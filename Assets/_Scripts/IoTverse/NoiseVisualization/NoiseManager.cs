using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Net.Http;
using UnityEngine;
using UnityEngine.UI;
using XCharts.Runtime;
using SimpleJSON;

public class NoiseManager : MonoBehaviour
{
    [Header("Spheres")]
    [SerializeField] private NoiseSphere[] spheres;

    [Header("Polling")]
    [SerializeField] private int pollIntervalMs = 500;
    [SerializeField] private int maxBufferedPoints = 256;

    [Header("Chart")]
    [SerializeField] public LineChart lineChart;

    [Header("UI (optional, for auto-wiring)")]
    [Tooltip("S1-S4 toggles (sphere visibility)")]
    [SerializeField] private Toggle[] sphereToggles; // length 4 (optional)
    [Tooltip("Sensor1-Sensor4 toggles (graph selection). Optional.")]
    [SerializeField] private Toggle[] sensorToggles; // length 4 (optional) - NOTE: ti imaš BUTTON, ne toggle
    [SerializeField] private Toggle mockDataToggle;  // optional
    [SerializeField] private Button startButton;     // not used but kept

    [Header("Noise range")]
    public float minDecibels = 30f;
    public float maxDecibels = 100f;

    [Header("Mode")]
    public bool onlyLiveData = true;

    private INoiseDataProvider noiseDataProvider;
    private int activeSensorIndex;
    private int currentSensorDisplayIndex = 0;

    private List<List<NoiseData>> currentSensorsData;

    private BlockingCollection<(NoiseData data, int sensorIndex)> dataQueue;
    private Thread fetcherThread;
    private bool isFetcherThreadRunning = false;

    // Main-thread pacing state
    private (NoiseData data, int sensorIndex)? pendingItem = null;
    private bool hasSync = false;

    private long startTimeMillisec;

    private long lastRemoteTimestamp = 0;
    private float lastLocalTimeSec = 0f;

    private int fetchLatestCount = 30;

    // Graph retention
    private List<(long timestamp, string label)> graphTimestamps = new List<(long, string)>();
    private const long graphRetentionMs = 30000;

    // --- NETWORK STATE BRIDGE ---
    private SharedNoiseCanvasState sharedState;

    // Thread-safe flag (fetcher thread MUST NOT touch Unity Toggle)
    private volatile bool _mockOnThreadSafe = false;

    // Cache for sphere actives (main thread apply)
    private bool[] _sphereActive = new bool[4] { true, true, true, true };

    private void Awake()
    {
        // Nemoj se oslanjati na Awake jer state često još nije spawnan.
        // Samo pokušaj, ali imamo EnsureSharedState() svugdje gdje treba.
        sharedState = FindObjectOfType<SharedNoiseCanvasState>(true);
    }

    private void Start()
    {
        startTimeMillisec = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        if (spheres == null || spheres.Length == 0)
        {
            Debug.LogError("No spheres assigned to NoiseManager!");
            return;
        }

        // Init per-sensor buffers
        currentSensorsData = new List<List<NoiseData>>();
        for (int i = 0; i < spheres.Length; i++)
            currentSensorsData.Add(new List<NoiseData>());

        InitializeChart();

        // Auto-wire UI (OPTIONAL) - koristi samo ako si stvarno popunio reference u inspectoru.
        WireUIIfAssigned();

        // Init blocking collection
        dataQueue = new BlockingCollection<(NoiseData, int)>(new ConcurrentQueue<(NoiseData, int)>());

        // Start fetcher thread
        isFetcherThreadRunning = true;
        fetcherThread = new Thread(FetcherThreadWork)
        {
            IsBackground = true
        };
        fetcherThread.Start();
        Debug.Log("Fetcher thread started.");
    }

    private void WireUIIfAssigned()
    {
        // Ako koristiš NoiseUIBridge i evente spajaš na njega, onda NE puni ova polja
        // i ovo se neće ništa attachati.

        if (sphereToggles != null && sphereToggles.Length >= 4)
        {
            for (int i = 0; i < 4; i++)
            {
                int idx = i;
                if (sphereToggles[idx] == null) continue;
                sphereToggles[idx].onValueChanged.AddListener((on) => OnSphereToggleChanged(idx, on));
            }
        }

        // Ovo je za slučaj da su Sensor1-4 TOGGLES. Ti imaš BUTTONe pa ti ne treba.
        if (sensorToggles != null && sensorToggles.Length >= 4)
        {
            for (int i = 0; i < 4; i++)
            {
                int idx = i;
                if (sensorToggles[idx] == null) continue;
                sensorToggles[idx].onValueChanged.AddListener((on) =>
                {
                    if (on) OnSensorSelected(idx);
                });
            }
        }

        if (mockDataToggle != null)
        {
            mockDataToggle.onValueChanged.AddListener(OnMockToggleChanged);
        }
    }

    private void OnDestroy()
    {
        isFetcherThreadRunning = false;
        if (fetcherThread != null && fetcherThread.IsAlive)
        {
            fetcherThread.Join(5000);
            Debug.Log("Fetcher thread stopped.");
        }
    }

    // ---------------------------
    // Shared state lazy-find
    // ---------------------------
    private bool EnsureSharedState()
    {
        if (sharedState != null) return true;
        sharedState = FindObjectOfType<SharedNoiseCanvasState>(true);
        return sharedState != null;
    }

    // ---------------------------
    // UI -> NETWORK requests
    // ---------------------------
    public void OnSphereToggleChanged(int sphereIndex, bool on)
    {
        EnsureSharedState();

        if (sharedState != null)
            sharedState.RequestSetSphere(sphereIndex, on);
        else
            SetSphereActiveLocal(sphereIndex, on);
    }

    // Ako ti Buttoni i dalje zovu DrawSensorData(int), ovo je kompatibilno
    public void DrawSensorData(int sensorIndex)
    {
        OnSensorSelected(sensorIndex);
    }

    // Preporuka: Buttoni zovu ovu metodu (OnClick)
    public void OnSensorSelected(int sensorIndex)
    {
        sensorIndex = Mathf.Clamp(sensorIndex, 0, 3);
        EnsureSharedState();

        if (sharedState != null)
            sharedState.RequestSetActiveSensor(sensorIndex);
        else
            ApplyGraphSensorLocal(sensorIndex);
    }

    public void OnMockToggleChanged(bool on)
    {
        EnsureSharedState();

        if (sharedState != null)
            sharedState.RequestSetMock(on);
        else
            _mockOnThreadSafe = on;
    }

    // ---------------------------
    // NETWORK -> LOCAL apply
    // Called by SharedNoiseCanvasState every Render when state changes
    // ---------------------------
    public void ApplyNetworkState(bool sphere0, bool sphere1, bool sphere2, bool sphere3, int activeSensorIndex, bool mockOn)
    {
        // spheres
        SetSphereActiveLocal(0, sphere0);
        SetSphereActiveLocal(1, sphere1);
        SetSphereActiveLocal(2, sphere2);
        SetSphereActiveLocal(3, sphere3);

        // mock
        _mockOnThreadSafe = mockOn;

        // graph selection
        ApplyGraphSensorLocal(activeSensorIndex);

        // keep UI in sync locally (ONLY if you assigned these refs)
        if (sphereToggles != null && sphereToggles.Length >= 4)
        {
            if (sphereToggles[0] != null) sphereToggles[0].SetIsOnWithoutNotify(sphere0);
            if (sphereToggles[1] != null) sphereToggles[1].SetIsOnWithoutNotify(sphere1);
            if (sphereToggles[2] != null) sphereToggles[2].SetIsOnWithoutNotify(sphere2);
            if (sphereToggles[3] != null) sphereToggles[3].SetIsOnWithoutNotify(sphere3);
        }

        if (mockDataToggle != null)
            mockDataToggle.SetIsOnWithoutNotify(mockOn);

        if (sensorToggles != null && sensorToggles.Length >= 4)
        {
            for (int i = 0; i < 4; i++)
                if (sensorToggles[i] != null)
                    sensorToggles[i].SetIsOnWithoutNotify(i == Mathf.Clamp(activeSensorIndex, 0, 3));
        }
    }

    private void SetSphereActiveLocal(int sphereIndex, bool active)
    {
        if (sphereIndex < 0 || sphereIndex >= spheres.Length) return;

        _sphereActive[sphereIndex] = active;

        if (spheres[sphereIndex] != null)
            spheres[sphereIndex].gameObject.SetActive(active);
    }

    private void ApplyGraphSensorLocal(int sensorIndex)
    {
        sensorIndex = Mathf.Clamp(sensorIndex, 0, Mathf.Max(0, currentSensorsData.Count - 1));
        if (sensorIndex == currentSensorDisplayIndex) return;

        currentSensorDisplayIndex = sensorIndex;
        List<NoiseData> sensorData = currentSensorsData[sensorIndex];
        DrawGraphPoints(sensorData);

        if (lineChart != null)
        {
            var title = lineChart.GetChartComponent<Title>();
            if (title != null) title.text = $"Noise Loudness - Sensor {sensorIndex + 1}";
            lineChart.RefreshChart();
        }
    }

    // ---------------------------
    // Main Update (graph + spheres)
    // ---------------------------
    private void Update()
    {
        if (pendingItem == null && dataQueue != null)
        {
            if (dataQueue.TryTake(out var temp, 0))
                pendingItem = temp;
        }

        if (pendingItem == null) return;

        var item = pendingItem.Value;
        var data = item.data;
        var sensorIndex = item.sensorIndex;

        float nowSec = Time.realtimeSinceStartup;

        if (!hasSync)
        {
            ApplySampleToSpheres(data, sensorIndex);
            currentSensorsData[sensorIndex].Add(data);

            if (sensorIndex == currentSensorDisplayIndex)
                AddSampleToGraph(data);

            hasSync = true;
            lastRemoteTimestamp = data.timestamp;
            lastLocalTimeSec = nowSec;
            pendingItem = null;
            return;
        }

        long remoteDeltaMs = data.timestamp - lastRemoteTimestamp;
        float localDeltaMs = (nowSec - lastLocalTimeSec) * 1000f;

        if (remoteDeltaMs <= 0 || localDeltaMs >= remoteDeltaMs)
        {
            ApplySampleToSpheres(data, sensorIndex);

            if (sensorIndex >= 0 && sensorIndex < currentSensorsData.Count)
            {
                currentSensorsData[sensorIndex].Add(data);

                long cutoffTime = data.timestamp - graphRetentionMs;
                currentSensorsData[sensorIndex].RemoveAll(d => d.timestamp < cutoffTime);
            }

            if (sensorIndex == currentSensorDisplayIndex)
                AddSampleToGraph(data);

            lastRemoteTimestamp = data.timestamp;
            lastLocalTimeSec += remoteDeltaMs / 1000f;

            pendingItem = null;
        }
    }

    // ---------------------------
    // Fetcher thread (NO Unity access!)
    // ---------------------------
    private void FetcherThreadWork()
    {
        long[] lastTimestamps = new long[spheres.Length];

        while (isFetcherThreadRunning)
        {
            try
            {
                var allData = new List<(NoiseData data, int sensorIndex)>();

                for (int sensorIndex = 0; sensorIndex < spheres.Length; sensorIndex++)
                {
                    if (_mockOnThreadSafe)
                    {
                        NoiseData[] data = generateMockData(sensorIndex);
                        foreach (NoiseData noiseData in data)
                            allData.Add((noiseData, sensorIndex));
                    }
                    else
                    {
                        NoiseData[] data = GetCurrentNoise(sensorIndex);
                        foreach (NoiseData noiseData in data)
                            allData.Add((noiseData, sensorIndex));
                    }
                }

                allData.Sort((a, b) => a.data.timestamp.CompareTo(b.data.timestamp));

                foreach (var item in allData)
                {
                    if (onlyLiveData && item.data.timestamp <= startTimeMillisec)
                        continue;

                    if (item.data.timestamp > lastTimestamps[item.sensorIndex])
                    {
                        lastTimestamps[item.sensorIndex] = item.data.timestamp;
                        dataQueue.Add(item);
                    }
                }

                Thread.Sleep(pollIntervalMs);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error in fetcher thread: {ex.Message}");
                Thread.Sleep(1000);
            }
        }
    }

    // ---------------------------
    // Chart helpers
    // ---------------------------
    private void InitializeChart()
    {
        if (lineChart == null)
        {
            Debug.LogError("LineChart reference is not set in NoiseManager!");
            return;
        }

        lineChart.ClearData();
        if (lineChart.series.Count == 0)
            lineChart.AddSerie<Line>("Noise Frequency");
    }

    private void DrawGraphPoints(List<NoiseData> data)
    {
        if (lineChart == null)
        {
            Debug.LogError("LineChart reference is not set in NoiseManager!");
            return;
        }

        lineChart.ClearData();

        if (lineChart.series.Count == 0)
        {
            var serie = lineChart.AddSerie<Line>("Noise Loudness");
            serie.symbol.show = true;
            serie.symbol.type = SymbolType.Circle;
        }

        for (int i = 0; i < data.Count; i++)
        {
            string label = FormatTimestamp(data[i].timestamp);
            lineChart.AddXAxisData(label);
            lineChart.AddData(0, data[i].decibels);
        }

        lineChart.RefreshChart();
    }

    private void AddSampleToGraph(NoiseData sample)
    {
        if (lineChart == null) return;

        string label = FormatTimestamp(sample.timestamp);
        lineChart.AddXAxisData(label);
        lineChart.AddData(0, sample.decibels);

        graphTimestamps.Add((sample.timestamp, label));

        long cutoffTime = sample.timestamp - graphRetentionMs;
        int removeCount = 0;

        for (int i = 0; i < graphTimestamps.Count; i++)
        {
            if (graphTimestamps[i].timestamp < cutoffTime) removeCount++;
            else break;
        }

        for (int i = 0; i < removeCount; i++)
        {
            if (lineChart.series.Count > 0)
                lineChart.series[0].RemoveData(0);

            var xAxis = lineChart.GetChartComponent<XAxis>(0);
            if (xAxis != null && xAxis.data.Count > 0)
                xAxis.RemoveData(0);
        }

        if (removeCount > 0)
            graphTimestamps.RemoveRange(0, removeCount);

        lineChart.RefreshChart();
    }

    // ---------------------------
    // Sphere apply
    // ---------------------------
    private void ApplySampleToSpheres(NoiseData sample, int sensorIndex)
    {
        if (sensorIndex < 0 || sensorIndex >= spheres.Length) return;
        if (!_sphereActive[sensorIndex]) return;

        float radius = MapDecibelsToRadius(sample.decibels, minDecibels, maxDecibels);
        spheres[sensorIndex].SetRadius(radius);
    }

    private float MapDecibelsToRadius(float decibels, float minDecibels, float maxDecibels)
    {
        float minRadius = 0.2f;
        float maxRadius = 0.7f;
        return Mathf.Lerp(minRadius, maxRadius, (decibels - minDecibels) / (maxDecibels - minDecibels));
    }

    private string FormatTimestamp(long unixMilliseconds)
    {
        DateTime dt = DateTimeOffset.FromUnixTimeMilliseconds(unixMilliseconds).LocalDateTime;
        return dt.ToString("HH:mm:ss");
    }

    // ---------------------------
    // Remote fetch
    // ---------------------------
    private NoiseData[] GetCurrentNoise(int sensorIndex = 0)
    {
        try
        {
            string baseUrl = "https://djx.entlab.hr/m2m/trusted/data";
            string resourceName = "dipProj25_noise_detector" + (sensorIndex + 1).ToString();
            int latestCount = fetchLatestCount;

            string url = $"{baseUrl}?usr=FER_Departments&latestNCount={latestCount}&res={resourceName}";

            var handler = new HttpClientHandler
            {
                ServerCertificateCustomValidationCallback = (message, cert, chain, errors) => true
            };

            using (var client = new HttpClient(handler))
            {
                client.Timeout = TimeSpan.FromSeconds(10);
                client.DefaultRequestHeaders.Add("Authorization", "PREAUTHENTICATED");
                client.DefaultRequestHeaders.Add("X-Requester-Id", "digiphy1");
                client.DefaultRequestHeaders.Add("X-Requester-Type", "domainApplication");
                client.DefaultRequestHeaders.TryAddWithoutValidation("Accept", "application/vnd.ericsson.simple.output+json;version=1.0");

                var response = client.GetAsync(url).Result;
                if (!response.IsSuccessStatusCode)
                {
                    string errorBody = response.Content.ReadAsStringAsync().Result;
                    Debug.LogError($"Request failed: {response.StatusCode}\nServer message: {errorBody}\nURL: {url}");
                    return Array.Empty<NoiseData>();
                }

                string json = response.Content.ReadAsStringAsync().Result;
                return ParseNoiseDataArray(json);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"GetCurrentNoise failed: {ex.GetType().Name}: {ex.Message}");
            return Array.Empty<NoiseData>();
        }
    }

    private NoiseData[] ParseNoiseDataArray(string json)
    {
        var result = new List<NoiseData>();
        try
        {
            var root = JSON.Parse(json);
            if (root == null || !root.IsArray) return Array.Empty<NoiseData>();

            foreach (JSONNode deviceNode in root.AsArray)
            {
                foreach (var gwKey in deviceNode.Keys)
                {
                    if (gwKey == "deviceId") continue;
                    var gwNode = deviceNode[gwKey];
                    if (gwNode == null) continue;

                    foreach (var sensorKey in gwNode.Keys)
                    {
                        var sensorNode = gwNode[sensorKey];
                        if (sensorNode == null) continue;

                        foreach (var tsKey in sensorNode.Keys)
                        {
                            var dataNode = sensorNode[tsKey];
                            if (dataNode == null) continue;

                            float noiseValue = dataNode["noise_detector"].AsFloat;

                            if (DateTime.TryParse(tsKey, null, System.Globalization.DateTimeStyles.RoundtripKind, out DateTime timestamp))
                            {
                                long timestampMs = new DateTimeOffset(timestamp).ToUnixTimeMilliseconds();
                                result.Add(new NoiseData(timestampMs, noiseValue));
                            }
                        }
                    }
                }
            }

            result.Sort((a, b) => a.timestamp.CompareTo(b.timestamp));
        }
        catch (Exception ex)
        {
            Debug.LogError($"ParseNoiseDataArray failed: {ex.Message}");
        }

        return result.ToArray();
    }

    // ---------------------------
    // Mock data
    // ---------------------------
    private NoiseData[] generateMockData(int sensorIndex)
    {
        int dataPointCount = 5;
        List<NoiseData> dataPoints = new List<NoiseData>();
        long currentTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        float phaseOffset = sensorIndex * (2f * Mathf.PI / Mathf.Max(1, spheres.Length));
        System.Random random = new System.Random(sensorIndex);

        for (int i = 0; i < dataPointCount; i++)
        {
            long timestamp = currentTime - (dataPointCount - i) * 200;
            float decibels = GenerateWaveNoiseDecibels(timestamp, phaseOffset, random);
            dataPoints.Add(new NoiseData(timestamp, decibels));
        }

        return dataPoints.ToArray();
    }

    private float GenerateWaveNoiseDecibels(long timestamp, float phaseOffset, System.Random random)
    {
        float baseValue = (minDecibels + maxDecibels) * 0.5f;
        float range = (maxDecibels - minDecibels) * 0.4f;

        float relativeTimeSeconds = (timestamp - startTimeMillisec) / 1000f;

        float primaryWave = Mathf.Sin((relativeTimeSeconds * 2f * Mathf.PI / 20f) + phaseOffset) * range;
        float secondaryWave = Mathf.Sin((relativeTimeSeconds * 2f * Mathf.PI / 5f) + phaseOffset * 0.5f) * range * 0.5f;

        float noiseAmount = range * 0.3f;
        float randomNoise = (float)(random.NextDouble() * 2 - 1) * noiseAmount;

        float decibels = baseValue + primaryWave + secondaryWave + randomNoise;
        return Mathf.Clamp(decibels, minDecibels, maxDecibels);
    }
}
