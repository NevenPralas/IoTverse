using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Net;
using System.Net.Http;
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
using UnityEngine;
using UnityEngine.Networking;
using XCharts.Runtime;
using SimpleJSON;
using System.Collections.ObjectModel;


public class NoiseManager : MonoBehaviour
{
    [SerializeField] private NoiseSphere[] spheres;
    [SerializeField] private int pollIntervalMs = 500;
    [SerializeField] private int maxBufferedPoints = 256;

    private INoiseDataProvider noiseDataProvider;
    private int activeSensorIndex;
    // XChart
    [SerializeField] public LineChart lineChart;

    // Button to start simulation
    [SerializeField] private UnityEngine.UI.Button startButton;

    private List<List<NoiseData>> currentSensorsData; // List of data for each sensor

    private int currentSensorDisplayIndex = 0;

    private int numSensors = 4;

    public float minDecibels = 30f;
    public float maxDecibels = 100f;

    // List of queues for each sensor
    private Queue<(NoiseData data, int sensorIndex)> incomingNoiseQueue = new Queue<(NoiseData data, int sensorIndex)>();

    // Main-thread pacing state
    private bool hasSync = false;

    private long startTimeMillisec;

    public bool onlyLiveData = true;

    private long lastRemoteTimestamp = 0;
    private float lastLocalTimeSec = 0f;       // Unity realtime when last sample applied
    private bool isFetcherCoroutineRunning = true;
    private int fetchLatestCount = 10; // Number of latest data points to fetch per request
    // Graph data tracking
    private List<(long timestamp, string label)> graphTimestamps = new List<(long, string)>();
    private const long graphRetentionMs = 30000; // 30 seconds

    // MockDataToggle element
    [SerializeField] public UnityEngine.UI.Toggle mockDataToggle;

    private void Start()
    {
        startTimeMillisec = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        if (spheres == null || spheres.Length == 0)
        {
            Debug.LogError("No spheres assigned to NoiseManager!");
            return;
        }

        // Initialize currentSensorsData with a list for each sensor
        currentSensorsData = new List<List<NoiseData>>();
        for (int i = 0; i < spheres.Length; i++)
        {
            currentSensorsData.Add(new List<NoiseData>());
        }

        InitializeChart();

        // Initialize queue
        incomingNoiseQueue = new Queue<(NoiseData data, int sensorIndex)>();

        // Start coroutine to fetch data periodically
        StartCoroutine(FetchDataCoroutine());

    }

    private void OnDestroy()
    {
        // Stop the fetcher coroutine safely
        isFetcherCoroutineRunning = false;
    }

    private void Update()
    {
        float nowSec = Time.realtimeSinceStartup;
        NoiseData data;
        int sensorIndex;

        if (incomingNoiseQueue.Count == 0)
        {
            return; // No data to process
        }

        if (!hasSync)
        {
            (data, sensorIndex) = incomingNoiseQueue.Dequeue();

            // First sample: apply immediately and set sync anchors
            ApplySampleToSpheres(data, sensorIndex);

            // Store data for all sensors (for later display switching)
            currentSensorsData[sensorIndex].Add(data);

            if (sensorIndex == currentSensorDisplayIndex)
            {
                AddSampleToGraph(data);
            }
            hasSync = true;
            lastRemoteTimestamp = data.timestamp;
            lastLocalTimeSec = nowSec;
            return;
        }

        long remoteDeltaMs = incomingNoiseQueue.Peek().data.timestamp - lastRemoteTimestamp;
        float localDeltaMs = (nowSec - lastLocalTimeSec) * 1000f;

        if (remoteDeltaMs <= 0 || localDeltaMs >= remoteDeltaMs)
        {
            (data, sensorIndex) = incomingNoiseQueue.Dequeue();

            // Time to apply (or catch up if local is ahead)
            ApplySampleToSpheres(data, sensorIndex);

            currentSensorsData[sensorIndex].Add(data);

            // Maintain 30-second retention for all sensors
            long cutoffTime = data.timestamp - graphRetentionMs;
            currentSensorsData[sensorIndex].RemoveAll(d => d.timestamp < cutoffTime);

            if (sensorIndex == currentSensorDisplayIndex)
            {
                AddSampleToGraph(data);
            }

            // Advance anchors by the remote delta to preserve pacing
            lastRemoteTimestamp = data.timestamp;
            lastLocalTimeSec += remoteDeltaMs / 1000f;
        }
    }

    private System.Collections.IEnumerator FetchDataCoroutine()
    {
        while (isFetcherCoroutineRunning)
        {
            FetchData();
            yield return new WaitForSeconds(2f); // Wait 2 seconds before next fetch
        }
    }


    private async void FetchData()
    {
        long[] lastTimestamps = new long[spheres.Length];

        try
        {
            // Collect all data from all sensors
            List<(NoiseData data, int sensorIndex)> allData = new List<(NoiseData, int)>();

            for (int sensorIndex = 0; sensorIndex < spheres.Length; sensorIndex++)
            {
                if (mockDataToggle.isOn)
                {
                    NoiseData[] data = generateMockData(sensorIndex);
                    foreach (NoiseData noiseData in data)
                        allData.Add((noiseData, sensorIndex));
                }
                else
                {
                    NoiseData[] data = await GetCurrentNoise(sensorIndex);
                    foreach (NoiseData noiseData in data)
                        allData.Add((noiseData, sensorIndex));
                }
            }

            // Sort by timestamp before enqueueing
            allData.Sort((a, b) => a.data.timestamp.CompareTo(b.data.timestamp));

            // Enqueue sorted data, filtering out old timestamps
            foreach (var item in allData)
            {
                if (onlyLiveData && item.data.timestamp <= startTimeMillisec)
                {
                    continue; // skip old data
                }

                if (item.data.timestamp > lastTimestamps[item.sensorIndex])
                {
                    lastTimestamps[item.sensorIndex] = item.data.timestamp;
                    incomingNoiseQueue.Enqueue(item);
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error while fetching data: {ex.Message}");
            Thread.Sleep(1000);
        }
    }


    private void InitializeChart()
    {
        if (lineChart == null)
        {
            Debug.LogError("LineChart reference is not set in NoiseManager!");
            return;
        }

        lineChart.ClearData();
        if (lineChart.series.Count == 0)
        {
            var serie = lineChart.AddSerie<Line>("Noise Frequency");
            // serie.symbol.show = true;
            // serie.symbol.type = SymbolType.Circle;
        }
    }

    public void DrawSensorData(int sensorIndex)
    {
        Debug.LogWarning("DrawSensorData called....................................................");
        if (sensorIndex < 0 || sensorIndex >= currentSensorsData.Count)
        {
            Debug.LogError("Invalid sensor index or no data available.");
            return;
        }

        if (sensorIndex == currentSensorDisplayIndex) return; // No change

        currentSensorDisplayIndex = sensorIndex;
        List<NoiseData> sensorData = currentSensorsData[sensorIndex];
        DrawGraphPoints(sensorData);

        // Change graph title to indicate sensor
        lineChart.GetChartComponent<Title>().text = $"Noise Loudness - Sensor {sensorIndex + 1}";
        lineChart.RefreshChart();
    }

    private void DrawGraphPoints(List<NoiseData> data)
    {
        Debug.Log("Drawing graph points in NoiseManager...............................................");
        if (lineChart != null)
        {
            // Clear existing data
            lineChart.ClearData();

            // Ensure we have a Line serie (only add if not already present)
            if (lineChart.series.Count == 0)
            {
                var serie = lineChart.AddSerie<Line>("Noise Loudness");
                // Enable symbols (dots) on the line
                serie.symbol.show = true;
                serie.symbol.type = SymbolType.Circle;
            }

            // Add data points
            for (int i = 0; i < data.Count; i++)
            {
                string label = FormatTimestamp(data[i].timestamp);
                lineChart.AddXAxisData(label);
                // Add Y-axis value for serie 0
                float roundedDecibels = Mathf.Round(data[i].decibels * 10f) / 10f;
                lineChart.AddData(0, roundedDecibels);
            }

            lineChart.RefreshChart();
        }
        else
        {
            Debug.LogError("LineChart reference is not set in NoiseManager!");
        }
    }

    private void AddSampleToGraph(NoiseData sample)
    {
        if (lineChart != null)
        {
            string label = FormatTimestamp(sample.timestamp);
            lineChart.AddXAxisData(label);
            float roundedDecibels = Mathf.Round(sample.decibels * 10f) / 10f;
            lineChart.AddData(0, roundedDecibels);

            currentSensorsData[currentSensorDisplayIndex].Add(sample);

            // Track timestamps
            graphTimestamps.Add((sample.timestamp, label));

            // Remove data points older than 30 seconds
            long cutoffTime = sample.timestamp - graphRetentionMs;
            int removeCount = 0;

            for (int i = 0; i < graphTimestamps.Count; i++)
            {
                if (graphTimestamps[i].timestamp < cutoffTime)
                {
                    removeCount++;
                }
                else
                {
                    break; // timestamps are sorted, so we can stop
                }
            }

            // Remove old data points from chart and tracking list
            for (int i = 0; i < removeCount; i++)
            {
                if (lineChart.series.Count > 0)
                {
                    lineChart.series[0].RemoveData(0); // Remove oldest data point from series 0
                }

                // Remove oldest x-axis label
                var xAxis = lineChart.GetChartComponent<XCharts.Runtime.XAxis>(0);
                if (xAxis != null && xAxis.data.Count > 0)
                {
                    xAxis.RemoveData(0);
                }
            }

            // Remove from tracking list
            if (removeCount > 0)
            {
                graphTimestamps.RemoveRange(0, removeCount);

                // Also remove from currentSensorsData for the current display sensor
                if (currentSensorDisplayIndex >= 0 && currentSensorDisplayIndex < currentSensorsData.Count)
                {
                    int currentDataCount = currentSensorsData[currentSensorDisplayIndex].Count;
                    if (removeCount <= currentDataCount)
                    {
                        currentSensorsData[currentSensorDisplayIndex].RemoveRange(0, removeCount);
                    }
                }
            }

            lineChart.RefreshChart();
        }
    }

    private void ApplySampleToSpheres(NoiseData sample, int sensorIndex)
    {
        // float minDecibels = currentSensorsData.Min(d => d.decibels);
        // float maxDecibels = currentSensorsData.Max(d => d.decibels);
        float radius = MapDecibelsToRadius(sample.decibels, minDecibels, maxDecibels);
        Debug.Log($"Applying sample to sphere {sensorIndex}: Decibels={sample.decibels}, Radius={radius}");
        spheres[sensorIndex].SetRadius(radius);

        Debug.Log($"Timestamp: {sample.timestamp}, Frequency: {sample.decibels}, Radius: {radius}");
    }

    private float MapDecibelsToRadius(float decibels, float minDecibels, float maxDecibels)
    {
        // Mapiramo decibele na radijus između 0.2 i 1.0
        float minRadius = 0.2f;
        float maxRadius = 0.7f;
        return Mathf.Lerp(minRadius, maxRadius, (decibels - minDecibels) / (maxDecibels - minDecibels));
    }

    private string FormatTimestamp(long unixMilliseconds)
    {
        DateTime dt = DateTimeOffset.FromUnixTimeMilliseconds(unixMilliseconds).LocalDateTime;
        return dt.ToString("HH:mm:ss");
    }

    private async Task<NoiseData[]> GetCurrentNoise(int sensorIndex = 0)
    {
        try
        {
            string baseUrl = "https://djx.entlab.hr/m2m/trusted/data";
            string resourceName = "dipProj25_noise_detector" + (sensorIndex + 1).ToString();
            string url = $"{baseUrl}?usr=FER_Departments&latestNCount={fetchLatestCount}&res={resourceName}";

            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                // Set custom headers
                request.SetRequestHeader("Authorization", "PREAUTHENTICATED");
                request.SetRequestHeader("X-Requester-Id", "digiphy1");
                request.SetRequestHeader("X-Requester-Type", "domainApplication");
                request.SetRequestHeader("Accept", "application/vnd.ericsson.simple.output+json;version=1.0");

                // Set timeout (in seconds)
                request.timeout = 10;

                // Send the request asynchronously
                var operation = request.SendWebRequest();

                // Wait for the request to complete
                while (!operation.isDone)
                {
                    await Task.Yield();
                }

                // Check for errors
                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError($"Request failed: {request.error}\nResponse Code: {request.responseCode}");
                    return Array.Empty<NoiseData>();
                }

                string json = request.downloadHandler.text;
                return ParseNoiseDataArray(json);
            }
        }
        catch (Exception ex)
        {
            string detailedError = ex.InnerException != null ? ex.InnerException.Message : ex.Message;
            Debug.LogError($"GetCurrentNoise failed: {ex.GetType().Name} -> {detailedError}");
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
                // Navigate: device -> gateway node -> sensor node -> timestamps
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

    // Generate mock noise data with realistic wave patterns and sensor-specific phases
    private NoiseData[] generateMockData(int sensorIndex)
    {
        int dataPointCount = 5;
        List<NoiseData> dataPoints = new List<NoiseData>();
        long currentTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        // Sensor-specific phase offset
        float phaseOffset = sensorIndex * (2f * Mathf.PI / Mathf.Max(1, spheres.Length));

        // Deterministic random per sensor
        System.Random random = new System.Random(sensorIndex);

        for (int i = 0; i < dataPointCount; i++)
        {
            long timestamp = currentTime - (dataPointCount - i) * 200; // 200ms intervals

            float decibels = GenerateWaveNoiseDecibels(timestamp, phaseOffset, random);
            dataPoints.Add(new NoiseData(timestamp, decibels));
        }

        return dataPoints.ToArray();
    }

    private float GenerateWaveNoiseDecibels(long timestamp, float phaseOffset, System.Random random)
    {
        // Create a realistic pattern using sine/cosine waves with multiple frequencies
        float baseValue = (minDecibels + maxDecibels) * 0.5f;
        float range = (maxDecibels - minDecibels) * 0.4f;

        // Use relative time from start (in seconds) for proper period calculation
        float relativeTimeSeconds = (timestamp - startTimeMillisec) / 1000f;

        // Primary wave: 20 second period with sensor-specific phase
        float primaryWave = Mathf.Sin((relativeTimeSeconds * 2f * Mathf.PI / 20f) + phaseOffset) * range;

        // Secondary wave: 5 second period for more variation
        float secondaryWave = Mathf.Sin((relativeTimeSeconds * 2f * Mathf.PI / 5f) + phaseOffset * 0.5f) * range * 0.5f;

        // Add random noise (30% of the range)
        float noiseAmount = range * 0.3f;
        float randomNoise = (float)(random.NextDouble() * 2 - 1) * noiseAmount;

        // Combine all components
        float decibels = baseValue + primaryWave + secondaryWave + randomNoise;

        // Clamp to valid range
        return Mathf.Clamp(decibels, minDecibels, maxDecibels);
    }
}