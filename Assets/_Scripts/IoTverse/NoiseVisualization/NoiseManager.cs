using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Net.Http;
using UnityEngine;
using XCharts.Runtime;
using SimpleJSON;


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

    public float minDecibels = 30f;
    public float maxDecibels = 100f;

    // Thread-safe queue for fetching data
    private BlockingCollection<(NoiseData data, int sensorIndex)> dataQueue;
    private Thread fetcherThread;
    private bool isFetcherThreadRunning = false;

    // Main-thread pacing state
    private (NoiseData data, int sensorIndex)? pendingItem = null; // next sample awaiting scheduled time
    private bool hasSync = false;
    private long lastRemoteTimestamp = 0;      // ms
    private float lastLocalTimeSec = 0f;       // Unity realtime when last sample applied
    
    private int fetchLatestCount = 30; // Number of latest data points to fetch per request
    // Graph data tracking
    private List<(long timestamp, string label)> graphTimestamps = new List<(long, string)>();
    private const long graphRetentionMs = 30000; // 30 seconds

    // MockDataToggle element
    [SerializeField] public UnityEngine.UI.Toggle mockDataToggle;

    private void Start()
    {
        // Test GetFromJedi
        Debug.Log("Testing GetFromJedi in NoiseManager...");
        NoiseData[] testData = GetCurrentNoise();
        Debug.Log($"Retrieved {testData.Length} data points from Data Jedi.");

        if (spheres == null || spheres.Length == 0)
        {
            Debug.LogError("No spheres assigned to NoiseManager!");
            return;
        }

        // activeSensorIndex = spheres[0].SensorIndex;

        InitializeChart();

        // Initialize blocking collection backed by a concurrent queue
        dataQueue = new BlockingCollection<(NoiseData, int)>(new ConcurrentQueue<(NoiseData, int)>());

        // Start the fetcher thread
        isFetcherThreadRunning = true;
        fetcherThread = new Thread(FetcherThreadWork);
        fetcherThread.IsBackground = true;
        fetcherThread.Start();
        Debug.Log("Fetcher thread started.");
    }

    private void OnDestroy()
    {
        // Stop the fetcher thread safely
        isFetcherThreadRunning = false;
        if (fetcherThread != null && fetcherThread.IsAlive)
        {
            fetcherThread.Join(5000); // Wait up to 5 seconds for thread to finish
            Debug.Log("Fetcher thread stopped.");
        }
    }

    private void Update()
    {
        // Pull next item if none pending
        if (pendingItem == null && dataQueue != null)
        {
            (NoiseData data, int sensorIndex) temp;
            if (dataQueue.TryTake(out temp, 0))
            {
                pendingItem = temp;
            }
        }

        if (pendingItem == null)
        {
            return; // nothing to do this frame
        }

        var item = pendingItem.Value;
        var data = item.data;
        var sensorIndex = item.sensorIndex;

        float nowSec = Time.realtimeSinceStartup;

        if (!hasSync)
        {
            // First sample: apply immediately and set sync anchors
            ApplySampleToSpheres(data, sensorIndex);
            if (sensorIndex == currentSensorDisplayIndex)
            {
                AddSampleToGraph(data);
            }
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
            // Time to apply (or catch up if local is ahead)
            ApplySampleToSpheres(data, sensorIndex);
            if (sensorIndex == currentSensorDisplayIndex)
            {
                AddSampleToGraph(data);
            }

            // Advance anchors by the remote delta to preserve pacing
            lastRemoteTimestamp = data.timestamp;
            lastLocalTimeSec += remoteDeltaMs / 1000f;

            pendingItem = null; // move to next sample next frame
        }
        // else: not enough local time has passed; keep pending and check next frame (non-blocking)
    }

    private void FetcherThreadWork()
    {
        long[] lastTimestamps = new long[spheres.Length];
        while (isFetcherThreadRunning)
        {
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
                        NoiseData[] data = GetCurrentNoise(sensorIndex);
                        foreach (NoiseData noiseData in data) 
                            allData.Add((noiseData, sensorIndex));
                    }
                }
                
                // Sort by timestamp before enqueueing
                allData.Sort((a, b) => a.data.timestamp.CompareTo(b.data.timestamp));
                
                // Enqueue sorted data, filtering out old timestamps
                foreach (var item in allData)
                {
                    if (item.data.timestamp > lastTimestamps[item.sensorIndex])
                    {
                        lastTimestamps[item.sensorIndex] = item.data.timestamp;
                        dataQueue.Add(item);
                    }
                }
                
                // Wait before next poll
                Thread.Sleep(pollIntervalMs);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error in fetcher thread: {ex.Message}");
                Thread.Sleep(1000); // Sleep on error to avoid tight loop
            }
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
        lineChart.GetChartComponent<Title>().text = $"Noise Loudness - Sensor {sensorIndex}";
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
                lineChart.AddData(0, data[i].decibels);
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
            lineChart.AddData(0, sample.decibels);
            
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

    private NoiseData[] GetCurrentNoise(int sensorIndex = 0)
    {
        try
        {
            string baseUrl = "https://djx.entlab.hr/m2m/trusted/data";
            string resourceName = "dipProj25_noise_detector" + (sensorIndex + 1).ToString();
            int latestCount = fetchLatestCount;

            // Match the working example_get.py params exactly
            string url = $"{baseUrl}?usr=FER_Departments&latestNCount={latestCount}&res={resourceName}";

            var handler = new System.Net.Http.HttpClientHandler();
            handler.ServerCertificateCustomValidationCallback = (message, cert, chain, errors) => true;

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
        catch (AggregateException aggEx)
        {
            var ex = aggEx.GetBaseException();
            if (ex.Message.Contains("NameResolutionFailure") || ex.GetType().Name == "WebException")
            {
                Debug.LogWarning($"GetCurrentNoise: Cannot resolve hostname 'djx.entlab.hr'. Check network/VPN.");
            }
            else
            {
                Debug.LogError($"GetCurrentNoise failed: {ex.GetType().Name}: {ex.Message}");
            }
            return Array.Empty<NoiseData>();
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

    // Generate <dataPointCount> amount of noise data with 500ms difference back in time
    private NoiseData[] generateMockData(int sensorIndex)
    {
        // Random noise between minDecibels and maxDecibels
        System.Random random = new System.Random();
        int dataPointCount = 5;
        List<NoiseData> dataPoints = new List<NoiseData>();
        long currentTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        for (int i = 0; i < dataPointCount; i++)
        {
            long timestamp = currentTime - (dataPointCount - i) * 200; // 200ms intervals
            float decibels = (float)(minDecibels + random.NextDouble() * (maxDecibels - minDecibels));
            dataPoints.Add(new NoiseData(timestamp, decibels));
        }
        return dataPoints.ToArray();
    }
}