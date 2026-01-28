using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class MockNoiseDataProvider : INoiseDataProvider
{
    private readonly float minDecibels;
    private readonly float maxDecibels;
    private readonly System.Random random;
    private readonly Dictionary<int, CancellationTokenSource> activeStreams;
    private readonly Dictionary<int, long> streamStartTimes = new Dictionary<int, long>();

    public MockNoiseDataProvider(float minDecibels = 30f, float maxDecibels = 90f)
    {
        this.minDecibels = minDecibels;
        this.maxDecibels = maxDecibels;
        this.random = new System.Random();
        this.activeStreams = new Dictionary<int, CancellationTokenSource>();
    }

    public NoiseData[] GetNoiseData(int sensorIndex, long start, long end)
    {
        if (start >= end)
        {
            return new NoiseData[0];
        }

        // Generate data points every 100ms
        long interval = 50;
        int dataPointCount = (int)((end - start) / interval) + 1;
        
        List<NoiseData> dataPoints = new List<NoiseData>();
        
        for (long timestamp = start; timestamp <= end; timestamp += interval)
        {
            float decibels = GenerateWaveNoiseDecibels(timestamp-start);
            dataPoints.Add(new NoiseData(timestamp, decibels));
        }

        return dataPoints.ToArray();
    }

    public async void GetNoiseDataAsync(int sensorIndex, Action<NoiseData[]> callback)
    {
        // Simulate async data fetching with a small delay
        await Task.Delay(100);

        // Generate some recent data (last 10 seconds)
        long currentTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        long startTime = currentTime - 10000; // 10 seconds ago

        NoiseData[] data = GetNoiseData(sensorIndex, startTime, currentTime);
        
        callback?.Invoke(data);
    }

    public void StartContinuousStream(int sensorIndex, Action<NoiseData> onDataReceived)
    {
        // Stop existing stream for this sensor if any
        StopContinuousStream(sensorIndex);

        // Store stream start time for this sensor
        streamStartTimes[sensorIndex] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        // Create cancellation token for this stream
        CancellationTokenSource cts = new CancellationTokenSource();
        activeStreams[sensorIndex] = cts;

        // Start continuous streaming task
        Task.Run(async () =>
        {
            while (!cts.Token.IsCancellationRequested)
            {
                try
                {
                    long currentTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    float decibels = GenerateWaveNoiseDecibels(currentTime);
                    NoiseData data = new NoiseData(currentTime, decibels);

                    // Invoke callback on main thread (Unity requirement)
                    UnityMainThreadDispatcher.Instance.Enqueue(() => onDataReceived?.Invoke(data));

                    // Regular updates every 100ms
                    await Task.Delay(100, cts.Token);
                }
                catch (TaskCanceledException)
                {
                    // Stream was stopped, exit gracefully
                    break;
                }
            }
        }, cts.Token);
    }

    public void StopContinuousStream(int sensorIndex)
    {
        if (activeStreams.TryGetValue(sensorIndex, out CancellationTokenSource cts))
        {
            cts.Cancel();
            cts.Dispose();
            activeStreams.Remove(sensorIndex);
        }
        streamStartTimes.Remove(sensorIndex);
    }

    private float GenerateWaveNoiseDecibels(long timestamp)
    {
        // Create a realistic pattern using sine/cosine waves with multiple frequencies
        float baseValue = (minDecibels + maxDecibels) * 0.5f;
        float range = (maxDecibels - minDecibels) * 0.4f;
        
        // Convert timestamp to seconds
        float timeSeconds = timestamp / 1000f;
        
        // Primary wave: slow oscillation (period of ~10 seconds)
        float primaryWave = Mathf.Sin(timestamp * Mathf.PI / 5000f) * range;

        Debug.Log($"Generated decibels at time {timestamp}ms: {baseValue + primaryWave}");
        
        // Secondary wave: faster oscillation for more variation (period of ~5 seconds)
        //float secondaryWave = Mathf.Sin(timeSeconds * Mathf.PI / 2.5f) * range * 0.5f;
        
        // Add random noise (30% of the range)
        float noiseAmount = range * 0.3f;
        float randomNoise = (float)(random.NextDouble() * 2 - 1) * noiseAmount;
        
        // Combine all components
        float decibels = baseValue + primaryWave + randomNoise;
        
        // Clamp to valid range
        return Mathf.Clamp(decibels, minDecibels, maxDecibels);
    }
}


