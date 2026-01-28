using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using SimpleJSON;
using System.Globalization;
using UnityEngine;

public class HttpNoiseDataProvider : INoiseDataProvider
{
    private readonly string baseUrl = "https://djx.entlab.hr/m2m/trusted/data";
    private readonly int pollIntervalMs = 750;
    private readonly HttpClient httpClient;
    private readonly string resource = "dipProj25_noise_detector";
    private readonly Dictionary<int, CancellationTokenSource> activeStreams = new Dictionary<int, CancellationTokenSource>();

    /// <param name="baseUrl">Data Jedi base url, e.g. https://djx.entlab.hr/m2m/trusted/data</param>
    /// <param name="resource">Resource name on Data Jedi (maps the sensor); if null uses sensorIndex as suffix.</param>
    public HttpNoiseDataProvider()
    {
        httpClient = new HttpClient();
    }

    public NoiseData[] GetNoiseData(int sensorIndex, long start, long end)
    {
        try
        {
            return FetchRangeAsync(sensorIndex, start, end, CancellationToken.None).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Debug.LogError($"[HttpNoiseDataProvider] Failed to fetch history: {ex.Message}");
            return Array.Empty<NoiseData>();
        }
    }

    public async void GetNoiseDataAsync(int sensorIndex, Action<NoiseData[]> callback)
    {
        try
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            long start = now - 30000; // last 30 seconds of history
            var data = await FetchRangeAsync(sensorIndex, start, now, CancellationToken.None);
            UnityMainThreadDispatcher.Instance.Enqueue(() => callback?.Invoke(data));
        }
        catch (Exception ex)
        {
            Debug.LogError($"[HttpNoiseDataProvider] Async history fetch failed: {ex.Message}");
            UnityMainThreadDispatcher.Instance.Enqueue(() => callback?.Invoke(Array.Empty<NoiseData>()));
        }
    }

    public void StartContinuousStream(int sensorIndex, Action<NoiseData> onDataReceived)
    {
        StopContinuousStream(sensorIndex);

        CancellationTokenSource cts = new CancellationTokenSource();
        activeStreams[sensorIndex] = cts;

        _ = Task.Run(() => StreamLoopAsync(sensorIndex, onDataReceived, cts.Token), cts.Token);
    }

    public void StopContinuousStream(int sensorIndex)
    {
        if (activeStreams.TryGetValue(sensorIndex, out CancellationTokenSource cts))
        {
            cts.Cancel();
            cts.Dispose();
            activeStreams.Remove(sensorIndex);
        }
    }

    private async Task StreamLoopAsync(int sensorIndex, Action<NoiseData> onDataReceived, CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                NoiseData latest = await FetchLatestAsync(sensorIndex, token);
                if (latest != null)
                {
                    UnityMainThreadDispatcher.Instance.Enqueue(() => onDataReceived?.Invoke(latest));
                }
            }
            catch (TaskCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[HttpNoiseDataProvider] Stream polling failed: {ex.Message}");
            }

            try
            {
                await Task.Delay(pollIntervalMs, token);
            }
            catch (TaskCanceledException)
            {
                break;
            }
        }
    }

    private async Task<NoiseData> FetchLatestAsync(int sensorIndex, CancellationToken token)
    {
        string url = BuildLatestUrl(sensorIndex);
        using (HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Get, url))
        {
            AddDefaultHeaders(request.Headers);
            using (HttpResponseMessage response = await httpClient.SendAsync(request, token))
            {
                response.EnsureSuccessStatusCode();
                string json = await response.Content.ReadAsStringAsync();
                return ParseSingle(json);
            }
        }
    }

    private string BuildLatestUrl(int sensorIndex)
    {
        // Data Jedi GET prediction: filter by resource and limit=1 to fetch newest sample.
        string res = ResolveResource(sensorIndex);
        return $"{baseUrl}?resource={Uri.EscapeDataString(res)}&limit=1&order=desc";
    }

    private string BuildRangeUrl(int sensorIndex, long start, long end)
    {
        // Data Jedi GET prediction: use ISO 8601 time bounds.
        string res = ResolveResource(sensorIndex);
        string fromIso = DateTimeOffset.FromUnixTimeMilliseconds(start).UtcDateTime.ToString("o", CultureInfo.InvariantCulture);
        string toIso = DateTimeOffset.FromUnixTimeMilliseconds(end).UtcDateTime.ToString("o", CultureInfo.InvariantCulture);
        return $"{baseUrl}?resource={Uri.EscapeDataString(res)}&from={Uri.EscapeDataString(fromIso)}&to={Uri.EscapeDataString(toIso)}&order=asc";
    }

    private string ResolveResource(int sensorIndex)
    {
        // If multiple sensors map to different resources, extend this mapping as needed.
        if (!string.IsNullOrEmpty(resource))
        {
            return resource;
        }

        return $"dipProj25_noise_detector_{sensorIndex}";
    }

    private void AddDefaultHeaders(System.Net.Http.Headers.HttpRequestHeaders headers)
    {
        headers.Remove("Authorization");
        headers.TryAddWithoutValidation("Authorization", "PREAUTHENTICATED");
        headers.TryAddWithoutValidation("X-Requester-Id", "digiphy1");
        headers.TryAddWithoutValidation("X-Requester-Type", "domainApplication");
        headers.TryAddWithoutValidation("Accept", "application/json");
    }

    private async Task<NoiseData[]> FetchRangeAsync(int sensorIndex, long start, long end, CancellationToken token)
    {
        string url = BuildRangeUrl(sensorIndex, start, end);
        using (HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Get, url))
        {
            AddDefaultHeaders(request.Headers);
            using (HttpResponseMessage response = await httpClient.SendAsync(request, token))
            {
                response.EnsureSuccessStatusCode();
                string json = await response.Content.ReadAsStringAsync();
                return ParseArray(json);
            }
        }
    }

    private NoiseData ParseSingle(string json)
    {
        var node = JSON.Parse(json);
        if (node == null)
        {
            return null;
        }

        node = ExtractPayloadNode(node, pickLatest: true);
        if (node == null)
        {
            return null;
        }

        return ParseEntry(node);
    }

    private NoiseData[] ParseArray(string json)
    {
        var node = JSON.Parse(json);
        if (node == null)
        {
            return Array.Empty<NoiseData>();
        }

        JSONArray array = ExtractPayloadArray(node);

        if (array == null)
        {
            NoiseData single = ParseSingle(json);
            return single == null ? Array.Empty<NoiseData>() : new[] { single };
        }

        List<NoiseData> results = new List<NoiseData>();
        foreach (var item in array)
        {
            var entry = item.Value;
            NoiseData parsed = ParseEntry(entry);
            if (parsed != null)
            {
                results.Add(parsed);
            }
        }

        return results.ToArray();
    }

    private JSONNode ExtractPayloadNode(JSONNode root, bool pickLatest)
    {
        if (root.HasKey("data"))
        {
            root = root["data"];
        }

        if (root.HasKey("contentNodes"))
        {
            var arr = root["contentNodes"] as JSONArray;
            if (arr != null)
            {
                if (arr.Count == 0)
                {
                    return null;
                }

                return pickLatest ? arr[arr.Count - 1] : arr[0];
            }
        }

        if (root is JSONArray arrayNode)
        {
            if (arrayNode.Count == 0)
            {
                return null;
            }

            return pickLatest ? arrayNode[arrayNode.Count - 1] : arrayNode[0];
        }

        return root;
    }

    private JSONArray ExtractPayloadArray(JSONNode root)
    {
        if (root.HasKey("data"))
        {
            root = root["data"];
        }

        if (root.HasKey("contentNodes"))
        {
            return root["contentNodes"].AsArray;
        }

        return root as JSONArray;
    }

    private NoiseData ParseEntry(JSONNode entry)
    {
        if (entry == null)
        {
            return null;
        }

        long timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        if (entry.HasKey("timestamp"))
        {
            timestamp = entry["timestamp"].AsLong;
        }
        else if (entry.HasKey("time"))
        {
            if (DateTimeOffset.TryParse(entry["time"], CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal, out var dto))
            {
                timestamp = dto.ToUnixTimeMilliseconds();
            }
        }

        float frequency = 0f;
        if (entry.HasKey("frequency"))
        {
            frequency = entry["frequency"].AsFloat;
        }
        else if (entry.HasKey("value"))
        {
            frequency = entry["value"].AsFloat;
        }
        else if (entry.HasKey("noise"))
        {
            frequency = entry["noise"].AsFloat;
        }

        return new NoiseData(timestamp, frequency);
    }
    
}
