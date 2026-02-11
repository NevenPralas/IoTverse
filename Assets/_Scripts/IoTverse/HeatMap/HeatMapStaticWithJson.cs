using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using XCharts.Runtime;

public class HeatMapStaticWithJson : MonoBehaviour
{
    [Header("Temperaturne vrijednosti na kutovima")]
    public float angle1 = 10.5f;
    public float angle2 = 15.4f;
    public float angle3 = 12.2f;
    public float angle4 = 10.8f;

    [Header("Postavke")]
    public int textureResolution = 1024;
    public bool autoUpdate = true;

    [Header("Mesh postavke")]
    public int meshSegmentsX = 80;
    public int meshSegmentsZ = 80;

    [Header("Globalna temperaturna ljestvica (STATIC)")]
    public float minGlobalTemp = 10f;
    public float maxGlobalTemp = 40f;

    [Header("Boje gradienta")]
    public Color coldColor = new Color(0, 0, 1);
    public Color midColor = new Color(1f, 0.5f, 0f);
    public Color warmColor = new Color(1, 0, 0);

    [Header("Forecast Time Labels (TMP)")]
    public TMP_Text labelC;
    public TMP_Text labelL1;
    public TMP_Text labelL2;
    public TMP_Text labelR1;
    public TMP_Text labelR2;

    [Header("Debug (optional TMP)")]
    public TMP_Text debugStatus;

    // Vizualni toggle
    public bool heatmapEnabled = true;

    private Mesh originalMesh;
    private Material originalMaterial;
    private Texture2D heatmapTexture;
    private Material planeMaterial;

    private float previousAngle1, previousAngle2, previousAngle3, previousAngle4;

    private bool meshGenerated = false;
    private float planeWidth;
    private float planeDepth;

    // =========================================================
    // Data Jedi TRUSTED (GET) - CURRENT
    // =========================================================
    [Header("Data Jedi TRUSTED (GET) - CURRENT")]
    [SerializeField] private string baseUrl = "https://djx.entlab.hr";
    [SerializeField] private string trustedPath = "/m2m/trusted/data";
    [SerializeField] private string usr = "FER_Departments";
    [SerializeField] private int latestNCount = 30;
    [SerializeField] private string resParamName = "res";

    [SerializeField] private string res1 = "dipProj25_temperature1";
    [SerializeField] private string res2 = "dipProj25_temperature2";
    [SerializeField] private string res3 = "dipProj25_temperature3";
    [SerializeField] private string res4 = "dipProj25_temperature4";

    [Tooltip("Key unutar timestamp objekta, kod vas 'temperature'")]
    [SerializeField] private string valueKey = "temperature";

    [Header("Data Jedi TRUSTED (GET) - FORECAST")]
    [SerializeField] private int latestNCountForecast = 30;
    [SerializeField] private string resPred1 = "dipProj25_predict_temp1";
    [SerializeField] private string resPred2 = "dipProj25_predict_temp2";
    [SerializeField] private string resPred3 = "dipProj25_predict_temp3";
    [SerializeField] private string resPred4 = "dipProj25_predict_temp4";

    [Header("Trusted headers")]
    [SerializeField] private string authorization = "PREAUTHENTICATED";
    [SerializeField] private string requesterId = "digiphy1";
    [SerializeField] private string requesterType = "domainApplication";
    [SerializeField] private string accept = "application/vnd.ericsson.simple.output+json;version=1.0";

    [Header("Polling")]
    [SerializeField] private float pollSeconds = 5.0f;
    [SerializeField] private int requestTimeoutSeconds = 15;
    [SerializeField] private bool acceptAnyCertificate = true;

    [Header("Debug")]
    [SerializeField] private bool debugLogs = true;
    [SerializeField] private int debugBodyMaxChars = 600;

    // =========================================================
    // Rolling window
    // =========================================================
    [Header("Rolling Window (shift svaki 1s)")]
    [SerializeField] private int windowSeconds = 30;
    [SerializeField] private int forecastSeconds = 29;
    [SerializeField] private int maxCachedSamplesPerSensor = 600;

    private float _rollTimer = 0f;

    [Serializable]
    private class TimestampValue
    {
        public DateTime timestampUtc;
        public float value;

        public TimestampValue(DateTime tsUtc, float val)
        {
            timestampUtc = tsUtc;
            value = val;
        }
    }

    private readonly List<TimestampValue> _sensor1Data = new List<TimestampValue>();
    private readonly List<TimestampValue> _sensor2Data = new List<TimestampValue>();
    private readonly List<TimestampValue> _sensor3Data = new List<TimestampValue>();
    private readonly List<TimestampValue> _sensor4Data = new List<TimestampValue>();

    private readonly List<TimestampValue> _pred1Data = new List<TimestampValue>();
    private readonly List<TimestampValue> _pred2Data = new List<TimestampValue>();
    private readonly List<TimestampValue> _pred3Data = new List<TimestampValue>();
    private readonly List<TimestampValue> _pred4Data = new List<TimestampValue>();

    private bool _hasLast1, _hasLast2, _hasLast3, _hasLast4;
    private float _last1, _last2, _last3, _last4;
    private DateTime _lastTs1Utc, _lastTs2Utc, _lastTs3Utc, _lastTs4Utc;

    private bool _hasPLast1, _hasPLast2, _hasPLast3, _hasPLast4;
    private float _pLast1, _pLast2, _pLast3, _pLast4;
    private DateTime _pLastTs1Utc, _pLastTs2Utc, _pLastTs3Utc, _pLastTs4Utc;

    private MeasurementData data;
    private MeasurementData forecastData;

    [Header("XCharts Referenca")]
    public GameObject lineChart;

    private Vector3 trackedPoint;
    private bool isTrackingPoint = false;

    private int cachedNowIndex = -1;
    private float chartUpdateTimer = 0f;
    private const float ChartUpdatePeriod = 1f;

    private bool forecastEnabled = false;

    private const int ForecastAxisCount = 61;

    [Header("Forecast Style (Serie 1)")]
    [SerializeField] private Color forecastColor = Color.gray;
    [SerializeField] private float dashLength = 6f;
    [SerializeField] private float gapLength = 4f;
    [SerializeField] private float dotLength = 0f;

    // internal cached components
    private bool _initialized = false;
    private MeshFilter _mf;
    private Renderer _r;
    private MeshCollider _mc;

    // Debug anti-spam / tracing
    private int _lastDataLen = -1;
    private int _lastForecastLen = -1;
    private float _pollTick = 0;
    private float _chartRefreshTick = 0;

    private void Log(string tag, string msg)
    {
        if (!debugLogs) return;
        Debug.Log($"{tag} {msg}");
    }

    private void Awake()
    {
        Log("[HEATMAP]", $"Awake() scene='{UnityEngine.SceneManagement.SceneManager.GetActiveScene().name}' obj='{name}' active={gameObject.activeInHierarchy}");
        EnsureInitialized();
    }

    private void EnsureInitialized()
    {
        if (_initialized) return;

        _mf = GetComponent<MeshFilter>();
        _r = GetComponent<Renderer>();
        _mc = GetComponent<MeshCollider>();

        if (_mf == null)
        {
            Debug.LogError("[HEATMAP] ERROR: MeshFilter missing!");
            return;
        }
        if (_r == null)
        {
            Debug.LogError("[HEATMAP] ERROR: Renderer missing!");
            return;
        }

        originalMesh = _mf.sharedMesh;
        originalMaterial = _r.sharedMaterial;

        if (originalMesh == null)
            Debug.LogError("[HEATMAP] ERROR: originalMesh is NULL (MeshFilter.sharedMesh).");
        if (originalMaterial == null)
            Debug.LogError("[HEATMAP] ERROR: originalMaterial is NULL (Renderer.sharedMaterial).");

        if (originalMesh != null)
        {
            planeWidth = originalMesh.bounds.size.x;
            planeDepth = originalMesh.bounds.size.z;
        }

        _initialized = true;

        Log("[HEATMAP]", $"Initialized OK. planeWidth={planeWidth:F3} planeDepth={planeDepth:F3} mesh='{(originalMesh != null ? originalMesh.name : "NULL")}'");
    }

    void Start()
    {
        EnsureInitialized();

        Log("[HEATMAP]", "Start() begin");
        SetForecastTmpLabelsEnabled(false);

        // chart reference diagnostics
        if (lineChart == null)
        {
            Log("[LINECHART]", "ERROR: lineChart reference is NULL in inspector!");
        }
        else
        {
            Log("[LINECHART]", $"lineChart ref OK: '{lineChart.name}' active={lineChart.activeInHierarchy}");

            var chart = lineChart.GetComponent<LineChart>();
            if (chart == null)
            {
                Log("[LINECHART]", "ERROR: lineChart GameObject has NO LineChart component!");
            }
            else
            {
                Log("[LINECHART]", $"LineChart component OK. seriesCount={chart.series.Count}");

                chart.ClearData();
                ForceYAxis10to40(chart);
                UpdateChartTitleWithDate(chart);

                EnsureTwoSeries(chart);
                ConfigureForecastSerie(chart);

                chart.RefreshChart();

                Log("[LINECHART]", "Chart initialized: ClearData + EnsureTwoSeries + RefreshChart done.");
            }
        }

        DeactivateHeatmap();
        SaveCurrentAngles();

        RebuildMeasurementDataRolling();
        RebuildForecastDataRolling();
        ApplyCornerTempsFromNow();

        Log("[HEATMAP]", $"After initial build: dataLen={(data?.measurements != null ? data.measurements.Length : 0)} " +
                        $"forecastLen={(forecastData?.measurements != null ? forecastData.measurements.Length : 0)} " +
                        $"angles=[{angle1:0.0},{angle2:0.0},{angle3:0.0},{angle4:0.0}]");

        StartCoroutine(PollDataJediLoop());
    }

    void Update()
    {
        // 1) roll tick each 1 sec
        _rollTimer += Time.deltaTime;
        if (_rollTimer >= 1f)
        {
            _rollTimer = 0f;

            RebuildMeasurementDataRolling();
            RebuildForecastDataRolling();
            ApplyCornerTempsFromNow();

            int dl = data?.measurements != null ? data.measurements.Length : 0;
            int fl = forecastData?.measurements != null ? forecastData.measurements.Length : 0;

            if (dl != _lastDataLen || fl != _lastForecastLen)
            {
                Log("[HEATMAP]", $"[ROLL-1S] rebuilt windows: dataLen={dl} forecastLen={fl} nowAngles=[{angle1:0.0},{angle2:0.0},{angle3:0.0},{angle4:0.0}]");
                _lastDataLen = dl;
                _lastForecastLen = fl;
            }
            else
            {
                Log("[HEATMAP]", $"[ROLL-1S] tick. nowAngles=[{angle1:0.0},{angle2:0.0},{angle3:0.0},{angle4:0.0}] cacheCounts curr=[{_sensor1Data.Count},{_sensor2Data.Count},{_sensor3Data.Count},{_sensor4Data.Count}] pred=[{_pred1Data.Count},{_pred2Data.Count},{_pred3Data.Count},{_pred4Data.Count}]");
            }

            if (isTrackingPoint)
            {
                Log("[LINECHART]", "[ROLL-1S] isTrackingPoint=true -> RefreshChartWindowForNow()");
                RefreshChartWindowForNow();
            }
        }

        // 2) heatmap render update only when enabled
        if (heatmapEnabled && autoUpdate && HasValuesChanged())
        {
            Log("[HEATMAP]", $"Heatmap values changed -> GenerateHeatmap() angles=[{angle1:0.0},{angle2:0.0},{angle3:0.0},{angle4:0.0}]");
            GenerateHeatmap();
            SaveCurrentAngles();
        }

        // 3) extra chart refresh tick
        if (isTrackingPoint && data != null && data.measurements != null && data.measurements.Length > 0)
        {
            chartUpdateTimer += Time.deltaTime;
            if (chartUpdateTimer >= ChartUpdatePeriod)
            {
                chartUpdateTimer = 0f;
                Log("[LINECHART]", "[CHART-1S] RefreshChartWindowForNow()");
                RefreshChartWindowForNow();
            }
        }
    }

    // =========================================================
    // Forecast toggle
    // =========================================================
    public void SetForecastEnabled(bool enabled)
    {
        forecastEnabled = enabled;
        Log("[LINECHART]", $"SetForecastEnabled({enabled})");
        SetForecastTmpLabelsEnabled(forecastEnabled);
        if (isTrackingPoint) RefreshChartWindowForNow();
    }

    public void UpdateChartForPoint(Vector3 worldPoint)
    {
        Log("[LINECHART]", $"UpdateChartForPoint(worldPoint={worldPoint}) called.");

        if (lineChart == null)
        {
            Log("[LINECHART]", "UpdateChartForPoint early-exit: lineChart is NULL");
            return;
        }
        if (data == null || data.measurements == null || data.measurements.Length == 0)
        {
            Log("[LINECHART]", $"UpdateChartForPoint early-exit: data missing. dataNull={data == null} len={(data?.measurements != null ? data.measurements.Length : 0)}");
            return;
        }

        trackedPoint = worldPoint;
        isTrackingPoint = true;

        Log("[LINECHART]", "isTrackingPoint=true -> RefreshChartWindowForNow()");
        RefreshChartWindowForNow();
    }

    // =========================================================
    // Data Jedi polling
    // =========================================================
    private IEnumerator PollDataJediLoop()
    {
        Log("[DATAJEDI]", $"PollDataJediLoop START pollSeconds={pollSeconds} baseUrl={baseUrl}");
        while (true)
        {
            _pollTick++;
            Log("[DATAJEDI]", $"--- POLL TICK #{_pollTick:0} ---");
            yield return FetchAndApplyLatestSamples();
            yield return new WaitForSeconds(pollSeconds);
        }
    }

    private IEnumerator FetchAndApplyLatestSamples()
    {
        // CURRENT
        yield return FetchOneRes(res1, 1, false);
        yield return FetchOneRes(res2, 2, false);
        yield return FetchOneRes(res3, 3, false);
        yield return FetchOneRes(res4, 4, false);

        // FORECAST
        yield return FetchOneRes(resPred1, 1, true);
        yield return FetchOneRes(resPred2, 2, true);
        yield return FetchOneRes(resPred3, 3, true);
        yield return FetchOneRes(resPred4, 4, true);

        ApplyCornerTempsFromNow();

        bool anyCurrent =
            _sensor1Data.Count > 0 ||
            _sensor2Data.Count > 0 ||
            _sensor3Data.Count > 0 ||
            _sensor4Data.Count > 0;

        bool anyPred =
            _pred1Data.Count > 0 ||
            _pred2Data.Count > 0 ||
            _pred3Data.Count > 0 ||
            _pred4Data.Count > 0;

        if (!anyCurrent)
        {
            SetStatus("[DATAJEDI] poll: NO CURRENT DATA parsed -> angles remain default");
        }
        else
        {
            string stale = BuildStaleStatus();
            string pred = anyPred ? " | forecast:OK" : " | forecast:EMPTY";
            SetStatus($"[DATAJEDI] poll OK | A1={angle1:0.0} A2={angle2:0.0} A3={angle3:0.0} A4={angle4:0.0}{stale}{pred}");
        }
    }

    private string BuildStaleStatus()
    {
        DateTime now = DateTime.UtcNow;
        int s1 = _hasLast1 ? (int)Math.Floor((now - _lastTs1Utc).TotalSeconds) : -1;
        int s2 = _hasLast2 ? (int)Math.Floor((now - _lastTs2Utc).TotalSeconds) : -1;
        int s3 = _hasLast3 ? (int)Math.Floor((now - _lastTs3Utc).TotalSeconds) : -1;
        int s4 = _hasLast4 ? (int)Math.Floor((now - _lastTs4Utc).TotalSeconds) : -1;
        return $" | age(s): [{s1},{s2},{s3},{s4}]";
    }

    private IEnumerator FetchOneRes(string resValue, int sensorIndex, bool isForecast)
    {
        int count = isForecast ? latestNCountForecast : latestNCount;

        string url =
            $"{baseUrl}{trustedPath}" +
            $"?usr={UnityWebRequest.EscapeURL(usr)}" +
            $"&latestNCount={count}" +
            $"&{UnityWebRequest.EscapeURL(resParamName)}={UnityWebRequest.EscapeURL(resValue)}";

        Log("[DATAJEDI]", $"GET {(isForecast ? "PRED" : "CURR")} sensor{sensorIndex} url='{url}'");

        using (var req = UnityWebRequest.Get(url))
        {
            req.SetRequestHeader("Authorization", authorization);
            req.SetRequestHeader("X-Requester-Id", requesterId);
            req.SetRequestHeader("X-Requester-Type", requesterType);
            req.SetRequestHeader("Accept", accept);

            if (acceptAnyCertificate)
                req.certificateHandler = new AcceptAllCertificates();

            req.timeout = requestTimeoutSeconds;

            yield return req.SendWebRequest();

            long code = req.responseCode;
            string body = req.downloadHandler != null ? req.downloadHandler.text : "";
            string bodyShort = body != null && body.Length > debugBodyMaxChars
                ? body.Substring(0, debugBodyMaxChars) + "..."
                : body;

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning($"[DATAJEDI] HTTP FAIL {(isForecast ? "PRED" : "CURR")} sensor{sensorIndex} res={resValue} code={code} err={req.error}");
                if (!string.IsNullOrEmpty(bodyShort)) Debug.LogWarning(bodyShort);
                yield break;
            }

            if (!TryParseTrustedJson(body, valueKey, out List<TimestampValue> series))
            {
                Debug.LogWarning($"[DATAJEDI] PARSE FAIL {(isForecast ? "PRED" : "CURR")} sensor{sensorIndex} res={resValue} code={code} bodyLen={(body != null ? body.Length : 0)}");
                if (!string.IsNullOrEmpty(bodyShort)) Debug.LogWarning(bodyShort);
                yield break;
            }

            if (isForecast)
                series = MaybeRemapForecastSeriesToFuture(series);

            ApplySeriesToCache(series, sensorIndex, isForecast);

            var list = GetSensorList(sensorIndex, isForecast);
            var last = list.Count > 0 ? list[list.Count - 1] : null;
            string lastLocal = last != null ? last.timestampUtc.ToLocalTime().ToString("HH:mm:ss.fff") : "N/A";
            float lastVal = last != null ? last.value : 0f;

            Log("[DATAJEDI]", $"OK {(isForecast ? "PRED" : "CURR")} sensor{sensorIndex} res={resValue} http={code} parsed={series.Count} cache={list.Count} last={lastLocal} val={lastVal:0.0}");
        }
    }

    private List<TimestampValue> GetSensorList(int sensorIndex, bool isForecast)
    {
        if (!isForecast)
        {
            return sensorIndex == 1 ? _sensor1Data :
                   sensorIndex == 2 ? _sensor2Data :
                   sensorIndex == 3 ? _sensor3Data :
                                      _sensor4Data;
        }

        return sensorIndex == 1 ? _pred1Data :
               sensorIndex == 2 ? _pred2Data :
               sensorIndex == 3 ? _pred3Data :
                                  _pred4Data;
    }

    private bool TryParseTrustedJson(string json, string key, out List<TimestampValue> series)
    {
        series = null;
        if (string.IsNullOrWhiteSpace(json)) return false;

        object rootObj = MiniJSON.Json.Deserialize(json);
        if (!(rootObj is List<object> arr) || arr.Count == 0) return false;
        if (!(arr[0] is Dictionary<string, object> root)) return false;

        Dictionary<string, object> gwNode = null;
        foreach (var kv in root)
        {
            if (kv.Key.Equals("deviceId", StringComparison.OrdinalIgnoreCase)) continue;
            gwNode = kv.Value as Dictionary<string, object>;
            if (gwNode != null) break;
        }
        if (gwNode == null) return false;

        Dictionary<string, object> sensorNode = null;
        foreach (var kv in gwNode)
        {
            sensorNode = kv.Value as Dictionary<string, object>;
            if (sensorNode != null) break;
        }
        if (sensorNode == null) return false;

        var list = new List<TimestampValue>();

        foreach (var kv in sensorNode)
        {
            string iso = kv.Key;
            if (!(kv.Value is Dictionary<string, object> vdict)) continue;

            float v = GetFloat(vdict, key);
            DateTime utc = ParseIsoToUtc(iso);
            if (utc == DateTime.MinValue) continue;

            list.Add(new TimestampValue(utc, v));
        }

        if (list.Count == 0) return false;

        list.Sort((a, b) => a.timestampUtc.CompareTo(b.timestampUtc));
        series = list;
        return true;
    }

    private List<TimestampValue> MaybeRemapForecastSeriesToFuture(List<TimestampValue> series)
    {
        if (series == null || series.Count == 0) return series;

        DateTime nowUtc = DateTime.UtcNow;
        DateTime maxTs = series[series.Count - 1].timestampUtc;

        if (maxTs <= nowUtc.AddSeconds(1))
        {
            double stepSec = EstimateMedianStepSeconds(series);
            if (stepSec < 1) stepSec = 5;

            var remapped = new List<TimestampValue>(series.Count);
            for (int i = 0; i < series.Count; i++)
            {
                DateTime ts = nowUtc.AddSeconds((i + 1) * stepSec);
                remapped.Add(new TimestampValue(ts, series[i].value));
            }
            remapped.Sort((a, b) => a.timestampUtc.CompareTo(b.timestampUtc));

            Log("[DATAJEDI]", $"Remapped forecast series to future: count={series.Count} stepSec~{stepSec:0.0}");
            return remapped;
        }

        return series;
    }

    private double EstimateMedianStepSeconds(List<TimestampValue> series)
    {
        if (series == null || series.Count < 2) return 5;

        var diffs = new List<double>();
        for (int i = 1; i < series.Count; i++)
        {
            double d = (series[i].timestampUtc - series[i - 1].timestampUtc).TotalSeconds;
            if (d > 0.2 && d < 600) diffs.Add(d);
        }

        if (diffs.Count == 0) return 5;

        diffs.Sort();
        int mid = diffs.Count / 2;
        return diffs.Count % 2 == 1 ? diffs[mid] : (diffs[mid - 1] + diffs[mid]) / 2.0;
    }

    private void ApplySeriesToCache(List<TimestampValue> series, int sensorIndex, bool isForecast)
    {
        if (series == null || series.Count == 0) return;

        var targetList = GetSensorList(sensorIndex, isForecast);

        foreach (var tv in series)
        {
            bool exists = false;
            for (int i = targetList.Count - 1; i >= 0; i--)
            {
                if ((tv.timestampUtc - targetList[i].timestampUtc).TotalSeconds > 10)
                    break;

                if (Math.Abs((targetList[i].timestampUtc - tv.timestampUtc).TotalSeconds) < 0.5)
                {
                    exists = true;
                    break;
                }
            }
            if (!exists) targetList.Add(tv);
        }

        targetList.Sort((a, b) => a.timestampUtc.CompareTo(b.timestampUtc));

        while (targetList.Count > maxCachedSamplesPerSensor)
            targetList.RemoveAt(0);

        if (targetList.Count > 0)
        {
            var last = targetList[targetList.Count - 1];

            if (!isForecast)
            {
                if (sensorIndex == 1) { _hasLast1 = true; _last1 = last.value; _lastTs1Utc = last.timestampUtc; }
                if (sensorIndex == 2) { _hasLast2 = true; _last2 = last.value; _lastTs2Utc = last.timestampUtc; }
                if (sensorIndex == 3) { _hasLast3 = true; _last3 = last.value; _lastTs3Utc = last.timestampUtc; }
                if (sensorIndex == 4) { _hasLast4 = true; _last4 = last.value; _lastTs4Utc = last.timestampUtc; }
            }
            else
            {
                if (sensorIndex == 1) { _hasPLast1 = true; _pLast1 = last.value; _pLastTs1Utc = last.timestampUtc; }
                if (sensorIndex == 2) { _hasPLast2 = true; _pLast2 = last.value; _pLastTs2Utc = last.timestampUtc; }
                if (sensorIndex == 3) { _hasPLast3 = true; _pLast3 = last.value; _pLastTs3Utc = last.timestampUtc; }
                if (sensorIndex == 4) { _hasPLast4 = true; _pLast4 = last.value; _pLastTs4Utc = last.timestampUtc; }
            }
        }

        Log("[DATAJEDI]", $"Cache updated {(isForecast ? "PRED" : "CURR")} sensor{sensorIndex}: cacheCount={targetList.Count}");
    }

    private void RebuildMeasurementDataRolling()
    {
        int n = Mathf.Max(2, windowSeconds);
        DateTime nowUtc = DateTime.UtcNow;

        data = new MeasurementData();
        data.measurements = new Measurement[n];

        for (int i = 0; i < n; i++)
        {
            DateTime ts = nowUtc.AddSeconds(-(n - 1 - i));
            float v1 = FindValueAtOrBefore(_sensor1Data, ts, _hasLast1, _last1);
            float v2 = FindValueAtOrBefore(_sensor2Data, ts, _hasLast2, _last2);
            float v3 = FindValueAtOrBefore(_sensor3Data, ts, _hasLast3, _last3);
            float v4 = FindValueAtOrBefore(_sensor4Data, ts, _hasLast4, _last4);

            data.measurements[i] = new Measurement
            {
                id = i,
                timestamp = ts.ToString("o"),
                temperature1 = v1,
                temperature2 = v2,
                temperature3 = v3,
                temperature4 = v4
            };
        }
    }

    private void RebuildForecastDataRolling()
    {
        int n = Mathf.Max(2, windowSeconds);
        int future = Mathf.Max(1, forecastSeconds);
        int total = n + future;

        DateTime nowUtc = DateTime.UtcNow;

        forecastData = new MeasurementData();
        forecastData.measurements = new Measurement[total];

        for (int i = 0; i < n; i++)
        {
            DateTime ts = nowUtc.AddSeconds(-(n - 1 - i));
            float v1 = FindValueAtOrBefore(_sensor1Data, ts, _hasLast1, _last1);
            float v2 = FindValueAtOrBefore(_sensor2Data, ts, _hasLast2, _last2);
            float v3 = FindValueAtOrBefore(_sensor3Data, ts, _hasLast3, _last3);
            float v4 = FindValueAtOrBefore(_sensor4Data, ts, _hasLast4, _last4);

            forecastData.measurements[i] = new Measurement
            {
                id = i,
                timestamp = ts.ToString("o"),
                temperature1 = v1,
                temperature2 = v2,
                temperature3 = v3,
                temperature4 = v4
            };
        }

        for (int i = 1; i <= future; i++)
        {
            DateTime ts = nowUtc.AddSeconds(i);

            float p1 = FindValueAtOrBefore(_pred1Data, ts, _hasPLast1, _pLast1);
            float p2 = FindValueAtOrBefore(_pred2Data, ts, _hasPLast2, _pLast2);
            float p3 = FindValueAtOrBefore(_pred3Data, ts, _hasPLast3, _pLast3);
            float p4 = FindValueAtOrBefore(_pred4Data, ts, _hasPLast4, _pLast4);

            if (!_hasPLast1 && _hasLast1) p1 = _last1;
            if (!_hasPLast2 && _hasLast2) p2 = _last2;
            if (!_hasPLast3 && _hasLast3) p3 = _last3;
            if (!_hasPLast4 && _hasLast4) p4 = _last4;

            int idx = (n - 1) + i;
            if (idx < 0 || idx >= forecastData.measurements.Length) continue;

            forecastData.measurements[idx] = new Measurement
            {
                id = idx,
                timestamp = ts.ToString("o"),
                temperature1 = p1,
                temperature2 = p2,
                temperature3 = p3,
                temperature4 = p4
            };
        }
    }

    private float FindValueAtOrBefore(List<TimestampValue> list, DateTime targetTsUtc, bool hasFallback, float fallback)
    {
        if (list == null || list.Count == 0)
            return hasFallback ? fallback : 0f;

        int lo = 0;
        int hi = list.Count - 1;
        int best = -1;

        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (list[mid].timestampUtc <= targetTsUtc)
            {
                best = mid;
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        if (best >= 0) return list[best].value;
        return hasFallback ? fallback : list[0].value;
    }

    private void ApplyCornerTempsFromNow()
    {
        if (data == null || data.measurements == null || data.measurements.Length == 0)
            return;

        var m = data.measurements[data.measurements.Length - 1];
        angle1 = m.temperature1;
        angle2 = m.temperature2;
        angle3 = m.temperature3;
        angle4 = m.temperature4;
    }

    private void RefreshChartWindowForNow()
    {
        if (lineChart == null)
        {
            Log("[LINECHART]", "RefreshChartWindowForNow early-exit: lineChart NULL");
            return;
        }
        if (data == null || data.measurements == null || data.measurements.Length == 0)
        {
            Log("[LINECHART]", "RefreshChartWindowForNow early-exit: data missing");
            return;
        }
        if (!isTrackingPoint)
        {
            Log("[LINECHART]", "RefreshChartWindowForNow called but isTrackingPoint=false (unexpected).");
        }

        int nowIdx = data.measurements.Length - 1;
        if (nowIdx != cachedNowIndex)
        {
            cachedNowIndex = nowIdx;
            ApplyCornerTempsForIndex(nowIdx);
        }

        // UV from point
        Vector3 local = transform.InverseTransformPoint(trackedPoint);
        float u = (local.x / (planeWidth * transform.lossyScale.x)) + 0.5f;
        float v = (local.z / (planeDepth * transform.lossyScale.z)) + 0.5f;
        u = Mathf.Clamp01(u);
        v = Mathf.Clamp01(v);

        var chart = lineChart.GetComponent<LineChart>();
        if (chart == null)
        {
            Log("[LINECHART]", "ERROR: lineChart has no LineChart component at runtime.");
            return;
        }

        chart.ClearData();
        EnsureTwoSeries(chart);
        ConfigureForecastSerie(chart);

        ForceYAxis10to40(chart);
        UpdateChartTitleWithDate(chart);

        if (!forecastEnabled)
        {
            SetForecastTmpLabelsEnabled(false);

            int endIdx = nowIdx;
            int startIdx = Mathf.Max(0, endIdx - 29);
            int pointCount = (endIdx - startIdx) + 1;

            int xCategoryCount = pointCount + 1;

            Log("[LINECHART]", $"[HIST] Building chart: startIdx={startIdx} endIdx={endIdx} pointCount={pointCount} xCategoryCount={xCategoryCount} uv=({u:F3},{v:F3})");

            for (int xi = 0; xi < xCategoryCount; xi++)
            {
                string xLabel = new string('\u200B', xi + 1);
                chart.AddXAxisData(xLabel);
            }

            int addedPoints = 0;

            for (int i = 0; i < pointCount; i++)
            {
                int idx = Mathf.Clamp(startIdx + i, 0, data.measurements.Length - 1);
                Measurement m = data.measurements[idx];

                float y = BilinearInterpolation(u, v, m.temperature1, m.temperature2, m.temperature3, m.temperature4);

                chart.AddData(0, y);
                chart.AddData(1, 0f);

                addedPoints++;

                if (i == pointCount - 1)
                    UpdateTemperatureText(y);
            }

            Log("[LINECHART]", $"[HIST] Added data points: {addedPoints} (serie0) + {addedPoints} dummy(serie1)");

            ResetSerieData(chart, 0);
            ResetSerieData(chart, 1);
            IgnoreAllPoints(chart, 1);
            SetPointRed(chart, 0, pointCount - 1);

            var xAxis = chart.EnsureChartComponent<XAxis>();
            xAxis.type = Axis.AxisType.Category;
            xAxis.axisLabel.formatter = "{value}";
            xAxis.interval = 0;

            chart.RefreshChart();

            _chartRefreshTick++;
            Log("[LINECHART]", $"[HIST] RefreshChart DONE tick#{_chartRefreshTick:0} chartSeriesCount={chart.series.Count}");
            return;
        }

        // Forecast ON
        SetForecastTmpLabelsEnabled(true);

        Log("[LINECHART]", $"[FORECAST] Building 61-point chart uv=({u:F3},{v:F3})");

        for (int xi = 0; xi < ForecastAxisCount; xi++)
            chart.AddXAxisData(new string('\u200B', xi + 1));

        DateTime nowUtc = DateTime.UtcNow;

        for (int i = 0; i < ForecastAxisCount; i++)
        {
            float y;

            if (i <= 29)
            {
                DateTime ts = nowUtc.AddSeconds(-(29 - i));
                float v1 = FindValueAtOrBefore(_sensor1Data, ts, _hasLast1, _last1);
                float v2 = FindValueAtOrBefore(_sensor2Data, ts, _hasLast2, _last2);
                float v3 = FindValueAtOrBefore(_sensor3Data, ts, _hasLast3, _last3);
                float v4 = FindValueAtOrBefore(_sensor4Data, ts, _hasLast4, _last4);
                y = BilinearInterpolation(u, v, v1, v2, v3, v4);
            }
            else if (i <= 58)
            {
                int offs = i - 29;
                DateTime ts = nowUtc.AddSeconds(offs);

                float p1 = FindValueAtOrBefore(_pred1Data, ts, _hasPLast1, _pLast1);
                float p2 = FindValueAtOrBefore(_pred2Data, ts, _hasPLast2, _pLast2);
                float p3 = FindValueAtOrBefore(_pred3Data, ts, _hasPLast3, _pLast3);
                float p4 = FindValueAtOrBefore(_pred4Data, ts, _hasPLast4, _pLast4);

                if (!_hasPLast1 && _hasLast1) p1 = _last1;
                if (!_hasPLast2 && _hasLast2) p2 = _last2;
                if (!_hasPLast3 && _hasLast3) p3 = _last3;
                if (!_hasPLast4 && _hasLast4) p4 = _last4;

                y = BilinearInterpolation(u, v, p1, p2, p3, p4);
            }
            else
            {
                y = 0f;
            }

            chart.AddData(0, y);
            chart.AddData(1, y);
        }

        ResetSerieData(chart, 0);
        ResetSerieData(chart, 1);

        UpdateTemperatureText(GetSerieY(chart, 0, 29));

        SetIgnoreRange(chart, 0, 30, 60, true);
        SetIgnoreRange(chart, 1, 0, 28, true);
        SetIgnoreRange(chart, 1, 59, 60, true);

        SetPointRed(chart, 0, 29);
        HideAllSymbols(chart, 1);

        UpdateForecastTmpTextsFromNow();

        var xAxisF = chart.EnsureChartComponent<XAxis>();
        xAxisF.type = Axis.AxisType.Category;
        xAxisF.axisLabel.formatter = "{value}";
        xAxisF.interval = 0;

        chart.RefreshChart();

        _chartRefreshTick++;
        Log("[LINECHART]", $"[FORECAST] RefreshChart DONE tick#{_chartRefreshTick:0} currCache=[{_sensor1Data.Count},{_sensor2Data.Count},{_sensor3Data.Count},{_sensor4Data.Count}] predCache=[{_pred1Data.Count},{_pred2Data.Count},{_pred3Data.Count},{_pred4Data.Count}]");
    }

    private void ApplyCornerTempsForIndex(int idx)
    {
        if (data == null || data.measurements == null || data.measurements.Length == 0) return;
        idx = Mathf.Clamp(idx, 0, data.measurements.Length - 1);

        Measurement m = data.measurements[idx];
        angle1 = m.temperature1;
        angle2 = m.temperature2;
        angle3 = m.temperature3;
        angle4 = m.temperature4;
    }

    private void SetForecastTmpLabelsEnabled(bool enabled)
    {
        if (labelC != null) labelC.enabled = enabled;
        if (labelL1 != null) labelL1.enabled = enabled;
        if (labelL2 != null) labelL2.enabled = enabled;
        if (labelR1 != null) labelR1.enabled = enabled;
        if (labelR2 != null) labelR2.enabled = enabled;
    }

    private void UpdateForecastTmpTextsFromNow()
    {
        DateTime nowLocal = DateTime.Now;

        DateTime tC = nowLocal;
        DateTime tL2 = nowLocal.AddSeconds(-29);
        DateTime tL1 = nowLocal.AddSeconds(-15);
        DateTime tR1 = nowLocal.AddSeconds(+15);
        DateTime tR2 = nowLocal.AddSeconds(+29);

        if (labelC != null) labelC.text = tC.ToString("HH:mm:ss");
        if (labelL2 != null) labelL2.text = tL2.ToString("HH:mm:ss");
        if (labelL1 != null) labelL1.text = tL1.ToString("HH:mm:ss");
        if (labelR1 != null) labelR1.text = tR1.ToString("HH:mm:ss");
        if (labelR2 != null) labelR2.text = tR2.ToString("HH:mm:ss");
    }

    private void ResetSerieData(LineChart chart, int serieIndex)
    {
        if (chart == null || chart.series == null || chart.series.Count <= serieIndex) return;
        var serie = chart.series[serieIndex];
        if (serie == null) return;

        for (int i = 0; i < serie.dataCount; i++)
        {
            var d = serie.data[i];
            d.ignore = false;
            d.state = SerieState.Normal;

            var item = d.EnsureComponent<ItemStyle>();
            item.show = true;
            item.opacity = 1f;

            var sym = d.EnsureComponent<SerieSymbol>();
            sym.show = true;
        }
    }

    private void EnsureTwoSeries(LineChart chart)
    {
        if (chart == null) return;
        if (chart.series.Count == 0) chart.AddSerie<Line>("Historical");
        if (chart.series.Count < 2) chart.AddSerie<Line>("Forecast");
    }

    private void ConfigureForecastSerie(LineChart chart)
    {
        if (chart == null || chart.series == null || chart.series.Count < 2) return;

        var s = chart.series[1];
        s.itemStyle.show = true;
        s.itemStyle.color = forecastColor;

        s.lineStyle.show = true;
        s.lineStyle.color = forecastColor;
        s.lineStyle.dashLength = dashLength;
        s.lineStyle.gapLength = gapLength;
        s.lineStyle.dotLength = dotLength;

        s.symbol.show = true;
        s.symbol.color = forecastColor;
    }

    private void HideAllSymbols(LineChart chart, int serieIndex)
    {
        if (chart == null || chart.series == null || chart.series.Count <= serieIndex) return;
        var serie = chart.series[serieIndex];
        if (serie == null) return;

        for (int i = 0; i < serie.dataCount; i++)
        {
            var d = serie.data[i];
            var sym = d.EnsureComponent<SerieSymbol>();
            sym.show = false;
        }
    }

    private void IgnoreAllPoints(LineChart chart, int serieIndex)
    {
        if (chart == null || chart.series == null || chart.series.Count <= serieIndex) return;
        var serie = chart.series[serieIndex];
        if (serie == null) return;

        for (int i = 0; i < serie.dataCount; i++)
            serie.data[i].ignore = true;
    }

    private void SetIgnoreRange(LineChart chart, int serieIndex, int fromInclusive, int toInclusive, bool ignore)
    {
        if (chart == null || chart.series == null || chart.series.Count <= serieIndex) return;
        var serie = chart.series[serieIndex];
        if (serie == null || serie.dataCount == 0) return;

        int from = Mathf.Clamp(fromInclusive, 0, serie.dataCount - 1);
        int to = Mathf.Clamp(toInclusive, 0, serie.dataCount - 1);
        if (to < from) return;

        for (int i = from; i <= to; i++)
            serie.data[i].ignore = ignore;
    }

    private void SetPointRed(LineChart chart, int serieIndex, int pointIndex)
    {
        if (chart == null || chart.series == null || chart.series.Count <= serieIndex) return;
        var serie = chart.series[serieIndex];
        if (serie == null || serie.dataCount == 0) return;

        int idx = Mathf.Clamp(pointIndex, 0, serie.dataCount - 1);

        for (int i = 0; i < serie.dataCount; i++)
            serie.data[i].state = SerieState.Normal;

        var d = serie.data[idx];
        d.state = SerieState.Emphasis;

        var item = d.EnsureComponent<ItemStyle>();
        item.show = true;
        item.color = Color.red;

        var sym = d.EnsureComponent<SerieSymbol>();
        sym.show = true;
        sym.color = Color.red;
    }

    private float GetSerieY(LineChart chart, int serieIndex, int pointIndex)
    {
        if (chart == null || chart.series == null || chart.series.Count <= serieIndex) return 0f;
        var serie = chart.series[serieIndex];
        if (serie == null || serie.dataCount <= pointIndex) return 0f;
        return (float)serie.data[pointIndex].data[1];
    }

    private void ForceYAxis10to40(LineChart chart)
    {
        var yAxis = chart.EnsureChartComponent<YAxis>();
        yAxis.minMaxType = Axis.AxisMinMaxType.Custom;
        yAxis.min = 10f;
        yAxis.max = 40f;
        yAxis.interval = 10f;
        yAxis.splitNumber = 3;
        yAxis.axisLabel.show = true;
        yAxis.axisLabel.formatter = "{value}";
    }

    private void UpdateChartTitleWithDate(LineChart chart)
    {
        var title = chart.EnsureChartComponent<Title>();
        title.text = $"Time Line - {DateTime.Now:dd/MM/yyyy}";
    }

    // Compatibility
    public float GetTemperatureAtUV(Vector2 uv)
    {
        return BilinearInterpolation(uv.x, uv.y, angle1, angle2, angle3, angle4);
    }

    public float GetTemperatureAtPointWorld(Vector3 worldPoint)
    {
        Vector3 local = transform.InverseTransformPoint(worldPoint);
        float uu = (local.x / (planeWidth * transform.lossyScale.x)) + 0.5f;
        float vv = (local.z / (planeDepth * transform.lossyScale.z)) + 0.5f;
        return GetTemperatureAtUV(new Vector2(Mathf.Clamp01(uu), Mathf.Clamp01(vv)));
    }

    public void UpdateTemperatureText(float temperature)
    {
        Log("[TEXT]", $"UpdateTemperatureText({temperature:F2}) -> finding AimOnGrip");

        AimOnGrip aimScript = FindObjectOfType<AimOnGrip>(true);
        if (aimScript == null)
        {
            Log("[TEXT]", "ERROR: AimOnGrip NOT found in scene (FindObjectOfType returned null).");
            return;
        }

        Log("[TEXT]", $"AimOnGrip found: '{aimScript.name}' active={aimScript.gameObject.activeInHierarchy}. Calling UpdateTemperatureDisplay...");
        aimScript.UpdateTemperatureDisplay(temperature);
    }

    // Heatmap toggles
    public void ToggleHeatmap() => SetHeatmapEnabled(!heatmapEnabled);

    public void SetHeatmapEnabled(bool enabled)
    {
        EnsureInitialized();
        if (!_initialized) return;

        if (enabled == heatmapEnabled) return;

        if (enabled) ActivateHeatmap();
        else DeactivateHeatmap();
    }

    private void ActivateHeatmap()
    {
        EnsureInitialized();
        if (!_initialized) return;
        if (originalMesh == null || originalMaterial == null)
        {
            Debug.LogError("[HEATMAP] ERROR: Cannot activate heatmap (originalMesh/originalMaterial invalid).");
            return;
        }

        heatmapEnabled = true;
        GenerateMeshFromExistingPlane();

        heatmapTexture = new Texture2D(textureResolution, textureResolution);
        heatmapTexture.filterMode = FilterMode.Bilinear;
        heatmapTexture.wrapMode = TextureWrapMode.Clamp;

        if (_r != null)
        {
            planeMaterial = new Material(Shader.Find("Standard"));
            planeMaterial.mainTexture = heatmapTexture;
            planeMaterial.SetFloat("_Metallic", 0f);
            planeMaterial.SetFloat("_Glossiness", 0.2f);
            _r.material = planeMaterial;
        }

        GenerateHeatmap();
        Log("[HEATMAP]", "ActivateHeatmap OK -> texture/material applied.");
    }

    private void DeactivateHeatmap()
    {
        EnsureInitialized();
        if (!_initialized) return;

        heatmapEnabled = false;

        if (_r != null && originalMaterial != null)
            _r.sharedMaterial = originalMaterial;

        if (_mf != null && originalMesh != null)
            _mf.sharedMesh = originalMesh;

        if (_mc != null && originalMesh != null)
            _mc.sharedMesh = originalMesh;

        meshGenerated = false;
        Log("[HEATMAP]", "DeactivateHeatmap done.");
    }

    public void GenerateMeshFromExistingPlane()
    {
        EnsureInitialized();
        if (!_initialized) return;

        MeshFilter meshFilter = _mf != null ? _mf : GetComponent<MeshFilter>();
        if (meshFilter == null) meshFilter = gameObject.AddComponent<MeshFilter>();

        Mesh mesh = new Mesh();
        mesh.name = "HeatMap Mesh";

        int vertCountX = meshSegmentsX + 1;
        int vertCountZ = meshSegmentsZ + 1;

        Vector3[] vertices = new Vector3[vertCountX * vertCountZ];
        Vector2[] uv = new Vector2[vertices.Length];
        Color[] colors = new Color[vertices.Length];

        for (int z = 0; z <= meshSegmentsZ; z++)
        {
            for (int x = 0; x <= meshSegmentsX; x++)
            {
                int i = z * vertCountX + x;
                float uu = x / (float)meshSegmentsX;
                float vv = z / (float)meshSegmentsZ;

                float baseY = 0f;
                if (originalMesh != null && originalMesh.vertices != null && originalMesh.vertices.Length > 0)
                    baseY = originalMesh.vertices[Mathf.Clamp(i, 0, originalMesh.vertices.Length - 1)].y;

                vertices[i] = new Vector3(
                    (uu - 0.5f) * planeWidth,
                    baseY,
                    (vv - 0.5f) * planeDepth
                );

                uv[i] = new Vector2(uu, vv);
                colors[i] = GetColorFromTemperature(BilinearInterpolation(uu, vv, angle1, angle2, angle3, angle4));
            }
        }

        int[] triangles = new int[meshSegmentsX * meshSegmentsZ * 6];
        int t = 0;
        for (int z = 0; z < meshSegmentsZ; z++)
        {
            for (int x = 0; x < meshSegmentsX; x++)
            {
                int i = z * vertCountX + x;
                triangles[t++] = i; triangles[t++] = i + vertCountX; triangles[t++] = i + 1;
                triangles[t++] = i + 1; triangles[t++] = i + vertCountX; triangles[t++] = i + vertCountX + 1;
            }
        }

        mesh.vertices = vertices;
        mesh.uv = uv;
        mesh.colors = colors;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();

        meshFilter.sharedMesh = mesh;

        if (_mc == null) _mc = GetComponent<MeshCollider>();
        if (_mc == null) _mc = gameObject.AddComponent<MeshCollider>();
        _mc.sharedMesh = mesh;

        meshGenerated = true;

        Log("[HEATMAP]", $"GenerateMeshFromExistingPlane OK. verts={vertices.Length} tris={triangles.Length / 3}");
    }

    void SaveCurrentAngles()
    {
        previousAngle1 = angle1;
        previousAngle2 = angle2;
        previousAngle3 = angle3;
        previousAngle4 = angle4;
    }

    bool HasValuesChanged()
    {
        return !Mathf.Approximately(angle1, previousAngle1) ||
               !Mathf.Approximately(angle2, previousAngle2) ||
               !Mathf.Approximately(angle3, previousAngle3) ||
               !Mathf.Approximately(angle4, previousAngle4);
    }

    void GenerateHeatmap()
    {
        if (heatmapTexture == null) return;

        for (int y = 0; y < textureResolution; y++)
        {
            for (int x = 0; x < textureResolution; x++)
            {
                float uu = x / (float)(textureResolution - 1);
                float vv = y / (float)(textureResolution - 1);
                heatmapTexture.SetPixel(x, y, GetColorFromTemperature(
                    BilinearInterpolation(uu, vv, angle1, angle2, angle3, angle4)));
            }
        }
        heatmapTexture.Apply();
        if (meshGenerated) UpdateVertexColors();
    }

    Color GetColorFromTemperature(float temp)
    {
        float tt = Mathf.InverseLerp(minGlobalTemp, maxGlobalTemp, temp);
        if (tt < 0.5f) return Color.Lerp(coldColor, midColor, tt * 2f);
        else return Color.Lerp(midColor, warmColor, (tt - 0.5f) * 2f);
    }

    void UpdateVertexColors()
    {
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        Color[] colors = mesh.colors;
        Vector2[] uvs = mesh.uv;

        for (int i = 0; i < colors.Length; i++)
            colors[i] = GetColorFromTemperature(BilinearInterpolation(uvs[i].x, uvs[i].y, angle1, angle2, angle3, angle4));

        mesh.colors = colors;
    }

    float BilinearInterpolation(float uu, float vv, float q11, float q21, float q12, float q22)
    {
        return Mathf.Lerp(Mathf.Lerp(q11, q21, uu), Mathf.Lerp(q12, q22, uu), vv);
    }

    private float GetFloat(Dictionary<string, object> dict, string key)
    {
        if (!dict.TryGetValue(key, out object obj)) return 0f;
        if (obj is long l) return l;
        if (obj is int i) return i;
        if (obj is double d) return (float)d;
        if (obj is float f) return f;
        if (obj is string s && float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out float r)) return r;
        return 0f;
    }

    private DateTime ParseIsoToUtc(string iso)
    {
        if (DateTime.TryParse(iso, CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dt))
            return dt.ToUniversalTime();
        return DateTime.MinValue;
    }

    private void SetStatus(string msg)
    {
        if (debugStatus != null) debugStatus.text = msg;
        if (debugLogs) Debug.Log(msg);
    }

    private class AcceptAllCertificates : CertificateHandler
    {
        protected override bool ValidateCertificate(byte[] certificateData) => true;
    }

    [Serializable]
    public class MeasurementData { public Measurement[] measurements; }

    [Serializable]
    public class Measurement
    {
        public int id;
        public string timestamp;
        public float temperature1;
        public float temperature2;
        public float temperature3;
        public float temperature4;
    }

    // ===== MiniJSON =====
    private static class MiniJSON
    {
        public static class Json
        {
            public static object Deserialize(string json)
            {
                if (json == null) return null;
                return Parser.Parse(json);
            }

            private sealed class Parser : IDisposable
            {
                private const string WORD_BREAK = "{}[],:\"";
                private StringReader json;

                private Parser(string jsonString) { json = new StringReader(jsonString); }

                public static object Parse(string jsonString)
                {
                    using (var instance = new Parser(jsonString))
                        return instance.ParseValue();
                }

                public void Dispose() { json = null; }

                private Dictionary<string, object> ParseObject()
                {
                    var table = new Dictionary<string, object>();
                    json.Read(); // '{'

                    while (true)
                    {
                        Token nextToken = NextToken;

                        if (nextToken == Token.NONE) return null;
                        if (nextToken == Token.CURLY_CLOSE)
                        {
                            json.Read(); // '}'
                            return table;
                        }

                        if (nextToken != Token.STRING) return null;
                        string name = ParseString();
                        if (name == null) return null;

                        if (NextToken != Token.COLON) return null;
                        json.Read(); // ':'

                        object value = ParseValue();
                        table[name] = value;

                        Token after = NextToken;
                        if (after == Token.COMMA)
                        {
                            json.Read(); // ','
                            continue;
                        }
                        if (after == Token.CURLY_CLOSE)
                        {
                            json.Read(); // '}'
                            return table;
                        }

                        return null;
                    }
                }

                private List<object> ParseArray()
                {
                    var array = new List<object>();
                    json.Read(); // '['

                    while (true)
                    {
                        Token nextToken = NextToken;
                        if (nextToken == Token.NONE) return null;
                        if (nextToken == Token.SQUARE_CLOSE)
                        {
                            json.Read(); // ']'
                            break;
                        }

                        array.Add(ParseValue());

                        nextToken = NextToken;
                        if (nextToken == Token.COMMA) json.Read();
                        else if (nextToken == Token.SQUARE_CLOSE) { }
                        else return null;
                    }

                    return array;
                }

                private object ParseValue()
                {
                    switch (NextToken)
                    {
                        case Token.STRING: return ParseString();
                        case Token.NUMBER: return ParseNumber();
                        case Token.CURLY_OPEN: return ParseObject();
                        case Token.SQUARE_OPEN: return ParseArray();
                        case Token.TRUE: ConsumeWord("true"); return true;
                        case Token.FALSE: ConsumeWord("false"); return false;
                        case Token.NULL: ConsumeWord("null"); return null;
                        default: return null;
                    }
                }

                private void ConsumeWord(string w)
                {
                    for (int i = 0; i < w.Length; i++) json.Read();
                }

                private string ParseString()
                {
                    var s = "";
                    json.Read(); // '"'
                    while (true)
                    {
                        if (json.Peek() == -1) break;
                        char c = NextChar;
                        if (c == '"') break;

                        if (c == '\\')
                        {
                            if (json.Peek() == -1) break;
                            c = NextChar;
                            if (c == '"') s += '"';
                            else if (c == '\\') s += '\\';
                            else if (c == '/') s += '/';
                            else if (c == 'b') s += '\b';
                            else if (c == 'f') s += '\f';
                            else if (c == 'n') s += '\n';
                            else if (c == 'r') s += '\r';
                            else if (c == 't') s += '\t';
                            else if (c == 'u')
                            {
                                var hex = new char[4];
                                for (int i = 0; i < 4; i++) hex[i] = NextChar;
                                s += (char)Convert.ToInt32(new string(hex), 16);
                            }
                        }
                        else s += c;
                    }
                    return s;
                }

                private object ParseNumber()
                {
                    string number = NextWord;
                    if (number.IndexOf('.') == -1)
                    {
                        if (long.TryParse(number, NumberStyles.Any, CultureInfo.InvariantCulture, out long parsedInt))
                            return parsedInt;
                        return 0L;
                    }
                    if (double.TryParse(number, NumberStyles.Any, CultureInfo.InvariantCulture, out double parsedDouble))
                        return parsedDouble;
                    return 0.0;
                }

                private void EatWhitespace()
                {
                    while (json.Peek() != -1 && char.IsWhiteSpace(PeekChar))
                        json.Read();
                }

                private char PeekChar => Convert.ToChar(json.Peek());
                private char NextChar => Convert.ToChar(json.Read());

                private string NextWord
                {
                    get
                    {
                        var word = "";
                        while (json.Peek() != -1 && !IsWordBreak(PeekChar))
                            word += NextChar;
                        return word;
                    }
                }

                private Token NextToken
                {
                    get
                    {
                        EatWhitespace();
                        if (json.Peek() == -1) return Token.NONE;

                        char c = PeekChar;
                        switch (c)
                        {
                            case '{': return Token.CURLY_OPEN;
                            case '}': return Token.CURLY_CLOSE;
                            case '[': return Token.SQUARE_OPEN;
                            case ']': return Token.SQUARE_CLOSE;
                            case ',': return Token.COMMA;
                            case '"': return Token.STRING;
                            case ':': return Token.COLON;
                            case '0':
                            case '1':
                            case '2':
                            case '3':
                            case '4':
                            case '5':
                            case '6':
                            case '7':
                            case '8':
                            case '9':
                            case '-': return Token.NUMBER;
                        }

                        string word = NextWord;
                        if (word == "false") return Token.FALSE;
                        if (word == "true") return Token.TRUE;
                        if (word == "null") return Token.NULL;
                        return Token.NONE;
                    }
                }

                private static bool IsWordBreak(char c) => char.IsWhiteSpace(c) || WORD_BREAK.IndexOf(c) != -1;

                private enum Token
                {
                    NONE, CURLY_OPEN, CURLY_CLOSE, SQUARE_OPEN, SQUARE_CLOSE,
                    COLON, COMMA, STRING, NUMBER, TRUE, FALSE, NULL
                }
            }

            private sealed class StringReader : IDisposable
            {
                private readonly string s;
                private int pos;
                public StringReader(string s) { this.s = s; pos = 0; }
                public void Dispose() { }
                public int Peek() => pos >= s.Length ? -1 : s[pos];
                public int Read() => pos >= s.Length ? -1 : s[pos++];
            }
        }
    }
}
