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
    // Data Jedi TRUSTED (GET)
    // =========================================================
    [Header("Data Jedi TRUSTED (GET)")]
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

    [Header("Trusted headers")]
    [SerializeField] private string authorization = "PREAUTHENTICATED";
    [SerializeField] private string requesterId = "digiphy1";
    [SerializeField] private string requesterType = "domainApplication";
    [SerializeField] private string accept = "application/vnd.ericsson.simple.output+json;version=1.0";

    [Header("Polling")]
    [SerializeField] private float pollSeconds = 5.0f;
    [SerializeField] private int requestTimeoutSeconds = 15;

    [Tooltip("Postman SSL cert OFF ekvivalent. Uključi ako endpoint vraća SSL chain koji Unity ne voli.")]
    [SerializeField] private bool acceptAnyCertificate = true;

    [Header("Debug")]
    [SerializeField] private bool debugLogs = true;
    [SerializeField] private int debugBodyMaxChars = 600;

    // =========================================================
    // Forecast (OSTAJE KAKO JE)
    // =========================================================
    [Header("Forecast (stari endpoint za testiranje)")]
    [SerializeField] private string forecastUrl = "http://127.0.0.1:8000/api/forecast";

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

    private const int PastCount = 29;
    private const int FutureCount = 29;

    private const int ForecastAxisCount = 61;

    [Header("Forecast Style (Serie 1)")]
    [SerializeField] private Color forecastColor = Color.gray;
    [SerializeField] private float dashLength = 6f;
    [SerializeField] private float gapLength = 4f;
    [SerializeField] private float dotLength = 0f;

    // ======== INTERNAL INIT / CACHED COMPONENTS ========
    private bool _initialized = false;
    private MeshFilter _mf;
    private Renderer _r;
    private MeshCollider _mc;

    // ======== SLIDING WINDOW ZA 4 SENZORA ========
    private const int MaxSlidingWindow = 30;

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

    private void Awake()
    {
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
            Debug.LogError("[HeatMap] MeshFilter missing!");
            return;
        }
        if (_r == null)
        {
            Debug.LogError("[HeatMap] Renderer missing!");
            return;
        }

        originalMesh = _mf.sharedMesh;
        originalMaterial = _r.sharedMaterial;

        if (originalMesh == null)
            Debug.LogError("[HeatMap] originalMesh je NULL (MeshFilter.sharedMesh). Provjeri da plane ima mesh!");
        if (originalMaterial == null)
            Debug.LogError("[HeatMap] originalMaterial je NULL (Renderer.sharedMaterial).");

        if (originalMesh != null)
        {
            planeWidth = originalMesh.bounds.size.x;
            planeDepth = originalMesh.bounds.size.z;
        }

        _initialized = true;
    }

    void Start()
    {
        EnsureInitialized();

        SetForecastTmpLabelsEnabled(false);

        if (lineChart != null)
        {
            var chart = lineChart.GetComponent<LineChart>();
            if (chart != null)
            {
                chart.ClearData();
                ForceYAxis10to40(chart);
                UpdateChartTitleWithDate(chart);

                EnsureTwoSeries(chart);
                ConfigureForecastSerie(chart);

                chart.RefreshChart();
            }
        }

        DeactivateHeatmap();
        SaveCurrentAngles();

        StartCoroutine(PollDataJediLoop());
        StartCoroutine(FetchForecastFromApi());
    }

    void Update()
    {
        if (!heatmapEnabled) return;

        if (autoUpdate && HasValuesChanged())
        {
            GenerateHeatmap();
            SaveCurrentAngles();
        }

        if (isTrackingPoint && data != null && data.measurements != null && data.measurements.Length > 0)
        {
            chartUpdateTimer += Time.deltaTime;
            if (chartUpdateTimer >= ChartUpdatePeriod)
            {
                chartUpdateTimer = 0f;
                RefreshChartWindowForNow();
            }
        }
    }

    // =========================================================
    // Data Jedi polling (4 endpoints)
    // =========================================================
    private IEnumerator PollDataJediLoop()
    {
        while (true)
        {
            yield return FetchAndApplyLatestSamples();
            yield return new WaitForSeconds(pollSeconds);
        }
    }

    private IEnumerator FetchAndApplyLatestSamples()
    {
        // sekvencijalno (lakše debug)
        yield return FetchOneRes(res1, 1);
        yield return FetchOneRes(res2, 2);
        yield return FetchOneRes(res3, 3);
        yield return FetchOneRes(res4, 4);

        UpdateCornerTemperaturesFromBuffers();
        RebuildMeasurementDataFromSlidingWindow();

        // status: nemoj “OK” ako nema ničega
        bool any =
            _sensor1Data.Count > 0 ||
            _sensor2Data.Count > 0 ||
            _sensor3Data.Count > 0 ||
            _sensor4Data.Count > 0;

        if (!any)
        {
            SetStatus("[HeatMap] DataJedi poll: NO DATA parsed (bufferi prazni) -> angles ostaju default");
        }
        else
        {
            SetStatus($"[HeatMap] DataJedi poll OK | A1={angle1:0.0} A2={angle2:0.0} A3={angle3:0.0} A4={angle4:0.0}");
        }
    }

    private IEnumerator FetchOneRes(string resValue, int sensorIndex)
    {
        string url =
            $"{baseUrl}{trustedPath}" +
            $"?usr={UnityWebRequest.EscapeURL(usr)}" +
            $"&latestNCount={latestNCount}" +
            $"&{UnityWebRequest.EscapeURL(resParamName)}={UnityWebRequest.EscapeURL(resValue)}";

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
                Debug.LogWarning($"[DataJedi] HTTP FAIL sensor{sensorIndex} res={resValue} code={code} err={req.error}");
                if (!string.IsNullOrEmpty(bodyShort)) Debug.LogWarning(bodyShort);
                yield break;
            }

            if (!TryParseTrustedJson(body, valueKey, out List<TimestampValue> series))
            {
                Debug.LogWarning($"[DataJedi] PARSE FAIL sensor{sensorIndex} res={resValue} code={code} bodyLen={(body != null ? body.Length : 0)}");
                if (!string.IsNullOrEmpty(bodyShort)) Debug.LogWarning(bodyShort);
                yield break;
            }

            ApplySeriesToSlidingWindow(series, sensorIndex);

            // debug: što smo stvarno dobili
            if (debugLogs)
            {
                var list = GetSensorList(sensorIndex);
                var last = list.Count > 0 ? list[list.Count - 1] : null;
                string lastLocal = last != null ? last.timestampUtc.ToLocalTime().ToString("HH:mm:ss.fff") : "N/A";
                float lastVal = last != null ? last.value : 0f;

                Debug.Log($"[DataJedi] OK sensor{sensorIndex} res={resValue} http={code} parsed={series.Count} buffer={list.Count} last={lastLocal} val={lastVal:0.0}");
            }
        }
    }

    private List<TimestampValue> GetSensorList(int sensorIndex)
    {
        return sensorIndex == 1 ? _sensor1Data :
               sensorIndex == 2 ? _sensor2Data :
               sensorIndex == 3 ? _sensor3Data :
                                  _sensor4Data;
    }

    // JSON format:
    // [
    //   {
    //     "deviceId": "...",
    //     "<GW_NODE>": {
    //       "<SENSOR_NODE>": {
    //         "<ISO>": { "temperature": 22.2 },
    //         ...
    //       }
    //     }
    //   }
    // ]
    private bool TryParseTrustedJson(string json, string key, out List<TimestampValue> series)
    {
        series = null;
        if (string.IsNullOrWhiteSpace(json)) return false;

        object rootObj = MiniJSON.Json.Deserialize(json);
        if (!(rootObj is List<object> arr) || arr.Count == 0) return false;
        if (!(arr[0] is Dictionary<string, object> root)) return false;

        // gateway node = prvi key koji nije deviceId
        Dictionary<string, object> gwNode = null;
        foreach (var kv in root)
        {
            if (kv.Key.Equals("deviceId", StringComparison.OrdinalIgnoreCase)) continue;
            gwNode = kv.Value as Dictionary<string, object>;
            if (gwNode != null) break;
        }
        if (gwNode == null) return false;

        // sensor node = prvi child dictionary (npr. Diplomski_Projekt_2025_dht11_spec)
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

        // sort po vremenu (najstarije prvo)
        list.Sort((a, b) => a.timestampUtc.CompareTo(b.timestampUtc));
        series = list;
        return true;
    }

    private void ApplySeriesToSlidingWindow(List<TimestampValue> series, int sensorIndex)
    {
        if (series == null || series.Count == 0) return;

        var targetList = GetSensorList(sensorIndex);

        // dodaj nove uzorke (skip duplikate)
        foreach (var tv in series)
        {
            bool exists = false;
            for (int i = 0; i < targetList.Count; i++)
            {
                if (Math.Abs((targetList[i].timestampUtc - tv.timestampUtc).TotalSeconds) < 0.5)
                {
                    exists = true;
                    break;
                }
            }
            if (!exists) targetList.Add(tv);
        }

        targetList.Sort((a, b) => a.timestampUtc.CompareTo(b.timestampUtc));

        while (targetList.Count > MaxSlidingWindow)
            targetList.RemoveAt(0);
    }

    private void UpdateCornerTemperaturesFromBuffers()
    {
        if (_sensor1Data.Count > 0) angle1 = _sensor1Data[_sensor1Data.Count - 1].value;
        if (_sensor2Data.Count > 0) angle2 = _sensor2Data[_sensor2Data.Count - 1].value;
        if (_sensor3Data.Count > 0) angle3 = _sensor3Data[_sensor3Data.Count - 1].value;
        if (_sensor4Data.Count > 0) angle4 = _sensor4Data[_sensor4Data.Count - 1].value;
    }

    private void RebuildMeasurementDataFromSlidingWindow()
    {
        var all = new HashSet<DateTime>();
        foreach (var tv in _sensor1Data) all.Add(tv.timestampUtc);
        foreach (var tv in _sensor2Data) all.Add(tv.timestampUtc);
        foreach (var tv in _sensor3Data) all.Add(tv.timestampUtc);
        foreach (var tv in _sensor4Data) all.Add(tv.timestampUtc);

        var sorted = new List<DateTime>(all);
        sorted.Sort();

        if (sorted.Count > MaxSlidingWindow)
            sorted = sorted.GetRange(sorted.Count - MaxSlidingWindow, MaxSlidingWindow);

        data = new MeasurementData();
        data.measurements = new Measurement[sorted.Count];

        for (int i = 0; i < sorted.Count; i++)
        {
            DateTime ts = sorted[i];

            data.measurements[i] = new Measurement
            {
                id = i,
                timestamp = ts.ToString("o"),
                temperature1 = FindValueAtOrBefore(_sensor1Data, ts),
                temperature2 = FindValueAtOrBefore(_sensor2Data, ts),
                temperature3 = FindValueAtOrBefore(_sensor3Data, ts),
                temperature4 = FindValueAtOrBefore(_sensor4Data, ts),
            };
        }

        if (isTrackingPoint) RefreshChartWindowForNow();
    }

    private float FindValueAtOrBefore(List<TimestampValue> list, DateTime targetTsUtc)
    {
        float last = 0f;
        for (int i = 0; i < list.Count; i++)
        {
            if (list[i].timestampUtc <= targetTsUtc)
                last = list[i].value;
            else
                break;
        }
        return last;
    }

    // =========================================================
    // Forecast (stari endpoint - bez promjena)
    // =========================================================
    private IEnumerator FetchForecastFromApi()
    {
        if (string.IsNullOrWhiteSpace(forecastUrl))
            yield break;

        using (UnityWebRequest req = UnityWebRequest.Get(forecastUrl))
        {
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning($"[HeatMap] Forecast API error: {req.error} ({forecastUrl})");
                yield break;
            }

            forecastData = JsonUtility.FromJson<MeasurementData>(req.downloadHandler.text);
            if (forecastData == null || forecastData.measurements == null || forecastData.measurements.Length == 0)
            {
                Debug.LogWarning("[HeatMap] forecast API vratio prazne podatke ili JSON nije u očekivanom formatu.");
                yield break;
            }

            if (forecastEnabled && isTrackingPoint) RefreshChartWindowForNow();
        }
    }

    public void SetForecastEnabled(bool enabled)
    {
        forecastEnabled = enabled;
        SetForecastTmpLabelsEnabled(forecastEnabled);

        if (forecastEnabled && (forecastData == null || forecastData.measurements == null || forecastData.measurements.Length == 0))
            StartCoroutine(FetchForecastFromApi());

        if (isTrackingPoint) RefreshChartWindowForNow();
    }

    public void UpdateChartForPoint(Vector3 worldPoint)
    {
        if (lineChart == null || data == null || data.measurements == null || data.measurements.Length == 0)
            return;

        trackedPoint = worldPoint;
        isTrackingPoint = true;
        RefreshChartWindowForNow();
    }

    private void RefreshChartWindowForNow()
    {
        if (lineChart == null || data == null || data.measurements == null || data.measurements.Length == 0)
            return;

        int nowIdx = data.measurements.Length - 1;

        if (nowIdx != cachedNowIndex)
        {
            cachedNowIndex = nowIdx;
            ApplyCornerTempsForIndex(nowIdx);
        }

        Vector3 local = transform.InverseTransformPoint(trackedPoint);
        float u = (local.x / (planeWidth * transform.lossyScale.x)) + 0.5f;
        float v = (local.z / (planeDepth * transform.lossyScale.z)) + 0.5f;
        u = Mathf.Clamp01(u);
        v = Mathf.Clamp01(v);

        var chart = lineChart.GetComponent<LineChart>();
        if (chart == null) return;

        chart.ClearData();
        EnsureTwoSeries(chart);
        ConfigureForecastSerie(chart);

        ForceYAxis10to40(chart);
        UpdateChartTitleWithDate(chart);

        if (!forecastEnabled || forecastData == null || forecastData.measurements == null || forecastData.measurements.Length == 0)
        {
            SetForecastTmpLabelsEnabled(false);

            int endIdx = nowIdx;
            int startIdx = Mathf.Max(0, endIdx - 29);
            int pointCount = (endIdx - startIdx) + 1;

            int xCategoryCount = pointCount + 1;
            int lastX = xCategoryCount - 1;

            int L0 = 0;
            int L1 = Mathf.Clamp(Mathf.RoundToInt(lastX * (6f / 30f)), 0, lastX);
            int L2 = Mathf.Clamp(Mathf.RoundToInt(lastX * (12f / 30f)), 0, lastX);
            int L3 = Mathf.Clamp(Mathf.RoundToInt(lastX * (18f / 30f)), 0, lastX);
            int L4 = Mathf.Clamp(Mathf.RoundToInt(lastX * (24f / 30f)), 0, lastX);
            int L5 = lastX;

            for (int xi = 0; xi < xCategoryCount; xi++)
            {
                bool isLabeled = (xi == L0 || xi == L1 || xi == L2 || xi == L3 || xi == L4 || xi == L5);
                string xLabel;

                if (isLabeled)
                {
                    int measurementIndex = (xi >= pointCount) ? endIdx : (startIdx + xi);
                    measurementIndex = Mathf.Clamp(measurementIndex, 0, data.measurements.Length - 1);
                    xLabel = FormatTimeLabelFromIso(data.measurements[measurementIndex].timestamp);
                }
                else xLabel = new string('\u200B', xi + 1);

                chart.AddXAxisData(xLabel);
            }

            for (int i = 0; i < pointCount; i++)
            {
                int idx = Mathf.Clamp(startIdx + i, 0, data.measurements.Length - 1);
                Measurement m = data.measurements[idx];
                float y = BilinearInterpolation(u, v, m.temperature1, m.temperature2, m.temperature3, m.temperature4);

                chart.AddData(0, y);
                chart.AddData(1, 0f);

                if (i == pointCount - 1) UpdateTemperatureText(y);
            }

            ResetSerieData(chart, 0);
            ResetSerieData(chart, 1);
            IgnoreAllPoints(chart, 1);

            SetPointRed(chart, 0, pointCount - 1);

            var xAxis = chart.EnsureChartComponent<XAxis>();
            xAxis.type = Axis.AxisType.Category;
            xAxis.axisLabel.formatter = "{value}";
            xAxis.interval = 0;

            chart.RefreshChart();
            return;
        }

        // forecast ON dio ostaje isti kao prije (nije ti sad bitan)
        SetForecastTmpLabelsEnabled(true);

        int histStart = Mathf.Max(0, nowIdx - PastCount);
        int histEnd = nowIdx;
        int histCount = (histEnd - histStart) + 1;

        int futStart = Mathf.Min(forecastData.measurements.Length - 1, nowIdx + 1);
        int futEnd = Mathf.Min(forecastData.measurements.Length - 1, nowIdx + FutureCount);
        int futCount = (futEnd >= futStart) ? (futEnd - futStart + 1) : 0;

        int combinedCount = histCount + futCount;

        for (int xi = 0; xi < ForecastAxisCount; xi++)
            chart.AddXAxisData(new string('\u200B', xi + 1));

        for (int i = 0; i < ForecastAxisCount; i++)
        {
            int ci = Mathf.Clamp(i, 0, 58);
            if (combinedCount > 0) ci = Mathf.Clamp(ci, 0, combinedCount - 1);

            float y = GetCombinedValueAtPoint(u, v, ci, histStart, histCount, futStart);

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

        UpdateForecastTmpTexts(nowIdx);

        var xAxisF = chart.EnsureChartComponent<XAxis>();
        xAxisF.type = Axis.AxisType.Category;
        xAxisF.axisLabel.formatter = "{value}";
        xAxisF.interval = 0;

        chart.RefreshChart();
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

    // ===== TMP label logic =====

    private void SetForecastTmpLabelsEnabled(bool enabled)
    {
        if (labelC != null) labelC.enabled = enabled;
        if (labelL1 != null) labelL1.enabled = enabled;
        if (labelL2 != null) labelL2.enabled = enabled;
        if (labelR1 != null) labelR1.enabled = enabled;
        if (labelR2 != null) labelR2.enabled = enabled;
    }

    private void UpdateForecastTmpTexts(int nowIdx)
    {
        int idxC = nowIdx;
        int idxL2 = Mathf.Max(0, nowIdx - 29);
        int idxL1 = Mathf.Max(0, nowIdx - 15);

        int idxR2 = Mathf.Min(3599, nowIdx + 29);
        int idxR1 = Mathf.Min(3599, nowIdx + 15);

        if (data != null && data.measurements != null && data.measurements.Length > 0)
        {
            if (labelC != null) labelC.text = FormatTimeLabelFromIso(data.measurements[Mathf.Clamp(idxC, 0, data.measurements.Length - 1)].timestamp);
            if (labelL2 != null) labelL2.text = FormatTimeLabelFromIso(data.measurements[Mathf.Clamp(idxL2, 0, data.measurements.Length - 1)].timestamp);
            if (labelL1 != null) labelL1.text = FormatTimeLabelFromIso(data.measurements[Mathf.Clamp(idxL1, 0, data.measurements.Length - 1)].timestamp);
        }

        if (forecastData != null && forecastData.measurements != null && forecastData.measurements.Length > 0)
        {
            if (labelR2 != null) labelR2.text = FormatTimeLabelFromIso(forecastData.measurements[Mathf.Clamp(idxR2, 0, forecastData.measurements.Length - 1)].timestamp);
            if (labelR1 != null) labelR1.text = FormatTimeLabelFromIso(forecastData.measurements[Mathf.Clamp(idxR1, 0, forecastData.measurements.Length - 1)].timestamp);
        }
    }

    // ======== Serie helpers ========

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

    private float GetCombinedValueAtPoint(float u, float v, int combinedIndex, int histStart, int histCount, int futStart)
    {
        if (combinedIndex < histCount)
        {
            int mIdx = Mathf.Clamp(histStart + combinedIndex, 0, data.measurements.Length - 1);
            Measurement m = data.measurements[mIdx];
            return BilinearInterpolation(u, v, m.temperature1, m.temperature2, m.temperature3, m.temperature4);
        }
        else
        {
            int fOffset = combinedIndex - histCount;
            int fIdx = Mathf.Clamp(futStart + fOffset, 0, forecastData.measurements.Length - 1);
            Measurement fm = forecastData.measurements[fIdx];
            return BilinearInterpolation(u, v, fm.temperature1, fm.temperature2, fm.temperature3, fm.temperature4);
        }
    }

    // ======== Formatting ========

    private string FormatTimeLabelFromIso(string isoTimestamp)
    {
        if (!DateTime.TryParse(isoTimestamp, CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out DateTime utc))
            return isoTimestamp;

        return utc.ToLocalTime().ToString("HH:mm:ss");
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

    // ======== AimOnGrip compatibility ========

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
        AimOnGrip aimScript = FindObjectOfType<AimOnGrip>();
        if (aimScript != null)
            aimScript.UpdateTemperatureDisplay(temperature);
    }

    // ======== Heatmap/Mesh ========

    public void ToggleHeatmap()
    {
        SetHeatmapEnabled(!heatmapEnabled);
    }

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
            Debug.LogError("[HeatMap] Ne mogu aktivirati heatmap jer originalMesh/originalMaterial nije validan.");
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

                vertices[i] = new Vector3(
                    (uu - 0.5f) * planeWidth,
                    originalMesh.vertices[Mathf.Clamp(i, 0, originalMesh.vertices.Length - 1)].y,
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

    // ===== helpers: parsing numbers & timestamps =====

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

    // ======== DATA MODEL (XCharts i ostale skripte očekuju ovo) ========
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

    // ===== MiniJSON (embedded) =====
    // FIX: ParseObject sada ispravno “pojede” zarez nakon svakog paira.
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

                        // ključ mora biti string
                        if (nextToken != Token.STRING) return null;
                        string name = ParseString();
                        if (name == null) return null;

                        // očekuj ':'
                        if (NextToken != Token.COLON) return null;
                        json.Read(); // ':'

                        // vrijednost
                        object value = ParseValue();
                        table[name] = value;

                        // nakon paira: ili ',' ili '}'
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

                        // nešto čudno
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
