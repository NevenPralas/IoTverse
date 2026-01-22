using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.Networking;
using TMPro;
using XCharts.Runtime;

public class HeatMapStaticWithDataJediAndForecast : MonoBehaviour
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
    public TMP_Text labelC;   // current
    public TMP_Text labelL1;  // between L2 and C
    public TMP_Text labelL2;  // far left past
    public TMP_Text labelR1;  // between C and R2
    public TMP_Text labelR2;  // far right future

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

    [Header("Data Jedi OUTPUT (GET)")]
    [Tooltip("Npr. https://161.53.19.19:56443")]
    [SerializeField] private string dataJediBaseUrl = "https://161.53.19.19:56443";

    [Tooltip("Credentialsi vaše grupe (za GET /m2m/data).")]
    [SerializeField] private string djUsername = "digiphy1";
    [SerializeField] private string djPassword = "dhStugrGu6cH4uIx";

    [Tooltip("Gateway Group URN (deviceId / dgg) npr. Grupa1ESP32")]
    [SerializeField] private string deviceId = "Grupa1ESP32";

    [Tooltip("Sensor Specification URN (sensorSpec) npr. Grupa1DHT11 (ili vaš)")]
    [SerializeField] private string sensorSpec = "Grupa1DHT11";

    [Tooltip("ResourceSpec URN-ovi (4 temperature resursa)")]
    [SerializeField] private string resourceSpec1 = "Grupa1Temperature1";
    [SerializeField] private string resourceSpec2 = "Grupa1Temperature2";
    [SerializeField] private string resourceSpec3 = "Grupa1Temperature3";
    [SerializeField] private string resourceSpec4 = "Grupa1Temperature4";

    [Tooltip("Koliko točaka povijesti: 30 => sada + 29 unazad.")]
    [SerializeField] private int historyCount = 30;

    [Tooltip("Polling interval u sekundama.")]
    [SerializeField] private float pollSeconds = 1.0f;

    [Tooltip("Ako platforma ima self-signed cert, uključi (curl -k ekvivalent).")]
    [SerializeField] private bool acceptAnyCertificate = true;

    // Accept header za Simplified JSON output
    // (dokument i primjeri koriste application/vnd.ericsson.simple.output+json;version=1.0) 
    private const string DJ_ACCEPT = "application/vnd.ericsson.simple.output+json;version=1.0";

    // ======= Forecast (OSTAJE STARI ENDPOINT za testiranje) =======
    [Header("Forecast (stari endpoint za testiranje)")]
    [SerializeField] private string forecastUrl = "http://127.0.0.1:8000/api/forecast";

    private MeasurementData data;          // historical “30 točaka” iz Data Jedi
    private MeasurementData forecastData;  // forecast iz starog endpointa

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

    private void Awake() => EnsureInitialized();

    private void EnsureInitialized()
    {
        if (_initialized) return;

        _mf = GetComponent<MeshFilter>();
        _r = GetComponent<Renderer>();
        _mc = GetComponent<MeshCollider>();

        if (_mf == null) { Debug.LogError("[HeatMap] MeshFilter missing!"); return; }
        if (_r == null) { Debug.LogError("[HeatMap] Renderer missing!"); return; }

        originalMesh = _mf.sharedMesh;
        originalMaterial = _r.sharedMaterial;

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

        // 1) Kreni polling historical sa Data Jedi
        StartCoroutine(PollDataJediLoop());

        // 2) Forecast ostaje “stari endpoint”, povuci jednom (ili kad se toggle uključi)
        StartCoroutine(FetchForecastFromOldApi());
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

    // zove UISwitcherForecast
    public void SetForecastEnabled(bool enabled)
    {
        forecastEnabled = enabled;
        SetForecastTmpLabelsEnabled(forecastEnabled);

        if (forecastEnabled && (forecastData == null || forecastData.measurements == null || forecastData.measurements.Length == 0))
            StartCoroutine(FetchForecastFromOldApi());

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

    // =======================
    // Data Jedi HISTORICAL LOOP
    // =======================

    private IEnumerator PollDataJediLoop()
    {
        while (true)
        {
            yield return FetchMeasurementsFromDataJedi();
            yield return new WaitForSeconds(pollSeconds);
        }
    }

    private IEnumerator FetchMeasurementsFromDataJedi()
    {
        // Koristimo latestNCount=30 umjesto “t1/t2” jer ti treba “zadnjih 29 + sada”.
        // Primjer u dokumentu: /m2m/data?dgg=Device1&sensorSpec=...&latestNCount=1 :contentReference[oaicite:4]{index=4}
        // Više resourceSpec u istom requestu: resourceSpec=A,B & t1/t2 u primjeru, ali radi i s latestNCount :contentReference[oaicite:5]{index=5}
        string resources = $"{resourceSpec1},{resourceSpec2},{resourceSpec3},{resourceSpec4}";
        string url =
            $"{dataJediBaseUrl}/m2m/data" +
            $"?dgg={UnityWebRequest.EscapeURL(deviceId)}" +
            $"&sensorSpec={UnityWebRequest.EscapeURL(sensorSpec)}" +
            $"&resourceSpec={UnityWebRequest.EscapeURL(resources)}" +
            $"&latestNCount={historyCount}";

        using (var req = UnityWebRequest.Get(url))
        {
            // U uputama je GET s -u username:password i Accept output+json;version=1.1 
            // Ovdje koristimo Simplified JSON adapter jer je lakši za “timestamp -> vrijednosti”.
            string auth = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes($"{djUsername}:{djPassword}"));
            req.SetRequestHeader("Authorization", "Basic " + auth);
            req.SetRequestHeader("Accept", DJ_ACCEPT);

            if (acceptAnyCertificate)
                req.certificateHandler = new AcceptAllCertificates();

            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                SetStatus($"DataJedi GET error: {req.error}");
                yield break;
            }

            string json = req.downloadHandler.text;

            if (!TryBuildMeasurementDataFromSimplifiedJson(json, out MeasurementData md))
            {
                SetStatus("DataJedi JSON parse fail (provjeri deviceId/sensorSpec/resourceSpec).");
                yield break;
            }

            data = md;

            // postavi kutove na “NOW” = zadnja točka u prozoru
            int nowIdx = data.measurements.Length - 1;
            ApplyCornerTempsForIndex(nowIdx);

            SetStatus($"DataJedi OK: {data.measurements.Length} pts, now={data.measurements[nowIdx].timestamp}");
        }
    }

    // =======================
    // Forecast ostaje stari endpoint (FastAPI JSON kao prije)
    // =======================

    private IEnumerator FetchForecastFromOldApi()
    {
        using (UnityWebRequest req = UnityWebRequest.Get(forecastUrl))
        {
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning($"[Forecast] old API error: {req.error} ({forecastUrl})");
                yield break;
            }

            forecastData = JsonUtility.FromJson<MeasurementData>(req.downloadHandler.text);
            if (forecastData == null || forecastData.measurements == null || forecastData.measurements.Length == 0)
            {
                Debug.LogWarning("[Forecast] old API vratio prazne podatke ili JSON nije u očekivanom formatu.");
                yield break;
            }

            if (forecastEnabled && isTrackingPoint) RefreshChartWindowForNow();
        }
    }

    // =======================
    // “Now index” u ovom modelu: zadnja točka od 30 (index 29)
    // =======================
    private int GetNowIndexInWindow()
    {
        if (data == null || data.measurements == null || data.measurements.Length == 0) return 0;
        return data.measurements.Length - 1;
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

    // =======================
    // CHART (tvoja logika, ali “historical window” je data.measurements[0..29])
    // =======================

    private void RefreshChartWindowForNow()
    {
        if (lineChart == null || data == null || data.measurements == null || data.measurements.Length == 0)
            return;

        int nowIdx = GetNowIndexInWindow();

        if (nowIdx != cachedNowIndex)
        {
            cachedNowIndex = nowIdx;
            ApplyCornerTempsForIndex(nowIdx);
        }

        // UV za kliknutu točku
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

        // ako forecast OFF ili nema forecastData -> prikaži samo historical (30 točaka)
        if (!forecastEnabled || forecastData == null || forecastData.measurements == null || forecastData.measurements.Length == 0)
        {
            SetForecastTmpLabelsEnabled(false);

            int pointCount = data.measurements.Length; // očekivano 30
            int lastX = pointCount - 1;

            // 6 labela (0, 6, 12, 18, 24, 29) kao prije, ostalo zero-width
            int L0 = 0;
            int L1 = Mathf.Clamp(Mathf.RoundToInt(lastX * (6f / 29f)), 0, lastX);
            int L2 = Mathf.Clamp(Mathf.RoundToInt(lastX * (12f / 29f)), 0, lastX);
            int L3 = Mathf.Clamp(Mathf.RoundToInt(lastX * (18f / 29f)), 0, lastX);
            int L4 = Mathf.Clamp(Mathf.RoundToInt(lastX * (24f / 29f)), 0, lastX);
            int L5 = lastX;

            for (int xi = 0; xi < pointCount; xi++)
            {
                bool isLabeled = (xi == L0 || xi == L1 || xi == L2 || xi == L3 || xi == L4 || xi == L5);
                string xLabel = isLabeled ? FormatTimeHhMmSs(data.measurements[xi].timestamp) : new string('\u200B', xi + 1);
                chart.AddXAxisData(xLabel);
            }

            for (int i = 0; i < pointCount; i++)
            {
                Measurement m = data.measurements[i];
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

        // ========= FORECAST ON (stari endpoint) =========
        SetForecastTmpLabelsEnabled(true);

        int histCount = data.measurements.Length;         // 30
        int futCount = Mathf.Min(FutureCount, forecastData.measurements.Length); // sigurnosno
        int combinedCount = histCount + futCount;

        // axis 61 (kao prije)
        for (int xi = 0; xi < ForecastAxisCount; xi++)
            chart.AddXAxisData(new string('\u200B', xi + 1));

        for (int i = 0; i < ForecastAxisCount; i++)
        {
            int ci = Mathf.Clamp(i, 0, combinedCount - 1);

            float y;
            if (ci < histCount)
            {
                var m = data.measurements[ci];
                y = BilinearInterpolation(u, v, m.temperature1, m.temperature2, m.temperature3, m.temperature4);
            }
            else
            {
                int fIdx = ci - histCount;
                fIdx = Mathf.Clamp(fIdx, 0, forecastData.measurements.Length - 1);
                var fm = forecastData.measurements[fIdx];
                y = BilinearInterpolation(u, v, fm.temperature1, fm.temperature2, fm.temperature3, fm.temperature4);
            }

            chart.AddData(0, y); // historical
            chart.AddData(1, y); // forecast
        }

        ResetSerieData(chart, 0);
        ResetSerieData(chart, 1);

        // NOW je na “historical” zadnjoj točki => index 29 u tvojoj logici
        int nowPoint = 29;
        UpdateTemperatureText(GetSerieY(chart, 0, nowPoint));

        // serie0: samo past+now (0..29)
        SetIgnoreRange(chart, 0, 30, 60, true);

        // serie1: now+future (29..58), ostalo ignore
        SetIgnoreRange(chart, 1, 0, 28, true);
        SetIgnoreRange(chart, 1, 59, 60, true);

        SetPointRed(chart, 0, nowPoint);
        HideAllSymbols(chart, 1);

        UpdateForecastTmpTexts_ForWindow(nowPoint);

        var xAxisF = chart.EnsureChartComponent<XAxis>();
        xAxisF.type = Axis.AxisType.Category;
        xAxisF.axisLabel.formatter = "{value}";
        xAxisF.interval = 0;

        chart.RefreshChart();
    }

    // TMP labels za forecast ON: uzmi iz historical prozora i iz forecastData
    private void UpdateForecastTmpTexts_ForWindow(int nowPointIndex)
    {
        // historical window: 0..29
        int idxC = nowPointIndex;
        int idxL2 = Mathf.Clamp(nowPointIndex - 29, 0, 29);
        int idxL1 = Mathf.Clamp(nowPointIndex - 15, 0, 29);

        int fR1 = Mathf.Clamp(15, 0, forecastData.measurements.Length - 1);
        int fR2 = Mathf.Clamp(29, 0, forecastData.measurements.Length - 1);

        if (data != null && data.measurements != null && data.measurements.Length > 0)
        {
            if (labelC != null) labelC.text = FormatTimeHhMmSs(data.measurements[idxC].timestamp);
            if (labelL2 != null) labelL2.text = FormatTimeHhMmSs(data.measurements[idxL2].timestamp);
            if (labelL1 != null) labelL1.text = FormatTimeHhMmSs(data.measurements[idxL1].timestamp);
        }

        if (forecastData != null && forecastData.measurements != null && forecastData.measurements.Length > 0)
        {
            if (labelR1 != null) labelR1.text = FormatTimeHhMmSs(forecastData.measurements[fR1].timestamp);
            if (labelR2 != null) labelR2.text = FormatTimeHhMmSs(forecastData.measurements[fR2].timestamp);
        }
    }

    private void SetForecastTmpLabelsEnabled(bool enabled)
    {
        if (labelC != null) labelC.enabled = enabled;
        if (labelL1 != null) labelL1.enabled = enabled;
        if (labelL2 != null) labelL2.enabled = enabled;
        if (labelR1 != null) labelR1.enabled = enabled;
        if (labelR2 != null) labelR2.enabled = enabled;
    }

    // =======================
    // Heatmap / Mesh (tvoje, bez promjena osim sitnih null-checkova)
    // =======================

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

                float y = 0f;
                if (originalMesh != null && originalMesh.vertices != null && originalMesh.vertices.Length > 0)
                    y = originalMesh.vertices[Mathf.Clamp(i, 0, originalMesh.vertices.Length - 1)].y;

                vertices[i] = new Vector3(
                    (uu - 0.5f) * planeWidth,
                    y,
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

    // =======================
    // XCharts helperi (tvoje)
    // =======================

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

    // =======================
    // AimOnGrip compatibility
    // =======================
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

    // =======================
    // Data Jedi JSON -> MeasurementData
    // =======================
    private bool TryBuildMeasurementDataFromSimplifiedJson(string json, out MeasurementData md)
    {
        md = new MeasurementData { measurements = Array.Empty<Measurement>() };
        if (string.IsNullOrWhiteSpace(json)) return false;

        object rootObj = MiniJSON.Json.Deserialize(json);
        if (!(rootObj is Dictionary<string, object> root)) return false;

        // default simplified json struktura:
        // {
        //   "GatewayGroupSpecification": { "SensorSpecification1": { "timestampISO": { "Res1": 10, "Res2": 20 ... }, ... } },
        //   "deviceId": "Device1"
        // }
        // 

        // Nađi “wrapper” (prvi key koji nije deviceId)
        Dictionary<string, object> wrapper = null;
        foreach (var kv in root)
        {
            if (kv.Key.Equals("deviceId", StringComparison.OrdinalIgnoreCase)) continue;
            wrapper = kv.Value as Dictionary<string, object>;
            if (wrapper != null) break;
        }
        if (wrapper == null) return false;

        // Nađi sensorSpec objekt (ili prvi child ako ime nije isto)
        Dictionary<string, object> sensorNode = null;
        if (wrapper.TryGetValue(sensorSpec, out object snObj))
            sensorNode = snObj as Dictionary<string, object>;

        if (sensorNode == null)
        {
            foreach (var kv in wrapper)
            {
                sensorNode = kv.Value as Dictionary<string, object>;
                if (sensorNode != null) break;
            }
        }
        if (sensorNode == null) return false;

        // sensorNode: keys su timestamp ISO, value je dict s ResourceSpec -> value
        var list = new List<Measurement>();
        foreach (var kv in sensorNode)
        {
            string tsIso = kv.Key;
            var resDict = kv.Value as Dictionary<string, object>;
            if (resDict == null) continue;

            float t1 = GetFloat(resDict, resourceSpec1);
            float t2 = GetFloat(resDict, resourceSpec2);
            float t3 = GetFloat(resDict, resourceSpec3);
            float t4 = GetFloat(resDict, resourceSpec4);

            list.Add(new Measurement
            {
                id = 0,
                timestamp = tsIso,
                temperature1 = t1,
                temperature2 = t2,
                temperature3 = t3,
                temperature4 = t4
            });
        }

        if (list.Count == 0) return false;

        // sort po timestampu (ISO string -> DateTime)
        list.Sort((a, b) =>
        {
            DateTime ta = ParseIso(a.timestamp);
            DateTime tb = ParseIso(b.timestamp);
            return ta.CompareTo(tb);
        });

        md.measurements = list.ToArray();
        return true;
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

    private DateTime ParseIso(string iso)
    {
        if (DateTime.TryParse(iso, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var dt))
            return dt.ToUniversalTime();
        return DateTime.MinValue;
    }

    private string FormatTimeHhMmSs(string isoTimestamp)
    {
        if (!DateTime.TryParse(isoTimestamp, out DateTime dtFromDb))
            return isoTimestamp;
        return dtFromDb.ToLocalTime().ToString("HH:mm:ss");
    }

    private void SetStatus(string msg)
    {
        if (debugStatus != null) debugStatus.text = msg;
    }

    private class AcceptAllCertificates : CertificateHandler
    {
        protected override bool ValidateCertificate(byte[] certificateData) => true;
    }

    // =======================
    // DTO (forecast stari format + naš historical)
    // =======================

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

    // =======================
    // MiniJSON (Unity-friendly)
    // =======================
    // Source: public domain style MiniJSON (embedded).
    // Minimalno: Deserialize u Dictionary<string, object>.
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

                private Parser(string jsonString)
                {
                    json = new StringReader(jsonString);
                }

                public static object Parse(string jsonString)
                {
                    using (var instance = new Parser(jsonString))
                    {
                        return instance.ParseValue();
                    }
                }

                public void Dispose()
                {
                    json.Dispose();
                    json = null;
                }

                private Dictionary<string, object> ParseObject()
                {
                    var table = new Dictionary<string, object>();

                    // {
                    json.Read();

                    while (true)
                    {
                        Token nextToken = NextToken;
                        if (nextToken == Token.NONE) return null;
                        if (nextToken == Token.CURLY_CLOSE) return table;

                        // key
                        string name = ParseString();
                        if (name == null) return null;

                        // :
                        if (NextToken != Token.COLON) return null;
                        json.Read();

                        // value
                        table[name] = ParseValue();
                    }
                }

                private List<object> ParseArray()
                {
                    var array = new List<object>();

                    // [
                    json.Read();

                    var parsing = true;
                    while (parsing)
                    {
                        Token nextToken = NextToken;

                        if (nextToken == Token.NONE) return null;
                        if (nextToken == Token.SQUARE_CLOSE) break;

                        object value = ParseValue();
                        array.Add(value);

                        nextToken = NextToken;
                        if (nextToken == Token.COMMA)
                        {
                            json.Read();
                        }
                        else if (nextToken == Token.SQUARE_CLOSE)
                        {
                            // ok
                        }
                        else
                        {
                            return null;
                        }
                    }

                    // ]
                    json.Read();
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
                        case Token.TRUE: json.Read(); json.Read(); json.Read(); json.Read(); return true;
                        case Token.FALSE: json.Read(); json.Read(); json.Read(); json.Read(); json.Read(); return false;
                        case Token.NULL: json.Read(); json.Read(); json.Read(); json.Read(); return null;
                        default: return null;
                    }
                }

                private string ParseString()
                {
                    var s = "";
                    char c;

                    // "
                    json.Read();

                    bool parsing = true;
                    while (parsing)
                    {
                        if (json.Peek() == -1) break;

                        c = NextChar;
                        if (c == '"')
                        {
                            parsing = false;
                            break;
                        }
                        else if (c == '\\')
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
                        else
                        {
                            s += c;
                        }
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
                    while (char.IsWhiteSpace(PeekChar)) json.Read();
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
                        switch (word)
                        {
                            case "false": return Token.FALSE;
                            case "true": return Token.TRUE;
                            case "null": return Token.NULL;
                        }
                        return Token.NONE;
                    }
                }

                private static bool IsWordBreak(char c) => char.IsWhiteSpace(c) || WORD_BREAK.IndexOf(c) != -1;

                private enum Token
                {
                    NONE,
                    CURLY_OPEN,
                    CURLY_CLOSE,
                    SQUARE_OPEN,
                    SQUARE_CLOSE,
                    COLON,
                    COMMA,
                    STRING,
                    NUMBER,
                    TRUE,
                    FALSE,
                    NULL
                }
            }

            private sealed class StringReader : IDisposable
            {
                private readonly string s;
                private int pos;

                public StringReader(string s) { this.s = s; pos = 0; }
                public void Dispose() { }

                public int Peek()
                {
                    if (pos >= s.Length) return -1;
                    return s[pos];
                }

                public int Read()
                {
                    if (pos >= s.Length) return -1;
                    return s[pos++];
                }
            }
        }
    }
}
