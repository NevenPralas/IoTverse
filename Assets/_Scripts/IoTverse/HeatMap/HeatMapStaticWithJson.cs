using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using XCharts.Runtime;
using TMPro;

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
    public TMP_Text labelC;   // current
    public TMP_Text labelL1;  // between L2 and C
    public TMP_Text labelL2;  // far left past
    public TMP_Text labelR1;  // between C and R2
    public TMP_Text labelR2;  // far right future

    public bool heatmapEnabled = true;
    private Mesh originalMesh;
    private Material originalMaterial;
    private Texture2D heatmapTexture;
    private Material planeMaterial;
    private float previousAngle1, previousAngle2, previousAngle3, previousAngle4;

    private bool meshGenerated = false;
    private float planeWidth;
    private float planeDepth;

    [Header("API (FastAPI)")]
    [SerializeField] private string apiUrl = "http://127.0.0.1:8000/api/measurements";
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

    // Forecast axis: 0..60 => 61 tickova
    private const int ForecastAxisCount = 61;
    private static readonly int[] ForecastLabelSlots = { 0, 12, 30, 48, 60 };

    [Header("Forecast Style (Serie 1)")]
    [SerializeField] private Color forecastColor = Color.gray;
    [SerializeField] private float dashLength = 6f;
    [SerializeField] private float gapLength = 4f;
    [SerializeField] private float dotLength = 0f;

    void Start()
    {
        // ugasi TMP label komponente po defaultu (ako su dodijeljene)
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

        originalMesh = GetComponent<MeshFilter>().mesh;
        originalMaterial = GetComponent<Renderer>().material;
        planeWidth = originalMesh.bounds.size.x;
        planeDepth = originalMesh.bounds.size.z;

        ActivateHeatmap();
        SaveCurrentAngles();

        StartCoroutine(FetchMeasurementsFromApi());
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

    // zove UISwitcherForecast
    public void SetForecastEnabled(bool enabled)
    {
        forecastEnabled = enabled;

        // upali/ugasi TMP komponente
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

    private IEnumerator FetchMeasurementsFromApi()
    {
        using (UnityWebRequest req = UnityWebRequest.Get(apiUrl))
        {
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[HeatMap] API error: {req.error} ({apiUrl})");
                yield break;
            }

            data = JsonUtility.FromJson<MeasurementData>(req.downloadHandler.text);
            if (data == null || data.measurements == null || data.measurements.Length == 0)
            {
                Debug.LogError("[HeatMap] measurements API vratio prazne podatke ili JSON nije u očekivanom formatu.");
                yield break;
            }

            Debug.Log($"[HeatMap] Učitano measurements: {data.measurements.Length}");
            ApplyCornerTempsForNowIndex(GetNowMappedIndex());
        }
    }

    private IEnumerator FetchForecastFromApi()
    {
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

            Debug.Log($"[HeatMap] Učitano forecast: {forecastData.measurements.Length}");
            if (forecastEnabled && isTrackingPoint) RefreshChartWindowForNow();
        }
    }

    private int GetNowMappedIndex()
    {
        DateTime now = DateTime.Now;
        int secondsFromHourStart = (now.Minute * 60) + now.Second;
        int maxIndex = data != null && data.measurements != null ? data.measurements.Length - 1 : 3599;
        return Mathf.Clamp(secondsFromHourStart, 0, maxIndex);
    }

    private void ApplyCornerTempsForNowIndex(int idx)
    {
        if (data == null || data.measurements == null || data.measurements.Length == 0) return;
        idx = Mathf.Clamp(idx, 0, data.measurements.Length - 1);

        Measurement m = data.measurements[idx];
        angle1 = m.temperature1;
        angle2 = m.temperature2;
        angle3 = m.temperature3;
        angle4 = m.temperature4;
    }

    private void RefreshChartWindowForNow()
    {
        if (lineChart == null || data == null || data.measurements == null || data.measurements.Length == 0)
            return;

        int nowIdx = GetNowMappedIndex();

        if (nowIdx != cachedNowIndex)
        {
            cachedNowIndex = nowIdx;
            ApplyCornerTempsForNowIndex(nowIdx);
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

        // ---------------- HISTORICAL OFF ----------------
        if (!forecastEnabled || forecastData == null || forecastData.measurements == null || forecastData.measurements.Length == 0)
        {
            // ugasi TMP label komponente (sigurno)
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
                    xLabel = FormatTimeHhMmSsUsingCurrentHour(data.measurements[measurementIndex].timestamp);
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

        // ---------------- FORECAST ON ----------------
        SetForecastTmpLabelsEnabled(true);

        int histStart = Mathf.Max(0, nowIdx - PastCount);
        int histEnd = nowIdx;
        int histCount = (histEnd - histStart) + 1;

        int futStart = Mathf.Min(forecastData.measurements.Length - 1, nowIdx + 1);
        int futEnd = Mathf.Min(forecastData.measurements.Length - 1, nowIdx + FutureCount);
        int futCount = (futEnd >= futStart) ? (futEnd - futStart + 1) : 0;

        int combinedCount = histCount + futCount;

        // ✅ Forecast ON: makni labele s grafa -> sve zero-width
        for (int xi = 0; xi < ForecastAxisCount; xi++)
        {
            chart.AddXAxisData(new string('\u200B', xi + 1));
        }

        // Data 61 u obje serije
        for (int i = 0; i < ForecastAxisCount; i++)
        {
            int ci = Mathf.Clamp(i, 0, 58);
            if (combinedCount > 0) ci = Mathf.Clamp(ci, 0, combinedCount - 1);

            float y = GetCombinedValueAtPoint(u, v, ci, histStart, histCount, futStart);

            chart.AddData(0, y); // historical
            chart.AddData(1, y); // forecast
        }

        ResetSerieData(chart, 0);
        ResetSerieData(chart, 1);

        // NOW = index 29 u ovom layoutu
        UpdateTemperatureText(GetSerieY(chart, 0, 29));

        // serie0: past + now (0..29)
        SetIgnoreRange(chart, 0, 30, 60, true);

        // serie1: now + future (29..58)
        SetIgnoreRange(chart, 1, 0, 28, true);
        SetIgnoreRange(chart, 1, 59, 60, true);

        SetPointRed(chart, 0, 29);
        HideAllSymbols(chart, 1);

        // ✅ postavi naših 5 TMP labela (vrijeme)
        UpdateForecastTmpTexts(nowIdx);

        var xAxisF = chart.EnsureChartComponent<XAxis>();
        xAxisF.type = Axis.AxisType.Category;
        xAxisF.axisLabel.formatter = "{value}";
        xAxisF.interval = 0;

        chart.RefreshChart();
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
        // Indeksi u konceptu:
        // L2 = now-29s (historical), C = now (historical), R2 = now+29 (forecast)
        // L1 = mid između L2 i C, R1 = mid između C i R2
        int idxC = nowIdx;
        int idxL2 = Mathf.Max(0, nowIdx - 29);
        int idxL1 = Mathf.Max(0, nowIdx - 15);

        int idxR2 = Mathf.Min(3599, nowIdx + 29);
        int idxR1 = Mathf.Min(3599, nowIdx + 15);

        // C, L1, L2 iz measurements
        if (data != null && data.measurements != null && data.measurements.Length > 0)
        {
            if (labelC != null) labelC.text = FormatTimeHhMmSsUsingCurrentHour(data.measurements[Mathf.Clamp(idxC, 0, data.measurements.Length - 1)].timestamp);
            if (labelL2 != null) labelL2.text = FormatTimeHhMmSsUsingCurrentHour(data.measurements[Mathf.Clamp(idxL2, 0, data.measurements.Length - 1)].timestamp);
            if (labelL1 != null) labelL1.text = FormatTimeHhMmSsUsingCurrentHour(data.measurements[Mathf.Clamp(idxL1, 0, data.measurements.Length - 1)].timestamp);
        }

        // R1, R2 iz forecast (ako postoji)
        if (forecastData != null && forecastData.measurements != null && forecastData.measurements.Length > 0)
        {
            if (labelR2 != null) labelR2.text = FormatTimeHhMmSsUsingCurrentHour(forecastData.measurements[Mathf.Clamp(idxR2, 0, forecastData.measurements.Length - 1)].timestamp);
            if (labelR1 != null) labelR1.text = FormatTimeHhMmSsUsingCurrentHour(forecastData.measurements[Mathf.Clamp(idxR1, 0, forecastData.measurements.Length - 1)].timestamp);
        }
    }

    // ======== Reset (pooling fix) ========

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

    // ======== Forecast mapping ========

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

    // ======== Axis/Title/Formatting ========

    private string FormatTimeHhMmSsUsingCurrentHour(string isoTimestamp)
    {
        if (!DateTime.TryParse(isoTimestamp, out DateTime dtFromDb))
            return isoTimestamp;

        DateTime now = DateTime.Now;
        DateTime display = new DateTime(now.Year, now.Month, now.Day, now.Hour, dtFromDb.Minute, dtFromDb.Second);
        return display.ToString("HH:mm:ss");
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
        if (heatmapEnabled) DeactivateHeatmap();
        else ActivateHeatmap();
    }

    private void ActivateHeatmap()
    {
        heatmapEnabled = true;
        GenerateMeshFromExistingPlane();

        heatmapTexture = new Texture2D(textureResolution, textureResolution);
        heatmapTexture.filterMode = FilterMode.Bilinear;
        heatmapTexture.wrapMode = TextureWrapMode.Clamp;

        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            planeMaterial = new Material(Shader.Find("Standard"));
            planeMaterial.mainTexture = heatmapTexture;
            planeMaterial.SetFloat("_Metallic", 0f);
            planeMaterial.SetFloat("_Glossiness", 0.2f);
            renderer.material = planeMaterial;
        }
        GenerateHeatmap();
    }

    private void DeactivateHeatmap()
    {
        heatmapEnabled = false;
        GetComponent<Renderer>().material = originalMaterial;
        GetComponent<MeshFilter>().mesh = originalMesh;
    }

    public void GenerateMeshFromExistingPlane()
    {
        MeshFilter meshFilter = GetComponent<MeshFilter>();
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

        meshFilter.mesh = mesh;

        if (GetComponent<MeshCollider>() == null) gameObject.AddComponent<MeshCollider>();
        GetComponent<MeshCollider>().sharedMesh = mesh;

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
        for (int y = 0; y < textureResolution; y++)
        {
            for (int x = 0; x < textureResolution; x++)
            {
                float uu = x / (float)(textureResolution - 1);
                float vv = y / (float)(textureResolution - 1);
                heatmapTexture.SetPixel(x, y, GetColorFromTemperature(BilinearInterpolation(uu, vv, angle1, angle2, angle3, angle4)));
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
}
