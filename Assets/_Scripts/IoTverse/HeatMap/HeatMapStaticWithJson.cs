using System;
using System.Collections;
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
    [Tooltip("U bazi su mjerenja spremljena od 14:00:00 do 14:59:59. Ovaj sat se koristi samo kao informacija (mapiramo po mm:ss).")]
    [SerializeField] private int storedHour = 14;

    private MeasurementData data;

    [Header("XCharts Referenca")]
    public GameObject lineChart;

    // Tracking točke koju korisnik klikne
    private Vector3 trackedPoint;
    private bool isTrackingPoint = false;

    // Cache za brži rad
    private int cachedNowIndex = -1;
    private float chartUpdateTimer = 0f;
    private const float ChartUpdatePeriod = 1f; // svake sekunde

    void Start()
    {
        // Očisti inicijalne fejk podatke na chartu + postavi Y os fiksno 10..40 + title s datumom
        if (lineChart != null)
        {
            LineChart chart = lineChart.GetComponent<LineChart>();
            if (chart != null)
            {
                chart.ClearData();
                ForceYAxis10to40(chart);
                UpdateChartTitleWithDate(chart);
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

    // --- KLIKNUTO NA POD (poziva AimOnGrip) ---
    public void UpdateChartForPoint(Vector3 worldPoint)
    {
        if (lineChart == null || data == null || data.measurements == null || data.measurements.Length == 0)
            return;

        trackedPoint = worldPoint;
        isTrackingPoint = true;

        RefreshChartWindowForNow();
    }

    // --- API: Fetch svih 3600 mjerenja (jednom), zatim u runtimeu samo biramo indeks ---
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

            string json = req.downloadHandler.text;
            data = JsonUtility.FromJson<MeasurementData>(json);

            if (data == null || data.measurements == null || data.measurements.Length == 0)
            {
                Debug.LogError("[HeatMap] API vratio prazne podatke ili JSON nije u očekivanom formatu.");
                yield break;
            }

            Debug.Log($"[HeatMap] Učitano mjerenja: {data.measurements.Length}");

            // Postavi heatmap odmah na "sada"
            ApplyCornerTempsForNowIndex(GetNowMappedIndex());
        }
    }

    // --- MAPIRANJE "sada" -> index 0..3599 (po mm:ss) ---
    private int GetNowMappedIndex()
    {
        DateTime now = DateTime.Now;
        int secondsFromHourStart = (now.Minute * 60) + now.Second;

        int maxIndex = data != null && data.measurements != null ? data.measurements.Length - 1 : 3599;
        secondsFromHourStart = Mathf.Clamp(secondsFromHourStart, 0, maxIndex);

        return secondsFromHourStart;
    }

    // --- Primijeni kutne temperature (angle1..4) iz mjerenja za dani index ---
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

    // --- Refreš 30-točkasti prozor za "sada" (29 prije + sada) ---
    // NE DIRAMO logiku labela.
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

        int endIdx = nowIdx;
        int startIdx = Mathf.Max(0, endIdx - 29);
        int pointCount = (endIdx - startIdx) + 1; // 1..30
        int xCategoryCount = pointCount + 1;      // 31 kad je pointCount 30

        Vector3 local = transform.InverseTransformPoint(trackedPoint);
        float u = (local.x / (planeWidth * transform.lossyScale.x)) + 0.5f;
        float v = (local.z / (planeDepth * transform.lossyScale.z)) + 0.5f;
        u = Mathf.Clamp01(u);
        v = Mathf.Clamp01(v);

        LineChart chart = lineChart.GetComponent<LineChart>();
        if (chart == null) return;

        chart.ClearData();

        ForceYAxis10to40(chart);
        UpdateChartTitleWithDate(chart);

        int lastX = xCategoryCount - 1;

        int L0 = 0;
        int L1 = Mathf.Clamp(Mathf.RoundToInt(lastX * (6f / 30f)), 0, lastX);
        int L2 = Mathf.Clamp(Mathf.RoundToInt(lastX * (12f / 30f)), 0, lastX);
        int L3 = Mathf.Clamp(Mathf.RoundToInt(lastX * (18f / 30f)), 0, lastX);
        int L4 = Mathf.Clamp(Mathf.RoundToInt(lastX * (24f / 30f)), 0, lastX);
        int L5 = lastX;

        // 1) X kategorije (31)
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
            else
            {
                xLabel = new string('\u200B', xi + 1);
            }

            chart.AddXAxisData(xLabel);
        }

        // 2) Podaci (30) ostaju na indeksima 0..29
        for (int i = 0; i < pointCount; i++)
        {
            int idx = Mathf.Clamp(startIdx + i, 0, data.measurements.Length - 1);
            Measurement m = data.measurements[idx];

            float tempAtPoint = BilinearInterpolation(u, v, m.temperature1, m.temperature2, m.temperature3, m.temperature4);

            chart.AddData(0, tempAtPoint);

            if (i == pointCount - 1)
                UpdateTemperatureText(tempAtPoint);
        }

        // ✅ OBOJI zadnju točku crveno + ostavi Emphasis state (bez diranja labela)
        SetLastPointEmphasisAndRed(chart, pointCount - 1);

        var xAxis = chart.EnsureChartComponent<XAxis>();
        xAxis.type = Axis.AxisType.Category;
        xAxis.axisLabel.formatter = "{value}";
        xAxis.interval = 0;

        chart.RefreshChart();
    }

    // ✅ Emphasis + crvena boja (ItemStyle + Symbol), bez ikakvog diranja X labela
    private void SetLastPointEmphasisAndRed(LineChart chart, int lastPointIndex)
    {
        if (chart == null || chart.series == null || chart.series.Count == 0) return;

        var serie = chart.series[0];
        if (serie == null || serie.dataCount == 0) return;

        int idx = Mathf.Clamp(lastPointIndex, 0, serie.dataCount - 1);

        // reset state svih točaka na Normal
        for (int i = 0; i < serie.dataCount; i++)
            serie.data[i].state = SerieState.Normal;

        // zadnja točka -> Emphasis
        var last = serie.data[idx];
        last.state = SerieState.Emphasis;

        // osiguraj komponente i postavi boju
        // (u tvojoj verziji itemStyle i symbol mogu biti null dok se ne EnsureComponent)
        var item = last.EnsureComponent<ItemStyle>();
        item.show = true;
        item.color = Color.red;

        var sym = last.EnsureComponent<SerieSymbol>();
        sym.show = true;
        sym.color = Color.red;
    }

    private string FormatTimeHhMmSsUsingCurrentHour(string isoTimestamp)
    {
        if (!DateTime.TryParse(isoTimestamp, out DateTime dtFromDb))
            return isoTimestamp;

        DateTime now = DateTime.Now;

        DateTime display = new DateTime(
            now.Year, now.Month, now.Day,
            now.Hour,
            dtFromDb.Minute,
            dtFromDb.Second
        );

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

    public void UpdateTemperatureText(float temperature)
    {
        AimOnGrip aimScript = FindObjectOfType<AimOnGrip>();
        if (aimScript != null)
            aimScript.UpdateTemperatureDisplay(temperature);
    }

    public void StopTracking()
    {
        isTrackingPoint = false;
        Debug.Log("[HeatMap] Zaustavljeno praćenje točke");
    }

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
                float u = x / (float)meshSegmentsX;
                float v = z / (float)meshSegmentsZ;

                vertices[i] = new Vector3(
                    (u - 0.5f) * planeWidth,
                    originalMesh.vertices[Mathf.Clamp(i, 0, originalMesh.vertices.Length - 1)].y,
                    (v - 0.5f) * planeDepth
                );

                uv[i] = new Vector2(u, v);
                colors[i] = GetColorFromTemperature(BilinearInterpolation(u, v, angle1, angle2, angle3, angle4));
            }
        }

        int[] triangles = new int[meshSegmentsX * meshSegmentsZ * 6];
        int t = 0;
        for (int z = 0; z < meshSegmentsZ; z++)
        {
            for (int x = 0; x < meshSegmentsX; x++)
            {
                int i = z * vertCountX + x;

                triangles[t++] = i;
                triangles[t++] = i + vertCountX;
                triangles[t++] = i + 1;

                triangles[t++] = i + 1;
                triangles[t++] = i + vertCountX;
                triangles[t++] = i + vertCountX + 1;
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

    bool HasValuesChanged()
    {
        return !Mathf.Approximately(angle1, previousAngle1) ||
               !Mathf.Approximately(angle2, previousAngle2) ||
               !Mathf.Approximately(angle3, previousAngle3) ||
               !Mathf.Approximately(angle4, previousAngle4);
    }

    void SaveCurrentAngles()
    {
        previousAngle1 = angle1;
        previousAngle2 = angle2;
        previousAngle3 = angle3;
        previousAngle4 = angle4;
    }

    void GenerateHeatmap()
    {
        for (int y = 0; y < textureResolution; y++)
        {
            for (int x = 0; x < textureResolution; x++)
            {
                float u = x / (float)(textureResolution - 1);
                float v = y / (float)(textureResolution - 1);
                heatmapTexture.SetPixel(x, y, GetColorFromTemperature(BilinearInterpolation(u, v, angle1, angle2, angle3, angle4)));
            }
        }
        heatmapTexture.Apply();
        if (meshGenerated) UpdateVertexColors();
    }

    Color GetColorFromTemperature(float temp)
    {
        float t = Mathf.InverseLerp(minGlobalTemp, maxGlobalTemp, temp);
        if (t < 0.5f) return Color.Lerp(coldColor, midColor, t * 2f);
        else return Color.Lerp(midColor, warmColor, (t - 0.5f) * 2f);
    }

    void UpdateVertexColors()
    {
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        Color[] colors = mesh.colors;
        Vector2[] uvs = mesh.uv;

        for (int i = 0; i < colors.Length; i++)
        {
            colors[i] = GetColorFromTemperature(BilinearInterpolation(uvs[i].x, uvs[i].y, angle1, angle2, angle3, angle4));
        }
        mesh.colors = colors;
    }

    float BilinearInterpolation(float u, float v, float q11, float q21, float q12, float q22)
    {
        return Mathf.Lerp(Mathf.Lerp(q11, q21, u), Mathf.Lerp(q12, q22, u), v);
    }

    public void RefreshHeatmap() { GenerateHeatmap(); }

    public void SetTemperatures(float t1, float t2, float t3, float t4)
    {
        angle1 = t1; angle2 = t2; angle3 = t3; angle4 = t4;
        GenerateHeatmap();
    }

    public float GetTemperatureAtUV(Vector2 uv)
    {
        return BilinearInterpolation(uv.x, uv.y, angle1, angle2, angle3, angle4);
    }

    public float GetTemperatureAtPointWorld(Vector3 worldPoint)
    {
        Vector3 local = transform.InverseTransformPoint(worldPoint);
        float u = (local.x / (planeWidth * transform.lossyScale.x)) + 0.5f;
        float v = (local.z / (planeDepth * transform.lossyScale.z)) + 0.5f;
        return GetTemperatureAtUV(new Vector2(Mathf.Clamp01(u), Mathf.Clamp01(v)));
    }

    // ---------------- JSON MODELI ----------------
    [Serializable]
    public class MeasurementData
    {
        public Measurement[] measurements;
    }

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
