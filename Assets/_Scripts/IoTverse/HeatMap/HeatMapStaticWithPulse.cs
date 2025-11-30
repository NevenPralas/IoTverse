using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeatMapStaticWithPulse : MonoBehaviour
{
    [Header("Temperaturne vrijednosti na kutovima")]
    public float angle1 = 10.5f; // Donji lijevi kut
    public float angle2 = 15.4f; // Donji desni kut
    public float angle3 = 12.2f; // Gornji lijevi kut
    public float angle4 = 10.8f; // Gornji desni kut

    [Header("Postavke")]
    [Tooltip("Veca rezolucija = gladji prijelazi")]
    public int textureResolution = 1024;
    public bool autoUpdate = true;

    [Header("Mesh postavke")]
    public int meshSegmentsX = 80;
    public int meshSegmentsZ = 80;

    [Header("Globalna temperaturna ljestvica (STATIC)")]
    public float minGlobalTemp = 0f;
    public float maxGlobalTemp = 40f;

    [Header("Boje gradienta")]
    public Color coldColor = new Color(0f, 0f, 1f);        // Plava (0°C)
    public Color purpleColor = new Color(0.45f, 0.15f, 0.8f); // Ljubicasta (~10°C)
    public Color midColor = new Color(1f, 0.5f, 0f);    // Narancasta (20°C)
    public Color lightRed = new Color(1f, 0.35f, 0.25f); // Svjetlocrvena (~30°C)
    public Color warmColor = new Color(0.85f, 0.05f, 0.05f); // Tamnija crvena (40°C)

    [Header("animirani efekti heatmapa")]
    public bool enableShimmer = true;
    [Range(0f, 1f)] public float shimmerStrength = 0.08f;
    public float shimmerSpeed = 1.2f;

    public bool enableNoiseWaves = true;
    public float noiseScale = 2.5f;
    public float noiseSpeed = 0.2f;
    [Range(0f, 5f)] public float noiseIntensity = 0.8f;

    public bool enablePulse = false;
    public float pulseSpeed = 0.8f;
    [Range(0f, 2f)] public float pulseIntensity = 0.2f;

    // internals
    private Texture2D heatmapTexture;
    private Material planeMaterial;
    private float previousAngle1, previousAngle2, previousAngle3, previousAngle4;
    private bool meshGenerated = false;

    // base temperature buffer (computed once per GenerateHeatmap)
    private float[,] baseHeat; // size [textureResolution, textureResolution]

    void Start()
    {
        // priprema jezgre
        GenerateMeshFromExistingPlane();

        heatmapTexture = new Texture2D(textureResolution, textureResolution, TextureFormat.RGBA32, false);
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

        // rezerviraj buffer
        baseHeat = new float[textureResolution, textureResolution];

        GenerateHeatmap(); // popuni baseHeat i teksturu
        SaveCurrentAngles();
    }

    void Update()
    {
        // Ako su temperature promijenjene → regeneriraj bazno polje
        if (autoUpdate && HasValuesChanged())
        {
            GenerateHeatmap();
            SaveCurrentAngles();
        }

        // Animiraj (svaki frame koristimo baseHeat + efekte)
        AnimateHeatmap();
    }

    // GENERIRAJ BAZNU TEMPERATURU (BILINEARNO)
    void GenerateHeatmap()
    {
        // Izracunaj bazne temperature u buffer (bez animacije)
        for (int y = 0; y < textureResolution; y++)
        {
            float v = y / (float)(textureResolution - 1);

            for (int x = 0; x < textureResolution; x++)
            {
                float u = x / (float)(textureResolution - 1);

                // bilinearna interpolacija izmedju 4 kuta
                float bottom = Mathf.Lerp(angle1, angle2, u); // donja ivica
                float top = Mathf.Lerp(angle3, angle4, u);    // gornja ivica
                float temp = Mathf.Lerp(bottom, top, v);

                baseHeat[x, y] = temp;
            }
        }

        // Postavi poctne boje (bez animacije) i apply
        for (int y = 0; y < textureResolution; y++)
        {
            for (int x = 0; x < textureResolution; x++)
            {
                float temp = baseHeat[x, y];
                Color c = GetColorFromTemperature(temp);
                heatmapTexture.SetPixel(x, y, c);
            }
        }

        heatmapTexture.Apply();

        // azuriraj vertex boje (ako mesh postoji) - koristimo bazne temperature za vertex boju
        if (meshGenerated)
            UpdateVertexColors();
    }

    // ANIMACIJA: koristi baseHeat + efekte -> upis u teksturu
    void AnimateHeatmap()
    {
        if (!enableShimmer && !enableNoiseWaves && !enablePulse)
            return;

        // Iteriraj pixele i pisi boju
        for (int y = 0; y < textureResolution; y++)
        {
            float v = y / (float)(textureResolution - 1);

            for (int x = 0; x < textureResolution; x++)
            {
                float u = x / (float)(textureResolution - 1);

                // pocetna temperatura iz baze
                float temp = baseHeat[x, y];

                // SHIMMER: mikro sinusne oscilacije (lagano pomicu temperaturu)
                if (enableShimmer)
                {
                    float phase = (u * 10f + v * 10f) + Time.time * shimmerSpeed;
                    temp += Mathf.Sin(phase) * shimmerStrength;
                }

                // NOISE WAVES: perlin noise blok koji lagano putuje
                if (enableNoiseWaves)
                {
                    float nx = (u + Time.time * noiseSpeed) * noiseScale;
                    float ny = (v + Time.time * noiseSpeed) * noiseScale;
                    float n = Mathf.PerlinNoise(nx, ny); // 0..1
                    n = (n - 0.5f) * 2f; // -1..1
                    temp += n * noiseIntensity;
                }

                // PULSE: globalni puls intenziteta
                if (enablePulse)
                {
                    temp += Mathf.Sin(Time.time * pulseSpeed) * pulseIntensity;
                }

                // mapi u boju i postavi pixel
                Color c = GetColorFromTemperature(temp);
                heatmapTexture.SetPixel(x, y, c);
            }
        }

        heatmapTexture.Apply();
    }

    // GENERIRANJE MESH-A
    void GenerateMeshFromExistingPlane()
    {
        Vector3 scale = transform.localScale;
        float width = 10f * scale.x;   // Unity standard plane 10x10
        float depth = 10f * scale.z;

        MeshFilter meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null) meshFilter = gameObject.AddComponent<MeshFilter>();

        Mesh mesh = new Mesh();
        mesh.name = "HeatMap Mesh";

        int vertCountX = meshSegmentsX + 1;
        int vertCountZ = meshSegmentsZ + 1;
        int vertCount = vertCountX * vertCountZ;

        Vector3[] vertices = new Vector3[vertCount];
        Vector2[] uv = new Vector2[vertCount];
        Color[] colors = new Color[vertCount];

        for (int z = 0; z <= meshSegmentsZ; z++)
        {
            for (int x = 0; x <= meshSegmentsX; x++)
            {
                int i = z * vertCountX + x;

                float u = x / (float)meshSegmentsX;
                float v = z / (float)meshSegmentsZ;

                float xPos = (u - 0.5f) * width;
                float zPos = (v - 0.5f) * depth;

                vertices[i] = new Vector3(xPos, 0f, zPos);
                uv[i] = new Vector2(u, v);

                // direktno bojenje na vertexu koristeci baznu bilinearnu interpolaciju
                float temp = BilinearInterpolation(u, v, angle1, angle2, angle3, angle4);
                colors[i] = GetColorFromTemperature(temp);
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

        // prilagodi transform skaliranje (isti postupak kao u originalnoj skripti)
        transform.localScale = new Vector3(0.25f, 1f, 0.25f);

        MeshCollider collider = GetComponent<MeshCollider>();
        if (collider == null) collider = gameObject.AddComponent<MeshCollider>();
        collider.sharedMesh = mesh;

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

    // MAPIRANJE TEMPERATURE U BOJU (FIKSNA SKALA 0..40)
    // Ukljucuje precizne prijelaze: 0 (blue) -> 10 (purple) -> 20 (orange) -> 30 (light red) -> 40 (cherry red)
    Color GetColorFromTemperature(float temp)
    {
        // normaliziraj prema globalnom fiksnom rasponu
        float t = Mathf.InverseLerp(minGlobalTemp, maxGlobalTemp, temp);
        t = Mathf.Clamp01(t);

        // zelimo tocku na 0, 0.25, 0.5, 0.75, 1.0 odgovarajuce na 0,10,20,30,40
        // mapiramo t u segment od 0..1 podijeljen u 4 dijela
        if (t <= 0.25f)
        {
            // 0.00..0.25 : coldColor -> purpleColor   (0..10)
            float local = t / 0.25f;
            return Color.Lerp(coldColor, purpleColor, local);
        }
        else if (t <= 0.5f)
        {
            // 0.25..0.5 : purpleColor -> midColor      (10..20)
            float local = (t - 0.25f) / 0.25f;
            return Color.Lerp(purpleColor, midColor, local);
        }
        else if (t <= 0.75f)
        {
            // 0.5..0.75 : midColor -> lightRed         (20..30)
            float local = (t - 0.5f) / 0.25f;
            return Color.Lerp(midColor, lightRed, local);
        }
        else
        {
            // 0.75..1.0 : lightRed -> warmColor        (30..40)
            float local = (t - 0.75f) / 0.25f;
            return Color.Lerp(lightRed, warmColor, local);
        }
    }

    // AZURIRANJE BAZNIH BOJA NA VERTEXIMA (ne animira vertex boje)
    void UpdateVertexColors()
    {
        MeshFilter mf = GetComponent<MeshFilter>();
        if (mf == null || mf.mesh == null) return;

        Mesh mesh = mf.mesh;
        Color[] colors = mesh.colors;
        Vector2[] uvs = mesh.uv;

        for (int i = 0; i < colors.Length; i++)
        {
            float u = uvs[i].x;
            float v = uvs[i].y;
            float temp = BilinearInterpolation(u, v, angle1, angle2, angle3, angle4);
            colors[i] = GetColorFromTemperature(temp);
        }

        mesh.colors = colors;
    }

    // BILINEARNA INTERPOLACIJA (pomocna)
    float BilinearInterpolation(float u, float v, float q11, float q21, float q12, float q22)
    {
        float bottom = Mathf.Lerp(q11, q21, u);
        float top = Mathf.Lerp(q12, q22, u);
        return Mathf.Lerp(bottom, top, v);
    }

    // JAVNE METODE
    public void RefreshHeatmap()
    {
        GenerateHeatmap();
    }

    public void SetTemperatures(float t1, float t2, float t3, float t4)
    {
        angle1 = t1;
        angle2 = t2;
        angle3 = t3;
        angle4 = t4;
        GenerateHeatmap();
    }
}
