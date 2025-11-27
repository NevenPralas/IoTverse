using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeatMapStatic : MonoBehaviour
{
    [Header("Temperaturne vrijednosti na kutovima")]
    public float angle1 = 10.5f; // Donji lijevi kut
    public float angle2 = 15.4f; // Donji desni kut
    public float angle3 = 12.2f; // Gornji lijevi kut
    public float angle4 = 10.8f; // Gornji desni kut

    [Header("Postavke")]
    [Tooltip("Veća rezolucija = glađi prijelazi")]
    public int textureResolution = 1024;
    public bool autoUpdate = true;

    [Header("Mesh postavke")]
    public int meshSegmentsX = 80;
    public int meshSegmentsZ = 80;

    [Header("GLOBALNA TEMPERATURNA SKALA (NE MIJENJA SE)")]
    public float minGlobalTemp = 0f;
    public float maxGlobalTemp = 40f;

    [Header("Boje gradienta")]
    public Color coldColor = new Color(0, 0, 1);        // Plava (0°C)
    public Color midColor = new Color(1f, 0.5f, 0f);    // Narančasta (20°C)
    public Color warmColor = new Color(1, 0, 0);        // Crvena (40°C)

    private Texture2D heatmapTexture;
    private Material planeMaterial;
    private float previousAngle1, previousAngle2, previousAngle3, previousAngle4;
    private bool meshGenerated = false;

    void Start()
    {
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
        SaveCurrentAngles();
    }

    void Update()
    {
        if (autoUpdate && HasValuesChanged())
        {
            GenerateHeatmap();
            SaveCurrentAngles();
        }
    }

    // ---------------------------------------------------------
    // MESH GENERACIJA
    // ---------------------------------------------------------
    void GenerateMeshFromExistingPlane()
    {
        Vector3 scale = transform.localScale;
        float width = 10f * scale.x;
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

                vertices[i] = new Vector3(xPos, 0, zPos);
                uv[i] = new Vector2(u, v);

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

        transform.localScale = new Vector3(0.25f, 1, 0.25f);

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

    // ---------------------------------------------------------
    // HEATMAP GENERACIJA (TEKSTURA)
    // ---------------------------------------------------------
    void GenerateHeatmap()
    {
        for (int y = 0; y < textureResolution; y++)
        {
            for (int x = 0; x < textureResolution; x++)
            {
                float u = x / (float)(textureResolution - 1);
                float v = y / (float)(textureResolution - 1);

                float temp = BilinearInterpolation(u, v, angle1, angle2, angle3, angle4);
                heatmapTexture.SetPixel(x, y, GetColorFromTemperature(temp));
            }
        }

        heatmapTexture.Apply();

        if (meshGenerated)
            UpdateVertexColors();
    }

    // ---------------------------------------------------------
    // STATIC GLOBAL COLOR MAPPING 0°C → 40°C
    // ---------------------------------------------------------
    Color GetColorFromTemperature(float temp)
    {
        float t = Mathf.InverseLerp(minGlobalTemp, maxGlobalTemp, temp);

        if (t < 0.5f)
            return Color.Lerp(coldColor, midColor, t * 2f);
        else
            return Color.Lerp(midColor, warmColor, (t - 0.5f) * 2f);
    }

    void UpdateVertexColors()
    {
        MeshFilter mf = GetComponent<MeshFilter>();
        if (mf == null || mf.mesh == null) return;

        Mesh mesh = mf.mesh;

        Color[] colors = mesh.colors;
        Vector2[] uvs = mesh.uv;

        for (int i = 0; i < colors.Length; i++)
        {
            float t = BilinearInterpolation(uvs[i].x, uvs[i].y, angle1, angle2, angle3, angle4);
            colors[i] = GetColorFromTemperature(t);
        }

        mesh.colors = colors;
    }

    // ---------------------------------------------------------
    // BILINEARNA INTERPOLACIJA
    // ---------------------------------------------------------
    float BilinearInterpolation(float u, float v, float q11, float q21, float q12, float q22)
    {
        float bottom = Mathf.Lerp(q11, q21, u);
        float top = Mathf.Lerp(q12, q22, u);
        return Mathf.Lerp(bottom, top, v);
    }

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
