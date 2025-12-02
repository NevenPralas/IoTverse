using System.Collections.Generic;
using UnityEngine;

public class HeatMapStatic : MonoBehaviour
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
    public float minGlobalTemp = 0f;
    public float maxGlobalTemp = 40f;

    [Header("Boje gradienta")]
    public Color coldColor = new Color(0, 0, 1);
    public Color midColor = new Color(1f, 0.5f, 0f);
    public Color warmColor = new Color(1, 0, 0);

    // Toggle
    public bool heatmapEnabled = true;
    private Mesh originalMesh;
    private Material originalMaterial;

    private Texture2D heatmapTexture;
    private Material planeMaterial;
    private float previousAngle1, previousAngle2, previousAngle3, previousAngle4;

    private bool meshGenerated = false;
    private float planeWidth;
    private float planeDepth;

    void Start()
    {
        originalMesh = GetComponent<MeshFilter>().mesh;
        originalMaterial = GetComponent<Renderer>().material;

        planeWidth = originalMesh.bounds.size.x;
        planeDepth = originalMesh.bounds.size.z;

        ActivateHeatmap();
        SaveCurrentAngles();
    }

    void Update()
    {
        if (!heatmapEnabled)
            return;

        if (autoUpdate && HasValuesChanged())
        {
            GenerateHeatmap();
            SaveCurrentAngles();
        }
    }

    public void ToggleHeatmap()
    {
        if (heatmapEnabled)
            DeactivateHeatmap();
        else
            ActivateHeatmap();
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

        Renderer renderer = GetComponent<Renderer>();
        renderer.material = originalMaterial;

        MeshFilter mf = GetComponent<MeshFilter>();
        mf.mesh = originalMesh;
    }

    public void GenerateMeshFromExistingPlane()
    {
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

                float xPos = (u - 0.5f) * planeWidth;
                float zPos = (v - 0.5f) * planeDepth;

                float yPos = originalMesh.vertices[Mathf.Clamp(i, 0, originalMesh.vertices.Length - 1)].y;

                vertices[i] = new Vector3(xPos, yPos, zPos);
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

    // =====================================================
    //   *** NEW: DOHVAT TEMPERATURE ***
    // =====================================================

    public float GetTemperatureAtUV(Vector2 uv)
    {
        return BilinearInterpolation(uv.x, uv.y, angle1, angle2, angle3, angle4);
    }

    public float GetTemperatureAtPointWorld(Vector3 worldPoint)
    {
        Vector3 local = transform.InverseTransformPoint(worldPoint);

        float meshWidth = originalMesh.bounds.size.x * transform.lossyScale.x;
        float meshDepth = originalMesh.bounds.size.z * transform.lossyScale.z;

        if (meshWidth == 0) meshWidth = planeWidth * transform.lossyScale.x;
        if (meshDepth == 0) meshDepth = planeDepth * transform.lossyScale.z;

        float u = (local.x / meshWidth) + 0.5f;
        float v = (local.z / meshDepth) + 0.5f;

        u = Mathf.Clamp01(u);
        v = Mathf.Clamp01(v);

        return GetTemperatureAtUV(new Vector2(u, v));
    }
}
