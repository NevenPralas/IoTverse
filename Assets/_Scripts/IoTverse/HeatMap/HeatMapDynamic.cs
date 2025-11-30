using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeatMapDynamic : MonoBehaviour
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
    [Tooltip("Vise segmenata = gladji gradijent (cca. 50-100)")]
    public int meshSegmentsX = 80;
    public int meshSegmentsZ = 80;

    [Header("Boje gradienta")]
    public Color coldColor = new Color(0, 0, 1);     // Plava  (0°C)
    public Color warmColor = new Color(1, 0, 0);     // Crvena (40°C)
    public Color midColor = new Color(1, 0.5f, 0);  // Narancasta (20°C)

    private Texture2D heatmapTexture;
    private Material planeMaterial;

    private float previousAngle1, previousAngle2, previousAngle3, previousAngle4;

    private bool meshGenerated = false;

    void Start()
    {
        // Mesh s originalnim dimenzijama
        GenerateMeshFromExistingPlane();

        // Stvaranje teksture
        heatmapTexture = new Texture2D(textureResolution, textureResolution);
        heatmapTexture.filterMode = FilterMode.Bilinear;
        heatmapTexture.wrapMode = TextureWrapMode.Clamp;

        // Postavljanje materijala
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

    void GenerateMeshFromExistingPlane()
    {
        // Dohvacanje dimenzije iz lokalnog skaliranja
        Vector3 scale = transform.localScale;
        float width = 10f * scale.x; // Unity Plane je 10x10
        float depth = 10f * scale.z;

        // Generiranje novog mesha
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null)
            meshFilter = gameObject.AddComponent<MeshFilter>();

        Mesh mesh = new Mesh();
        mesh.name = "HeatMap Mesh";

        int vertCountX = meshSegmentsX + 1;
        int vertCountZ = meshSegmentsZ + 1;
        int vertCount = vertCountX * vertCountZ;

        Vector3[] vertices = new Vector3[vertCount];
        Vector2[] uv = new Vector2[vertCount];
        Color[] colors = new Color[vertCount];

        // Vertex
        for (int z = 0; z <= meshSegmentsZ; z++)
        {
            for (int x = 0; x <= meshSegmentsX; x++)
            {
                int index = z * vertCountX + x;

                float xPercent = x / (float)meshSegmentsX;
                float zPercent = z / (float)meshSegmentsZ;

                float xPos = (xPercent - 0.5f) * width;
                float zPos = (zPercent - 0.5f) * depth;

                vertices[index] = new Vector3(xPos, 0, zPos);
                uv[index] = new Vector2(xPercent, zPercent);

                float temp = BilinearInterpolation(xPercent, zPercent, angle1, angle2, angle3, angle4);
                float minTemp = Mathf.Min(angle1, angle2, angle3, angle4);
                float maxTemp = Mathf.Max(angle1, angle2, angle3, angle4);

                if (Mathf.Approximately(minTemp, maxTemp))
                {
                    minTemp -= 0.5f;
                    maxTemp += 0.5f;
                }

                float normalizedTemp = Mathf.InverseLerp(minTemp, maxTemp, temp);
                colors[index] = GetColorFromTemperature(normalizedTemp);
            }
        }

        // Trokuti
        int[] triangles = new int[meshSegmentsX * meshSegmentsZ * 6];
        int triIndex = 0;

        for (int z = 0; z < meshSegmentsZ; z++)
        {
            for (int x = 0; x < meshSegmentsX; x++)
            {
                int vertIndex = z * vertCountX + x;

                triangles[triIndex++] = vertIndex;
                triangles[triIndex++] = vertIndex + vertCountX;
                triangles[triIndex++] = vertIndex + 1;

                triangles[triIndex++] = vertIndex + 1;
                triangles[triIndex++] = vertIndex + vertCountX;
                triangles[triIndex++] = vertIndex + vertCountX + 1;
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
        if (collider == null)
            collider = gameObject.AddComponent<MeshCollider>();
        collider.sharedMesh = mesh;

        meshGenerated = true;

        Debug.Log($"Mesh generiran: {width:F2}x{depth:F2}m sa {meshSegmentsX}x{meshSegmentsZ} segmenata");
    }

    bool HasValuesChanged()
    {
        return
            !Mathf.Approximately(angle1, previousAngle1) ||
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
        float minTemp = Mathf.Min(angle1, angle2, angle3, angle4);
        float maxTemp = Mathf.Max(angle1, angle2, angle3, angle4);

        if (Mathf.Approximately(minTemp, maxTemp))
        {
            minTemp -= 0.5f;
            maxTemp += 0.5f;
        }

        for (int y = 0; y < textureResolution; y++)
        {
            for (int x = 0; x < textureResolution; x++)
            {
                float u = x / (float)(textureResolution - 1);
                float v = y / (float)(textureResolution - 1);

                float temperature = BilinearInterpolation(u, v, angle1, angle2, angle3, angle4);
                float normalizedTemp = Mathf.InverseLerp(minTemp, maxTemp, temperature);

                Color pixelColor = GetColorFromTemperature(normalizedTemp);
                heatmapTexture.SetPixel(x, y, pixelColor);
            }
        }

        heatmapTexture.Apply();

        if (meshGenerated)
            UpdateVertexColors();
    }

    void UpdateVertexColors()
    {
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null || meshFilter.mesh == null)
            return;

        Mesh mesh = meshFilter.mesh;
        Color[] colors = mesh.colors;
        Vector2[] uvs = mesh.uv;

        float minTemp = Mathf.Min(angle1, angle2, angle3, angle4);
        float maxTemp = Mathf.Max(angle1, angle2, angle3, angle4);

        if (Mathf.Approximately(minTemp, maxTemp))
        {
            minTemp -= 0.5f;
            maxTemp += 0.5f;
        }

        for (int i = 0; i < colors.Length; i++)
        {
            float temp = BilinearInterpolation(uvs[i].x, uvs[i].y, angle1, angle2, angle3, angle4);
            float normalizedTemp = Mathf.InverseLerp(minTemp, maxTemp, temp);
            colors[i] = GetColorFromTemperature(normalizedTemp);
        }

        mesh.colors = colors;
    }

    float BilinearInterpolation(float u, float v, float q11, float q21, float q12, float q22)
    {
        float r1 = Mathf.Lerp(q11, q21, u);
        float r2 = Mathf.Lerp(q12, q22, u);
        return Mathf.Lerp(r1, r2, v);
    }

    Color GetColorFromTemperature(float t)
    {
        if (t < 0.5f)
        {
            return Color.Lerp(coldColor, midColor, t * 2f);
        }
        else
        {
            return Color.Lerp(midColor, warmColor, (t - 0.5f) * 2f);
        }
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
