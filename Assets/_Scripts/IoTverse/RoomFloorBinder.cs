using System.Collections;
using UnityEngine;

public class RoomFloorBinder : MonoBehaviour
{
    [Header("Template (scene object)")]
    [SerializeField] private HeatMapStaticWithJson template;

    [Header("Find targets")]
    [SerializeField] private string floorName = "FLOOR";
    [SerializeField] private string planeMeshName = "PlaneMesh(PrefabSpawner Clone)";
    [SerializeField] private string cubeName = "Cube";

    [Header("Rename (PREPORUČENO)")]
    [SerializeField] private bool renameToFloor = true;
    [SerializeField] private string floorObjectName = "Floor";

    [Header("Layer settings")]
    [Tooltip("Layer koji AimOnGrip već raycasta")]
    [SerializeField] private string targetLayerName = "Environment";

    [Tooltip("Primijeni layer i na svu djecu (PREPORUČENO)")]
    [SerializeField] private bool applyLayerRecursively = true;

    [Header("Retry")]
    [SerializeField] private float retryEverySeconds = 0.25f;
    [SerializeField] private float giveUpAfterSeconds = 20f;

    [Header("After bind")]
    [SerializeField] private bool enableHeatmapOnBind = true;

    private bool alreadyBound = false;

    private void Start()
    {
        StartCoroutine(FindAndBind());
    }

    private IEnumerator FindAndBind()
    {
        float t = 0f;

        while (t < giveUpAfterSeconds)
        {
            if (alreadyBound)
                yield break;

            var cube = FindFloorCube();
            if (cube != null)
            {
                BindHeatmapTo(cube);
                alreadyBound = true;
                yield break;
            }

            t += retryEverySeconds;
            yield return new WaitForSeconds(retryEverySeconds);
        }

        Debug.LogError("[RoomFloorBinder] FLOOR nije pronađen. Provjeri nazive.");
    }

    private GameObject FindFloorCube()
    {
        var floor = GameObject.Find(floorName);
        if (floor == null) return null;

        Transform plane = floor.transform.Find(planeMeshName);
        if (plane == null) return null;

        Transform cube = plane.Find(cubeName);
        if (cube == null) return null;

        return cube.gameObject;
    }

    private void BindHeatmapTo(GameObject cube)
    {
        if (template == null)
        {
            Debug.LogError("[RoomFloorBinder] Template nije postavljen.");
            return;
        }

        //-----------------------------------
        // ✅ RENAME (NOVO)
        //-----------------------------------
        if (renameToFloor)
        {
            cube.name = floorObjectName;
            Debug.Log($"[RoomFloorBinder] Objekt preimenovan u '{floorObjectName}'.");
        }

        //-----------------------------------
        // ✅ SET LAYER
        //-----------------------------------
        int layerIndex = LayerMask.NameToLayer(targetLayerName);

        if (layerIndex == -1)
        {
            Debug.LogError($"[RoomFloorBinder] Layer '{targetLayerName}' ne postoji! Dodaj ga u Project Settings > Tags and Layers.");
        }
        else
        {
            SetLayer(cube, layerIndex);

            if (applyLayerRecursively)
            {
                foreach (Transform child in cube.GetComponentsInChildren<Transform>(true))
                    SetLayer(child.gameObject, layerIndex);
            }

            Debug.Log($"[RoomFloorBinder] Layer '{targetLayerName}' postavljen.");
        }

        //-----------------------------------
        // Collider
        //-----------------------------------
        if (cube.GetComponent<Collider>() == null)
        {
            if (cube.GetComponent<MeshFilter>() != null)
                cube.AddComponent<MeshCollider>();
            else
                cube.AddComponent<BoxCollider>();
        }

        //-----------------------------------
        // Heatmap
        //-----------------------------------
        var heat = cube.GetComponent<HeatMapStaticWithJson>();
        if (heat == null)
            heat = cube.AddComponent<HeatMapStaticWithJson>();

        CopySerialized(template, heat);

        heat.enabled = true;

        if (enableHeatmapOnBind)
            heat.SetHeatmapEnabled(true);

        Debug.Log("[RoomFloorBinder] Heatmap bindan na FLOOR.");
    }

    private void SetLayer(GameObject obj, int layer)
    {
        obj.layer = layer;
    }

    private static void CopySerialized(HeatMapStaticWithJson from, HeatMapStaticWithJson to)
    {
        var json = JsonUtility.ToJson(from);
        JsonUtility.FromJsonOverwrite(json, to);
    }
}
