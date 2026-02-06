using System.Collections;
using UnityEngine;

public class RoomFloorBinder : MonoBehaviour
{
    [Header("Template (scene object)")]
    [SerializeField] private HeatMapStaticWithJson template;

    [Header("Find targets (path under ROOM scene)")]
    [SerializeField] private string floorName = "FLOOR";
    [SerializeField] private string planeMeshName = "PlaneMesh(PrefabSpawner Clone)";
    [SerializeField] private string cubeName = "Cube";

    [Header("Rename (optional, recommended)")]
    [SerializeField] private bool renameToFloor = true;
    [SerializeField] private string floorObjectName = "Floor";

    [Header("Layer settings")]
    [Tooltip("Layer koji AimOnGrip već raycasta (ne mijenjamo AimOnGrip)")]
    [SerializeField] private string targetLayerName = "Environment";
    [Tooltip("Primijeni layer i na svu djecu (preporučeno)")]
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
            if (alreadyBound) yield break;

            var target = FindFloorTarget();
            if (target != null)
            {
                BindTo(target);
                alreadyBound = true;
                yield break;
            }

            t += retryEverySeconds;
            yield return new WaitForSeconds(retryEverySeconds);
        }

        Debug.LogError("[RoomFloorBinder] Nisam našao FLOOR target u zadanom roku. Provjeri nazive (FLOOR/PlaneMesh/Cube).");
    }

    private GameObject FindFloorTarget()
    {
        // Find FLOOR anywhere in the scene
        var floor = GameObject.Find(floorName);
        if (floor == null) return null;

        // Find plane mesh child
        Transform plane = floor.transform.Find(planeMeshName);
        if (plane == null) return null;

        // Find cube child
        Transform cube = plane.Find(cubeName);
        if (cube == null) return null;

        return cube.gameObject;
    }

    private void BindTo(GameObject target)
    {
        if (template == null)
        {
            Debug.LogError("[RoomFloorBinder] Template nije postavljen (drag HeatmapTemplate u Inspector).");
            return;
        }

        // Rename for easier debug / stable find
        if (renameToFloor)
        {
            target.name = floorObjectName;
            Debug.Log($"[RoomFloorBinder] Target preimenovan u '{floorObjectName}'.");
        }

        // Set layer for raycasts (AimOnGrip stays unchanged)
        int layerIndex = LayerMask.NameToLayer(targetLayerName);
        if (layerIndex == -1)
        {
            Debug.LogError($"[RoomFloorBinder] Layer '{targetLayerName}' ne postoji! Dodaj ga u Project Settings > Tags and Layers.");
        }
        else
        {
            SetLayer(target, layerIndex);

            if (applyLayerRecursively)
            {
                foreach (Transform tr in target.GetComponentsInChildren<Transform>(true))
                    SetLayer(tr.gameObject, layerIndex);
            }

            Debug.Log($"[RoomFloorBinder] Layer '{targetLayerName}' postavljen (recursive={applyLayerRecursively}).");
        }

        // Ensure collider for raycast
        EnsureCollider(target);

        // Add HeatMapStaticWithJson to THIS runtime floor object
        var heat = target.GetComponent<HeatMapStaticWithJson>();
        if (heat == null)
            heat = target.AddComponent<HeatMapStaticWithJson>();

        // Copy all serialized fields from template -> runtime instance
        CopySerialized(template, heat);

        heat.enabled = true;

        if (enableHeatmapOnBind)
            heat.SetHeatmapEnabled(true);

        Debug.Log("[RoomFloorBinder] Heatmap bindan na runtime FLOOR target.");
    }

    private void EnsureCollider(GameObject target)
    {
        if (target.GetComponent<Collider>() != null)
            return;

        // Prefer MeshCollider if mesh exists, otherwise BoxCollider
        if (target.GetComponent<MeshFilter>() != null)
            target.AddComponent<MeshCollider>();
        else
            target.AddComponent<BoxCollider>();
    }

    private void SetLayer(GameObject obj, int layer)
    {
        obj.layer = layer;
    }

    private static void CopySerialized(HeatMapStaticWithJson from, HeatMapStaticWithJson to)
    {
        // Copies serialized values like inspector settings.
        var json = JsonUtility.ToJson(from);
        JsonUtility.FromJsonOverwrite(json, to);
    }
}
