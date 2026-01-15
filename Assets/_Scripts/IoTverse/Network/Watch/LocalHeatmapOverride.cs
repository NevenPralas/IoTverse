using UnityEngine;

public class LocalHeatmapOverride : MonoBehaviour
{
    // True = ovaj klijent skriva heatmap lokalno
    [SerializeField] private bool hideHeatmapLocally = false;

    public bool HideHeatmapLocally => hideHeatmapLocally;

    public void Toggle()
    {
        hideHeatmapLocally = !hideHeatmapLocally;
        Debug.Log($"[LocalHeatmapOverride] HideHeatmapLocally = {hideHeatmapLocally} (obj={gameObject.name}, id={GetInstanceID()})");
    }

    public void SetHidden(bool hidden)
    {
        hideHeatmapLocally = hidden;
        Debug.Log($"[LocalHeatmapOverride] HideHeatmapLocally = {hideHeatmapLocally} (obj={gameObject.name}, id={GetInstanceID()})");
    }
}
