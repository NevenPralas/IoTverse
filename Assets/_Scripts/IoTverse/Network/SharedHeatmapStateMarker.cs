using Fusion;
using UnityEngine;

public class SharedHeatmapStateMarker : NetworkBehaviour
{
    [Header("Scene floor lookup")]
    [SerializeField] private string floorName = "Floor";

    [Header("Default state (StateAuthority only)")]
    [SerializeField] private bool defaultHeatmapOn = true;

    [Networked] public NetworkBool HeatmapOn { get; set; }

    private HeatMapStaticWithJson hm;
    private LocalHeatmapOverride localOverride;

    private bool _lastAppliedFinalShow;

    public override void Spawned()
    {
        var floor = GameObject.Find(floorName);
        if (floor == null)
        {
            Debug.LogError($"[SharedHeatmapStateMarker] Ne mogu naći scene floor '{floorName}'.");
            return;
        }

        hm = floor.GetComponentInChildren<HeatMapStaticWithJson>(true);
        if (hm == null)
        {
            Debug.LogError("[SharedHeatmapStateMarker] Ne mogu naći HeatMapStaticWithJson na Floor-u ili childovima.");
            return;
        }

        localOverride = hm.GetComponent<LocalHeatmapOverride>();
        if (localOverride == null)
            localOverride = hm.gameObject.AddComponent<LocalHeatmapOverride>();

        if (Object.HasStateAuthority)
            HeatmapOn = defaultHeatmapOn;

        // Force apply jednom
        _lastAppliedFinalShow = !ComputeFinalShow();
        Apply();
    }

    public override void Render() => Apply();

    private bool ComputeFinalShow()
    {
        bool localHidden = localOverride != null && localOverride.HideHeatmapLocally;
        return HeatmapOn && !localHidden;
    }

    private void Apply()
    {
        if (hm == null) return;

        bool finalShow = ComputeFinalShow();
        if (finalShow == _lastAppliedFinalShow) return;

        _lastAppliedFinalShow = finalShow;
        hm.SetHeatmapEnabled(finalShow);
    }
}
