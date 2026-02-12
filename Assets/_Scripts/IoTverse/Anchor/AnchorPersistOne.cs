using UnityEngine;

public class AnchorPersistOne : MonoBehaviour
{
    [SerializeField] private Transform pivot;
    [SerializeField] private bool snapPivotToAnchor = true;
    [SerializeField] private bool setScaleToOne = true;

    private const string PlayerPrefsKey = "IOTVERSE_SINGLE_ANCHOR_UUID";

    // Spoji u BuildingBlock event:
    // On Anchor Create Completed (OVRSpatialAnchor, OperationResult)
    public void OnAnchorCreateCompleted(OVRSpatialAnchor createdAnchor, OVRSpatialAnchor.OperationResult result)
    {
        if (createdAnchor == null)
        {
            Debug.LogError("[AnchorPersistOne] createdAnchor is null.");
            return;
        }

        if (result != OVRSpatialAnchor.OperationResult.Success)
        {
            Debug.LogWarning($"[AnchorPersistOne] Anchor create failed: {result}");
            return;
        }

        AttachPivot(createdAnchor);

        // SAVE + store UUID (async)
        SaveAndStoreAsync(createdAnchor);
    }

    private void AttachPivot(OVRSpatialAnchor anchor)
    {
        if (pivot == null)
        {
            Debug.LogError("[AnchorPersistOne] Pivot nije postavljen.");
            return;
        }

        pivot.SetParent(anchor.transform, false);

        if (snapPivotToAnchor)
        {
            pivot.localPosition = Vector3.zero;
            pivot.localRotation = Quaternion.identity;
            if (setScaleToOne) pivot.localScale = Vector3.one;
        }
    }

    private async void SaveAndStoreAsync(OVRSpatialAnchor anchor)
    {
        // 1) UUID (ovo će ti možda biti drugačije po verziji; ako pukne, javi error)
        string uuidString = anchor.Uuid.ToString();

        // 2) Save (ovo mora postojati u tvojoj verziji, inače ćeš dobiti compile error)
        try
        {
            var saveResult = await anchor.SaveAnchorAsync();
            Debug.Log($"[AnchorPersistOne] Save completed: {saveResult}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[AnchorPersistOne] Save failed: {e}");
            // svejedno spremimo UUID lokalno (ali load možda neće radit)
        }

        PlayerPrefs.SetString(PlayerPrefsKey, uuidString);
        PlayerPrefs.Save();

        Debug.Log($"[AnchorPersistOne] Stored UUID = {uuidString}");
    }
}
