using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class AnchorAutoAttach : MonoBehaviour
{
    [SerializeField] private Transform pivot;

    [Header("Snap")]
    [SerializeField] private bool snapToAnchor = true;
    [SerializeField] private bool setScaleToOne = true;

    [Header("Delete old (keep only 1 anchor)")]
    [SerializeField] private bool deletePreviousAnchor = true;
    [SerializeField] private float waitBeforeDeleteOldSeconds = 0.2f;

    private OVRSpatialAnchor _currentAnchor;

    // Spoji u Spatial Anchor Core Building Block event:
    // On Anchor Create Completed (OVRSpatialAnchor, OperationResult)
    public void OnAnchorCreateCompleted(OVRSpatialAnchor createdAnchor, OVRSpatialAnchor.OperationResult result)
    {
        if (createdAnchor == null)
        {
            Debug.LogError("[AnchorAutoAttach] createdAnchor is null.");
            return;
        }

        if (result != OVRSpatialAnchor.OperationResult.Success)
        {
            Debug.LogWarning($"[AnchorAutoAttach] Anchor create failed: {result}");
            return;
        }

        var old = _currentAnchor;

        AttachPivot(createdAnchor);
        _currentAnchor = createdAnchor;

        if (deletePreviousAnchor && old != null && old != createdAnchor)
        {
            // ne coroutine s task.Status/Result, nego async brisanje
            StartCoroutine(DeleteOldAfterDelay(old));
        }
    }

    private void AttachPivot(OVRSpatialAnchor anchor)
    {
        if (pivot == null)
        {
            Debug.LogError("[AnchorAutoAttach] Pivot nije postavljen.");
            return;
        }

        pivot.SetParent(anchor.transform, false);

        if (snapToAnchor)
        {
            pivot.localPosition = Vector3.zero;
            pivot.localRotation = Quaternion.identity;
            if (setScaleToOne) pivot.localScale = Vector3.one;
        }

        Debug.Log($"[AnchorAutoAttach] Pivot attached to '{anchor.name}'.");
    }

    private IEnumerator DeleteOldAfterDelay(OVRSpatialAnchor old)
    {
        if (waitBeforeDeleteOldSeconds > 0f)
            yield return new WaitForSeconds(waitBeforeDeleteOldSeconds);

        // pozovi async delete bez Task.Status/Result
        DeleteOldAnchorAsync(old);
    }

    private async void DeleteOldAnchorAsync(OVRSpatialAnchor old)
    {
        if (old == null) return;

        Debug.Log($"[AnchorAutoAttach] Deleting old anchor '{old.name}'...");

        try
        {
            // U tvojoj verziji ovo vraća OVRTask<OVRResult<OVRAnchor.EraseResult>>
            var eraseResult = await old.EraseAnchorAsync();

            // OVRResult obično ima .IsSuccess (ovisno o verziji). Ako nema, samo logaj eraseResult.
            // Sigurno loganje bez oslanjanja na polja koja možda ne postoje:
            Debug.Log($"[AnchorAutoAttach] Erase completed: {eraseResult}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[AnchorAutoAttach] Erase failed with exception: {e}");
        }

        // makni GO iz scene (lokalno)
        if (old != null && old.gameObject != null)
            Destroy(old.gameObject);
    }

    public void OnAnchorsLoadCompleted(List<OVRSpatialAnchor> anchors)
    {
        if (anchors == null || anchors.Count == 0)
        {
            Debug.LogWarning("[AnchorAutoAttach] Load completed but no anchors returned.");
            return;
        }

        // uzmi zadnji (ili prvi, kako želiš)
        var a = anchors[anchors.Count - 1];
        AttachPivot(a);
        _currentAnchor = a;
    }
}
