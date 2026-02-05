using System.Collections;
using UnityEngine;

public class AnchorAttachToLast : MonoBehaviour
{
    [Header("Scene object you want to attach to anchor")]
    [SerializeField] private Transform objectToAttach; // npr. Pivot

    [Header("Timing")]
    [SerializeField] private float retryEverySeconds = 0.2f;
    [SerializeField] private float giveUpAfterSeconds = 10f;

    [Header("Attach behavior")]
    [Tooltip("Ako je true: objekt ostaje gdje je u svijetu, samo postane child anchora.")]
    [SerializeField] private bool keepWorldPosition = true;

    private Transform _lastAttachedAnchor;

    // Pozovi ovo iz UI/button eventa ili direktno nakon A, ali i ručno možeš.
    public void AttachToNewestAnchor()
    {
        StopAllCoroutines();
        StartCoroutine(AttachRoutine());
    }

    private IEnumerator AttachRoutine()
    {
        if (objectToAttach == null)
        {
            Debug.LogError("[AnchorAttachToLast] objectToAttach nije postavljen.");
            yield break;
        }

        float t = 0f;
        while (t < giveUpAfterSeconds)
        {
            // nađi sve anchore u sceni, uzmi "najnoviji" (zadnji u listi)
            var anchors = FindObjectsOfType<OVRSpatialAnchor>(true);
            if (anchors != null && anchors.Length > 0)
            {
                var newest = anchors[anchors.Length - 1].transform;

                // ako smo već attachali na taj anchor, nema potrebe opet
                if (_lastAttachedAnchor == newest)
                    yield break;

                objectToAttach.SetParent(newest, keepWorldPosition);
                _lastAttachedAnchor = newest;

                Debug.Log($"[AnchorAttachToLast] '{objectToAttach.name}' attached to anchor '{newest.name}'.");
                yield break;
            }

            t += retryEverySeconds;
            yield return new WaitForSeconds(retryEverySeconds);
        }

        Debug.LogError("[AnchorAttachToLast] Nisam našao nijedan OVRSpatialAnchor u sceni u zadanom roku.");
    }
}
