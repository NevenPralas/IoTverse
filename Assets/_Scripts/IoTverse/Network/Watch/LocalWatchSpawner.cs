using System.Collections;
using TMPro;
using UnityEngine;

/// <summary>
/// Lokalno spawnanje sata za SVAKOG igrača (host i join).
/// Ne koristi Fusion spawn. Samo Instantiate na lokalnom avataru.
/// </summary>
public class LocalWatchSpawner : MonoBehaviour
{
    [Header("Watch prefab (local only)")]
    [SerializeField] private GameObject watchPrefab;

    [Header("Attach to this joint name on the LOCAL avatar")]
    [SerializeField] private string wristJointName = "Joint RightHandWrist";

    [Header("Local pose offset")]
    [SerializeField] private Vector3 localPosition = new Vector3(0.0049f, 0.0208f, 0.0042f);
    [SerializeField] private Vector3 localEulerRotation = new Vector3(-165.493f, 161.189f, -2.58099f);

    [Header("Optional UI text to enable (local)")]
    [SerializeField] private TextMeshProUGUI text1;
    [SerializeField] private TextMeshProUGUI text2;
    [SerializeField] private TextMeshProUGUI text3;
    [SerializeField] private TextMeshProUGUI text4;

    private bool _spawned;

    private void Start()
    {
        StartCoroutine(SpawnWhenReady());
    }

    private IEnumerator SpawnWhenReady()
    {
        // Čekaj dok se ne pojavi zglob u local avatar hijerarhiji
        while (!_spawned)
        {
            var parent = GameObject.Find(wristJointName)?.transform;

            if (parent != null)
            {
                SpawnWatch(parent);
                yield break;
            }

            yield return null;
        }
    }

    private void SpawnWatch(Transform parent)
    {
        if (_spawned) return;

        if (watchPrefab == null)
        {
            Debug.LogError("[LocalWatchSpawner] watchPrefab nije postavljen!");
            return;
        }

        // Ako već postoji (npr. restart), ne radi duplikat
        if (GameObject.FindWithTag("HeatController") != null)
        {
            Debug.Log("[LocalWatchSpawner] HeatController već postoji, ne spawnam duplikat.");
            _spawned = true;
            return;
        }

        var instance = Instantiate(watchPrefab, parent);
        instance.transform.localPosition = localPosition;
        instance.transform.localRotation = Quaternion.Euler(localEulerRotation);

        if (text1) text1.enabled = true;
        if (text2) text2.enabled = true;
        if (text3) text3.enabled = true;
        if (text4) text4.enabled = true;

        _spawned = true;
        Debug.Log("[LocalWatchSpawner] Watch spawned locally.");
    }
}
