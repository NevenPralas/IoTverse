using System.Collections;
using TMPro;
using UnityEngine;

public class AimOnGrip : MonoBehaviour
{
    [Header("Debug")]
    [SerializeField] private bool debugLogs = true;
    [SerializeField] private bool debugRaycastWhileGripping = true; // ako te spam-a, stavi false

    [Header("References")]
    [SerializeField] private GameObject aimObject;
    [SerializeField] private Transform rightHandAnchor;

    [Header("Behavior")]
    [SerializeField] private float defaultDistance = 0.45f;
    [SerializeField] private bool useRaycast = true;
    [SerializeField] private LayerMask raycastMask = ~0;
    [SerializeField] private float raycastMaxDistance = 10f;

    [Header("Smoothing & Visuals")]
    [SerializeField] private bool smoothMovement = true;
    [SerializeField] private float positionSmoothTime = 0.06f;
    [SerializeField] private float rotationSmoothTime = 0.06f;
    [SerializeField] private bool scaleOnShow = true;
    [SerializeField] private float scaleShowSpeed = 12f;
    [SerializeField] private Vector3 hiddenScale = Vector3.zero;
    [SerializeField] private Vector3 shownScale = new Vector3(0.03f, 0.03f, 0.03f);

    [Header("Haptics (optional)")]
    [SerializeField] private bool useHaptics = true;
    [SerializeField] private float hapticDuration = 0.05f;
    [SerializeField] private float hapticStrength = 0.5f;

    [Header("Laser Beam")]
    [SerializeField] private LineRenderer laserLine;
    [SerializeField] private float laserMaxLength = 10f;

    [Header("Material Flash")]
    public Material flashMaterial;
    public Material defaultLaserMaterial;
    public Material defaultAimMaterial;
    public float flashDuration = 1f;

    private Vector3 velocity;
    private bool isVisible = false;
    private bool flashing = false;

    public GameObject text;
    private SharedAimCanvasState sharedState;

    // --- Debug state (anti-spam) ---
    private bool _lastGripHeld = false;
    private float _lastGripValue = -999f;
    private float _lastIndexValue = -999f;
    private bool _lastA = false;
    private OVRInput.Controller _lastActive = OVRInput.Controller.None;

    private bool _lastHit = false;
    private string _lastHitName = "";
    private int _lastHitLayer = -999;

    void Log(string msg)
    {
        if (debugLogs)
            Debug.Log("[AimOnGrip] " + msg);
    }

    void Start()
    {
        Log("Start() called.");

        if (aimObject != null)
        {
            aimObject.SetActive(true);
            aimObject.transform.localScale = hiddenScale;
            Log("Aim object initialized.");

            if (defaultAimMaterial == null)
            {
                Renderer r = aimObject.GetComponent<Renderer>();
                if (r != null)
                {
                    defaultAimMaterial = r.material;
                    Log("Default aim material cached from aimObject renderer.");
                }
                else
                {
                    Log("WARNING: aimObject has no Renderer, can't cache defaultAimMaterial.");
                }
            }
        }
        else
        {
            Debug.LogError("[AimOnGrip] AimObject is NULL!");
        }

        if (rightHandAnchor == null)
        {
            var r = GameObject.Find("RightHandAnchor");
            if (r != null)
            {
                rightHandAnchor = r.transform;
                Log("RightHandAnchor auto-found via GameObject.Find('RightHandAnchor').");
            }
            else
            {
                Debug.LogError("[AimOnGrip] RightHandAnchor NOT found! (GameObject named 'RightHandAnchor' missing)");
            }
        }

        if (rightHandAnchor != null)
        {
            Log($"RightHandAnchor OK: name='{rightHandAnchor.name}' path='{GetTransformPath(rightHandAnchor)}'");
        }

        if (laserLine != null && defaultLaserMaterial == null)
        {
            defaultLaserMaterial = laserLine.material;
            Log("Default laser material cached from laserLine.");
        }

        sharedState = FindObjectOfType<SharedAimCanvasState>(true);
        if (sharedState != null)
            Log("SharedAimCanvasState found in scene.");
        else
            Log("WARNING: SharedAimCanvasState NOT found on Start().");
    }

    void Update()
    {
        // --- Ensure sharedState exists ---
        if (sharedState == null)
        {
            sharedState = FindObjectOfType<SharedAimCanvasState>(true);
            if (sharedState != null)
                Log("SharedAimCanvasState found during Update().");
        }

        // --- Input ---
        var active = OVRInput.GetActiveController();

        float gripValue = OVRInput.Get(OVRInput.Axis1D.SecondaryHandTrigger);   // "grip" (hand trigger) na Touch kontrolerima
        float indexValue = OVRInput.Get(OVRInput.Axis1D.SecondaryIndexTrigger); // "trigger" (index trigger)

        bool gripHeld = gripValue > 0.12f;

        bool aDown = OVRInput.GetDown(OVRInput.Button.One);

        // Log input only on change (anti-spam)
        if (active != _lastActive ||
            Mathf.Abs(gripValue - _lastGripValue) > 0.05f ||
            Mathf.Abs(indexValue - _lastIndexValue) > 0.05f ||
            gripHeld != _lastGripHeld)
        {
            Log($"Input: ActiveController={active} | SecondaryHandTrigger(grip?)={gripValue:F2} held={gripHeld} | SecondaryIndexTrigger(trigger)={indexValue:F2}");
            _lastActive = active;
            _lastGripValue = gripValue;
            _lastIndexValue = indexValue;
            _lastGripHeld = gripHeld;
        }

        // --- Raycast + targets ---
        Vector3 targetPos = Vector3.zero;
        Quaternion targetRot = Quaternion.identity;
        RaycastHit hitInfo = new RaycastHit();
        bool hitEnvironment = false;

        if (gripHeld && rightHandAnchor != null)
        {
            Ray ray = new Ray(rightHandAnchor.position, rightHandAnchor.forward);

            if (useRaycast)
            {
                if (Physics.Raycast(ray, out hitInfo, raycastMaxDistance, raycastMask))
                {
                    hitEnvironment = true;
                    targetPos = hitInfo.point + hitInfo.normal * 0.01f;
                    targetRot = Quaternion.LookRotation(-hitInfo.normal);
                }
                else
                {
                    // fallback (ako želiš): možeš koristiti defaultDistance, ali zadržavam istu funkcionalnost kao prije
                    // targetPos = rightHandAnchor.position + rightHandAnchor.forward * defaultDistance;
                    // targetRot = rightHandAnchor.rotation;
                }
            }
        }

        // Raycast log: samo kad se promijeni hit/miss ili objekt koji pogađa
        if (debugLogs && debugRaycastWhileGripping && gripHeld && rightHandAnchor != null)
        {
            if (hitEnvironment != _lastHit ||
                (hitEnvironment && (hitInfo.collider != null) &&
                 (hitInfo.collider.name != _lastHitName || hitInfo.collider.gameObject.layer != _lastHitLayer)))
            {
                if (hitEnvironment && hitInfo.collider != null)
                {
                    Log($"Raycast HIT: '{hitInfo.collider.name}' layer={hitInfo.collider.gameObject.layer} point={hitInfo.point} normal={hitInfo.normal}");
                    _lastHitName = hitInfo.collider.name;
                    _lastHitLayer = hitInfo.collider.gameObject.layer;
                }
                else
                {
                    Log($"Raycast MISS: origin={rightHandAnchor.position} forward={rightHandAnchor.forward} mask={raycastMask.value} maxDist={raycastMaxDistance}");
                    _lastHitName = "";
                    _lastHitLayer = -999;
                }

                _lastHit = hitEnvironment;
            }
        }

        // SHOW / HIDE (ista logika kao prije: mora biti gripHeld AND hitEnvironment)
        bool shouldShow = gripHeld && hitEnvironment;

        if (shouldShow && !isVisible)
        {
            isVisible = true;
            Log("Aim became VISIBLE.");

            if (useHaptics)
                StartCoroutine(PulseHaptics(hapticStrength, hapticDuration));
        }
        else if (!shouldShow && isVisible)
        {
            isVisible = false;
            Log("Aim became HIDDEN.");
        }

        // CLICK (A while gripping)
        if (gripHeld && aDown)
        {
            Log("A button pressed while gripping -> Flash + (if hit) send point.");

            StartCoroutine(FlashMaterials());

            if (hitEnvironment)
            {
                Log("Raycast hit -> sending point to SharedAimCanvasState.");

                if (sharedState != null)
                    sharedState.RequestSetPoint(hitInfo.point);
                else
                    Debug.LogError("[AimOnGrip] SharedAimCanvasState missing when trying to send point!");
            }
            else
            {
                Log("A pressed but NO surface hit (hitEnvironment=false).");
            }
        }

        // Move/scale aimObject
        if (aimObject != null)
        {
            if (smoothMovement)
            {
                Vector3 newPos = shouldShow ? targetPos : aimObject.transform.position;
                Quaternion newRot = shouldShow ? targetRot : aimObject.transform.rotation;

                aimObject.transform.position = Vector3.SmoothDamp(
                    aimObject.transform.position,
                    newPos,
                    ref velocity,
                    positionSmoothTime
                );

                aimObject.transform.rotation = Quaternion.Slerp(
                    aimObject.transform.rotation,
                    newRot,
                    Time.deltaTime / Mathf.Max(0.0001f, rotationSmoothTime)
                );
            }
            else if (shouldShow)
            {
                aimObject.transform.position = targetPos;
                aimObject.transform.rotation = targetRot;
            }

            if (scaleOnShow)
            {
                aimObject.transform.localScale = Vector3.Lerp(
                    aimObject.transform.localScale,
                    shouldShow ? shownScale : hiddenScale,
                    Time.deltaTime * scaleShowSpeed
                );
            }
        }

        // Laser
        if (laserLine != null)
        {
            laserLine.enabled = shouldShow;

            if (shouldShow && rightHandAnchor != null)
            {
                laserLine.SetPosition(0, rightHandAnchor.position);
                laserLine.SetPosition(1, targetPos);
            }
        }
    }

    private IEnumerator PulseHaptics(float s, float d)
    {
        Log($"Haptics pulse triggered. strength={s:F2} duration={d:F2}");

        OVRInput.SetControllerVibration(1f, s, OVRInput.Controller.RTouch);
        yield return new WaitForSeconds(d);
        OVRInput.SetControllerVibration(0f, 0f, OVRInput.Controller.RTouch);
    }

    private IEnumerator FlashMaterials()
    {
        if (flashing)
        {
            Log("Flash skipped (already flashing).");
            yield break;
        }

        flashing = true;
        Log("Flash started.");

        if (laserLine) laserLine.material = flashMaterial;

        if (aimObject)
        {
            Renderer r = aimObject.GetComponent<Renderer>();
            if (r != null) r.material = flashMaterial;
        }

        yield return new WaitForSeconds(flashDuration);

        if (laserLine) laserLine.material = defaultLaserMaterial;

        if (aimObject)
        {
            Renderer r = aimObject.GetComponent<Renderer>();
            if (r != null) r.material = defaultAimMaterial;
        }

        flashing = false;
        Log("Flash ended.");
    }

    public void UpdateTemperatureDisplay(float temperature)
    {
        Log($"Temperature updated: {temperature:F2}°C");

        if (text != null)
        {
            var tmp = text.GetComponent<TextMeshProUGUI>();
            if (tmp != null)
                tmp.text = $"Temperature = {temperature:F2}°C";
            else
                Log("WARNING: Text object exists but has no TextMeshProUGUI component.");
        }
        else
        {
            Log("WARNING: Text object is NULL!");
        }
    }

    private static string GetTransformPath(Transform t)
    {
        if (t == null) return "(null)";
        string path = t.name;
        while (t.parent != null)
        {
            t = t.parent;
            path = t.name + "/" + path;
        }
        return path;
    }
}
