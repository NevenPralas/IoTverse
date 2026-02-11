using System.Collections;
using TMPro;
using UnityEngine;

public class AimOnGrip : MonoBehaviour
{
    [Header("Debug")]
    [SerializeField] private bool debugLogs = true;
    [SerializeField] private bool debugRaycastWhileGripping = true;
    [SerializeField] private bool debugTextEveryUpdate = false;

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

    [Header("UI Text (TMP)")]
    public GameObject text;

    private Vector3 velocity;
    private bool isVisible = false;
    private bool flashing = false;

    private SharedAimCanvasState sharedState;

    // Debug state anti-spam
    private bool _lastGripHeld = false;
    private float _lastGripValue = -999f;
    private float _lastIndexValue = -999f;
    private OVRInput.Controller _lastActive = OVRInput.Controller.None;

    private bool _lastHit = false;
    private string _lastHitName = "";
    private int _lastHitLayer = -999;

    private float _textProbeTimer = 0f;

    private void Log(string tag, string msg)
    {
        if (!debugLogs) return;
        Debug.Log($"{tag} {msg}");
    }

    void Start()
    {
        Log("[AIM]", "Start()");

        // aimObject init
        if (aimObject != null)
        {
            aimObject.SetActive(true);
            aimObject.transform.localScale = hiddenScale;
            Log("[AIM]", $"Aim object OK: '{aimObject.name}'");

            if (defaultAimMaterial == null)
            {
                Renderer r = aimObject.GetComponent<Renderer>();
                if (r != null)
                {
                    defaultAimMaterial = r.material;
                    Log("[AIM]", "Cached defaultAimMaterial from aimObject renderer.");
                }
                else
                {
                    Log("[AIM]", "WARNING: aimObject has no Renderer, can't cache defaultAimMaterial.");
                }
            }
        }
        else
        {
            Debug.LogError("[AIM] ERROR: AimObject is NULL!");
        }

        // rightHandAnchor auto-find
        if (rightHandAnchor == null)
        {
            var r = GameObject.Find("RightHandAnchor");
            if (r != null)
            {
                rightHandAnchor = r.transform;
                Log("[AIM]", $"RightHandAnchor auto-found. Path='{GetTransformPath(rightHandAnchor)}'");
            }
            else
            {
                Debug.LogError("[AIM] ERROR: RightHandAnchor NOT found! (GameObject named 'RightHandAnchor' missing)");
            }
        }
        else
        {
            Log("[AIM]", $"RightHandAnchor set via inspector. Path='{GetTransformPath(rightHandAnchor)}'");
        }

        // laser material cache
        if (laserLine != null && defaultLaserMaterial == null)
        {
            defaultLaserMaterial = laserLine.material;
            Log("[AIM]", "Cached defaultLaserMaterial from laserLine.");
        }

        // Shared state find
        sharedState = FindObjectOfType<SharedAimCanvasState>(true);
        if (sharedState != null)
            Log("[AIM]", $"SharedAimCanvasState found: '{sharedState.name}' activeInHierarchy={sharedState.gameObject.activeInHierarchy}");
        else
            Log("[AIM]", "WARNING: SharedAimCanvasState NOT found on Start().");

        // Text probe
        ProbeText("[TEXT]");
    }

    void Update()
    {
        // periodically probe text ref (AR issue)
        _textProbeTimer += Time.deltaTime;
        if (_textProbeTimer >= 1f)
        {
            _textProbeTimer = 0f;
            ProbeText("[TEXT]");
        }

        // Ensure sharedState exists
        if (sharedState == null)
        {
            sharedState = FindObjectOfType<SharedAimCanvasState>(true);
            if (sharedState != null)
                Log("[AIM]", "SharedAimCanvasState found during Update().");
        }

        // Input
        var active = OVRInput.GetActiveController();
        float gripValue = OVRInput.Get(OVRInput.Axis1D.SecondaryHandTrigger);
        float indexValue = OVRInput.Get(OVRInput.Axis1D.SecondaryIndexTrigger);
        bool gripHeld = gripValue > 0.12f;
        bool aDown = OVRInput.GetDown(OVRInput.Button.One);

        // Log input only on change
        if (active != _lastActive ||
            Mathf.Abs(gripValue - _lastGripValue) > 0.05f ||
            Mathf.Abs(indexValue - _lastIndexValue) > 0.05f ||
            gripHeld != _lastGripHeld)
        {
            Log("[AIM]", $"Input: Active={active} | Grip={gripValue:F2} held={gripHeld} | Trigger={indexValue:F2}");
            _lastActive = active;
            _lastGripValue = gripValue;
            _lastIndexValue = indexValue;
            _lastGripHeld = gripHeld;
        }

        // Raycast
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
            }
        }

        // Raycast log on change
        if (debugLogs && debugRaycastWhileGripping && gripHeld && rightHandAnchor != null)
        {
            if (hitEnvironment != _lastHit ||
                (hitEnvironment && hitInfo.collider != null &&
                 (hitInfo.collider.name != _lastHitName || hitInfo.collider.gameObject.layer != _lastHitLayer)))
            {
                if (hitEnvironment && hitInfo.collider != null)
                {
                    Log("[AIM]", $"Raycast HIT: '{hitInfo.collider.name}' layer={hitInfo.collider.gameObject.layer} point={hitInfo.point}");
                    _lastHitName = hitInfo.collider.name;
                    _lastHitLayer = hitInfo.collider.gameObject.layer;
                }
                else
                {
                    Log("[AIM]", $"Raycast MISS: origin={rightHandAnchor.position} forward={rightHandAnchor.forward} mask={raycastMask.value} maxDist={raycastMaxDistance}");
                    _lastHitName = "";
                    _lastHitLayer = -999;
                }
                _lastHit = hitEnvironment;
            }
        }

        bool shouldShow = gripHeld && hitEnvironment;

        if (shouldShow && !isVisible)
        {
            isVisible = true;
            Log("[AIM]", "Aim VISIBLE");
            if (useHaptics) StartCoroutine(PulseHaptics(hapticStrength, hapticDuration));
        }
        else if (!shouldShow && isVisible)
        {
            isVisible = false;
            Log("[AIM]", "Aim HIDDEN");
        }

        // Click A while gripping
        if (gripHeld && aDown)
        {
            Log("[AIM]", "A pressed while gripping -> Flash + send point (if hit).");
            StartCoroutine(FlashMaterials());

            if (hitEnvironment)
            {
                Log("[AIM]", $"Sending point to SharedAimCanvasState: {hitInfo.point}");
                if (sharedState != null)
                    sharedState.RequestSetPoint(hitInfo.point);
                else
                    Debug.LogError("[AIM] ERROR: SharedAimCanvasState missing when trying to send point!");
            }
            else
            {
                Log("[AIM]", "A pressed but NO surface hit.");
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

        if (debugTextEveryUpdate)
        {
            // optional: spammy, but good for AR debug
            ProbeText("[TEXT]");
        }
    }

    private void ProbeText(string tag)
    {
        if (text == null)
        {
            Log(tag, "text GameObject is NULL (no reference).");
            return;
        }

        var tmp = text.GetComponent<TextMeshProUGUI>();
        if (tmp == null)
        {
            Log(tag, $"text GameObject='{text.name}' has NO TextMeshProUGUI component.");
            return;
        }

        Log(tag, $"text OK: go='{text.name}' enabled={tmp.enabled} activeInHierarchy={text.activeInHierarchy} current='{tmp.text}'");
    }

    private IEnumerator PulseHaptics(float s, float d)
    {
        Log("[AIM]", $"Haptics: strength={s:F2} duration={d:F2}");
        OVRInput.SetControllerVibration(1f, s, OVRInput.Controller.RTouch);
        yield return new WaitForSeconds(d);
        OVRInput.SetControllerVibration(0f, 0f, OVRInput.Controller.RTouch);
    }

    private IEnumerator FlashMaterials()
    {
        if (flashing)
        {
            Log("[AIM]", "Flash skipped (already flashing).");
            yield break;
        }

        flashing = true;
        Log("[AIM]", "Flash START");

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
        Log("[AIM]", "Flash END");
    }

    public void UpdateTemperatureDisplay(float temperature)
    {
        Log("[TEXT]", $"UpdateTemperatureDisplay({temperature:F2}) called.");

        if (text == null)
        {
            Log("[TEXT]", "WARNING: text reference is NULL -> cannot display temperature.");
            return;
        }

        var tmp = text.GetComponent<TextMeshProUGUI>();
        if (tmp == null)
        {
            Log("[TEXT]", $"WARNING: text='{text.name}' has no TextMeshProUGUI component.");
            return;
        }

        tmp.text = $"Temperature = {temperature:F2}°C";
        Log("[TEXT]", $"TMP updated OK -> '{tmp.text}' (enabled={tmp.enabled}, active={text.activeInHierarchy})");
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
