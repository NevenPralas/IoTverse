using System.Collections;
using TMPro;
using UnityEngine;

public class AimOnGrip : MonoBehaviour
{
    [Header("Debug")]
    [SerializeField] private bool debugLogs = true;

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
                    Log("Default aim material cached.");
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
                Log("RightHandAnchor auto-found.");
            }
            else
            {
                Debug.LogError("[AimOnGrip] RightHandAnchor NOT found!");
            }
        }

        if (laserLine != null && defaultLaserMaterial == null)
        {
            defaultLaserMaterial = laserLine.material;
            Log("Default laser material cached.");
        }

        sharedState = FindObjectOfType<SharedAimCanvasState>(true);

        if (sharedState != null)
            Log("SharedAimCanvasState found.");
        else
            Debug.LogWarning("[AimOnGrip] SharedAimCanvasState NOT found on start.");
    }

    void Update()
    {
        if (sharedState == null)
        {
            sharedState = FindObjectOfType<SharedAimCanvasState>(true);
            if (sharedState != null)
                Log("SharedAimCanvasState found during Update.");
        }

        float gripValue = OVRInput.Get(OVRInput.Axis1D.SecondaryHandTrigger);
        bool gripHeld = gripValue > 0.12f;

        Vector3 targetPos = Vector3.zero;
        Quaternion targetRot = Quaternion.identity;
        RaycastHit hitInfo = new RaycastHit();

        bool hitEnvironment = false;

        if (gripHeld && rightHandAnchor != null)
        {
            Ray ray = new Ray(rightHandAnchor.position, rightHandAnchor.forward);

            if (Physics.Raycast(ray, out hitInfo, raycastMaxDistance, raycastMask))
            {
                hitEnvironment = true;
                targetPos = hitInfo.point + hitInfo.normal * 0.01f;
                targetRot = Quaternion.LookRotation(-hitInfo.normal);
            }
        }

        // SHOW / HIDE logs (bez frame spamma)
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

        // CLICK
        if (gripHeld && OVRInput.GetDown(OVRInput.Button.One))
        {
            Log("A button pressed while gripping.");

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
                Log("A pressed but NO surface hit.");
            }
        }

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

            aimObject.transform.localScale = Vector3.Lerp(
                aimObject.transform.localScale,
                shouldShow ? shownScale : hiddenScale,
                Time.deltaTime * scaleShowSpeed
            );
        }

        if (laserLine != null)
        {
            laserLine.enabled = shouldShow;

            if (shouldShow)
            {
                laserLine.SetPosition(0, rightHandAnchor.position);
                laserLine.SetPosition(1, targetPos);
            }
        }
    }

    private IEnumerator PulseHaptics(float s, float d)
    {
        Log("Haptics pulse triggered.");

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
            text.GetComponent<TextMeshProUGUI>().text =
                $"Temperature = {temperature:F2}°C";
        }
        else
        {
            Debug.LogWarning("[AimOnGrip] Text object is NULL!");
        }
    }
}
