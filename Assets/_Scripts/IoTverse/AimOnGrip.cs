using System.Collections;
using TMPro;
using UnityEngine;

public class AimOnGrip : MonoBehaviour
{
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

    // UI text reference (ostaje)
    public GameObject text;

    private SharedAimCanvasState sharedState;

    void Start()
    {
        if (aimObject != null)
        {
            aimObject.SetActive(true);
            aimObject.transform.localScale = hiddenScale;
            if (defaultAimMaterial == null)
            {
                Renderer r = aimObject.GetComponent<Renderer>();
                if (r != null) defaultAimMaterial = r.material;
            }
        }

        if (rightHandAnchor == null)
        {
            var r = GameObject.Find("RightHandAnchor");
            if (r != null) rightHandAnchor = r.transform;
        }

        if (laserLine != null && defaultLaserMaterial == null)
            defaultLaserMaterial = laserLine.material;

        sharedState = FindObjectOfType<SharedAimCanvasState>(true);
    }

    void Update()
    {
        if (sharedState == null)
            sharedState = FindObjectOfType<SharedAimCanvasState>(true);

        float gripValue = OVRInput.Get(OVRInput.Axis1D.SecondaryHandTrigger);
        bool gripHeld = gripValue > 0.12f;

        bool hitEnvironment = false;
        Vector3 targetPos = Vector3.zero;
        Quaternion targetRot = Quaternion.identity;
        RaycastHit hitInfo = new RaycastHit();

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

        // Klik (A) dok držiš grip: flash + NETWORK update shared canvasa
        if (gripHeld && OVRInput.GetDown(OVRInput.Button.One))
        {
            StartCoroutine(FlashMaterials());
            if (hitEnvironment)
            {
                if (sharedState != null)
                    sharedState.RequestSetPoint(hitInfo.point);
                else
                    Debug.LogError("[AimOnGrip] SharedAimCanvasState nije pronađen (nije spawnan?).");
            }
        }

        bool shouldShow = gripHeld && hitEnvironment;
        if (shouldShow && !isVisible)
        {
            isVisible = true;
            if (useHaptics) StartCoroutine(PulseHaptics(hapticStrength, hapticDuration));
        }
        else if (!shouldShow && isVisible)
        {
            isVisible = false;
        }

        if (aimObject != null)
        {
            if (smoothMovement)
            {
                Vector3 newPos = shouldShow ? targetPos : aimObject.transform.position;
                Quaternion newRot = shouldShow ? targetRot : aimObject.transform.rotation;
                aimObject.transform.position = Vector3.SmoothDamp(aimObject.transform.position, newPos, ref velocity, positionSmoothTime);
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
        OVRInput.SetControllerVibration(1f, s, OVRInput.Controller.RTouch);
        yield return new WaitForSeconds(d);
        OVRInput.SetControllerVibration(0f, 0f, OVRInput.Controller.RTouch);
    }

    private IEnumerator FlashMaterials()
    {
        if (flashing) yield break;
        flashing = true;

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
    }

    // koristi SharedAimCanvasState za update teksta (lokalno)
    public void UpdateTemperatureDisplay(float temperature)
    {
        if (text != null)
        {
            text.GetComponent<TextMeshProUGUI>().text = $"Temperature = {temperature:F2}°C";
        }
    }
}
