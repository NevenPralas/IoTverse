using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.XR;

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
    private Quaternion rotVelocity = Quaternion.identity;
    private bool isVisible = false;
    private bool flashing = false;

    public GameObject text;
    public GameObject canvas;
    public GameObject portal;
    void Start()
    {
        if (aimObject == null)
        {
            Debug.LogWarning("AimOnGrip: aimObject nije postavljen.");
        }
        else
        {
            aimObject.SetActive(true);
            aimObject.transform.localScale = hiddenScale;
        }

        if (rightHandAnchor == null)
        {
            var r = GameObject.Find("RightHandAnchor");
            if (r != null) rightHandAnchor = r.transform;
        }

        if (laserLine != null && defaultLaserMaterial == null)
            defaultLaserMaterial = laserLine.material;

        if (aimObject != null && defaultAimMaterial == null)
        {
            Renderer r = aimObject.GetComponent<Renderer>();
            if (r != null)
                defaultAimMaterial = r.material;
        }
    }

    void Update()
    {
        float gripValue = OVRInput.Get(OVRInput.Axis1D.SecondaryHandTrigger);
        bool gripHeld = gripValue > 0.12f;

        // ---------------------------------------------------
        // RAYCAST
        // ---------------------------------------------------
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

        // ---------------------------------------------------
        // A BUTTON — tek SAD, nakon raycasta!
        // ---------------------------------------------------
        if (gripHeld && OVRInput.GetDown(OVRInput.Button.One))
        {
            StartCoroutine(FlashMaterials());

            if (hitEnvironment)
            {
                LogTemperatureAtHit(hitInfo);
            }
        }

        // ---------------------------------------------------
        // VISIBLE TOGGLE
        // ---------------------------------------------------
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

        // ---------------------------------------------------
        // MOVE AIM OBJECT
        // ---------------------------------------------------
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
                    positionSmoothTime);

                aimObject.transform.rotation = Quaternion.Slerp(
                    aimObject.transform.rotation,
                    newRot,
                    Time.deltaTime / Mathf.Max(0.0001f, rotationSmoothTime));
            }
            else
            {
                if (shouldShow)
                {
                    aimObject.transform.position = targetPos;
                    aimObject.transform.rotation = targetRot;
                }
            }

            Vector3 targetScale = shouldShow ? shownScale : hiddenScale;
            aimObject.transform.localScale = Vector3.Lerp(
                aimObject.transform.localScale,
                targetScale,
                Time.deltaTime * scaleShowSpeed);
        }

        // ---------------------------------------------------
        // LASER
        // ---------------------------------------------------
        if (laserLine != null)
        {
            if (shouldShow && rightHandAnchor != null)
            {
                laserLine.enabled = true;
                laserLine.SetPosition(0, rightHandAnchor.position);
                laserLine.SetPosition(1, hitEnvironment ?
                    targetPos :
                    rightHandAnchor.position + rightHandAnchor.forward * laserMaxLength);
            }
            else
            {
                laserLine.enabled = false;
            }
        }
    }

    private void LogTemperatureAtHit(RaycastHit hit)
    {
        // Postavljanje Canvasa na mjesto klika
        float heightOffset = 1.2f; // koliko iznad pogotka
        Vector3 targetPos = hit.point + Vector3.up * heightOffset;
        canvas.transform.position = targetPos;

        Transform cam = Camera.main != null ? Camera.main.transform : null;

        if (cam != null)
        {
            Vector3 lookDir = cam.position - canvas.transform.position;
            lookDir.y = 0; // ostaje uspravan

            if (lookDir.sqrMagnitude > 0.001f)
            {
                // Okreni prema igracu
                canvas.transform.rotation = Quaternion.LookRotation(lookDir);

                // Ispravi zrcaljenje (dodaj 180°)
                canvas.transform.Rotate(0, 180f, 0, Space.Self);
            }
        }

        canvas.GetComponent<Canvas>().enabled = true;
        portal.SetActive(true);

        Vector2 uv = hit.textureCoord;

        HeatMapStatic hm = hit.collider.GetComponent<HeatMapStatic>();
        if (hm == null) hm = hit.collider.GetComponentInParent<HeatMapStatic>();

        if (hm == null)
        {
            Debug.LogWarning("[AimOnGrip] Hit object has no HeatMapStatic.");
            return;
        }

        bool uvValid = !(Mathf.Approximately(uv.x, 0f) && Mathf.Approximately(uv.y, 0f));

        float temp;

        if (uvValid)
        {
            temp = hm.GetTemperatureAtUV(uv);
            Debug.Log($"[HeatMap] Temperature at UV {uv} = {temp:F2}°C");

            text.GetComponent<TextMeshProUGUI>().text = $"Temperature = {temp:F2}°C";
        }
        else
        {
            temp = hm.GetTemperatureAtPointWorld(hit.point);
            Debug.Log($"[HeatMap] Temperature at world {hit.point} = {temp:F2}°C");
        }
    }

    private IEnumerator PulseHaptics(float strength, float duration)
    {
        OVRInput.SetControllerVibration(1f, strength, OVRInput.Controller.RTouch);
        yield return new WaitForSeconds(duration);
        OVRInput.SetControllerVibration(0f, 0f, OVRInput.Controller.RTouch);
    }

    private IEnumerator FlashMaterials()
    {
        if (flashing) yield break;
        flashing = true;

        if (laserLine != null && flashMaterial != null)
            laserLine.material = flashMaterial;

        if (aimObject != null && flashMaterial != null)
        {
            Renderer r = aimObject.GetComponent<Renderer>();
            if (r != null)
                r.material = flashMaterial;
        }

        yield return new WaitForSeconds(flashDuration);

        if (laserLine != null && defaultLaserMaterial != null)
            laserLine.material = defaultLaserMaterial;

        if (aimObject != null && defaultAimMaterial != null)
        {
            Renderer r = aimObject.GetComponent<Renderer>();
            if (r != null)
                r.material = defaultAimMaterial;
        }

        flashing = false;
    }
}
