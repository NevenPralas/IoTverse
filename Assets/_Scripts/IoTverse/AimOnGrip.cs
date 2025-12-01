using System.Collections;
using UnityEngine;
using UnityEngine.XR;

public class AimOnGrip : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private GameObject aimObject;           // prefab / objekt koji će se pojaviti
    [SerializeField] private Transform rightHandAnchor;     // obično RightHandAnchor iz OVRCameraRig

    [Header("Behavior")]
    [Tooltip("Udaljenost od ruke kada nema hit-a")]
    [SerializeField] private float defaultDistance = 0.45f;
    [Tooltip("Koristi raycast da nišan stane na površinu (ako ima hit)")]
    [SerializeField] private bool useRaycast = true;
    [Tooltip("Layer mask za raycast (npr. Environment)")]
    [SerializeField] private LayerMask raycastMask = ~0;
    [Tooltip("Maks udaljenost raycast-a")]
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
    [SerializeField] private float hapticStrength = 0.5f; // 0..1

    [Header("Laser Beam")]
    [SerializeField] private LineRenderer laserLine;
    [SerializeField] private float laserMaxLength = 10f;

    [Header("Material Flash")]
    public Material flashMaterial;          // materijal za kratki flash
    public Material defaultLaserMaterial;   // default materijal LineRenderer-a
    public Material defaultAimMaterial;     // default materijal nišana
    public float flashDuration = 1f;      // trajanje flash efekta

    // internal
    private Vector3 velocity;
    private Quaternion rotVelocity = Quaternion.identity;
    private bool isVisible = false;
    private bool flashing = false;

    // A button usage
    private static readonly InputFeatureUsage<bool> AButtonUsage = CommonUsages.primaryButton;

    void Start()
    {
        if (aimObject == null)
        {
            Debug.LogWarning("AimOnGrip: aimObject nije postavljen u inspectoru.");
        }
        else
        {
            aimObject.SetActive(true); // držimo aktivnim da lakše skaliramo i pozicioniramo
            aimObject.transform.localScale = hiddenScale;
        }

        // pokušaj automatski naći RightHandAnchor ako nije postavljen
        if (rightHandAnchor == null)
        {
            var r = GameObject.Find("RightHandAnchor");
            if (r != null) rightHandAnchor = r.transform;
        }

        // postavi default materijale ako nisu postavljeni
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
        // --- 1) Čitanje grip gumba ---
        float gripValue = OVRInput.Get(OVRInput.Axis1D.SecondaryHandTrigger);
        bool gripHeld = gripValue > 0.12f;

        // --- 2) Provjera A button-a dok je grip držen ---
        if (gripHeld && OVRInput.GetDown(OVRInput.Button.One)) // A button
        {
            StartCoroutine(FlashMaterials());
        }

        // --- 3) Raycast prema Environment layeru ---
        bool hitEnvironment = false;
        Vector3 targetPos = Vector3.zero;
        Quaternion targetRot = Quaternion.identity;

        if (gripHeld && rightHandAnchor != null)
        {
            Ray ray = new Ray(rightHandAnchor.position, rightHandAnchor.forward);

            if (Physics.Raycast(ray, out RaycastHit hit, raycastMaxDistance, raycastMask))
            {
                hitEnvironment = true;
                targetPos = hit.point + hit.normal * 0.01f;
                targetRot = Quaternion.LookRotation(-hit.normal);
            }
        }

        // --- 4) Vidljivost nišana ---
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

        // --- 5) Update pozicije/rotacije i scale nišana ---
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

        // --- 6) Update laserske zrake ---
        if (laserLine != null)
        {
            if (shouldShow && rightHandAnchor != null)
            {
                laserLine.enabled = true;
                laserLine.SetPosition(0, rightHandAnchor.position);
                laserLine.SetPosition(1, hitEnvironment ? targetPos : rightHandAnchor.position + rightHandAnchor.forward * laserMaxLength);
            }
            else
            {
                laserLine.enabled = false;
            }
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
