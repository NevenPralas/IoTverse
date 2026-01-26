using UnityEngine;
using UnityEngine.UI;

public class SliderTransformController : MonoBehaviour
{
    [Header("Target (object you want to move/rotate)")]
    public Transform target;

    [Header("Sliders")]
    public Slider sliderPositionX;   // Slider_Position_X
    public Slider sliderPositionZ;   // Slider_Position_Z
    public Slider sliderRotationY;   // Slider_Rotation_Y

    [Header("Position settings")]
    public float positionRange = 10f; // +/- 10 around original position at slider=0.5

    [Header("Rotation settings")]
    [Tooltip("Half range around original at slider=0.5. 90 means: 0 -> -90, 0.5 -> 0, 1 -> +90 (relative to original).")]
    public float rotationHalfRange = 90f;

    [Tooltip("If true uses localPosition/localRotation (recommended when target has children).")]
    public bool useLocal = true;

    private Vector3 originalPos;
    private Quaternion originalRot;

    private void Awake()
    {
        if (target == null)
            target = transform;
    }

    private void Start()
    {
        // Save originals (0.5 reference point)
        originalPos = useLocal ? target.localPosition : target.position;
        originalRot = useLocal ? target.localRotation : target.rotation;

        SetupSlider(sliderPositionX, 0.5f);
        SetupSlider(sliderPositionZ, 0.5f);
        SetupSlider(sliderRotationY, 0.5f);

        if (sliderPositionX) sliderPositionX.onValueChanged.AddListener(OnSliderChanged);
        if (sliderPositionZ) sliderPositionZ.onValueChanged.AddListener(OnSliderChanged);
        if (sliderRotationY) sliderRotationY.onValueChanged.AddListener(OnSliderChanged);

        Apply();
    }

    private void OnDestroy()
    {
        if (sliderPositionX) sliderPositionX.onValueChanged.RemoveListener(OnSliderChanged);
        if (sliderPositionZ) sliderPositionZ.onValueChanged.RemoveListener(OnSliderChanged);
        if (sliderRotationY) sliderRotationY.onValueChanged.RemoveListener(OnSliderChanged);
    }

    private void SetupSlider(Slider s, float defaultValue)
    {
        if (!s) return;
        s.minValue = 0f;
        s.maxValue = 1f;
        s.value = defaultValue;
    }

    private void OnSliderChanged(float _)
    {
        Apply();
    }

    private void Apply()
    {
        // ---- POSITION ----
        Vector3 pos = useLocal ? target.localPosition : target.position;

        if (sliderPositionX)
        {
            float offsetX = (sliderPositionX.value - 0.5f) * 2f * positionRange; // -10..+10
            pos.x = originalPos.x + offsetX;
        }

        if (sliderPositionZ)
        {
            float offsetZ = (sliderPositionZ.value - 0.5f) * 2f * positionRange; // -10..+10
            pos.z = originalPos.z + offsetZ;
        }

        if (useLocal) target.localPosition = pos;
        else target.position = pos;

        // ---- ROTATION (0.5 = original, 0 = original-90, 1 = original+90) ----
        if (sliderRotationY)
        {
            float offsetY = (sliderRotationY.value - 0.5f) * 2f * rotationHalfRange; // -90..+90
            Quaternion rot = originalRot * Quaternion.Euler(0f, offsetY, 0f);

            if (useLocal) target.localRotation = rot;
            else target.rotation = rot;
        }
    }
}
