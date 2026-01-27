using UnityEngine;
using UnityEngine.UI;

public class SliderTransformController : MonoBehaviour
{
    [Header("Target (object you want to move/rotate)")]
    public Transform target;

    [Header("Sliders")]
    public Slider sliderPositionX;
    public Slider sliderPositionY;
    public Slider sliderPositionZ;
    public Slider sliderRotationY;

    [Header("Buttons")]
    public Button buttonPosXMinus;
    public Button buttonPosXPlus;
    public Button buttonPosYMinus;
    public Button buttonPosYPlus;
    public Button buttonPosZMinus;
    public Button buttonPosZPlus;
    public Button buttonRotYMinus;
    public Button buttonRotYPlus;

    [Header("Position settings")]
    public float positionRange = 10f; // +/- around original at slider=0.5

    [Header("Rotation settings")]
    public float rotationHalfRange = 90f;

    [Header("Button step sizes")]
    public float positionStep = 0.1f;
    public float rotationStepDegrees = 1f;

    [Tooltip("Rotation space: true = localRotation, false = world rotation")]
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
        // Position koristi SUPROTAN space od rotacije
        bool positionUsesLocal = !useLocal;

        originalPos = positionUsesLocal ? target.localPosition : target.position;
        originalRot = useLocal ? target.localRotation : target.rotation;

        SetupSlider(sliderPositionX);
        SetupSlider(sliderPositionY);
        SetupSlider(sliderPositionZ);
        SetupSlider(sliderRotationY);

        if (sliderPositionX) sliderPositionX.onValueChanged.AddListener(OnSliderChanged);
        if (sliderPositionY) sliderPositionY.onValueChanged.AddListener(OnSliderChanged);
        if (sliderPositionZ) sliderPositionZ.onValueChanged.AddListener(OnSliderChanged);
        if (sliderRotationY) sliderRotationY.onValueChanged.AddListener(OnSliderChanged);

        if (buttonPosXMinus) buttonPosXMinus.onClick.AddListener(() => NudgePositionX(-positionStep));
        if (buttonPosXPlus) buttonPosXPlus.onClick.AddListener(() => NudgePositionX(+positionStep));

        if (buttonPosYMinus) buttonPosYMinus.onClick.AddListener(() => NudgePositionY(-positionStep));
        if (buttonPosYPlus) buttonPosYPlus.onClick.AddListener(() => NudgePositionY(+positionStep));

        if (buttonPosZMinus) buttonPosZMinus.onClick.AddListener(() => NudgePositionZ(-positionStep));
        if (buttonPosZPlus) buttonPosZPlus.onClick.AddListener(() => NudgePositionZ(+positionStep));

        if (buttonRotYMinus) buttonRotYMinus.onClick.AddListener(() => NudgeRotationY(-rotationStepDegrees));
        if (buttonRotYPlus) buttonRotYPlus.onClick.AddListener(() => NudgeRotationY(+rotationStepDegrees));

        Apply();
    }

    private void SetupSlider(Slider s)
    {
        if (!s) return;
        s.minValue = 0f;
        s.maxValue = 1f;
        s.value = 0.5f;
    }

    private void OnSliderChanged(float _)
    {
        Apply();
    }

    // -------- BUTTON NUDGES --------

    private void NudgePositionX(float delta)
    {
        sliderPositionX.value = Mathf.Clamp01(sliderPositionX.value + UnitsToSliderDelta(delta));
    }

    private void NudgePositionY(float delta)
    {
        sliderPositionY.value = Mathf.Clamp01(sliderPositionY.value + UnitsToSliderDelta(delta));
    }

    private void NudgePositionZ(float delta)
    {
        sliderPositionZ.value = Mathf.Clamp01(sliderPositionZ.value + UnitsToSliderDelta(delta));
    }

    private void NudgeRotationY(float deltaDegrees)
    {
        sliderRotationY.value = Mathf.Clamp01(
            sliderRotationY.value + DegreesToSliderDelta(deltaDegrees)
        );
    }

    private float UnitsToSliderDelta(float units)
    {
        return units / (2f * Mathf.Max(0.0001f, positionRange));
    }

    private float DegreesToSliderDelta(float degrees)
    {
        return degrees / (2f * Mathf.Max(0.0001f, rotationHalfRange));
    }

    // -------- APPLY --------

    private void Apply()
    {
        bool positionUsesLocal = !useLocal;

        Vector3 pos = positionUsesLocal ? target.localPosition : target.position;

        if (sliderPositionX)
            pos.x = originalPos.x + (sliderPositionX.value - 0.5f) * 2f * positionRange;

        if (sliderPositionY)
            pos.y = originalPos.y + (sliderPositionY.value - 0.5f) * 2f * positionRange;

        if (sliderPositionZ)
            pos.z = originalPos.z + (sliderPositionZ.value - 0.5f) * 2f * positionRange;

        if (positionUsesLocal) target.localPosition = pos;
        else target.position = pos;

        if (sliderRotationY)
        {
            float offsetY = (sliderRotationY.value - 0.5f) * 2f * rotationHalfRange;
            Quaternion rot = originalRot * Quaternion.Euler(0f, offsetY, 0f);

            if (useLocal) target.localRotation = rot;
            else target.rotation = rot;
        }
    }
}
