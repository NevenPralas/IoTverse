using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace UISwitcher
{
    public class UISwitcherForecast : UINullableToggle
    {
        private HeatMapStaticWithJson heatmap;

        private readonly Vector2 _min = new Vector2(0f, 0.5f);
        private readonly Vector2 _max = new Vector2(1f, 0.5f);
        private readonly Vector2 _middle = new Vector2(0.5f, 0.5f);

        [Header("UI")]
        [SerializeField] private Graphic backgroundGraphic;
        [SerializeField] private RectTransform tipRect;

        [Header("Colors")]
        [SerializeField] private Color onColor = new Color(0.2f, 0.8f, 0.2f, 1f);
        [SerializeField] private Color offColor = new Color(0.8f, 0.2f, 0.2f, 1f);
        [SerializeField] private Color nullColor = Color.white;

        [Header("Label (TMP)")]
        public TMP_Text modeText;

        protected override void Start()
        {
            base.Start();
            heatmap = FindObjectOfType<HeatMapStaticWithJson>();

            // prikaži default stanje
            OnChanged(isOnNullable);
        }

        protected override void OnChanged(bool? obj)
        {
            if (obj.HasValue)
            {
                if (obj.Value) SetOn();
                else SetOff();
            }
            else
            {
                SetNull();
            }
        }

        private void SetOn()
        {
            SetAnchors(_max);
            if (backgroundGraphic != null) backgroundGraphic.color = onColor;
            if (modeText != null) modeText.text = "Forecast";

            if (heatmap != null) heatmap.SetForecastEnabled(true);
        }

        private void SetOff()
        {
            SetAnchors(_min);
            if (backgroundGraphic != null) backgroundGraphic.color = offColor;
            if (modeText != null) modeText.text = "Historical Data";

            if (heatmap != null) heatmap.SetForecastEnabled(false);
        }

        private void SetNull()
        {
            SetAnchors(_middle);
            if (backgroundGraphic != null) backgroundGraphic.color = nullColor;
            if (modeText != null) modeText.text = "Historical Data";

            if (heatmap != null) heatmap.SetForecastEnabled(false);
        }

        private void SetAnchors(Vector2 anchor)
        {
            if (tipRect == null) return;
            tipRect.anchorMin = anchor;
            tipRect.anchorMax = anchor;
            tipRect.pivot = anchor;
        }

        // ===================== VR INPUT (META XR) =====================
        void Update()
        {
            // Y button na Quest = OVRInput.Button.Four
            if (OVRInput.GetDown(OVRInput.Button.Four))
            {
                ToggleState();
            }
        }

        private void ToggleState()
        {
            bool? v = isOnNullable;
            isOnNullable = !(v ?? false); // null -> false -> true
            // OnChanged će se pozvati kroz UINullableToggle
        }
    }
}
