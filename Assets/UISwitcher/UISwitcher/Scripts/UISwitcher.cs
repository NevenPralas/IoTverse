using UnityEngine;
using UnityEngine.UI;

namespace UISwitcher
{
    public class UISwitcher : UINullableToggle
    {
        private HeatMapStaticWithJson heatmap;

        private readonly Vector2 _min = new(0, 0.5f);
        private readonly Vector2 _max = new(1, 0.5f);
        private readonly Vector2 _middle = new(0.5f, 0.5f);

        [SerializeField] private Graphic backgroundGraphic;
        [SerializeField] private Color onColor, offColor, nullColor;
        [SerializeField] private RectTransform tipRect;

        private GameObject Aim;
        private GameObject LaserBeam;
        private GameObject AimObject;

        protected override void Start()
        {
            base.Start();
            heatmap = FindObjectOfType<HeatMapStaticWithJson>();

            Aim = GameObject.Find("AimController");
            LaserBeam = GameObject.Find("LaserBeam");
            AimObject = GameObject.Find("AimObject");
        }

        protected override void OnChanged(bool? obj)
        {
            if (obj.HasValue)
                if (obj.Value) SetOn();
                else SetOff();
            else
                SetNull();
        }

        private void SetOn()
        {
            SetAnchors(_max);
            backgroundGraphic.color = onColor;
        }

        private void SetOff()
        {
            SetAnchors(_min);
            backgroundGraphic.color = offColor;
        }

        private void SetNull()
        {
            SetAnchors(_middle);
            backgroundGraphic.color = nullColor;
        }

        private void SetAnchors(Vector2 anchor)
        {
            tipRect.anchorMin = anchor;
            tipRect.anchorMax = anchor;
            tipRect.pivot = anchor;
        }

        // ===================== VR INPUT (META XR) =====================
        void Update()
        {
            // X button je OVRInput.Button.Three
            if (OVRInput.GetDown(OVRInput.Button.Three))
            {
                ToggleState();
            }
        }

        private void ToggleState()
        {
            heatmap.ToggleHeatmap();

            bool? v = isOnNullable;
            isOnNullable = !(v ?? false);

            if (Aim.GetComponent<AimOnGrip>().isActiveAndEnabled)
            {
                Aim.GetComponent<AimOnGrip>().enabled = false;
            }
            else
            {
                Aim.GetComponent<AimOnGrip>().enabled = true;
            }

            if (LaserBeam.activeSelf)
            {
                LaserBeam.SetActive(false);
            }
            else
            {
                Invoke("InvokeLaserBeam", 0.5f);
            }

            if (AimObject.activeSelf)
            {
                AimObject.SetActive(false);
            }
            else
            {
                Invoke("InvokeAimObject", 0.5f);
            }

        }

        void InvokeLaserBeam()
        {
            LaserBeam.SetActive(true);
        }

        void InvokeAimObject()
        {
            AimObject.SetActive(true);
        }

    }
}
