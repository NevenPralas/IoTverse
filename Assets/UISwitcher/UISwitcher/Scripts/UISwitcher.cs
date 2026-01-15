using UnityEngine;
using UnityEngine.UI;

namespace UISwitcher
{
    public class UISwitcher : UINullableToggle
    {
        private LocalHeatmapOverride localOverride;

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

            // lokalni override je na heatmap objektu; Find(true) je ok jer je lokalni po klijentu
            localOverride = FindObjectOfType<LocalHeatmapOverride>(true);

            Aim = GameObject.Find("AimController");
            LaserBeam = GameObject.Find("LaserBeam");
            AimObject = GameObject.Find("AimObject");
        }

        protected override void OnChanged(bool? obj)
        {
            if (obj.HasValue)
            {
                if (obj.Value) SetOn();
                else SetOff();
            }
            else SetNull();
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

        void Update()
        {
            if (OVRInput.GetDown(OVRInput.Button.Three))
            {
                ToggleState();
            }
        }

        private void ToggleState()
        {
            if (localOverride == null)
            {
                var hm = FindObjectOfType<HeatMapStaticWithJson>(true);
                if (hm != null)
                {
                    localOverride = hm.GetComponent<LocalHeatmapOverride>();
                    if (localOverride == null)
                        localOverride = hm.gameObject.AddComponent<LocalHeatmapOverride>();
                }
            }

            if (localOverride != null)
                localOverride.Toggle();
            else
                Debug.LogError("[UISwitcher] Ne mogu naći HeatMapStaticWithJson/LocalHeatmapOverride.");

            // UI vizual (tvoj postojeći toggle izgled)
            bool? v = isOnNullable;
            isOnNullable = !(v ?? false);

            // tvoje postojeće lokalne stvari
            if (Aim && Aim.TryGetComponent<AimOnGrip>(out var grip))
                grip.enabled = !grip.enabled;

            if (LaserBeam)
            {
                if (LaserBeam.activeSelf) LaserBeam.SetActive(false);
                else Invoke(nameof(InvokeLaserBeam), 0.5f);
            }

            if (AimObject)
            {
                if (AimObject.activeSelf) AimObject.SetActive(false);
                else Invoke(nameof(InvokeAimObject), 0.5f);
            }
        }

        void InvokeLaserBeam() { if (LaserBeam) LaserBeam.SetActive(true); }
        void InvokeAimObject() { if (AimObject) AimObject.SetActive(true); }
    }
}
