using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

// ==== VR Input cleanup ====
using XRDevice = UnityEngine.XR.InputDevice;
using XRCommon = UnityEngine.XR.CommonUsages;

namespace UISwitcher
{
    public class UISwitcher : UINullableToggle
    {
        // ---- VR A button handling ----
        private bool aButtonDown = false;
        private XRDevice rightController;

        private HeatMapStatic heatmap;

        private readonly Vector2 _min = new(0, 0.5f);
        private readonly Vector2 _max = new(1, 0.5f);
        private readonly Vector2 _middle = new(0.5f, 0.5f);

        [SerializeField] private Graphic backgroundGraphic;
        [SerializeField] private Color onColor, offColor, nullColor;
        [SerializeField] private RectTransform tipRect;

        private Color backgroundColor
        {
            set
            {
                if (backgroundGraphic == null) return;
                backgroundGraphic.color = value;
            }
        }

        protected override void OnChanged(bool? obj)
        {
            if (obj.HasValue)
            {
                if (obj.Value)
                    SetOn();
                else
                    SetOff();
            }
            else
            {
                SetNull();
            }
        }

        private void SetOn()
        {
            SetAnchors(_max);
            backgroundColor = onColor;
        }

        private void SetOff()
        {
            SetAnchors(_min);
            backgroundColor = offColor;
        }

        private void SetNull()
        {
            SetAnchors(_middle);
            backgroundColor = nullColor;
        }

        private void SetAnchors(Vector2 anchor)
        {
            tipRect.anchorMin = anchor;
            tipRect.anchorMax = anchor;
            tipRect.pivot = anchor;
        }

        // ================ VR INIT ======================
        protected override void Start()
        {
            base.Start();

            heatmap = FindObjectOfType<HeatMapStatic>();

            var found = new List<XRDevice>();
            UnityEngine.XR.InputDevices.GetDevicesWithCharacteristics(
                UnityEngine.XR.InputDeviceCharacteristics.Right |
                UnityEngine.XR.InputDeviceCharacteristics.Controller,
                found
            );

            if (found.Count > 0)
                rightController = found[0];
        }

        // ================ VR UPDATE ======================
        void Update()
        {
            // Ako kontroler nije validan → pokusaj ponovno pronaci
            if (!rightController.isValid)
            {
                var list = new List<XRDevice>();
                UnityEngine.XR.InputDevices.GetDevicesWithCharacteristics(
                    UnityEngine.XR.InputDeviceCharacteristics.Right |
                    UnityEngine.XR.InputDeviceCharacteristics.Controller,
                    list
                );

                if (list.Count > 0)
                    rightController = list[0];
            }

            // ---- Citanje A buttona ----
            if (rightController.TryGetFeatureValue(XRCommon.primaryButton, out bool pressed))
            {
                if (pressed && !aButtonDown)
                {
                    aButtonDown = true;
                    ToggleState(); 
                }
                else if (!pressed)
                {
                    aButtonDown = false;
                }
            }
        }

        // ================ TOGGLE BEZ PROMJENA LOGIKE ======================
        private void ToggleState()
        {
            heatmap.ToggleHeatmap();
            bool? v = isOnNullable;

            // Ako zelis samo ON/OFF:
            isOnNullable = !(v ?? false);

            // Ako zelis ON → OFF → NULL → ON:
            // if (!v.HasValue) isOnNullable = true;
            // else if (v.Value) isOnNullable = false;
            // else isOnNullable = null;
        }

    }
}
