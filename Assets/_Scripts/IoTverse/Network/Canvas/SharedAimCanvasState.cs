using Fusion;
using UnityEngine;

public class SharedAimCanvasState : NetworkBehaviour
{
    [Header("Scene refs (assign or auto-find)")]
    [SerializeField] private GameObject canvas;
    [SerializeField] private GameObject portal;

    [Header("Optional: look at camera")]
    [SerializeField] private float canvasHeight = 1.2f;

    [Networked] public NetworkBool CanvasOn { get; set; }
    [Networked] public Vector3 HitPoint { get; set; }

    private HeatMapStaticWithJson hm;
    private AimOnGrip aim;

    // cache da ne radimo sve svaki Render frame
    private bool _lastOn;
    private Vector3 _lastPoint;

    public override void Spawned()
    {
        if (canvas == null)
        {
            var c = GameObject.Find("Canvas");
            if (c != null) canvas = c;
        }

        if (portal == null)
        {
            var p = GameObject.Find("Portal green");
            if (p != null) portal = p;
        }

        hm = FindObjectOfType<HeatMapStaticWithJson>(true);
        aim = FindObjectOfType<AimOnGrip>(true);

        // init cache da force apply
        _lastOn = !CanvasOn;
        _lastPoint = HitPoint + Vector3.one * 9999f;

        ApplyIfChanged();
    }

    public override void Render()
    {
        ApplyIfChanged();
    }

    private void ApplyIfChanged()
    {
        bool on = CanvasOn;
        Vector3 p = HitPoint;

        if (on == _lastOn && (!on || (p - _lastPoint).sqrMagnitude < 0.000001f))
            return;

        _lastOn = on;
        _lastPoint = p;

        // Canvas enable/disable + position/rotation
        if (canvas != null)
        {
            var c = canvas.GetComponent<Canvas>();
            if (c != null) c.enabled = on;

            if (on)
            {
                canvas.transform.position = p + Vector3.up * canvasHeight;

                var cam = Camera.main != null ? Camera.main.transform : null;
                if (cam != null)
                {
                    Vector3 lookDir = cam.position - canvas.transform.position;
                    lookDir.y = 0;
                    if (lookDir.sqrMagnitude > 0.001f)
                    {
                        canvas.transform.rotation = Quaternion.LookRotation(lookDir);
                        canvas.transform.Rotate(0, 180f, 0);
                    }
                }
            }
        }

        if (portal != null)
            portal.SetActive(on);

        // Update chart + temperature (lokalno na svakom klijentu)
        if (on && hm != null)
        {
            hm.UpdateChartForPoint(p);
            float temp = hm.GetTemperatureAtPointWorld(p);
            if (aim != null) aim.UpdateTemperatureDisplay(temp);
        }
    }

    public void RequestSetPoint(Vector3 point)
    {
        if (Object.HasStateAuthority)
        {
            HitPoint = point;
            CanvasOn = true;
        }
        else
        {
            RPC_SetPoint(point);
        }
    }

    [Rpc(RpcSources.All, RpcTargets.StateAuthority)]
    private void RPC_SetPoint(Vector3 point)
    {
        HitPoint = point;
        CanvasOn = true;
    }

    public void RequestHide()
    {
        if (Object.HasStateAuthority)
            CanvasOn = false;
        else
            RPC_Hide();
    }

    [Rpc(RpcSources.All, RpcTargets.StateAuthority)]
    private void RPC_Hide()
    {
        CanvasOn = false;
    }
}
