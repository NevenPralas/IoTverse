using Fusion;
using UnityEngine;

public class SharedNoiseCanvasState : NetworkBehaviour
{
    [Networked] public NetworkBool Sphere1On { get; set; }
    [Networked] public NetworkBool Sphere2On { get; set; }
    [Networked] public NetworkBool Sphere3On { get; set; }
    [Networked] public NetworkBool Sphere4On { get; set; }
    [Networked] public NetworkBool MockOn { get; set; }
    [Networked] public int ActiveSensorIndex { get; set; } // 0..3

    private NoiseManager _noiseManager;

    // Cache da ne zovemo Apply svaki frame bez potrebe
    private bool _lastS1, _lastS2, _lastS3, _lastS4, _lastMock;
    private int _lastActive;

    public override void Spawned()
    {
        _noiseManager = FindObjectOfType<NoiseManager>(true);

        if (Object.HasStateAuthority)
        {
            Sphere1On = true;
            Sphere2On = true;
            Sphere3On = true;
            Sphere4On = true;
            MockOn = false;
            ActiveSensorIndex = 0;
        }

        // Force apply
        _lastS1 = !Sphere1On;
        _lastS2 = !Sphere2On;
        _lastS3 = !Sphere3On;
        _lastS4 = !Sphere4On;
        _lastMock = !MockOn;
        _lastActive = ActiveSensorIndex + 999;

        ApplyIfChanged();
    }

    public override void Render()
    {
        ApplyIfChanged();
    }

    private void ApplyIfChanged()
    {
        bool s1 = Sphere1On;
        bool s2 = Sphere2On;
        bool s3 = Sphere3On;
        bool s4 = Sphere4On;
        bool mk = MockOn;
        int active = Mathf.Clamp(ActiveSensorIndex, 0, 3);

        if (s1 == _lastS1 && s2 == _lastS2 && s3 == _lastS3 && s4 == _lastS4 && mk == _lastMock && active == _lastActive)
            return;

        _lastS1 = s1; _lastS2 = s2; _lastS3 = s3; _lastS4 = s4;
        _lastMock = mk;
        _lastActive = active;

        if (_noiseManager == null)
            _noiseManager = FindObjectOfType<NoiseManager>(true);

        if (_noiseManager != null)
        {
            _noiseManager.ApplyNetworkState(s1, s2, s3, s4, active, mk);
        }
    }

    // ---- Requests ----
    public void RequestSetSphere(int index, bool on)
    {
        index = Mathf.Clamp(index, 0, 3);
        if (Object.HasStateAuthority) SetSphereInternal(index, on);
        else RPC_SetSphere(index, on);
    }

    [Rpc(RpcSources.All, RpcTargets.StateAuthority)]
    private void RPC_SetSphere(int index, bool on)
    {
        Debug.Log($"[SharedNoiseCanvasState] RPC_SetSphere index={index} on={on}");
        SetSphereInternal(index, on);
    }

    private void SetSphereInternal(int index, bool on)
    {
        switch (index)
        {
            case 0: Sphere1On = on; break;
            case 1: Sphere2On = on; break;
            case 2: Sphere3On = on; break;
            case 3: Sphere4On = on; break;
        }
    }

    public void RequestSetMock(bool on)
    {
        if (Object.HasStateAuthority) MockOn = on;
        else RPC_SetMock(on);
    }

    [Rpc(RpcSources.All, RpcTargets.StateAuthority)]
    private void RPC_SetMock(bool on)
    {
        MockOn = on;
    }

    public void RequestSetActiveSensor(int index)
    {
        index = Mathf.Clamp(index, 0, 3);
        if (Object.HasStateAuthority) ActiveSensorIndex = index;
        else RPC_SetActiveSensor(index);
    }

    [Rpc(RpcSources.All, RpcTargets.StateAuthority)]
    private void RPC_SetActiveSensor(int index)
    {
        ActiveSensorIndex = Mathf.Clamp(index, 0, 3);
    }
}
