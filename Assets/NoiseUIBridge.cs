using UnityEngine;

public class NoiseUIBridge : MonoBehaviour
{
    private SharedNoiseCanvasState _state;

    private void Awake()
    {
        // Awake je često prerano (state još nije spawnan) -> samo pokušaj
        LazyFindState();
        Debug.Log($"[NoiseUIBridge] Awake | stateFound={_state != null}");
    }

    private void Start()
    {
        // Start se događa kasnije -> velika šansa da je state već spawnan
        if (_state == null)
        {
            LazyFindState();
            Debug.Log($"[NoiseUIBridge] Start | stateFound={_state != null}");
        }
    }

    private bool LazyFindState()
    {
        if (_state != null) return true;

        _state = FindObjectOfType<SharedNoiseCanvasState>(true);
        return _state != null;
    }

    // --- S1-S4 toggles (OnValueChanged bool) ---
    public void SetSphere0(bool on)
    {
        Debug.Log($"[NoiseUIBridge] SetSphere0({on})");
        if (!LazyFindState()) return;
        _state.RequestSetSphere(0, on);
    }
    public void SetSphere1(bool on) { if (!LazyFindState()) return; _state.RequestSetSphere(1, on); }
    public void SetSphere2(bool on) { if (!LazyFindState()) return; _state.RequestSetSphere(2, on); }
    public void SetSphere3(bool on) { if (!LazyFindState()) return; _state.RequestSetSphere(3, on); }

    // --- Sensor buttons (OnClick) ---
    public void SelectSensor0() { if (!LazyFindState()) return; _state.RequestSetActiveSensor(0); }
    public void SelectSensor1() { if (!LazyFindState()) return; _state.RequestSetActiveSensor(1); }
    public void SelectSensor2() { if (!LazyFindState()) return; _state.RequestSetActiveSensor(2); }
    public void SelectSensor3() { if (!LazyFindState()) return; _state.RequestSetActiveSensor(3); }

    // --- Mock toggle (OnValueChanged bool) ---
    public void SetMock(bool on) { if (!LazyFindState()) return; _state.RequestSetMock(on); }
}
