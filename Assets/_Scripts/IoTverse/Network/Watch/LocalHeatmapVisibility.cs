using UnityEngine;

/// <summary>
/// Lokalno paljenje/gašenje prikaza heatmape bez diranja Network state.
/// Stavi na isti objekt koji ima HeatMapStaticWithJson ili na parent floora.
/// </summary>
public class LocalHeatmapVisibility : MonoBehaviour
{
    [SerializeField] private Renderer[] renderersToToggle;

    private bool _visible = true;

    private void Awake()
    {
        if (renderersToToggle == null || renderersToToggle.Length == 0)
        {
            // automatski pokupi sve renderere ispod ovog objekta
            renderersToToggle = GetComponentsInChildren<Renderer>(true);
        }
    }

    public void SetVisible(bool visible)
    {
        _visible = visible;

        if (renderersToToggle == null) return;

        for (int i = 0; i < renderersToToggle.Length; i++)
        {
            if (renderersToToggle[i] != null)
                renderersToToggle[i].enabled = _visible;
        }
    }

    public void Toggle()
    {
        SetVisible(!_visible);
    }

    public bool IsVisible => _visible;
}
