using UnityEngine;

public class CanvasQuit : MonoBehaviour
{
    public GameObject canvas;   // Canvas kojeg gasimo
    public GameObject portal;
    void Update()
    {
        // B dugme na desnom kontroleru
        if (OVRInput.GetDown(OVRInput.Button.Two))
        {
            if (canvas != null)
            {
                Canvas c = canvas.GetComponent<Canvas>();
                if (c != null)
                {
                    c.enabled = false;
                    portal.SetActive(false);
                }
            }
        }
    }
}
