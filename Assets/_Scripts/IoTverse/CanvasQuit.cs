using UnityEngine;

public class CanvasQuit : MonoBehaviour
{
    public GameObject canvas;   // Canvas kojeg gasimo
    public GameObject portal;
    void Update()
    {
        // X dugme na lijevom kontroleru
        if (OVRInput.GetDown(OVRInput.Button.Three))
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
