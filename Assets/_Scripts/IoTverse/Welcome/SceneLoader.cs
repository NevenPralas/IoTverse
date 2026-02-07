using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneLoader : MonoBehaviour
{
    public void LoadVR()
    {
        SceneManager.LoadScene("Meta_vr");
    }

    public void LoadAR()
    {
        SceneManager.LoadScene("Meta_ar");
    }
}
