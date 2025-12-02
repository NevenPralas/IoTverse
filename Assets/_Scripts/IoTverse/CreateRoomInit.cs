using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreateRoomInit : MonoBehaviour
{
    public GameObject floor;
    public GameObject myPrefab;

    public void SetEverything()
    {
        if (floor.GetComponent<HeatMapStatic>().isActiveAndEnabled == false)
        {
            floor.GetComponent<HeatMapStatic>().enabled = true;
        }

        Invoke(nameof(SetHeatController), 3f);
    }

    public void SetHeatController()
    {
        Debug.LogError("Pokrecemo nakon 3 sekunde!");
        if (GameObject.FindWithTag("HeatController") == null)
        {
            Transform parent = GameObject.Find("Joint RightHandWrist")?.transform;

            if (parent == null)
            {
                Debug.LogError("Ne mogu pronaci Joint RightHandWrist u LocalAvatar hijerarhiji!");
            }
            else
            {
                Debug.LogError("Pronasao Joint RightHandWrist u LocalAvatar hijerarhiji!");
                GameObject instance = Instantiate(myPrefab, parent);
                instance.transform.localPosition = new Vector3(0.0049f, 0.0208f, 0.0042f);
                instance.transform.localRotation = Quaternion.Euler(-165.493f, 161.189f, -2.58099f);
            }
        }
    }
}
