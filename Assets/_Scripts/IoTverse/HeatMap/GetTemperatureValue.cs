using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class GetTemperatureValue : MonoBehaviour
{
    private HeatMapStaticWithJson heatMap;

    void Start()
    {
        heatMap = FindObjectOfType<HeatMapStaticWithJson>();

        if (gameObject.name == "digi1T") //angle4
            if (!Mathf.Approximately(heatMap.angle4, Mathf.Round(heatMap.angle4)))
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle4.ToString() + "°C";
            else
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle4.ToString() + ".0°C";
        else if (gameObject.name == "digi2T") //angle2
            if (!Mathf.Approximately(heatMap.angle2, Mathf.Round(heatMap.angle2)))
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle2.ToString() + "°C";
            else
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle2.ToString() + ".0°C";
        else if (gameObject.name == "digi3T") //angle1
            if (!Mathf.Approximately(heatMap.angle1, Mathf.Round(heatMap.angle1)))
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle1.ToString() + "°C";
            else
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle1.ToString() + ".0°C";
        else if (gameObject.name == "digi4T") //angle3
            if (!Mathf.Approximately(heatMap.angle3, Mathf.Round(heatMap.angle3)))
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle3.ToString() + "°C";
            else
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle3.ToString() + ".0°C";

    }



    void Update()
    {
         heatMap = FindObjectOfType<HeatMapStaticWithJson>();

        if (gameObject.name == "digi1T") //angle4
            if (!Mathf.Approximately(heatMap.angle4, Mathf.Round(heatMap.angle4)))
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle4.ToString() + "°C";
            else
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle4.ToString() + ".0°C";
        else if (gameObject.name == "digi2T") //angle2
            if (!Mathf.Approximately(heatMap.angle2, Mathf.Round(heatMap.angle2)))
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle2.ToString() + "°C";
            else
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle2.ToString() + ".0°C";
        else if (gameObject.name == "digi3T") //angle1
            if (!Mathf.Approximately(heatMap.angle1, Mathf.Round(heatMap.angle1)))
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle1.ToString() + "°C";
            else
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle1.ToString() + ".0°C";
        else if (gameObject.name == "digi4T") //angle3
            if (!Mathf.Approximately(heatMap.angle3, Mathf.Round(heatMap.angle3)))
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle3.ToString() + "°C";
            else
                gameObject.GetComponent<TextMeshProUGUI>().text = heatMap.angle3.ToString() + ".0°C";
    } 
}
