using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class FetchDataDemo : MonoBehaviour
{
    string JsonUrl = "https://mocki.io/v1/7cbe89ca-6296-49db-8827-7d9237496c41";
    MeasurementData data;
    int currentIndex = 0;

    void Start()
    {
        StartCoroutine(FetchJson());
    }

    IEnumerator FetchJson()
    {
        using (UnityWebRequest req = UnityWebRequest.Get(JsonUrl))
        {
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Error fetching JSON: " + req.error);
            }
            else
            {
                string json = req.downloadHandler.text;
                data = JsonUtility.FromJson<MeasurementData>(json);

                Debug.Log("JSON Loaded! Starting playback...");
                StartCoroutine(PlayMeasurements());
            }
        }
    }

    IEnumerator PlayMeasurements()
    {
        while (currentIndex < data.measurements.Length)
        {
            Measurement m = data.measurements[currentIndex];

            Debug.Log(
                "Index: " + currentIndex +
                " | Timestamp: " + m.timestamp +
                " | T1: " + m.temperature1 +
                " | T2: " + m.temperature2 +
                " | T3: " + m.temperature3 +
                " | T4: " + m.temperature4
            );

            currentIndex++;
            yield return new WaitForSeconds(1f); 
        }

        Debug.Log("Playback finished!");
    }
}

[System.Serializable]
public class MeasurementData
{
    public Measurement[] measurements;
}

[System.Serializable]
public class Measurement
{
    public string timestamp;
    public float temperature1;
    public float temperature2;
    public float temperature3;
    public float temperature4;
}
