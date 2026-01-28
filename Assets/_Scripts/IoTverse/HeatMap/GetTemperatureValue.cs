using TMPro;
using UnityEngine;

public class GetTemperatureValue : MonoBehaviour
{
    private HeatMapStaticWithJson heatMap;
    private TextMeshProUGUI tmp;

    [Header("Update settings")]
    [Tooltip("Koliko često osvježavati tekst (sekunde). 0.2 = 5x u sekundi.")]
    [SerializeField] private float refreshInterval = 0.2f;

    private float timer = 0f;

    private enum SensorKind { None, Angle1, Angle2, Angle3, Angle4 }
    private SensorKind sensorKind = SensorKind.None;

    void Awake()
    {
        tmp = GetComponent<TextMeshProUGUI>();
        if (tmp == null)
        {
            Debug.LogError($"[GetTemperatureValue] Nema TextMeshProUGUI na objektu: {gameObject.name}");
        }

        // Odredi koji kutni senzor prikazuje ovaj tekst prema imenu objekta
        // (zadržano kao u tvojoj originalnoj skripti)
        if (gameObject.name == "digi1T") sensorKind = SensorKind.Angle4; // angle4
        else if (gameObject.name == "digi2T") sensorKind = SensorKind.Angle2; // angle2
        else if (gameObject.name == "digi3T") sensorKind = SensorKind.Angle1; // angle1
        else if (gameObject.name == "digi4T") sensorKind = SensorKind.Angle3; // angle3
        else sensorKind = SensorKind.None;
    }

    void Start()
    {
        heatMap = FindObjectOfType<HeatMapStaticWithJson>();
        if (heatMap == null)
        {
            Debug.LogError("[GetTemperatureValue] HeatMapStaticWithJson nije pronađen u sceni.");
            return;
        }

        // odmah postavi tekst
        UpdateText();
    }

    void Update()
    {
        if (heatMap == null || tmp == null || sensorKind == SensorKind.None)
            return;

        timer += Time.deltaTime;
        if (timer >= refreshInterval)
        {
            timer = 0f;
            UpdateText();
        }
    }

    private void UpdateText()
    {
        float value = GetValue();
        // uvijek 1 decimalno mjesto (npr. 23.0°C, 23.4°C)
        tmp.text = $"{value:F1}°C";
    }

    private float GetValue()
    {
        switch (sensorKind)
        {
            case SensorKind.Angle1: return heatMap.angle1;
            case SensorKind.Angle2: return heatMap.angle2;
            case SensorKind.Angle3: return heatMap.angle3;
            case SensorKind.Angle4: return heatMap.angle4;
            default: return 0f;
        }
    }
}
