using UnityEngine;

[CreateAssetMenu(fileName = "Settings", menuName = "Settings levels")]
public class SettingsLevels : ScriptableObject
{
    public float SoundLevels;

    private void OnEnable()
    {
        SoundLevels = 1f;
    }
}
