

using System;

[Serializable]
public class NoiseData
{
    public long timestamp;
    public float decibels;

    public NoiseData(long timestamp, float decibels)
    {
        this.timestamp = timestamp;
        this.decibels = decibels;
    }
}