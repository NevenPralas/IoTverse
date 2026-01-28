

public interface INoiseDataProvider {
    NoiseData[] GetNoiseData(int sensorIndex, long start, long end);
    void GetNoiseDataAsync(int sensorIndex, System.Action<NoiseData[]> callback);
    void StartContinuousStream(int sensorIndex, System.Action<NoiseData> onDataReceived);
    void StopContinuousStream(int sensorIndex);
}