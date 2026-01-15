using System;
using System.Collections;
using System.Collections.Generic;
using Fusion;
using Fusion.Sockets;
using UnityEngine;

public class SharedWorldSpawner : MonoBehaviour, INetworkRunnerCallbacks
{
    [Header("Prefabs (NetworkObject)")]
    public NetworkObject heatmapMarkerPrefab;
    public NetworkObject sharedAimCanvasStatePrefab;

    private NetworkRunner _boundRunner;
    private bool _callbacksAdded;

    private NetworkObject _spawnedHeatmapMarker;
    private NetworkObject _spawnedAimState;

    private bool _spawnQueued;

    private void Update()
    {
        var running = FindRunningRunner();
        if (running == null)
            return;

        if (_boundRunner == running && _callbacksAdded)
            return;

        Unbind();

        _boundRunner = running;
        _boundRunner.AddCallbacks(this);
        _callbacksAdded = true;

        Debug.Log($"[SharedWorldSpawner] Bound to RUNNING runner: {_boundRunner.name} | Mode={_boundRunner.GameMode} | IsServer={_boundRunner.IsServer}");
    }

    private NetworkRunner FindRunningRunner()
    {
        var runners = FindObjectsOfType<NetworkRunner>(true);
        if (runners == null || runners.Length == 0)
            return null;

        for (int i = 0; i < runners.Length; i++)
        {
            if (runners[i] != null && runners[i].IsRunning)
                return runners[i];
        }
        return null;
    }

    private void Unbind()
    {
        if (_boundRunner != null && _callbacksAdded)
        {
            _boundRunner.RemoveCallbacks(this);
            _callbacksAdded = false;
            Debug.Log("[SharedWorldSpawner] Unbound from previous runner.");
        }
        _boundRunner = null;
    }

    private bool HeatmapMarkerAlreadyExists() => FindObjectOfType<SharedHeatmapStateMarker>(true) != null;
    private bool AimStateAlreadyExists() => FindObjectOfType<SharedAimCanvasState>(true) != null;

    // --- Callbacks ---
    public void OnPlayerJoined(NetworkRunner runner, PlayerRef player)
    {
        Debug.Log($"[SharedWorldSpawner] OnPlayerJoined | player={player} | Mode={runner.GameMode} | IsServer={runner.IsServer}");
        QueueSpawn(runner);
    }

    public void OnSceneLoadDone(NetworkRunner runner)
    {
        Debug.Log($"[SharedWorldSpawner] OnSceneLoadDone | Mode={runner.GameMode} | IsServer={runner.IsServer}");
        QueueSpawn(runner);
    }

    private void QueueSpawn(NetworkRunner runner)
    {
        if (_spawnQueued) return;
        _spawnQueued = true;

        StartCoroutine(SpawnAfterOneFrame(runner));
    }

    private IEnumerator SpawnAfterOneFrame(NetworkRunner runner)
    {
        yield return null;
        TrySpawnSharedObjects(runner);
        _spawnQueued = false;
    }

    private void TrySpawnSharedObjects(NetworkRunner runner)
    {
        // Shared mode: samo prvi u sobi spawna (PlayerCount==1)
        int playerCount = 0;
        if (runner.SessionInfo.IsValid)
            playerCount = runner.SessionInfo.PlayerCount;

        bool shouldSpawn = runner.GameMode != GameMode.Shared ? runner.IsServer : (playerCount == 1);
        if (!shouldSpawn) return;

        // 1) Heatmap marker
        if (_spawnedHeatmapMarker == null && !HeatmapMarkerAlreadyExists())
        {
            if (heatmapMarkerPrefab == null)
            {
                Debug.LogError("[SharedWorldSpawner] heatmapMarkerPrefab nije postavljen!");
            }
            else
            {
                _spawnedHeatmapMarker = runner.Spawn(heatmapMarkerPrefab, Vector3.zero, Quaternion.identity, inputAuthority: null);
                Debug.Log("[SharedWorldSpawner] Spawned Heatmap marker.");
            }
        }

        // 2) Shared aim/canvas state
        if (_spawnedAimState == null && !AimStateAlreadyExists())
        {
            if (sharedAimCanvasStatePrefab == null)
            {
                Debug.LogError("[SharedWorldSpawner] sharedAimCanvasStatePrefab nije postavljen!");
            }
            else
            {
                _spawnedAimState = runner.Spawn(sharedAimCanvasStatePrefab, Vector3.zero, Quaternion.identity, inputAuthority: null);
                Debug.Log("[SharedWorldSpawner] Spawned SharedAimCanvasState.");
            }
        }
    }

    private void OnDestroy()
    {
        Unbind();
    }

    // --- Ostali callbackovi prazni ---
    public void OnPlayerLeft(NetworkRunner runner, PlayerRef player) { }
    public void OnInput(NetworkRunner runner, NetworkInput input) { }
    public void OnInputMissing(NetworkRunner runner, PlayerRef player, NetworkInput input) { }
    public void OnShutdown(NetworkRunner runner, ShutdownReason shutdownReason) { }
    public void OnConnectedToServer(NetworkRunner runner) { }
    public void OnDisconnectedFromServer(NetworkRunner runner, NetDisconnectReason reason) { }
    public void OnConnectRequest(NetworkRunner runner, NetworkRunnerCallbackArgs.ConnectRequest request, byte[] token) { }
    public void OnConnectFailed(NetworkRunner runner, NetAddress remoteAddress, NetConnectFailedReason reason) { }
    public void OnUserSimulationMessage(NetworkRunner runner, SimulationMessagePtr message) { }
    public void OnSessionListUpdated(NetworkRunner runner, List<SessionInfo> sessionList) { }
    public void OnCustomAuthenticationResponse(NetworkRunner runner, Dictionary<string, object> data) { }
    public void OnHostMigration(NetworkRunner runner, HostMigrationToken hostMigrationToken) { }
    public void OnReliableDataReceived(NetworkRunner runner, PlayerRef player, ReliableKey key, ArraySegment<byte> data) { }
    public void OnReliableDataProgress(NetworkRunner runner, PlayerRef player, ReliableKey key, float progress) { }
    public void OnSceneLoadStart(NetworkRunner runner) { }
    public void OnObjectEnterAOI(NetworkRunner runner, NetworkObject obj, PlayerRef player) { }
    public void OnObjectExitAOI(NetworkRunner runner, NetworkObject obj, PlayerRef player) { }
}
