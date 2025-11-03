using Fusion.Sockets;
using Fusion;
using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;
using WebSocketSharp;
using Meta.XR.MultiplayerBlocks.Fusion;
using UnityEngine.UI;

namespace Digiphy
{
    public class FusionConnection : SingletonPersistent<FusionConnection>, INetworkRunnerCallbacks
    {
        [SerializeField] private AvatarSpawnerFusion _avatarSpawnerFusion = null;
        [SerializeField] private int _playerCount = 2;
        [SerializeField] private NetworkRunner _runner = null;
        [SerializeField] private VoiceSetup _voiceSetup = null;
        [SerializeField] private Button _createARoomButton;
        private List<SessionInfo> _sessions = new List<SessionInfo>();

        public List<SessionInfo> Sessions => _sessions;

        private void Awake()
        {
            base.Awake();
            ConnectToLobby();
        }

        public void ConnectToLobby()
        {
            _runner.JoinSessionLobby(SessionLobby.Shared);
        }

        public async void CreateSession(string sessionName)
        {
            _runner.ProvideInput = false;

            await _runner.StartGame(new StartGameArgs()
            {
                GameMode = GameMode.Shared,
                SessionName = sessionName,
                PlayerCount = _playerCount,
            });
        }

        public async void JoinSession(string sessionName)
        {
            _runner.ProvideInput = true;

            await _runner.StartGame(new StartGameArgs()
            {
                GameMode = GameMode.Shared,
                SessionName = sessionName,
            });
        }

        public void LeaveSession()
        {
            _runner.Shutdown();
            _runner = gameObject.AddComponent<NetworkRunner>();
        }

        public void OnSessionListUpdated(NetworkRunner runner, List<SessionInfo> sessionList)
        {
            _createARoomButton.interactable = true;
            if (_sessions == null || (_sessions.Count == 0 && sessionList.Count > 0))
            {
                _sessions = sessionList;
                RoomManager.Instance.UpdateSessionList();
            }
            else _sessions = sessionList;
        }

        public void OnPlayerJoined(NetworkRunner runner, PlayerRef player)
        {
            Debug.Log("On Player Joined");

            if (player == runner.LocalPlayer)
            {
                _avatarSpawnerFusion?.SpawnAvatar();
                ArSpawner.Instance?.JoinedNetworkSession(_runner);
            }
        }


        public void OnPlayerLeft(NetworkRunner runner, PlayerRef player)
        {
            Debug.Log("On Player Left");
            TestingSetupManager.Instance.PlayerDisconnected(runner, player);
        }

        public void OnDisconnectedFromServer(NetworkRunner runner, NetDisconnectReason reason)
        {
            Debug.Log("On Disconnected From Server");
        }

        public void OnShutdown(NetworkRunner runner, ShutdownReason shutdownReason)
        {
            Debug.Log("On Shut down, reason: " + shutdownReason.ToString());
        }

        private void OnApplicationQuit()
        {
            base.OnApplicationQuit();

            if (_runner != null && _runner.IsRunning) _runner.Shutdown();
        }

        #region UnusedCallbacks
        public void OnInput(NetworkRunner runner, NetworkInput input)
        {
            Debug.Log("On Input");
        }

        public void OnConnectedToServer(NetworkRunner runner)
        {
            Debug.Log("On Connected to server");
        }

        public void OnConnectFailed(NetworkRunner runner, NetAddress remoteAddress, NetConnectFailedReason reason)
        {
            Debug.Log("On Connect Failed");
        }

        public void OnConnectRequest(NetworkRunner runner, NetworkRunnerCallbackArgs.ConnectRequest request, byte[] token)
        {
            Debug.Log("On Connect Request");
        }

        public void OnCustomAuthenticationResponse(NetworkRunner runner, Dictionary<string, object> data)
        {
            Debug.Log("On Custom Authentication Response");
        }

        public void OnHostMigration(NetworkRunner runner, HostMigrationToken hostMigrationToken)
        {
            Debug.Log("On Host Migration");
        }

        public void OnInputMissing(NetworkRunner runner, PlayerRef player, NetworkInput input)
        {
            Debug.Log("On Input Missing");
        }

        public void OnObjectEnterAOI(NetworkRunner runner, NetworkObject obj, PlayerRef player)
        {
            Debug.Log("On Object Enter AOI");
        }

        public void OnObjectExitAOI(NetworkRunner runner, NetworkObject obj, PlayerRef player)
        {
            Debug.Log("OnO bject Exit AOI");
        }

        public void OnReliableDataProgress(NetworkRunner runner, PlayerRef player, ReliableKey key, float progress)
        {
            Debug.Log("On Reliable Data Progress");
        }

        public void OnReliableDataReceived(NetworkRunner runner, PlayerRef player, ReliableKey key, ArraySegment<byte> data)
        {
            Debug.Log("On Reliable Data Received");
        }

        public void OnSceneLoadDone(NetworkRunner runner)
        {
            Debug.Log("On Scene Load Done");
        }

        public void OnSceneLoadStart(NetworkRunner runner)
        {
            Debug.Log("On Scene Load Start");
        }

        public void OnUserSimulationMessage(NetworkRunner runner, SimulationMessagePtr message)
        {
            Debug.Log("On User Simulation Message");
        }
        #endregion
    }
}
