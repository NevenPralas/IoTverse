using Fusion;
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Digiphy
{
    public class RoomManager : Singleton<RoomManager>
    {
        [SerializeField] private Button _refreshButton = null;
        [SerializeField] private Button _joinButton = null;
        [SerializeField] private Button _createRoomButton = null;
        [SerializeField] private SessionDataView _sessionDataViewPrefab = null;
        [SerializeField] private ToggleGroup _sessionListContainer = null;
        [SerializeField] private string _userType;
        [SerializeField] private GameObject _roomJoinMenu;
        [SerializeField] private GameObject _chessSettingMenu;
        private List<SessionDataView> _sessions = new List<SessionDataView>();
        private System.Random _rand = new System.Random();
        private string _sessionData;

        private void Awake()
        {
            base.Awake();
            _refreshButton.onClick.AddListener(UpdateSessionList);
            _joinButton.onClick.AddListener(() => 
            {
                _roomJoinMenu.SetActive(false);
                _chessSettingMenu.SetActive(true);
                FusionConnection.Instance.JoinSession(_sessionData);
            });
            _joinButton.interactable = false;
            _createRoomButton.onClick.AddListener(() =>
            {
                _roomJoinMenu.SetActive(false);
                _chessSettingMenu.SetActive(true);
                string numbers = "";

                for (int i = 0; i < 4; i++)
                {
                    numbers += _rand.Next(0, 100);
                }
                FusionConnection.Instance.CreateSession(_userType + "[" + numbers + "]");
            });
        }

        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.W))
            {
                _roomJoinMenu.SetActive(false);
                _chessSettingMenu.SetActive(true);
                string numbers = "";

                for (int i = 0; i < 4; i++)
                {
                    numbers += _rand.Next(0, 100);
                }
                FusionConnection.Instance.CreateSession(_userType + "[" + numbers + "]");
            }

            if (Input.GetKeyDown(KeyCode.E))
            {
                UpdateSessionList();
            }
        }

        public void UpdateSessionList()
        {
            List<SessionInfo> sessionList = FusionConnection.Instance.Sessions;
            _sessions.ForEach(session => Destroy(session.gameObject));
            _sessions.Clear();

            foreach (SessionInfo sessionInfo in sessionList)
            {
                SessionDataView sessionDataView = Instantiate(_sessionDataViewPrefab, _sessionListContainer.transform);
                _sessions.Add(sessionDataView);
                string sessionName = sessionInfo.Name;
                int playerCount = sessionInfo.PlayerCount;
                int maxPlayerCount = sessionInfo.MaxPlayers;

                sessionDataView.ShowSession(sessionName, playerCount, SessionOnToggle, _sessionListContainer);
            }
        }

        private void SessionOnToggle(bool isOn, string sessionData)
        {
            if (isOn)
            {
                _sessionData = sessionData;
                _joinButton.interactable = true;
            }
            else if (sessionData == _sessionData) _joinButton.interactable = false;
        }
    }
}
