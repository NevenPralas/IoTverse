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
        private List<SessionDataView> _sessions = new List<SessionDataView>();
        private System.Random _rand = new System.Random();
        private string _sessionData;

        private void Awake()
        {
            _refreshButton.onClick.AddListener(UpdateSessionList);
            _joinButton.onClick.AddListener(() => FusionConnection.Instance.JoinSession(_sessionData));
            _joinButton.interactable = false;
            _createRoomButton.onClick.AddListener(() =>
            {
                string numbers = "";

                for (int i = 0; i < numbers.Length; i++)
                {
                    numbers += _rand.Next(0, 100);
                }
                FusionConnection.Instance.CreateSession(_userType + "[" + numbers + "]");
            });
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
