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
using UnityEngine.UI;

namespace Digiphy
{
    public class SessionDataView : MonoBehaviour
    {
        [SerializeField] private TextMeshProUGUI _sessionName = null;
        [SerializeField] private Toggle _toggle = null;

        public void ShowSession(string sessionName, int playerCount, Action<bool, string> onToggle, ToggleGroup toggleGroup)
        {
            _sessionName.text = sessionName + " " + playerCount;
            _toggle.onValueChanged.AddListener((isOn) => onToggle.Invoke(isOn, sessionName));
            _toggle.group = toggleGroup;
        }
    }
}
