using ChessMainLoop;
using Fusion;
using System;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace Digiphy
{
    public class TestingSetupManager : Singleton<TestingSetupManager>
    {
        [SerializeField] private AnchorPlacement _anchorPlacement;
        [Header("Menus")]
        [SerializeField] private GameObject _testMenuControllsAndSize;
        [SerializeField] private GameObject _testMenuCouch;
        [Header("Menu Toggles")]
        [SerializeField] private Toggle _testMenuControllsAndSizeToggle;
        [SerializeField] private Toggle _testMenuCouchToggle;
        [Header("Setup Buttons")]
        [SerializeField] private Button _buttonSmallKinematic;
        [SerializeField] private Button _buttonSmallLogic;
        [SerializeField] private Button _buttonSmallSnapping;
        [SerializeField] private Button _buttonLargeKinematic;
        [SerializeField] private Button _buttonLargeLogic;
        [SerializeField] private Button _buttonLargeSnapping;
        [SerializeField] private Button _buttonChessInBetween;
        [SerializeField] private Button _buttonChessInFront;
        [Header("Chess Prefabs")]
        [SerializeField] private GameObject _chessKinematic;
        [SerializeField] private GameObject _chessLogic;
        [SerializeField] private GameObject _chessSnapping;
        [SerializeField] private GameObject _chessLargeKinematic;
        [SerializeField] private GameObject _chessLargeLogic;
        [SerializeField] private GameObject _chessLargeSnapping;
        [Header("Additional toggles")]
        [SerializeField] private Toggle _chessMoveableToggle;
        [SerializeField] private Button _resetAnchorButton;
        private Transform _chessLocationTable;
        private Transform _chessLocationCouchInFront;
        private Transform _chessLocationCouchInBetween;
        private Transform _chessLocationLarge;
        private NetworkRunner _runner;
        private ChessInstanceController _currentChess;

        public bool IsChessMoveable => _chessMoveableToggle.isOn;
        public Action<bool> ChessMoveableChanged;

        private void Start()
        {
            if (_resetAnchorButton != null) _resetAnchorButton.onClick.AddListener(() => _anchorPlacement.Enabled = true);
        }

        public void Init(NetworkRunner runer)
        {
            _runner = runer;
            _testMenuControllsAndSizeToggle.onValueChanged.AddListener(value => 
                _testMenuControllsAndSize.SetActive(value));
            _testMenuCouchToggle.onValueChanged.AddListener(value =>
                _testMenuCouch.SetActive(value));

            _buttonSmallKinematic.onClick.AddListener(() => 
                CreateChess(_chessKinematic, _chessLocationTable));
            _buttonSmallLogic.onClick.AddListener(() => 
                CreateChess(_chessLogic, _chessLocationTable));
            _buttonSmallSnapping.onClick.AddListener(() => 
                CreateChess(_chessSnapping, _chessLocationTable));
            _buttonLargeKinematic.onClick.AddListener(() =>
                CreateChess(_chessLargeKinematic, _chessLocationLarge));
            _buttonLargeLogic.onClick.AddListener(() =>
                CreateChess(_chessLargeLogic, _chessLocationLarge));
            _buttonLargeSnapping.onClick.AddListener(() =>
                CreateChess(_chessLargeSnapping, _chessLocationLarge));
            _buttonChessInBetween.onClick.AddListener(() =>
                CreateChess(_chessLogic, _chessLocationCouchInBetween));
            _buttonChessInFront.onClick.AddListener(() =>
                CreateChess(_chessLogic, _chessLocationCouchInFront));

            _chessMoveableToggle.onValueChanged.AddListener(value => ChessMoveableChanged.Invoke(value));
        }

        public void SetChessLocations(Transform chessTable, Transform chessCouchInFront, 
            Transform chessCouchInBetween, Transform chessLarge)
        {
            _chessLocationTable = chessTable;
            _chessLocationCouchInFront = chessCouchInFront;
            _chessLocationCouchInBetween = chessCouchInBetween;
            _chessLocationLarge = chessLarge;
        }

        public void CreateChess(GameObject chessPrefab, Transform transform)
        {
            if (_runner.ActivePlayers.Count() < 2) return;
            if (_currentChess)
            {
                _currentChess.RPC_RemoveChess();
            }
            NetworkObject chess = _runner.Spawn(chessPrefab, transform.position, transform.rotation);
        }

        public void ChessCreated(NetworkObject chess)
        {
            _currentChess = chess.GetComponent<ChessInstanceController>();
        }

        public void ChessDeleted(NetworkObject chess)
        {
            if (_currentChess == chess) _currentChess = null;
        }

        public void PlayerDisconnected(NetworkRunner runner, PlayerRef player)
        {
            _currentChess?.RPC_RemoveChess();
        }
    }
}
