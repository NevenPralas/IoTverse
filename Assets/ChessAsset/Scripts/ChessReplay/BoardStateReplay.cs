using System.Collections.Generic;
using UnityEngine;

namespace ChessReplay
{
    public class BoardStateReplay : MonoBehaviour
    {
        [SerializeField]
        private int _boardSize;
        public int BoardSize { get => _boardSize; }
        private Transform[,] _gridState;
        [SerializeField]
        private GameObject _blackPieces;
        [SerializeField]
        private GameObject _whitePieces;
        public static float Displacement = 1.5f;
        private Dictionary<int, GameObject> _killedDict;
        private Dictionary<int, GameObject[]> _promotedOnesDict;

        #region Private piece prefabs
        [SerializeField]
        private GameObject _whiteQueen;
        [SerializeField]
        private GameObject _blackQueen;
        [SerializeField]
        private GameObject _whiteBishop;
        [SerializeField]
        private GameObject _blackBishop;
        [SerializeField]
        private GameObject _whiteRook;
        [SerializeField]
        private GameObject _blackRook;
        [SerializeField]
        private GameObject _whiteKnight;
        [SerializeField]
        private GameObject _blackKnight;
        #endregion


        private static BoardStateReplay _instance;
        public static BoardStateReplay Instance { get => _instance; }

        private void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Destroy(this.gameObject);
            }
            else
            {
                _instance = this;
            }
        }      

        /// <summary>
        /// Resets grid state and positions of all pieces.
        /// </summary>
        public void InitializeGrid()
        {
            _promotedOnesDict = new Dictionary<int, GameObject[]>();
            _gridState = new Transform[_boardSize, _boardSize];
            _killedDict = new Dictionary<int, GameObject>();

            for (int i = 0; i < _gridState.GetLength(0); i++)
            {
                for (int j = 0; j < _gridState.GetLength(1); j++)
                {
                    _gridState[i, j] = null;
                }
            }

            foreach (Transform child in _blackPieces.transform)
            {
                int x = (int)(child.localPosition.x / Displacement);
                int y = (int)(child.localPosition.z / Displacement);
                _gridState[x, y] = child;
            }
            foreach (Transform child in _whitePieces.transform)
            {
                int x = (int)(child.localPosition.x / Displacement);
                int y = (int)(child.localPosition.z / Displacement);
                _gridState[x, y] = child;
            }
        }

        /// <summary>
        /// Moves piece located on first paramter to location of second parameter. Stores pieces killed or promoted by turn counter parameter.
        /// </summary>
        /// If end position vector is negative, based on the value chooses a piece to replace pawn with for promotion or kills en passanted piece
        public void MovePiece(Vector2 _startPosition, Vector2 _endPosition, int _turnCount)
        {
            if (_gridState == null)
            {
                InitializeGrid();
            }

            //Parameter -1 signifies piece got passanted, and from -2 to -9 resembles index of piece mesh to replace pawn with
            switch (_endPosition.x)
            {
                case -1:
                    _killedDict.Add(_turnCount, _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject);
                    _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject.SetActive(false);
                    _gridState[(int)_startPosition.x, (int)_startPosition.y] = null;
                    return;
                case -2:
                    PromotePawn(_startPosition, _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject, _blackQueen, _turnCount);
                    return;
                case -3:
                    PromotePawn(_startPosition, _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject, _whiteQueen, _turnCount);
                    return;
                case -4:
                    PromotePawn(_startPosition, _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject, _blackRook, _turnCount);
                    return;
                case -5:
                    PromotePawn(_startPosition, _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject, _whiteRook, _turnCount);
                    return;
                case -6:
                    PromotePawn(_startPosition, _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject, _blackBishop, _turnCount);
                    return;
                case -7:
                    PromotePawn(_startPosition, _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject, _whiteBishop, _turnCount);
                    return;
                case -8:
                    PromotePawn(_startPosition, _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject, _blackKnight, _turnCount);
                    return;
                case -9:
                    PromotePawn(_startPosition, _gridState[(int)_startPosition.x, (int)_startPosition.y].gameObject, _whiteKnight, _turnCount);
                    return;
                default:
                    break;
            }

            if(_gridState[(int)_endPosition.x, (int)_endPosition.y] != null)
            {
                _killedDict.Add(_turnCount, _gridState[(int)_endPosition.x, (int)_endPosition.y].gameObject);
                _gridState[(int)_endPosition.x, (int)_endPosition.y].gameObject.SetActive(false);
            }

            _gridState[(int)_endPosition.x, (int)_endPosition.y] = _gridState[(int)_startPosition.x, (int)_startPosition.y];
            _gridState[(int)_startPosition.x, (int)_startPosition.y] = null;
            _gridState[(int)_endPosition.x, (int)_endPosition.y].localPosition = new Vector3(_endPosition.x * Displacement, 
                _gridState[(int)_endPosition.x, (int)_endPosition.y].localPosition.y, _endPosition.y * Displacement);
        }

        public void PromotePawn(Vector3 _endPosition, GameObject _pawn, GameObject _prefab, int _turnCount)
        {
            GameObject _piece = Instantiate(_prefab, _pawn.transform.parent);
            _piece.transform.position = _pawn.transform.position;
            _piece.transform.localScale = _pawn.transform.localScale;
            _gridState[(int)_endPosition.x, (int)_endPosition.y] = _piece.transform;
            GameObject[] _promotedPair =
            {
                _pawn,
                _piece
            };
            _promotedOnesDict.Add(_turnCount, _promotedPair);
            _pawn.SetActive(false);
        }

        /// <summary>
        /// Replays last move by playing it backwards.
        /// </summary>
        public void UndoPiece(Vector2 _startPosition, Vector2 _endPosition, int _turnCount)
        {
            MovePiece(_startPosition, _endPosition, _turnCount);

            if (_killedDict.TryGetValue(_turnCount, out GameObject _killed))
            {
                _killedDict.Remove(_turnCount);
                _killed.SetActive(true);
                _gridState[(int)(_killed.transform.localPosition.x / Displacement), (int)(_killed.transform.localPosition.z / Displacement)] = _killed.transform;
            }
            
            if (_promotedOnesDict.TryGetValue(_turnCount, out GameObject[] _promotionPair))            
            {
                _promotedOnesDict.Remove(_turnCount);
                _promotionPair[0].SetActive(true);
                _promotionPair[0].transform.localPosition = _promotionPair[1].transform.localPosition;
                _gridState[(int)(_promotionPair[0].transform.localPosition.x / Displacement),
                    (int)(_promotionPair[0].transform.localPosition.z / Displacement)] = _promotionPair[0].transform;
                Destroy(_promotionPair[1]);
            }
        }
    }
}