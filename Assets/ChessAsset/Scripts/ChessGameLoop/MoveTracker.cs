using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace ChessMainLoop
{
    public class MoveTracker : MonoBehaviour
    {
        private List<List<Vector2>> _moves;
        private static readonly string[] _files =
        {
            "/save1.json",
            "/save2.json",
            "/save3.json",
            "/save4.json"
        };

        private static MoveTracker _instance;
        public static MoveTracker Instance { get => _instance; }

        private void Awake()
        {
            _instance = this;
        }

        private void Start()
        {
            _moves = new List<List<Vector2>>();
        }

        public void AddMove(int _xOld, int _yOld, int _xNew, int _yNew, int _moveOrder)
        {
            //if (_moves.Count <= _moveOrder)
            //{
            //    _moves.Add(new List<Vector2>());
            //}

            //_moves[_moveOrder].Add(new Vector2(_xOld, _yOld));
            //_moves[_moveOrder].Add(new Vector2(_xNew, _yNew));
        }

        public List<List<Vector2>> Moves()
        {
            return _moves;
        }

        public void ResetMoves()
        {
            _moves = new List<List<Vector2>>();
        }

        public void SaveGame(int _fileIndex)
        {
            string _json = "";
            for (int i = 0; i < _moves.Count; i++) 
            {
                var _myclass=new MovesSerializator(_moves[i], i);
                _json =_json + "\n" + JsonUtility.ToJson(_myclass);            
            }

            File.WriteAllText(Application.persistentDataPath + _files[_fileIndex], _json);


        }

        /// <summary>
        /// Class for json serialization of moves.
        /// </summary>
        public class MovesSerializator
        {
            public List<Vector2> MoveList;
            public int TurnOrder;

            public MovesSerializator(List<Vector2> _moveList, int _turnOrder)
            {
                MoveList = _moveList;
                TurnOrder = _turnOrder;
            }
        }
    }
}
