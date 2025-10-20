using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace ChessReplay
{
    public static class DataLoader
    {
        private static readonly string[] _files =
        {
            "/save1.json",
            "/save2.json",
            "/save3.json",
            "/save4.json"
        };

        /// <summary>
        /// Loads moveset data from file selected by index paramter
        /// </summary>
        /// <returns>Moveset data</returns>
        public static List<List<Vector2>> LoadData(int _fileIndex)
        {
            List<List<Vector2>> _moves = new List<List<Vector2>>();

            if (File.Exists(Application.persistentDataPath + _files[_fileIndex]))
            {
                string[] _json = File.ReadAllText(Application.persistentDataPath + _files[_fileIndex]).Split("\n");
                MovesSerializator _moveInstance;

                foreach (string _turn in _json)
                {
                    if (_turn.Length == 0)
                    {
                        continue;
                    }
                    _moveInstance = JsonUtility.FromJson<MovesSerializator>(_turn);
                    _moves.Add(_moveInstance.MoveList);
                }
            }
        
            return _moves;
        }

        /// <summary>
        /// Class for json data serialization of moves.
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
