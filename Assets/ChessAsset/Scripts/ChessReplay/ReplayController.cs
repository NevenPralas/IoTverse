using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace ChessReplay
{ 
    public class ReplayController : MonoBehaviour
    {
        private List<List<Vector2>> _moveList;
        private int _turnCount;
        private IEnumerator _automaticTurns;
        [SerializeField]
        private float _turnSpeed = 1f;
        public float TurnSpeed { get => _turnSpeed; set => _turnSpeed = value; }
        [SerializeField]
        private AudioSource _moveSound;
        private float _timeSinceLeft;
        private float _timeSinceRight;

        /// <summary>
        /// Loads moveset data from file selected by index parameter and starts autoplay.
        /// </summary>
        public void Initialize(int _fileIndex)
        {
            _moveList = DataLoader.LoadData(_fileIndex);
            _automaticTurns = AutomaticTurns();
            _turnCount = 0;
            StartAutoPlay();
        }

        private void Update()
        {
            if (Input.GetKey(KeyCode.LeftArrow))
            {
                if (Input.GetKeyDown(KeyCode.LeftArrow))
                {
                    LastTurn();
                    _timeSinceLeft = Time.time;
                }
                else if (Time.time - _timeSinceLeft > _turnSpeed)
                {
                    LastTurn();
                    _timeSinceLeft = Time.time;
                }
            }

            if (Input.GetKey(KeyCode.RightArrow))
            {
                if (Input.GetKeyDown(KeyCode.RightArrow))
                {
                    NextTurn();
                    _timeSinceRight = Time.time;
                }
                else if (Time.time - _timeSinceRight > _turnSpeed)
                {
                    NextTurn();
                    _timeSinceRight = Time.time;
                }
            }

            if (Input.GetKeyDown(KeyCode.Space))
            {
                StartAutoPlay();
            }
        }

        public void StartAutoPlay()
        {
            StartCoroutine(_automaticTurns);
        }

        /// <summary>
        /// Stops autoplay of turns and plays past turn if it exists.
        /// </summary>
        public void LastTurn()
        {
            if (_turnCount > 0)
            {
                StopAutoplay();

                _turnCount--;
                List<Vector2> _turnMoves = _moveList[_turnCount];
                BoardStateReplay.Instance.UndoPiece(_turnMoves[1], _turnMoves[0], _turnCount);
                if (_turnMoves.Count > 2 && _turnMoves[3].x >= 0)
                {
                    BoardStateReplay.Instance.UndoPiece(_turnMoves[3], _turnMoves[2], _turnCount);
                }
                _moveSound.Play();
            }
        }

        /// <summary>
        /// Stops autoplay of turns and plays folowing turn if it exists.
        /// </summary>
        public void NextTurn()
        {
            if (_turnCount < _moveList.Count)
            {
                StopAutoplay();

                List<Vector2> _turnMoves = _moveList[_turnCount];

                BoardStateReplay.Instance.MovePiece(_turnMoves[0], _turnMoves[1], _turnCount);
                if (_turnMoves.Count > 2)
                {
                    BoardStateReplay.Instance.MovePiece(_turnMoves[2], _turnMoves[3], _turnCount);
                }
                _turnCount++;
                _moveSound.Play();
            }
        }

        private IEnumerator AutomaticTurns()
        {
            while (_turnCount < _moveList.Count)
            {
                List<Vector2> _turnMoves = _moveList[_turnCount];

                BoardStateReplay.Instance.MovePiece(_turnMoves[0], _turnMoves[1], _turnCount);
                if (_turnMoves.Count > 2)
                {
                    BoardStateReplay.Instance.MovePiece(_turnMoves[2], _turnMoves[3], _turnCount);
                }
                _moveSound.Play();
                _turnCount++;

                yield return new WaitForSeconds(_turnSpeed);
            }
        }

        public void StopAutoplay()
        {
            StopCoroutine(_automaticTurns);
        }
    }
}