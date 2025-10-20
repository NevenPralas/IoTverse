using ChessEnums;
using ChessMainLoop;
using Digiphy;
using Fusion;
using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace Tutorial
{
    public class GameManagerLogic : Singleton<GameManagerLogic>
    {
        [SerializeField] private PieceLogic _piece;
        [SerializeField] private List<PathLogic> _pathsFirst;
        [SerializeField] private List<PathLogic> _pathsSecond;
        [SerializeField] private PathLogic _firstActive;
        [SerializeField] private PathLogic _secondActive;
        private bool _isSecond;

        internal void PathSelected()
        {
            if (_isSecond) 
                foreach (var item in _pathsSecond)
                {
                    item.gameObject.SetActive(false);
                }
            else
                foreach (var item in _pathsFirst)
                {
                    item.gameObject.SetActive(false);
                }

            if (_isSecond) _secondActive.gameObject.SetActive(false);
            else _firstActive.gameObject.SetActive(false);


            Vector3 target;
            if(_isSecond) target = _secondActive.transform.localPosition;
            else target = _firstActive.transform.localPosition;
            _isSecond = !_isSecond;

            target.y = _piece.transform.localPosition.y;
            _piece.Moving();
            AnimationManagerLogic.Instance.MovePiece(_piece, target);
        }

        public void PieceMoved()
        {
            _piece.Moved();
        }

        internal void PieceSelected()
        {
            if (_isSecond)
                foreach (var item in _pathsSecond)
                {
                    item.gameObject.SetActive(true);
                }
            else
                foreach (var item in _pathsFirst)
                {
                    item.gameObject.SetActive(true);
                }

            if (_isSecond) _secondActive.gameObject.SetActive(true);
            else _firstActive.gameObject.SetActive(true);
        }
    }
}