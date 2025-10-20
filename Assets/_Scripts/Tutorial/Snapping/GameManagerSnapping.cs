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
    public class GameManagerSnapping : Singleton<GameManagerSnapping>
    {
        [SerializeField] private AudioSource _moveSound;
        [SerializeField] private PieceSnapping _piece;
        [SerializeField] private List<PathSnapping> _pathsFirst;
        [SerializeField] private List<PathSnapping> _pathsSecond;
        [SerializeField] private PathSnapping _firstActive;
        [SerializeField] private PathSnapping _secondActive;
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
            _piece.transform.localPosition = target;
            _piece.Moving();
            _moveSound.Play();
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