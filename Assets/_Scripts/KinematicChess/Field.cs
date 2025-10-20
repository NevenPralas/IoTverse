using ChessEnums;
using Digiphy;
using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace KinematicChess
{
    public class Field : MonoBehaviour
    {
        [SerializeField] private List<Piece> _piecesInContact = new List<Piece>();
        private MeshRenderer _renderer;
        private bool _selected;

        public bool IsSelected => _selected;

        public void Init()
        {
            _renderer = GetComponent<MeshRenderer>();
        }

        public void SetInitialPiece(Piece piece)
        {
            _piecesInContact.Add(piece);
        }

        public void Selected()
        {
            _renderer.enabled = true;
            _selected = true;
        }

        public void Unselected()
        {
            _renderer.enabled = false;
            _selected = false;
        }

        private void OnTriggerEnter(Collider other)
        {
            if(other.TryGetComponent(out Piece piece))
            {
                if (!_piecesInContact.Contains(piece)) _piecesInContact.Add(piece);
            }
        }

        private void OnTriggerExit(Collider other)
        {
            if (other.TryGetComponent(out Piece piece))
            {
                if (_piecesInContact.Contains(piece)) _piecesInContact.Remove(piece);
            }
        }

        public bool IsInContact() => _piecesInContact.Count > 0;
    }
}