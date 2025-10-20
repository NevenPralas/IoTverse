using Oculus.Interaction;
using System;
using UnityEngine;

namespace ChessMainLoop
{
    public delegate void PathSelect(PathPiece piece);

    public class PathPiece : MonoBehaviour
    {
        [SerializeField] private Grabbable _grabbable;
        [SerializeField] private PathType _type;
        private int _row;
        private int _column;
        private Color _startColor;
        private Renderer _renderer;
        private Piece _assignedPiece = null;
        private Piece _assignedCastle = null;
        private bool _selectedForQuest = false;

        public PathType PathType => _type;
        public (int Row, int Column) Location
        {
            get => (_row, _column);
            set
            {
                _row = value.Row;
                _column = value.Column;
            }
        }

        public Piece AssignedPiece { get => _assignedPiece; set => _assignedPiece = value; }
        public Piece AssignedCastle { get => _assignedCastle; set => _assignedCastle = value; }

        public static event PathSelect PathSelect;


        void OnEnable()
        {
            PieceController.PieceMoved += Disable;
        }
    
        void OnDisable()
        {
            PieceController.PieceMoved -= Disable;
        }

        private void Awake()
        {
            _renderer = GetComponent<Renderer>();
            _startColor = _renderer.material.color;
            _grabbable.WhenPointerEventRaised += ProcessPointerEvent;
        }

        private void OnMouseEnter() => HoverEnter();

        private void OnMouseExit() => HoverEnd();

        private void OnMouseDown() => Selected();

        public void ProcessPointerEvent(PointerEvent evt)
        {
            switch (evt.Type)
            {
                case PointerEventType.Hover:
                    HoverEnter();
                    break;
                case PointerEventType.Unhover:
                    HoverEnd();
                    break;
                case PointerEventType.Select:
                    Selected();
                    break;
                case PointerEventType.Unselect:
                    break;
                case PointerEventType.Move:
                    break;
                case PointerEventType.Cancel:
                    break;
            }
        }

        public void HoverEnter()
        {
            _renderer.material.color = Color.white;
        }

        public void HoverEnd()
        {
            if (_selectedForQuest) _renderer.material.color = Color.green;
            else _renderer.material.color = _startColor;
        }

        /// <summary>
        /// Disables the path gameobject. Also resets all refrences pieces have to it.
        /// </summary>
        private void Disable()
        {
            if (_assignedPiece != null)
            {
                _assignedPiece.AssignedAsEnemy = null;
                _assignedPiece = null;
            }
            else if (_assignedCastle != null)
            {
                _assignedCastle.AssignedAsCastle = null;
                _assignedCastle = null;
            }
            ObjectPool.Instance.RemoveHighlightPath(this);
        }

        public void AssignPiece(Piece _piece)
        {
            _assignedPiece = _piece;
            _assignedPiece.AssignedAsEnemy = this;
        }

        public void AssignCastle(Piece _piece)
        {
            _assignedCastle = _piece;
            _piece.AssignedAsCastle = this;
        }

        /// <summary>
        /// Sets the path as selected target for piece movement.
        /// </summary>
        public void Selected()
        {
            _renderer.material.color = _startColor;
            _selectedForQuest = false;
            PathSelect?.Invoke(this);
        }

        internal void SelectedForQuest()
        {
            _renderer = GetComponent<Renderer>();
            _selectedForQuest = true;
            _renderer.material.color = Color.green;
        }
    }
}
