using Digiphy;
using Oculus.Interaction;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace ChessMainLoop
{
    public delegate void Selected(Piece self);
    /// <summary>
    /// Represents side colors of players.
    /// </summary>
    public enum SideColor
    {
        White,
        Black,
        None,
        Both
    }

    public abstract class Piece : MonoBehaviour
    {
        #region Private fields and corresponding public properties
        [SerializeField] private SideColor _pieceColor;
        [SerializeField] protected Renderer _renderer;
        [SerializeField] private Grabbable _grabbable;
        [SerializeField] protected int _row;
        [SerializeField] protected int _column;
        [SerializeField] private bool _isSnappingPiece;
        [SerializeField] private Transform _middle;
        private bool _isActive = false;
        private Color _startColor;
        private (int Row, int Column) _startLocation;
        private bool _hasMoved = false;
        private PathPiece _assignedAsCastle = null;
        private PathPiece _assignedAsEnemy = null;
        private Pawn _wasPawn = null;
        private PathPiece _howeredSnappingPath = null;
        private Piece _howeredSnappingPiece = null;
        protected bool _selectedForQuest = false;
        protected int _questDistance = 0;
        private float _originalY = 0;
        private Quaternion _originalRotation = Quaternion.identity;

        public bool IsSnappinPiece => _isSnappingPiece;
        public SideColor PieceColor { get => _pieceColor; }
        public (int Row, int Column) Location => (_row, _column); 
        public bool IsActive { get => _isActive; set { _isActive = false; _renderer.material.color = _startColor; } }
        public bool HasMoved { get => _hasMoved; set => _hasMoved = value; }
        public Transform visual => _grabbable.transform;
        public PathPiece AssignedAsEnemy 
        { 
            get => _assignedAsEnemy; 
            set 
            {
                _assignedAsEnemy = value; 
                if(value != null && !_isSnappingPiece)
                {
                    _grabbable.gameObject.SetActive(true);
                }
                else if(_pieceColor != GameManager.Instance.LocalPlayer)
                {
                    _grabbable.gameObject.SetActive(false);
                }
            }
        }
        public PathPiece AssignedAsCastle { get => _assignedAsCastle; set { _assignedAsCastle = value; _renderer.material.color = _startColor; } }
        public Pawn WasPawn { get => _wasPawn; set => _wasPawn = value; }
        public float OriginalY => _originalY;
        #endregion

        private void Start()
        {
            _originalY = transform.localPosition.y;
            _originalRotation = transform.localRotation;
        }

        public static event Selected Selected;

        public abstract void CreatePath();
        public abstract bool IsAttackingKing(int row, int column);
        public abstract bool CanMove(int row, int column);

        private void Awake()
        {
            _grabbable.WhenPointerEventRaised += ProcessPointerEvent;
            _startLocation = (_row, _column);
            _startColor = _renderer.material.color;
        }

        private void Update()
        {
            if (!_isActive) return;
            _howeredSnappingPath?.HoverEnd();
            _howeredSnappingPath = null;
            _howeredSnappingPiece?.HoverEnd();
            _howeredSnappingPiece = null;
            RaycastHit[] raycastHits = Physics.RaycastAll(_grabbable.transform.position, Vector3.down);
            foreach (RaycastHit hit in raycastHits)
            {
                if(hit.collider.TryGetComponent(out PathPiece path))
                {
                    if (path == _howeredSnappingPath) return;
                    path.HoverEnter();
                    _howeredSnappingPath = path;
                }
                else if(hit.collider.TryGetComponent(out Piece piece))
                {
                    if (piece == this) continue;
                    if (piece == _howeredSnappingPiece) return;
                    piece.PieceHowered();
                    _howeredSnappingPiece = piece;
                }
            }
        }

        private void GrabbEnd()
        {
            Vector3 raycastPosition = _grabbable.transform.position;
            transform.localPosition = new Vector3(_row * BoardState.Offset, _originalY, _column * BoardState.Offset);
            transform.localRotation = _originalRotation;

            _howeredSnappingPath?.HoverEnd();
            _howeredSnappingPath = null;
            _howeredSnappingPiece?.HoverEnd();
            _howeredSnappingPiece = null;
            if(_isSnappingPiece) SnappingPointer.Instance.Unset();

            PathPiece selectedPath = null;
            Piece selectedPiece = null;

            RaycastHit[] raycastHits = Physics.RaycastAll(raycastPosition, Vector3.down);
            foreach (RaycastHit hit in raycastHits)
            {
                if (hit.collider.TryGetComponent(out PathPiece path))
                {
                    selectedPath = path;
                    break;
                }
                else if (hit.collider.TryGetComponent(out Piece piece))
                {
                    if (piece == this) continue;
                    selectedPiece = piece;
                    break;
                }
            }

            transform.localPosition = new Vector3(_row * BoardState.Offset, transform.localPosition.y, _column * BoardState.Offset);
            
            selectedPath?.Selected();
            selectedPiece?.PieceSelected();
        }

        private void OnMouseEnter() => PieceHowered();

        private void OnMouseExit() => HoverEnd();

        private void OnMouseDown() => PieceSelected();

        public void ProcessPointerEvent(PointerEvent evt)
        {
            switch (evt.Type)
            {
                case PointerEventType.Hover:
                    PieceHowered();
                    break;
                case PointerEventType.Unhover:
                    HoverEnd();
                    break;
                case PointerEventType.Select:
                    PieceSelected();
                    break;
                case PointerEventType.Unselect:
                    if (_isSnappingPiece) GrabbEnd();
                    break;
                case PointerEventType.Move:
                    break;
                case PointerEventType.Cancel:
                    break;
            }
        }        

        //If its turn players piece sets it as selected and sets path pieces for it. If the piece is target of enemy or castle calls select method of path object this piece is assing to.
        public void PieceSelected()
        {
            if (_isActive == false && GameManager.Instance.TurnPlayer == _pieceColor && _assignedAsCastle == false 
                && GameManager.Instance.IsPieceMoving == false && GameManager.Instance.IsPromotingPawn == false && GameManager.Instance.IsPlayerTurn) 
            {
                if (PieceController.Instance.PieceSelected(this))
                {
                    _isActive = true;
                    if (_isSnappingPiece) SnappingPointer.Instance.SetFigure(transform);
                    CreatePath();
                    _renderer.material.color = Color.yellow;
                }
            }
            else if (_assignedAsEnemy)
            {
                _assignedAsEnemy.Selected();
            }
            else if (_assignedAsCastle)
            {
                _assignedAsCastle.Selected();
            }
            else if(_isSnappingPiece && _isActive)
            {
                SnappingPointer.Instance.SetFigure(transform);
            }
        }

        public void PieceHowered()
        {
            if (GameManager.Instance.TurnPlayer == _pieceColor && PieceController.Instance.AnyActive == false && GameManager.Instance.IsPlayerTurn)
            {
                _renderer.material.color = Color.yellow;
            }
            else if (_assignedAsEnemy)
            {
                _renderer.material.color = Color.red;
            }
            else if (_assignedAsCastle)
            {
                _renderer.material.color = Color.yellow;
            }
        }

        public void HoverEnd()
        {
            if ((_isActive == false) || _assignedAsEnemy || _assignedAsCastle)
            {
                if (_selectedForQuest)
                {
                    _renderer.material.color = Color.green;
                }
                else
                {
                    _renderer.material.color = _startColor;
                }
            }

        }

        public void Die()
        {
            if (BoardState.Instance.GetField(_row, _column) == this)
            {
                BoardState.Instance.ClearField(_row, _column);
            }
            ObjectPool.Instance.AddPiece(this);
        }

        public void ResetPiece()
        {
            _row = _startLocation.Row; 
            _column = _startLocation.Column;
            _renderer.material.color = _startColor;
            _wasPawn = null;
            _hasMoved = false;
        }

        public virtual void Move(int newRow, int newColumn)
        {
            MoveTracker.Instance.AddMove(_row, _column, newRow, newColumn, GameManager.Instance.TurnCount);
            if (this is Pawn && GameManager.Instance.Passantable)
            {
                int _direction = PieceColor == SideColor.Black ? 1 : -1;

                if (_column == GameManager.Instance.Passantable.Location.Column)
                {
                    if (_row  == GameManager.Instance.Passantable.Location.Row && _row != newRow)
                    {
                        MoveTracker.Instance.AddMove(newRow - _direction, newColumn, -1, -1, GameManager.Instance.TurnCount);
                    }
                }
            }

            BoardState.Instance.SetField(this, newRow, newColumn);
            _row = newRow;
            _column = newColumn;

            _hasMoved = true;
            GameManager.Instance.Passantable = null;
        }

        public void PiecePromoted(Pawn promotingPawn)
        {
            WasPawn = promotingPawn;
            HasMoved = true;
            _row = promotingPawn.Location.Row;
            _column = promotingPawn.Location.Column;
            transform.localPosition = promotingPawn.transform.localPosition;
            transform.localPosition = new Vector3(transform.localPosition.x, 0, transform.localPosition.z);
        }
    }
}
