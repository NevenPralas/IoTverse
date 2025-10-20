using ChessEnums;
using Digiphy;
using System.Collections.Generic;
using UnityEngine;

namespace ChessMainLoop
{
    public class BoardState : SingletonReplaceable<BoardState>
    {
        private Piece[,] _gridState;
        [SerializeField]
        private int _boardSize;
        public int BoardSize { get => _boardSize; }
        [SerializeField]
        private List<Piece> _blackPieces;
        [SerializeField]
        private List<Piece> _whitePieces;
        [SerializeField]
        private Queue<Piece> _promotedPieces;
        public static float Offset = 1.5f;

        private void Awake()
        {
            base.Awake();
            _gridState = new Piece[_boardSize, _boardSize];
            InitializeGrid();
            _promotedPieces = new Queue<Piece>();  
        }

        public void InitializeGrid()
        {
            for(int i = 0; i < _boardSize; i++)
            {
                for(int j = 0; j < _boardSize; j++)
                {
                    _gridState[i, j] = null;
                }
            }

            for (int i = 0; i < _blackPieces.Count; i++) 
            {
                Piece piece = _blackPieces[i];
                if (!piece.gameObject.activeSelf) continue;
                (int Row, int Column) location = piece.Location;
                _gridState[location.Row, location.Column] = piece;
                Vector3 position = new Vector3(location.Row * Offset, 
                    piece.transform.localPosition.y, piece.Location.Column * Offset);
                piece.transform.localPosition = position;
            } 

            for(int i = 0; i < _whitePieces.Count; i++)
            {
                Piece piece = _whitePieces[i];
                if (!piece.gameObject.activeSelf) continue;
                (int Row, int Column) location = piece.Location;
                _gridState[location.Row, location.Column] = piece;
                Vector3 position = new Vector3(location.Row * Offset,
                    piece.transform.localPosition.y, piece.Location.Column * Offset);
                piece.transform.localPosition = position;
            }
        }

        /// <summary>
        /// Retrieves current state of that fied on board
        /// </summary>
        /// <returns>Piece reference of the piece on the field, or null if not occupied</returns>
        public Piece GetField(int row, int column) => _gridState[row, column];

        public void SetField(Piece piece, int newRow, int newColumn)
        {
            _gridState[piece.Location.Row, piece.Location.Column] = null;
            _gridState[newRow, newColumn] = piece;
        }

        public void ClearField(int row, int column)
        {
            _gridState[row, column] = null;
        }

        /// <summary>
        /// Checks if cooridantes are inside board borders
        /// </summary>
        public bool IsInBorders(int row, int column)
        {
            bool check = (row >= 0 && row < _boardSize && column >= 0 && column < _boardSize);
            return check;
        }

        /// <summary>
        /// Mocks the translation of the piece to the target position and check if it would result in check.
        /// </summary>
        /// <returns>Wether translation performed on the piece would result in a check state</returns>
        public SideColor SimulateCheckState(int rowOld, int columnOld, int rowNew, int columnNew)
        {
            Piece missplaced = _gridState[rowNew, columnNew];
            _gridState[rowNew, columnNew] = _gridState[rowOld, columnOld];
            _gridState[rowOld, columnOld] = null;

            SideColor checkSide = CheckStateCalculator.CalculateCheck(_gridState);

            _gridState[rowOld, columnOld] = _gridState[rowNew, columnNew];
            _gridState[rowNew, columnNew] = missplaced;

            return checkSide;
        }

        public SideColor CheckIfGameOver()
        {
            return GameEndCalculator.CheckIfGameEnd(_gridState);
        }

        public void ResetPieces()
        {
            foreach (Piece piece in _blackPieces)
            {
                piece.ResetPiece();
            }
            foreach (Piece piece in _whitePieces)
            {
                piece.ResetPiece();
            }

            while (_promotedPieces.Count > 0)
            {
                Destroy(_promotedPieces.Dequeue());
            }

            InitializeGrid();
        }

        /// <summary>
        /// Replaces pawn being promoted with the selected piece.
        /// </summary>
        public void PromotePawn(Pawn promotingPawn, Piece piece, ChessPieceType pieceIndex)
        {
            _promotedPieces.Enqueue(piece);
            MoveTracker.Instance.AddMove(promotingPawn.Location.Row, promotingPawn.Location.Column, (int)pieceIndex, (int)pieceIndex, GameManager.Instance.TurnCount - 1);
            _gridState[promotingPawn.Location.Row, promotingPawn.Location.Column] = piece;
            piece.PiecePromoted(promotingPawn);
            promotingPawn.gameObject.SetActive(false);
        }
    }
}
