using System.Collections.Generic;
using UnityEngine;

namespace ChessMainLoop
{
    public class King : Piece
    {
        /// <summary>
        /// Lookup table containing knight movement directions
        /// </summary>
        private static readonly int[,] LookupMoves =
        {
           { 1, 1 },
           { 1, 0 },
           { 1, -1 },
           { 0, -1 },
           { -1, -1 },
           { -1, 0 },
           { -1, 1 },
           { 0, 1 }

        };

        [SerializeField] private List<Rook> _rooks;

        public override void CreatePath()
        {
            bool allowed;

            //Checks surrounding of each position from lookup table for nearby enemy king. If there is no king present tries to create path on that position.
            for (int i = 0; i < LookupMoves.GetLength(0); i++)
            {
                allowed = true;
                int newRow = _row + LookupMoves[i, 0];
                int newColumn = _column + LookupMoves[i, 1];
                for (int j = 0; j < LookupMoves.GetLength(0); j++)
                {
                    if (!BoardState.Instance.IsInBorders(newRow + LookupMoves[j, 0], newColumn + LookupMoves[j, 1])) continue;

                    Piece piece = BoardState.Instance.GetField(newRow + LookupMoves[j, 0], newColumn + LookupMoves[j, 1]);
                    if (piece is King && piece != this)
                    {
                        allowed = false;
                    }
                }

                if (allowed)
                {
                    PathManager.CreatePathInSpotDirection(this, LookupMoves[i, 0], LookupMoves[i, 1]);
                }
            }

            if (HasMoved) return;

            foreach (Piece rook in _rooks)
            {
                if (!rook.HasMoved) PathManager.CreateCastleSpot(this, rook);
            }
        }

        public override bool IsAttackingKing(int row, int column)
        {
            return false;
        }

        public override bool CanMove(int row, int column)
        {
            for (int i = 0; i < LookupMoves.GetLength(0); i++)
            {
                if (GameEndCalculator.CanMoveToSpot(row, column, LookupMoves[i, 0], LookupMoves[i, 1], PieceColor))
                {
                    return true;
                }
            }

            return false;
        }
    }
}