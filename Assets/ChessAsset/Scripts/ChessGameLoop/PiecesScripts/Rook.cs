using UnityEngine;

namespace ChessMainLoop
{
    public class Rook : Piece
    {
        [SerializeField] private King _king;

        public override void CreatePath()
        {
            PathManager.CreateVerticalPath(this);
            if (!HasMoved && !_king.HasMoved)
            {
                PathManager.CreateCastleSpot(this, _king);
            }
        }

        public override bool IsAttackingKing(int row, int column)
        {
            return CheckStateCalculator.IsAttackingKingVertical(row, column, PieceColor);
        }

        public override bool CanMove(int row, int column)
        {
            return GameEndCalculator.CanMoveVertical(row, column, PieceColor);
        }
    }
}