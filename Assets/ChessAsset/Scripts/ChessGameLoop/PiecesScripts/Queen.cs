namespace ChessMainLoop
{
    public class Queen : Piece
    {
        public override void CreatePath()
        {
            PathManager.CreateDiagonalPath(this);
            PathManager.CreateVerticalPath(this);
        }

        public override bool IsAttackingKing(int row, int column)
        {
            return CheckStateCalculator.IsAttackingKingVertical(row, column, PieceColor) || CheckStateCalculator.IsAttackingKingDiagonal(row, column, PieceColor);
        }

        public override bool CanMove(int row, int column)
        {
            return GameEndCalculator.CanMoveDiagonal(row, column, PieceColor) || GameEndCalculator.CanMoveVertical(row, column, PieceColor);
        }
    }
}