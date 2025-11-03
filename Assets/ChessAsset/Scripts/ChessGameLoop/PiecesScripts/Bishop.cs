namespace ChessMainLoop
{
    public class Bishop : Piece
    {
        public override void CreatePath()
        {
            PathManager.CreateDiagonalPath(this);
        }

        public override bool IsAttackingKing(int row, int column)
        {
            return CheckStateCalculator.IsAttackingKingDiagonal(row, column, PieceColor);
        }

        public override bool CanMove(int row, int column)
        {
            return GameEndCalculator.CanMoveDiagonal(row, column, PieceColor);
        }
    }
}