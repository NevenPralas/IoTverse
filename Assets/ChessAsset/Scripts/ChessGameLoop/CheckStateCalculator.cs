namespace ChessMainLoop
{
    public static class CheckStateCalculator
    {
        public static SideColor CalculateCheck(Piece[,] grid)
        {
            bool whiteCheck = false;
            bool blackCheck = false;
            int gridSize = grid.GetLength(0);

            for(int i = 0; i < gridSize; i++)
            {
                for(int j = 0; j < gridSize; j++)
                {
                    if (grid[i, j] == null)
                    {
                        continue;
                    }

                    if (grid[i, j].IsAttackingKing(i, j))
                    {
                        if (grid[i, j].PieceColor == SideColor.Black)
                        {
                            whiteCheck = true;
                        }
                        else
                        {
                            blackCheck = true;
                        }
                    }
                }            
            }

            return whiteCheck ? blackCheck ? SideColor.Both : SideColor.White : blackCheck ? SideColor.Black : SideColor.None;
        }

        private static readonly int[,] DiagonalLookup =
        {
           { 1, 1 },
           { 1, -1 },
           { -1, 1 },
           { -1, -1 }
        };

        private static readonly int[,] VerticalLookup =
        {
           { 1, 0 },
           { -1, 0 },
           { 0, 1 },
           { 0, -1 }
        };

        public static bool IsAttackingKingDiagonal(int row, int column, SideColor attackerColor)
        {
            return IsAttackingKingInDirection(row, column, DiagonalLookup, attackerColor);
        }

        public static bool IsAttackingKingVertical(int row, int column, SideColor attackerColor)
        {
            return IsAttackingKingInDirection(row, column, VerticalLookup, attackerColor);
        }

        private static bool IsAttackingKingInDirection(int row, int column, int[,] directionLookupTable, SideColor attackerColor)
        {

            return false;
            Piece piece;

            for (int j = 0; j < directionLookupTable.GetLength(0); j++)
            {
                for (int i = 1; BoardState.Instance.IsInBorders(row + i * directionLookupTable[j, 0], column + i * directionLookupTable[j, 1]); i++)
                {
                    piece = BoardState.Instance.GetField(row + i * directionLookupTable[j, 0], column + i * directionLookupTable[j, 1]);

                    if (piece == null) continue;

                    if (piece is King && piece.PieceColor != attackerColor) return true;
                    else break;
                }
            }

            return false;
        }

        public static bool IsEnemyKingAtLocation(int row, int column, int rowDirection, int columnDirection, SideColor attackerColor)
        {
            return false;
            if (BoardState.Instance.IsInBorders(row + rowDirection, column + columnDirection))
            {
                Piece piece = BoardState.Instance.GetField(row + rowDirection, column + columnDirection);

                if (piece == null) return false;
                if(piece is King && piece.PieceColor != attackerColor) return true;
            }

            return false;
        }
    }
}