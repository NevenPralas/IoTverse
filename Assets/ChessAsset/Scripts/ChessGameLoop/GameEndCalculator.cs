namespace ChessMainLoop
{
    /// <summary>
    /// Contains methods for checking if state on the board resembles check.
    /// </summary>
    public static class GameEndCalculator 
    {
        /// <summary>
        /// Checks if turn player has any available turns left.
        /// </summary>
        /// <returns>Color of winner side</returns>
        public static SideColor CheckIfGameEnd(Piece[,] grid)
        {
            SideColor _turnPlayer = GameManager.Instance.TurnPlayer;
            int boardSize = grid.GetLength(0);

            /* Checks each field on the board, and if the field contains a piece with the same color as turn player 
             * checks if the piece can move. For the first such piece that can move returns indication that there isn't a checkmate, 
             * and if no such piece can be found returns color of winner player.
             */
            for (int i = 0; i < boardSize; i++)
            {
                for (int j = 0; j < boardSize; j++)
                {
                    if (grid[i, j] == null) continue;

                    if (grid[i, j].PieceColor == _turnPlayer && grid[i, j].CanMove(i, j))
                    {
                        return SideColor.None;
                    }
                }
            }

            if (GameManager.Instance.CheckedSide == _turnPlayer)
            {
                return _turnPlayer == SideColor.Black ? SideColor.White : SideColor.Black;
            }

            return SideColor.Both;
        }

        #region Lookup tables for movement directions
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
        #endregion

        public static bool CanMoveDiagonal(int row, int column, SideColor _attackerColor)
        {
            return CanMove(row, column, DiagonalLookup, _attackerColor);
        }

        public static bool CanMoveVertical(int row, int column, SideColor _attackerColor)
        {
            return CanMove(row, column, VerticalLookup, _attackerColor);
        }

        private static bool CanMove(int row, int column, int[,] _lookupTable, SideColor _attackerColor)
        {
            Piece _piece;

            //Checks in each direction from lookup table if there is an available spot to move which doesnt result in check for piece player side
            for (int j = 0; j < _lookupTable.GetLength(0); j++)
            {
                for (int i = 1; BoardState.Instance.IsInBorders(row + i * _lookupTable[j, 0], column + i * _lookupTable[j, 1]); i++)
                {
                    SideColor _checkSide = BoardState.Instance.SimulateCheckState(row, column, row + i * _lookupTable[j, 0], column + i * _lookupTable[j, 1]);
                    if (_checkSide == _attackerColor || _checkSide == SideColor.Both) break;

                    _piece = BoardState.Instance.GetField(row + i * _lookupTable[j, 0], column + i * _lookupTable[j, 1]);

                    if (_piece == null) return true;
                    else if (_piece.PieceColor != _attackerColor) return true;
                    else break;
                }
            }

            return false;
        }

        public static bool CanMoveToSpot(int row, int column, int rowDirection, int columnDirection, SideColor _attackerColor)
        {
            Piece _piece;

            //Checks if the target location is available for moving and doesnt result in a check
            if (!BoardState.Instance.IsInBorders(row + rowDirection, column + columnDirection)) return false;

            SideColor _checkSide = BoardState.Instance.SimulateCheckState(row, column, row + rowDirection, column + columnDirection);
            _piece = BoardState.Instance.GetField(row + rowDirection, column + columnDirection);

            if (_checkSide == _attackerColor || _checkSide == SideColor.Both) return false;

            if (_piece == null) return true;
            else return _piece.PieceColor != _attackerColor;
        }
    }
}