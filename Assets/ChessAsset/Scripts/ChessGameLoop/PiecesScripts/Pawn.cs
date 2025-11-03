using UnityEngine;

namespace ChessMainLoop
{
    public class Pawn : Piece
    {
        public override void CreatePath()
        {
            int direction = PieceColor == SideColor.Black ? 1 : -1;

            //Following two sections check if there are enemy pieces diagonally in looking direction of the pawn, and if there are creates attack path under them
            CreateAttackSpace(direction, 1);
            CreateAttackSpace(direction, -1);

            //Following two sections check left and right of the pawn for enemy pawns, and if they are passantable creates attack field behind them
            CreatePassantSpace(direction, 1);
            CreatePassantSpace(direction, -1);

            //Following sections check if path forward in the direction pawn is facing is empty for one and two spaces, and if they are creates walk path them.
            if (!BoardState.Instance.IsInBorders(_row + direction, _column)) return;
            Piece piece = BoardState.Instance.GetField(_row + direction, _column);
            if (piece != null) return;

            PathManager.CreatePathInSpotDirection(this, direction, 0);

            if (!BoardState.Instance.IsInBorders(_row + direction * 2, _column)) return;
            piece = BoardState.Instance.GetField(_row + direction * 2, _column);
            if (HasMoved || piece != null) return;

            PathManager.CreatePathInSpotDirection(this, direction * 2, 0);
        }

        private void CreateAttackSpace(int rowDirection, int columnDirection)
        {
            if (!BoardState.Instance.IsInBorders(_row + rowDirection, _column + columnDirection) == true) return;
            Piece piece = BoardState.Instance.GetField(_row + rowDirection, _column + columnDirection);
            if (piece != null && piece.PieceColor != PieceColor)
            {
                PathManager.CreatePathInSpotDirection(this, rowDirection, columnDirection);
            }
        }

        private void CreatePassantSpace(int rowDirection, int columnDirection)
        {
            if (!BoardState.Instance.IsInBorders(_row, _column + columnDirection) == true) return;
            Piece piece = BoardState.Instance.GetField(_row, _column + columnDirection);
            if (piece != null && piece.PieceColor != PieceColor && piece == GameManager.Instance.Passantable)
            {
                PathManager.CreatePassantSpot(piece, _row + rowDirection, _column + columnDirection);
            }
        }

        /// <summary>
        /// Adds checks for making the piece passantable if it moved for two sapces and promoting the pawn if it reached the end of the board to Move method of base class.
        /// </summary>
        public override void Move(int newRow, int newColumn)
        {
            int oldRow = _row;

            base.Move(newRow, newColumn);

            if (Mathf.Abs(oldRow - newRow) == 2)
            {
                GameManager.Instance.Passantable = this;
            }

            if (newRow == 0 || newRow == BoardState.Instance.BoardSize - 1)
            {
                GameManager.Instance.PawnPromoting(this);
            }
        }

        public override bool IsAttackingKing(int row, int column)
        {
            int _direction = PieceColor == SideColor.Black ? 1 : -1;

            if (CheckStateCalculator.IsEnemyKingAtLocation(row, column, _direction, 1, PieceColor))
            {
                return true;
            }

            if (CheckStateCalculator.IsEnemyKingAtLocation(row, column, _direction, -1, PieceColor))
            {
                return true;
            }

            return false;
        }

        public override bool CanMove(int row, int column)
        {
            int _direction = PieceColor == SideColor.Black ? 1 : -1;

            //Following two sections perform checks if there are attackable units diagonally in looking direction of the pawn, and if moving to them would not resolve in a check for turn player
            if (BoardState.Instance.IsInBorders(row + _direction, column + 1))
            {
                Piece piece = BoardState.Instance.GetField(row + _direction, column + 1);
                if (piece != null && piece.PieceColor != PieceColor)
                {
                    if (GameEndCalculator.CanMoveToSpot(row, column, _direction, 1, PieceColor))
                    {
                        return true;
                    }
                }
            }

            if (BoardState.Instance.IsInBorders(row + _direction, column - 1))
            {
                Piece piece = BoardState.Instance.GetField(row + _direction, column - 1);
                if (piece != null && piece.PieceColor != PieceColor)
                {
                    if (GameEndCalculator.CanMoveToSpot(row, column, _direction, -1, PieceColor))
                    {
                        return true;
                    }
                }
            }

            //Following sections check if one in looking direction of the pawn is awailable for moving to
            if (!BoardState.Instance.IsInBorders(row + _direction, column)) return false;
            if (BoardState.Instance.GetField(row + _direction, column) != null) return false;

            if (GameEndCalculator.CanMoveToSpot(row, column, _direction, 0, PieceColor)) return true;

            return false;
        }
    }
}
