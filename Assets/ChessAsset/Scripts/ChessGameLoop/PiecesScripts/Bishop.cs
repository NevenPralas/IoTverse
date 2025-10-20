using System.Collections.Generic;
using UnityEngine;

namespace ChessMainLoop
{
    public class Bishop : Piece
    {
        public override void CreatePath()
        {
            List<PathPiece> allPaths = PathManager.CreateDiagonalPath(this);
            List<PathPiece> availablePaths = new List<PathPiece>();

            foreach (PathPiece path in allPaths)
            {
                if (Mathf.Abs(path.Location.Row - Location.Row) <= _questDistance && Mathf.Abs(path.Location.Column - Location.Column) <= _questDistance
                    && (Mathf.Abs(path.Location.Row - Location.Row) == _questDistance || Mathf.Abs(path.Location.Column - Location.Column) == _questDistance))
                {
                    availablePaths.Add(path);
                }
            }

            if (availablePaths.Count == 0) availablePaths = allPaths;

            PathPiece selectedPath = availablePaths[Random.Range(0, availablePaths.Count)];
            selectedPath.SelectedForQuest();
            GameManager.Instance.PathPieceSlectedForQuest(selectedPath);
        }

        public override bool IsAttackingKing(int row, int column)
        {
            return CheckStateCalculator.IsAttackingKingDiagonal(row, column, PieceColor);
        }

        public override bool CanMove(int row, int column)
        {
            return GameEndCalculator.CanMoveDiagonal(row, column, PieceColor);
        }

        internal override int GetPathLength()
        {
            return PathManager.GetDiagonalPathLength(this);
        }

        internal override bool HasSinglePath()
        {
            return PathManager.CheckDiagonalPathAtDistance(this, 1);
        }

        internal override bool HasLongPath()
        {
            return PathManager.CheckDiagonalPathAtDistance(this, 3);
        }
    }
}