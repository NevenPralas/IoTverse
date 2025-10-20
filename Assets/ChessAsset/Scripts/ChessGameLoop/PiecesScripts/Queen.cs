using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace ChessMainLoop
{
    public class Queen : Piece
    {    
        public override void CreatePath()
        {
            List<PathPiece> allPaths = PathManager.CreateDiagonalPath(this).Concat(PathManager.CreateVerticalPath(this)).ToList();
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
            return CheckStateCalculator.IsAttackingKingVertical(row, column, PieceColor) || CheckStateCalculator.IsAttackingKingDiagonal(row, column, PieceColor);
        }

        public override bool CanMove(int _xPosition, int _yPosition)
        {
            return GameEndCalculator.CanMoveDiagonal(_xPosition, _yPosition, PieceColor) || GameEndCalculator.CanMoveVertical(_xPosition, _yPosition, PieceColor);
        }

        internal override int GetPathLength()
        {
            return PathManager.GetDiagonalPathLength(this) + PathManager.GetVerticalPathLength(this);
        }

        internal override bool HasSinglePath()
        {
            return PathManager.CheckDiagonalPathAtDistance(this, 1) || PathManager.CheckVerticalPathAtDistance(this, 1);
        }

        internal override bool HasLongPath()
        {
            return PathManager.CheckDiagonalPathAtDistance(this, 3) || PathManager.CheckVerticalPathAtDistance(this, 3);
        }
    }
}