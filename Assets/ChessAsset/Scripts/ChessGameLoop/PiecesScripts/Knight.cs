using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace ChessMainLoop
{
    public class Knight : Piece
    {
        /// <summary>
        /// Lookup table containing knight movement directions
        /// </summary>
        private static readonly int[,] LookupMoves =
        {
           { 1, -2 },
           { 2, -1 },
           { 2, 1 },
           { 1, 2 },
           { -1, 2 },
           { -2, 1 },
           { -2, -1 },
           { -1, -2 }
        };

        public override void CreatePath()
        {
            List<PathPiece> allPaths = new List<PathPiece>();
            for (int i = 0; i < LookupMoves.GetLength(0); i++)
            {
                PathPiece path = PathManager.CreatePathInSpotDirection(this, LookupMoves[i, 0], LookupMoves[i, 1]);
                if(path != null) allPaths.Add(path);
            }
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
            for (int i = 0; i < LookupMoves.GetLength(0); i++)
            {
                if (CheckStateCalculator.IsEnemyKingAtLocation(row, column, LookupMoves[i, 0], LookupMoves[i, 1], PieceColor))
                {
                    return true;
                }
            }

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

        internal override int GetPathLength()
        {
            int count = 0;
            for (int i = 0; i < LookupMoves.GetLength(0); i++)
            {
                if(PathManager.GetPathInSpotDirectionLength(this, LookupMoves[i, 0], LookupMoves[i, 1])) count++;
            }
            return count;
        }

        internal override bool HasSinglePath()
        {
            return false;
        }

        internal override bool HasLongPath()
        {
            return true;
        }
    }
}
