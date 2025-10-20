using System.Collections.Generic;
using UnityEngine;

namespace ChessMainLoop
{
    /// <summary>
    /// Contains methods for calculating viable positions piece can move to, and placing path fields on them with appropriate color
    /// </summary>
    public static class PathManager
    {
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

        #region Normal Path Creation
        public static List<PathPiece> CreateDiagonalPath(Piece caller)
        {
            return CreatePathOnDirection(caller, DiagonalLookup);
        }

        public static List<PathPiece> CreateVerticalPath(Piece caller)
        {
            return CreatePathOnDirection(caller, VerticalLookup);
        }

        /// <summary>
        /// Checks for available spots for directions specified in lookup table and sets path field on them. Stops at first enemy or unavailable field in each direction.
        /// </summary>
        private static List<PathPiece> CreatePathOnDirection(Piece caller, int[,] lookupTable)
        {
            int startRow = caller.Location.Row;
            int startColumn = caller.Location.Column;
            List<PathPiece> paths = new List<PathPiece>();

            for (int j = 0; j < lookupTable.GetLength(0); j++)
            {
                for (int i = 1; ; i++)
                {
                    int newRow = startRow + i * lookupTable[j, 0];
                    int newColumn = startColumn + i * lookupTable[j, 1];
                    if (!BoardState.Instance.IsInBorders(newRow, newColumn)) break;
                    PathPiece path = CreatePath(caller, startRow, startColumn, newRow, newColumn);
                    if (path != null) paths.Add(path);
                    if (BoardState.Instance.GetField(newRow, newColumn) != null) break;
                }
            }

            return paths;
        }

        /// <summary>
        /// Checks if the field located at callers position translated by direction parameters is free to move. 
        /// </summary>
        public static PathPiece CreatePathInSpotDirection(Piece caller, int rowDirection, int columnDirection)
        {
            int startRow = caller.Location.Row;
            int startColumn = caller.Location.Column;

            int newRow = startRow + rowDirection;
            int newColumn = startColumn + columnDirection;
            return CreatePath(caller, startRow, startColumn, newRow, newColumn);
        }

        private static PathPiece CreatePath(Piece caller, int startRow, int startColumn, int newRow, int newColumn)
        {
            if (!BoardState.Instance.IsInBorders(newRow, newColumn)) return null;
            SideColor checkSide = BoardState.Instance.SimulateCheckState(startRow, startColumn, newRow, newColumn);

            if (checkSide == caller.PieceColor || checkSide == SideColor.Both) return null;

            Piece piece = BoardState.Instance.GetField(newRow, newColumn);
            GameObject path;
            if (piece == null)
            {
                path = ObjectPool.Instance.GetHighlightPath(PathType.Yellow);
            }
            else if (piece.PieceColor != caller.PieceColor)
            {
                path = ObjectPool.Instance.GetHighlightPath(PathType.Red);
                path.GetComponent<PathPiece>().AssignPiece(piece);
            }
            else return null;

            path.GetComponent<PathPiece>().Location = (newRow, newColumn);

            Vector3 position = new Vector3();

            position.x = newRow * BoardState.Offset;
            position.z = newColumn * BoardState.Offset;
            position.y = path.transform.localPosition.y;

            path.transform.localPosition = position;
            return path.GetComponent<PathPiece>();
        }

        #endregion

        #region Path Length

        public static int GetDiagonalPathLength(Piece caller)
        {
            return GetPathOnDirectionLength(caller, DiagonalLookup);
        }

        public static int GetVerticalPathLength(Piece caller)
        {
            return GetPathOnDirectionLength(caller, VerticalLookup);
        }

        /// <summary>
        /// Checks for available spots for directions specified in lookup table and sets path field on them. Stops at first enemy or unavailable field in each direction.
        /// </summary>
        private static int GetPathOnDirectionLength(Piece caller, int[,] lookupTable)
        {
            int startRow = caller.Location.Row;
            int startColumn = caller.Location.Column;
            int pathCount = 0;

            for (int j = 0; j < lookupTable.GetLength(0); j++)
            {
                for (int i = 1; ; i++)
                {
                    int newRow = startRow + i * lookupTable[j, 0];
                    int newColumn = startColumn + i * lookupTable[j, 1];
                    if (!BoardState.Instance.IsInBorders(newRow, newColumn)) break;
                    if(TryCreatePath(caller, startRow, startColumn, newRow, newColumn)) pathCount++;
                    if (BoardState.Instance.GetField(newRow, newColumn) != null) break;
                }
            }

            return pathCount;
        }

        /// <summary>
        /// Checks if the field located at callers position translated by direction parameters is free to move. 
        /// </summary>
        public static bool GetPathInSpotDirectionLength(Piece caller, int rowDirection, int columnDirection)
        {
            int startRow = caller.Location.Row;
            int startColumn = caller.Location.Column;

            int newRow = startRow + rowDirection;
            int newColumn = startColumn + columnDirection;
            return TryCreatePath(caller, startRow, startColumn, newRow, newColumn);
        }

        private static bool TryCreatePath(Piece caller, int startRow, int startColumn, int newRow, int newColumn)
        {
            if (!BoardState.Instance.IsInBorders(newRow, newColumn)) return false;
            SideColor checkSide = BoardState.Instance.SimulateCheckState(startRow, startColumn, newRow, newColumn);

            if (checkSide == caller.PieceColor || checkSide == SideColor.Both) return false;

            Piece piece = BoardState.Instance.GetField(newRow, newColumn);
            if (piece == null || piece.PieceColor != caller.PieceColor)
            {
                return true;
            }
            else return false;
        }

        #endregion

        #region Path Distance Check

        public static bool CheckDiagonalPathAtDistance(Piece caller, int distance)
        {
            return CheckPathOnDirectionAtDistance(caller, DiagonalLookup, distance);
        }

        public static bool CheckVerticalPathAtDistance(Piece caller, int distance)
        {
            return CheckPathOnDirectionAtDistance(caller, VerticalLookup, distance);
        }

        /// <summary>
        /// Checks for available spots for directions specified in lookup table and sets path field on them. Stops at first enemy or unavailable field in each direction.
        /// </summary>
        private static bool CheckPathOnDirectionAtDistance(Piece caller, int[,] lookupTable, int distance)
        {
            int startRow = caller.Location.Row;
            int startColumn = caller.Location.Column;

            for (int j = 0; j < lookupTable.GetLength(0); j++)
            {
                for (int i = 1; ; i++)
                {
                    int newRow = startRow + i * lookupTable[j, 0];
                    int newColumn = startColumn + i * lookupTable[j, 1];
                    if (!BoardState.Instance.IsInBorders(newRow, newColumn)) break;
                    if (newRow <= distance && newColumn <= distance && (newRow == distance || newColumn == distance) &&
                        TryCreatePath(caller, startRow, startColumn, newRow, newColumn)) return true;
                    if (BoardState.Instance.GetField(newRow, newColumn) != null) break;
                }
            }
            return false;
        }

        /// <summary>
        /// Checks if the field located at callers position translated by direction parameters is free to move. 
        /// </summary>
        public static bool CheckPathInSpotDirectionAtDistance(Piece caller, int rowDirection, int columnDirection, int distance)
        {
            int startRow = caller.Location.Row;
            int startColumn = caller.Location.Column;

            int newRow = startRow + rowDirection;
            int newColumn = startColumn + columnDirection;
            if(rowDirection <= distance && columnDirection <= distance && (rowDirection == distance || columnDirection == distance))
            return TryCreatePath(caller, startRow, startColumn, newRow, newColumn);

            else return false;
        }

        #endregion

        #region Common

        public static void CreatePassantSpot(Piece target, int row, int column)
        {
            PathPiece path = ObjectPool.Instance.GetHighlightPath(PathType.Red).GetComponent<PathPiece>();
            path.AssignPiece(target);
            path.Location = (row, column);

            Vector3 _position = new Vector3();
            _position.x = row * BoardState.Offset; 
            _position.z = column * BoardState.Offset; 
            _position.y = path.transform.localPosition.y;

            path.transform.localPosition = _position;
        }

        /// <summary>
        /// Checks if there is a piece that can be castled with at target location and if that castle action would result in check for turn player.
        /// </summary>
        public static void CreateCastleSpot(Piece caller, Piece target)
        {
            if (GameManager.Instance.CheckedSide == caller.PieceColor) return;

            int rowCaller = caller.Location.Row;
            int columnCaller = caller.Location.Column;
            int rowTarget = target.Location.Row;
            int columnTarget = target.Location.Column;

            //Check to see if there are any pieces between rook and king
            columnCaller += columnTarget > columnCaller ? 1 : -1;
            while (columnCaller != columnTarget) 
            {
                if (BoardState.Instance.GetField(rowCaller, columnCaller) != null) return;
                columnCaller += columnTarget > columnCaller ? 1 : -1;
            }

            columnCaller = caller.Location.Column;
            int columnMedian = (int)Mathf.Ceil((columnCaller + columnTarget) / 2f);

            if(BoardState.Instance.SimulateCheckState(rowCaller, columnCaller, rowCaller, columnMedian) == caller.PieceColor)
            {
                return;
            }
            if (BoardState.Instance.SimulateCheckState(rowTarget, columnTarget, rowTarget, columnMedian) == caller.PieceColor)
            {
                return;
            }

            PathPiece path = ObjectPool.Instance.GetHighlightPath(PathType.Yellow).GetComponent<PathPiece>();
            path.Location = (rowTarget, columnTarget);

            Vector3 position = new Vector3();
            path.AssignCastle(target);
            position.x = rowTarget * BoardState.Offset;
            position.z = columnTarget * BoardState.Offset;
            position.y = path.transform.localPosition.y;

            path.transform.localPosition = position;
        }
        #endregion

    }
}
