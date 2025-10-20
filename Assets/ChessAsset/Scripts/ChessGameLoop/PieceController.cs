using Digiphy;
using Fusion;
using System.Collections;
using UnityEngine;

namespace ChessMainLoop
{
    public class PieceController : SingletonNetworkedReplaceable<PieceController>
    {
        private Piece _activePiece;
        public bool AnyActive { get => _activePiece != null; }

        public static event PieceMoved PieceMoved;

        void OnEnable()
        {
            _activePiece = null;
            PathPiece.PathSelect += PathSelected;
        }

        void OnDisable()
        {
            PathPiece.PathSelect -= PathSelected;
        }

        /// <summary>
        /// Upon selecting path to move selected piece to starts the moving coroutine and clears all active paths
        /// </summary>
        private void PathSelected(PathPiece _path)
        {
            if(!GameManager.Instance.PathSelected(_path)) return;

            _activePiece.QuestUnselect();
            Piece _assignedEnemy = _path.AssignedPiece;
            Piece _assignedCastle = _path.AssignedCastle;
            GameManager.Instance.IsPieceMoving = true;
            if (_assignedCastle != null)
            {
                _path.AssignedCastle.AssignedAsCastle = null;
            }        
            PieceMoved?.Invoke();

            int oldRow = _activePiece.Location.Row;
            int oldColumn = _activePiece.Location.Column;
            int newRow = _path.Location.Row;
            int newColumn = _path.Location.Column;

            _activePiece.IsActive = false;
            if (_assignedCastle)
            {
                RPC_CastleMove(oldRow, oldColumn, newRow, newColumn);
            }
            else
            {
                int enemyRow = _assignedEnemy ? _assignedEnemy.Location.Row : -1;
                int enemyColumn = _assignedEnemy ? _assignedEnemy.Location.Column : -1;
                RPC_RegularMove(oldRow, oldColumn, newRow, newColumn, enemyRow, enemyColumn);
            }
        }

        [Rpc(sources: RpcSources.All, targets: RpcTargets.All, HostMode = RpcHostMode.SourceIsServer)]
        public void RPC_RegularMove(int oldRow, int oldColumn, int newRow, int newColumn, int enemyRow, int enemyColumn)
        {
            StartCoroutine(PieceRegularMover(oldRow, oldColumn, newRow, newColumn, enemyRow, enemyColumn));
        }

        [Rpc(sources: RpcSources.All, targets: RpcTargets.All, HostMode = RpcHostMode.SourceIsServer)]
        public void RPC_CastleMove(int oldRow, int oldColumn, int castleRow, int castleColumn)
        {
            StartCoroutine(PieceCastleMover(oldRow, oldColumn, castleRow, castleColumn));
        }

        /// <summary>
        /// Moves the selected piece to target path position. Has special cases for castling.
        /// </summary>
        private IEnumerator PieceRegularMover(int oldRow, int oldColumn, int newRow, int newColumn, int enemyRow, int enemyColumn)
        {
            Vector3 targetPosition = new Vector3();

            SideColor checkSide = BoardState.Instance.SimulateCheckState(oldRow, oldColumn, newRow, newColumn);
            GameManager.Instance.CheckedSide = checkSide;


            Piece movingPiece = BoardState.Instance.GetField(oldRow, oldColumn);
            Piece enemy = null;
            if (BoardState.Instance.IsInBorders(enemyRow, enemyColumn)) enemy = BoardState.Instance.GetField(enemyRow, enemyColumn);
            targetPosition.x = newRow * BoardState.Offset;
            targetPosition.y = movingPiece.OriginalY;
            targetPosition.z = newColumn * BoardState.Offset;
            movingPiece.Move(newRow, newColumn);
            if (!movingPiece.IsSnappinPiece)
            {           
                AnimationManager.Instance.MovePiece(movingPiece, targetPosition, enemy);
                while (AnimationManager.Instance.IsActive == true)
                {
                    yield return null;
                }
            }
            else
            {
                movingPiece.transform.localPosition = targetPosition;
                AnimationManager.Instance.MoveSound.Play();
                enemy?.Die();   
            }

            _activePiece = null;
            GameManager.Instance.IsPieceMoving = false;
            GameManager.Instance.ChangeTurn();
        }

        private IEnumerator PieceCastleMover(int callerRow, int callerColumn, int castleRow, int castleColumn)
        {
            Vector3 targetPositionKing = new Vector3();
            Vector3 targetPositionRook = new Vector3();

            Piece firstPiece = BoardState.Instance.GetField(callerRow, callerColumn);
            Piece secondPiece = BoardState.Instance.GetField(castleRow, castleColumn);
            Piece king = firstPiece is King ? firstPiece : secondPiece;
            Piece rook = firstPiece is Rook ? firstPiece : secondPiece;

            //If target is a castling position performs special castling action. Position calculations are done differently if the target is a King or a Rook          
            int columnMedian = (int)Mathf.Ceil((king.Location.Column + rook.Location.Column) / 2f);
            int rookNewColumn = columnMedian > king.Location.Column ? columnMedian - 1 : columnMedian + 1;
            SideColor checkedSide;

            targetPositionKing.x = callerRow * BoardState.Offset;
            targetPositionKing.y = 0;
            targetPositionKing.z = columnMedian * BoardState.Offset;

            targetPositionRook.x = callerRow * BoardState.Offset;
            targetPositionRook.y = 0;
            targetPositionRook.z = rookNewColumn * BoardState.Offset;

            king.Move(callerRow, columnMedian);
            AnimationManager.Instance.MovePiece(king, targetPositionKing, null);
            while (AnimationManager.Instance.IsActive == true)
            {
                yield return null;
            }

            checkedSide = BoardState.Instance.SimulateCheckState(callerRow, rook.Location.Column, callerRow, rookNewColumn);

            rook.Move(callerRow, rookNewColumn);
            AnimationManager.Instance.MovePiece(rook, targetPositionRook, null);
            while (AnimationManager.Instance.IsActive == true)
            {
                yield return null;
            }            

            GameManager.Instance.CheckedSide = checkedSide;
            GameManager.Instance.Passantable = null;
           
            _activePiece = null;
            GameManager.Instance.IsPieceMoving = false;
            GameManager.Instance.ChangeTurn();
        }

        /// <summary>
        /// Replaces status of selected piece with newly selected piece.
        /// </summary>
        public bool PieceSelected(Piece _piece)
        {
            if(!GameManager.Instance.PieceSelected(_piece)) { return false; }

            if (_activePiece)
            {
                _activePiece.IsActive = false;
                PieceMoved?.Invoke();
            }

            _activePiece = _piece;
            return true;
        }
    }
}