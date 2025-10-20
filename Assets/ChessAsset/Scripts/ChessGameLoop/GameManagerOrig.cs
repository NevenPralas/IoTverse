using ChessEnums;
using Digiphy;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace ChessMainLoop
{
    public delegate void PieceMoved();

    public class GameManagerOrig : SingletonNetworkedReplaceable<GameManagerOrig>
    {
        private int _turnCount = 0;
        public int TurnCount { get => _turnCount; }
        private SideColor _turnPlayer;
        public SideColor TurnPlayer { get => _turnPlayer; set => _turnPlayer = value; }
        private SideColor _localPlayer;
        public SideColor LocalPlayer { get => _localPlayer; set => _localPlayer = value; }
        public bool IsPlayerTurn => _localPlayer == _turnPlayer;
        private SideColor _checkedSide;
        public SideColor CheckedSide { get => _checkedSide; set => _checkedSide = Check(value); }
        private Pawn _passantable = null;
        public Pawn Passantable { get => _passantable; set => _passantable = value; }
        private Pawn _promotingPawn = null;
        public bool IsPromotingPawn => _promotingPawn != null;
        public (int Row, int Column) PromotingPawnLocation => _promotingPawn.Location;
        [SerializeField] private AudioSource _checkSound;
        [SerializeField] private TextMeshPro _winnerText;
        [SerializeField] private List<GameObject> _blackInteractors;
        [SerializeField] private List<GameObject> _whiteInteractors;

        private bool _isPieceMoving = false;
        public bool IsPieceMoving { get => _isPieceMoving; set => _isPieceMoving = value; }

        public override void Spawned()
        {
            base.Spawned();

            _checkedSide = SideColor.None;
            _turnPlayer = SideColor.White;
            if (Runner.IsSharedModeMasterClient)
            {
                _localPlayer = SideColor.White;
                foreach(GameObject interactor in _blackInteractors)
                {
                    interactor.SetActive(false);
                }
            }
            else
            {
                _localPlayer = SideColor.Black;
                foreach (GameObject interactor in _whiteInteractors)
                {
                    interactor.SetActive(false);
                }
            }
        }

        /// <summary>
        /// Returns color of checked player and if there is a check plays check sound.
        /// </summary>
        /// <param name="_checkSide"></param>
        /// <returns>Color of player that is checked</returns>
        private SideColor Check(SideColor _checkSide)
        {
            if(_checkedSide == SideColor.None && _checkSide != SideColor.None)
            {
                _checkSound.Play();
            }
            return _checkSide == SideColor.Both ? _turnPlayer == SideColor.White ? SideColor.Black : SideColor.White : _checkSide;
        }

        public void ChangeTurn()
        {
            if (_promotingPawn) return;

            if (_turnPlayer == SideColor.White)
            {
                _turnPlayer = SideColor.Black;
            }
            else if (_turnPlayer == SideColor.Black)
            {
                _turnPlayer = SideColor.White;
            }
            SideColor _winner = BoardState.Instance.CheckIfGameOver();
            if (_winner != SideColor.None)
            {
                GameEnd(_winner);
            }
            _turnCount++;
        }

        public void GameEnd(SideColor _winner)
        {
            _winnerText.gameObject.SetActive(true);
            _winnerText.text = $"{_winner.ToString()} is the winner!!!";
            _turnPlayer = SideColor.None;
        }

        /// <summary>
        /// Resets state variables and starts a new round.
        /// </summary>
        public void Restart()
        {
            ObjectPool.Instance.ResetPieces();
            BoardState.Instance.ResetPieces();
            MoveTracker.Instance.ResetMoves();
            _turnCount = 0;
            _turnPlayer = SideColor.White;
            _checkedSide = SideColor.None;
            _passantable = null;
        }

        public void PawnPromoting(Pawn _pawn)
        {
            _promotingPawn = _pawn;
            if (_pawn.PieceColor != LocalPlayer) return;
            PromotionController.Instance.PawnPromotionMenu(_pawn.PieceColor);
        }

        /// <summary>
        /// Replaces pawn that is getting promoted with selected piece, then checks for checkmate.
        /// </summary>
        public void SelectedPromotion(Piece _piece, ChessPieceType pieceIndex, (int Row, int Column) pawnLocation)
        {
            Pawn pawn = (Pawn)BoardState.Instance.GetField(pawnLocation.Row, pawnLocation.Column);

            _piece.transform.parent = pawn.transform.parent;            
            _piece.transform.localScale = pawn.transform.localScale;
            _piece.HasMoved = true;
            _promotingPawn = null;
            BoardState.Instance.PromotePawn(pawn, _piece, pieceIndex);

            ChangeTurn();
        }
    }
}