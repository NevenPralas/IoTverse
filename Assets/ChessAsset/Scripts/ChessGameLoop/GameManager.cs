using ChessEnums;
using Digiphy;
using Oculus.Interaction;
using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace ChessMainLoop
{
    public class GameManager : SingletonNetworkedReplaceable<GameManager>
    {
        private int _turnCount = -1;
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
        private List<Piece> _selectablePieces = new List<Piece>();
        private float _time;
        private float _time2;
        private float _distance;
        private float _grabbAmount;
        private float _maxPing;
        private float _pingAmount;
        private int _pingCount;

        private bool _isPieceMoving = false;
        public bool IsPieceMoving { get => _isPieceMoving; set => _isPieceMoving = value; }
        private Piece _selectedPieceForQuest;
        private PathPiece _selectedPathPieceForQuest;

        private int _pathErrors;
        private int _pieceErrors;

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
                foreach(GameObject interactor in _whiteInteractors)
                {
                    if (!interactor.activeSelf) continue;
                    Piece piece = interactor.transform.parent.GetComponent<Piece>();
                    if(piece is Queen || piece is Bishop || piece is Rook)
                    {
                        _selectablePieces.Add(piece);
                    }
                }

                SetSelectableFigure();
            }
            else
            {
                _localPlayer = SideColor.Black;
                foreach (GameObject interactor in _whiteInteractors)
                {
                    interactor.SetActive(false);
                }
                foreach (GameObject interactor in _blackInteractors)
                {
                    if (!interactor.activeSelf) continue;
                    Piece piece = interactor.transform.parent.GetComponent<Piece>();
                    if (piece is Queen || piece is Bishop || piece is Rook)
                    {
                        _selectablePieces.Add(piece);
                    }
                }
            }

            DistanceGrabInteractor.GrabbedAction += ObjectGrabbed;
        }

        private void ObjectGrabbed(Transform grabber, Transform grabbed)
        {
            if(grabbed.transform.parent != null && grabbed.transform.parent.TryGetComponent(out Piece piece) 
                ||  grabbed.transform.parent.TryGetComponent(out PathPiece path))
            {
                _distance += Vector3.Distance(grabber.position, grabbed.position);
                _grabbAmount++;
                UiManager.Instance.ShowDistance(_distance / _grabbAmount, _turnCount);
            }
        }

        private void Update()
        {
            _time += Time.deltaTime;
            _time2 += Time.deltaTime;

            if(_time2 > 1 && _turnPlayer != SideColor.None)
            {
                _time2 -= 1;
                float rtt = (float)(Runner.GetPlayerRtt(Runner.LocalPlayer) * 1000);
                _pingAmount += rtt;
                if(rtt > _maxPing) _maxPing = rtt;
                _pingCount++;
                UiManager.Instance.ShowPing(rtt, _maxPing);
            }
        }

        private void SetSelectableFigure()
        {
            _pathErrors = 0;
            _pieceErrors = 0;
            _time = 0;
            _grabbAmount = 0;
            _distance = 0;
            if (_turnCount < 5) SetSelectableFigureShort();
            else SetSelectableFigureLong();
        }

        private void SetSelectableFigureShort()
        {
            List<(int, Piece)> pieceLenghts = new List<(int, Piece)>();

            foreach(Piece piece in _selectablePieces)
            {
                if (piece.gameObject.activeSelf && piece.HasSinglePath())
                {
                    int length = piece.GetPathLength();
                    if(length > 5) pieceLenghts.Add((length, piece));
                }
            }

            if (pieceLenghts.Count == 0)
            {
                foreach (Piece piece in _selectablePieces)
                {
                    if (piece.gameObject.activeSelf && piece.HasSinglePath())
                    {
                        pieceLenghts.Add((piece.GetPathLength(), piece));
                    }
                }
            }

            if (pieceLenghts.Count == 0)
            {
                foreach (Piece piece in _selectablePieces)
                {
                    if (piece.gameObject.activeSelf)
                    {
                        pieceLenghts.Add((piece.GetPathLength(), piece));
                    }
                }
            }

            int index = UnityEngine.Random.Range(0, pieceLenghts.Count);
            _selectedPieceForQuest = pieceLenghts[index].Item2;
            pieceLenghts[index].Item2.QuestSelectSingle();
        }

        private void SetSelectableFigureLong()
        {
            List<(int, Piece)> pieceLenghts = new List<(int, Piece)>();

            foreach (Piece piece in _selectablePieces)
            {
                if (piece.gameObject.activeSelf && piece.HasLongPath())
                {
                    int length = piece.GetPathLength();
                    if (length > 5)
                        pieceLenghts.Add((length, piece));
                }
            }

            if (pieceLenghts.Count == 0)
            {
                foreach (Piece piece in _selectablePieces)
                {
                    if (piece.gameObject.activeSelf && piece.HasLongPath())
                        pieceLenghts.Add((piece.GetPathLength(), piece));
                }
            }

            if (pieceLenghts.Count == 0)
            {
                foreach (Piece piece in _selectablePieces)
                {
                    if (piece.gameObject.activeSelf) pieceLenghts.Add((piece.GetPathLength(), piece));
                }
            }

            int index = UnityEngine.Random.Range(0, pieceLenghts.Count);
            _selectedPieceForQuest = pieceLenghts[index].Item2;
            pieceLenghts[index].Item2.QuestSelectLong();
        }

        public void PathPieceSlectedForQuest(PathPiece pathPiece) => _selectedPathPieceForQuest = pathPiece;

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
            if (_turnPlayer == LocalPlayer)
            {
                UiManager.Instance.ShowPathError(_pathErrors, _turnCount);
                UiManager.Instance.ShowPieceError(_pieceErrors, _turnCount);
                if (_grabbAmount == 0) UiManager.Instance.ShowDistance(0, _turnCount);
                else UiManager.Instance.ShowDistance(_distance / _grabbAmount, _turnCount);
                UiManager.Instance.ShowTime((int)_time, _turnCount);
            }

            _turnCount++;
            if(_turnCount > 8)
            {
                GameEnd();
            }

            if (_turnPlayer == SideColor.White)
            {
                _turnPlayer = SideColor.Black;
            }
            else if (_turnPlayer == SideColor.Black)
            {
                _turnPlayer = SideColor.White;
            }

            if (_turnPlayer == LocalPlayer) SetSelectableFigure();
            //SideColor _winner = BoardState.Instance.CheckIfGameOver();
            //if (_winner != SideColor.None)
            //{
            //    GameEnd(_winner);
            //}

        }

        public void GameEnd()
        {
            _turnPlayer = SideColor.None;
            _winnerText.gameObject.SetActive(true);
            UiManager.Instance.ShowPing(_pingAmount / _pingCount, _maxPing);
        }

        /// <summary>
        /// Resets state variables and starts a new round.
        /// </summary>
        public void Restart()
        {
            ObjectPool.Instance.ResetPieces();
            BoardState.Instance.ResetPieces();
            MoveTracker.Instance.ResetMoves();
            _turnCount = -1;
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

        internal bool PathSelected(PathPiece path)
        {
            if(path != _selectedPathPieceForQuest)
            {
                _pathErrors++;
                UiManager.Instance.ShowPathError(_pathErrors, _turnCount);
                return false;
            }
            else return true;
        }

        internal bool PieceSelected(Piece piece)
        {
            if (piece != _selectedPieceForQuest)
            {
                _pieceErrors++;
                UiManager.Instance.ShowPieceError(_pieceErrors, _turnCount);
                return false;
            }
            else return true;
        }
    }
}