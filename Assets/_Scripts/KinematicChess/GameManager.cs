using ChessEnums;
using ChessMainLoop;
using Digiphy;
using Fusion;
using Oculus.Interaction;
using System.Collections.Generic;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;

namespace KinematicChess
{
    public class GameManager : SingletonNetworkedReplaceable<GameManager>
    {
        [SerializeField] private List<Piece> _pieces;
        [SerializeField] private List<Field> _fields;
        [SerializeField] private TextMeshPro _endText;
        private SideColor _sideColor;
        private SideColor _currentTurn;
        private Field _selectedField;
        private Piece _selectedPiece;
        private int _turnCount = -1;
        private int _pieceErrors;
        private int _pathErrors;
        private float _time;
        private float _distance;
        private float _grabbAmount;
        private float _maxPing;
        private float _pingAmount;
        private int _pingCount;
        private float _time2;

        public override void Spawned()
        {
            base.Spawned();

            foreach (var item in _fields)
            {
                if(item != null) item.Init();
            }
            foreach (var item in _pieces)
            {
                if (item != null) item.Init();
            }

            if (Runner.IsSharedModeMasterClient)
            {
                _sideColor = SideColor.White;
                SetSelectedElementsWhite();
            }
            else _sideColor = SideColor.Black;
            _currentTurn = SideColor.White;
            DistanceGrabInteractor.GrabbedAction += ObjectGrabbed;
        }

        private void ObjectGrabbed(Transform grabber, Transform grabbed)
        {
            if (_currentTurn != _sideColor) return;
            if (grabbed.transform.parent != null && grabbed.transform.parent.TryGetComponent(out Piece piece))
            {
                _distance += Vector3.Distance(grabber.position, grabbed.position);
                _grabbAmount++;
                UiManager.Instance.ShowDistance(_distance / _grabbAmount, _turnCount);
            }
        }

        private void SetSelectedElementsWhite()
        {
            _pieceErrors = 0;
            _pathErrors = 0;
            _time = 0;
            _distance = 0;
            _grabbAmount = 0;
            UiManager.Instance.ShowPathError(_pathErrors, _turnCount);
            UiManager.Instance.ShowPieceError(_pieceErrors, _turnCount);
            UiManager.Instance.ShowDistance(0, _turnCount);

            int selected = Random.Range(16, 32);
            while (_pieces[selected] == null) selected = Random.Range(16, 32);
            _selectedPiece = _pieces[selected];
            _selectedPiece.Selected();

            if (_turnCount < 4) SetSelectableFieldShort();
            else SetSelectableFieldLong();
        }

        private void SetSelectedElementsBlack()
        {
            _pieceErrors = 0;
            _pathErrors = 0;
            _time = 0;
            _distance = 0;
            _grabbAmount = 0;
            UiManager.Instance.ShowPathError(_pathErrors, _turnCount);
            UiManager.Instance.ShowPieceError(_pieceErrors, _turnCount);
            UiManager.Instance.ShowDistance(0, _turnCount);

            int selected = Random.Range(0, 16);
            while(_pieces[selected] == null) selected = Random.Range(0, 16);
            _selectedPiece = _pieces[selected];
            _selectedPiece.Selected();
            if (_turnCount < 5) SetSelectableFieldShort();
            else SetSelectableFieldLong();
        }

        private void SetSelectableFieldShort()
        {
            List<(float, Field)> selectableFields = new List<(float, Field)>();
            foreach (var item in _fields)
            {
                if (item.IsInContact()) continue;
                float distance = Vector3.Distance(_selectedPiece.transform.position, item.transform.position);
                bool added = false;
                for(int i = 0; i < selectableFields.Count; i++)
                {
                    if (selectableFields[i].Item1 > distance)
                    {
                        selectableFields.Insert(i, (distance, item));
                        added = true;
                        break;
                    }
                }
                
                if(!added) selectableFields.Add((distance, item));
            }

            _selectedField = selectableFields[0].Item2;
            _selectedField.Selected();
        }

        private void SetSelectableFieldLong()
        {
            List<(float, Field)> selectableFields = new List<(float, Field)>();
            foreach (var item in _fields)
            {
                if (item.IsInContact()) continue;
                float distance = Vector3.Distance(_selectedPiece.transform.position, item.transform.position);
                bool added = false;
                for (int i = 0; i < selectableFields.Count; i++)
                {
                    if (selectableFields[i].Item1 > distance)
                    {
                        selectableFields.Insert(i, (distance, item));
                        added = true;
                        break;
                    }
                }

                if (!added) selectableFields.Add((distance, item));
            }

            _selectedField = selectableFields[selectableFields.Count / 2].Item2;
            _selectedField.Selected();
        }

        private void Update()
        {
            _time += Time.deltaTime;
            _time2 += Time.deltaTime;

            if (_time2 > 1 && _currentTurn != SideColor.None)
            {
                _time2 -= 1;
                float rtt = (float)(Runner.GetPlayerRtt(Runner.LocalPlayer) * 1000);
                _pingAmount += rtt;
                if (rtt > _maxPing) _maxPing = rtt;
                _pingCount++;
                UiManager.Instance.ShowPing(rtt, _maxPing);
            }
        }

        public void PieceGrabbed(bool selected)
        {
            if (_currentTurn != _sideColor) return;
            if(!selected)
            {
                _pieceErrors++;
                UiManager.Instance.ShowPieceError(_pieceErrors, _turnCount);
            } 
        }

        public void PieceGrabEnd(bool selected)
        {
            if (_currentTurn != _sideColor) return;

            if (!selected)
            {
                _pathErrors++;
                UiManager.Instance.ShowPathError(_pathErrors, _turnCount);
            }
            else
            {
                UiManager.Instance.ShowTime((int)_time, _turnCount);
                _selectedField.Unselected();
                _selectedPiece.Unselected();
                RPC_ChangeTurn();
            }
        }


        [Rpc(sources:RpcSources.All, targets:RpcTargets.All)]
        public void RPC_ChangeTurn()
        {
            _turnCount++;
            //_sideColor = _sideColor == SideColor.White ? SideColor.Black : SideColor.White;
            if (_turnCount > 8)
            {
                _endText.gameObject.SetActive(true);
                _currentTurn = SideColor.None;
                UiManager.Instance.ShowPing(_pingAmount / _pingCount, _maxPing);
            }
            else
            {
                if (_currentTurn == SideColor.White) _currentTurn = SideColor.Black;
                else _currentTurn = SideColor.White;
                if (_currentTurn == _sideColor && _sideColor == SideColor.White) SetSelectedElementsWhite();
                else if (_currentTurn == _sideColor && _sideColor == SideColor.Black) SetSelectedElementsBlack();
            }
        }

    }
}