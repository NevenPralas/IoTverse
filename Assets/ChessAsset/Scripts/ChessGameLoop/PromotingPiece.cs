using Oculus.Interaction;
using UnityEngine;
using ChessEnums;

namespace ChessMainLoop
{
    public class PromotingPiece :  MonoBehaviour
    {
        [SerializeField] private Grabbable _grabbable;
        [SerializeField] private Renderer _renderer;
        [SerializeField] private ChessPieceType _pieceIndex;
        private Color _startColor;

        private void Start()
        {
            _grabbable.WhenPointerEventRaised += ProcessPointerEvent;
            _startColor = _renderer.material.color;
        }

        private void OnMouseEnter() => PieceHowered();

        private void OnMouseExit() => HoverEnd();

        private void OnMouseDown() => PieceSelected();

        public void ProcessPointerEvent(PointerEvent evt)
        {
            switch (evt.Type)
            {
                case PointerEventType.Hover:
                    PieceHowered();
                    break;
                case PointerEventType.Unhover:
                    HoverEnd();
                    break;
                case PointerEventType.Select:
                    PieceSelected();
                    break;
                case PointerEventType.Unselect:
                    break;
                case PointerEventType.Move:
                    break;
                case PointerEventType.Cancel:
                    break;
            }
        }

        public void PieceSelected()
        {
            PromotionController.Instance.RPC_PieceSelected(_pieceIndex, GameManager.Instance.PromotingPawnLocation.Row, GameManager.Instance.PromotingPawnLocation.Column);
        }

        public void PieceHowered()
        {
            _renderer.material.color = Color.yellow;
        }

        public void HoverEnd()
        {
            _renderer.material.color = _startColor;
        }
    }
}