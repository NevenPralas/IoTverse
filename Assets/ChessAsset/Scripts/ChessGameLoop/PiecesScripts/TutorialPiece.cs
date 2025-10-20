using Oculus.Interaction;
using UnityEngine;

namespace ChessMainLoop
{

    public class TutorialPiece : MonoBehaviour
    {
        [SerializeField] private Grabbable _grabbable;
        [SerializeField] private TutorialPiece _otherPiece;
        [SerializeField] private Renderer _renderer;
        private Color _startColor;
        private bool _selected;

        private void Start()
        {
            _startColor = _renderer.material.color;
            _grabbable.WhenPointerEventRaised += ProcessPointerEvent;
        }

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

        //If its turn players piece sets it as selected and sets path pieces for it. If the piece is target of enemy or castle calls select method of path object this piece is assing to.
        public void PieceSelected()
        {
            _selected = true;
            _renderer.material.color = Color.yellow;
            _otherPiece.OtherSelected();
        }

        public void PieceHowered()
        {
            _renderer.material.color = Color.yellow;
        }

        public void HoverEnd()
        {
            if(!_selected) _renderer.material.color = _startColor;
        }

        public void OtherSelected()
        {
            _selected = false;
            _renderer.material.color = _startColor;
        }
    }
}
