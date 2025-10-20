using ChessEnums;
using ChessMainLoop;
using Digiphy;
using Fusion;
using Oculus.Interaction;
using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace Tutorial
{
    public class PieceLogic : MonoBehaviour
    {
        private MeshRenderer _renderer;
        private Grabbable _grabbable;
        private bool _selected;
        private Color _startColor;

        private void Start()
        {
            _renderer = GetComponent<MeshRenderer>();
            _startColor = _renderer.material.color; 
            _renderer.material.color = Color.green;
            _grabbable = GetComponentInChildren<Grabbable>();
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
                    GrabStart();
                    break;
                case PointerEventType.Unselect:
                    break;
                case PointerEventType.Move:
                    break;
                case PointerEventType.Cancel:
                    break;
            }
        }

        public void Moving()
        {
            _selected = false;
            _renderer.material.color = _startColor;
        }

        public void Moved()
        {
            _renderer.material.color = Color.green;
        }

        private void GrabStart()
        {
            if (AnimationManagerLogic.Instance.IsActive) return;
            if (_selected) return;
            _selected = true;
            _renderer.material.color = Color.yellow;
            GameManagerLogic.Instance.PieceSelected();
        }

        private void HoverEnd()
        {
            if (AnimationManagerLogic.Instance.IsActive) return;
            if (_selected) _renderer.material.color = Color.yellow;
            else _renderer.material.color = Color.green;
        }

        private void PieceHowered()
        {
            if (AnimationManagerLogic.Instance.IsActive) return;
            _renderer.material.color = Color.yellow;
        }
    }
}