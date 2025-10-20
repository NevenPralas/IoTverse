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
    public class PathLogic : MonoBehaviour
    {
        [SerializeField] private bool _selectable;
        private MeshRenderer _renderer;
        private Grabbable _grabbable;
        private Color _originalColor;

        private void Awake()
        {
            _renderer = GetComponent<MeshRenderer>();
            if(_selectable)_renderer.material.color = Color.green;
            _originalColor = _renderer.material.color;
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

        private void GrabStart()
        {
            if (_selectable) GameManagerLogic.Instance.PathSelected();
        }

        private void HoverEnd()
        {
            _renderer.material.color = _originalColor;
        }

        private void PieceHowered()
        {
            _renderer.material.color = Color.yellow;
        }

        private void OnEnable()
        {
            if (_selectable) _renderer.material.color = Color.green;
        }
    }
}