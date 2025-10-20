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
    public class PieceSnapping : MonoBehaviour
    {
        private MeshRenderer _renderer;
        private Grabbable _grabbable;
        private bool _selected;
        private PathSnapping _howeredSnappingPath = null;
        private Vector3 _originalPosition;
        private bool _grabbed;

        public Transform visual => _grabbable.transform;

        private void Start()
        {
            _originalPosition = transform.localPosition;
            _renderer = GetComponentInChildren<MeshRenderer>();
            _renderer.material.color = Color.green;
            _grabbable = GetComponentInChildren<Grabbable>();
            _grabbable.WhenPointerEventRaised += ProcessPointerEvent;
        }

        private void Update()
        {
            if (!_grabbed) return;
            RaycastHit[] raycastHits = Physics.RaycastAll(transform.position, Vector3.down);
            foreach (RaycastHit hit in raycastHits)
            {
                if (hit.collider.TryGetComponent(out PathSnapping path))
                {
                    if (path == _howeredSnappingPath) return;
                    path.PieceHowered();
                    _howeredSnappingPath?.HoverEnd();
                    _howeredSnappingPath = path;
                }
            }

            _howeredSnappingPath?.HoverEnd();
            _howeredSnappingPath = null;
        }

        private void GrabbEnd()
        {
            SnappingPointerSnapping.Instance.Unset();
            _grabbed = false;
            Vector3 raycastPosition = _grabbable.transform.position;
            transform.localPosition = _originalPosition;
            transform.rotation = Quaternion.identity;

            _howeredSnappingPath?.HoverEnd();
            _howeredSnappingPath = null;

            PathSnapping selectedPath = null;

            RaycastHit[] raycastHits = Physics.RaycastAll(raycastPosition, Vector3.down);
            foreach (RaycastHit hit in raycastHits)
            {
                if (hit.collider.TryGetComponent(out PathSnapping path))
                {
                    selectedPath = path;
                    break;
                }
            }

            selectedPath?.GrabStart();
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
                    GrabbEnd();
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
            _renderer.material.color = Color.green;
            _originalPosition = transform.localPosition;
        }

        private void GrabStart()
        {
            SnappingPointerSnapping.Instance.SetFigure(transform);
            if (_selected) return;
            _selected = true;
            _renderer.material.color = Color.yellow;
            GameManagerSnapping.Instance.PieceSelected();
            _grabbed = true;
        }

        private void HoverEnd()
        {
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