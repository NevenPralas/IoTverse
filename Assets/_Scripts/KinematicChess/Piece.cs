using ChessEnums;
using Digiphy;
using Fusion;
using Oculus.Interaction;
using Oculus.Platform.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;

namespace KinematicChess
{
    public class Piece : NetworkBehaviour
    {
        [SerializeField] private Field _initialField;
        [SerializeField] private List<Field> _fieldsInContact = new List<Field>();
        [SerializeField] private Vector3 _initialPosition;
        private MeshRenderer _renderer;
        private Grabbable _grabbable;
        private bool _selected;
        private Color _startColor;

        public bool IsSelected => _selected;

        public override void Spawned()
        {
            base.Spawned();
            transform.localPosition = _initialPosition;
        }

        public void Init()
        {
            _renderer = GetComponentInChildren<MeshRenderer>();
            _startColor = _renderer.material.color;
            _grabbable = GetComponentInChildren<Grabbable>();
            _fieldsInContact.Add(_initialField);
            _initialField.SetInitialPiece(this);
            _grabbable.WhenPointerEventRaised += ProcessPointerEvent;
        }

        public void Selected()
        {
            _selected = true;
            _renderer.material.color = Color.green;
            _grabbable.MovingEnabled = true;
        }

        public void Unselected()
        {
            _renderer.material.color = _startColor;
            _selected = false;
            _grabbable.MovingEnabled = true;
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
                    if (_selected) GrabbEnd();
                    break;
                case PointerEventType.Move:
                    break;
                case PointerEventType.Cancel:
                    break;
            }
        }

        public void GrabStart()
        {
            GameManager.Instance.PieceGrabbed(_selected);
        }

        public void PieceHowered()
        {
            _renderer.material.color = Color.yellow;
        }

        public void HoverEnd()
        {
            if (_selected) _renderer.material.color = Color.green;
            else _renderer.material.color = _startColor;
        }

        private void GrabbEnd()
        {
            if (_fieldsInContact.Count == 0) return;
            Field selectedField = _fieldsInContact.FirstOrDefault(field => field.IsSelected);
            if (selectedField == null) GameManager.Instance.PieceGrabEnd(false);
            else
            {
                float x = selectedField.transform.localPosition.x;
                x = Math.Abs(x - transform.localPosition.x);
                float z = selectedField.transform.localPosition.z;
                z = Math.Abs(z - transform.localPosition.z);
                if (x > 0.75 || z > 0.75) GameManager.Instance.PieceGrabEnd(false);
                else GameManager.Instance.PieceGrabEnd(true);
            }
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.TryGetComponent(out Field field))
            {
                if (!_fieldsInContact.Contains(field)) _fieldsInContact.Add(field);
            }
        }

        private void OnTriggerExit(Collider other)
        {
            if (other.TryGetComponent(out Field field))
            {
                if (_fieldsInContact.Contains(field)) _fieldsInContact.Remove(field);
            }
        }

        public bool IsInContact(Field field) => _fieldsInContact.Contains(field);
    }
}