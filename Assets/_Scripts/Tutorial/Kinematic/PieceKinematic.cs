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
    public class PieceKinematic : MonoBehaviour
    {
        private MeshRenderer _renderer;
        private Grabbable _grabbable;
        private PathKinematic _pathInContact;

        private void Start()
        {
            _renderer = GetComponentInChildren<MeshRenderer>();
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
                    GrabEnd();
                    break;
                case PointerEventType.Move:
                    break;
                case PointerEventType.Cancel:
                    break;
            }
        }

        private void GrabEnd()
        {
            _renderer.material.color = Color.green;

            if (!_pathInContact) return;

            float x = _pathInContact.transform.localPosition.x;
            x = Math.Abs(x - transform.localPosition.x);
            float z = _pathInContact.transform.localPosition.z;
            z = Math.Abs(z - transform.localPosition.z);
            if (x < 0.75 || z < 0.75)
            {
                _pathInContact.Selected();
                _pathInContact = null;
            }
        }

        private void GrabStart()
        {
            _renderer.material.color = Color.yellow;
        }

        private void HoverEnd()
        {
            _renderer.material.color = Color.green;
        }

        private void PieceHowered()
        {
            _renderer.material.color = Color.yellow;
        }

        private void OnTriggerEnter(Collider other)
        {
            if(other.TryGetComponent(out PathKinematic path))
            {
                if (path.gameObject.activeSelf) _pathInContact = path;
            } 
        }

        private void OnTriggerExit(Collider other)
        {
            if (other.TryGetComponent(out PathKinematic path))
            {
                if (path == _pathInContact) _pathInContact = null;
            }
        }
    }
}