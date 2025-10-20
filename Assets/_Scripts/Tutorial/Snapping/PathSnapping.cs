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
    public class PathSnapping : MonoBehaviour
    {
        [SerializeField] private bool _selectable;
        private MeshRenderer _renderer;
        private Color _originalColor;

        private void Awake()
        {
            _renderer = GetComponent<MeshRenderer>();
            if(_selectable)_renderer.material.color = Color.green;
            _originalColor = _renderer.material.color;
        }

        public void GrabStart()
        {
            if (_selectable) GameManagerSnapping.Instance.PathSelected();
        }

        public void HoverEnd()
        {
            _renderer.material.color = _originalColor;
        }

        public void PieceHowered()
        {
            _renderer.material.color = Color.yellow;
        }

        private void OnEnable()
        {
            if (_selectable) _renderer.material.color = Color.green;
        }
    }
}