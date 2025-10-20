using Fusion;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Digiphy
{
    public class TurnToCamera : MonoBehaviour 
    {
        private Transform _camera;

        private void Start()
        {
            _camera = Camera.main.transform;
        }
        private void Update()
        {
            transform.LookAt(_camera);
            transform.Rotate(0, 180, 0);
        }
    }
}
