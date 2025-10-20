using ChessEnums;
using ChessMainLoop;
using Digiphy;
using Fusion;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace Tutorial
{
    public class PathKinematic : MonoBehaviour
    {
        [SerializeField] private PathKinematic _other;

        public void Selected()
        {
            gameObject.SetActive(false);
            _other.gameObject.SetActive(true);
        }

    }
}