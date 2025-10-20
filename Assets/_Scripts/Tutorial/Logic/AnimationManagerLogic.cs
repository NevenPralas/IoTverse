using ChessEnums;
using ChessMainLoop;
using Digiphy;
using Fusion;
using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace Tutorial
{
    public class AnimationManagerLogic : Singleton<AnimationManagerLogic>
    {
        [SerializeField] private AudioSource _moveSound;
        private bool _isActive;
        public bool IsActive => _isActive;

        public void MovePiece(PieceLogic piece, Vector3 target)
        {
            _isActive = true;
            StartCoroutine(MoveAnimation(piece, target));
        }

        /// <summary>
        /// Moves the piece to target location with root motion animations and translation.
        /// </summary>
        private IEnumerator MoveAnimation(PieceLogic piece, Vector3 target)
        {
            Vector3 upLocation = new Vector3(piece.transform.localPosition.x, piece.transform.localPosition.y + 3, piece.transform.localPosition.z);
            Vector3 upRotation = piece.transform.eulerAngles;

            int rotation = -45;
            upRotation.z = rotation;

            Quaternion upQuaternion = Quaternion.Euler(upRotation);
            while (piece.transform.localPosition != upLocation || piece.transform.rotation != upQuaternion)
            {
                piece.transform.localPosition = Vector3.MoveTowards(piece.transform.localPosition, upLocation, 12.5f * Time.deltaTime);
                piece.transform.rotation = Quaternion.RotateTowards(piece.transform.rotation, upQuaternion, 15 * 12.5f * Time.deltaTime);
                yield return null;
            }

            //performs translation to target position
            target.y = piece.transform.localPosition.y;
            while (piece.transform.localPosition != target)
            {
                piece.transform.localPosition = Vector3.MoveTowards(piece.transform.localPosition, target, 12.5f * Time.deltaTime);
                yield return null;
            }

            _moveSound.Play();

            Vector3 downLocation = new Vector3(piece.transform.localPosition.x, piece.transform.localPosition.y - 3, piece.transform.localPosition.z);
            Vector3 downRotation = piece.transform.eulerAngles;
            downRotation.x = 0f;
            downRotation.z = 0f;
            Quaternion downQuaternion = Quaternion.Euler(downRotation);
            while (piece.transform.localPosition != downLocation || piece.transform.rotation != downQuaternion)
            {
                piece.transform.localPosition = Vector3.MoveTowards(piece.transform.localPosition, downLocation, 12.5f * Time.deltaTime);
                piece.transform.rotation = Quaternion.RotateTowards(piece.transform.rotation, downQuaternion, 15 * 12.5f * Time.deltaTime);
                yield return null;
            }

            _isActive = false;
            GameManagerLogic.Instance.PieceMoved();
        }
    }
}