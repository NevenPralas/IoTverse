using System.Collections;
using UnityEngine;
using Digiphy;

namespace ChessMainLoop
{
    public class AnimationManager : SingletonReplaceable<AnimationManager>
    {
        #region Private animator prefabs
        [SerializeField]
        private RuntimeAnimatorController _whiteAnimator;
        [SerializeField]
        private RuntimeAnimatorController _blackAnimator;
        [SerializeField]
        private RuntimeAnimatorController _whiteKnightAnimator;
        [SerializeField]
        private RuntimeAnimatorController _blackKnightAnimator;
        [SerializeField]
        private RuntimeAnimatorController _whiteBishopAnimator;
        [SerializeField]
        private RuntimeAnimatorController _blackBishopAnimator;
        #endregion

        [SerializeField] private float _moveSpeed = 20f;
        [SerializeField] private AudioSource _moveSound;
        private bool _isActive = false;

        public bool IsActive { get => _isActive; }       
        public AudioSource MoveSound => _moveSound;

        /// <summary>
        /// Assigns animator controller to piece based on piece type.
        /// </summary>
        /// <returns>Animator controler for piece parameter</returns>
        public RuntimeAnimatorController Assign(Piece piece)
        {
            if (piece.PieceColor == SideColor.White)
            {
                if(piece is Knight || piece is King)
                {
                    return _whiteKnightAnimator;
                }

                if(piece is Bishop)
                {
                    return _whiteBishopAnimator;
                }

                return _whiteAnimator;
            }
            else
            {
                if (piece is Knight || piece is King)
                {
                    return _blackKnightAnimator;
                }

                if (piece is Bishop)
                {
                    return _blackBishopAnimator;
                }

                return _blackAnimator;
            }
        }

        public void MovePiece(Piece piece, Vector3 target, Piece killTarget)
        {
            _isActive = true;
            StartCoroutine(MoveAnimation(piece, target, killTarget));
        }

        /// <summary>
        /// Moves the piece to target location with root motion animations and translation.
        /// </summary>
        private IEnumerator MoveAnimation(Piece piece, Vector3 target, Piece killTarget)
        {
            Vector3 upLocation = new Vector3(piece.transform.localPosition.x, piece.transform.localPosition.y + 3, piece.transform.localPosition.z);
            Vector3 upRotation = piece.transform.eulerAngles;

            int rotation = 45;
            if (piece.PieceColor == SideColor.White) rotation *= -1;
            if(piece is Bishop) rotation *= -1;
            if (piece is King || piece is Knight) upRotation.x = rotation;
            else upRotation.z = rotation;

            Quaternion upQuaternion = Quaternion.Euler(upRotation);
            while (piece.transform.localPosition != upLocation || piece.transform.rotation != upQuaternion)
            {
                piece.transform.localPosition = Vector3.MoveTowards(piece.transform.localPosition, upLocation, _moveSpeed * Time.deltaTime);
                piece.transform.rotation = Quaternion.RotateTowards(piece.transform.rotation, upQuaternion, 15 * _moveSpeed * Time.deltaTime);
                yield return null;
            }

            //performs translation to target position
            target.y = piece.transform.localPosition.y;
            while (piece.transform.localPosition != target)
            {
                piece.transform.localPosition = Vector3.MoveTowards(piece.transform.localPosition, target, _moveSpeed * Time.deltaTime);
                yield return null;
            }

            _moveSound.Play();

            //Perfoms root motion animation that puts piece back down
            if (killTarget != null)
            {
                killTarget.Die();
            }

            Vector3 downLocation = new Vector3(piece.transform.localPosition.x, piece.transform.localPosition.y - 3, piece.transform.localPosition.z);
            Vector3 downRotation = piece.transform.eulerAngles;
            downRotation.x = 0f;
            downRotation.z = 0f;
            Quaternion downQuaternion = Quaternion.Euler(downRotation);
            while (piece.transform.localPosition != downLocation || piece.transform.rotation != downQuaternion)
            {
                piece.transform.localPosition = Vector3.MoveTowards(piece.transform.localPosition, downLocation, _moveSpeed * Time.deltaTime);
                piece.transform.rotation = Quaternion.RotateTowards(piece.transform.rotation, downQuaternion, 15 * _moveSpeed * Time.deltaTime);
                yield return null;
            }

            _isActive = false;
        }
    }
}
