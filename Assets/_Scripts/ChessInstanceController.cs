using Fusion;
using UnityEngine;
using UnityEngine.UI;

namespace Digiphy
{
    public class ChessInstanceController : NetworkBehaviour
    {
        [SerializeField] private MeshCollider _meshCollider;
        [SerializeField] private GameObject _grabbable;
        [SerializeField] private GameObject _board;

        public override void Spawned()
        {
            base.Spawned();
            if (TestingSetupManager.Instance == null) return;
            TestingSetupManager.Instance.ChessCreated(Object);

            _meshCollider.enabled = TestingSetupManager.Instance.IsChessMoveable;
            _grabbable.SetActive(TestingSetupManager.Instance.IsChessMoveable);

            TestingSetupManager.Instance.ChessMoveableChanged += ChessMoveableChanged;
        }

        private void ChessMoveableChanged(bool value)
        {
            _meshCollider.enabled = value;
            _grabbable.SetActive(value);
        }

        [Rpc(RpcSources.All, RpcTargets.StateAuthority)]
        public void RPC_RemoveChess() => Runner.Despawn(Object);

        public void SetScale(Vector3 scale) => _board.transform.localScale = scale;
    }
}