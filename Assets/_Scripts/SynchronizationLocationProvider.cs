using Fusion;
using UnityEngine;

namespace Digiphy
{
    public class SynchronizationLocationProvider : NetworkBehaviour
    {
        [SerializeField] private GameObject _tutorial; 

        public override void Spawned()
        {
            if (TestingSetupManager.Instance == null) return;
            TestingSetupManager.Instance.Init(Runner);
            if (VrRoomSynchronizer.Instance == null) return;

            VrRoomSynchronizer.Instance.SynchronizeRoomWithAr(Runner, transform);
            _tutorial.SetActive(false);
        }
    }
}
