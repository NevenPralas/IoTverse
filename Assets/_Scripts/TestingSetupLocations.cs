using Fusion;
using UnityEngine;
using UnityEngine.UI;

namespace Digiphy
{
    public class TestingSetupLocations : MonoBehaviour
    {
        [Header("Chess Locations")]
        [SerializeField] private Transform _chessLocationTable;
        [SerializeField] private Transform _chessLocationCouchInFront;
        [SerializeField] private Transform _chessLocationCouchInBetween;
        [SerializeField] private Transform _chessLocationLarge;

        private void Start()
        {
            TestingSetupManager.Instance.SetChessLocations(
                _chessLocationTable, _chessLocationCouchInFront, _chessLocationCouchInBetween, _chessLocationLarge);
        }
    }
}
