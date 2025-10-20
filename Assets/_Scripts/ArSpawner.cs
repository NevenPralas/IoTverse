using Fusion;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Digiphy
{
    public class ArSpawner : Singleton<SpawnManager>
    {
        private enum RoomType
        {
            ChessTestPosition = 1,
            ChessOfficePosition = 2,
            ChessLabPosition = 3,
            ChessLabNew = 4
        }

        [SerializeField] private GameObject _chessPositionsPrefab;
        [SerializeField] private GameObject _synchronizationLocationPrefab;
        [SerializeField] private RoomType _roomType;
        [SerializeField] private Vector3 _chessOffest;
        private NetworkRunner _runner;

        public void PlayerJoined(NetworkRunner runner, PlayerRef player)
        {
            if (player != runner.LocalPlayer) return;

            _runner = runner;
            LoadAnchors();
        }

        private void Update()
        {
            GameObject avatar = GameObject.Find("S0_L0_M1_V0_optimized_geom,0");
            if (avatar == null) return;

            MeshRenderer renderer = avatar.GetComponent<MeshRenderer>();
            renderer.enabled = false;
            enabled = false;
        }

        private async void LoadAnchors()
        {
            string uuid = PlayerPrefs.GetString(_roomType.ToString());
            Guid guid = new Guid(uuid);

            var options = new OVRSpatialAnchor.LoadOptions
            {
                Timeout = 0,
                StorageLocation = OVRSpace.StorageLocation.Local,
                Uuids = new Guid[] { guid }
            };

            OVRResult<List<OVRSpatialAnchor.UnboundAnchor>, OVRAnchor.FetchResult> oVRResult = await OVRSpatialAnchor.LoadUnboundAnchorsAsync(new List<Guid> { guid },
                            new List<OVRSpatialAnchor.UnboundAnchor>());
            foreach (OVRSpatialAnchor.UnboundAnchor anchor in oVRResult.Value)
            {
                if (anchor.Localized) OnLocalized(anchor);
                else
                {
                    var result = await anchor.LocalizeAsync();
                    if (result) OnLocalized(anchor);
                }
            };
        }

        private void OnLocalized(OVRSpatialAnchor.UnboundAnchor unboundAnchor)
        {
            Pose pose;
            unboundAnchor.TryGetPose(out pose);
            Vector3 oldRotation = pose.rotation.eulerAngles;
            Quaternion newRotation = Quaternion.Euler(new Vector3(0, oldRotation.y + 177, 0));
            _runner.Spawn(_synchronizationLocationPrefab, pose.position + _chessOffest, newRotation);
        }
    }
}
