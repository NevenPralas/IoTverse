using Fusion;
using OVR.OpenVR;
using Photon.Voice;
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Digiphy
{
    public class ArSpawner : Singleton<ArSpawner>
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
        [SerializeField] private Slider _rotationSlider;
        private NetworkRunner _runner;
        private NetworkObject _spawnedRoom;
        private GameObject _avatar;

        private void Start()
        {
            _rotationSlider.value = PlayerPrefs.GetFloat("rotation");
            _rotationSlider.onValueChanged.AddListener(RotationSliderChanged);
        }

        private void RotationSliderChanged(float value)
        {
            float rotation = 360 * value;
            _spawnedRoom.transform.GetComponent<NetworkTransform>().enabled = false;
            _spawnedRoom.transform.rotation = Quaternion.Euler(new Vector3(0, _spawnedRoom.transform.rotation.y + rotation, 0));
            PlayerPrefs.SetFloat("rotation", value);
        }

        public void JoinedNetworkSession(NetworkRunner runner)
        {
            _runner = runner;
            LoadAnchors();
        }

        private void Update()
        {
            if (_avatar != null) return;
            _avatar = GameObject.Find("S0_L0_M1_V0_optimized_geom,0");
            if (_avatar == null) return;

            MeshRenderer renderer = _avatar.GetComponent<MeshRenderer>();
            renderer.enabled = false;
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
            float rotation = 360 * _rotationSlider.value;
            Quaternion newRotation = Quaternion.Euler(new Vector3(0, oldRotation.y + rotation, 0));
            _spawnedRoom = _runner.Spawn(_synchronizationLocationPrefab, pose.position + _chessOffest, newRotation);
        }
    }
}
