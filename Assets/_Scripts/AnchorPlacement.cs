using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Digiphy
{
    public class AnchorPlacement : MonoBehaviour
    {
        private enum RoomType
        {
            ChessTestPosition = 1,
            ChessOfficePosition = 2,
            ChessLabPosition = 3,
            ChessLabNew = 4
        }

        [SerializeField] private OVRSpatialAnchor _anchorPrefab;
        [SerializeField] private InputActionReference _placeCommand;
        [SerializeField] private Transform _rightController;
        [SerializeField] private RoomType _roomType;
        [SerializeField] private bool _enabled;

        public bool Enabled { get => _enabled; set => _enabled = value; }

        private void Update()
        {
            if (!_enabled) return;
            if (_placeCommand.action.WasPressedThisFrame())
            {
                CreateSpatialAnchor();
            }
            else if(OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
            {
                CreateSpatialAnchor();
            }
        }

        private void CreateSpatialAnchor()
        {
            if (PlayerPrefs.HasKey(_roomType.ToString()))
            {
                LoadAnchors();
                return;
            }


            OVRSpatialAnchor anchor = Instantiate(_anchorPrefab, 
                _rightController.position,
                Quaternion.Euler(0, _rightController.rotation.y, 0));

            Canvas canvas = anchor.gameObject.GetComponentInChildren<Canvas>();
            TextMeshProUGUI uuid = canvas.gameObject.transform.GetChild(0).GetComponent<TextMeshProUGUI>();            
            TextMeshProUGUI savedStatusText = canvas.gameObject.transform.GetChild(1).GetComponent<TextMeshProUGUI>();

            StartCoroutine(CreateAnchor(anchor, uuid, savedStatusText));
        }

        private IEnumerator CreateAnchor(OVRSpatialAnchor anchor, TextMeshProUGUI uuid, TextMeshProUGUI savedStatus)
        {
            while(!anchor.Created && !anchor.Localized)
            {
                yield return new WaitForEndOfFrame();
            }

            uuid.text = anchor.Uuid.ToString();

            SaveAnchor(anchor, savedStatus);        }

        private async void SaveAnchor(OVRSpatialAnchor anchor, TextMeshProUGUI savedStatus)
        {
            var saveResult = await anchor.SaveAnchorAsync();
            savedStatus.text = saveResult.Success ? "Saved" : "Failed saving";
            if (saveResult.Success)
            {
                PlayerPrefs.SetString(_roomType.ToString(), anchor.Uuid.ToString());
                _enabled = false;
            }
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
            foreach(OVRSpatialAnchor.UnboundAnchor anchor in oVRResult.Value)
            {
                if (anchor.Localized) OnLocalized(anchor);
                else
                {
                    var result = await anchor.LocalizeAsync();
                    if (result) OnLocalized(anchor);
                }
            }; 
        }

        private async void OnLocalized(OVRSpatialAnchor.UnboundAnchor unboundAnchor)
        {
            Pose pose;
            unboundAnchor.TryGetPose(out pose);
            var spatialAnchor = Instantiate(_anchorPrefab, pose.position, pose.rotation);
            unboundAnchor.BindTo(spatialAnchor);

            if(spatialAnchor.TryGetComponent<OVRSpatialAnchor>(out var achors))
            {
                Canvas canvas = spatialAnchor.gameObject.GetComponentInChildren<Canvas>();
                TextMeshProUGUI uuid = canvas.gameObject.transform.GetChild(0).GetComponent<TextMeshProUGUI>();
                TextMeshProUGUI savedStatusText = canvas.gameObject.transform.GetChild(1).GetComponent<TextMeshProUGUI>();

                uuid.text = spatialAnchor.Uuid.ToString();
                savedStatusText.text = "Loaded from device";
                var resilt = await spatialAnchor.EraseAnchorAsync();
                if (resilt.Status == OVRAnchor.EraseResult.Success)
                {
                    PlayerPrefs.DeleteKey(_roomType.ToString());
                    savedStatusText.text = "Deleted save";
                }
            }
        }
    }
}
