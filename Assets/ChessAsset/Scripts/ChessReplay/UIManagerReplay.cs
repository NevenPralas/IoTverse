using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace ChessReplay
{
    public class UIManagerReplay : MonoBehaviour
    {
        #region Private menu references
        [SerializeField]
        private GameObject _pauseMenu;
        [SerializeField]
        private GameObject _settingsMenu;
        [SerializeField]
        private GameObject _pauseButton;
        [SerializeField]
        private Slider _volumeSlider;
        [SerializeField]
        private List<AudioSource> _sounds;
        [SerializeField]
        private SettingsLevels _settings;
        [SerializeField]
        private ReplayController _replayController;
        [SerializeField]
        private TMP_InputField _autplaySpeed;
        [SerializeField]
        private List<TMP_Text> _saves;
        [SerializeField]
        private GameObject _filesMenu;
        #endregion

        private static UIManagerReplay _instance;
        public static UIManagerReplay Instance { get => _instance; }

        private void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Destroy(this.gameObject);
            }
            else
            {
                _instance = this;
            }
        }

        private void Start()
        {
            foreach (AudioSource _sound in _sounds)
            {
                _sound.volume = _settings.SoundLevels;
            }
            _volumeSlider.value = _settings.SoundLevels;

            for(int i = 0; i < _saves.Count; i++)
            {
                if (File.Exists(Application.persistentDataPath + "/save" + (i + 1) + ".json") == false){
                    _saves[i].SetText("Empty");
                }
            }
        }

        public void Pause()
        {
            _pauseMenu.SetActive(true);
            _replayController.StopAutoplay();
        }

        public void Resume()
        {
            _pauseMenu.SetActive(false);
        }

        public void Settings()
        {
            _settingsMenu.SetActive(true);
        }

        public void ReturnFromSettings()
        {
            _settingsMenu.SetActive(false);
        }

        public void VolumeChanged()
        {
            foreach (AudioSource _sound in _sounds)
            {
                _sound.volume = _volumeSlider.value;
            }
            _settings.SoundLevels = _volumeSlider.value;
        }

        public void MainMenu()
        {
            SceneManager.LoadScene("MainMenu");
        }

        public void Quit()
        {
            Application.Quit();
        }

        public void AutoplaySpeedChange()
        {
            try
            {
                float.Parse(_autplaySpeed.text);
                _replayController.TurnSpeed = float.Parse(_autplaySpeed.text);
            }
            catch
            {

            }
        }

        public void NextTurn()
        {
            _replayController.NextTurn();
        }

        public void LastTurn()
        {
            _replayController.LastTurn();
        }

        public void StartAutoplay()
        {
            _replayController.StartAutoPlay();
        }

        public void Save1()
        {
            if(string.Compare(_saves[0].text, "Empty") == 0)
            {
                return;
            }

            _replayController.Initialize(0);
            _filesMenu.SetActive(false);
        }

        public void Save2()
        {
            if (string.Compare(_saves[1].text, "Empty") == 0)
            {
                return;
            }

            _replayController.Initialize(1);
            _filesMenu.SetActive(false);
        }

        public void Save3()
        {
            if (string.Compare(_saves[2].text, "Empty") == 0)
            {
                return;
            }

            _replayController.Initialize(2);
            _filesMenu.SetActive(false);
        }

        public void Save4()
        {
            if (string.Compare(_saves[3].text, "Empty") == 0)
            {
                return;
            }

            _replayController.Initialize(3);
            _filesMenu.SetActive(false);
        }
    }
}
