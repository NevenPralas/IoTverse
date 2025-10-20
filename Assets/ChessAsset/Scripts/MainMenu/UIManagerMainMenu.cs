using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace MainMenu
{
    public class UIManagerMainMenu : MonoBehaviour
    {
        [SerializeField]
        private GameObject _settingsMenu;
        [SerializeField]
        private GameObject _loadingScreen;
        [SerializeField]
        private SettingsLevels _settings;
        [SerializeField]
        private Slider _volumeSlider;
        [SerializeField]
        private List<AudioSource> _sounds;
        [SerializeField]
        private Slider _loadingSlider;
        [SerializeField]
        private TextMeshProUGUI _loadPercent;

        private static UIManagerMainMenu _instance;
        public static UIManagerMainMenu Instance { get => _instance; }

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
        }

        public void Play()
        {
            StartCoroutine(AsyncLoading("ChessGameLoop"));
            _loadingScreen.SetActive(true);
        }

        public void Replay()
        {
            StartCoroutine(AsyncLoading("ChessReplay"));
            _loadingScreen.SetActive(true);
        }

        /// <summary>
        /// Loads scene gives as parameter by name. Displays loading screen while waiting with filling progress bar.
        /// </summary>
        IEnumerator AsyncLoading(string _scene)
        {
            AsyncOperation _loading = SceneManager.LoadSceneAsync(_scene);

            while (_loading.isDone == false)
            {
                _loadingSlider.value = _loading.progress;
                _loadPercent.SetText((int)(_loading.progress * 100) + "%");
                yield return new WaitForSeconds(0.01f);
            }
        }

        public void Settings()
        {
            _settingsMenu.SetActive(true);
        }

        public void VolumeChanged()
        {
            foreach (AudioSource _sound in _sounds)
            {
                _sound.volume = _volumeSlider.value;
            }
            _settings.SoundLevels = _volumeSlider.value;
        }

        public void ReturnFromSettings()
        {
            _settingsMenu.SetActive(false);
        }

        public void Quit()
        {
            Application.Quit();
        }
    }
}
