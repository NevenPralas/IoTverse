using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace ChessMainLoop
{
    public class UIManagerGameLoop : MonoBehaviour
    {
        #region Private menu references
        [SerializeField]
        private GameObject _pauseMenu;
        [SerializeField]
        private GameObject _settingsMenu;
        [SerializeField]
        private GameObject _gameOverMenu;
        [SerializeField]
        private GameObject _pawnPromotionMenu;
        [SerializeField]
        private GameObject _pauseButton;
        [SerializeField]
        private Slider _volumeSlider;
        [SerializeField]
        private List<AudioSource> _sounds;
        [SerializeField]
        private TextMeshProUGUI _winnerText;
        [SerializeField]
        private SettingsLevels _settings;
        #endregion

        #region Private piece prefabs
        [SerializeField]
        private Queen _whiteQueen;
        [SerializeField]
        private Queen _blackQueen;
        [SerializeField]
        private Bishop _whiteBishop;
        [SerializeField]
        private Bishop _blackBishop;
        [SerializeField]
        private Rook _whiteRook;
        [SerializeField]
        private Rook _blackRook;
        [SerializeField]
        private Knight _whiteKnight;
        [SerializeField]
        private Knight _blackKnight;
        [SerializeField]
        private List<TMP_Text> _saves;
        [SerializeField]
        private GameObject _filesMenu;
        #endregion

        private SideColor _pawnColor = SideColor.None;

        private static UIManagerGameLoop _instance;
        public static UIManagerGameLoop Instance { get => _instance; }

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

            for (int i = 0; i < _saves.Count; i++)
            {
                if (File.Exists(Application.persistentDataPath + "/save" + (i + 1) + ".json") == false)
                {
                    _saves[i].SetText("Empty");
                }
            }
        }

        public void Pause()
        {
            _pauseMenu.SetActive(true);
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

        public void ReturnFromSave()
        {
            _filesMenu.SetActive(false);
            _gameOverMenu.SetActive(true);
        }

        public void VolumeChanged()
        {
            foreach(AudioSource _sound in _sounds)
            {
                _sound.volume = _volumeSlider.value;
            }
            _settings.SoundLevels = _volumeSlider.value;
        }

        public void PlayAgain()
        {
            _pauseButton.SetActive(true);
            _gameOverMenu.SetActive(false);
            GameManager.Instance.Restart();
        }

        public void GameOver(SideColor _winner)
        {
            _pauseButton.SetActive(false);
            _gameOverMenu.SetActive(true);
            if(_winner == SideColor.Both)
            {
                _winnerText.SetText("DRAW");
            }
            else
            {
                _winnerText.SetText(_winner+" WINS");
            }
        }

        public void MainMenu()
        {
            SceneManager.LoadScene("MainMenu");
        }

        public void Quit()
        {
            Application.Quit();
        }

        public void PawnPromotionMenu(SideColor _color)
        {
            _pawnColor = _color;
            _pawnPromotionMenu.SetActive(true);
        }

        //public void SelectedQueen()
        //{
        //    if (GameManager.Instance.IsPieceMoving == true)
        //    {
        //        return;
        //    }

        //    if (_pawnColor == SideColor.Black)
        //    {
        //        GameManager.Instance.SelectedPromotion(Instantiate(_blackQueen), -2);
        //    }
        //    else
        //    {
        //        GameManager.Instance.SelectedPromotion(Instantiate(_whiteQueen), -3);
        //    }
        //    _pawnColor = SideColor.None;
        //    _pawnPromotionMenu.SetActive(false);
        //}

        //public void SelectedRook()
        //{
        //    if (GameManager.Instance.IsPieceMoving == true)
        //    {
        //        return;
        //    }

        //    if (_pawnColor == SideColor.Black)
        //    {
        //        GameManager.Instance.SelectedPromotion(Instantiate(_blackRook), -4);
        //    }
        //    else
        //    {
        //        GameManager.Instance.SelectedPromotion(Instantiate(_whiteRook), -5);
        //    }
        //    _pawnColor = SideColor.None;
        //    _pawnPromotionMenu.SetActive(false);
        //}

        //public void SelectedBishop()
        //{
        //    if (GameManager.Instance.IsPieceMoving == true)
        //    {
        //        return;
        //    }

        //    if (_pawnColor == SideColor.Black)
        //    {
        //        GameManager.Instance.SelectedPromotion(Instantiate(_blackBishop), -6);
        //    }
        //    else
        //    {
        //        GameManager.Instance.SelectedPromotion(Instantiate(_whiteBishop), -7);
        //    }
        //    _pawnColor = SideColor.None;
        //    _pawnPromotionMenu.SetActive(false);
        //}

        //public void SelectedKnight()
        //{
        //    if (GameManager.Instance.IsPieceMoving == true)
        //    {
        //        return;
        //    }

        //    if (_pawnColor == SideColor.Black)
        //    {
        //        GameManager.Instance.SelectedPromotion(Instantiate(_blackKnight), -8);
        //    }
        //    else
        //    {
        //        GameManager.Instance.SelectedPromotion(Instantiate(_whiteKnight), -9);
        //    }
        //    _pawnColor = SideColor.None;
        //    _pawnPromotionMenu.SetActive(false);
        //}

        public void SaveGame()
        {
            _filesMenu.SetActive(true);
            _gameOverMenu.SetActive(false);
        }

        public void Save1()
        {
            MoveTracker.Instance.SaveGame(0);
            _filesMenu.SetActive(false);
            _gameOverMenu.SetActive(true);
            _saves[0].SetText("SAVE 1");
        }

        public void Save2()
        {
            MoveTracker.Instance.SaveGame(1);
            _filesMenu.SetActive(false);
            _gameOverMenu.SetActive(true);
            _saves[1].SetText("SAVE 2");
        }

        public void Save3()
        {
            MoveTracker.Instance.SaveGame(2);
            _filesMenu.SetActive(false);
            _gameOverMenu.SetActive(true);
            _saves[2].SetText("SAVE 3");
        }

        public void Save4()
        {
            MoveTracker.Instance.SaveGame(3);
            _filesMenu.SetActive(false);
            _gameOverMenu.SetActive(true);
            _saves[3].SetText("SAVE 4");
        }
    }
}
