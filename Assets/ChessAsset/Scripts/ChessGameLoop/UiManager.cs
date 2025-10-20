using Digiphy;
using System;
using TMPro;
using UnityEngine;

namespace ChessMainLoop
{
    public class UiManager : SingletonReplaceable<UiManager>
    {
        [SerializeField] private TextMeshPro _pieceErrorText1;
        [SerializeField] private TextMeshPro _pathErrorText1;
        [SerializeField] private TextMeshPro _distanceText1;
        [SerializeField] private TextMeshPro _timeText1;
        [SerializeField] private TextMeshPro _pieceErrorText2;
        [SerializeField] private TextMeshPro _pathErrorText2;
        [SerializeField] private TextMeshPro _distanceText2;
        [SerializeField] private TextMeshPro _timeText2;
        [SerializeField] private TextMeshPro _pieceErrorText3;
        [SerializeField] private TextMeshPro _pathErrorText3;
        [SerializeField] private TextMeshPro _distanceText3;
        [SerializeField] private TextMeshPro _timeText3;
        [SerializeField] private TextMeshPro _pieceErrorText4;
        [SerializeField] private TextMeshPro _pathErrorText4;
        [SerializeField] private TextMeshPro _distanceText4;
        [SerializeField] private TextMeshPro _timeText4;
        [SerializeField] private TextMeshPro _rtt;
        [SerializeField] private TextMeshPro _mrtt;

        public void ShowPieceError(int amount, int turn)
        {
            if (turn <= 0) return;

            if (turn % 2 == 0) turn = (turn / 2);
            else
            {
                turn++;
                turn = turn / 2;
            }

            switch (turn)
            {
                case 1:
                    _pieceErrorText1.text = "Pi: " + amount.ToString();
                    break;
                case 2:
                    _pieceErrorText2.text = "Pi: " + amount.ToString();
                    break;
                case 3:
                    _pieceErrorText3.text = "Pi: " + amount.ToString();
                    break;
                case 4:
                    _pieceErrorText4.text = "Pi: " + amount.ToString();
                    break;
            }
        }

        public void ShowPathError(int amount, int turn)
        {
            if (turn <= 0) return;

            if (turn % 2 == 0)turn = (turn / 2);
            else
            {
                turn++;
                turn = turn / 2;
            }

            switch (turn)
            {
                case 1:
                    _pathErrorText1.text = "Pa: " + amount.ToString();
                    break;
                case 2:
                    _pathErrorText2.text = "Pa: " + amount.ToString();
                    break;
                case 3:
                    _pathErrorText3.text = "Pa: " + amount.ToString();
                    break;
                case 4:
                    _pathErrorText4.text = "Pa: " + amount.ToString();
                    break;
            }
        }

        public void ShowDistance(float amount, int turn)
        {
            amount = amount * 100;
            if (turn <= 0) return;

            if (turn % 2 == 0) turn = (turn / 2);
            else
            {
                turn++;
                turn = turn / 2;
            }

            switch (turn)
            {
                case 1:
                    _distanceText1.text = "Di: " + amount.ToString("0.##");
                    break;
                case 2:
                    _distanceText2.text = "Di: " + amount.ToString("0.##");
                    break;
                case 3:
                    _distanceText3.text = "Di: " + amount.ToString("0.##");
                    break;
                case 4:
                    _distanceText4.text = "Di: " + amount.ToString("0.##");
                    break;
            }
        }

        public void ShowTime(int seconds, int turn)
        {
            if (turn <= 0) return;

            if (turn % 2 == 0) turn = (turn / 2);
            else
            {
                turn++;
                turn = turn / 2;
            }

            switch (turn)
            {
                case 1:
                    _timeText1.text = "Ti: " + seconds.ToString();
                    break;
                case 2:
                    _timeText2.text = "Ti: " + seconds.ToString();
                    break;
                case 3:
                    _timeText3.text = "Ti: " + seconds.ToString();
                    break;
                case 4:
                    _timeText4.text = "Ti: " + seconds.ToString();
                    break;
            }
        }

        public void ShowPing(float avgPing, float maxPing)
        {
            _rtt.text = "Rtt: " + avgPing.ToString("0.##");
            _mrtt.text = "MRtt: " + maxPing.ToString("0.##");
        }

        public void ShowDatas()
        {
            _pieceErrorText1.gameObject.SetActive(true);
            _pathErrorText1.gameObject.SetActive(true);
            _timeText1.gameObject.SetActive(true);
            _pieceErrorText2.gameObject.SetActive(true);
            _pathErrorText2.gameObject.SetActive(true);
            _timeText2.gameObject.SetActive(true);
            _pieceErrorText3.gameObject.SetActive(true);
            _pathErrorText3.gameObject.SetActive(true);
            _timeText3.gameObject.SetActive(true);
            _pieceErrorText4.gameObject.SetActive(true);
            _pathErrorText4.gameObject.SetActive(true);
            _timeText4.gameObject.SetActive(true);
        }

        public void HideDatas()
        {
            _pieceErrorText1.gameObject.SetActive(false);
            _pathErrorText1.gameObject.SetActive(false);
            _timeText1.gameObject.SetActive(false);
            _pieceErrorText2.gameObject.SetActive(false);
            _pathErrorText2.gameObject.SetActive(false);
            _timeText2.gameObject.SetActive(false);
            _pieceErrorText3.gameObject.SetActive(false);
            _pathErrorText3.gameObject.SetActive(false);
            _timeText3.gameObject.SetActive(false);
            _pieceErrorText4.gameObject.SetActive(false);
            _pathErrorText4.gameObject.SetActive(false);
            _timeText4.gameObject.SetActive(false);
        }

        internal void ResetData()
        {
            _pieceErrorText1.text = "Pi: 0";
            _pathErrorText1.text = "Pa: 0";
            _timeText1.text = "Ti: 0";
            _pieceErrorText2.text = "Pi: 0";
            _pathErrorText2.text = "Pa: 0";
            _timeText2.text = "Ti: 0";
            _pieceErrorText3.text = "Pi: 0";
            _pathErrorText3.text = "Pa: 0";
            _timeText3.text = "Ti: 0";
            _pieceErrorText4.text = "Pi: 0";
            _pathErrorText4.text = "Pa: 0";
            _timeText4.text = "Ti: 0";
        }
    }
}