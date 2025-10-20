using ChessMainLoop;
using Fusion;
using UnityEngine;
using UnityEngine.UI;

namespace Digiphy
{
    public class SnappingPointer : SingletonReplaceable<SnappingPointer>
    {
        private Vector3 offest;
        private Transform _figure;

        private void Start()
        {
            offest = new Vector3(0, -6, 0) * transform.parent.localScale.y;
            gameObject.SetActive(false);
        }

        private void Update()
        {
            transform.position = _figure.position;
            transform.position += offest;
        }

        public void SetFigure(Transform figure)
        {
            _figure = figure.GetComponent<Piece>().visual;
            gameObject.SetActive(true);
        }

        public void Unset()
        {
            _figure = null;
            gameObject.SetActive(false);
        }
    }
}