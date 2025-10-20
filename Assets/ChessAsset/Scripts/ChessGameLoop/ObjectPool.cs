using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using Digiphy;

namespace ChessMainLoop
{
    public enum PathType
    {
        Yellow = 1,
        Red = 2,
    }

    /// <summary>
    /// Contains pools for objects and methods to put objects and recieve objects to and from pools.
    /// </summary>
    public  class ObjectPool : SingletonReplaceable<ObjectPool>
    {
        private Dictionary<PathType, Queue<GameObject>> _poolDictionary;
        [SerializeField]
        private List<PathPiece> _prefabs;
        private Queue<Piece> _pieces;

        private void Start()
        {
            _poolDictionary = new Dictionary<PathType, Queue<GameObject>>();
            _pieces = new Queue<Piece>();

            Queue<GameObject> queue;


            queue = new Queue<GameObject>();
            _poolDictionary.Add(PathType.Yellow, queue);
            queue = new Queue<GameObject>();
            _poolDictionary.Add(PathType.Red, queue);
        }

        /// <summary>
        /// Returns number of path objects indexed by name equal to quantity parameter. Gets objects from pool or instantiates new ones if quantity in pool isnt enough.
        /// </summary>
        /// <returns>List of path objects quantity long</returns>
        public GameObject[] GetHighlightPaths(int _quantity, PathType type)
        {
            GameObject[] _paths = new GameObject[_quantity];

            for(int i = 0; i < _quantity; i++)
            {
                if (_poolDictionary[type].Count > 0)
                {
                    _paths[i] = _poolDictionary[type].Dequeue();
                    _paths[i].SetActive(true);
                }
                else
                {
                    _paths[i]= Instantiate(_prefabs.Where(obj => obj.PathType == type).SingleOrDefault().gameObject, transform.parent);
                }
            }

            return _paths;
        }

        /// <summary>
        /// Returns a singular path object indexed by name
        /// </summary>
        /// <returns>Path object of name</returns>
        public GameObject GetHighlightPath(PathType type)
        {
            GameObject _path;
            if (_poolDictionary[type].Count > 0)
            {
                _path=_poolDictionary[type].Dequeue();
                _path.SetActive(true);
            }
            else
            {
                _path = Instantiate(_prefabs.Where(obj => obj.PathType == type).SingleOrDefault().gameObject, transform.parent);            
            }

            return _path;
        }

        /// <summary>
        /// Disables a path object and poots it back into pool
        /// </summary>
        public void RemoveHighlightPath(PathPiece _path)
        {
            _poolDictionary[_path.PathType].Enqueue(_path.gameObject);
            _path.gameObject.SetActive(false);
        }

        public void AddPiece(Piece _piece)
        {
            _pieces.Enqueue(_piece);
            _piece.gameObject.SetActive(false);
        }

        public void ResetPieces()
        {
            while (_pieces.Count > 0)
            {
                _pieces.Dequeue().gameObject.SetActive(true);
            }
        }
    }
}
