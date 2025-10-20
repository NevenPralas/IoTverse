using Fusion;
using UnityEngine;

namespace Digiphy
{
    public abstract class Singleton<T> : MonoBehaviour where T : MonoBehaviour
    {
        private static T _instance = null;
        public static T Instance { get => _instance; private set => _instance = value; }
        protected virtual void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Destroy(this.gameObject);
            }
            else
            {
                Instance = this as T;
            }

            Init();
        }

        protected virtual void Init()
        {
            if (Instance == null) Instance = this as T;
        }

        protected virtual void OnApplicationQuit()
        {
            Instance = null;
            Destroy(gameObject);
        }
    }

    public abstract class SingletonReplaceable<T> : MonoBehaviour where T : MonoBehaviour
    {
        private static T _instance = null;
        public static T Instance { get => _instance; private set => _instance = value; }
        protected virtual void Awake()
        {
            Instance = this as T;

            Init();
        }

        protected virtual void Init()
        {
            if (Instance == null) Instance = this as T;
        }

        protected virtual void OnApplicationQuit()
        {
            Instance = null;
            Destroy(gameObject);
        }
    }

    public abstract class SingletonNetworkedReplaceable<T> : NetworkBehaviour where T : NetworkBehaviour
    {
        private static T _instance = null;
        public static T Instance { get => _instance; private set => _instance = value; }
        public override void Spawned()
        {
            base.Spawned();

            //if (_instance != null && _instance != this && _instance.Object)
            //{
            //    Runner.Despawn(_instance.Object);
            //}

            Instance = this as T;
            Init();
        }

        protected virtual void Init()
        {
            if (Instance == null) Instance = this as T;
        }

        protected virtual void OnApplicationQuit()
        {
            Instance = null;
            //if(Object) Runner.Despawn(Object);
        }
    }

    public abstract class SingletonNetworked<T> : NetworkBehaviour where T : NetworkBehaviour
    {
        private static T _instance = null;
        public static T Instance { get => _instance; private set => _instance = value; }
        public override void Spawned()
        {
            base.Spawned();

            //if (_instance != null && _instance != this && _instance.Object)
            //{
            //    Runner.Despawn(Object);
            //}
            //else
            //{
                Instance = this as T;
            //}

            Init();
        }

        protected virtual void Init()
        {
            if (Instance == null) Instance = this as T;
        }

        protected virtual void OnApplicationQuit()
        {
            Instance = null;
            //if(Object) Runner.Despawn(Object);
        }
    }

    public abstract class SingletonPersistent<T> : Singleton<T> where T : MonoBehaviour
    {
        protected virtual void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this.gameObject);
            }
            else
            {
                DontDestroyOnLoad(gameObject);
                base.Awake();
            }

            Init();
        }
    }
}
