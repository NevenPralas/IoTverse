using System;
using System.Collections.Generic;
using UnityEngine;

public class UnityMainThreadDispatcher : MonoBehaviour
{
    private static UnityMainThreadDispatcher instance;
    private readonly Queue<Action> executionQueue = new Queue<Action>();

    public static UnityMainThreadDispatcher Instance
    {
        get
        {
            if (instance == null)
            {
                // Try to find existing instance
                instance = FindObjectOfType<UnityMainThreadDispatcher>();

                // Create new instance if none exists
                if (instance == null)
                {
                    GameObject go = new GameObject("UnityMainThreadDispatcher");
                    instance = go.AddComponent<UnityMainThreadDispatcher>();
                    DontDestroyOnLoad(go);
                }
            }
            return instance;
        }
    }

    private void Awake()
    {
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else if (instance != this)
        {
            Destroy(gameObject);
        }
    }

    private void Update()
    {
        // Execute all queued actions on the main thread
        lock (executionQueue)
        {
            while (executionQueue.Count > 0)
            {
                executionQueue.Dequeue()?.Invoke();
            }
        }
    }

    public void Enqueue(Action action)
    {
        if (action == null) return;

        lock (executionQueue)
        {
            executionQueue.Enqueue(action);
        }
    }
}
