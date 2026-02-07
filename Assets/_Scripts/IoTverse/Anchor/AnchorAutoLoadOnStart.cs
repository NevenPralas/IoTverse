using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using UnityEngine;

public class AnchorAutoLoadOnStart : MonoBehaviour
{
    [Header("Auto-load on start")]
    [Tooltip("Koliko sekundi pričekati nakon starta (Platform Init / OVR init) prije loada.")]
    [SerializeField] private float delaySeconds = 0.8f;

    [Tooltip("Ako je true, pokušava opet dok ne uspije ili do timeouta.")]
    [SerializeField] private bool retryUntilSuccess = true;

    [SerializeField] private float retryEverySeconds = 1.0f;
    [SerializeField] private float giveUpAfterSeconds = 15f;

    [Header("Optional: drag your Sample Spatial Anchor Controller here (if you can)")]
    [SerializeField] private MonoBehaviour explicitLoader; // može biti bilo koji component na tom GO-u

    // Kandidati naziva metoda koje SDK često ima (različite verzije)
    private static readonly string[] KnownLoadMethodNames =
    {
        "LoadAnchors",
        "LoadAllAnchors",
        "LoadAnchorsFromStorage",
        "LoadSavedAnchors",
        "Load",
        "LoadAsync",
        "LoadAll",
        "LoadAllAsync"
    };

    private void Start()
    {
        StartCoroutine(AutoLoadRoutine());
    }

    private IEnumerator AutoLoadRoutine()
    {
        if (delaySeconds > 0f)
            yield return new WaitForSeconds(delaySeconds);

        float t = 0f;

        while (true)
        {
            bool ok = TryInvokeLoad();

            if (ok)
            {
                Debug.Log("[AnchorAutoLoader] ✅ Load invoked.");
                yield break;
            }

            if (!retryUntilSuccess)
            {
                Debug.LogWarning("[AnchorAutoLoader] ❌ Load not invoked (retry disabled).");
                yield break;
            }

            t += retryEverySeconds;
            if (t >= giveUpAfterSeconds)
            {
                Debug.LogError("[AnchorAutoLoader] ❌ Gave up trying to invoke anchor load. (No matching method found)");
                yield break;
            }

            yield return new WaitForSeconds(retryEverySeconds);
        }
    }

    private bool TryInvokeLoad()
    {
        // 1) Ako si ručno zadao component u Inspectoru, pokušaj njega prvo
        if (explicitLoader != null)
        {
            if (InvokeLoadOnComponent(explicitLoader))
                return true;
        }

        // 2) Inače: nađi sve MonoBehaviour-e u sceni i traži one koji izgledaju kao anchor loader
        var all = FindObjectsOfType<MonoBehaviour>(true);

        // Prioritet: sve što u imenu tipa ima "SampleSpatialAnchor" ili "SpatialAnchor"
        var candidates = all
            .Where(mb => mb != null)
            .OrderByDescending(mb =>
            {
                string n = mb.GetType().Name;
                if (n.IndexOf("SampleSpatialAnchor", StringComparison.OrdinalIgnoreCase) >= 0) return 3;
                if (n.IndexOf("SpatialAnchor", StringComparison.OrdinalIgnoreCase) >= 0) return 2;
                if (n.IndexOf("Anchor", StringComparison.OrdinalIgnoreCase) >= 0) return 1;
                return 0;
            })
            .ToArray();

        foreach (var mb in candidates)
        {
            // Probaj pozvati poznate metode po imenu
            foreach (var methodName in KnownLoadMethodNames)
            {
                if (TryInvokeNoArgMethod(mb, methodName))
                {
                    Debug.Log($"[AnchorAutoLoader] Called {mb.GetType().Name}.{methodName}()");
                    return true;
                }
            }

            // Ako nema poznatih imena: probaj naći bilo koju public/protected metodu bez argumenata
            // koja sadrži "Load" i "Anchor" u nazivu.
            var methods = mb.GetType().GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

            foreach (var mi in methods)
            {
                if (mi.GetParameters().Length != 0) continue;

                string name = mi.Name;
                bool looksLikeLoad =
                    name.IndexOf("Load", StringComparison.OrdinalIgnoreCase) >= 0 &&
                    name.IndexOf("Anchor", StringComparison.OrdinalIgnoreCase) >= 0;

                if (!looksLikeLoad) continue;

                try
                {
                    mi.Invoke(mb, null);
                    Debug.Log($"[AnchorAutoLoader] Called {mb.GetType().Name}.{mi.Name}()");
                    return true;
                }
                catch { /* ignore */ }
            }
        }

        return false;
    }

    private bool InvokeLoadOnComponent(MonoBehaviour mb)
    {
        foreach (var methodName in KnownLoadMethodNames)
        {
            if (TryInvokeNoArgMethod(mb, methodName))
            {
                Debug.Log($"[AnchorAutoLoader] (explicit) Called {mb.GetType().Name}.{methodName}()");
                return true;
            }
        }

        // fallback search
        var methods = mb.GetType().GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        foreach (var mi in methods)
        {
            if (mi.GetParameters().Length != 0) continue;

            string name = mi.Name;
            bool looksLikeLoad =
                name.IndexOf("Load", StringComparison.OrdinalIgnoreCase) >= 0 &&
                name.IndexOf("Anchor", StringComparison.OrdinalIgnoreCase) >= 0;

            if (!looksLikeLoad) continue;

            try
            {
                mi.Invoke(mb, null);
                Debug.Log($"[AnchorAutoLoader] (explicit) Called {mb.GetType().Name}.{mi.Name}()");
                return true;
            }
            catch { /* ignore */ }
        }

        return false;
    }

    private bool TryInvokeNoArgMethod(MonoBehaviour mb, string methodName)
    {
        try
        {
            var mi = mb.GetType().GetMethod(methodName, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (mi == null) return false;
            if (mi.GetParameters().Length != 0) return false;

            mi.Invoke(mb, null);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
