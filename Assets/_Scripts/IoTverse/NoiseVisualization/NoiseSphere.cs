using System.Collections;
using UnityEngine;

public class NoiseSphere : MonoBehaviour
{
    [SerializeField] private float currentRadius = 1f;
    [SerializeField] private float minRadius = 0.1f;
    [SerializeField] private float maxRadius = 0.8f;

    public int SensorIndex;
    
    private float targetRadius;
    private Coroutine radiusCoroutine;

    private void Awake()
    {
        targetRadius = currentRadius;
        UpdateScale();
    }

    /// <summary>
    /// Set the radius of the sphere with smooth easing animation
    /// </summary>
    /// <param name="newRadius">Target radius</param>
    /// <param name="duration">Animation duration in seconds</param>
    /// <param name="easeType">Type of easing function to use</param>
    public void SetRadius(float newRadius, float duration = 0.1f, EaseType easeType = EaseType.EaseOutQuad)
    {
        targetRadius = Mathf.Clamp(newRadius, minRadius, maxRadius);
        
        if (radiusCoroutine != null)
        {
            StopCoroutine(radiusCoroutine);
        }
        
        radiusCoroutine = StartCoroutine(AnimateRadius(targetRadius, duration, easeType));
    }

    /// <summary>
    /// Set the radius instantly without animation
    /// </summary>
    public void SetRadiusInstant(float newRadius)
    {
        if (radiusCoroutine != null)
        {
            StopCoroutine(radiusCoroutine);
            radiusCoroutine = null;
        }
        
        currentRadius = Mathf.Clamp(newRadius, minRadius, maxRadius);
        targetRadius = currentRadius;
        UpdateScale();
    }

    private IEnumerator AnimateRadius(float targetRadius, float duration, EaseType easeType)
    {
        float startRadius = currentRadius;
        float elapsed = 0f;

        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / duration);
            float easedT = ApplyEase(t, easeType);
            
            currentRadius = Mathf.Lerp(startRadius, targetRadius, easedT);
            UpdateScale();
            
            yield return null;
        }

        currentRadius = targetRadius;
        UpdateScale();
        radiusCoroutine = null;
    }

    private void UpdateScale()
    {
        transform.localScale = Vector3.one * (currentRadius * 2f); // Diameter = radius * 2
    }

    private float ApplyEase(float t, EaseType easeType)
    {
        switch (easeType)
        {
            case EaseType.Linear:
                return t;
                
            case EaseType.EaseInQuad:
                return t * t;
                
            case EaseType.EaseOutQuad:
                return t * (2f - t);
                
            case EaseType.EaseInOutQuad:
                return t < 0.5f ? 2f * t * t : -1f + (4f - 2f * t) * t;
                
            case EaseType.EaseInCubic:
                return t * t * t;
                
            case EaseType.EaseOutCubic:
                float f = t - 1f;
                return f * f * f + 1f;
                
            case EaseType.EaseInOutCubic:
                return t < 0.5f ? 4f * t * t * t : (t - 1f) * (2f * t - 2f) * (2f * t - 2f) + 1f;
                
            case EaseType.EaseInElastic:
                if (t == 0f || t == 1f) return t;
                return -Mathf.Pow(2f, 10f * (t - 1f)) * Mathf.Sin((t - 1.1f) * 5f * Mathf.PI);
                
            case EaseType.EaseOutElastic:
                if (t == 0f || t == 1f) return t;
                return Mathf.Pow(2f, -10f * t) * Mathf.Sin((t - 0.1f) * 5f * Mathf.PI) + 1f;
                
            case EaseType.EaseInBounce:
                return 1f - ApplyEase(1f - t, EaseType.EaseOutBounce);
                
            case EaseType.EaseOutBounce:
                if (t < 1f / 2.75f)
                    return 7.5625f * t * t;
                else if (t < 2f / 2.75f)
                {
                    t -= 1.5f / 2.75f;
                    return 7.5625f * t * t + 0.75f;
                }
                else if (t < 2.5f / 2.75f)
                {
                    t -= 2.25f / 2.75f;
                    return 7.5625f * t * t + 0.9375f;
                }
                else
                {
                    t -= 2.625f / 2.75f;
                    return 7.5625f * t * t + 0.984375f;
                }
                
            default:
                return t;
        }
    }

    public float GetCurrentRadius() => currentRadius;
    public float GetTargetRadius() => targetRadius;
}

public enum EaseType
{
    Linear,
    EaseInQuad,
    EaseOutQuad,
    EaseInOutQuad,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    EaseInElastic,
    EaseOutElastic,
    EaseInBounce,
    EaseOutBounce
}
