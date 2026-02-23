// DDASystem.cs - More gradual difficulty adjustment logic
using UnityEngine;

public class DDASystem : MonoBehaviour
{
    private float currentDifficulty = 5f;
    private float minDifficulty;
    private float maxDifficulty;

    [Header("DDA Parameters - More Gradual")]
    public float adaptationRate = 0.8f; // REDUCED from 1.5f for more gradual changes
    public float targetWinRate = 0.4f; // Target performance level
    public float performanceWeight = 0.8f;
    public float timeWeight = 0.2f;

    [Header("Debug")]
    public bool showDebugLogs = true;

    [Header("Death Handling - Less Drastic")]
    public float deathPenalty = 0.8f; // REDUCED from 1.5f for less drastic drops
    public float maxDeathPenalty = 2.0f; // Cap on how much difficulty can drop per death
    public int deathCount = 0;

    [Header("Difficulty Change Limits")]
    public float maxDifficultyChangePerUpdate = 0.5f; // NEW - Cap on difficulty change per update
    public float difficultyChangeSmoothing = 0.7f; // NEW - Smoothing factor for changes

    public void Initialize(float min, float max)
    {
        minDifficulty = min;
        maxDifficulty = max;
        currentDifficulty = (min + max) / 2f;

        if (showDebugLogs)
            Debug.Log($"DDA System initialized: Min={min}, Max={max}, Current={currentDifficulty}");
    }

    public float CalculateNewDifficulty(PlayerMetrics metrics)
    {
        // Calculate performance score (0-1)
        float performanceScore = CalculatePerformanceScore(metrics);

        // Calculate time pressure factor
        float timePressure = CalculateTimePressure(metrics);

        // Combine factors
        float overallPerformance = (performanceScore * performanceWeight) + (timePressure * timeWeight);

        // Calculate desired difficulty change
        float performanceDelta = overallPerformance - targetWinRate;
        float rawDifficultyChange = performanceDelta * adaptationRate;

        // NEW: Apply maximum change limit to prevent drastic swings
        float cappedDifficultyChange = Mathf.Clamp(rawDifficultyChange,
            -maxDifficultyChangePerUpdate, maxDifficultyChangePerUpdate);

        // NEW: Apply smoothing to make changes more gradual
        float smoothedChange = cappedDifficultyChange * difficultyChangeSmoothing;

        float oldDifficulty = currentDifficulty;
        currentDifficulty = Mathf.Clamp(currentDifficulty + smoothedChange, minDifficulty, maxDifficulty);

        if (showDebugLogs)
        {
            Debug.Log($"DDA Update: Performance={overallPerformance:F3}, Target={targetWinRate:F3}, " +
                     $"Delta={performanceDelta:F3}, Raw Change={rawDifficultyChange:F3}, " +
                     $"Smoothed Change={smoothedChange:F3}, " +
                     $"Difficulty: {oldDifficulty:F2} -> {currentDifficulty:F2}");
        }

        return currentDifficulty;
    }

    public void OnPlayerDeath()
    {
        deathCount++;
        float oldDifficulty = currentDifficulty;

        // NEW: Scale death penalty based on current difficulty
        // Higher difficulty = larger penalty, but still capped
        float scaledPenalty = deathPenalty;
        if (currentDifficulty > 7f)
        {
            scaledPenalty = deathPenalty * 1.3f; // Slightly higher penalty at high difficulty
        }
        else if (currentDifficulty < 3f)
        {
            scaledPenalty = deathPenalty * 0.7f; // Lower penalty at low difficulty
        }

        // Apply maximum penalty cap
        scaledPenalty = Mathf.Min(scaledPenalty, maxDeathPenalty);

        // Reduce difficulty based on death penalty
        currentDifficulty = Mathf.Max(minDifficulty, currentDifficulty - scaledPenalty);

        if (showDebugLogs)
            Debug.Log($"PLAYER DEATH #{deathCount}: Scaled penalty={scaledPenalty:F2}, " +
                     $"Difficulty reduced from {oldDifficulty:F2} to {currentDifficulty:F2}");
    }

    private float CalculatePerformanceScore(PlayerMetrics metrics)
    {
        if (metrics.totalEnemiesEncountered == 0) return 0.5f;

        // Calculate component scores (0-1 range)
        float killRatio = (float)metrics.enemiesKilled / metrics.totalEnemiesEncountered;
        float healthRatio = metrics.currentHealth / metrics.maxHealth;
        float accuracyScore = metrics.accuracy;

        // Factor in death penalty - more deaths = lower performance score
        float deathPenaltyFactor = 1f;
        if (deathCount > 0 && metrics.survivalTime > 0)
        {
            float deathsPerMinute = (deathCount / (metrics.survivalTime / 60f));
            // REDUCED death penalty factor for less drastic impact
            deathPenaltyFactor = Mathf.Clamp01(1f - (deathsPerMinute * 0.1f)); // Reduced from 0.2f
        }

        // Weighted combination
        float performanceScore = (killRatio * 0.4f) + (healthRatio * 0.3f) + (accuracyScore * 0.3f);
        performanceScore *= deathPenaltyFactor;

        if (showDebugLogs && Time.frameCount % 180 == 0) // Every 3 seconds at 60fps
        {
            Debug.Log($"Performance Components: Kill={killRatio:P0}, Health={healthRatio:P0}, " +
                     $"Accuracy={accuracyScore:P0}, Deaths={deathCount}, " +
                     $"DeathPenalty={deathPenaltyFactor:F2}, Combined={performanceScore:F3}");
        }

        return Mathf.Clamp01(performanceScore);
    }

    private float CalculateTimePressure(PlayerMetrics metrics)
    {
        // Higher survival time = better performance = higher score
        float timeScore = Mathf.Clamp01(metrics.survivalTime / 300f); // 5 minutes max

        // Also consider damage rate - lower damage rate = better performance
        float damageScore = 1f - Mathf.Clamp01(metrics.damageTakenRate / 60f); // Normalize against 60 damage/min

        return (timeScore * 0.7f) + (damageScore * 0.3f);
    }

    public float GetCurrentDifficulty() => currentDifficulty;

    // NEW: More gentle emergency difficulty reduction
    public void ForceReduceDifficulty(float amount = 1.0f) // REDUCED from 2f
    {
        float oldDifficulty = currentDifficulty;
        currentDifficulty = Mathf.Max(minDifficulty, currentDifficulty - amount);

        if (showDebugLogs)
            Debug.Log($"EMERGENCY difficulty reduction: {oldDifficulty:F2} -> {currentDifficulty:F2}");
    }

    public void SetDifficulty(float newDifficulty)
    {
        currentDifficulty = Mathf.Clamp(newDifficulty, minDifficulty, maxDifficulty);
    }

    // Manual difficulty adjustment for testing
    [ContextMenu("Increase Difficulty")]
    public void IncreaseDifficulty()
    {
        SetDifficulty(currentDifficulty + 1f);
        Debug.Log($"Manually increased difficulty to: {currentDifficulty:F2}");
    }

    [ContextMenu("Decrease Difficulty")]
    public void DecreaseDifficulty()
    {
        SetDifficulty(currentDifficulty - 1f);
        Debug.Log($"Manually decreased difficulty to: {currentDifficulty:F2}");
    }

    // NEW: Context menu options for testing gradual changes
    [ContextMenu("Test Gradual Increase")]
    public void TestGradualIncrease()
    {
        SetDifficulty(currentDifficulty + 0.5f);
        Debug.Log($"Gradual increase to: {currentDifficulty:F2}");
    }

    [ContextMenu("Test Gradual Decrease")]
    public void TestGradualDecrease()
    {
        SetDifficulty(currentDifficulty - 0.5f);
        Debug.Log($"Gradual decrease to: {currentDifficulty:F2}");
    }
}