// GameManager.cs - Enhanced with higher death threshold and gentler emergency handling
using UnityEngine;

public class GameManager : MonoBehaviour
{
    [Header("Game References")]
    public Transform player;
    public GameObject enemyPrefab;
    public Transform[] spawnPoints;
    public Canvas uiCanvas;

    [Header("DDA Settings")]
    public float difficultyUpdateInterval = 3f;
    public float maxDifficulty = 10f;
    public float minDifficulty = 0.5f;

    [Header("Emergency Settings - More Lenient")]
    public float emergencyHealthThreshold = 0.2f; // REDUCED from 0.3f
    public float emergencyKillRatioThreshold = 0.15f; // REDUCED from 0.2f
    public int maxConsecutiveDeaths = 10; // INCREASED from 3 to 10 deaths
    public float deathTimeWindow = 120f; // INCREASED from 60f to 120f (2 minutes)

    [Header("Emergency Response - Less Drastic")]
    public float emergencyDifficultyReduction = 2.0f; // NEW - How much to reduce difficulty in emergency
    public float emergencyHealthBoost = 0.4f; // NEW - How much health to give (40% instead of 30%)
    public float emergencyEnemyRemovalRatio = 0.5f; // NEW - What fraction of enemies to remove (50% instead of 75%)

    private DDASystem ddaSystem;
    private EnemySpawner enemySpawner;
    private PlayerPerformanceTracker performanceTracker;
    private DDAVisualizer visualizer;
    private PlayerHealth playerHealth;

    // Death tracking for emergency system
    private float[] recentDeathTimes;
    private int deathIndex = 0;

    void Start()
    {
        InitializeSystems();
        InitializeDeathTracking();
    }

    void InitializeSystems()
    {
        // Initialize DDA system
        ddaSystem = gameObject.AddComponent<DDASystem>();
        ddaSystem.Initialize(minDifficulty, maxDifficulty);

        // Initialize spawner
        enemySpawner = gameObject.AddComponent<EnemySpawner>();
        enemySpawner.Initialize(enemyPrefab, spawnPoints);

        // Initialize performance tracker
        performanceTracker = player.GetComponent<PlayerPerformanceTracker>();
        if (performanceTracker == null)
            performanceTracker = player.gameObject.AddComponent<PlayerPerformanceTracker>();

        // Initialize player health
        playerHealth = player.GetComponent<PlayerHealth>();

        // Initialize visualizer
        visualizer = gameObject.AddComponent<DDAVisualizer>();
        visualizer.Initialize(uiCanvas);

        // Sync initial difficulty across all systems
        float initialDifficulty = ddaSystem.GetCurrentDifficulty();
        performanceTracker.SetCurrentDifficulty(initialDifficulty);
        enemySpawner.UpdateDifficulty(initialDifficulty);

        // Start difficulty adjustment loop
        InvokeRepeating(nameof(UpdateDifficulty), difficultyUpdateInterval, difficultyUpdateInterval);
    }

    void InitializeDeathTracking()
    {
        recentDeathTimes = new float[maxConsecutiveDeaths];
        for (int i = 0; i < recentDeathTimes.Length; i++)
        {
            recentDeathTimes[i] = -deathTimeWindow - 1f; // Initialize to old times
        }
    }

    void Update()
    {
        // Manual difficulty testing (optional)
        if (Input.GetKeyDown(KeyCode.Alpha1)) ddaSystem.SetDifficulty(1f);
        if (Input.GetKeyDown(KeyCode.Alpha5)) ddaSystem.SetDifficulty(5f);
        if (Input.GetKeyDown(KeyCode.Alpha9)) ddaSystem.SetDifficulty(9f);

        // Check for emergency situations
        CheckEmergencyConditions();
    }

    void UpdateDifficulty()
    {
        var metrics = performanceTracker.GetCurrentMetrics();

        // Standard emergency difficulty reduction for struggling players
        bool isStruggling = metrics.healthPercentage < emergencyHealthThreshold &&
                           metrics.killRatio < emergencyKillRatioThreshold;

        if (isStruggling)
        {
            // NEW: Use gentler difficulty reduction instead of forcing large drops
            ddaSystem.ForceReduceDifficulty(0.8f); // Reduced from 1f
            Debug.Log("Standard emergency difficulty reduction activated!");
        }

        var newDifficulty = ddaSystem.CalculateNewDifficulty(metrics);

        // Apply difficulty changes
        enemySpawner.UpdateDifficulty(newDifficulty);

        // IMPORTANT: Update performance tracker with the new difficulty BEFORE updating visualizer
        performanceTracker.SetCurrentDifficulty(newDifficulty);

        // Now update visualizer with synchronized metrics
        var updatedMetrics = performanceTracker.GetCurrentMetrics();
        visualizer.UpdateDisplay(updatedMetrics, newDifficulty);

        Debug.Log($"Difficulty adjusted to: {newDifficulty:F2} | Kill Ratio: {metrics.killRatio:P0} | " +
                 $"Health: {metrics.healthPercentage:P0} | Deaths: {metrics.deathCount}");
    }

    void CheckEmergencyConditions()
    {
        // Check if player died recently (this would be called by PlayerHealth)
        if (playerHealth != null && playerHealth.IsDead())
        {
            // This is handled by PlayerHealth calling OnPlayerDeath
            return;
        }

        // Check for too many recent deaths
        if (HasTooManyRecentDeaths())
        {
            TriggerEmergencyMode();
        }
    }

    public void OnPlayerDeath()
    {
        // Record death time
        recentDeathTimes[deathIndex] = Time.time;
        deathIndex = (deathIndex + 1) % maxConsecutiveDeaths;

        int recentDeaths = CountRecentDeaths();
        Debug.Log($"Player death recorded at time {Time.time:F1}. Recent deaths: {recentDeaths}/{maxConsecutiveDeaths}");

        // Notify DDA system
        ddaSystem.OnPlayerDeath();

        // Get updated difficulty and sync across all systems
        float newDifficulty = ddaSystem.GetCurrentDifficulty();
        performanceTracker.SetCurrentDifficulty(newDifficulty);
        enemySpawner.UpdateDifficulty(newDifficulty);

        // Update visualizer with synchronized data
        var updatedMetrics = performanceTracker.GetCurrentMetrics();
        visualizer.UpdateDisplay(updatedMetrics, newDifficulty);

        // Check if emergency mode should be triggered
        if (HasTooManyRecentDeaths())
        {
            TriggerEmergencyMode();
        }
    }

    bool HasTooManyRecentDeaths()
    {
        float currentTime = Time.time;
        int recentDeaths = 0;

        for (int i = 0; i < recentDeathTimes.Length; i++)
        {
            if (currentTime - recentDeathTimes[i] <= deathTimeWindow)
            {
                recentDeaths++;
            }
        }

        return recentDeaths >= maxConsecutiveDeaths;
    }

    void TriggerEmergencyMode()
    {
        int recentDeaths = CountRecentDeaths();
        Debug.Log($"EMERGENCY MODE ACTIVATED - {recentDeaths} deaths in {deathTimeWindow}s window! " +
                 $"(Threshold: {maxConsecutiveDeaths})");

        // NEW: More gradual emergency difficulty reduction instead of dropping to minimum
        float currentDifficulty = ddaSystem.GetCurrentDifficulty();
        float newEmergencyDifficulty = Mathf.Max(minDifficulty, currentDifficulty - emergencyDifficultyReduction);

        ddaSystem.SetDifficulty(newEmergencyDifficulty);

        // Sync the emergency difficulty across all systems
        performanceTracker.SetCurrentDifficulty(newEmergencyDifficulty);

        // NEW: Remove fewer enemies - more gradual response
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        int enemiesToRemove = Mathf.FloorToInt(enemies.Length * emergencyEnemyRemovalRatio);
        enemiesToRemove = Mathf.Max(0, Mathf.Min(enemiesToRemove, enemies.Length - 1)); // Keep at least 1 enemy

        for (int i = 0; i < enemiesToRemove; i++)
        {
            if (enemies[i] != null)
                Destroy(enemies[i]);
        }

        Debug.Log($"Emergency response: Difficulty {currentDifficulty:F1} -> {newEmergencyDifficulty:F1}, " +
                 $"Removed {enemiesToRemove}/{enemies.Length} enemies");

        // Update spawner with emergency difficulty
        enemySpawner.UpdateDifficulty(newEmergencyDifficulty);

        // Give player some health if they're alive but low
        if (playerHealth != null && playerHealth.IsAlive() && playerHealth.GetHealthPercentage() < 0.6f)
        {
            float healthToAdd = playerHealth.GetMaxHealth() * emergencyHealthBoost; // Increased from 30% to 40%
            performanceTracker.AddHealth(healthToAdd);
            Debug.Log($"Emergency health boost: +{healthToAdd:F0} HP");
        }

        // Reset death tracking to prevent immediate re-triggering
        InitializeDeathTracking();
    }

    // Method to be called by PlayerHealth when player dies
    public void NotifyPlayerDeath()
    {
        OnPlayerDeath();
    }

    int CountRecentDeaths()
    {
        float currentTime = Time.time;
        int count = 0;

        for (int i = 0; i < recentDeathTimes.Length; i++)
        {
            if (currentTime - recentDeathTimes[i] <= deathTimeWindow)
            {
                ++count;
            }
        }

        return count;
    }

    // Reset game state
    [ContextMenu("Reset Game")]
    public void ResetGame()
    {
        // Reset DDA to middle difficulty
        float resetDifficulty = (minDifficulty + maxDifficulty) / 2f;
        ddaSystem.SetDifficulty(resetDifficulty);
        performanceTracker.SetCurrentDifficulty(resetDifficulty);

        // Reset performance tracker
        performanceTracker.ResetMetrics();

        // Clear enemies
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        foreach (GameObject enemy in enemies)
        {
            Destroy(enemy);
        }

        // Update spawner with reset difficulty
        enemySpawner.UpdateDifficulty(resetDifficulty);

        // Reset death tracking
        InitializeDeathTracking();

        Debug.Log($"Game state reset! Difficulty: {resetDifficulty:F1}");
    }

    // NEW: Debug methods for testing emergency conditions
    [ContextMenu("Test Emergency Mode")]
    public void TestEmergencyMode()
    {
        // Simulate recent deaths
        float currentTime = Time.time;
        for (int i = 0; i < maxConsecutiveDeaths; i++)
        {
            recentDeathTimes[i] = currentTime - (i * 5f); // Space deaths 5 seconds apart
        }

        TriggerEmergencyMode();
        Debug.Log("Emergency mode manually triggered for testing!");
    }

    [ContextMenu("Clear Death History")]
    public void ClearDeathHistory()
    {
        InitializeDeathTracking();
        Debug.Log("Death history cleared!");
    }

    [ContextMenu("Show Death Status")]
    public void ShowDeathStatus()
    {
        int recentDeaths = CountRecentDeaths();
        Debug.Log($"Recent deaths: {recentDeaths}/{maxConsecutiveDeaths} in last {deathTimeWindow}s");

        // Show individual death times
        float currentTime = Time.time;
        for (int i = 0; i < recentDeathTimes.Length; i++)
        {
            float timeSinceDeath = currentTime - recentDeathTimes[i];
            if (timeSinceDeath <= deathTimeWindow)
            {
                Debug.Log($"Death #{i}: {timeSinceDeath:F1}s ago");
            }
        }
    }
}