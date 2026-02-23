using UnityEngine;
using System.Collections;

public class EnemySpawner : MonoBehaviour
{
    private GameObject enemyPrefab;
    private Transform[] spawnPoints;
    private float currentDifficulty = 5f;
    private float lastDifficulty = 5f; // NEW - Track previous difficulty for gradual changes

    [Header("Spawn Settings - More Gradual")]
    public float baseSpawnRate = 4f; // Base time between spawns
    public int baseEnemyCount = 1;
    public float spawnRateReduction = 0.6f; // REDUCED from 0.8f for more gradual changes
    public float enemyCountIncrease = 1.5f; // REDUCED from 2f for more gradual changes

    [Header("Gradual Adjustment")]
    public float difficultyChangeThreshold = 0.5f; // NEW - Only adjust spawning if difficulty changes by this much
    public float maxEnemyRemovalPerUpdate = 3; // NEW - Limit how many enemies to remove at once

    [Header("Debug")]
    public bool showSpawnDebug = true;

    private Coroutine spawnCoroutine;

    public void Initialize(GameObject prefab, Transform[] spawns)
    {
        enemyPrefab = prefab;
        spawnPoints = spawns;
        StartSpawning();
    }

    public void UpdateDifficulty(float newDifficulty)
    {
        float difficultyChange = Mathf.Abs(newDifficulty - currentDifficulty);
        float oldDifficulty = currentDifficulty;
        currentDifficulty = newDifficulty;

        // NEW: Only make adjustments if difficulty change is significant enough
        if (difficultyChange < difficultyChangeThreshold)
        {
            if (showSpawnDebug)
            {
                Debug.Log($"Difficulty change too small ({difficultyChange:F2} < {difficultyChangeThreshold:F2}), " +
                         $"skipping spawn adjustment");
            }
            return;
        }

        // If difficulty dropped significantly, remove some enemies (but gradually)
        if (oldDifficulty - newDifficulty > difficultyChangeThreshold)
        {
            RemoveExcessEnemiesGradually(oldDifficulty - newDifficulty);
        }

        // Restart spawning with new parameters
        if (spawnCoroutine != null)
            StopCoroutine(spawnCoroutine);
        StartSpawning();

        if (showSpawnDebug)
        {
            LogSpawnParameters();
        }

        lastDifficulty = currentDifficulty;
    }

    void LogSpawnParameters()
    {
        float difficultyFactor = (currentDifficulty - 1f) / 9f; // 0-1 range

        // FIXED: Higher difficulty = FASTER spawning, MORE enemies
        float spawnRate = baseSpawnRate * (1f - (difficultyFactor * spawnRateReduction));
        int enemyCount = Mathf.RoundToInt(baseEnemyCount + (difficultyFactor * enemyCountIncrease));

        spawnRate = Mathf.Max(0.8f, spawnRate); // INCREASED minimum from 0.5s to 0.8s for less aggressive spawning
        enemyCount = Mathf.Max(1, enemyCount); // Minimum 1 enemy

        Debug.Log($"=== SPAWN PARAMETERS (Difficulty {currentDifficulty:F1}) ===\n" +
                 $"Difficulty Factor: {difficultyFactor:F2}\n" +
                 $"Spawn Rate: {spawnRate:F1}s (faster = harder)\n" +
                 $"Enemy Count: {enemyCount} per wave\n" +
                 $"Enemies Per Minute: ~{(60f / spawnRate) * enemyCount:F1}");
    }

    // NEW: More gradual enemy removal based on difficulty drop
    void RemoveExcessEnemiesGradually(float difficultyDrop)
    {
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");

        // Calculate how many to remove based on difficulty drop (more gradual)
        float removalFactor = Mathf.Clamp01(difficultyDrop / 3f); // Scale removal based on drop size
        int enemiesToRemove = Mathf.FloorToInt(enemies.Length * removalFactor * 0.3f); // Reduced removal rate

        // Apply maximum removal limit
        enemiesToRemove = Mathf.Min(enemiesToRemove, (int)maxEnemyRemovalPerUpdate);
        enemiesToRemove = Mathf.Min(enemiesToRemove, enemies.Length - 1); // Always keep at least 1

        for (int i = 0; i < enemiesToRemove; i++)
        {
            if (enemies[i] != null)
                Destroy(enemies[i]);
        }

        if (showSpawnDebug && enemiesToRemove > 0)
        {
            Debug.Log($"Gradually removed {enemiesToRemove} enemies due to difficulty drop of {difficultyDrop:F1} " +
                     $"({enemies.Length} -> {enemies.Length - enemiesToRemove})");
        }
    }

    // Keep the old method for emergency situations
    void RemoveExcessEnemies()
    {
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        int enemiesToRemove = Mathf.Min(enemies.Length / 2, 5);

        for (int i = 0; i < enemiesToRemove; i++)
        {
            if (enemies[i] != null)
                Destroy(enemies[i]);
        }

        Debug.Log($"Removed {enemiesToRemove} enemies due to difficulty drop");
    }

    void StartSpawning()
    {
        spawnCoroutine = StartCoroutine(SpawnEnemies());
    }

    IEnumerator SpawnEnemies()
    {
        while (true)
        {
            // Calculate spawn parameters based on difficulty
            float difficultyFactor = (currentDifficulty - 1f) / 9f; // 0-1 range

            // FIXED: Higher difficulty = FASTER spawning, MORE enemies
            float spawnRate = baseSpawnRate * (1f - (difficultyFactor * spawnRateReduction));
            int enemyCount = Mathf.RoundToInt(baseEnemyCount + (difficultyFactor * enemyCountIncrease));

            // Apply limits (more conservative)
            spawnRate = Mathf.Max(0.8f, spawnRate); // INCREASED minimum from 0.5 second to 0.8 seconds
            enemyCount = Mathf.Max(1, enemyCount); // Minimum 1 enemy

            // NEW: Check current enemy count to avoid overwhelming the player
            GameObject[] currentEnemies = GameObject.FindGameObjectsWithTag("Enemy");
            int maxEnemiesAllowed = Mathf.RoundToInt(5 + (difficultyFactor * 10)); // Scale max enemies with difficulty

            if (currentEnemies.Length >= maxEnemiesAllowed)
            {
                if (showSpawnDebug)
                {
                    Debug.Log($"Skipping spawn wave - too many enemies ({currentEnemies.Length}/{maxEnemiesAllowed})");
                }
                yield return new WaitForSeconds(spawnRate * 0.5f); // Wait half the normal time before checking again
                continue;
            }

            // Adjust enemy count if we're near the limit
            if (currentEnemies.Length + enemyCount > maxEnemiesAllowed)
            {
                enemyCount = maxEnemiesAllowed - currentEnemies.Length;
            }

            if (showSpawnDebug)
            {
                Debug.Log($"Spawning {enemyCount} enemies, next wave in {spawnRate:F1}s " +
                         $"(Difficulty: {currentDifficulty:F1}, Current enemies: {currentEnemies.Length})");
            }

            // Spawn the enemies for this wave
            for (int i = 0; i < enemyCount; i++)
            {
                SpawnEnemy();
                yield return new WaitForSeconds(0.3f); // INCREASED delay from 0.2f for more gradual spawning
            }

            // Wait for next wave
            yield return new WaitForSeconds(spawnRate);
        }
    }

    void SpawnEnemy()
    {
        if (spawnPoints.Length == 0) return;

        Transform spawnPoint = spawnPoints[Random.Range(0, spawnPoints.Length)];
        GameObject enemy = Instantiate(enemyPrefab, spawnPoint.position, spawnPoint.rotation);

        // Apply difficulty scaling to enemy
        EnemyAI enemyAI = enemy.GetComponent<EnemyAI>();
        if (enemyAI != null)
            enemyAI.SetDifficulty(currentDifficulty);

        // Notify performance tracker
        PlayerPerformanceTracker tracker = FindAnyObjectByType<PlayerPerformanceTracker>();
        if (tracker != null)
            tracker.OnEnemyEncountered();
    }

    // Manual testing methods
    [ContextMenu("Test Low Difficulty")]
    public void TestLowDifficulty()
    {
        UpdateDifficulty(1f);
    }

    [ContextMenu("Test High Difficulty")]
    public void TestHighDifficulty()
    {
        UpdateDifficulty(9f);
    }

    [ContextMenu("Test Gradual Difficulty Increase")]
    public void TestGradualIncrease()
    {
        UpdateDifficulty(currentDifficulty + 0.5f);
    }

    [ContextMenu("Test Gradual Difficulty Decrease")]
    public void TestGradualDecrease()
    {
        UpdateDifficulty(currentDifficulty - 0.5f);
    }

    [ContextMenu("Show Current Enemy Count")]
    public void ShowCurrentEnemyCount()
    {
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        Debug.Log($"Current enemy count: {enemies.Length}");
    }
}