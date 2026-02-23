// PlayerPerformanceTracker.cs - Enhanced with death tracking
using UnityEngine;

[System.Serializable]
public class PlayerMetrics
{
    [Header("Survival Metrics")]
    public float survivalTime;
    public float currentHealth;
    public float maxHealth;
    public float healthPercentage;
    public int deathCount;
    public float timeSinceLastDeath;

    [Header("Combat Metrics")]
    public int enemiesKilled;
    public int totalEnemiesEncountered;
    public float killRatio;
    public int shotsFired;
    public int shotsHit;
    public float accuracy;
    public float avgReactionTime;

    [Header("Movement Metrics")]
    public float distanceMoved;
    public float avgMovementSpeed;
    public bool isCurrentlyMoving;

    [Header("Resource Metrics")]
    public int resourcesCollected;
    public int reloadsPerformed;

    [Header("Difficulty Metrics")]
    public float currentDifficulty;
    public float timeSinceLastHit;
    public float damageTakenRate;
}

public class PlayerPerformanceTracker : MonoBehaviour
{
    [Header("Performance Settings")]
    public float maxHealth = 100f;
    public float reactionTimeWindow = 2f;

    [Header("Death Tracking")]
    public bool resetStatsOnDeath = false; // Whether to reset some stats on death
    public float deathPenaltyDuration = 30f; // How long death affects performance calculation

    [Header("Debug Info")]
    public bool showDebugInfo = true;

    // Components
    private PlayerHealth playerHealth;
    private PlayerShooting playerShooting;
    private PlayerController playerController;

    // Metrics tracking
    private PlayerMetrics metrics = new PlayerMetrics();
    private float gameStartTime;
    private Vector3 lastPosition;
    private float lastHealthUpdateTime;
    private float totalDamageTaken;
    private float lastDeathTime = -1f;

    // Reaction time tracking
    private float lastEnemySpottedTime;
    private bool enemyInSight;
    private float reactionTimeSum;
    private int reactionTimeCount;

    // Movement tracking
    private float movementStartTime;
    private float totalMovementTime;

    // Health tracking
    public float currentHealth { get; set; }

    // Death tracking
    private int currentLifeKills = 0;
    private float currentLifeStartTime = 0f;

    // Cached difficulty to avoid FindObjectOfType calls
    private float cachedDifficulty = 5f;

    void Start()
    {
        InitializeComponents();
        InitializeMetrics();
    }

    void InitializeComponents()
    {
        playerHealth = GetComponent<PlayerHealth>();
        playerShooting = GetComponent<PlayerShooting>();
        playerController = GetComponent<PlayerController>();

        if (playerHealth != null)
        {
            maxHealth = playerHealth.GetMaxHealth();
            currentHealth = playerHealth.GetCurrentHealth();
        }
        else
        {
            currentHealth = maxHealth;
        }
    }

    void InitializeMetrics()
    {
        gameStartTime = Time.time;
        currentLifeStartTime = Time.time;
        lastPosition = transform.position;
        lastHealthUpdateTime = Time.time;

        metrics.maxHealth = maxHealth;
        metrics.currentHealth = currentHealth;
        metrics.healthPercentage = 1f;
        metrics.accuracy = 0f;
        metrics.killRatio = 0f;
        metrics.timeSinceLastHit = 0f;
        metrics.deathCount = 0;
        metrics.timeSinceLastDeath = 0f;

        // Initialize with default difficulty that will be updated by GameManager
        cachedDifficulty = 5f;
        metrics.currentDifficulty = cachedDifficulty;
    }

    void Update()
    {
        UpdateTimeMetrics();
        UpdateMovementMetrics();
        UpdateHealthMetrics();
        UpdateCombatMetrics();
        UpdateReactionTime();
        UpdateDeathMetrics();

        if (showDebugInfo)
            DisplayDebugInfo();
    }

    void UpdateTimeMetrics()
    {
        metrics.survivalTime = Time.time - gameStartTime;
        metrics.timeSinceLastHit += Time.deltaTime;
    }

    void UpdateDeathMetrics()
    {
        if (lastDeathTime >= 0)
        {
            metrics.timeSinceLastDeath = Time.time - lastDeathTime;
        }
        else
        {
            metrics.timeSinceLastDeath = metrics.survivalTime;
        }
    }

    void UpdateMovementMetrics()
    {
        float distanceThisFrame = Vector3.Distance(transform.position, lastPosition);
        metrics.distanceMoved += distanceThisFrame;
        lastPosition = transform.position;

        if (playerController != null)
        {
            bool wasMoving = metrics.isCurrentlyMoving;
            metrics.isCurrentlyMoving = playerController.IsMoving();

            if (metrics.isCurrentlyMoving)
            {
                if (!wasMoving)
                    movementStartTime = Time.time;
                totalMovementTime += Time.deltaTime;
            }
        }

        if (totalMovementTime > 0)
            metrics.avgMovementSpeed = metrics.distanceMoved / totalMovementTime;
    }

    void UpdateHealthMetrics()
    {
        if (playerHealth != null)
        {
            float newHealth = playerHealth.GetCurrentHealth();

            if (newHealth < currentHealth)
            {
                float damageTaken = currentHealth - newHealth;
                totalDamageTaken += damageTaken;
                metrics.timeSinceLastHit = 0f;
                OnDamageTaken(damageTaken);
            }

            currentHealth = newHealth;
        }

        metrics.currentHealth = currentHealth;
        metrics.healthPercentage = currentHealth / maxHealth;

        float timeAlive = Time.time - gameStartTime;
        if (timeAlive > 0)
            metrics.damageTakenRate = (totalDamageTaken / timeAlive) * 60f;
    }

    void UpdateCombatMetrics()
    {
        if (metrics.totalEnemiesEncountered > 0)
            metrics.killRatio = (float)metrics.enemiesKilled / metrics.totalEnemiesEncountered;
        else
            metrics.killRatio = 0f;

        if (metrics.shotsFired > 0)
            metrics.accuracy = (float)metrics.shotsHit / metrics.shotsFired;
        else
            metrics.accuracy = 0f;

        if (reactionTimeCount > 0)
            metrics.avgReactionTime = reactionTimeSum / reactionTimeCount;
    }

    void UpdateReactionTime()
    {
        Collider[] nearbyEnemies = Physics.OverlapSphere(transform.position, 10f);
        bool enemyVisible = false;

        foreach (Collider col in nearbyEnemies)
        {
            if (col.CompareTag("Enemy") || col.GetComponent<EnemyAI>() != null)
            {
                Vector3 directionToEnemy = (col.transform.position - transform.position).normalized;
                if (Vector3.Dot(transform.forward, directionToEnemy) > 0.5f)
                {
                    enemyVisible = true;
                    break;
                }
            }
        }

        if (enemyVisible && !enemyInSight)
        {
            lastEnemySpottedTime = Time.time;
        }

        enemyInSight = enemyVisible;
    }

    // Death and respawn handling
    public void OnPlayerDeath()
    {
        metrics.deathCount++;
        lastDeathTime = Time.time;
        currentLifeKills = 0;
        currentLifeStartTime = Time.time;

        Debug.Log($"Player death #{metrics.deathCount} recorded. Total survival time: {metrics.survivalTime:F1}s");

        // Optional: Reset certain stats on death
        if (resetStatsOnDeath)
        {
            // Reset current life stats
            metrics.timeSinceLastHit = 0f;
        }
    }

    public void OnPlayerRespawn()
    {
        currentHealth = maxHealth;
        metrics.currentHealth = currentHealth;
        metrics.healthPercentage = 1f;
        currentLifeStartTime = Time.time;

        Debug.Log($"Player respawned. Deaths so far: {metrics.deathCount}");
    }

    // Public methods for other scripts
    public void OnShotFired()
    {
        metrics.shotsFired++;

        if (enemyInSight && Time.time - lastEnemySpottedTime < reactionTimeWindow)
        {
            float reactionTime = Time.time - lastEnemySpottedTime;
            reactionTimeSum += reactionTime;
            reactionTimeCount++;
        }
    }

    public void OnShotHit()
    {
        metrics.shotsHit++;
    }

    public void OnEnemyKilled()
    {
        metrics.enemiesKilled++;
        currentLifeKills++;
    }

    public void OnEnemyEncountered()
    {
        metrics.totalEnemiesEncountered++;
    }

    public void OnResourceCollected()
    {
        metrics.resourcesCollected++;
    }

    public void OnReloadPerformed()
    {
        metrics.reloadsPerformed++;
    }

    public void TakeDamage(float damage)
    {
        if (playerHealth != null)
        {
            playerHealth.TakeDamage(damage);
        }
        else
        {
            currentHealth = Mathf.Max(0, currentHealth - damage);
            totalDamageTaken += damage;
            metrics.timeSinceLastHit = 0f;
            OnDamageTaken(damage);
        }
    }

    void OnDamageTaken(float damage)
    {
        Debug.Log($"Player took {damage} damage. Health: {currentHealth}/{maxHealth}");
    }

    public PlayerMetrics GetCurrentMetrics()
    {
        metrics.currentDifficulty = cachedDifficulty;
        return metrics;
    }

    // Method to set difficulty from GameManager (eliminates FindObjectOfType)
    public void SetCurrentDifficulty(float difficulty)
    {
        cachedDifficulty = difficulty;
        metrics.currentDifficulty = difficulty;
    }

    // Get performance score adjusted for recent deaths
    public float GetAdjustedPerformanceScore()
    {
        var currentMetrics = GetCurrentMetrics();

        // Base performance
        float basePerformance = 0.5f;
        if (currentMetrics.totalEnemiesEncountered > 0)
        {
            float killRatio = (float)currentMetrics.enemiesKilled / currentMetrics.totalEnemiesEncountered;
            float healthRatio = currentMetrics.currentHealth / currentMetrics.maxHealth;
            float accuracyScore = currentMetrics.accuracy;
            basePerformance = (killRatio * 0.4f) + (healthRatio * 0.3f) + (accuracyScore * 0.3f);
        }

        // Apply death penalty if recent death
        if (currentMetrics.timeSinceLastDeath < deathPenaltyDuration)
        {
            float deathPenalty = 1f - (currentMetrics.timeSinceLastDeath / deathPenaltyDuration);
            basePerformance *= (1f - deathPenalty * 0.5f); // Up to 50% reduction
        }

        return Mathf.Clamp01(basePerformance);
    }

    void DisplayDebugInfo()
    {
        if (!showDebugInfo) return;

        string debugText = $"=== PLAYER PERFORMANCE ===\n" +
                          $"Survival Time: {metrics.survivalTime:F1}s\n" +
                          $"Deaths: {metrics.deathCount}\n" +
                          $"Time Since Death: {metrics.timeSinceLastDeath:F1}s\n" +
                          $"Health: {metrics.currentHealth:F0}/{metrics.maxHealth:F0} ({metrics.healthPercentage:P0})\n" +
                          $"Kills: {metrics.enemiesKilled}/{metrics.totalEnemiesEncountered} ({metrics.killRatio:P0})\n" +
                          $"This Life Kills: {currentLifeKills}\n" +
                          $"Accuracy: {metrics.shotsHit}/{metrics.shotsFired} ({metrics.accuracy:P0})\n" +
                          $"Distance: {metrics.distanceMoved:F1}m\n" +
                          $"Difficulty: {metrics.currentDifficulty:F1}";

        if (Time.frameCount % 60 == 0)
        {
            Debug.Log(debugText);
        }
    }

    public void ResetMetrics()
    {
        float preservedDifficulty = cachedDifficulty; // Preserve current difficulty

        metrics = new PlayerMetrics();
        InitializeMetrics();

        // Restore the difficulty
        cachedDifficulty = preservedDifficulty;
        metrics.currentDifficulty = preservedDifficulty;

        gameStartTime = Time.time;
        currentLifeStartTime = Time.time;
        totalDamageTaken = 0f;
        reactionTimeSum = 0f;
        reactionTimeCount = 0;
        totalMovementTime = 0f;
        lastDeathTime = -1f;
        currentLifeKills = 0;
    }

    public void AddHealth(float amount)
    {
        if (playerHealth != null)
        {
            currentHealth = playerHealth.GetCurrentHealth();
        }
        else
        {
            currentHealth = Mathf.Min(maxHealth, currentHealth + amount);
        }
    }
}