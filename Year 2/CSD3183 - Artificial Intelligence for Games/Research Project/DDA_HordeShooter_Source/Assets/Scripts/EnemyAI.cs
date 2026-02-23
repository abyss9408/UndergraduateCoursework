// EnemyAI.cs - Basic enemy with difficulty-scaled behavior
using UnityEngine;

public class EnemyAI : MonoBehaviour
{
    [Header("Enemy Settings")]
    public float baseSpeed = 3f;
    public float baseHealth = 50f;
    public float baseDamage = 10f;
    public float attackRange = 1.5f;
    public float attackCooldown = 1f;

    private Transform player;
    private float currentHealth;
    private float lastAttackTime;
    private float difficultyMultiplier = 1f;

    // Scaled properties
    private float scaledSpeed;
    private float scaledDamage;

    void Start()
    {
        player = GameObject.FindGameObjectWithTag("Player")?.transform;
        currentHealth = baseHealth;
        ApplyDifficultyScaling();
    }

    public void SetDifficulty(float difficulty)
    {
        difficultyMultiplier = difficulty / 5f; // Normalize around 5
        ApplyDifficultyScaling();
    }

    void ApplyDifficultyScaling()
    {
        scaledSpeed = baseSpeed * (0.8f + 0.4f * difficultyMultiplier);
        scaledDamage = baseDamage * (0.7f + 0.6f * difficultyMultiplier);
        currentHealth = baseHealth * (0.8f + 0.4f * difficultyMultiplier);
    }

    void Update()
    {
        if (player == null) return;

        MoveTowardsPlayer();
        TryAttack();
    }

    void MoveTowardsPlayer()
    {
        float distanceToPlayer = Vector3.Distance(transform.position, player.position);

        if (distanceToPlayer > attackRange)
        {
            Vector3 direction = (player.position - transform.position).normalized;
            transform.position += direction * scaledSpeed * Time.deltaTime;
            transform.LookAt(player);
        }
    }

    void TryAttack()
    {
        if (Time.time - lastAttackTime < attackCooldown) return;

        float distanceToPlayer = Vector3.Distance(transform.position, player.position);
        if (distanceToPlayer <= attackRange)
        {
            AttackPlayer();
            lastAttackTime = Time.time;
        }
    }

    void AttackPlayer()
    {
        PlayerPerformanceTracker playerTracker = player.GetComponent<PlayerPerformanceTracker>();
        if (playerTracker != null)
            playerTracker.TakeDamage(scaledDamage);
    }

    public void TakeDamage(float damage)
    {
        currentHealth -= damage;
        if (currentHealth <= 0)
        {
            Die();
        }
    }

    void Die()
    {
        PlayerPerformanceTracker tracker = FindAnyObjectByType<PlayerPerformanceTracker>();
        if (tracker != null)
            tracker.OnEnemyKilled();

        Destroy(gameObject);
    }
}