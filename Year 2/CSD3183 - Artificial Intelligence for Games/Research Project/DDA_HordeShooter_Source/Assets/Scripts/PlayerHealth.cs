// PlayerHealth.cs - Updated with respawn immunity system
using UnityEngine;
using System.Collections;

public class PlayerHealth : MonoBehaviour
{
    [Header("Health Settings")]
    public float maxHealth = 100f;
    public float healthRegenRate = 5f;
    public float regenDelay = 3f;

    [Header("Respawn Immunity")]
    public float immunityDuration = 3f; // 3 seconds of immunity after respawn
    public float flashRate = 0.2f; // How fast to flash during immunity
    public Color immunityColor = new Color(1f, 1f, 1f, 0.5f); // Semi-transparent white

    [Header("Visual Effects")]
    public GameObject damageEffect;
    public float damageEffectDuration = 0.5f;

    [Header("Death Settings")]
    public float respawnDelay = 2f;
    public Vector3 respawnPosition = Vector3.up;

    private float currentHealth;
    private float lastDamageTime;
    private PlayerPerformanceTracker performanceTracker;
    private Renderer playerRenderer;
    private Color originalColor;
    private bool isDead = false;
    private UIManager uiManager; // Reference to UI manager

    // Immunity system
    private bool isImmune = false;
    private float immunityStartTime;
    private Coroutine immunityFlashCoroutine;

    void Start()
    {
        currentHealth = maxHealth;
        performanceTracker = GetComponent<PlayerPerformanceTracker>();
        playerRenderer = GetComponentInChildren<Renderer>();
        uiManager = FindAnyObjectByType<UIManager>(); // Cache UI manager

        if (playerRenderer != null)
            originalColor = playerRenderer.material.color;

        if (performanceTracker != null)
        {
            performanceTracker.maxHealth = maxHealth;
            performanceTracker.currentHealth = currentHealth;
        }
    }

    void Update()
    {
        if (!isDead)
        {
            HandleHealthRegen();
            UpdatePerformanceTracker();
            UpdateImmunity();
        }
    }

    void UpdateImmunity()
    {
        if (isImmune)
        {
            float immunityElapsed = Time.time - immunityStartTime;

            if (immunityElapsed >= immunityDuration)
            {
                EndImmunity();
            }
        }
    }

    void HandleHealthRegen()
    {
        if (currentHealth < maxHealth && Time.time - lastDamageTime > regenDelay)
        {
            currentHealth += healthRegenRate * Time.deltaTime;
            currentHealth = Mathf.Clamp(currentHealth, 0, maxHealth);
        }
    }

    void UpdatePerformanceTracker()
    {
        if (performanceTracker != null)
        {
            performanceTracker.currentHealth = currentHealth;
        }
    }

    public void TakeDamage(float damage)
    {
        if (isDead) return;

        // Check immunity - prevent damage during immunity period
        if (isImmune)
        {
            Debug.Log($"Player is immune! Damage blocked. Immunity time remaining: {GetImmunityTimeRemaining():F1}s");
            return;
        }

        currentHealth -= damage;
        currentHealth = Mathf.Clamp(currentHealth, 0, maxHealth);
        lastDamageTime = Time.time;

        ShowDamageEffect();

        Debug.Log($"Player took {damage} damage. Health: {currentHealth}/{maxHealth}");

        if (currentHealth <= 0)
        {
            Die();
        }
    }

    void ShowDamageEffect()
    {
        // Use UIManager for damage effect instead of local gameobject
        if (uiManager != null)
        {
            uiManager.ShowDamageEffect();
        }

        // Keep the old system as fallback
        if (damageEffect != null)
        {
            damageEffect.SetActive(true);
            Invoke(nameof(HideDamageEffect), damageEffectDuration);
        }

        // Flash red (only if not immune - don't interfere with immunity flashing)
        if (playerRenderer != null && !isImmune)
        {
            StartCoroutine(DamageFlash());
        }
    }

    void HideDamageEffect()
    {
        if (damageEffect != null)
            damageEffect.SetActive(false);
    }

    System.Collections.IEnumerator DamageFlash()
    {
        if (playerRenderer != null)
        {
            playerRenderer.material.color = Color.red;
            yield return new WaitForSeconds(0.1f);

            // Only restore original color if not immune (immunity has its own flashing)
            if (!isImmune)
            {
                playerRenderer.material.color = originalColor;
            }
        }
    }

    void Die()
    {
        if (isDead) return;

        isDead = true;

        // End any active immunity
        EndImmunity();

        Debug.Log("Player died!");

        GameManager gameManager = FindAnyObjectByType<GameManager>();
        if (gameManager != null)
        {
            gameManager.NotifyPlayerDeath();
        }
        else
        {
            DDASystem ddaSystem = FindAnyObjectByType<DDASystem>();
            if (ddaSystem != null)
            {
                ddaSystem.OnPlayerDeath();
            }
        }

        if (performanceTracker != null)
        {
            performanceTracker.OnPlayerDeath();
        }

        if (playerRenderer != null)
        {
            playerRenderer.material.color = Color.gray;
        }

        PlayerController controller = GetComponent<PlayerController>();
        if (controller != null)
        {
            controller.enabled = false;
        }

        PlayerShooting shooting = GetComponent<PlayerShooting>();
        if (shooting != null)
        {
            shooting.enabled = false;
        }

        Invoke(nameof(Respawn), respawnDelay);
    }

    void Respawn()
    {
        isDead = false;
        currentHealth = maxHealth;
        transform.position = respawnPosition;

        Debug.Log("Player respawned with immunity!");

        if (playerRenderer != null)
        {
            playerRenderer.material.color = originalColor;
        }

        PlayerController controller = GetComponent<PlayerController>();
        if (controller != null)
        {
            controller.enabled = true;
        }

        PlayerShooting shooting = GetComponent<PlayerShooting>();
        if (shooting != null)
        {
            shooting.enabled = true;
        }

        if (performanceTracker != null)
        {
            performanceTracker.OnPlayerRespawn();
        }

        // Start immunity period after respawn
        StartImmunity();

        EnemySpawner spawner = FindAnyObjectByType<EnemySpawner>();
        DDASystem ddaSystem = FindAnyObjectByType<DDASystem>();
        if (spawner != null && ddaSystem != null)
        {
            spawner.UpdateDifficulty(ddaSystem.GetCurrentDifficulty());
        }
    }

    void StartImmunity()
    {
        isImmune = true;
        immunityStartTime = Time.time;

        Debug.Log($"Immunity started! Duration: {immunityDuration}s");

        // Start immunity visual effect
        if (immunityFlashCoroutine != null)
        {
            StopCoroutine(immunityFlashCoroutine);
        }
        immunityFlashCoroutine = StartCoroutine(ImmunityFlashEffect());
    }

    void EndImmunity()
    {
        if (!isImmune) return;

        isImmune = false;

        Debug.Log("Immunity ended!");

        // Stop immunity visual effect
        if (immunityFlashCoroutine != null)
        {
            StopCoroutine(immunityFlashCoroutine);
            immunityFlashCoroutine = null;
        }

        // Restore original appearance
        if (playerRenderer != null)
        {
            playerRenderer.material.color = originalColor;
        }
    }

    System.Collections.IEnumerator ImmunityFlashEffect()
    {
        while (isImmune)
        {
            if (playerRenderer != null)
            {
                // Flash between immunity color and original color
                playerRenderer.material.color = immunityColor;
                yield return new WaitForSeconds(flashRate);

                playerRenderer.material.color = originalColor;
                yield return new WaitForSeconds(flashRate);
            }
            else
            {
                yield return new WaitForSeconds(flashRate);
            }
        }
    }

    // Public getters
    public float GetCurrentHealth() => currentHealth;
    public float GetMaxHealth() => maxHealth;
    public float GetHealthPercentage() => currentHealth / maxHealth;
    public bool IsAlive() => !isDead;
    public bool IsDead() => isDead;
    public bool IsImmune() => isImmune;
    public float GetImmunityTimeRemaining() => isImmune ? Mathf.Max(0, immunityDuration - (Time.time - immunityStartTime)) : 0f;

    // Debug and testing methods
    [ContextMenu("Force Death")]
    public void ForceDeath()
    {
        TakeDamage(currentHealth);
    }

    [ContextMenu("Force Respawn")]
    public void ForceRespawn()
    {
        if (isDead)
        {
            CancelInvoke(nameof(Respawn));
            Respawn();
        }
    }

    [ContextMenu("Test Immunity")]
    public void TestImmunity()
    {
        if (!isDead && !isImmune)
        {
            StartImmunity();
            Debug.Log("Immunity manually activated for testing!");
        }
    }

    [ContextMenu("End Immunity")]
    public void ForceEndImmunity()
    {
        EndImmunity();
        Debug.Log("Immunity manually ended!");
    }

    [ContextMenu("Test Damage During Immunity")]
    public void TestDamageDuringImmunity()
    {
        if (isImmune)
        {
            TakeDamage(50f);
            Debug.Log($"Attempted damage during immunity. Should be blocked! Time remaining: {GetImmunityTimeRemaining():F1}s");
        }
        else
        {
            Debug.Log("Player is not immune. Start immunity first with 'Test Immunity'");
        }
    }

    // Method to add health (used by emergency system)
    public void AddHealth(float amount)
    {
        currentHealth = Mathf.Min(maxHealth, currentHealth + amount);
        Debug.Log($"Health added: +{amount}. Current: {currentHealth}/{maxHealth}");
    }
}