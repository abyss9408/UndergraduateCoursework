// UIManager.cs - Centralized UI management with proper positioning
using UnityEngine;
using UnityEngine.UI;

public class UIManager : MonoBehaviour
{
    [Header("UI Panels")]
    public Canvas mainCanvas;
    public GameObject healthPanel;
    public GameObject ammoPanel; // RE-ADDED
    public GameObject debugPanel;
    public GameObject damageOverlay;
    public GameObject crosshairPanel; // NEW - Simple crosshair

    [Header("Font Sizes")]
    public int healthTextSize = 18;
    public int ammoTextSize = 20; // RE-ADDED
    public int reloadTextSize = 16; // RE-ADDED
    public int debugTextSize = 14;

    [Header("Crosshair Settings")]
    public bool showCrosshair = true;
    public Color crosshairColor = Color.white;
    public int crosshairSize = 20;
    public int crosshairThickness = 2;

    [Header("Debug Settings")]
    public bool showDebugUI = true;
    public KeyCode toggleDebugKey = KeyCode.F1;

    [Header("Health UI")]
    public Slider healthBar;
    public Text healthText;
    public Image healthFill;

    [Header("Ammo UI")] // RE-ADDED
    public Text ammoText;
    public Text reloadText;

    [Header("Debug UI")]
    public Text performanceText;
    public Text gameManagerText;
    public Text spawnerText;
    public Text controlsText;

    [Header("Damage Effect")]
    public Image damageOverlayImage;
    public float damageAlpha = 0.3f; // Reduced from full opacity

    private PlayerHealth playerHealth;
    private PlayerShooting playerShooting;
    private PlayerPerformanceTracker performanceTracker;
    private GameManager gameManager;
    private EnemySpawner enemySpawner;

    void Start()
    {
        SetupCanvas();
        SetupUIReferences();
        CreateUILayout();
        CacheComponents();
    }

    void SetupCanvas()
    {
        if (mainCanvas == null)
            mainCanvas = FindAnyObjectByType<Canvas>();

        // Ensure resolution independence
        CanvasScaler scaler = mainCanvas.GetComponent<CanvasScaler>();
        if (scaler == null)
            scaler = mainCanvas.gameObject.AddComponent<CanvasScaler>();

        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
        scaler.matchWidthOrHeight = 0.5f; // Balance between width and height

        // Also ensure proper render mode
        mainCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
        mainCanvas.sortingOrder = 100; // Make sure UI renders on top
    }

    // Helper method to get safe area margins (useful for mobile/different aspect ratios)
    Vector4 GetSafeAreaMargins()
    {
        Rect safeArea = Screen.safeArea;
        Vector4 margins = new Vector4();

        // Calculate margins as percentage of screen
        margins.x = safeArea.x / Screen.width; // Left margin
        margins.y = safeArea.y / Screen.height; // Bottom margin
        margins.z = (Screen.width - safeArea.width - safeArea.x) / Screen.width; // Right margin  
        margins.w = (Screen.height - safeArea.height - safeArea.y) / Screen.height; // Top margin

        // Add extra padding for better visual spacing
        margins.x = Mathf.Max(margins.x, 0.02f); // Minimum 2% margin
        margins.y = Mathf.Max(margins.y, 0.02f);
        margins.z = Mathf.Max(margins.z, 0.02f);
        margins.w = Mathf.Max(margins.w, 0.02f);

        return margins;
    }

    void CreateUILayout()
    {
        CreateHealthPanel();
        CreateAmmoPanel(); // RE-ADDED - Keep the bottom-left ammo UI
        CreateDebugPanel();
        CreateCrosshair(); // NEW - Simple crosshair
        SetupDamageOverlay();
    }

    void CreateHealthPanel()
    {
        if (healthPanel == null)
        {
            healthPanel = new GameObject("HealthPanel");
            healthPanel.transform.SetParent(mainCanvas.transform, false);

            RectTransform rect = healthPanel.AddComponent<RectTransform>();
            // Top-RIGHT corner with proper safe margins
            rect.anchorMin = new Vector2(0.68f, 0.85f); // Start from right side
            rect.anchorMax = new Vector2(0.98f, 0.98f); // End near right edge
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;

            CreateHealthBar(healthPanel);
        }
    }

    void CreateHealthBar(GameObject parent)
    {
        // Background
        GameObject bgObj = new GameObject("HealthBG");
        bgObj.transform.SetParent(parent.transform, false);
        RectTransform bgRect = bgObj.AddComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.offsetMin = Vector2.zero;
        bgRect.offsetMax = Vector2.zero;

        Image bgImage = bgObj.AddComponent<Image>();
        bgImage.color = new Color(0, 0, 0, 0.7f);

        // Health slider
        GameObject sliderObj = new GameObject("HealthSlider");
        sliderObj.transform.SetParent(parent.transform, false);
        RectTransform sliderRect = sliderObj.AddComponent<RectTransform>();
        sliderRect.anchorMin = new Vector2(0, 0.4f);
        sliderRect.anchorMax = new Vector2(1, 0.8f);
        sliderRect.offsetMin = new Vector2(10, 0);
        sliderRect.offsetMax = new Vector2(-10, 0);

        healthBar = sliderObj.AddComponent<Slider>();
        healthBar.minValue = 0;
        healthBar.maxValue = 1;

        // Fill area
        GameObject fillArea = new GameObject("Fill Area");
        fillArea.transform.SetParent(sliderObj.transform, false);
        RectTransform fillRect = fillArea.AddComponent<RectTransform>();
        fillRect.anchorMin = Vector2.zero;
        fillRect.anchorMax = Vector2.one;
        fillRect.offsetMin = Vector2.zero;
        fillRect.offsetMax = Vector2.zero;

        GameObject fill = new GameObject("Fill");
        fill.transform.SetParent(fillArea.transform, false);
        RectTransform fillTransform = fill.AddComponent<RectTransform>();
        fillTransform.anchorMin = Vector2.zero;
        fillTransform.anchorMax = Vector2.one;
        fillTransform.offsetMin = Vector2.zero;
        fillTransform.offsetMax = Vector2.zero;

        healthFill = fill.AddComponent<Image>();
        healthFill.color = Color.green;

        healthBar.fillRect = fillTransform;

        // Health text
        GameObject textObj = new GameObject("HealthText");
        textObj.transform.SetParent(parent.transform, false);
        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0, 0);
        textRect.anchorMax = new Vector2(1, 0.4f);
        textRect.offsetMin = new Vector2(10, 5);
        textRect.offsetMax = new Vector2(-10, 0);

        healthText = textObj.AddComponent<Text>();
        healthText.text = "Health: 100/100";
        healthText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        healthText.fontSize = healthTextSize; // Use configurable size
        healthText.color = Color.white;
        healthText.alignment = TextAnchor.MiddleRight; // Right-aligned for top-right position
    }

    void CreateAmmoPanel()
    {
        if (ammoPanel == null)
        {
            ammoPanel = new GameObject("AmmoPanel");
            ammoPanel.transform.SetParent(mainCanvas.transform, false);

            RectTransform rect = ammoPanel.AddComponent<RectTransform>();
            // Bottom-LEFT corner to avoid overlap with health (now top-right)
            rect.anchorMin = new Vector2(0.02f, 0.02f); // Left side, bottom
            rect.anchorMax = new Vector2(0.25f, 0.15f); // Small panel on left
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;

            // Background
            Image bg = ammoPanel.AddComponent<Image>();
            bg.color = new Color(0, 0, 0, 0.7f);

            // Ammo text
            GameObject textObj = new GameObject("AmmoText");
            textObj.transform.SetParent(ammoPanel.transform, false);
            RectTransform textRect = textObj.AddComponent<RectTransform>();
            textRect.anchorMin = new Vector2(0, 0.5f);
            textRect.anchorMax = new Vector2(1, 1);
            textRect.offsetMin = new Vector2(10, 0);
            textRect.offsetMax = new Vector2(-10, -5);

            ammoText = textObj.AddComponent<Text>();
            ammoText.text = "Ammo: 30/30";
            ammoText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            ammoText.fontSize = ammoTextSize; // Use configurable size
            ammoText.color = Color.white;
            ammoText.alignment = TextAnchor.MiddleLeft; // Left-aligned for bottom-left position

            // Reload text
            GameObject reloadObj = new GameObject("ReloadText");
            reloadObj.transform.SetParent(ammoPanel.transform, false);
            RectTransform reloadRect = reloadObj.AddComponent<RectTransform>();
            reloadRect.anchorMin = new Vector2(0, 0);
            reloadRect.anchorMax = new Vector2(1, 0.5f);
            reloadRect.offsetMin = new Vector2(10, 5);
            reloadRect.offsetMax = new Vector2(-10, 0);

            reloadText = reloadObj.AddComponent<Text>();
            reloadText.text = "";
            reloadText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            reloadText.fontSize = reloadTextSize; // Use configurable size
            reloadText.color = Color.yellow;
            reloadText.alignment = TextAnchor.MiddleLeft; // Left-aligned for bottom-left position
        }
    }

    void CreateDebugPanel()
    {
        if (debugPanel == null)
        {
            debugPanel = new GameObject("DebugPanel");
            debugPanel.transform.SetParent(mainCanvas.transform, false);

            RectTransform rect = debugPanel.AddComponent<RectTransform>();
            // Left side, middle-to-top, avoiding ammo panel at bottom-left
            rect.anchorMin = new Vector2(0.02f, 0.15f); // Start higher to avoid ammo panel
            rect.anchorMax = new Vector2(0.35f, 0.7f); // Don't go too wide
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;

            // Background with low opacity so it's less intrusive
            Image bg = debugPanel.AddComponent<Image>();
            bg.color = new Color(0, 0, 0, 0.5f);

            CreateDebugSections(debugPanel);
        }

        debugPanel.SetActive(showDebugUI);
    }

    void CreateDebugSections(GameObject parent)
    {
        // Performance section
        GameObject perfSection = CreateDebugSection(parent, "Performance", new Vector2(0, 0.65f), new Vector2(1, 1), Color.white);
        performanceText = perfSection.GetComponentInChildren<Text>();

        // Game Manager section  
        GameObject gmSection = CreateDebugSection(parent, "Game Manager", new Vector2(0, 0.35f), new Vector2(1, 0.65f), Color.cyan);
        gameManagerText = gmSection.GetComponentInChildren<Text>();

        // Spawner section
        GameObject spawnSection = CreateDebugSection(parent, "Spawner", new Vector2(0, 0.15f), new Vector2(1, 0.35f), Color.yellow);
        spawnerText = spawnSection.GetComponentInChildren<Text>();

        // Controls section (made taller for more text)
        GameObject controlSection = CreateDebugSection(parent, "Controls", new Vector2(0, 0), new Vector2(1, 0.15f), Color.green);
        controlsText = controlSection.GetComponentInChildren<Text>();
    }

    GameObject CreateDebugSection(GameObject parent, string title, Vector2 anchorMin, Vector2 anchorMax, Color textColor)
    {
        GameObject section = new GameObject($"{title}Section");
        section.transform.SetParent(parent.transform, false);

        RectTransform rect = section.AddComponent<RectTransform>();
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.offsetMin = new Vector2(5, 5);
        rect.offsetMax = new Vector2(-5, -5);

        Text text = section.AddComponent<Text>();
        text.text = title;
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        text.fontSize = debugTextSize; // Use configurable size
        text.color = textColor;
        text.alignment = TextAnchor.UpperLeft;

        return section;
    }

    void SetupDamageOverlay()
    {
        if (damageOverlay == null)
        {
            damageOverlay = new GameObject("DamageOverlay");
            damageOverlay.transform.SetParent(mainCanvas.transform, false);

            RectTransform rect = damageOverlay.AddComponent<RectTransform>();
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.one;
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;

            damageOverlayImage = damageOverlay.AddComponent<Image>();
            damageOverlayImage.color = new Color(1, 0, 0, 0); // Start transparent

            // Make sure it doesn't block other UI
            damageOverlay.GetComponent<Image>().raycastTarget = false;
        }

        damageOverlay.SetActive(false);
    }

    void CreateCrosshair()
    {
        if (crosshairPanel == null)
        {
            crosshairPanel = new GameObject("CrosshairPanel");
            crosshairPanel.transform.SetParent(mainCanvas.transform, false);

            RectTransform rect = crosshairPanel.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0.5f, 0.5f); // Center of screen
            rect.anchorMax = new Vector2(0.5f, 0.5f);
            rect.anchoredPosition = Vector2.zero; // CRITICAL: Must be zero for center
            rect.sizeDelta = new Vector2(crosshairSize + 10, crosshairSize + 10);

            // Create horizontal line
            GameObject horizontal = new GameObject("CrosshairHorizontal");
            horizontal.transform.SetParent(crosshairPanel.transform, false);

            RectTransform hRect = horizontal.AddComponent<RectTransform>();
            hRect.anchorMin = new Vector2(0.5f, 0.5f);
            hRect.anchorMax = new Vector2(0.5f, 0.5f);
            hRect.anchoredPosition = Vector2.zero; // CRITICAL: Must be zero for center
            hRect.sizeDelta = new Vector2(crosshairSize, crosshairThickness);
            hRect.pivot = new Vector2(0.5f, 0.5f); // ADDED: Ensure center pivot

            Image hImage = horizontal.AddComponent<Image>();
            hImage.color = crosshairColor;
            hImage.raycastTarget = false;

            // Create vertical line
            GameObject vertical = new GameObject("CrosshairVertical");
            vertical.transform.SetParent(crosshairPanel.transform, false);

            RectTransform vRect = vertical.AddComponent<RectTransform>();
            vRect.anchorMin = new Vector2(0.5f, 0.5f);
            vRect.anchorMax = new Vector2(0.5f, 0.5f);
            vRect.anchoredPosition = Vector2.zero; // CRITICAL: Must be zero for center
            vRect.sizeDelta = new Vector2(crosshairThickness, crosshairSize);
            vRect.pivot = new Vector2(0.5f, 0.5f); // ADDED: Ensure center pivot

            Image vImage = vertical.AddComponent<Image>();
            vImage.color = crosshairColor;
            vImage.raycastTarget = false;

            Debug.Log($"Crosshair created at screen center with size {crosshairSize}x{crosshairThickness}");
        }

        crosshairPanel.SetActive(showCrosshair);
    }

    void Update()
    {
        if (Input.GetKeyDown(toggleDebugKey))
        {
            ToggleDebugUI();
        }

        UpdateUI();
    }

    void UpdateUI()
    {
        UpdateHealthUI();
        UpdateAmmoUI(); // RE-ADDED
        if (showDebugUI)
            UpdateDebugUI();
    }

    void UpdateHealthUI()
    {
        if (playerHealth != null && healthBar != null)
        {
            float healthPercent = playerHealth.GetHealthPercentage();
            healthBar.value = healthPercent;

            // Update health text with immunity status
            if (healthText != null)
            {
                string healthDisplay = $"Health: {playerHealth.GetCurrentHealth():F0}/{playerHealth.GetMaxHealth():F0}";

                // Add immunity indicator if active
                if (playerHealth.IsImmune())
                {
                    float immunityTimeLeft = playerHealth.GetImmunityTimeRemaining();
                    healthDisplay += $"\nIMMUNE: {immunityTimeLeft:F1}s";
                }

                healthText.text = healthDisplay;
            }

            // Health bar color - special color during immunity
            if (healthFill != null)
            {
                if (playerHealth.IsImmune())
                {
                    // Flash between cyan and normal color during immunity
                    float flashValue = Mathf.PingPong(Time.time * 3f, 1f);
                    Color immuneColor = Color.Lerp(Color.cyan, Color.Lerp(Color.red, Color.green, healthPercent), flashValue);
                    healthFill.color = immuneColor;
                }
                else
                {
                    healthFill.color = Color.Lerp(Color.red, Color.green, healthPercent);
                }
            }
        }
    }

    void UpdateAmmoUI()
    {
        if (playerShooting != null && ammoText != null)
        {
            ammoText.text = $"Ammo: {playerShooting.GetCurrentAmmo()}/{playerShooting.GetMaxAmmo()}";
            ammoText.color = playerShooting.GetCurrentAmmo() > 0 ? Color.white : Color.red;

            if (reloadText != null)
                reloadText.text = playerShooting.IsReloading() ? "RELOADING..." : "";
        }
    }

    void UpdateDebugUI()
    {
        if (performanceTracker != null && performanceText != null)
        {
            var metrics = performanceTracker.GetCurrentMetrics();
            performanceText.text = $"PERFORMANCE:\n" +
                                 $"Survival: {metrics.survivalTime:F1}s\n" +
                                 $"Deaths: {metrics.deathCount}\n" +
                                 $"Health: {metrics.healthPercentage:P0}\n" +
                                 $"Kills: {metrics.enemiesKilled}/{metrics.totalEnemiesEncountered}\n" +
                                 $"Accuracy: {metrics.accuracy:P0}";
        }

        if (gameManager != null && gameManagerText != null)
        {
            DDASystem ddaSystem = FindAnyObjectByType<DDASystem>();
            if (ddaSystem != null)
            {
                float currentDifficulty = ddaSystem.GetCurrentDifficulty();
                int enemyCount = GameObject.FindGameObjectsWithTag("Enemy").Length;

                gameManagerText.text = $"GAME MANAGER:\n" +
                                     $"Difficulty: {currentDifficulty:F1}/10\n" +
                                     $"Enemies: {enemyCount}\n" +
                                     $"Status: {GetDifficultyStatus(currentDifficulty)}";
            }
        }

        if (spawnerText != null)
        {
            EnemySpawner spawner = FindAnyObjectByType<EnemySpawner>();
            if (spawner != null)
            {
                spawnerText.text = $"SPAWNER:\n" +
                                 $"Active Enemies: {GameObject.FindGameObjectsWithTag("Enemy").Length}\n" +
                                 $"Auto-adjusting to difficulty\n" +
                                 $"Wave system active";
            }
            else
            {
                spawnerText.text = $"SPAWNER:\n" +
                                 $"Not found";
            }
        }

        if (controlsText != null)
        {
            controlsText.text = $"CONTROLS:\n" +
                              $"1: Difficulty = 1 (Easy)\n" +
                              $"5: Difficulty = 5 (Normal)\n" +
                              $"9: Difficulty = 9 (Hard)\n" +
                              $"F1: Toggle Debug UI";
        }
    }

    public void ShowDamageEffect()
    {
        if (damageOverlay != null)
        {
            damageOverlay.SetActive(true);
            damageOverlayImage.color = new Color(1, 0, 0, damageAlpha);
            Invoke(nameof(HideDamageEffect), 0.5f);
        }
    }

    void HideDamageEffect()
    {
        if (damageOverlay != null)
        {
            damageOverlay.SetActive(false);
            damageOverlayImage.color = new Color(1, 0, 0, 0);
        }
    }

    public void ToggleDebugUI()
    {
        showDebugUI = !showDebugUI;
        if (debugPanel != null)
            debugPanel.SetActive(showDebugUI);
    }

    void CacheComponents()
    {
        playerHealth = FindAnyObjectByType<PlayerHealth>();
        playerShooting = FindAnyObjectByType<PlayerShooting>(); // RE-ADDED
        performanceTracker = FindAnyObjectByType<PlayerPerformanceTracker>();
        gameManager = FindAnyObjectByType<GameManager>();
        enemySpawner = FindAnyObjectByType<EnemySpawner>();
    }

    void SetupUIReferences()
    {
        // This method can be called to ensure UI elements are properly referenced
        // Useful when UI is created dynamically
    }

    // Helper method to get difficulty status text
    string GetDifficultyStatus(float difficulty)
    {
        if (difficulty <= 2f) return "Very Easy";
        else if (difficulty <= 4f) return "Easy";
        else if (difficulty <= 6f) return "Normal";
        else if (difficulty <= 8f) return "Hard";
        else return "Very Hard";
    }

    // Debug method to visualize safe area and UI bounds
    [ContextMenu("Force Create UI Now")]
    public void ForceCreateUI()
    {
        Debug.Log("Forcing UI creation...");
        SetupCanvas();
        CreateUILayout();
        Debug.Log("UI creation complete. Check hierarchy for new UI elements.");
    }

    [ContextMenu("Debug UI Bounds")]
    public void DebugUIBounds()
    {
        Vector4 margins = GetSafeAreaMargins();
        Debug.Log($"Safe Area Margins - Left: {margins.x:P1}, Bottom: {margins.y:P1}, Right: {margins.z:P1}, Top: {margins.w:P1}");
        Debug.Log($"Screen Resolution: {Screen.width}x{Screen.height}");
        Debug.Log($"Safe Area: {Screen.safeArea}");

        if (mainCanvas != null)
        {
            CanvasScaler scaler = mainCanvas.GetComponent<CanvasScaler>();
            if (scaler != null)
            {
                Debug.Log($"Canvas Scale Factor: {mainCanvas.scaleFactor}");
                Debug.Log($"Reference Resolution: {scaler.referenceResolution}");
            }
        }
    }

    // Method to force refresh UI positioning (useful for testing)
    [ContextMenu("Refresh UI Layout")]
    public void RefreshUILayout()
    {
        if (healthPanel != null) DestroyImmediate(healthPanel);
        if (ammoPanel != null) DestroyImmediate(ammoPanel);
        if (debugPanel != null) DestroyImmediate(debugPanel);

        CreateUILayout();
    }

    [ContextMenu("Increase Font Sizes")]
    public void IncreaseFontSizes()
    {
        healthTextSize += 2;
        ammoTextSize += 2;
        reloadTextSize += 2;
        debugTextSize += 2;
        RefreshUILayout();
        Debug.Log($"Font sizes increased! Health: {healthTextSize}, Ammo: {ammoTextSize}, Debug: {debugTextSize}");
    }

    [ContextMenu("Decrease Font Sizes")]
    public void DecreaseFontSizes()
    {
        healthTextSize = Mathf.Max(10, healthTextSize - 2);
        ammoTextSize = Mathf.Max(12, ammoTextSize - 2);
        reloadTextSize = Mathf.Max(10, reloadTextSize - 2);
        debugTextSize = Mathf.Max(8, debugTextSize - 2);
        RefreshUILayout();
        Debug.Log($"Font sizes decreased! Health: {healthTextSize}, Ammo: {ammoTextSize}, Debug: {debugTextSize}");
    }

    [ContextMenu("Debug Crosshair Info")]
    public void DebugCrosshairInfo()
    {
        if (crosshairPanel != null)
        {
            Debug.Log($"Crosshair Panel Active: {crosshairPanel.activeInHierarchy}");
            Debug.Log($"Crosshair Position: {crosshairPanel.transform.position}");

            Transform horizontal = crosshairPanel.transform.Find("CrosshairHorizontal");
            Transform vertical = crosshairPanel.transform.Find("CrosshairVertical");

            if (horizontal != null)
            {
                Image hImg = horizontal.GetComponent<Image>();
                Debug.Log($"Horizontal Line - Active: {horizontal.gameObject.activeInHierarchy}, Color: {hImg?.color}, Size: {horizontal.GetComponent<RectTransform>()?.sizeDelta}");
            }

            if (vertical != null)
            {
                Image vImg = vertical.GetComponent<Image>();
                Debug.Log($"Vertical Line - Active: {vertical.gameObject.activeInHierarchy}, Color: {vImg?.color}, Size: {vertical.GetComponent<RectTransform>()?.sizeDelta}");
            }
        }
        else
        {
            Debug.Log("Crosshair Panel is NULL!");
        }
    }
}