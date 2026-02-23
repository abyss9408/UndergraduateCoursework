// DDAVisualizer.cs - UI system to show DDA in action
using UnityEngine;
using UnityEngine.UI;

public class DDAVisualizer : MonoBehaviour
{
    [Header("UI Elements")]
    public Text difficultyText;
    public Text performanceText;
    public Text metricsText;
    public Slider difficultySlider;
    public GameObject uiPanel;

    private Canvas canvas;

    public void Initialize(Canvas uiCanvas)
    {
        canvas = uiCanvas;
        CreateUI();
    }

    void CreateUI()
    {
        // Create main panel
        GameObject panel = new GameObject("DDA Panel");
        panel.transform.SetParent(canvas.transform, false);

        RectTransform panelRect = panel.AddComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(0, 0.7f);
        panelRect.anchorMax = new Vector2(0.4f, 1f);
        panelRect.offsetMin = Vector2.zero;
        panelRect.offsetMax = Vector2.zero;

        Image panelImage = panel.AddComponent<Image>();
        panelImage.color = new Color(0, 0, 0, 0.7f);

        uiPanel = panel;

        // Create difficulty slider
        CreateDifficultySlider(panel);

        // Create text elements
        CreateTextElements(panel);
    }

    void CreateDifficultySlider(GameObject parent)
    {
        GameObject sliderGO = new GameObject("Difficulty Slider");
        sliderGO.transform.SetParent(parent.transform, false);

        RectTransform sliderRect = sliderGO.AddComponent<RectTransform>();
        sliderRect.anchorMin = new Vector2(0.1f, 0.8f);
        sliderRect.anchorMax = new Vector2(0.9f, 0.9f);
        sliderRect.offsetMin = Vector2.zero;
        sliderRect.offsetMax = Vector2.zero;

        difficultySlider = sliderGO.AddComponent<Slider>();
        difficultySlider.minValue = 1f;
        difficultySlider.maxValue = 10f;
        difficultySlider.value = 5f;

        // Add slider visuals (simplified)
        GameObject background = new GameObject("Background");
        background.transform.SetParent(sliderGO.transform, false);
        RectTransform bgRect = background.AddComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.offsetMin = Vector2.zero;
        bgRect.offsetMax = Vector2.zero;
        Image bgImage = background.AddComponent<Image>();
        bgImage.color = Color.gray;

        GameObject handle = new GameObject("Handle");
        handle.transform.SetParent(sliderGO.transform, false);
        RectTransform handleRect = handle.AddComponent<RectTransform>();
        handleRect.sizeDelta = new Vector2(20, 20);
        Image handleImage = handle.AddComponent<Image>();
        handleImage.color = Color.white;

        difficultySlider.targetGraphic = handleImage;
        difficultySlider.handleRect = handleRect;
    }

    void CreateTextElements(GameObject parent)
    {
        // Difficulty text
        difficultyText = CreateText("Difficulty: 5.0", parent, new Vector2(0.1f, 0.65f), new Vector2(0.9f, 0.75f));

        // Performance text
        performanceText = CreateText("Performance: Analyzing...", parent, new Vector2(0.1f, 0.5f), new Vector2(0.9f, 0.6f));

        // Metrics text
        metricsText = CreateText("Metrics:\nKills: 0\nAccuracy: 0%", parent, new Vector2(0.1f, 0.1f), new Vector2(0.9f, 0.45f));
    }

    Text CreateText(string content, GameObject parent, Vector2 anchorMin, Vector2 anchorMax)
    {
        GameObject textGO = new GameObject("Text");
        textGO.transform.SetParent(parent.transform, false);

        RectTransform textRect = textGO.AddComponent<RectTransform>();
        textRect.anchorMin = anchorMin;
        textRect.anchorMax = anchorMax;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;

        Text text = textGO.AddComponent<Text>();
        text.text = content;
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        text.fontSize = 14;
        text.color = Color.white;

        return text;
    }

    public void UpdateDisplay(PlayerMetrics metrics, float difficulty)
    {
        if (difficultySlider != null)
            difficultySlider.value = difficulty;

        if (difficultyText != null)
            difficultyText.text = $"Difficulty: {difficulty:F1}";

        if (performanceText != null)
        {
            float performance = metrics.totalEnemiesEncountered > 0 ?
                (float)metrics.enemiesKilled / metrics.totalEnemiesEncountered : 0f;
            performanceText.text = $"Performance: {performance:P0}";
        }

        if (metricsText != null)
        {
            metricsText.text = $"Metrics:\n" +
                $"Kills: {metrics.enemiesKilled}\n" +
                $"Accuracy: {metrics.accuracy:P0}\n" +
                $"Health: {metrics.currentHealth:F0}/{metrics.maxHealth:F0}\n" +
                $"Time elapsed: {metrics.survivalTime:F0}s";
        }
    }
}