package edu.singaporetech.githubstar

import android.content.Context
import android.util.Log
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotDisplayed
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.filter
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onChildren
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToIndex
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import androidx.test.core.app.ApplicationProvider
import androidx.test.espresso.Espresso
import androidx.test.espresso.matcher.ViewMatchers.assertThat
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import junit.framework.TestCase.assertEquals
import org.hamcrest.Matchers.greaterThan
import org.junit.Before
import org.junit.FixMethodOrder
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestWatcher
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters

@FixMethodOrder(MethodSorters.NAME_ASCENDING) // run in alphabet order of method name
@RunWith(AndroidJUnit4::class)
@LargeTest
class UIInstrumentedTest {
    companion object {
        private val testContext: Context = ApplicationProvider.getApplicationContext()
        private val TAG = "GradingInstrumentedTest"
        private val REPORT_ITEM_MAX_LENGTH = 350
        private const val DATASTORE_PATH = "datastore/"
        private const val PREFERENCE_EXTENSION = ".preferences_pb"
    }

    // UiAutomator device
    private lateinit var device: UiDevice

    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)

    @get:Rule
    var composeTestRule = createAndroidComposeRule(MainActivity::class.java)

    @Before
    fun setupUiAutomator() {
        device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
    }

    fun isNotCached() = SemanticsMatcher("isNotCached") { node -> node.layoutInfo.isPlaced }

    @Test
    fun a1_Test() {
        Log.d(
            "Test", """
            ### 1. Running an app
            - ensure the names and tags are correct
            """.trimIndent()
        )

        composeTestRule
            .onAllNodes(
                hasTestTag("repoList")
            )
            .onFirst()
            .assertExists()

        composeTestRule
            .onAllNodes(
                hasTestTag("inputRepo")
            )
            .onFirst()
            .assertExists()
        composeTestRule.onNodeWithTag("inputRepo").performTextInput("flutter")
        composeTestRule.onNodeWithTag("inputRepo").assert(hasText("flutter"))

        composeTestRule
            .onAllNodes(
                hasText("FIND")
            )
            .onFirst()
            .assertExists()
        composeTestRule.onNodeWithText("FIND").performClick()

        Thread.sleep(2000)

        composeTestRule
            .onAllNodes(
                hasTestTag("repoName")
            )
            .onFirst()
            .assertExists()
    }
}