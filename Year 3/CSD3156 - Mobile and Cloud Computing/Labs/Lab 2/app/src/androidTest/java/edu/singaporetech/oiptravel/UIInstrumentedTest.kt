package edu.singaporetech.oiptravel

import android.content.Context
import android.util.Log
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotDisplayed
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
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

    //ensure testTags and button on the first screen are correct
    @Test
    fun a1_TestTheTags() {
        composeTestRule
            .onAllNodes(
                hasTestTag("Redmond")
            )
            .onFirst()
            .assertExists()

        composeTestRule
            .onAllNodes(
                hasTestTag("Bilbao")
            )
            .onFirst()
            .assertExists()

        composeTestRule
            .onNode(
                hasText(ignoreCase = true, substring = true, text ="Currency Converter")
            )
            .assertExists()
    }
}