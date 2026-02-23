package edu.singaporetech.fourdigits

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation3.runtime.NavEntry
import androidx.navigation3.ui.NavDisplay
import edu.singaporetech.fourdigits.navigation.DisplayScreen
import edu.singaporetech.fourdigits.navigation.GeneratorScreen
import edu.singaporetech.fourdigits.ui.theme.FourdigitsTheme
import edu.singaporetech.fourdigits.viewmodel.FourDigitViewModel

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            FourdigitsTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    FourDigitNavigation()
                }
            }
        }
    }
}

@Composable
fun FourDigitNavigation() {
    val context = LocalContext.current
    val application = context.applicationContext as FourDigitApp
    val viewModel: FourDigitViewModel = viewModel(factory = FourDigitViewModel.Factory)

    val backstack = remember { mutableStateListOf<NavEntry<Any>>() }

    if (backstack.isEmpty()) {
        backstack.add(
            NavEntry(GeneratorScreen) {
                edu.singaporetech.fourdigits.ui.GeneratorScreen(
                    viewModel = viewModel,
                    onNavigateToDisplay = {
                        backstack.add(
                            NavEntry(DisplayScreen) {
                                edu.singaporetech.fourdigits.ui.DisplayScreen(
                                    viewModel = viewModel,
                                    application = application
                                )
                            }
                        )
                    }
                )
            }
        )
    }

    NavDisplay(
        entries = backstack,
        onBack = {
            if (backstack.size > 1) {
                backstack.removeLast()
            }
        }
    )
}