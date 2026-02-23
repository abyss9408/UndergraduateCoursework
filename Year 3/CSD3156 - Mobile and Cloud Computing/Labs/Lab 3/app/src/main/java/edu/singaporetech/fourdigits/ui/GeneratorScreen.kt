package edu.singaporetech.fourdigits.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import edu.singaporetech.fourdigits.viewmodel.FourDigitViewModel

@Composable
fun GeneratorScreen(
    viewModel: FourDigitViewModel,
    onNavigateToDisplay: () -> Unit
) {
    val currentFourDigit by viewModel.currentFourDigit.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = currentFourDigit?.toString() ?: "",
            fontSize = 48.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier
                .testTag("FourDigit")
                .padding(bottom = 32.dp)
        )

        Button(
            onClick = { viewModel.generateFourDigit() },
            modifier = Modifier
                .padding(8.dp)
                .width(200.dp)
        ) {
            Text("Generate")
        }

        Button(
            onClick = onNavigateToDisplay,
            modifier = Modifier
                .padding(8.dp)
                .width(200.dp)
        ) {
            Text("Display All")
        }
    }
}
