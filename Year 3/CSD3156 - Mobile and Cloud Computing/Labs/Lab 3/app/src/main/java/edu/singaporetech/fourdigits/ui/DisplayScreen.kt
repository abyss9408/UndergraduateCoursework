package edu.singaporetech.fourdigits.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import edu.singaporetech.fourdigits.FourDigitApp
import edu.singaporetech.fourdigits.data.FourDigit
import edu.singaporetech.fourdigits.viewmodel.FourDigitViewModel
import kotlinx.coroutines.launch

@Composable
fun DisplayScreen(
    viewModel: FourDigitViewModel,
    application: FourDigitApp
) {
    val allFourDigits by viewModel.allFourDigits.collectAsState()
    val isGridView by application.userPreferencesRepository.isGridView.collectAsState(initial = false)
    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        // Main content area that takes up remaining space
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        ) {
            if (isGridView) {
                LazyVerticalGrid(
                    columns = GridCells.Fixed(4),
                    contentPadding = PaddingValues(8.dp),
                    modifier = Modifier.fillMaxSize()
                ) {
                    items(allFourDigits) { fourDigit ->
                        FourDigitGridItem(fourDigit = fourDigit)
                    }
                }
            } else {
                LazyColumn(
                    contentPadding = PaddingValues(8.dp),
                    modifier = Modifier.fillMaxSize()
                ) {
                    items(allFourDigits) { fourDigit ->
                        FourDigitListItem(fourDigit = fourDigit)
                    }
                }
            }
        }

        // Bottom controls
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier.padding(bottom = 8.dp)
            ) {
                Text(
                    text = "Grid View",
                    modifier = Modifier.padding(end = 8.dp)
                )
                Switch(
                    checked = isGridView,
                    onCheckedChange = { newValue ->
                        scope.launch {
                            application.userPreferencesRepository.saveGridViewPreference(newValue)
                        }
                    },
                    modifier = Modifier.testTag("GridSwitch")
                )
            }

            Button(
                onClick = { viewModel.resetAllData() },
                modifier = Modifier.width(200.dp)
            ) {
                Text("Reset")
            }
        }
    }
}

@Composable
fun FourDigitListItem(fourDigit: FourDigit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "#${fourDigit.id}",
                fontSize = 14.sp,
                fontWeight = FontWeight.Normal
            )
            Text(
                text = fourDigit.value.toString(),
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
fun FourDigitGridItem(fourDigit: FourDigit) {
    Card(
        modifier = Modifier
            .padding(4.dp)
            .aspectRatio(1f),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "#${fourDigit.id}",
                fontSize = 10.sp,
                fontWeight = FontWeight.Normal
            )
            Text(
                text = fourDigit.value.toString(),
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}
