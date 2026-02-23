package edu.singaporetech.oiptravel

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults.topAppBarColors
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.navigation3.runtime.NavEntry
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.ui.NavDisplay
import edu.singaporetech.oiptravel.ui.theme.OIPTravelTheme

enum class OIPCampus{
    REDMOND,
    BILBAO
}
data object Home
data class Currency(val campus:OIPCampus)

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContent {
            OIPTravelTheme {
                Scaffold(modifier = Modifier.fillMaxSize(),
                    topBar = {
                        TopAppBar(
                            colors = topAppBarColors(
                                containerColor = MaterialTheme.colorScheme.primaryContainer,
                                titleContentColor = MaterialTheme.colorScheme.primary,
                            ),
                            title = {
                                Text("OIP Travel App")
                            }
                        )
                    },
                    bottomBar = {
                    }) { innerPadding ->
                    NavLogic(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
    @Composable
    fun NavLogic(modifier: Modifier = Modifier) {
        val backStack = remember { mutableStateListOf<Any>(Home) }

        NavDisplay(
            backStack = backStack,
            modifier = modifier.fillMaxSize(),
            onBack = { backStack.removeLastOrNull() },
            entryProvider = entryProvider {
                entry<Home> {
                    HomeScreen(
                        onNavigateToCurrency = { campus ->
                            backStack.add(Currency(campus))
                        }
                    )
                }
                entry<Currency> { route ->
                    CurrencyScreen(campus = route.campus)
                }
            }
        )
    }

    @Composable
    fun HomeScreen(onNavigateToCurrency: (OIPCampus) -> Unit) {
        var selectedCampus by remember { mutableStateOf(OIPCampus.REDMOND) }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Top section: Two campus columns in a Row
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                // Redmond Column
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .padding(8.dp)
                        .background(
                            color = if (selectedCampus == OIPCampus.REDMOND)
                                MaterialTheme.colorScheme.primaryContainer
                            else
                                MaterialTheme.colorScheme.surface,
                            shape = RoundedCornerShape(8.dp)
                        )
                        .border(
                            width = if (selectedCampus == OIPCampus.REDMOND) 3.dp else 1.dp,
                            color = if (selectedCampus == OIPCampus.REDMOND)
                                MaterialTheme.colorScheme.primary
                            else
                                MaterialTheme.colorScheme.outline,
                            shape = RoundedCornerShape(8.dp)
                        )
                        .testTag("Redmond")
                        .clickable { selectedCampus = OIPCampus.REDMOND }
                        .padding(4.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Redmond",
                        style = MaterialTheme.typography.titleMedium,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                    Image(
                        painter = painterResource(id = R.drawable.redmond),
                        contentDescription = "Redmond",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .aspectRatio(16f/9f)
                    )
                }

                // Bilbao Column
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .padding(8.dp)
                        .background(
                            color = if (selectedCampus == OIPCampus.BILBAO)
                                MaterialTheme.colorScheme.primaryContainer
                            else
                                MaterialTheme.colorScheme.surface,
                            shape = RoundedCornerShape(8.dp)
                        )
                        .border(
                            width = if (selectedCampus == OIPCampus.BILBAO) 3.dp else 1.dp,
                            color = if (selectedCampus == OIPCampus.BILBAO)
                                MaterialTheme.colorScheme.primary
                            else
                                MaterialTheme.colorScheme.outline,
                            shape = RoundedCornerShape(8.dp)
                        )
                        .testTag("Bilbao")
                        .clickable { selectedCampus = OIPCampus.BILBAO }
                        .padding(4.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Bilbao",
                        style = MaterialTheme.typography.titleMedium,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                    Image(
                        painter = painterResource(id = R.drawable.bilbao),
                        contentDescription = "Bilbao",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .aspectRatio(16f/9f)
                    )
                }
            }

            // Middle section: Scrollable description text
            Text(
                text = when (selectedCampus) {
                    OIPCampus.REDMOND -> stringResource(id = R.string.redmond)
                    OIPCampus.BILBAO -> stringResource(id = R.string.bilbao)
                },
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(vertical = 16.dp),
                style = MaterialTheme.typography.bodyMedium
            )

            // Bottom section: Currency Converter button
            Button(
                onClick = { onNavigateToCurrency(selectedCampus) },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 16.dp)
            ) {
                Text("Currency Converter")
            }
        }
    }

    @Composable
    fun CurrencyScreen(campus: OIPCampus) {
        var exchangeRate by remember { mutableStateOf("") }
        var inputValue by remember { mutableStateOf("") }
        var isSGDToForeign by remember { mutableStateOf(true) }
        var resultText by remember { mutableStateOf("") }

        val currencyCode = when (campus) {
            OIPCampus.REDMOND -> "USD"
            OIPCampus.BILBAO -> "EUR"
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Top
        ) {
            // Title
            Text(
                text = "Currency Converter for ${campus.name.lowercase().replaceFirstChar { it.uppercase() }}",
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.padding(bottom = 24.dp)
            )

            // Exchange Rate TextField
            TextField(
                value = exchangeRate,
                onValueChange = { exchangeRate = it },
                label = {
                    Text(
                        if (isSGDToForeign)
                            "Exchange rate, 1 SGD = $exchangeRate $currencyCode"
                        else
                            "Exchange rate, 1 $currencyCode = $exchangeRate SGD"
                    )
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp)
                    .testTag("ExchangeRate")
            )

            // Input Value TextField
            TextField(
                value = inputValue,
                onValueChange = { inputValue = it },
                label = {
                    Text(if (isSGDToForeign) "SGD" else currencyCode)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 24.dp)
                    .testTag("InputValue")
            )

            // Buttons Row
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 24.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Button(onClick = {
                    // Convert logic
                    val rate = exchangeRate.toDoubleOrNull()
                    val amount = inputValue.toIntOrNull()

                    if (rate != null && amount != null) {
                        if (isSGDToForeign) {
                            val result = amount * rate
                            resultText = "$amount SGD = %.2f $currencyCode".format(result)
                        } else {
                            val result = amount * rate
                            resultText = "$amount $currencyCode = %.2f SGD".format(result)
                        }
                    }
                }) {
                    Text("Convert")
                }

                Button(onClick = {
                    // Swap conversion direction
                    isSGDToForeign = !isSGDToForeign
                    resultText = ""
                }) {
                    Text("Swap")
                }
            }

            // Result Text
            if (resultText.isNotEmpty()) {
                Text(
                    text = resultText,
                    style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.padding(top = 16.dp)
                )
            }
        }
    }
}
