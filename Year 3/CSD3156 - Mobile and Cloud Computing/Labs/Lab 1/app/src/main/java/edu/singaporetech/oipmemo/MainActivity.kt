package edu.singaporetech.oipmemo

import android.content.res.Configuration
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.ThumbUp
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.AbsoluteAlignment
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import edu.singaporetech.oipmemo.ui.theme.OIPMemoTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            OIPMemoTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    OIPMemoryApp()
                }
            }
        }
    }
}

@Composable
fun OIPMemoryApp() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            text = "OIP Memory",
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp)
        )

        // Top section: Campus likes
        CampusLikesSection()

        Spacer(modifier = Modifier.height(16.dp))

        // Bottom section: Memories list
        MemoriesList(data.momentsList)
    }
}

@Composable
fun CampusLikesSection() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        data.campusList.forEachIndexed { index, campus ->
            val testTag = when (index) {
                0 -> "LikeRedmond"
                1 -> "LikeBilbao"
                2 -> "LikeSingapore"
                else -> "LikeCampus$index"
            }
            CampusColumn(campus = campus, testTag = testTag)
        }
    }
}

@Composable
fun CampusColumn(campus: MemorableData, testTag: String) {
    var likeCount by rememberSaveable { mutableIntStateOf(0) }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .testTag(testTag)
            .clickable { likeCount++ }
            .padding(8.dp)
    ) {
        Image(
            painter = painterResource(id = campus.imgDrawable),
            contentDescription = campus.title,
            modifier = Modifier
                .size(80.dp)
                .clip(CircleShape)
                .border(2.dp, MaterialTheme.colorScheme.primary, CircleShape)
        )

        Spacer(modifier = Modifier.height(4.dp))

        Text(
            text = campus.title,
            style = MaterialTheme.typography.bodyMedium
        )

        Row(
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Filled.ThumbUp,
                contentDescription = "Like",
                modifier = Modifier.size(16.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                text = "Like $likeCount",
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Composable
fun MemoriesList(moments: List<MemorableData>) {
    LazyColumn(
        modifier = Modifier.testTag("LazyColumn")
    ) {
        items(moments.size) { index ->
            MemoryItem(moment = moments[index], initiallyExpanded = index == 0)
        }
    }
}

@Composable
fun MemoryItem(moment: MemorableData, initiallyExpanded: Boolean = false) {
    var isExpanded by rememberSaveable { mutableStateOf(initiallyExpanded) }
    var isStarred by rememberSaveable { mutableStateOf(false) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
            .clickable { isExpanded = !isExpanded }
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Image(
                painter = painterResource(id = moment.imgDrawable),
                contentDescription = moment.title,
                modifier = Modifier
                    .size(64.dp)
                    .clip(CircleShape)
                    .border(1.dp, MaterialTheme.colorScheme.outline, CircleShape)
            )

            Spacer(modifier = Modifier.width(16.dp))

            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = moment.title,
                    style = MaterialTheme.typography.titleMedium
                )

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = moment.description,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = if (isExpanded) Int.MAX_VALUE else 2
                )

                if (isExpanded) {
                    Spacer(modifier = Modifier.height(8.dp))

                    val starColor by animateColorAsState(
                        targetValue = if (isStarred) Color.Yellow else Color.Gray,
                        label = "starColor"
                    )

                    Button(
                        onClick = { isStarred = !isStarred },
                        modifier = Modifier.testTag("Star")
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Star,
                            contentDescription = "Star",
                            tint = starColor
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(if (isStarred) "Starred" else "Star")
                    }
                }
            }
        }
    }
}