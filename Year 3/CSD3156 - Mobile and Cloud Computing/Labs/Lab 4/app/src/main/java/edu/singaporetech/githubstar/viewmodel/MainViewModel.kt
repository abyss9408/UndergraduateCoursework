package edu.singaporetech.githubstar.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.room.Room
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import edu.singaporetech.githubstar.data.local.RepoDatabase
import edu.singaporetech.githubstar.data.local.RepoEntity
import edu.singaporetech.githubstar.data.repository.RepoRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

@HiltViewModel
class MainViewModel @Inject constructor(
    @ApplicationContext private val context: Context
) : ViewModel() {

    // Room database – Room itself manages the underlying connection pool,
    // so one instance per ViewModel is safe.
    private val database = Room.databaseBuilder(
        context, RepoDatabase::class.java, "repo_db"
    ).build()

    // Retrofit client
    private val retrofit = Retrofit.Builder()
        .baseUrl("https://api.github.com/")
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    // Repository wired with the DAO and the API service
    private val repository = RepoRepository(
        repoDao = database.repoDao(),
        gitHubApiService = retrofit.create(
            edu.singaporetech.githubstar.data.remote.GitHubApiService::class.java
        )
    )

    /** Live stream of every repo currently in the local database, ordered by stars DESC. */
    val repos: StateFlow<List<RepoEntity>> = repository.getAllRepos()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList()
        )

    /** Kicks off the search-and-cache flow for one repository name. */
    fun searchRepo(name: String) {
        if (name.isBlank()) return
        viewModelScope.launch {
            try {
                repository.searchAndAddRepo(name.trim())
            } catch (_: Exception) {
                // network / parse error – silently ignored
            }
        }
    }
}
