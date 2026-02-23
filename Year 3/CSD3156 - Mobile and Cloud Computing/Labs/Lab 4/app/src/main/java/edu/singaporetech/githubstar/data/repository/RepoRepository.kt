package edu.singaporetech.githubstar.data.repository

import edu.singaporetech.githubstar.data.local.RepoDao
import edu.singaporetech.githubstar.data.local.RepoEntity
import edu.singaporetech.githubstar.data.remote.GitHubApiService
import kotlinx.coroutines.flow.Flow

class RepoRepository(
    private val repoDao: RepoDao,
    private val gitHubApiService: GitHubApiService
) {

    /** Emits the full list of locally stored repos, updated in real-time via Room Flow. */
    fun getAllRepos(): Flow<List<RepoEntity>> = repoDao.getAllRepos()

    /**
     * 1. If the repo is already in the local DB, do nothing.
     * 2. Otherwise query GitHub for the top-50 Android repos sorted by stars.
     * 3. If the repo appears in that list, persist it locally.
     */
    suspend fun searchAndAddRepo(repoName: String) {
        val existing = repoDao.getRepoByName(repoName)
        if (existing != null) return

        val response = gitHubApiService.searchRepositories()

        val found = response.items.find { it.name.equals(repoName, ignoreCase = true) }
        if (found != null) {
            repoDao.insertRepo(
                RepoEntity(name = found.name, starCount = found.stargazersCount)
            )
        }
    }
}
