package edu.singaporetech.githubstar.data.remote

import retrofit2.http.GET
import retrofit2.http.Query

interface GitHubApiService {

    @GET("search/repositories")
    suspend fun searchRepositories(
        @Query("sort") sort: String = "stars",
        @Query("q") query: String = "Android",
        @Query("per_page") perPage: Int = 50
    ): GitHubSearchResponse
}
