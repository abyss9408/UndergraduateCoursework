package edu.singaporetech.githubstar.data.remote

import com.google.gson.annotations.SerializedName

data class GitHubSearchResponse(
    @SerializedName("total_count") val totalCount: Int,
    val items: List<GitHubRepo>
)

data class GitHubRepo(
    val name: String,
    @SerializedName("stargazers_count") val stargazersCount: Int
)
