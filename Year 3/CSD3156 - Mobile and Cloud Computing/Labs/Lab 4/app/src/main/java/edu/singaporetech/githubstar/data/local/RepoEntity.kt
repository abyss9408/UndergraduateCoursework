package edu.singaporetech.githubstar.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "repos")
data class RepoEntity(
    @PrimaryKey
    val name: String,
    val starCount: Int
)
