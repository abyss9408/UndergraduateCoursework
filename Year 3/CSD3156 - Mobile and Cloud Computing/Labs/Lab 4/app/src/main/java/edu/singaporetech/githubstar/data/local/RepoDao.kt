package edu.singaporetech.githubstar.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface RepoDao {

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertRepo(repo: RepoEntity)

    @Query("SELECT * FROM repos ORDER BY starCount DESC")
    fun getAllRepos(): Flow<List<RepoEntity>>

    @Query("SELECT * FROM repos WHERE name = :name")
    suspend fun getRepoByName(name: String): RepoEntity?
}
