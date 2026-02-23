package edu.singaporetech.fourdigits.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface FourDigitDao {
    @Insert
    suspend fun insert(fourDigit: FourDigit)

    @Query("SELECT * FROM four_digit ORDER BY id ASC")
    fun getAllFourDigits(): Flow<List<FourDigit>>

    @Query("DELETE FROM four_digit")
    suspend fun deleteAll()

    @Query("SELECT * FROM four_digit ORDER BY id DESC LIMIT 1")
    fun getLatest(): Flow<FourDigit?>
}
