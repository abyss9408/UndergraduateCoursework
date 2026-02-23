package edu.singaporetech.fourdigits.data

import kotlinx.coroutines.flow.Flow

class FourDigitRepository(private val fourDigitDao: FourDigitDao) {

    val allFourDigits: Flow<List<FourDigit>> = fourDigitDao.getAllFourDigits()

    val latestFourDigit: Flow<FourDigit?> = fourDigitDao.getLatest()

    suspend fun insert(fourDigit: FourDigit) {
        fourDigitDao.insert(fourDigit)
    }

    suspend fun deleteAll() {
        fourDigitDao.deleteAll()
    }
}
