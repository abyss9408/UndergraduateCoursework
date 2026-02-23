package edu.singaporetech.fourdigits

import android.app.Application
import edu.singaporetech.fourdigits.data.FourDigitRepository
import edu.singaporetech.fourdigits.data.FourDigitRoomDatabase
import edu.singaporetech.fourdigits.data.UserPreferencesRepository

class FourDigitApp : Application() {
    // Initialize your database and repository
    private val database by lazy { FourDigitRoomDatabase.getDatabase(this) }
    val repository by lazy { FourDigitRepository(database.fourDigitDao()) }
    val userPreferencesRepository by lazy { UserPreferencesRepository(this) }
}