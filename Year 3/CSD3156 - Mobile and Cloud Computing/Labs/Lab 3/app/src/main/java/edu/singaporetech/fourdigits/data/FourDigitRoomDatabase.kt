package edu.singaporetech.fourdigits.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [FourDigit::class], version = 1, exportSchema = false)
abstract class FourDigitRoomDatabase : RoomDatabase() {
    abstract fun fourDigitDao(): FourDigitDao

    companion object {
        @Volatile
        private var INSTANCE: FourDigitRoomDatabase? = null

        fun getDatabase(context: Context): FourDigitRoomDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    FourDigitRoomDatabase::class.java,
                    "four_digit_database"
                )
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
