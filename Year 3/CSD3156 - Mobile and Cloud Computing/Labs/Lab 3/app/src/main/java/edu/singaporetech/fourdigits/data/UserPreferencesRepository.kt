package edu.singaporetech.fourdigits.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

class UserPreferencesRepository(private val context: Context) {

    private object PreferencesKeys {
        val GRID_VIEW = booleanPreferencesKey("grid_view")
    }

    val isGridView: Flow<Boolean> = context.dataStore.data
        .map { preferences ->
            preferences[PreferencesKeys.GRID_VIEW] ?: false
        }

    suspend fun saveGridViewPreference(isGridView: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PreferencesKeys.GRID_VIEW] = isGridView
        }
    }
}
