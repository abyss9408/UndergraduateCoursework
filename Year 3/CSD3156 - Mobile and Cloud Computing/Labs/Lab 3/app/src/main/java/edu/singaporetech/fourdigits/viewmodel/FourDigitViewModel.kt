package edu.singaporetech.fourdigits.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import edu.singaporetech.fourdigits.FourDigitApp
import edu.singaporetech.fourdigits.data.FourDigit
import edu.singaporetech.fourdigits.data.FourDigitRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlin.random.Random

class FourDigitViewModel(private val repository: FourDigitRepository) : ViewModel() {

    private val _currentFourDigit = MutableStateFlow<Int?>(null)
    val currentFourDigit: StateFlow<Int?> = _currentFourDigit.asStateFlow()

    val allFourDigits: StateFlow<List<FourDigit>> = repository.allFourDigits
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    init {
        // Observe latest four digit from database to restore state after rotation
        viewModelScope.launch {
            repository.latestFourDigit.collect { latest ->
                if (_currentFourDigit.value == null && latest != null) {
                    _currentFourDigit.value = latest.value
                }
            }
        }
    }

    fun generateFourDigit() {
        val randomNumber = Random.nextInt(1000, 10000)
        _currentFourDigit.value = randomNumber
        viewModelScope.launch {
            repository.insert(FourDigit(value = randomNumber))
        }
    }

    fun resetAllData() {
        viewModelScope.launch {
            repository.deleteAll()
            _currentFourDigit.value = null
        }
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val application = (this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY] as FourDigitApp)
                FourDigitViewModel(application.repository)
            }
        }
    }
}
