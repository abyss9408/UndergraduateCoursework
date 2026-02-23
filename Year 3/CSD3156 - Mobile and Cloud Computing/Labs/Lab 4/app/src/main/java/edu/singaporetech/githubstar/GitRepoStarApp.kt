package edu.singaporetech.githubstar

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class GitRepoStarApp: Application() {
    override fun onCreate() {
        super.onCreate()
    }
}