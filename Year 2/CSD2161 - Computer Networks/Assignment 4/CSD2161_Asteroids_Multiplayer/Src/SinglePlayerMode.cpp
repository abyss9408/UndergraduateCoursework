/******************************************************************************/
/*!
\file		SinglePlayerMode.cpp
\author 	Michael Henry Lazaroo
\par    	email: m.lazaroo\@digipen.edu
\date   	March 29, 2025
\brief		This source file implements the SinglePlayerMode class.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#include "SinglePlayerMode.h"
#include "main.h"

// Initialize the game mode
bool SinglePlayerMode::Initialize() {
    // Load asteroids game state
    GameStateAsteroidsLoad();

    // Initialize asteroids game state
    GameStateAsteroidsInit();

    return true;
}

// Update the game mode
void SinglePlayerMode::Update() {
    // Update asteroids game state
    GameStateAsteroidsUpdate();
}

// Draw the game mode
void SinglePlayerMode::Draw() {
    // Draw asteroids game state
    GameStateAsteroidsDraw();
}

// Free resources
void SinglePlayerMode::Free() {
    // Free asteroids game state resources
    GameStateAsteroidsFree();
}

// Unload resources
void SinglePlayerMode::Unload() {
    // Unload asteroids game state resources
    GameStateAsteroidsUnload();
}