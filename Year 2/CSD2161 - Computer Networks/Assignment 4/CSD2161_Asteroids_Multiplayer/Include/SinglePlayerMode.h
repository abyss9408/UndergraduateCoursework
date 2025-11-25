/******************************************************************************/
/*!
\file		SinglePlayerMode.h
\author 	Michael Henry Lazaroo
\par    	email: m.lazaroo\@digipen.edu
\date   	March 29, 2025
\brief		This header file declares the SinglePlayerMode class.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef SINGLE_PLAYER_MODE_H_
#define SINGLE_PLAYER_MODE_H_

#include "GameMode.h"

// SinglePlayerMode class - wraps the existing single player code
class SinglePlayerMode : public GameMode {
public:
    SinglePlayerMode() = default;
    ~SinglePlayerMode() = default;

    // Initialize the game mode
    bool Initialize() override;

    // Update the game mode
    void Update() override;

    // Draw the game mode
    void Draw() override;

    // Free resources
    void Free() override;

    // Unload resources
    void Unload() override;
};

#endif // SINGLE_PLAYER_MODE_H_