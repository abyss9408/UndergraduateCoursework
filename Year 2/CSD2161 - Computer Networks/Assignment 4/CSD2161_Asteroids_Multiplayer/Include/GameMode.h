/******************************************************************************/
/*!
\file		GameMode.h
\author 	Michael Henry Lazaroo
\par    	email: m.lazaroo\@digipen.edu
\date   	March 29, 2025
\brief		This header file declares the abstract GameMode class.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef GAME_MODE_H_
#define GAME_MODE_H_

// Abstract base class for game modes
class GameMode {
public:
    GameMode() = default;
    virtual ~GameMode() = default;

    // Initialize the game mode
    virtual bool Initialize() = 0;

    // Update the game mode
    virtual void Update() = 0;

    // Draw the game mode
    virtual void Draw() = 0;

    // Free resources
    virtual void Free() = 0;

    // Unload resources
    virtual void Unload() = 0;
};

#endif // GAME_MODE_H_