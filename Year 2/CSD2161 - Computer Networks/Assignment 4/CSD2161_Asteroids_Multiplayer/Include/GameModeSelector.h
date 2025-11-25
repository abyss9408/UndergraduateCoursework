/******************************************************************************/
/*!
\file		GameModeSelector.h
\author 	Michael Henry Lazaroo
\par    	email: m.lazaroo\@digipen.edu
\date   	March 29, 2025
\brief		This header file declares the GameModeSelector class, which handles
            the selection between single player and multiplayer modes.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef GAME_MODE_SELECTOR_H_
#define GAME_MODE_SELECTOR_H_

#include <Windows.h>
#include <string>

enum class GameModeType {
    SINGLE_PLAYER,
    MULTIPLAYER_CLIENT,
    MULTIPLAYER_SERVER,
    QUIT
};

class GameModeSelector {
public:
    GameModeSelector() = default;
    ~GameModeSelector() = default;

    // Shows the mode selection dialog and returns the selected mode
    GameModeType ShowModeSelection();

    // Gets server address for multiplayer client mode
    std::string GetServerAddress();

private:
    // Helper function to show a simple input dialog
    std::string ShowInputDialog(const char* prompt, const char* defaultValue);
};

#endif // GAME_MODE_SELECTOR_H_