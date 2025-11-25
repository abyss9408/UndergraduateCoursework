/******************************************************************************/
/*!
\file		GameModeSelector.cpp
\author 	Michael Henry Lazaroo
\par    	email: m.lazaroo\@digipen.edu
\date   	March 29, 2025
\brief		This source file implements the GameModeSelector class, which handles
            the selection between single player and multiplayer modes.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#include "GameModeSelector.h"
#include "ConfigReader.h"
#include <commdlg.h>
#include <iostream>

GameModeType GameModeSelector::ShowModeSelection() {
    int result = MessageBoxA(NULL,
        "Select game mode:\n\n"
        "YES - Single Player\n"
        "NO - Multiplayer Client\n"
        "CANCEL - Multiplayer Server",
        "Spaceships Game Mode Selection",
        MB_YESNOCANCEL | MB_ICONQUESTION);

    switch (result) {
    case IDYES:
        return GameModeType::SINGLE_PLAYER;
    case IDNO:
        return GameModeType::MULTIPLAYER_CLIENT;
    case IDCANCEL:
        return GameModeType::MULTIPLAYER_SERVER;
    default:
        return GameModeType::QUIT;
    }
}

std::string GameModeSelector::GetServerAddress() {
    // First try to read from config file
    ConfigReader config;
    const std::string configFilename = "network.cfg";
    std::string serverAddress = "127.0.0.1"; // Default fallback address

    if (config.LoadFromFile(configFilename)) {
        // Config file exists, try to read server address
        std::string configAddress = config.GetString("ServerAddress", "");
        if (!configAddress.empty()) {
            // Found valid server address in config
            std::cout << "Using server address from config file: " << configAddress << std::endl;
            return configAddress;
        }
    }

    // Config file doesn't exist or doesn't have server address
    // Ask user for server address
    std::string userAddress = ShowInputDialog("Enter server IP address:", serverAddress.c_str());

    // Save the user's choice to config file for next time
    config.SetString("ServerAddress", userAddress);
    if (config.SaveToFile(configFilename)) {
        std::cout << "Saved server address to config file" << std::endl;
    }
    else {
        std::cout << "Failed to save server address to config file" << std::endl;
    }

    return userAddress;
}

std::string GameModeSelector::ShowInputDialog(const char* prompt, const char* defaultValue) {
    char buffer[256] = { 0 };
    strncpy_s(buffer, defaultValue, sizeof(buffer) - 1);

    OPENFILENAMEA ofn = { 0 };
    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = buffer;
    ofn.nMaxFile = sizeof(buffer);
    ofn.lpstrTitle = prompt;
    ofn.Flags = OFN_HIDEREADONLY | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST | OFN_ENABLEHOOK | OFN_ENABLESIZING;

    // Abuse GetSaveFileName as an input dialog (hack but works for simple cases)
    if (GetSaveFileNameA(&ofn)) {
        // Extract just the filename part as the input
        char* lastSlash = strrchr(buffer, '\\');
        if (lastSlash) {
            return std::string(lastSlash + 1);
        }
        return std::string(buffer);
    }

    return std::string(defaultValue);
}