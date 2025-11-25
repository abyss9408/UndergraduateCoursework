/******************************************************************************/
/*!
\file		Main.cpp
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	March 29, 2025
\brief		This source file contains the 'WinMain' function. Program execution
			begins and ends there.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "main.h"
#include "GameModeSelector.h"
#include "SinglePlayerMode.h"
#include "MultiplayerMode.h"
#include <memory>

// ---------------------------------------------------------------------------
// Globals
float	 g_dt;
double	 g_appTime;
s8 g_font;

/******************************************************************************/
/*!
	Starting point of the application
*/
/******************************************************************************/
int WINAPI WinMain(HINSTANCE instanceH, HINSTANCE prevInstanceH, LPSTR command_line, int show)
{
	UNREFERENCED_PARAMETER(prevInstanceH);
	UNREFERENCED_PARAMETER(command_line);

	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	// Show game mode selection dialog
	GameModeSelector selector;
	GameModeType selectedMode = selector.ShowModeSelection();

	// Handle quit selection
	if (selectedMode == GameModeType::QUIT) {
		return 0;
	}

	// If standalone server was selected, launch the server executable
	if (selectedMode == GameModeType::MULTIPLAYER_SERVER) {
		// Launch the server executable
		STARTUPINFO si;
		PROCESS_INFORMATION pi;

		ZeroMemory(&si, sizeof(si));
		si.cb = sizeof(si);
		ZeroMemory(&pi, sizeof(pi));

		// Start the server executable
		if (!CreateProcess("AsteroidsServer.exe", NULL, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
			MessageBox(NULL, "Failed to launch server. Make sure AsteroidsServer.exe exists in the same directory.", "Error", MB_OK | MB_ICONERROR);
			return 1;
		}

		// We don't need to wait for the server, it will run independently
		CloseHandle(pi.hProcess);
		CloseHandle(pi.hThread);

		return 0;
	}

	// Initialize the system
	AESysInit(instanceH, show, 800, 600, 1, 60, false, NULL);

	// Create font here, and use it for all levels
	g_font = AEGfxCreateFont("../Resources/Fonts/Arial Italic.ttf", 24);

	// Changing the window title
	AESysSetWindowTitle("Spaceships Game!");

	// Set background color
	AEGfxSetBackgroundColor(0.0f, 0.0f, 0.0f);

	// Create the appropriate game mode
	std::unique_ptr<GameMode> gameMode;

	if (selectedMode == GameModeType::SINGLE_PLAYER) {
		gameMode = std::make_unique<SinglePlayerMode>();
	}
	else if (selectedMode == GameModeType::MULTIPLAYER_CLIENT) {
		std::string serverAddress = selector.GetServerAddress();
		gameMode = std::make_unique<MultiplayerMode>(serverAddress);
	}

	// Initialize the game mode
	if (!gameMode->Initialize()) {
		MessageBox(NULL, "Failed to initialize game mode.", "Error", MB_OK | MB_ICONERROR);
		AESysExit();
		return 1;
	}

	// Main game loop
	bool isRunning = true;

	while (isRunning) {
		AESysFrameStart();

		// Update the game mode
		gameMode->Update();

		// Draw the game mode
		gameMode->Draw();

		AESysFrameEnd();

		// Check if the window was closed or ESC was pressed
		if ((AESysDoesWindowExist() == false) || AEInputCheckTriggered(AEVK_ESCAPE)) {
			isRunning = false;
		}

		g_dt = (float)AEFrameRateControllerGetFrameTime();
		g_appTime += g_dt;
	}

	// Free resources
	gameMode->Free();
	gameMode->Unload();

	// Free font
	AEGfxDestroyFont(g_font);

	// Free the system
	AESysExit();

	return 0;
}