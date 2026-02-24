/******************************************************************************/
/*!
\file		Main.cpp
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 05, 2024
\brief		This source file contains the 'WinMain' function. Program execution
			begins and ends there.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

// MultiplayerMode.h must come first: it defines WIN32_LEAN_AND_MEAN and
// includes <winsock2.h> before main.h can pull in <windows.h> (and winsock.h).
#include "MultiplayerMode.h"
#include "main.h"
#include <memory>

// ---------------------------------------------------------------------------
// Globals
float	 g_dt;
double	 g_appTime;

// Multiplayer globals (declared extern in Main.h)
bool       g_isMultiplayer  = false;
int        g_localPlayerId  = 0;
IGameMode* g_networkMode    = nullptr;


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
		_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	#endif


	// Initialise WinSock (needed for multiplayer; safe to call in SP too)
	UDPSocket::InitWinsock();

	// Initialize the system
	AESysInit (instanceH, show, 800, 600, 1, 60, false, NULL);

	// Changing the window title
	AESysSetWindowTitle("Asteroids Demo!");

	//set background color
	AEGfxSetBackgroundColor(0.0f, 0.0f, 0.0f);



	GameStateMgrInit(GS_MENU);

	while(gGameStateCurr != GS_QUIT)
	{
		// reset the system modules
		AESysReset();

		// If not restarting, load the gamestate
		if(gGameStateCurr != GS_RESTART)
		{
			GameStateMgrUpdate();
			GameStateLoad();
		}
		else
			gGameStateNext = gGameStateCurr = gGameStatePrev;

		// Initialize the gamestate
		GameStateInit();

		while(gGameStateCurr == gGameStateNext)
		{
			AESysFrameStart();

			GameStateUpdate();

			GameStateDraw();
			
			AESysFrameEnd();

			// Force quit if the window was closed
			if (AESysDoesWindowExist() == false)
				gGameStateNext = GS_QUIT;

			g_dt = (f32)AEFrameRateControllerGetFrameTime();
			g_appTime += g_dt;
		}
		
		GameStateFree();

		if(gGameStateNext != GS_RESTART)
			GameStateUnload();

		gGameStatePrev = gGameStateCurr;
		gGameStateCurr = gGameStateNext;
	}

	// Clean up network mode if still active
	if (g_networkMode)
	{
		g_networkMode->Shutdown();
		delete g_networkMode;
		g_networkMode = nullptr;
	}

	// free the system
	AESysExit();

	// Shutdown WinSock last
	UDPSocket::ShutdownWinsock();
}