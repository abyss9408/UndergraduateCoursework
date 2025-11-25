/******************************************************************************/
/*!
\file		main.cpp
\author 	Bryan Ang Wei Ze
\par    	email: digipen\@digipen.edu
\date   	February 28, 2024
\brief		This source file contains the 'WinMain' function. Program execution
			begins and ends there.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#include "main.h"
#include <memory>

// ---------------------------------------------------------------------------
// Globals
float	 g_dt;
double	 g_appTime;
s8		 g_font;
int		 level;


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

	// Initialize the system
	AESysInit (instanceH, show, 800, 600, 1, 60, false, NULL);
	
	//Create your font here, and use it for all your levels
	g_font = AEGfxCreateFont("../Resources/Fonts/Arial Italic.ttf", 24);

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

			// check if forcing the application to quit
			if (AESysDoesWindowExist() == false)
				gGameStateNext = GS_QUIT;

			g_dt = (f32)AEFrameRateControllerGetFrameTime();
			
			//capping the game loop - delta time, to 1/60.0f
			if (g_dt > 0.01667f)	//0.01667f = 1/60.0f
				g_dt = 0.01667f;
				
			g_appTime += g_dt;
		}
		
		GameStateFree();

		if(gGameStateNext != GS_RESTART)
			GameStateUnload();

		gGameStatePrev = gGameStateCurr;
		gGameStateCurr = gGameStateNext;
	}
	
	//free you font here
	AEGfxDestroyFont(g_font);

	// free the system
	AESysExit();
}