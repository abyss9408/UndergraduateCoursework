/******************************************************************************/
/*!
\file		main.cpp
\author 	Elie Hosry, ehosry, 00000000
\par    	ehosry@digipen.edu
\date   	Jul 03 2023
\brief  	This file includes the main application game loop from start to finish

Copyright (C) 2023 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#include "main.h"
#include <memory>


// ---------------------------------------------------------------------------
// Globals
float	 g_dt = 0.01667f;
double	 g_appTime;

s8	fontId = 0;

/******************************************************************************/
/*!
	Starting point of the application
*/
/******************************************************************************/
int WINAPI WinMain(HINSTANCE instanceH, HINSTANCE prevInstanceH, LPSTR command_line, int show)
{
	// Enable run-time memory check for debug builds.
	#if defined(DEBUG) | defined(_DEBUG)
		_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	#endif

	UNREFERENCED_PARAMETER(prevInstanceH);
	UNREFERENCED_PARAMETER(command_line);
	// Initialize the system
	AESysInit(instanceH, show, 1024, 768, 1, 60, false, nullptr);

	TestMyMath();

	//Fonts Assets
	fontId = AEGfxCreateFont("..\\Bin\\Resources\\Fonts\\Arial Italic.ttf", 28);

	GameStateMgrInit(GS_STATE::GS_CAGE);

	while(gGameStateCurr != GS_STATE::GS_QUIT)
	{
		// reset the system modules
		AESysReset();

		// If not restarting, load the gamestate
		if(gGameStateCurr != GS_STATE::GS_RESTART)
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
			if ((AESysDoesWindowExist() == false) || AEInputCheckTriggered(AEVK_ESCAPE))
				gGameStateNext = GS_STATE::GS_QUIT;

			g_dt = (f32)AEFrameRateControllerGetFrameTime();

			//hack
			if (g_dt > 0.01667f)
				g_dt = 0.01667f;
			
			g_appTime += g_dt;
		}
		
		GameStateFree();

		if(gGameStateNext != GS_STATE::GS_RESTART)
			GameStateUnload();

		gGameStatePrev = gGameStateCurr;
		gGameStateCurr = gGameStateNext;
	}

	AEGfxDestroyFont(fontId);

	// free the system
	AESysExit();
}