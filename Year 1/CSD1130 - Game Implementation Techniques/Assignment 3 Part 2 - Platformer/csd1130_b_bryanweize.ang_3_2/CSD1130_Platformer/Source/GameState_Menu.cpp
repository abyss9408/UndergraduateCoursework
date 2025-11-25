/******************************************************************************/
/*!
\file		GameState_Menu.cpp
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 28, 2024
\brief		This source file implements the Load, Initialize, Update, Draw, Free
			and Unload functions of GameState_Menu.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#include "main.h"

/******************************************************************************/
/*!
	"Load" function of this state
*/
/******************************************************************************/
void GameStateMenuLoad(void)
{

}

/******************************************************************************/
/*!
	"Init" function of this state
*/
/******************************************************************************/
void GameStateMenuInit(void)
{

}

/******************************************************************************/
/*!
	"Update" function of this state
*/
/******************************************************************************/
void GameStateMenuUpdate(void)
{
	if (AEInputCheckTriggered(AEVK_1))
	{
		level = 1;
		gGameStateNext = GS_PLATFORM;
	}

	if (AEInputCheckTriggered(AEVK_2))
	{
		level = 2;
		gGameStateNext = GS_PLATFORM;
	}

	if (AEInputCheckTriggered(AEVK_Q))
	{
		gGameStateNext = GS_QUIT;
	}
}

/******************************************************************************/
/*!
	"Draw" function of this state
*/
/******************************************************************************/
void GameStateMenuDraw(void)
{
	AEGfxSetBackgroundColor(0.f, 0.f, 0.f);
	/*AEGfxSetRenderMode(AE_GFX_RM_COLOR);
	AEGfxSetColorToMultiply(1.0f, 1.0f, 1.0f, 1.0f);
	AEGfxSetBlendMode(AE_GFX_BM_BLEND);
	AEGfxSetTransparency(1.0f);*/

	AEGfxPrint(g_font, "Platformer", -0.25f, 0.5f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f);
	AEGfxPrint(g_font, "Press '1' for Level 1", -0.25f, 0.3f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
	AEGfxPrint(g_font, "Press '2' for Level 2", -0.25f, 0.2f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
	AEGfxPrint(g_font, "Press 'Q' to Quit", -0.25f, 0.1f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
}

/******************************************************************************/
/*!
	"Free" function of this state
*/
/******************************************************************************/
void GameStateMenuFree(void)
{

}

/******************************************************************************/
/*!
	"Unload" function of this state
*/
/******************************************************************************/
void GameStateMenuUnload(void)
{

}