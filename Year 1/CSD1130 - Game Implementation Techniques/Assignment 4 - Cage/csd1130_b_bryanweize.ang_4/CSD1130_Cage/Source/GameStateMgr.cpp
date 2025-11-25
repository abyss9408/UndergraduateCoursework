/******************************************************************************/
/*!
\file		GameStateMgr.cpp
\author 	Elie Hosry, ehosry, 00000000
\par    	ehosry@digipen.edu
\date   	Jul 03 2023
\brief  	This file includes the implementation of the game state manager

Copyright (C) 2023 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#include "main.h"

// ---------------------------------------------------------------------------
// globals

// variables to keep track the current, previous and next game state
GS_STATE	gGameStateInit;
GS_STATE	gGameStateCurr;
GS_STATE	gGameStatePrev;
GS_STATE	gGameStateNext;

// pointer to functions for game state life cycles functions
void (*GameStateLoad)()		= 0;
void (*GameStateInit)()		= 0;
void (*GameStateUpdate)()	= 0;
void (*GameStateDraw)()		= 0;
void (*GameStateFree)()		= 0;
void (*GameStateUnload)()	= 0;

/******************************************************************************/
/*!

*/
/******************************************************************************/
void GameStateMgrInit(GS_STATE gameStateInit)
{
	// set the initial game state
	gGameStateInit = gameStateInit;

	// reset the current, previoud and next game
	gGameStateCurr = 
	gGameStatePrev = 
	gGameStateNext = gGameStateInit;

	// call the update to set the function pointers
	GameStateMgrUpdate();
}

/******************************************************************************/
/*!

*/
/******************************************************************************/
void GameStateMgrUpdate()
{
	if ((gGameStateCurr == GS_STATE::GS_RESTART) || (gGameStateCurr == GS_STATE::GS_QUIT))
		return;

	switch (gGameStateCurr)
	{
	case GS_STATE::GS_CAGE:
		GameStateLoad	= GameStateCageLoad;
		GameStateInit	= GameStateCageInit;
		GameStateUpdate	= GameStateCageUpdate;
		GameStateDraw	= GameStateCageDraw;
		GameStateFree	= GameStateCageFree;
		GameStateUnload	= GameStateCageUnload;
		break;
	default:
		AE_FATAL_ERROR("invalid state!!");
	}
}
