/******************************************************************************/
/*!
\file		GameState_Cage.h
\author 	Elie Hosry, ehosry, 00000000
\par    	ehosry@digipen.edu
\date   	Jul 03 2023
\brief  	This file includes declarations of 6 functions related to the Cage level

Copyright (C) 2023 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#ifndef CSD1130_GAME_STATE_PLAY_H_
#define CSD1130_GAME_STATE_PLAY_H_


// ---------------------------------------------------------------------------

void GameStateCageLoad(void);
void GameStateCageInit(void);
void GameStateCageUpdate(void);
void GameStateCageDraw(void);
void GameStateCageFree(void);
void GameStateCageUnload(void);

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_PLAY_H_