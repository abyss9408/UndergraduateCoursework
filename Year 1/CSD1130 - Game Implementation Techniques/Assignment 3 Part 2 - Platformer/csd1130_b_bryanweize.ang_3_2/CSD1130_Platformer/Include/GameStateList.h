/******************************************************************************/
/*!
\file		GameStateList.h
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 28, 2024
\brief		This source file contains enumerations for the list of all game states

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#ifndef CSD1130_GAME_STATE_LIST_H_
#define CSD1130_GAME_STATE_LIST_H_

// ---------------------------------------------------------------------------
// game state list

enum
{
	// list of all game states 
	GS_MENU = 0,
	GS_PLATFORM,
	
	// special game state. Do not change
	GS_RESTART,
	GS_QUIT, 
	GS_NUM
};


// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_LIST_H_