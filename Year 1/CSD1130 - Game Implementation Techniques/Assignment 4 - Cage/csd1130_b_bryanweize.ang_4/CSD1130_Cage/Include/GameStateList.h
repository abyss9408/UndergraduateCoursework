/******************************************************************************/
/*!
\file		GameStateList.h
\author 	Elie Hosry, ehosry, 00000000
\par    	ehosry@digipen.edu
\date   	Jul 03 2023
\brief  	This file includes an enumeration of possible game states identifiers

Copyright (C) 2023 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#ifndef CSD1130_GAME_STATE_LIST_H_
#define CSD1130_GAME_STATE_LIST_H_

// ---------------------------------------------------------------------------
// game state list

enum class GS_STATE
{
	// list of all game states 
	GS_CAGE = 0, 
	
	// special game state. Do not change
	GS_RESTART,
	GS_QUIT, 
	GS_NUM
};

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_LIST_H_