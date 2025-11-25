/******************************************************************************/
/*!
\file		GameState_Menu.h
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 28, 2024
\brief		This header file declares the Load, Initialize, Update, Draw, Free
			and Unload functions of GameState_Menu.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef CSD1130_GAME_STATE_MENU_H_
#define CSD1130_GAME_STATE_MENU_H_


// ---------------------------------------------------------------------------

void GameStateMenuLoad(void);
void GameStateMenuInit(void);
void GameStateMenuUpdate(void);
void GameStateMenuDraw(void);
void GameStateMenuFree(void);
void GameStateMenuUnload(void);

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_MENU_H_