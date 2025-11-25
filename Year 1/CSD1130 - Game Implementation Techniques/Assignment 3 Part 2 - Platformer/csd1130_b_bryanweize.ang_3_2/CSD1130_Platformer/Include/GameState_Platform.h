/******************************************************************************/
/*!
\file		GameState_Platform.h
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 28, 2024
\brief		This header file declares the Load, Initialize, Update, Draw, Free
			and Unload functions of GameState_Platform.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#ifndef CSD1130_GAME_STATE_PLAY_H_
#define CSD1130_GAME_STATE_PLAY_H_


// ---------------------------------------------------------------------------

void GameStatePlatformLoad(void);
void GameStatePlatformInit(void);
void GameStatePlatformUpdate(void);
void GameStatePlatformDraw(void);
void GameStatePlatformFree(void);
void GameStatePlatformUnload(void);

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_PLAY_H_