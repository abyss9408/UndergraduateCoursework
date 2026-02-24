/******************************************************************************/
/*!
\file		GameState_Asteroids.h
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 05, 2024
\brief		This header file declares the Load, Initialize, Update, Draw, Free
			and Unload functions of GameState_Asteroids.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef CSD1130_GAME_STATE_PLAY_H_
#define CSD1130_GAME_STATE_PLAY_H_

#include <cstdint>

// ---------------------------------------------------------------------------
// Core state functions
void GameStateAsteroidsLoad(void);
void GameStateAsteroidsInit(void);
void GameStateAsteroidsUpdate(void);
void GameStateAsteroidsDraw(void);
void GameStateAsteroidsFree(void);
void GameStateAsteroidsUnload(void);

// ---------------------------------------------------------------------------
// Multiplayer helper functions – called by MultiplayerMode when messages arrive
void MP_ApplyShipState(int playerId, float posX, float posY,
                       float velX, float velY, float dir);
void MP_SpawnAsteroid(uint16_t serverId, float px, float py,
                      float vx, float vy, float sx, float sy);
void MP_CorrectAsteroid(uint16_t serverId, float px, float py,
                        float vx, float vy);
void MP_MarkAsteroidDying(uint16_t serverId);
void MP_DestroyAsteroid(uint16_t serverId);
void MP_SpawnBullet(uint16_t bulletId, int ownerId,
                    float px, float py, float vx, float vy, float dir);
void MP_DestroyBullet(uint16_t bulletId);
void MP_UpdateScore(int playerId, uint32_t newScore);
void MP_SetGameOver(int winnerId);

// ---------------------------------------------------------------------------

#endif // CSD1130_GAME_STATE_PLAY_H_


