/******************************************************************************/
/*!
\file   GameModeSelector.h
\brief  State functions for GS_MENU, GS_LOBBY, GS_SCORE_SCREEN;
        also the IGameMode factory.
*/
/******************************************************************************/
#pragma once

#include <cstdint>

class IGameMode;

// ---------------------------------------------------------------------------
// Factory: creates a SinglePlayerMode or MultiplayerMode instance
IGameMode* CreateGameMode(bool isMultiplayer);

// ---------------------------------------------------------------------------
// GS_MENU state functions
void GameStateMenuLoad();
void GameStateMenuInit();
void GameStateMenuUpdate();
void GameStateMenuDraw();
void GameStateMenuFree();
void GameStateMenuUnload();

// ---------------------------------------------------------------------------
// GS_LOBBY state functions
void GameStateLobbyLoad();
void GameStateLobbyInit();
void GameStateLobbyUpdate();
void GameStateLobbyDraw();
void GameStateLobbyFree();
void GameStateLobbyUnload();

// ---------------------------------------------------------------------------
// GS_SCORE_SCREEN state functions
void GameStateScoreLoad();
void GameStateScoreInit();
void GameStateScoreUpdate();
void GameStateScoreDraw();
void GameStateScoreFree();
void GameStateScoreUnload();

// ---------------------------------------------------------------------------
// Called by MultiplayerMode when MSG_GAME_OVER is received, so the score
// screen can display the leaderboard data.
void SetScoreScreenData(int winnerId,
                        const uint32_t  scores[4],
                        int             playerCount,
                        const char      topNames[5][16],
                        const uint32_t  topScores[5]);
