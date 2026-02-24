/******************************************************************************/
/*!
\file   GameModeSelector.cpp
\brief  Menu, Lobby and Score-screen state implementations; IGameMode factory
*/
/******************************************************************************/
// MultiplayerMode.h must come first: it defines WIN32_LEAN_AND_MEAN and
// includes <winsock2.h> before main.h can pull in <windows.h> (and winsock.h).
#include "MultiplayerMode.h"
#include "main.h"
#include "GameModeSelector.h"
#include "SinglePlayerMode.h"
#include "ConfigReader.h"

#include <cstring>
#include <cstdio>

// ============================================================================
// Score-screen persistent data (set by SetScoreScreenData)
// ============================================================================
static int      ss_winnerId     = -1;
static int      ss_playerCount  = 1;
static uint32_t ss_scores[4]    = {};
static char     ss_topNames[5][17] = {};
static uint32_t ss_topScores[5] = {};

void SetScoreScreenData(int winnerId,
                        const uint32_t  scores[4],
                        int             playerCount,
                        const char      topNames[5][16],
                        const uint32_t  topScores[5])
{
    ss_winnerId    = winnerId;
    ss_playerCount = playerCount;
    for (int i = 0; i < 4; ++i)
        ss_scores[i] = scores[i];
    for (int i = 0; i < 5; ++i)
    {
        memcpy(ss_topNames[i], topNames[i], 16);
        ss_topNames[i][16] = '\0';
        ss_topScores[i] = topScores[i];
    }
}

// ============================================================================
// IGameMode factory
// ============================================================================
IGameMode* CreateGameMode(bool isMultiplayer)
{
    if (!isMultiplayer)
        return new SinglePlayerMode();

    NetworkConfig cfg = ConfigReader::Load();
    return new MultiplayerMode(cfg);
}

// ============================================================================
// GS_MENU
// ============================================================================
void GameStateMenuLoad()   {}
void GameStateMenuUnload() {}
void GameStateMenuFree()   {}

void GameStateMenuInit()
{
    printf("=== ASTEROIDS MULTIPLAYER ===\n");
    printf("  1. Single Player\n");
    printf("  2. Multiplayer\n");
    printf("  ESC. Quit\n");
}

void GameStateMenuUpdate()
{
    if (AEInputCheckTriggered(AEVK_1))
    {
        // Single-player path
        g_isMultiplayer = false;
        if (g_networkMode)
        {
            g_networkMode->Shutdown();
            delete g_networkMode;
            g_networkMode = nullptr;
        }
        gGameStateNext = GS_ASTEROIDS;
    }
    else if (AEInputCheckTriggered(AEVK_2))
    {
        // Multiplayer path – lobby will create the network mode
        gGameStateNext = GS_LOBBY;
    }
    else if (AEInputCheckTriggered(AEVK_ESCAPE))
    {
        gGameStateNext = GS_QUIT;
    }
}

void GameStateMenuDraw()
{
    // Text is already printed to console in Init; nothing to draw on-screen
    // (A production version would render text via AEGfxPrint)
}

// ============================================================================
// GS_LOBBY
// ============================================================================
void GameStateLobbyLoad()   {}
void GameStateLobbyUnload() {}

void GameStateLobbyInit()
{
    g_isMultiplayer = true;

    // Clean up any previous network mode
    if (g_networkMode)
    {
        g_networkMode->Shutdown();
        delete g_networkMode;
        g_networkMode = nullptr;
    }

    g_networkMode = CreateGameMode(true);
    g_networkMode->Init();   // sends MSG_CONNECT_REQUEST

    printf("[Lobby] Connecting to server...\n");
}

void GameStateLobbyUpdate()
{
    if (!g_networkMode)
    {
        gGameStateNext = GS_MENU;
        return;
    }

    // Poll for server messages
    g_networkMode->applyNetworkState();

    // Transition to game when server says start
    if (g_networkMode->isGameStarted())
    {
        g_localPlayerId = g_networkMode->getLocalPlayerId();
        gGameStateNext  = GS_ASTEROIDS;
        return;
    }

    // Connection timed out
    if (!static_cast<MultiplayerMode*>(g_networkMode)->isConnected()
        && /* give 5 s to connect */ true)
    {
        // stay in lobby and keep trying
    }

    // ESC cancels and returns to menu
    if (AEInputCheckTriggered(AEVK_ESCAPE))
    {
        g_networkMode->Shutdown();
        delete g_networkMode;
        g_networkMode   = nullptr;
        g_isMultiplayer = false;
        gGameStateNext  = GS_MENU;
    }
}

void GameStateLobbyDraw()
{
    // Status is printed in Init; nothing to render on screen per-frame
}

void GameStateLobbyFree()
{
    // If transitioning away from lobby but NOT to the game, destroy network mode
    if (gGameStateNext != GS_ASTEROIDS)
    {
        if (g_networkMode)
        {
            g_networkMode->Shutdown();
            delete g_networkMode;
            g_networkMode   = nullptr;
            g_isMultiplayer = false;
        }
    }
}

// ============================================================================
// GS_SCORE_SCREEN
// ============================================================================
void GameStateScoreLoad()   {}
void GameStateScoreUnload() {}

void GameStateScoreInit()
{
    printf("\n=== ROUND OVER ===\n");
    if (ss_winnerId >= 0)
        printf("Winner: Player %d (%s)\n", ss_winnerId + 1,
               ss_topNames[0][0] ? "" : "");

    printf("Scores:\n");
    for (int i = 0; i < ss_playerCount; ++i)
        printf("  Player %d: %u pts\n", i + 1, ss_scores[i]);

    printf("\n--- Top 5 Leaderboard ---\n");
    for (int i = 0; i < 5; ++i)
    {
        if (ss_topScores[i] == 0) break;
        printf("  %d. %-16s %u\n", i + 1, ss_topNames[i], ss_topScores[i]);
    }
    printf("\nPress any key to return to menu.\n");
}

void GameStateScoreUpdate()
{
    // Return to menu on Enter, Space or ESC
    if (AEInputCheckTriggered(AEVK_RETURN) ||
        AEInputCheckTriggered(AEVK_SPACE)  ||
        AEInputCheckTriggered(AEVK_ESCAPE))
    {
        gGameStateNext = GS_MENU;
    }
}

void GameStateScoreDraw()
{
    // Text already printed in Init; nothing extra to draw
}

void GameStateScoreFree()
{
    // Clean up network mode when returning to menu
    if (g_networkMode)
    {
        g_networkMode->Shutdown();
        delete g_networkMode;
        g_networkMode   = nullptr;
        g_isMultiplayer = false;
    }

    // Reset score data
    ss_winnerId    = -1;
    ss_playerCount = 1;
    memset(ss_scores,    0, sizeof(ss_scores));
    memset(ss_topNames,  0, sizeof(ss_topNames));
    memset(ss_topScores, 0, sizeof(ss_topScores));
}
