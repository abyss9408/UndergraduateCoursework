/******************************************************************************/
/*!
\file   GameMode.h
\brief  Abstract interface implemented by SinglePlayerMode and MultiplayerMode
*/
/******************************************************************************/
#pragma once

#include <cstdint>

class IGameMode
{
public:
    virtual ~IGameMode() = default;

    // Called once when entering the game / lobby
    virtual void Init()     = 0;
    // Called once when leaving the game / lobby
    virtual void Shutdown() = 0;

    // Called every game frame with the current input bit-mask and frame delta time
    virtual void onLocalInput(uint8_t inputBits, float dt) = 0;

    // Drain the inbound message queue and apply state to the game (called once per frame)
    virtual void applyNetworkState() = 0;

    // Client detected bullet-asteroid collision; report to server (no-op in SP)
    virtual void reportAsteroidHit(uint16_t asteroidId, uint16_t bulletId) = 0;

    // True after MSG_GAME_OVER received (MP) or after SP end conditions met
    virtual bool isGameOver() const = 0;

    // True when the game has started (always true for SP; true after MSG_GAME_START for MP)
    virtual bool isGameStarted() const = 0;

    // 0 for SP; server-assigned player index (0-3) for MP
    virtual int  getLocalPlayerId() const = 0;
};
