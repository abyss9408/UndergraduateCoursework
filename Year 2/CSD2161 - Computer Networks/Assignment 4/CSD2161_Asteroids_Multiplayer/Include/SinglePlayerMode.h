/******************************************************************************/
/*!
\file   SinglePlayerMode.h
\brief  IGameMode no-op implementation for single-player
*/
/******************************************************************************/
#pragma once

#include "GameMode.h"

class SinglePlayerMode : public IGameMode
{
public:
    void Init()                                        override {}
    void Shutdown()                                    override {}
    void onLocalInput(uint8_t, float)                  override {}
    void applyNetworkState()                           override {}
    void reportAsteroidHit(uint16_t, uint16_t)         override {}
    bool isGameOver()     const                        override { return false; }
    bool isGameStarted()  const                        override { return true;  }
    int  getLocalPlayerId() const                      override { return 0;     }
};
