/******************************************************************************/
/*!
\file   MultiplayerMode.h
\brief  IGameMode implementation for networked multiplayer over UDP
*/
/******************************************************************************/
#pragma once

#include "GameMode.h"
#include "ConfigReader.h"
#include "../Network/UDPSocket.h"
#include "../Network/NetworkMessage.h"

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

class MultiplayerMode : public IGameMode
{
public:
    explicit MultiplayerMode(const NetworkConfig& cfg);
    ~MultiplayerMode() override;

    void Init()                                override;
    void Shutdown()                            override;
    void onLocalInput(uint8_t inputBits, float dt) override;
    void applyNetworkState()                   override;
    void reportAsteroidHit(uint16_t asteroidId, uint16_t bulletId) override;
    bool isGameOver()     const                override;
    bool isGameStarted()  const                override;
    int  getLocalPlayerId() const              override;

    // Lobby helpers
    bool isConnected()    const { return _connected.load();    }
    int  getPlayerCount() const { return _playerCount;         }
    const char* getPlayerName(int i) const;

private:
    // Thread function: recvfrom loop
    void ReceiveThreadFunc();

    // Send a message to the server
    void SendMsg(MessageType type, uint8_t senderId,
                 const uint8_t* payload = nullptr, uint16_t payloadLen = 0);

    // Process a single inbound message (called from main thread in applyNetworkState)
    void HandleMessage(const NetworkMessage& msg);

    // -----------------------------------------------------------------------
    NetworkConfig  _cfg;
    UDPSocket      _socket;
    sockaddr_in    _serverAddr{};

    std::thread              _recvThread;
    std::mutex               _inboundMutex;
    std::queue<NetworkMessage> _inboundQueue;

    std::atomic<bool>        _running{false};
    std::atomic<bool>        _connected{false};
    std::atomic<bool>        _gameStarted{false};
    std::atomic<bool>        _gameOver{false};

    int      _localPlayerId{0};
    int      _playerCount{0};
    char     _playerNames[4][17]{};

    std::atomic<uint16_t> _seqNum{0};
    std::atomic<uint16_t> _ackSeqNum{0};  // used only in recv thread

    double   _lastHeartbeatSent{0.0};
    double   _lastServerMsg{0.0};
};
