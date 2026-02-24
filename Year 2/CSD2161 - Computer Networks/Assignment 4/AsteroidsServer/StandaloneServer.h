/******************************************************************************/
/*!
\file   StandaloneServer.h
\brief  Authoritative game server for Asteroids multiplayer
*/
/******************************************************************************/
#pragma once

#include "UDPSocket.h"
#include "NetworkMessage.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Server-side constants
static constexpr int   MAX_PLAYERS       = 4;
static constexpr int   MAX_ASTEROIDS     = 256;
static constexpr int   MAX_BULLETS       = 512;
static constexpr float WIN_SCORE         = 5000.f;
static constexpr float WIN_MIN_X         = -400.f;
static constexpr float WIN_MAX_X         =  400.f;
static constexpr float WIN_MIN_Y         = -300.f;
static constexpr float WIN_MAX_Y         =  300.f;

static constexpr float SHIP_SCALE        = 16.f;
static constexpr float SHIP_ACCEL        = 100.f;
static constexpr float SHIP_ROT_SPEED    = 6.2831853f; // 2*PI
static constexpr float BULLET_SPEED      = 400.f;
static constexpr float BULLET_SCALE_X    = 20.f;
static constexpr float BULLET_SCALE_Y    = 3.f;
static constexpr float AST_MIN_SCALE     = 10.f;
static constexpr float AST_MAX_SCALE     = 60.f;

static constexpr double ACK_TIMEOUT_SEC  = 0.5;   // force-ACK after 500 ms
static constexpr double HB_TIMEOUT_SEC  = 3.0;    // kick player after 3 s silence
static constexpr double CONFIRM_INTERVAL = 0.2;   // retransmit CONFIRM every 200 ms
static constexpr int   CONFIRM_RETRIES   = 3;

// ---------------------------------------------------------------------------
struct PlayerInfo
{
    sockaddr_in addr{};
    char        name[17]   = {};
    bool        active     = false;
    uint32_t    score      = 0;
    float       posX       = 0.f;
    float       posY       = 0.f;
    float       velX       = 0.f;
    float       velY       = 0.f;
    float       dirCurr    = 0.f;
    uint8_t     inputBits  = 0;
    double      lastMsgTime = 0.0;
    uint16_t    lastSeq    = 0;
};

struct AsteroidState
{
    bool     active       = false;
    bool     dying        = false;   // waiting for all ACKs
    uint16_t id           = 0;
    float    posX         = 0.f;
    float    posY         = 0.f;
    float    velX         = 0.f;
    float    velY         = 0.f;
    float    scaleX       = 20.f;
    float    scaleY       = 20.f;
    uint8_t  ackBits      = 0;       // one bit per player, set on ACK
    uint8_t  expectedAckBits = 0;    // which players we expect
    double   destroyTime  = 0.0;
    int      scoringPlayer = -1;
    int      confirmRetries = 0;
    double   nextRetryTime  = 0.0;
};

struct BulletState
{
    bool     active   = false;
    uint16_t id       = 0;
    int      ownerId  = 0;
    float    posX     = 0.f;
    float    posY     = 0.f;
    float    velX     = 0.f;
    float    velY     = 0.f;
    float    dirCurr  = 0.f;
};

// ---------------------------------------------------------------------------
// Outbound packet for the send queue
struct OutPacket
{
    sockaddr_in dest;
    std::vector<uint8_t> data;
};

// ---------------------------------------------------------------------------
class StandaloneServer
{
public:
    StandaloneServer(uint16_t port, int minPlayers, int maxPlayers);
    ~StandaloneServer();

    void Run();  // blocks until server is stopped

private:
    // -----------------------------------------------------------------------
    // Thread functions
    void ReceiveThreadFunc();
    void GameLoopFunc();
    void SendThreadFunc();

    // -----------------------------------------------------------------------
    // Inbound handlers
    void HandleConnect(const NetworkMessage& msg, const sockaddr_in& from);
    void HandleDisconnect(const NetworkMessage& msg, const sockaddr_in& from);
    void HandleHeartbeat(const NetworkMessage& msg, const sockaddr_in& from);
    void HandlePlayerInput(const NetworkMessage& msg, const sockaddr_in& from);
    void HandleAsteroidHit(const NetworkMessage& msg, const sockaddr_in& from);
    void HandleAsteroidDestroyAck(const NetworkMessage& msg, const sockaddr_in& from);

    // -----------------------------------------------------------------------
    // Game-loop helpers
    void ProcessInboundMessages();
    void TickShips(float dt);
    void TickBullets(float dt);
    void TickAsteroids(float dt);
    void CheckCollisions();
    void CheckHeartbeatTimeouts();

    // -----------------------------------------------------------------------
    // Asteroid lifecycle
    void    SpawnInitialAsteroids();
    uint16_t SpawnAsteroid(float px, float py, float vx, float vy,
                           float sx, float sy);
    void    InitiateAsteroidDestroy(uint16_t astIdx, int scoringPlayer);
    void    CheckPendingAsteroidAcks();
    void    ConfirmAsteroidDestroy(uint16_t astIdx);

    // -----------------------------------------------------------------------
    // Broadcast helpers
    void BroadcastToAll(const NetworkMessage& msg);
    void SendTo(const NetworkMessage& msg, int playerIdx);
    void Enqueue(const NetworkMessage& msg, int playerIdx);    // -> send queue
    void EnqueueAll(const NetworkMessage& msg);               // -> send queue all
    void BroadcastShipStates();
    void BroadcastAsteroidCorrections();

    // -----------------------------------------------------------------------
    // Score
    void UpdateScore(int playerIdx, uint32_t addScore);
    void CheckWinCondition();
    void SendGameOver(int winnerId);
    void SaveScore(const char* name, uint32_t score);

    // -----------------------------------------------------------------------
    // Utility
    uint8_t BuildExpectedAckBits() const;
    int     FindPlayer(const sockaddr_in& addr) const;
    double  NowSec()  const;
    float   RandF(float lo, float hi);
    float   RandFExcl(float lo, float hi, float exLo, float exHi);
    float   WrapVal(float v, float lo, float hi) const;

    // -----------------------------------------------------------------------
    // State
    UDPSocket   _socket;
    uint16_t    _port;
    int         _minPlayers;
    int         _maxPlayers;

    std::atomic<bool>   _running{false};
    std::atomic<bool>   _gameRunning{false};

    // Players
    PlayerInfo  _players[MAX_PLAYERS];
    int         _connectedCount = 0;

    // Game objects
    AsteroidState _asteroids[MAX_ASTEROIDS];
    BulletState   _bullets[MAX_BULLETS];

    uint16_t    _nextAsteroidId = 0;
    uint16_t    _nextBulletId   = 0;
    uint16_t    _serverSeq      = 0;

    // Inbound queue (recv thread -> game loop)
    struct InPacket { NetworkMessage msg; sockaddr_in from; };
    std::mutex              _inMutex;
    std::condition_variable _inCV;
    std::queue<InPacket>    _inboundQueue;

    // Outbound queue (game loop -> send thread)
    std::mutex              _outMutex;
    std::condition_variable _outCV;
    std::queue<OutPacket>   _outboundQueue;

    std::thread _recvThread;
    std::thread _sendThread;

    // RNG
    uint32_t _rngSeed = 0;

    // Tick counter for periodic sends
    int _tickCount = 0;
};
