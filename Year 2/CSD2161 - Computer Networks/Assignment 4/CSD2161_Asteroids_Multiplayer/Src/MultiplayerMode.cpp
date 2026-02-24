/******************************************************************************/
/*!
\file   MultiplayerMode.cpp
\brief  Networked multiplayer mode: connection, receive thread, message dispatch
*/
/******************************************************************************/
#include "MultiplayerMode.h"
#include "GameState_Asteroids.h"    // MP_* helpers
#include "GameModeSelector.h"       // SetScoreScreenData

#include <cstring>
#include <chrono>
#include <cstdio>

// ---------------------------------------------------------------------------
// Helper: current time in seconds (used for heartbeat timing)
static double NowSec()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
MultiplayerMode::MultiplayerMode(const NetworkConfig& cfg)
    : _cfg(cfg)
{
}

MultiplayerMode::~MultiplayerMode()
{
    Shutdown();
}

// ---------------------------------------------------------------------------
void MultiplayerMode::Init()
{
    // Initialise server address
    memset(&_serverAddr, 0, sizeof(_serverAddr));
    _serverAddr.sin_family = AF_INET;
    _serverAddr.sin_port   = htons(_cfg.serverPort);
    inet_pton(AF_INET, _cfg.serverIp.c_str(), &_serverAddr.sin_addr);

    // Bind to any local port
    _socket.Bind("", 0);
    _socket.SetRecvTimeout(100); // 100 ms timeout so recv thread can check _running

    _running = true;
    _recvThread = std::thread(&MultiplayerMode::ReceiveThreadFunc, this);

    // Send connect request: payload = name[16]
    uint8_t payload[16]{};
    strncpy_s(reinterpret_cast<char*>(payload), 16,
              _cfg.playerName.c_str(), _TRUNCATE);
    SendMsg(MSG_CONNECT_REQUEST, 0xFF, payload, 16);

    _lastHeartbeatSent = NowSec();
    _lastServerMsg     = NowSec();
}

// ---------------------------------------------------------------------------
void MultiplayerMode::Shutdown()
{
    if (!_running.exchange(false))
        return;   // already shut down

    _socket.Close();
    if (_recvThread.joinable())
        _recvThread.join();
}

// ---------------------------------------------------------------------------
void MultiplayerMode::SendMsg(MessageType type, uint8_t senderId,
                              const uint8_t* payload, uint16_t payloadLen)
{
    NetworkMessage msg{};
    msg.header.msgType    = static_cast<uint8_t>(type);
    msg.header.senderId   = senderId;
    msg.header.seqNum     = ++_seqNum;
    msg.header.payloadLen = payloadLen;
    if (payload && payloadLen > 0)
        msg.payload.assign(payload, payload + payloadLen);

    std::vector<uint8_t> buf;
    BuildPacket(msg, buf);
    _socket.SendTo(buf.data(), buf.size(), _serverAddr);
}

// ---------------------------------------------------------------------------
void MultiplayerMode::onLocalInput(uint8_t inputBits, float dt)
{
    if (!_gameStarted || _gameOver)
        return;

    // Send input: inputBits(1) + dt(4) = 5 bytes
    uint8_t payload[5];
    payload[0] = inputBits;
    memcpy(payload + 1, &dt, 4);

    SendMsg(MSG_PLAYER_INPUT,
            static_cast<uint8_t>(_localPlayerId),
            payload, 5);

    // Heartbeat every 1 second
    double now = NowSec();
    if (now - _lastHeartbeatSent > 1.0)
    {
        SendMsg(MSG_HEARTBEAT, static_cast<uint8_t>(_localPlayerId));
        _lastHeartbeatSent = now;
    }
}

// ---------------------------------------------------------------------------
void MultiplayerMode::applyNetworkState()
{
    // Detect server timeout (3 seconds without message)
    if (_connected && NowSec() - _lastServerMsg > 3.0)
    {
        printf("[MP] Server timeout – disconnecting\n");
        _connected   = false;
        _gameStarted = false;
        // Transition will be handled by lobby/game update functions
    }

    // Drain inbound queue
    std::queue<NetworkMessage> local;
    {
        std::lock_guard<std::mutex> lock(_inboundMutex);
        std::swap(local, _inboundQueue);
    }
    while (!local.empty())
    {
        HandleMessage(local.front());
        local.pop();
    }
}

// ---------------------------------------------------------------------------
void MultiplayerMode::reportAsteroidHit(uint16_t asteroidId, uint16_t bulletId)
{
    if (!_gameStarted || _gameOver)
        return;

    // payload: asteroidId(2) + bulletId(2) = 4 bytes
    uint8_t payload[4];
    memcpy(payload + 0, &asteroidId, 2);
    memcpy(payload + 2, &bulletId,   2);
    SendMsg(MSG_ASTEROID_HIT,
            static_cast<uint8_t>(_localPlayerId),
            payload, 4);
}

// ---------------------------------------------------------------------------
bool MultiplayerMode::isGameOver()    const { return _gameOver.load();    }
bool MultiplayerMode::isGameStarted() const { return _gameStarted.load(); }
int  MultiplayerMode::getLocalPlayerId() const { return _localPlayerId;   }

const char* MultiplayerMode::getPlayerName(int i) const
{
    if (i < 0 || i >= 4) return "";
    return _playerNames[i];
}

// ---------------------------------------------------------------------------
// Receive thread: recvfrom loop, sends ACKs immediately for ASTEROID_DESTROY
void MultiplayerMode::ReceiveThreadFunc()
{
    std::vector<uint8_t> buf;
    sockaddr_in from{};

    while (_running)
    {
        int received = _socket.RecvFrom(buf, from);
        if (received <= 0)
            continue;

        NetworkMessage msg;
        if (!ParsePacket(buf.data(), static_cast<size_t>(received), msg))
            continue;

        _lastServerMsg = NowSec();

        // Send ACK immediately for asteroid destroy (plan requirement)
        if (msg.header.msgType == MSG_ASTEROID_DESTROY && msg.payload.size() >= 2)
        {
            uint16_t astId;
            memcpy(&astId, msg.payload.data(), 2);

            uint8_t ackPay[2];
            memcpy(ackPay, &astId, 2);

            NetworkMessage ack{};
            ack.header.msgType  = MSG_ASTEROID_DESTROY_ACK;
            ack.header.senderId = static_cast<uint8_t>(_localPlayerId);
            ack.header.seqNum   = ++_ackSeqNum;
            ack.payload.assign(ackPay, ackPay + 2);

            std::vector<uint8_t> ackBuf;
            BuildPacket(ack, ackBuf);
            _socket.SendTo(ackBuf.data(), ackBuf.size(), _serverAddr);
        }

        // Push to inbound queue for main thread processing
        std::lock_guard<std::mutex> lock(_inboundMutex);
        _inboundQueue.push(std::move(msg));
    }
}

// ---------------------------------------------------------------------------
// Inline helpers for float parsing
static float ReadF32(const uint8_t* p) { float v; memcpy(&v, p, 4); return v; }
static uint16_t ReadU16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t ReadU32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }

// ---------------------------------------------------------------------------
void MultiplayerMode::HandleMessage(const NetworkMessage& msg)
{
    const auto& pay = msg.payload;

    switch (msg.header.msgType)
    {
    // -----------------------------------------------------------------------
    case MSG_CONNECT_ACCEPT:
        // payload: assignedId(1), playerCount(1), names[4][16]
        if (pay.size() >= 2)
        {
            _localPlayerId = pay[0];
            _playerCount   = pay[1];
            for (int i = 0; i < 4 && 2 + i * 16 + 16 <= (int)pay.size(); ++i)
            {
                memcpy(_playerNames[i], &pay[2 + i * 16], 16);
                _playerNames[i][16] = '\0';
            }
            _connected = true;
            printf("[MP] Connected as player %d (total players: %d)\n",
                   _localPlayerId, _playerCount);
        }
        break;

    case MSG_CONNECT_REJECT:
        printf("[MP] Connection rejected (reason: %d)\n",
               pay.empty() ? 0 : pay[0]);
        break;

    case MSG_DISCONNECT:
        if (pay.size() >= 1)
            printf("[MP] Player %d disconnected\n", pay[0]);
        break;

    case MSG_HEARTBEAT_ACK:
        // lastServerMsg already updated above
        break;

    // -----------------------------------------------------------------------
    case MSG_GAME_START:
        // payload: rngSeed(4)
        _gameStarted = true;
        printf("[MP] Game started!\n");
        break;

    // -----------------------------------------------------------------------
    case MSG_SHIP_STATE:
        // payload: playerId(1), posX(4), posY(4), velX(4), velY(4), dir(4) = 21
        if (pay.size() >= 21)
        {
            int  pid  = pay[0];
            float px  = ReadF32(&pay[1]);
            float py  = ReadF32(&pay[5]);
            float vx  = ReadF32(&pay[9]);
            float vy  = ReadF32(&pay[13]);
            float dir = ReadF32(&pay[17]);
            MP_ApplyShipState(pid, px, py, vx, vy, dir);
        }
        break;

    case MSG_BULLET_SPAWN:
        // payload: bulletId(2), ownerId(1), posX(4), posY(4), velX(4), velY(4), dir(4) = 23
        if (pay.size() >= 23)
        {
            uint16_t bid    = ReadU16(&pay[0]);
            int      owner  = pay[2];
            float    px     = ReadF32(&pay[3]);
            float    py     = ReadF32(&pay[7]);
            float    vx     = ReadF32(&pay[11]);
            float    vy     = ReadF32(&pay[15]);
            float    dir    = ReadF32(&pay[19]);
            MP_SpawnBullet(bid, owner, px, py, vx, vy, dir);
        }
        break;

    case MSG_BULLET_DESTROY:
        // payload: bulletId(2)
        if (pay.size() >= 2)
            MP_DestroyBullet(ReadU16(&pay[0]));
        break;

    // -----------------------------------------------------------------------
    case MSG_ASTEROID_SPAWN:
        // payload: id(2), posX(4), posY(4), velX(4), velY(4), scaleX(4), scaleY(4) = 26
        if (pay.size() >= 26)
        {
            uint16_t id = ReadU16(&pay[0]);
            float px    = ReadF32(&pay[2]);
            float py    = ReadF32(&pay[6]);
            float vx    = ReadF32(&pay[10]);
            float vy    = ReadF32(&pay[14]);
            float sx    = ReadF32(&pay[18]);
            float sy    = ReadF32(&pay[22]);
            MP_SpawnAsteroid(id, px, py, vx, vy, sx, sy);
        }
        break;

    case MSG_ASTEROID_CORRECT:
        // payload: id(2), posX(4), posY(4), velX(4), velY(4) = 18
        if (pay.size() >= 18)
        {
            uint16_t id = ReadU16(&pay[0]);
            float px    = ReadF32(&pay[2]);
            float py    = ReadF32(&pay[6]);
            float vx    = ReadF32(&pay[10]);
            float vy    = ReadF32(&pay[14]);
            MP_CorrectAsteroid(id, px, py, vx, vy);
        }
        break;

    case MSG_ASTEROID_DESTROY:
        // ACK already sent from receive thread; mark locally dying
        if (pay.size() >= 2)
            MP_MarkAsteroidDying(ReadU16(&pay[0]));
        break;

    case MSG_ASTEROID_DESTROY_CONFIRM:
        // payload: asteroidId(2)
        if (pay.size() >= 2)
            MP_DestroyAsteroid(ReadU16(&pay[0]));
        break;

    // -----------------------------------------------------------------------
    case MSG_SCORE_UPDATE:
        // payload: playerId(1), score(4) = 5
        if (pay.size() >= 5)
            MP_UpdateScore(pay[0], ReadU32(&pay[1]));
        break;

    case MSG_GAME_OVER:
        // payload: winnerId(1), scores[4](16), top5_names[5][16]+top5_scores[5][4] = 117
        if (pay.size() >= 17)
        {
            int winnerId = pay[0];

            uint32_t scores[4] = {};
            for (int i = 0; i < 4; ++i)
                scores[i] = ReadU32(&pay[1 + i * 4]);

            char     topNames[5][16] = {};
            uint32_t topScores[5]    = {};
            if (pay.size() >= 117)
            {
                for (int i = 0; i < 5; ++i)
                {
                    memcpy(topNames[i], &pay[17 + i * 20], 16);
                    topScores[i] = ReadU32(&pay[17 + i * 20 + 16]);
                }
            }
            MP_SetGameOver(winnerId);
            SetScoreScreenData(winnerId, scores, _playerCount,
                               topNames, topScores);
            _gameOver = true;
        }
        break;

    default:
        break;
    }
}
