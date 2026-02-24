/******************************************************************************/
/*!
\file   StandaloneServer.cpp
\brief  Authoritative Asteroids server – game loop, physics, lockstep ACK
*/
/******************************************************************************/
#include "StandaloneServer.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <algorithm>

// ===========================================================================
// Local helpers
// ===========================================================================
static double ClockNow()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static float ReadF32(const uint8_t* p) { float v; memcpy(&v, p, 4); return v; }
static uint16_t ReadU16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }

// ===========================================================================
StandaloneServer::StandaloneServer(uint16_t port, int minPlayers, int maxPlayers)
    : _port(port), _minPlayers(minPlayers), _maxPlayers(maxPlayers)
{
    _rngSeed = static_cast<uint32_t>(std::time(nullptr));
    // Seed a simple LCG used in RandF
}

StandaloneServer::~StandaloneServer()
{
    _running = false;
    _socket.Close();
    if (_recvThread.joinable()) _recvThread.join();
    if (_sendThread.joinable()) _sendThread.join();
}

// ===========================================================================
void StandaloneServer::Run()
{
    if (!_socket.Bind("0.0.0.0", _port))
    {
        printf("[Server] Failed to bind to port %u\n", _port);
        return;
    }
    _socket.SetRecvTimeout(100);
    printf("[Server] Listening on port %u (min=%d max=%d)\n",
           _port, _minPlayers, _maxPlayers);

    _running = true;
    _recvThread = std::thread(&StandaloneServer::ReceiveThreadFunc, this);
    _sendThread = std::thread(&StandaloneServer::SendThreadFunc, this);

    GameLoopFunc();  // blocks until _running = false

    _running = false;
    // Wake up send thread
    _outCV.notify_all();
    if (_sendThread.joinable()) _sendThread.join();
    if (_recvThread.joinable()) _recvThread.join();
}

// ===========================================================================
// Receive thread
// ===========================================================================
void StandaloneServer::ReceiveThreadFunc()
{
    std::vector<uint8_t> buf;
    sockaddr_in from{};

    while (_running)
    {
        int r = _socket.RecvFrom(buf, from);
        if (r <= 0) continue;

        NetworkMessage msg;
        if (!ParsePacket(buf.data(), static_cast<size_t>(r), msg))
            continue;

        std::lock_guard<std::mutex> lock(_inMutex);
        _inboundQueue.push({ msg, from });
        _inCV.notify_one();
    }
}

// ===========================================================================
// Send thread
// ===========================================================================
void StandaloneServer::SendThreadFunc()
{
    while (_running)
    {
        std::unique_lock<std::mutex> lock(_outMutex);
        _outCV.wait_for(lock, std::chrono::milliseconds(10),
                        [this] { return !_outboundQueue.empty() || !_running; });

        while (!_outboundQueue.empty())
        {
            OutPacket pkt = std::move(_outboundQueue.front());
            _outboundQueue.pop();
            lock.unlock();
            _socket.SendTo(pkt.data.data(), pkt.data.size(), pkt.dest);
            lock.lock();
        }
    }
}

// ===========================================================================
// Game loop (main server thread)
// ===========================================================================
void StandaloneServer::GameLoopFunc()
{
    using namespace std::chrono;
    const microseconds TICK_DUR(1000000 / 60);
    auto lastTick = steady_clock::now();

    while (_running)
    {
        auto now = steady_clock::now();
        if (now - lastTick < TICK_DUR)
        {
            std::this_thread::sleep_for(milliseconds(1));
            continue;
        }
        float dt = duration<float>(now - lastTick).count();
        lastTick = now;
        ++_tickCount;

        ProcessInboundMessages();

        if (_gameRunning)
        {
            TickShips(dt);
            TickBullets(dt);
            TickAsteroids(dt);
            CheckCollisions();
            CheckPendingAsteroidAcks();

            if (_tickCount % 2 == 0)  BroadcastShipStates();
            if (_tickCount % 10 == 0) BroadcastAsteroidCorrections();
        }

        CheckHeartbeatTimeouts();
    }
}

// ===========================================================================
// Inbound message processing
// ===========================================================================
void StandaloneServer::ProcessInboundMessages()
{
    std::queue<InPacket> local;
    {
        std::lock_guard<std::mutex> lock(_inMutex);
        std::swap(local, _inboundQueue);
    }
    while (!local.empty())
    {
        auto& pkt = local.front();
        switch (pkt.msg.header.msgType)
        {
        case MSG_CONNECT_REQUEST:  HandleConnect(pkt.msg, pkt.from);            break;
        case MSG_DISCONNECT:       HandleDisconnect(pkt.msg, pkt.from);         break;
        case MSG_HEARTBEAT:        HandleHeartbeat(pkt.msg, pkt.from);          break;
        case MSG_PLAYER_INPUT:     HandlePlayerInput(pkt.msg, pkt.from);        break;
        case MSG_ASTEROID_HIT:     HandleAsteroidHit(pkt.msg, pkt.from);        break;
        case MSG_ASTEROID_DESTROY_ACK: HandleAsteroidDestroyAck(pkt.msg, pkt.from); break;
        default: break;
        }
        local.pop();
    }
}

// ===========================================================================
// Handlers
// ===========================================================================
void StandaloneServer::HandleConnect(const NetworkMessage& msg, const sockaddr_in& from)
{
    // If player already connected (reconnect), restore state
    int existingIdx = FindPlayer(from);

    // Extract name
    char name[17] = {};
    if (msg.payload.size() >= 16)
    {
        memcpy(name, msg.payload.data(), 16);
        name[16] = '\0';
    }

    if (existingIdx >= 0)
    {
        // Reconnect
        printf("[Server] Player %d reconnected\n", existingIdx);
        _players[existingIdx].lastMsgTime = ClockNow();
        // TODO: send full game state sync
        return;
    }

    // Find free slot
    int slot = -1;
    for (int i = 0; i < _maxPlayers && i < MAX_PLAYERS; ++i)
    {
        if (!_players[i].active) { slot = i; break; }
    }

    if (slot < 0)
    {
        // Reject: full
        NetworkMessage rej{};
        rej.header.msgType  = MSG_CONNECT_REJECT;
        rej.header.senderId = 0xFF;
        rej.header.seqNum   = ++_serverSeq;
        rej.payload         = {1}; // reason = full
        Enqueue(rej, -1);  // send directly
        std::vector<uint8_t> buf;
        BuildPacket(rej, buf);
        _socket.SendTo(buf.data(), buf.size(), from);
        return;
    }

    _players[slot].active      = true;
    _players[slot].addr        = from;
    _players[slot].score       = 0;
    _players[slot].posX        = 0.f;
    _players[slot].posY        = static_cast<float>(slot) * 60.f - 90.f;
    _players[slot].velX        = 0.f;
    _players[slot].velY        = 0.f;
    _players[slot].dirCurr     = 0.f;
    _players[slot].inputBits   = 0;
    _players[slot].lastMsgTime = ClockNow();
    strncpy_s(_players[slot].name, 17, name, _TRUNCATE);
    ++_connectedCount;

    printf("[Server] Player %d connected: '%s' (%d/%d)\n",
           slot, name, _connectedCount, _minPlayers);

    // Build accept: assignedId(1) + playerCount(1) + names[4][16]
    NetworkMessage acc{};
    acc.header.msgType  = MSG_CONNECT_ACCEPT;
    acc.header.senderId = 0xFF;
    acc.header.seqNum   = ++_serverSeq;
    acc.payload.resize(2 + 4 * 16, 0);
    acc.payload[0] = static_cast<uint8_t>(slot);
    acc.payload[1] = static_cast<uint8_t>(_connectedCount);
    for (int i = 0; i < MAX_PLAYERS; ++i)
        memcpy(&acc.payload[2 + i * 16], _players[i].name, 16);

    std::vector<uint8_t> buf;
    BuildPacket(acc, buf);
    _socket.SendTo(buf.data(), buf.size(), from);

    // Start game when minPlayers reached
    if (_connectedCount >= _minPlayers && !_gameRunning)
    {
        printf("[Server] Starting game!\n");
        _gameRunning = true;

        // Send GAME_START FIRST so clients can initialise before receiving spawns
        NetworkMessage gs{};
        gs.header.msgType  = MSG_GAME_START;
        gs.header.senderId = 0xFF;
        gs.header.seqNum   = ++_serverSeq;
        gs.payload.resize(4);
        memcpy(gs.payload.data(), &_rngSeed, 4);
        EnqueueAll(gs);

        SpawnInitialAsteroids();  // queued after GAME_START
    }
}

void StandaloneServer::HandleDisconnect(const NetworkMessage& msg, const sockaddr_in& from)
{
    int idx = FindPlayer(from);
    if (idx < 0) return;
    printf("[Server] Player %d disconnected\n", idx);
    _players[idx].active = false;
    --_connectedCount;
}

void StandaloneServer::HandleHeartbeat(const NetworkMessage& msg, const sockaddr_in& from)
{
    int idx = FindPlayer(from);
    if (idx >= 0) _players[idx].lastMsgTime = ClockNow();

    NetworkMessage ack{};
    ack.header.msgType  = MSG_HEARTBEAT_ACK;
    ack.header.senderId = 0xFF;
    ack.header.seqNum   = ++_serverSeq;
    std::vector<uint8_t> buf;
    BuildPacket(ack, buf);
    _socket.SendTo(buf.data(), buf.size(), from);
}

void StandaloneServer::HandlePlayerInput(const NetworkMessage& msg, const sockaddr_in& from)
{
    int idx = FindPlayer(from);
    if (idx < 0) return;
    _players[idx].lastMsgTime = ClockNow();

    if (msg.payload.size() >= 1)
    {
        _players[idx].inputBits = msg.payload[0];
    }

    // Shoot handling: create bullet when SHOOT bit is triggered
    // We track the previous inputBits to detect rising edge
    static uint8_t prevBits[MAX_PLAYERS] = {};
    uint8_t cur = _players[idx].inputBits;
    if ((cur & INPUT_SHOOT) && !(prevBits[idx] & INPUT_SHOOT))
    {
        // Spawn 3 bullets (matching SP game: spread)
        float dirs[3] = {
            _players[idx].dirCurr + 0.15708f,  // +PI/20
            _players[idx].dirCurr,
            _players[idx].dirCurr - 0.15708f   // -PI/20
        };
        for (int k = 0; k < 3; ++k)
        {
            float d  = dirs[k];
            float vx = cosf(d) * BULLET_SPEED;
            float vy = sinf(d) * BULLET_SPEED;

            // Find free bullet slot
            int bslot = -1;
            for (int b = 0; b < MAX_BULLETS; ++b)
                if (!_bullets[b].active) { bslot = b; break; }

            if (bslot < 0) continue;

            _bullets[bslot].active  = true;
            _bullets[bslot].id      = _nextBulletId++;
            _bullets[bslot].ownerId = idx;
            _bullets[bslot].posX    = _players[idx].posX;
            _bullets[bslot].posY    = _players[idx].posY;
            _bullets[bslot].velX    = vx;
            _bullets[bslot].velY    = vy;
            _bullets[bslot].dirCurr = d;

            // Broadcast MSG_BULLET_SPAWN
            // payload: bulletId(2), ownerId(1), posX(4),posY(4),velX(4),velY(4),dir(4) = 23
            NetworkMessage bs{};
            bs.header.msgType  = MSG_BULLET_SPAWN;
            bs.header.senderId = 0xFF;
            bs.header.seqNum   = ++_serverSeq;
            bs.payload.resize(23);
            memcpy(&bs.payload[0],  &_bullets[bslot].id,   2);
            bs.payload[2] = static_cast<uint8_t>(idx);
            memcpy(&bs.payload[3],  &_players[idx].posX, 4);
            memcpy(&bs.payload[7],  &_players[idx].posY, 4);
            memcpy(&bs.payload[11], &vx, 4);
            memcpy(&bs.payload[15], &vy, 4);
            memcpy(&bs.payload[19], &d,  4);
            EnqueueAll(bs);
        }
    }
    prevBits[idx] = cur;
}

void StandaloneServer::HandleAsteroidHit(const NetworkMessage& msg, const sockaddr_in& from)
{
    int idx = FindPlayer(from);
    if (idx < 0 || msg.payload.size() < 4) return;

    uint16_t astId = ReadU16(&msg.payload[0]);
    uint16_t bulId = ReadU16(&msg.payload[2]);

    // Find asteroid by ID
    for (int a = 0; a < MAX_ASTEROIDS; ++a)
    {
        AsteroidState& ast = _asteroids[a];
        if (!ast.active || ast.dying || ast.id != astId)
            continue;

        // Validate bullet ownership for the reporting player
        bool bulletValid = false;
        for (int b = 0; b < MAX_BULLETS; ++b)
        {
            if (_bullets[b].active && _bullets[b].id == bulId)
            {
                bulletValid = true;
                // Destroy the bullet
                _bullets[b].active = false;

                NetworkMessage bd{};
                bd.header.msgType  = MSG_BULLET_DESTROY;
                bd.header.senderId = 0xFF;
                bd.header.seqNum   = ++_serverSeq;
                bd.payload.resize(2);
                memcpy(bd.payload.data(), &bulId, 2);
                EnqueueAll(bd);
                break;
            }
        }
        if (!bulletValid) return;

        InitiateAsteroidDestroy(static_cast<uint16_t>(a), idx);
        return;
    }
}

void StandaloneServer::HandleAsteroidDestroyAck(const NetworkMessage& msg,
                                                const sockaddr_in& from)
{
    int idx = FindPlayer(from);
    if (idx < 0 || msg.payload.size() < 2) return;

    uint16_t astId = ReadU16(&msg.payload[0]);
    for (int a = 0; a < MAX_ASTEROIDS; ++a)
    {
        AsteroidState& ast = _asteroids[a];
        if (ast.active && ast.dying && ast.id == astId)
        {
            ast.ackBits |= (1 << idx);
            return;
        }
    }
}

// ===========================================================================
// Physics ticks
// ===========================================================================
void StandaloneServer::TickShips(float dt)
{
    for (int i = 0; i < MAX_PLAYERS; ++i)
    {
        PlayerInfo& p = _players[i];
        if (!p.active) continue;

        if (p.inputBits & INPUT_UP)
        {
            p.velX += cosf(p.dirCurr) * SHIP_ACCEL * dt;
            p.velY += sinf(p.dirCurr) * SHIP_ACCEL * dt;
        }
        if (p.inputBits & INPUT_DOWN)
        {
            p.velX -= cosf(p.dirCurr) * SHIP_ACCEL * dt;
            p.velY -= sinf(p.dirCurr) * SHIP_ACCEL * dt;
        }
        if (p.inputBits & INPUT_LEFT)
            p.dirCurr += SHIP_ROT_SPEED * dt;
        if (p.inputBits & INPUT_RIGHT)
            p.dirCurr -= SHIP_ROT_SPEED * dt;

        // Velocity dampening
        p.velX *= 0.99f;
        p.velY *= 0.99f;

        p.posX += p.velX * dt;
        p.posY += p.velY * dt;

        // Screen wrap
        p.posX = WrapVal(p.posX, WIN_MIN_X - SHIP_SCALE, WIN_MAX_X + SHIP_SCALE);
        p.posY = WrapVal(p.posY, WIN_MIN_Y - SHIP_SCALE, WIN_MAX_Y + SHIP_SCALE);
    }
}

void StandaloneServer::TickBullets(float dt)
{
    for (int b = 0; b < MAX_BULLETS; ++b)
    {
        BulletState& bul = _bullets[b];
        if (!bul.active) continue;

        bul.posX += bul.velX * dt;
        bul.posY += bul.velY * dt;

        // Destroy when out of bounds
        if (bul.posX < WIN_MIN_X || bul.posX > WIN_MAX_X ||
            bul.posY < WIN_MIN_Y || bul.posY > WIN_MAX_Y)
        {
            bul.active = false;
            NetworkMessage bd{};
            bd.header.msgType  = MSG_BULLET_DESTROY;
            bd.header.senderId = 0xFF;
            bd.header.seqNum   = ++_serverSeq;
            bd.payload.resize(2);
            memcpy(bd.payload.data(), &bul.id, 2);
            EnqueueAll(bd);
        }
    }
}

void StandaloneServer::TickAsteroids(float dt)
{
    for (int a = 0; a < MAX_ASTEROIDS; ++a)
    {
        AsteroidState& ast = _asteroids[a];
        if (!ast.active || ast.dying) continue;

        ast.posX += ast.velX * dt;
        ast.posY += ast.velY * dt;

        ast.posX = WrapVal(ast.posX, WIN_MIN_X - ast.scaleX, WIN_MAX_X + ast.scaleX);
        ast.posY = WrapVal(ast.posY, WIN_MIN_Y - ast.scaleY, WIN_MAX_Y + ast.scaleY);
    }
}

// ===========================================================================
// Collision detection (server-authoritative, simple AABB)
// ===========================================================================
static bool AabbOverlap(float ax, float ay, float ahw, float ahh,
                         float bx, float by, float bhw, float bhh)
{
    return fabsf(ax - bx) < (ahw + bhw) && fabsf(ay - by) < (ahh + bhh);
}

void StandaloneServer::CheckCollisions()
{
    for (int a = 0; a < MAX_ASTEROIDS; ++a)
    {
        AsteroidState& ast = _asteroids[a];
        if (!ast.active || ast.dying) continue;

        float ahw = ast.scaleX * 0.5f;
        float ahh = ast.scaleY * 0.5f;

        // Asteroid vs bullets
        for (int b = 0; b < MAX_BULLETS; ++b)
        {
            BulletState& bul = _bullets[b];
            if (!bul.active) continue;

            if (AabbOverlap(ast.posX, ast.posY, ahw, ahh,
                            bul.posX, bul.posY, BULLET_SCALE_X * 0.5f, BULLET_SCALE_Y * 0.5f))
            {
                // Destroy bullet, initiate asteroid destroy
                int owner = bul.ownerId;
                bul.active = false;

                NetworkMessage bd{};
                bd.header.msgType  = MSG_BULLET_DESTROY;
                bd.header.senderId = 0xFF;
                bd.header.seqNum   = ++_serverSeq;
                bd.payload.resize(2);
                memcpy(bd.payload.data(), &bul.id, 2);
                EnqueueAll(bd);

                InitiateAsteroidDestroy(static_cast<uint16_t>(a), owner);
                break;
            }
        }
    }
}

// ===========================================================================
// Asteroid lifecycle – lockstep destroy
// ===========================================================================
void StandaloneServer::SpawnInitialAsteroids()
{
    struct { float px, py, vx, vy, sx, sy; } init[4] = {
        {  90.f, -220.f, -60.f,  -30.f, AST_MIN_SCALE, AST_MAX_SCALE },
        {-260.f, -250.f,  39.f, -130.f, AST_MAX_SCALE, AST_MIN_SCALE },
        {-260.f,  100.f,  90.f,   20.f, AST_MIN_SCALE, AST_MIN_SCALE },
        { 120.f, -180.f, -90.f,   76.f, AST_MAX_SCALE, AST_MAX_SCALE },
    };
    for (auto& d : init)
        SpawnAsteroid(d.px, d.py, d.vx, d.vy, d.sx, d.sy);
}

uint16_t StandaloneServer::SpawnAsteroid(float px, float py,
                                          float vx, float vy,
                                          float sx, float sy)
{
    int slot = -1;
    for (int a = 0; a < MAX_ASTEROIDS; ++a)
        if (!_asteroids[a].active) { slot = a; break; }
    if (slot < 0) return 0xFFFF;

    AsteroidState& ast = _asteroids[slot];
    ast            = {};
    ast.active     = true;
    ast.id         = _nextAsteroidId++;
    ast.posX = px;  ast.posY = py;
    ast.velX = vx;  ast.velY = vy;
    ast.scaleX= sx; ast.scaleY = sy;

    // Broadcast MSG_ASTEROID_SPAWN
    // payload: id(2), posX(4),posY(4),velX(4),velY(4),scaleX(4),scaleY(4) = 26
    NetworkMessage sp{};
    sp.header.msgType  = MSG_ASTEROID_SPAWN;
    sp.header.senderId = 0xFF;
    sp.header.seqNum   = ++_serverSeq;
    sp.payload.resize(26);
    memcpy(&sp.payload[0],  &ast.id,    2);
    memcpy(&sp.payload[2],  &px,  4);
    memcpy(&sp.payload[6],  &py,  4);
    memcpy(&sp.payload[10], &vx,  4);
    memcpy(&sp.payload[14], &vy,  4);
    memcpy(&sp.payload[18], &sx,  4);
    memcpy(&sp.payload[22], &sy,  4);
    EnqueueAll(sp);

    return ast.id;
}

void StandaloneServer::InitiateAsteroidDestroy(uint16_t astIdx, int scoringPlayer)
{
    AsteroidState& ast = _asteroids[astIdx];
    if (!ast.active || ast.dying) return;

    ast.dying        = true;
    ast.destroyTime  = ClockNow();
    ast.scoringPlayer= scoringPlayer;
    ast.ackBits      = 0;
    ast.expectedAckBits = BuildExpectedAckBits();
    ast.confirmRetries  = 0;
    ast.nextRetryTime   = ClockNow() + CONFIRM_INTERVAL;

    // Award score
    if (scoringPlayer >= 0)
        UpdateScore(scoringPlayer, 100);

    // Broadcast MSG_ASTEROID_DESTROY
    // payload: id(2), scoringPlayer(1), spawnCount(1)
    NetworkMessage d{};
    d.header.msgType  = MSG_ASTEROID_DESTROY;
    d.header.senderId = 0xFF;
    d.header.seqNum   = ++_serverSeq;
    d.payload.resize(4);
    memcpy(&d.payload[0], &ast.id, 2);
    d.payload[2] = static_cast<uint8_t>(scoringPlayer < 0 ? 0xFF : scoringPlayer);
    d.payload[3] = 0; // child spawn count (sent later via CONFIRM)
    EnqueueAll(d);
}

void StandaloneServer::CheckPendingAsteroidAcks()
{
    double now = ClockNow();

    for (int a = 0; a < MAX_ASTEROIDS; ++a)
    {
        AsteroidState& ast = _asteroids[a];
        if (!ast.active || !ast.dying) continue;

        // Force-set ACK bits for timed-out players
        if (now - ast.destroyTime > ACK_TIMEOUT_SEC)
            ast.ackBits = ast.expectedAckBits; // force all

        // All ACKs collected?
        if ((ast.ackBits & ast.expectedAckBits) == ast.expectedAckBits)
        {
            ConfirmAsteroidDestroy(static_cast<uint16_t>(a));
            continue;
        }

        // Retransmit CONFIRM for stragglers
        if (now >= ast.nextRetryTime && ast.confirmRetries < CONFIRM_RETRIES)
        {
            // Resend MSG_ASTEROID_DESTROY to players that haven't ACK'd
            NetworkMessage d{};
            d.header.msgType  = MSG_ASTEROID_DESTROY;
            d.header.senderId = 0xFF;
            d.header.seqNum   = ++_serverSeq;
            d.payload.resize(4);
            memcpy(&d.payload[0], &ast.id, 2);
            d.payload[2] = static_cast<uint8_t>(
                ast.scoringPlayer < 0 ? 0xFF : ast.scoringPlayer);
            d.payload[3] = 0;
            for (int p = 0; p < MAX_PLAYERS; ++p)
            {
                if (_players[p].active && !(ast.ackBits & (1 << p)))
                    Enqueue(d, p);
            }
            ++ast.confirmRetries;
            ast.nextRetryTime = now + CONFIRM_INTERVAL;
        }

        // Give up after retries exceeded, force confirm
        if (ast.confirmRetries >= CONFIRM_RETRIES)
            ast.ackBits = ast.expectedAckBits;
    }
}

void StandaloneServer::ConfirmAsteroidDestroy(uint16_t astIdx)
{
    AsteroidState& ast = _asteroids[astIdx];

    // Broadcast MSG_ASTEROID_DESTROY_CONFIRM
    NetworkMessage cf{};
    cf.header.msgType  = MSG_ASTEROID_DESTROY_CONFIRM;
    cf.header.senderId = 0xFF;
    cf.header.seqNum   = ++_serverSeq;
    cf.payload.resize(2);
    memcpy(cf.payload.data(), &ast.id, 2);
    EnqueueAll(cf);

    // Spawn 1-2 replacement asteroids
    int count = 1 + (rand() % 2);
    for (int k = 0; k < count; ++k)
    {
        float sx = RandF(AST_MIN_SCALE, AST_MAX_SCALE);
        float sy = RandF(AST_MIN_SCALE, AST_MAX_SCALE);
        float px = RandFExcl(WIN_MIN_X * 2.f, WIN_MAX_X * 2.f, WIN_MIN_X, WIN_MAX_X);
        float py = RandFExcl(WIN_MIN_Y * 2.f, WIN_MAX_Y * 2.f, WIN_MIN_Y, WIN_MAX_Y);
        float vx = RandF(-75.f, 75.f);
        float vy = RandF(-100.f, 100.f);
        SpawnAsteroid(px, py, vx, vy, sx, sy);
    }

    // Deactivate
    ast.active = false;
    ast.dying  = false;
}

// ===========================================================================
// Periodic broadcasts
// ===========================================================================
void StandaloneServer::BroadcastShipStates()
{
    for (int i = 0; i < MAX_PLAYERS; ++i)
    {
        if (!_players[i].active) continue;

        NetworkMessage msg{};
        msg.header.msgType  = MSG_SHIP_STATE;
        msg.header.senderId = 0xFF;
        msg.header.seqNum   = ++_serverSeq;
        msg.payload.resize(21);
        msg.payload[0] = static_cast<uint8_t>(i);
        memcpy(&msg.payload[1],  &_players[i].posX,    4);
        memcpy(&msg.payload[5],  &_players[i].posY,    4);
        memcpy(&msg.payload[9],  &_players[i].velX,    4);
        memcpy(&msg.payload[13], &_players[i].velY,    4);
        memcpy(&msg.payload[17], &_players[i].dirCurr, 4);
        EnqueueAll(msg);
    }
}

void StandaloneServer::BroadcastAsteroidCorrections()
{
    // Send MSG_ASTEROID_SPAWN (26 bytes) with current position instead of CORRECT (18 bytes).
    // Clients that already know the asteroid treat it as a position correction (idempotent).
    // Clients that missed the initial spawn (lobby-to-game race) get to create the asteroid here.
    for (int a = 0; a < MAX_ASTEROIDS; ++a)
    {
        AsteroidState& ast = _asteroids[a];
        if (!ast.active || ast.dying) continue;

        NetworkMessage msg{};
        msg.header.msgType  = MSG_ASTEROID_SPAWN;
        msg.header.senderId = 0xFF;
        msg.header.seqNum   = ++_serverSeq;
        msg.payload.resize(26);
        memcpy(&msg.payload[0],  &ast.id,     2);
        memcpy(&msg.payload[2],  &ast.posX,   4);
        memcpy(&msg.payload[6],  &ast.posY,   4);
        memcpy(&msg.payload[10], &ast.velX,   4);
        memcpy(&msg.payload[14], &ast.velY,   4);
        memcpy(&msg.payload[18], &ast.scaleX, 4);
        memcpy(&msg.payload[22], &ast.scaleY, 4);
        EnqueueAll(msg);
    }
}

// ===========================================================================
// Score / Win
// ===========================================================================
void StandaloneServer::UpdateScore(int playerIdx, uint32_t addScore)
{
    if (playerIdx < 0 || playerIdx >= MAX_PLAYERS || !_players[playerIdx].active)
        return;

    _players[playerIdx].score += addScore;

    NetworkMessage su{};
    su.header.msgType  = MSG_SCORE_UPDATE;
    su.header.senderId = 0xFF;
    su.header.seqNum   = ++_serverSeq;
    su.payload.resize(5);
    su.payload[0] = static_cast<uint8_t>(playerIdx);
    memcpy(&su.payload[1], &_players[playerIdx].score, 4);
    EnqueueAll(su);

    CheckWinCondition();
}

void StandaloneServer::CheckWinCondition()
{
    for (int i = 0; i < MAX_PLAYERS; ++i)
    {
        if (_players[i].active && _players[i].score >= static_cast<uint32_t>(WIN_SCORE))
        {
            SendGameOver(i);
            _gameRunning = false;
            return;
        }
    }
}

void StandaloneServer::SendGameOver(int winnerId)
{
    printf("[Server] Game over! Winner: Player %d (%s) with %u pts\n",
           winnerId, _players[winnerId].name, _players[winnerId].score);

    // Save winner's score to leaderboard
    SaveScore(_players[winnerId].name, _players[winnerId].score);

    // Build top-5 from scores.txt
    struct Entry { char name[17]; uint32_t score; };
    std::vector<Entry> entries;
    {
        std::ifstream f("scores.txt");
        std::string line;
        while (std::getline(f, line))
        {
            auto c1 = line.find(',');
            auto c2 = line.find(',', c1 + 1);
            if (c1 == std::string::npos) continue;
            Entry e{};
            strncpy_s(e.name, 17, line.substr(0, c1).c_str(), _TRUNCATE);
            e.score = static_cast<uint32_t>(std::stoul(line.substr(c1 + 1, c2 - c1 - 1)));
            entries.push_back(e);
        }
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.score > b.score; });
    if (entries.size() > 5) entries.resize(5);

    // MSG_GAME_OVER payload: winnerId(1) + scores[4](16) + top5(100) = 117
    NetworkMessage go{};
    go.header.msgType  = MSG_GAME_OVER;
    go.header.senderId = 0xFF;
    go.header.seqNum   = ++_serverSeq;
    go.payload.resize(117, 0);
    go.payload[0] = static_cast<uint8_t>(winnerId);
    for (int i = 0; i < MAX_PLAYERS; ++i)
        memcpy(&go.payload[1 + i * 4], &_players[i].score, 4);

    for (int i = 0; i < (int)entries.size() && i < 5; ++i)
    {
        memcpy(&go.payload[17 + i * 20],      entries[i].name,  16);
        memcpy(&go.payload[17 + i * 20 + 16], &entries[i].score, 4);
    }
    EnqueueAll(go);
}

void StandaloneServer::SaveScore(const char* name, uint32_t score)
{
    std::ofstream f("scores.txt", std::ios::app);
    if (!f.is_open()) return;

    std::time_t t = std::time(nullptr);
    char buf[32];
    struct tm tmBuf;
    localtime_s(&tmBuf, &t);
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmBuf);
    f << name << "," << score << "," << buf << "\n";
}

// ===========================================================================
// Heartbeat timeout
// ===========================================================================
void StandaloneServer::CheckHeartbeatTimeouts()
{
    double now = ClockNow();
    for (int i = 0; i < MAX_PLAYERS; ++i)
    {
        if (!_players[i].active) continue;
        if (now - _players[i].lastMsgTime > HB_TIMEOUT_SEC)
        {
            printf("[Server] Player %d timed out\n", i);
            _players[i].active = false;
            --_connectedCount;

            // Force-ACK all pending asteroid destroys for this player
            for (int a = 0; a < MAX_ASTEROIDS; ++a)
            {
                if (_asteroids[a].active && _asteroids[a].dying)
                    _asteroids[a].ackBits |= (1 << i);
            }

            // Broadcast disconnect
            NetworkMessage dc{};
            dc.header.msgType  = MSG_DISCONNECT;
            dc.header.senderId = 0xFF;
            dc.header.seqNum   = ++_serverSeq;
            dc.payload         = { static_cast<uint8_t>(i), 1 };
            EnqueueAll(dc);
        }
    }
}

// ===========================================================================
// Queue / send helpers
// ===========================================================================
void StandaloneServer::Enqueue(const NetworkMessage& msg, int playerIdx)
{
    if (playerIdx >= 0 && playerIdx < MAX_PLAYERS && !_players[playerIdx].active)
        return;
    std::vector<uint8_t> buf;
    BuildPacket(msg, buf);

    OutPacket pkt;
    pkt.dest = (playerIdx >= 0) ? _players[playerIdx].addr : sockaddr_in{};
    pkt.data = std::move(buf);

    std::lock_guard<std::mutex> lock(_outMutex);
    _outboundQueue.push(std::move(pkt));
    _outCV.notify_one();
}

void StandaloneServer::EnqueueAll(const NetworkMessage& msg)
{
    std::vector<uint8_t> buf;
    BuildPacket(msg, buf);

    std::lock_guard<std::mutex> lock(_outMutex);
    for (int i = 0; i < MAX_PLAYERS; ++i)
    {
        if (!_players[i].active) continue;
        OutPacket pkt;
        pkt.dest = _players[i].addr;
        pkt.data = buf;
        _outboundQueue.push(std::move(pkt));
    }
    _outCV.notify_one();
}

void StandaloneServer::BroadcastToAll(const NetworkMessage& msg)
{
    EnqueueAll(msg);
}

void StandaloneServer::SendTo(const NetworkMessage& msg, int playerIdx)
{
    Enqueue(msg, playerIdx);
}

// ===========================================================================
// Utility
// ===========================================================================
uint8_t StandaloneServer::BuildExpectedAckBits() const
{
    uint8_t bits = 0;
    for (int i = 0; i < MAX_PLAYERS; ++i)
        if (_players[i].active) bits |= (1 << i);
    return bits;
}

int StandaloneServer::FindPlayer(const sockaddr_in& addr) const
{
    for (int i = 0; i < MAX_PLAYERS; ++i)
    {
        if (_players[i].active &&
            _players[i].addr.sin_addr.s_addr == addr.sin_addr.s_addr &&
            _players[i].addr.sin_port        == addr.sin_port)
            return i;
    }
    return -1;
}

double StandaloneServer::NowSec() const { return ClockNow(); }

float StandaloneServer::RandF(float lo, float hi)
{
    return lo + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / (hi - lo));
}

float StandaloneServer::RandFExcl(float lo, float hi, float exLo, float exHi)
{
    float v;
    do { v = RandF(lo, hi); } while (v >= exLo && v <= exHi);
    return v;
}

float StandaloneServer::WrapVal(float v, float lo, float hi) const
{
    float range = hi - lo;
    while (v < lo)  v += range;
    while (v >= hi) v -= range;
    return v;
}
