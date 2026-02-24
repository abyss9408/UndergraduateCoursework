/******************************************************************************/
/*!
\file   NetworkMessage.h
\brief  Network message header and types for Asteroids multiplayer
*/
/******************************************************************************/
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Message types
enum MessageType : uint8_t
{
    // Connection
    MSG_CONNECT_REQUEST          = 0x01,  // C->S  payload: name[16]
    MSG_CONNECT_ACCEPT           = 0x02,  // S->C  payload: assignedId(1), playerCount(1), names[4][16]
    MSG_CONNECT_REJECT           = 0x03,  // S->C  payload: reason(1)
    MSG_DISCONNECT               = 0x04,  // both  payload: playerId(1), reason(1)
    MSG_HEARTBEAT                = 0x05,  // C->S  no payload
    MSG_HEARTBEAT_ACK            = 0x06,  // S->C  no payload

    // Game flow
    MSG_GAME_START               = 0x10,  // S->C  payload: rngSeed(4)
    MSG_GAME_STATE_SYNC          = 0x11,  // S->C  full snapshot
    MSG_GAME_OVER                = 0x12,  // S->C  payload: winnerId(1), scores[4](16), top5(100)

    // Real-time
    MSG_PLAYER_INPUT             = 0x20,  // C->S  payload: inputBits(1), dt(4)
    MSG_SHIP_STATE               = 0x21,  // S->C  payload: playerId(1), posX(4),posY(4),velX(4),velY(4),dir(4)
    MSG_BULLET_SPAWN             = 0x22,  // S->C  payload: bulletId(2), ownerId(1), pos(8), vel(8), dir(4)
    MSG_BULLET_DESTROY           = 0x23,  // S->C  payload: bulletId(2)

    // Asteroids (lockstep on destroy)
    MSG_ASTEROID_SPAWN           = 0x30,  // S->C  payload: id(2), pos(8), vel(8), scale(8)
    MSG_ASTEROID_CORRECT         = 0x31,  // S->C  payload: id(2), pos(8), vel(8)
    MSG_ASTEROID_HIT             = 0x32,  // C->S  payload: asteroidId(2), bulletId(2)
    MSG_ASTEROID_DESTROY         = 0x33,  // S->C  payload: id(2), scoringPlayer(1), spawnCount(1), newIds[](4 each)
    MSG_ASTEROID_DESTROY_ACK     = 0x34,  // C->S  payload: asteroidId(2)
    MSG_ASTEROID_DESTROY_CONFIRM = 0x35,  // S->C  payload: asteroidId(2)

    // Score
    MSG_SCORE_UPDATE             = 0x40,  // S->C  payload: playerId(1), score(4)
};

// ---------------------------------------------------------------------------
// Input bit flags (used in MSG_PLAYER_INPUT)
enum InputBits : uint8_t
{
    INPUT_UP    = 0x01,
    INPUT_DOWN  = 0x02,
    INPUT_LEFT  = 0x04,
    INPUT_RIGHT = 0x08,
    INPUT_SHOOT = 0x10,
};

// ---------------------------------------------------------------------------
// Wire-format header (8 bytes, packed)
#pragma pack(push, 1)
struct MsgHeader
{
    uint8_t  msgType;       // MessageType enum
    uint8_t  senderId;      // 0-3 = players; 0xFF = server
    uint16_t seqNum;        // sequence number
    uint16_t payloadLen;    // payload length in bytes
    uint16_t checksum;      // XOR of header bytes 0-5
};
#pragma pack(pop)

struct NetworkMessage
{
    MsgHeader            header;
    std::vector<uint8_t> payload;
};

// ---------------------------------------------------------------------------
// Helpers
uint16_t ComputeChecksum(const uint8_t* data, size_t len);
bool     BuildPacket(const NetworkMessage& msg, std::vector<uint8_t>& outBuffer);
bool     ParsePacket(const uint8_t* data, size_t len, NetworkMessage& outMsg);
