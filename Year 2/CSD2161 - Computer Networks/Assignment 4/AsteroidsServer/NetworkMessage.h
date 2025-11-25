/******************************************************************************/
/*!
\file		NetworkMessage.h
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	March 29, 2025
\brief		This header file declares the network message classes for game communication.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/
#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <string>

// Maximum message size
const size_t MAX_MESSAGE_SIZE = 8192;

// Simple Vector2 structure for network transmission
struct NetVector2 {
    float x;
    float y;

    NetVector2() : x(0.0f), y(0.0f) {}
    NetVector2(float _x, float _y) : x(_x), y(_y) {}
};

// Message types
enum class MessageType : uint8_t {
    JOIN_REQUEST,            // Client requests to join server
    JOIN_RESPONSE,           // Server responds to join request
    GAME_START,              // Server signals game start
    PLAYER_INPUT,            // Client sends input updates
    GAME_STATE_UPDATE,       // Server sends game state updates
    OBJECT_CREATE,           // Create a new game object
    OBJECT_DESTROY,          // Destroy a game object
    OBJECT_DESTROY_ACK,      // Acknowledge object destruction
    SCORE_UPDATE,            // Update player scores
    GAME_END,                // Game has ended
    PING,                    // Ping message for latency measurement
    PONG,                    // Pong response to ping
    DISCONNECT               // Client or server disconnecting
};

// Base message class
class NetworkMessage {
public:
    virtual ~NetworkMessage() = default;

    // Serialize message into buffer for network transmission
    virtual std::vector<uint8_t> Serialize() const = 0;

    // Get message type
    virtual MessageType GetType() const = 0;

    // Static method to deserialize a buffer into the appropriate message type
    static NetworkMessage* Deserialize(const std::vector<uint8_t>& buffer);
};

// Join request message
class JoinRequestMessage : public NetworkMessage {
public:
    JoinRequestMessage() : m_playerName("Player") {}
    explicit JoinRequestMessage(const std::string& playerName) : m_playerName(playerName) {}

    MessageType GetType() const override { return MessageType::JOIN_REQUEST; }

    std::vector<uint8_t> Serialize() const override;
    static JoinRequestMessage* Deserialize(const std::vector<uint8_t>& buffer);

    std::string GetPlayerName() const { return m_playerName; }

private:
    std::string m_playerName;
};

// Join response message
class JoinResponseMessage : public NetworkMessage {
public:
    JoinResponseMessage() : m_playerID(0), m_accepted(false) {}
    JoinResponseMessage(uint8_t playerID, bool accepted) : m_playerID(playerID), m_accepted(accepted) {}

    MessageType GetType() const override { return MessageType::JOIN_RESPONSE; }

    std::vector<uint8_t> Serialize() const override;
    static JoinResponseMessage* Deserialize(const std::vector<uint8_t>& buffer);

    uint8_t GetPlayerID() const { return m_playerID; }
    bool IsAccepted() const { return m_accepted; }

private:
    uint8_t m_playerID;
    bool m_accepted;
};

// Game start message
class GameStartMessage : public NetworkMessage {
public:
    GameStartMessage() = default;
    GameStartMessage(uint32_t seed) : m_randomSeed(seed) {}

    MessageType GetType() const override { return MessageType::GAME_START; }

    std::vector<uint8_t> Serialize() const override;
    static GameStartMessage* Deserialize(const std::vector<uint8_t>& buffer);

    uint32_t GetRandomSeed() const { return m_randomSeed; }

private:
    uint32_t m_randomSeed = 0;
};

// Player input message
class PlayerInputMessage : public NetworkMessage {
public:
    PlayerInputMessage() : m_playerID(0), m_inputFlags(0), m_direction(0.0f), m_timestamp(0) {}
    PlayerInputMessage(uint8_t playerID, uint8_t inputFlags, float direction, uint32_t timestamp)
        : m_playerID(playerID), m_inputFlags(inputFlags), m_direction(direction), m_timestamp(timestamp) {
    }

    MessageType GetType() const override { return MessageType::PLAYER_INPUT; }

    std::vector<uint8_t> Serialize() const override;
    static PlayerInputMessage* Deserialize(const std::vector<uint8_t>& buffer);

    uint8_t GetPlayerID() const { return m_playerID; }
    uint8_t GetInputFlags() const { return m_inputFlags; }
    float GetDirection() const { return m_direction; }
    uint32_t GetTimestamp() const { return m_timestamp; }

    // Input flags
    static const uint8_t INPUT_UP = 0x01;
    static const uint8_t INPUT_DOWN = 0x02;
    static const uint8_t INPUT_LEFT = 0x04;
    static const uint8_t INPUT_RIGHT = 0x08;
    static const uint8_t INPUT_FIRE = 0x10;

private:
    uint8_t m_playerID;
    uint8_t m_inputFlags;
    float m_direction;
    uint32_t m_timestamp;
};

// Game object data structure for network transmission
struct NetworkGameObject {
    uint32_t id;
    uint8_t type;
    uint8_t ownerID;
    NetVector2 position;
    NetVector2 velocity;
    float direction;
    NetVector2 scale;
};

// Game state update message
class GameStateUpdateMessage : public NetworkMessage {
public:
    GameStateUpdateMessage() = default;

    MessageType GetType() const override { return MessageType::GAME_STATE_UPDATE; }

    std::vector<uint8_t> Serialize() const override;
    static GameStateUpdateMessage* Deserialize(const std::vector<uint8_t>& buffer);

    void AddObject(const NetworkGameObject& object) { m_objects.push_back(object); }
    const std::vector<NetworkGameObject>& GetObjects() const { return m_objects; }

    void SetSequenceNumber(uint32_t seq) { m_sequenceNumber = seq; }
    uint32_t GetSequenceNumber() const { return m_sequenceNumber; }

private:
    std::vector<NetworkGameObject> m_objects;
    uint32_t m_sequenceNumber = 0;
};

// Object create message
class ObjectCreateMessage : public NetworkMessage {
public:
    ObjectCreateMessage() = default;
    ObjectCreateMessage(const NetworkGameObject& object) : m_object(object) {}

    MessageType GetType() const override { return MessageType::OBJECT_CREATE; }

    std::vector<uint8_t> Serialize() const override;
    static ObjectCreateMessage* Deserialize(const std::vector<uint8_t>& buffer);

    const NetworkGameObject& GetObject() const { return m_object; }

private:
    NetworkGameObject m_object;
};

// Object destroy message
class ObjectDestroyMessage : public NetworkMessage {
public:
    ObjectDestroyMessage() : m_objectID(0), m_sequenceNumber(0) {}
    ObjectDestroyMessage(uint32_t objectID, uint32_t sequenceNumber)
        : m_objectID(objectID), m_sequenceNumber(sequenceNumber) {
    }

    MessageType GetType() const override { return MessageType::OBJECT_DESTROY; }

    std::vector<uint8_t> Serialize() const override;
    static ObjectDestroyMessage* Deserialize(const std::vector<uint8_t>& buffer);

    uint32_t GetObjectID() const { return m_objectID; }
    uint32_t GetSequenceNumber() const { return m_sequenceNumber; }

private:
    uint32_t m_objectID;
    uint32_t m_sequenceNumber;
};

// Object destroy acknowledgment message
class ObjectDestroyAckMessage : public NetworkMessage {
public:
    ObjectDestroyAckMessage() : m_objectID(0), m_sequenceNumber(0), m_playerID(0) {}
    ObjectDestroyAckMessage(uint32_t objectID, uint32_t sequenceNumber, uint8_t playerID)
        : m_objectID(objectID), m_sequenceNumber(sequenceNumber), m_playerID(playerID) {
    }

    MessageType GetType() const override { return MessageType::OBJECT_DESTROY_ACK; }

    std::vector<uint8_t> Serialize() const override;
    static ObjectDestroyAckMessage* Deserialize(const std::vector<uint8_t>& buffer);

    uint32_t GetObjectID() const { return m_objectID; }
    uint32_t GetSequenceNumber() const { return m_sequenceNumber; }
    uint8_t GetPlayerID() const { return m_playerID; }

private:
    uint32_t m_objectID;
    uint32_t m_sequenceNumber;
    uint8_t m_playerID;
};

// Score update message
class ScoreUpdateMessage : public NetworkMessage {
public:
    ScoreUpdateMessage() = default;

    MessageType GetType() const override { return MessageType::SCORE_UPDATE; }

    std::vector<uint8_t> Serialize() const override;
    static ScoreUpdateMessage* Deserialize(const std::vector<uint8_t>& buffer);

    void SetScore(uint8_t playerID, uint32_t score) {
        if (playerID < 4) {
            m_scores[playerID] = score;
        }
    }

    uint32_t GetScore(uint8_t playerID) const {
        return (playerID < 4) ? m_scores[playerID] : 0;
    }

private:
    uint32_t m_scores[4] = { 0 };
};

// Game end message
class GameEndMessage : public NetworkMessage {
public:
    GameEndMessage() : m_winnerID(0) {}
    explicit GameEndMessage(uint8_t winnerID) : m_winnerID(winnerID) {}

    MessageType GetType() const override { return MessageType::GAME_END; }

    std::vector<uint8_t> Serialize() const override;
    static GameEndMessage* Deserialize(const std::vector<uint8_t>& buffer);

    uint8_t GetWinnerID() const { return m_winnerID; }

private:
    uint8_t m_winnerID;
};

// Ping message
class PingMessage : public NetworkMessage {
public:
    PingMessage() : m_timestamp(0) {}
    explicit PingMessage(uint64_t timestamp) : m_timestamp(timestamp) {}

    MessageType GetType() const override { return MessageType::PING; }

    std::vector<uint8_t> Serialize() const override;
    static PingMessage* Deserialize(const std::vector<uint8_t>& buffer);

    uint64_t GetTimestamp() const { return m_timestamp; }

private:
    uint64_t m_timestamp;
};

// Pong message
class PongMessage : public NetworkMessage {
public:
    PongMessage() : m_timestamp(0) {}
    explicit PongMessage(uint64_t timestamp) : m_timestamp(timestamp) {}

    MessageType GetType() const override { return MessageType::PONG; }

    std::vector<uint8_t> Serialize() const override;
    static PongMessage* Deserialize(const std::vector<uint8_t>& buffer);

    uint64_t GetTimestamp() const { return m_timestamp; }

private:
    uint64_t m_timestamp;
};

// Disconnect message
class DisconnectMessage : public NetworkMessage {
public:
    DisconnectMessage() : m_playerID(0) {}
    explicit DisconnectMessage(uint8_t playerID) : m_playerID(playerID) {}

    MessageType GetType() const override { return MessageType::DISCONNECT; }

    std::vector<uint8_t> Serialize() const override;
    static DisconnectMessage* Deserialize(const std::vector<uint8_t>& buffer);

    uint8_t GetPlayerID() const { return m_playerID; }

private:
    uint8_t m_playerID;
};

template<typename T>
void AppendToBuffer(std::vector<uint8_t>& buffer, const T& value) {
    size_t originalSize = buffer.size();
    buffer.resize(originalSize + sizeof(T));
    std::memcpy(buffer.data() + originalSize, &value, sizeof(T));
}

template<typename T>
T ReadFromBuffer(const std::vector<uint8_t>& buffer, size_t& offset) {
    T value;
    std::memcpy(&value, buffer.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}