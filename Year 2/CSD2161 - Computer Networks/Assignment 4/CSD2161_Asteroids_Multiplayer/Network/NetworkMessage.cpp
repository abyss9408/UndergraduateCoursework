/******************************************************************************/
/*!
\file		NetworkMessage.cpp
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	March 29, 2025
\brief		This source file implements the network message classes for game communication.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#include "NetworkMessage.h"

void AppendStringToBuffer(std::vector<uint8_t>& buffer, const std::string& str) {
    uint16_t length = static_cast<uint16_t>(str.length());
    AppendToBuffer(buffer, length);

    size_t originalSize = buffer.size();
    buffer.resize(originalSize + length);
    std::memcpy(buffer.data() + originalSize, str.data(), length);
}

std::string ReadStringFromBuffer(const std::vector<uint8_t>& buffer, size_t& offset) {
    uint16_t length = ReadFromBuffer<uint16_t>(buffer, offset);

    std::string str(reinterpret_cast<const char*>(buffer.data() + offset), length);
    offset += length;

    return str;
}

// NetworkMessage static deserialize
NetworkMessage* NetworkMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 1) {
        return nullptr;
    }

    MessageType type = static_cast<MessageType>(buffer[0]);

    switch (type) {
    case MessageType::JOIN_REQUEST:
        return JoinRequestMessage::Deserialize(buffer);
    case MessageType::JOIN_RESPONSE:
        return JoinResponseMessage::Deserialize(buffer);
    case MessageType::GAME_START:
        return GameStartMessage::Deserialize(buffer);
    case MessageType::PLAYER_INPUT:
        return PlayerInputMessage::Deserialize(buffer);
    case MessageType::GAME_STATE_UPDATE:
        return GameStateUpdateMessage::Deserialize(buffer);
    case MessageType::OBJECT_CREATE:
        return ObjectCreateMessage::Deserialize(buffer);
    case MessageType::OBJECT_DESTROY:
        return ObjectDestroyMessage::Deserialize(buffer);
    case MessageType::OBJECT_DESTROY_ACK:
        return ObjectDestroyAckMessage::Deserialize(buffer);
    case MessageType::SCORE_UPDATE:
        return ScoreUpdateMessage::Deserialize(buffer);
    case MessageType::GAME_END:
        return GameEndMessage::Deserialize(buffer);
    case MessageType::PING:
        return PingMessage::Deserialize(buffer);
    case MessageType::PONG:
        return PongMessage::Deserialize(buffer);
    case MessageType::DISCONNECT:
        return DisconnectMessage::Deserialize(buffer);
    default:
        return nullptr;
    }
}

// JoinRequestMessage
std::vector<uint8_t> JoinRequestMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    AppendStringToBuffer(buffer, m_playerName);
    return buffer;
}

JoinRequestMessage* JoinRequestMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 3 || buffer[0] != static_cast<uint8_t>(MessageType::JOIN_REQUEST)) {
        return nullptr;
    }

    size_t offset = 1;
    std::string playerName = ReadStringFromBuffer(buffer, offset);

    return new JoinRequestMessage(playerName);
}

// JoinResponseMessage
std::vector<uint8_t> JoinResponseMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    buffer.push_back(m_playerID);
    buffer.push_back(m_accepted ? 1 : 0);
    return buffer;
}

JoinResponseMessage* JoinResponseMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 3 || buffer[0] != static_cast<uint8_t>(MessageType::JOIN_RESPONSE)) {
        return nullptr;
    }

    uint8_t playerID = buffer[1];
    bool accepted = buffer[2] != 0;

    return new JoinResponseMessage(playerID, accepted);
}

// GameStartMessage
std::vector<uint8_t> GameStartMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    AppendToBuffer(buffer, m_randomSeed);
    return buffer;
}

GameStartMessage* GameStartMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 5 || buffer[0] != static_cast<uint8_t>(MessageType::GAME_START)) {
        return nullptr;
    }

    size_t offset = 1;
    uint32_t seed = ReadFromBuffer<uint32_t>(buffer, offset);

    return new GameStartMessage(seed);
}

// PlayerInputMessage
std::vector<uint8_t> PlayerInputMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    buffer.push_back(m_playerID);
    buffer.push_back(m_inputFlags);
    AppendToBuffer(buffer, m_direction);
    AppendToBuffer(buffer, m_timestamp);
    return buffer;
}

PlayerInputMessage* PlayerInputMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 11 || buffer[0] != static_cast<uint8_t>(MessageType::PLAYER_INPUT)) {
        return nullptr;
    }

    uint8_t playerID = buffer[1];
    uint8_t inputFlags = buffer[2];

    size_t offset = 3;
    float direction = ReadFromBuffer<float>(buffer, offset);
    uint32_t timestamp = ReadFromBuffer<uint32_t>(buffer, offset);

    return new PlayerInputMessage(playerID, inputFlags, direction, timestamp);
}

// GameStateUpdateMessage
std::vector<uint8_t> GameStateUpdateMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    AppendToBuffer(buffer, m_sequenceNumber);

    uint16_t objectCount = static_cast<uint16_t>(m_objects.size());
    AppendToBuffer(buffer, objectCount);

    for (const auto& obj : m_objects) {
        AppendToBuffer(buffer, obj.id);
        buffer.push_back(obj.type);
        buffer.push_back(obj.ownerID);
        AppendToBuffer(buffer, obj.position.x);
        AppendToBuffer(buffer, obj.position.y);
        AppendToBuffer(buffer, obj.velocity.x);
        AppendToBuffer(buffer, obj.velocity.y);
        AppendToBuffer(buffer, obj.direction);
        AppendToBuffer(buffer, obj.scale.x);
        AppendToBuffer(buffer, obj.scale.y);
    }

    return buffer;
}

GameStateUpdateMessage* GameStateUpdateMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 7 || buffer[0] != static_cast<uint8_t>(MessageType::GAME_STATE_UPDATE)) {
        return nullptr;
    }

    size_t offset = 1;
    uint32_t sequenceNumber = ReadFromBuffer<uint32_t>(buffer, offset);
    uint16_t objectCount = ReadFromBuffer<uint16_t>(buffer, offset);

    GameStateUpdateMessage* message = new GameStateUpdateMessage();
    message->SetSequenceNumber(sequenceNumber);

    for (uint16_t i = 0; i < objectCount; ++i) {
        if (offset + 31 > buffer.size()) {  // Size of NetworkGameObject serialized
            delete message;
            return nullptr;
        }

        NetworkGameObject obj;
        obj.id = ReadFromBuffer<uint32_t>(buffer, offset);
        obj.type = buffer[offset++];
        obj.ownerID = buffer[offset++];
        obj.position.x = ReadFromBuffer<float>(buffer, offset);
        obj.position.y = ReadFromBuffer<float>(buffer, offset);
        obj.velocity.x = ReadFromBuffer<float>(buffer, offset);
        obj.velocity.y = ReadFromBuffer<float>(buffer, offset);
        obj.direction = ReadFromBuffer<float>(buffer, offset);
        obj.scale.x = ReadFromBuffer<float>(buffer, offset);
        obj.scale.y = ReadFromBuffer<float>(buffer, offset);

        message->AddObject(obj);
    }

    return message;
}

// ObjectCreateMessage
std::vector<uint8_t> ObjectCreateMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));

    AppendToBuffer(buffer, m_object.id);
    buffer.push_back(m_object.type);
    buffer.push_back(m_object.ownerID);
    AppendToBuffer(buffer, m_object.position.x);
    AppendToBuffer(buffer, m_object.position.y);
    AppendToBuffer(buffer, m_object.velocity.x);
    AppendToBuffer(buffer, m_object.velocity.y);
    AppendToBuffer(buffer, m_object.direction);
    AppendToBuffer(buffer, m_object.scale.x);
    AppendToBuffer(buffer, m_object.scale.y);

    return buffer;
}

ObjectCreateMessage* ObjectCreateMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 31 || buffer[0] != static_cast<uint8_t>(MessageType::OBJECT_CREATE)) {
        return nullptr;
    }

    size_t offset = 1;

    NetworkGameObject obj;
    obj.id = ReadFromBuffer<uint32_t>(buffer, offset);
    obj.type = buffer[offset++];
    obj.ownerID = buffer[offset++];
    obj.position.x = ReadFromBuffer<float>(buffer, offset);
    obj.position.y = ReadFromBuffer<float>(buffer, offset);
    obj.velocity.x = ReadFromBuffer<float>(buffer, offset);
    obj.velocity.y = ReadFromBuffer<float>(buffer, offset);
    obj.direction = ReadFromBuffer<float>(buffer, offset);
    obj.scale.x = ReadFromBuffer<float>(buffer, offset);
    obj.scale.y = ReadFromBuffer<float>(buffer, offset);

    return new ObjectCreateMessage(obj);
}

// ObjectDestroyMessage
std::vector<uint8_t> ObjectDestroyMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    AppendToBuffer(buffer, m_objectID);
    AppendToBuffer(buffer, m_sequenceNumber);
    return buffer;
}

ObjectDestroyMessage* ObjectDestroyMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 9 || buffer[0] != static_cast<uint8_t>(MessageType::OBJECT_DESTROY)) {
        return nullptr;
    }

    size_t offset = 1;
    uint32_t objectID = ReadFromBuffer<uint32_t>(buffer, offset);
    uint32_t sequenceNumber = ReadFromBuffer<uint32_t>(buffer, offset);

    return new ObjectDestroyMessage(objectID, sequenceNumber);
}

// ObjectDestroyAckMessage
std::vector<uint8_t> ObjectDestroyAckMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    AppendToBuffer(buffer, m_objectID);
    AppendToBuffer(buffer, m_sequenceNumber);
    buffer.push_back(m_playerID);
    return buffer;
}

ObjectDestroyAckMessage* ObjectDestroyAckMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 10 || buffer[0] != static_cast<uint8_t>(MessageType::OBJECT_DESTROY_ACK)) {
        return nullptr;
    }

    size_t offset = 1;
    uint32_t objectID = ReadFromBuffer<uint32_t>(buffer, offset);
    uint32_t sequenceNumber = ReadFromBuffer<uint32_t>(buffer, offset);
    uint8_t playerID = buffer[offset];

    return new ObjectDestroyAckMessage(objectID, sequenceNumber, playerID);
}

// ScoreUpdateMessage
std::vector<uint8_t> ScoreUpdateMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));

    for (int i = 0; i < 4; ++i) {
        AppendToBuffer(buffer, m_scores[i]);
    }

    return buffer;
}

ScoreUpdateMessage* ScoreUpdateMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 17 || buffer[0] != static_cast<uint8_t>(MessageType::SCORE_UPDATE)) {
        return nullptr;
    }

    ScoreUpdateMessage* message = new ScoreUpdateMessage();

    size_t offset = 1;
    for (int i = 0; i < 4; ++i) {
        uint32_t score = ReadFromBuffer<uint32_t>(buffer, offset);
        message->SetScore(i, score);
    }

    return message;
}

// GameEndMessage
std::vector<uint8_t> GameEndMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    buffer.push_back(m_winnerID);
    return buffer;
}

GameEndMessage* GameEndMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 2 || buffer[0] != static_cast<uint8_t>(MessageType::GAME_END)) {
        return nullptr;
    }

    uint8_t winnerID = buffer[1];

    return new GameEndMessage(winnerID);
}

// PingMessage
std::vector<uint8_t> PingMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    AppendToBuffer(buffer, m_timestamp);
    return buffer;
}

PingMessage* PingMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 9 || buffer[0] != static_cast<uint8_t>(MessageType::PING)) {
        return nullptr;
    }

    size_t offset = 1;
    uint64_t timestamp = ReadFromBuffer<uint64_t>(buffer, offset);

    return new PingMessage(timestamp);
}

// PongMessage
std::vector<uint8_t> PongMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    AppendToBuffer(buffer, m_timestamp);
    return buffer;
}

PongMessage* PongMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 9 || buffer[0] != static_cast<uint8_t>(MessageType::PONG)) {
        return nullptr;
    }

    size_t offset = 1;
    uint64_t timestamp = ReadFromBuffer<uint64_t>(buffer, offset);

    return new PongMessage(timestamp);
}

// DisconnectMessage
std::vector<uint8_t> DisconnectMessage::Serialize() const {
    std::vector<uint8_t> buffer;
    buffer.push_back(static_cast<uint8_t>(GetType()));
    buffer.push_back(m_playerID);
    return buffer;
}

DisconnectMessage* DisconnectMessage::Deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 2 || buffer[0] != static_cast<uint8_t>(MessageType::DISCONNECT)) {
        return nullptr;
    }

    uint8_t playerID = buffer[1];

    return new DisconnectMessage(playerID);
}