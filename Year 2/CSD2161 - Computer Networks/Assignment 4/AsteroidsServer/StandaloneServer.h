/******************************************************************************/
/*!
\file		StandaloneServer.h
\author 	Low Yue Jun
\par    	email: yuejun.low\@digipen.edu
\date   	March 29, 2025
\brief		This header file declares the StandaloneServer class.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef STANDALONE_SERVER_H_
#define STANDALONE_SERVER_H_

#include "UDPSocket.h"
#include "NetworkMessage.h"
#include <map>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>

// Forward declarations
struct ServerGameObj;
struct ServerPlayer;
struct AABB;

// Structure to represent a point in 2D space for the server
struct Vec2 {
    float x, y;

    Vec2() : x(0.0f), y(0.0f) {}
    Vec2(float _x, float _y) : x(_x), y(_y) {}

    // Convert to NetVector2
    NetVector2 ToNetVector2() const {
        return NetVector2(x, y);
    }

    // Convert from NetVector2
    static Vec2 FromNetVector2(const NetVector2& netVec) {
        return Vec2(netVec.x, netVec.y);
    }
};

// Structure to hold object data on the server
struct ServerGameObj {
    uint32_t id;
    uint8_t type;
    uint8_t ownerID;
    Vec2 position;
    Vec2 velocity;
    float direction;
    Vec2 scale;
    bool isActive;

    // Collision data
    AABB* boundingBox;

    ServerGameObj() :
        id(0), type(0), ownerID(0), direction(0.0f), isActive(false), boundingBox(nullptr) {
    }
};

// Structure to hold player data on the server
struct ServerPlayer {
    uint8_t id;
    std::string name;
    bool isConnected;
    NetworkEndpoint endpoint;
    uint32_t score;
    uint8_t lives;
    ServerGameObj* ship;
    std::chrono::steady_clock::time_point lastHeardTime;
    bool timeoutWarningIssued;  // Add this flag

    ServerPlayer() :
        id(0), name(""), isConnected(false), score(0), lives(3), ship(nullptr), timeoutWarningIssued(false) {
    }
};

// Structure to hold pending destroy events
struct ServerDestroyEvent {
    uint32_t objectID;
    uint32_t sequenceNumber;
    std::map<uint8_t, bool> playerAcks;

    ServerDestroyEvent() : objectID(0), sequenceNumber(0) {}
};

// StandaloneServer class
class StandaloneServer {
public:
    StandaloneServer();
    ~StandaloneServer();

    // Initialize the server
    bool Initialize();

    // Run the server (blocking)
    void Run();

    // Shutdown the server
    void Shutdown();

private:
    // Network communication
    UDPSocket m_socket;
    std::map<uint8_t, ServerPlayer> m_players;
    std::atomic<bool> m_isRunning;
    std::mutex m_gameStateMutex;  // Protects m_gameObjects and m_players from race conditions

    // Game state
    std::vector<ServerGameObj> m_gameObjects;
    uint32_t m_nextObjectID;
    uint32_t m_currentSequence;
    std::map<uint32_t, ServerDestroyEvent> m_pendingDestroyEvents;
    float m_gameRestartTimer; // Timer for automatic game restart after win

    // Game settings
    uint32_t m_randomSeed;
    std::mt19937 m_rng;
    bool m_gameStarted;
    std::chrono::steady_clock::time_point m_gameStartTime;

    // Game loop thread
    std::thread m_gameLoopThread;

    // Process network messages
    void OnMessageReceived(const NetworkMessage* message, const NetworkEndpoint& sender);

    // Game loop
    void GameLoopThreadFunc();

    // Handle different message types
    void HandleJoinRequest(const JoinRequestMessage* message, const NetworkEndpoint& sender);
    void HandlePlayerInput(const PlayerInputMessage* message, const NetworkEndpoint& sender);
    void HandleObjectDestroyAck(const ObjectDestroyAckMessage* message, const NetworkEndpoint& sender);
    void HandlePing(const PingMessage* message, const NetworkEndpoint& sender);
    void HandleDisconnect(const DisconnectMessage* message, const NetworkEndpoint& sender);

    // Send game state updates to all players
    void BroadcastGameState();

    // Initialize the game state
    void InitializeGame();

    // Update the game state
    void UpdateGame(float deltaTime);

    // Check for collisions
    void CheckCollisions();

    // Process input for a player
    void ProcessPlayerInput(ServerPlayer& player, uint8_t inputFlags, float direction);

    // Create game objects
    ServerGameObj* CreateGameObject(uint8_t type, uint8_t ownerID, const Vec2& position, const Vec2& velocity, float direction, const Vec2& scale);

    // Destroy a game object
    void DestroyGameObject(uint32_t objectID);

    // Convert server object to network object
    NetworkGameObject ConvertToNetworkObject(const ServerGameObj& obj);

    // Check if all acknowledgments have been received for a destroy event
    bool AreAllAcksReceived(const ServerDestroyEvent& event);

    // Check for timeouts
    void CheckTimeouts();

    // Find an active player by ID
    ServerPlayer* FindPlayer(uint8_t playerID);

    // Find a free player slot
    uint8_t FindFreePlayerSlot();

    // Check if the game should start
    bool ShouldStartGame();

    // Start the game
    void StartGame();

    // Check if game is over
    bool IsGameOver();

    // Handle game over
    void HandleGameOver();
};

#endif // STANDALONE_SERVER_H_