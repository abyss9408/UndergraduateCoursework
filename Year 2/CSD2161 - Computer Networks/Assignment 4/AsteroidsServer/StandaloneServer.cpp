/******************************************************************************/
/*!
\file		StandaloneServer.cpp
\author 	Bryan Ang Wei Ze (50%)
\co-author 	Low Yue Jun (50%)
\par    	email: bryanweize.ang\@digipen.edu
\par    	email: yuejun.low\@digipen.edu
\date   	March 29, 2025
\brief		This source file implements the StandaloneServer class.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#define _USE_MATH_DEFINES

#include "StandaloneServer.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <thread>

// Define AABB struct for collision detection (similar to the one in the game)
struct AABB {
    Vec2 min;
    Vec2 max;
};

// Constants
const unsigned short SERVER_PORT = 27015;
const float GAME_UPDATE_RATE = 0.02f; // 50 updates per second
const float STATE_BROADCAST_RATE = 0.0167f; // 60 broadcasts per second (reduced latency, matches typical client frame rate)
const float TIMEOUT_DURATION = 30.0f; // 30 seconds timeout
const float GAME_START_DELAY = 1.0f; // 1 second delay before game starts
const unsigned int MAX_PLAYERS = 4;
const float MAX_VELOCITY = 300.0f;
const float WINDOW_WIDTH = 800.0f;
const float WINDOW_HEIGHT = 600.0f;
const float BOUNDING_RECT_SIZE = 1.0f;
const uint32_t WIN_SCORE_THRESHOLD = 5000;
const float GAME_RESTART_DELAY = 5.0f; // 5 seconds delay before game restarts

// Game object types (copied from GameState_Asteroids.cpp)
enum TYPE {
    TYPE_SHIP = 0,
    TYPE_BULLET,
    TYPE_WALL,
    TYPE_ASTEROID,
    TYPE_EXPLOSION,
    TYPE_NUM
};

// Input flags
const uint8_t INPUT_UP = PlayerInputMessage::INPUT_UP;
const uint8_t INPUT_DOWN = PlayerInputMessage::INPUT_DOWN;
const uint8_t INPUT_LEFT = PlayerInputMessage::INPUT_LEFT;
const uint8_t INPUT_RIGHT = PlayerInputMessage::INPUT_RIGHT;
const uint8_t INPUT_FIRE = PlayerInputMessage::INPUT_FIRE;

// Ship constants
const float SHIP_ACCEL_FORWARD = 100.0f;
const float SHIP_ACCEL_BACKWARD = 100.0f;
const float SHIP_ROT_SPEED = 2.0f * M_PI;
const float SHIP_SCALE_X = 16.0f;
const float SHIP_SCALE_Y = 16.0f;
const float BULLET_SPEED = 400.0f;
const float BULLET_SCALE_X = 20.0f;
const float BULLET_SCALE_Y = 3.0f;
const float ASTEROID_MIN_SCALE_X = 10.0f;
const float ASTEROID_MAX_SCALE_X = 60.0f;
const float ASTEROID_MIN_SCALE_Y = 10.0f;
const float ASTEROID_MAX_SCALE_Y = 60.0f;
const float ASTEROID_MIN_VEL_X = -75.0f;
const float ASTEROID_MAX_VEL_X = 75.0f;
const float ASTEROID_MIN_VEL_Y = -100.0f;
const float ASTEROID_MAX_VEL_Y = 100.0f;

double Wrap(double value, double min, double max) {
    // Calculate the range
    double range = max - min;

    // Handle the case where range is zero to avoid division by zero
    if (range == 0) {
        return min;
    }

    // Adjust the value to be relative to min
    value = value - min;

    // Use modulo to wrap within the range
    // For negative values, we need to add the range to ensure proper wrapping
    value = value - (range * floor(value / range));

    // Re-adjust to the original range
    return value + min;
}

// Constructor
StandaloneServer::StandaloneServer()
    : m_nextObjectID(1), m_currentSequence(0), m_randomSeed(0), 
    m_gameStarted(false), m_isRunning(false), m_gameRestartTimer(0.0f) {
    // Add this line to prevent vector resizing
    m_gameObjects.reserve(500);
}

// Destructor
StandaloneServer::~StandaloneServer() {
    Shutdown();
}

// Initialize the server
bool StandaloneServer::Initialize() {
    // Initialize network socket
    if (!m_socket.Initialize()) {
        std::cerr << "Failed to initialize UDP socket" << std::endl;
        return false;
    }

    // Bind to server port
    if (!m_socket.Bind(SERVER_PORT)) {
        std::cerr << "Failed to bind UDP socket to port " << SERVER_PORT << std::endl;
        return false;
    }

    // Set message callback
    m_socket.SetMessageCallback([this](const NetworkMessage* message, const NetworkEndpoint& sender) {
        this->OnMessageReceived(message, sender);
    });

    // Start receiving messages
    if (!m_socket.StartReceiving()) {
        std::cerr << "Failed to start receiving messages" << std::endl;
        return false;
    }

    // Initialize the random number generator
    std::random_device rd;
    m_randomSeed = rd();
    m_rng.seed(m_randomSeed);

    std::cout << "Server initialized and listening on port " << SERVER_PORT << std::endl;
    std::cout << "Server IP address: " << m_socket.GetLocalAddress() << std::endl;

    return true;
}

// Run the server
void StandaloneServer::Run() {
    m_isRunning = true;

    // Start the game loop thread
    m_gameLoopThread = std::thread(&StandaloneServer::GameLoopThreadFunc, this);

}

// Shutdown the server
void StandaloneServer::Shutdown() {
    // Stop the game loop
    m_isRunning = false;

    // Wait for the thread to finish
    if (m_gameLoopThread.joinable()) {
        m_gameLoopThread.join();
    }

    // Stop receiving network messages
    m_socket.StopReceiving();

    // Clean up game objects
    m_gameObjects.clear();

    // Notify connected players
    for (auto& pair : m_players) {
        if (pair.second.isConnected) {
            DisconnectMessage disconnect(0); // 0 = server
            m_socket.SendMessage(&disconnect, pair.second.endpoint);
        }
    }

    std::cout << "Server shut down" << std::endl;
}

// Process network messages
void StandaloneServer::OnMessageReceived(const NetworkMessage* message, const NetworkEndpoint& sender) {
    // Handle different message types
    switch (message->GetType()) {
    case MessageType::JOIN_REQUEST:
        HandleJoinRequest(static_cast<const JoinRequestMessage*>(message), sender);
        break;

    case MessageType::PLAYER_INPUT:
        HandlePlayerInput(static_cast<const PlayerInputMessage*>(message), sender);
        break;

    case MessageType::OBJECT_DESTROY_ACK:
        HandleObjectDestroyAck(static_cast<const ObjectDestroyAckMessage*>(message), sender);
        break;

    case MessageType::PING:
        HandlePing(static_cast<const PingMessage*>(message), sender);
        break;

    case MessageType::DISCONNECT:
        HandleDisconnect(static_cast<const DisconnectMessage*>(message), sender);
        break;

    default:
        break;
    }
}

// Game loop
void StandaloneServer::GameLoopThreadFunc() {
    auto lastUpdateTime = std::chrono::steady_clock::now();
    float timeSinceLastBroadcast = 0.0f;

    while (m_isRunning) {
        auto currentTime = std::chrono::steady_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastUpdateTime).count();
        lastUpdateTime = currentTime;

        for (auto& pair : m_players) {
            if (pair.second.isConnected && pair.second.ship) {
                // Validate ship position at the start of each game loop
                ServerGameObj* ship = pair.second.ship;

                /*if (std::isnan(ship->position.x) || std::isinf(ship->position.x) ||
                    std::isnan(ship->position.y) || std::isinf(ship->position.y) ||
                    fabs(ship->position.x) > 10000.0f || fabs(ship->position.y) > 10000.0f) {

                    std::cout << "WARNING: Ship position corrupted in game loop for player "
                        << (int)pair.first << ": (" << ship->position.x << ","
                        << ship->position.y << ") - resetting" << std::endl;

                    ship->position.x = 0.0f;
                    ship->position.y = 0.0f;
                    ship->velocity.x = 0.0f;
                    ship->velocity.y = 0.0f;
                }*/
            }
        }

        // Check timeouts
        CheckTimeouts();

        // Check if the game should start
        if (!m_gameStarted && ShouldStartGame()) {
            StartGame();
        }

        // Update the game state
        if (m_gameStarted) {
            UpdateGame(deltaTime);

            // Broadcast game state at a fixed rate
            timeSinceLastBroadcast += deltaTime;
            if (timeSinceLastBroadcast >= STATE_BROADCAST_RATE) {
                BroadcastGameState();
                timeSinceLastBroadcast = 0.0f;
            }

            // Check if game is over
            if (m_gameStarted && IsGameOver()) {
                if (m_gameRestartTimer <= 0) {
                    // If game is already over and timer expired, restart
                    m_gameStarted = false;
                    m_gameObjects.clear();
                    m_pendingDestroyEvents.clear();

                    // Reset player scores and reset ships
                    for (auto& pair : m_players) {
                        if (pair.second.isConnected) {
                            pair.second.score = 0;
                            pair.second.lives = 3;
                            pair.second.ship = nullptr;
                        }
                    }

                    // Wait a bit before starting a new game
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                    // Check if we should start a new game
                    if (ShouldStartGame()) {
                        StartGame();
                    }
                }
                else {
                    // First time detected game over
                    if (m_gameRestartTimer == GAME_RESTART_DELAY) {
                        HandleGameOver();
                    }

                    // Count down restart timer
                    m_gameRestartTimer -= deltaTime;
                }
            }
        }

        // Sleep to limit CPU usage (reduced from 5ms to 0.5ms for better input responsiveness)
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

// Handle join request message
// In StandaloneServer.cpp, update the HandleJoinRequest method to:

void StandaloneServer::HandleJoinRequest(const JoinRequestMessage* message, const NetworkEndpoint& sender) {
    // Find a free player slot
    uint8_t playerID = FindFreePlayerSlot();

    // Check if we have room for another player
    bool accepted = (playerID < MAX_PLAYERS);

    // Send join response
    JoinResponseMessage response(playerID, accepted);
    m_socket.SendMessage(&response, sender);

    if (accepted) {
        // Add player to the list
        ServerPlayer player;
        player.id = playerID;
        player.name = message->GetPlayerName();
        player.isConnected = true;
        player.endpoint = sender;
        player.lastHeardTime = std::chrono::steady_clock::now();

        m_players[playerID] = player;

        std::cout << "Player " << static_cast<int>(playerID) << " (" << player.name << ") connected from "
            << sender.address << ":" << sender.port << std::endl;

        // If game is already in progress, send the current game state to the new player
        if (m_gameStarted) {
            // Send game start message with the current random seed
            GameStartMessage startMessage(m_randomSeed);
            m_socket.SendMessage(&startMessage, sender);

            // Create a game state update message for the new player
            GameStateUpdateMessage stateMessage;
            stateMessage.SetSequenceNumber(m_currentSequence++);

            // Add all active game objects to the message
            for (const auto& obj : m_gameObjects) {
                if (obj.isActive) {
                    stateMessage.AddObject(ConvertToNetworkObject(obj));
                }
            }

            // Send full game state to new player
            m_socket.SendMessage(&stateMessage, sender);

            // Create score update message
            ScoreUpdateMessage scoreMessage;
            for (const auto& pair : m_players) {
                scoreMessage.SetScore(pair.first, pair.second.score);
            }
            m_socket.SendMessage(&scoreMessage, sender);

            // Create player's ship
            Vec2 position;
            position.x = (playerID % 2) ? 100.0f : -100.0f;
            position.y = (playerID < 2) ? 100.0f : -100.0f;

            Vec2 velocity = { 0.0f, 0.0f };
            Vec2 scale = { 16.0f, 16.0f }; // SHIP_SCALE_X, SHIP_SCALE_Y

            std::cout << "Creating ship for player " << (int)playerID
                << " at position (" << position.x << "," << position.y << ")" << std::endl;

            ServerGameObj* ship = CreateGameObject(TYPE_SHIP, playerID, position, velocity, 0.0f, scale);
            if (ship) {
                //if (std::isnan(ship->position.x) || std::isinf(ship->position.x) ||
                //    std::isnan(ship->position.y) || std::isinf(ship->position.y) ||
                //    fabs(ship->position.x) > 10000.0f || fabs(ship->position.y) > 10000.0f) {

                //    std::cout << "WARNING: Ship position corrupted immediately after creation!" << std::endl;
                //    ship->position = position; // Reset to intended position
                //}

                player.ship = ship;
                m_players[playerID].ship = ship;

                // Send object create message for the new ship to all players
                ObjectCreateMessage createMessage(ConvertToNetworkObject(*ship));
                for (auto& pair : m_players) {
                    if (pair.second.isConnected) {
                        m_socket.SendMessage(&createMessage, pair.second.endpoint);
                    }
                }
            }
        }
        else if (ShouldStartGame()) {
            // If we now have enough players to start the game, do so
            StartGame();
        }
    }
    else {
        std::cout << "Rejected connection from " << sender.address << ":" << sender.port
            << " (server full)" << std::endl;
    }
}

// Handle player input message
void StandaloneServer::HandlePlayerInput(const PlayerInputMessage* message, const NetworkEndpoint& sender) {
    // Find the player
    ServerPlayer* player = FindPlayer(message->GetPlayerID());

    if (player && player->isConnected && player->endpoint.address == sender.address) {
        // Update last heard time
        player->lastHeardTime = std::chrono::steady_clock::now();

        // Reset the timeout warning flag when we receive input
        player->timeoutWarningIssued = false;

        //std::cout << "Player found\n";  // This is being logged correctly

        // Process input
        ProcessPlayerInput(*player, message->GetInputFlags(), message->GetDirection());
    }
}

// Handle object destroy acknowledgment message
void StandaloneServer::HandleObjectDestroyAck(const ObjectDestroyAckMessage* message, const NetworkEndpoint& sender) {
    // Find the player
    ServerPlayer* player = FindPlayer(message->GetPlayerID());

    if (player && player->isConnected && player->endpoint.address == sender.address) {
        // Update last heard time
        player->lastHeardTime = std::chrono::steady_clock::now();

        // Find the destroy event
        auto it = m_pendingDestroyEvents.find(message->GetObjectID());
        if (it != m_pendingDestroyEvents.end() && it->second.sequenceNumber == message->GetSequenceNumber()) {
            // Mark the acknowledgment as received
            it->second.playerAcks[player->id] = true;

            // Check if all acknowledgments have been received
            if (AreAllAcksReceived(it->second)) {
                // Remove the destroy event
                m_pendingDestroyEvents.erase(it);
            }
        }
    }
}

// Handle ping message
void StandaloneServer::HandlePing(const PingMessage* message, const NetworkEndpoint& sender) {
    // Find the player
    for (auto& pair : m_players) {
        if (pair.second.isConnected && pair.second.endpoint.address == sender.address) {
            // Update last heard time
            pair.second.lastHeardTime = std::chrono::steady_clock::now();

            // Reset timeout warning
            pair.second.timeoutWarningIssued = false;

            // Send pong response
            PongMessage pongMessage(message->GetTimestamp());
            m_socket.SendMessage(&pongMessage, sender);

            break;
        }
    }
}

// Handle disconnect message
void StandaloneServer::HandleDisconnect(const DisconnectMessage* message, const NetworkEndpoint& sender) {
    // Find the player
    ServerPlayer* player = FindPlayer(message->GetPlayerID());

    if (player && player->isConnected && player->endpoint.address == sender.address) {
        //std::cout << "Player " << static_cast<int>(player->id) << " disconnected" << std::endl;

        // Mark the player as disconnected
        player->isConnected = false;

        // Destroy the player's ship
        if (player->ship) {
            DestroyGameObject(player->ship->id);
            player->ship = nullptr;
        }

        // Check if the game should end
        if (m_gameStarted && IsGameOver()) {
            HandleGameOver();
        }
    }
}

// Broadcast game state to all players
void StandaloneServer::BroadcastGameState() {
    if (!m_gameStarted) {
        return;
    }

    // Create game state update message
    GameStateUpdateMessage stateMessage;
    stateMessage.SetSequenceNumber(m_currentSequence++);

    // Track number of objects by type
    int shipCount = 0, bulletCount = 0, asteroidCount = 0;

    // First collect all active bullets (highest priority)
    for (const auto& obj : m_gameObjects) {
        if (obj.isActive && obj.type == TYPE_BULLET) {
            // Double-check position values before adding to state update
            if (!std::isnan(obj.position.x) && !std::isnan(obj.position.y) &&
                !std::isinf(obj.position.x) && !std::isinf(obj.position.y)) {
                stateMessage.AddObject(ConvertToNetworkObject(obj));
                bulletCount++;
            }
            else {
                std::cout << "WARNING: Skipping bullet with invalid position: ID=" << obj.id << std::endl;
            }
        }
    }

    // Then add all active ships
    for (const auto& obj : m_gameObjects) {
        if (obj.isActive && obj.type == TYPE_SHIP) {
            stateMessage.AddObject(ConvertToNetworkObject(obj));
            shipCount++;
        }
    }

    // Finally add asteroids
    for (const auto& obj : m_gameObjects) {
        if (obj.isActive && obj.type == TYPE_ASTEROID) {
            stateMessage.AddObject(ConvertToNetworkObject(obj));
            asteroidCount++;
        }
    }

    // Only send if we have objects to update
    if (stateMessage.GetObjects().size() > 0) {
        // Send the message to all connected players
        for (auto& pair : m_players) {
            if (pair.second.isConnected) {
                m_socket.SendMessage(&stateMessage, pair.second.endpoint);
            }
        }
    }

    // Create and send score update message as usual
    ScoreUpdateMessage scoreMessage;
    for (const auto& pair : m_players) {
        scoreMessage.SetScore(pair.first, pair.second.score);
    }
    for (auto& pair : m_players) {
        if (pair.second.isConnected) {
            m_socket.SendMessage(&scoreMessage, pair.second.endpoint);
        }
    }
}

// Initialize the game state
void StandaloneServer::InitializeGame() {
    // Clear existing game objects
    m_gameObjects.clear();
    m_nextObjectID = 1;

    // Create ships for all connected players
    for (auto& pair : m_players) {
        if (pair.second.isConnected) {
            // Reset player stats
            pair.second.score = 0;
            pair.second.lives = 3;

            // Create player ship
            Vec2 position;
            position.x = (pair.first % 2) ? 100.0f : -100.0f;
            position.y = (pair.first < 2) ? 100.0f : -100.0f;

            Vec2 velocity = { 0.0f, 0.0f };
            Vec2 scale = { SHIP_SCALE_X, SHIP_SCALE_Y };

            ServerGameObj* ship = CreateGameObject(TYPE_SHIP, pair.first, position, velocity, 0.0f, scale);
            if (ship) {
                pair.second.ship = ship;
            }
        }
    }

    // Create initial asteroids
    for (int i = 0; i < 8; ++i) {
        Vec2 position;
        position.x = std::uniform_real_distribution<float>(-WINDOW_WIDTH / 2 + 50, WINDOW_WIDTH / 2 - 50)(m_rng);
        position.y = std::uniform_real_distribution<float>(-WINDOW_HEIGHT / 2 + 50, WINDOW_HEIGHT / 2 - 50)(m_rng);

        // Make sure asteroids don't spawn too close to ships
        bool tooClose = false;
        for (const auto& pair : m_players) {
            if (pair.second.ship) {
                float dx = position.x - pair.second.ship->position.x;
                float dy = position.y - pair.second.ship->position.y;
                float distSquared = dx * dx + dy * dy;

                if (distSquared < 150.0f * 150.0f) {
                    tooClose = true;
                    break;
                }
            }
        }

        if (tooClose) {
            --i; // Try again
            continue;
        }

        Vec2 velocity;
        velocity.x = std::uniform_real_distribution<float>(ASTEROID_MIN_VEL_X, ASTEROID_MAX_VEL_X)(m_rng);
        velocity.y = std::uniform_real_distribution<float>(ASTEROID_MIN_VEL_Y, ASTEROID_MAX_VEL_Y)(m_rng);

        Vec2 scale;
        scale.x = std::uniform_real_distribution<float>(ASTEROID_MIN_SCALE_X, ASTEROID_MAX_SCALE_X)(m_rng);
        scale.y = std::uniform_real_distribution<float>(ASTEROID_MIN_SCALE_Y, ASTEROID_MAX_SCALE_Y)(m_rng);

        CreateGameObject(TYPE_ASTEROID, 0, position, velocity, 0.0f, scale);
    }
}

// Update the game state
void StandaloneServer::UpdateGame(float deltaTime) {
    // Lock to prevent race condition with input processing thread
    std::lock_guard<std::mutex> lock(m_gameStateMutex);

    // Update positions of all game objects
    for (auto& obj : m_gameObjects) {
        if (obj.isActive) {
            // VALIDATION: Check position BEFORE update
            //if (obj.type == TYPE_SHIP && (std::isnan(obj.position.x) || std::isinf(obj.position.x) ||
            //    std::isnan(obj.position.y) || std::isinf(obj.position.y) ||
            //    fabs(obj.position.x) > 10000.0f || fabs(obj.position.y) > 10000.0f)) {
            //    std::cout << "WARNING: Ship with ID " << obj.id << " has invalid position: ("
            //        << obj.position.x << "," << obj.position.y
            //        << ") - resetting position" << std::endl;

            //    // Reset to center of screen
            //    obj.position.x = 0.0f;
            //    obj.position.y = 0.0f;
            //    obj.velocity.x = 0.0f;
            //    obj.velocity.y = 0.0f;
            //}

            // Save previous position
            Vec2 prevPos = obj.position;

            // Check for invalid velocity values
            /*if (std::isnan(obj.velocity.x) || std::isinf(obj.velocity.x) ||
                std::isnan(obj.velocity.y) || std::isinf(obj.velocity.y)) {
                std::cout << "WARNING: Ship velocity is invalid: ("
                    << obj.velocity.x << "," << obj.velocity.y << ")" << std::endl;
                obj.velocity.x = 0.0f;
                obj.velocity.y = 0.0f;
            }*/

            // Check for unusually large deltaTime
            //if (deltaTime > 1.0f) {
            //    std::cout << "WARNING: Unusually large deltaTime: " << deltaTime << std::endl;
            //    deltaTime = 0.02f; // Safe fallback value
            //}

            // Use safer position update with validation
            Vec2 newPos;
            newPos.x = obj.position.x + obj.velocity.x * deltaTime;
            newPos.y = obj.position.y + obj.velocity.y * deltaTime;

            // Check if position update would cause corruption
            //if (std::isnan(newPos.x) || std::isinf(newPos.x) ||
            //    std::isnan(newPos.y) || std::isinf(newPos.y) ||
            //    fabs(newPos.x) > 10000.0f || fabs(newPos.y) > 10000.0f) {

            //    std::cout << "WARNING: Position update would cause corruption: from ("
            //        << obj.position.x << "," << obj.position.y << ") to ("
            //        << newPos.x << "," << newPos.y << "), velocity: ("
            //        << obj.velocity.x << "," << obj.velocity.y
            //        << "), deltaTime: " << deltaTime << std::endl;

            //    // Don't update position, just reset velocity
            //    obj.velocity.x = 0.0f;
            //    obj.velocity.y = 0.0f;
            //}
            //else {
            //    // Safe to update
            //    
            //}

            obj.position = newPos;

            // Wrap position around screen edges for ships and asteroids
            if (obj.type == TYPE_SHIP || obj.type == TYPE_ASTEROID) {
                // Add debug to track wrapping
                bool wrapped = false;
                Vec2 beforeWrap = obj.position;

                // Wrap X
                if (obj.position.x < -WINDOW_WIDTH / 2 - obj.scale.x) {
                    obj.position.x = WINDOW_WIDTH / 2 + obj.scale.x;
                    wrapped = true;
                }
                else if (obj.position.x > WINDOW_WIDTH / 2 + obj.scale.x) {
                    obj.position.x = -WINDOW_WIDTH / 2 - obj.scale.x;
                    wrapped = true;
                }

                // Wrap Y
                if (obj.position.y < -WINDOW_HEIGHT / 2 - obj.scale.y) {
                    obj.position.y = WINDOW_HEIGHT / 2 + obj.scale.y;
                    wrapped = true;
                }
                else if (obj.position.y > WINDOW_HEIGHT / 2 + obj.scale.y) {
                    obj.position.y = -WINDOW_HEIGHT / 2 - obj.scale.y;
                    wrapped = true;
                }

                // Check if wrapping caused corruption
                //if (wrapped) {
                //    if (std::isnan(obj.position.x) || std::isinf(obj.position.x) ||
                //        std::isnan(obj.position.y) || std::isinf(obj.position.y) ||
                //        fabs(obj.position.x) > 10000.0f || fabs(obj.position.y) > 10000.0f) {

                //        std::cout << "WARNING: Position wrapping caused corruption: before ("
                //            << beforeWrap.x << "," << beforeWrap.y << "), after ("
                //            << obj.position.x << "," << obj.position.y << ")" << std::endl;

                //        // Reset to center
                //        obj.position.x = 0.0f;
                //        obj.position.y = 0.0f;
                //    }
                //}
            }

            // VALIDATION: Check position AFTER update
            if (obj.type == TYPE_SHIP && (std::isnan(obj.position.x) || std::isinf(obj.position.x) ||
                std::isnan(obj.position.y) || std::isinf(obj.position.y) ||
                fabs(obj.position.x) > 10000.0f || fabs(obj.position.y) > 10000.0f)) {
                std::cout << "WARNING: Ship position update caused invalid position: ("
                    << obj.position.x << "," << obj.position.y
                    << ") from velocity: (" << obj.velocity.x << "," << obj.velocity.y
                    << ") with deltaTime: " << deltaTime << std::endl;

                // Revert to previous position and stop movement
                obj.position = prevPos;
                obj.velocity.x = 0.0f;
                obj.velocity.y = 0.0f;
            }

            // Check if bullets are out of bounds
            if (obj.type == TYPE_BULLET) {
                // Check for invalid position
                if (std::isnan(obj.position.x) || std::isinf(obj.position.x) ||
                    std::isnan(obj.position.y) || std::isinf(obj.position.y) ||
                    fabs(obj.position.x) > 10000.0f || fabs(obj.position.y) > 10000.0f) {
                    obj.isActive = false;
                    std::cout << "WARNING: Deactivating bullet with invalid position" << std::endl;
                }
                // Normal out-of-bounds check
                else if (obj.position.x < -WINDOW_WIDTH / 2 || obj.position.x > WINDOW_WIDTH / 2 ||
                    obj.position.y < -WINDOW_HEIGHT / 2 || obj.position.y > WINDOW_HEIGHT / 2) {
                    obj.isActive = false;
                }
            }

            // Update bounding box
            if (obj.boundingBox) {
                obj.boundingBox->min.x = (-BOUNDING_RECT_SIZE / 2.0f) * obj.scale.x + obj.position.x;
                obj.boundingBox->min.y = (-BOUNDING_RECT_SIZE / 2.0f) * obj.scale.y + obj.position.y;
                obj.boundingBox->max.x = (BOUNDING_RECT_SIZE / 2.0f) * obj.scale.x + obj.position.x;
                obj.boundingBox->max.y = (BOUNDING_RECT_SIZE / 2.0f) * obj.scale.y + obj.position.y;
            }
        }
    }

    // Check for collisions
    CheckCollisions();
}

// Check for collisions between game objects
void StandaloneServer::CheckCollisions() {
    // Check for asteroid-ship and asteroid-bullet collisions
    for (auto& asteroid : m_gameObjects) {
        if (!asteroid.isActive || asteroid.type != TYPE_ASTEROID) {
            continue;
        }

        for (auto& obj : m_gameObjects) {
            if (!obj.isActive || obj.type == TYPE_ASTEROID) {
                continue;
            }

            // Simple AABB collision check
            if (asteroid.boundingBox && obj.boundingBox) {
                bool collision = !(asteroid.boundingBox->max.x < obj.boundingBox->min.x ||
                    asteroid.boundingBox->min.x > obj.boundingBox->max.x ||
                    asteroid.boundingBox->max.y < obj.boundingBox->min.y ||
                    asteroid.boundingBox->min.y > obj.boundingBox->max.y);

                if (collision) {
                    if (obj.type == TYPE_SHIP) {
                        // Ship-asteroid collision
                        ServerPlayer* player = FindPlayer(obj.ownerID);
                        if (player) {
                            // Reduce player lives
                            if (player->lives > 0) {
                                player->lives--;
                            }

                            // Reset ship position
                            obj.position.x = 0.0f;
                            obj.position.y = 0.0f;
                            obj.velocity.x = 0.0f;
                            obj.velocity.y = 0.0f;
                            obj.direction = 0.0f;

                            // Create explosion effect
                            Vec2 explosionPos = asteroid.position;
                            Vec2 explosionVel = { 0.0f, 0.0f };
                            Vec2 explosionScale = { 50.0f, 50.0f };
                            CreateGameObject(TYPE_EXPLOSION, 0, explosionPos, explosionVel, 0.0f, explosionScale);

                            // Destroy the asteroid
                            DestroyGameObject(asteroid.id);

                            // Create a new asteroid
                            Vec2 newPos;
                            do {
                                newPos.x = std::uniform_real_distribution<float>(-WINDOW_WIDTH / 2 + 50, WINDOW_WIDTH / 2 - 50)(m_rng);
                                newPos.y = std::uniform_real_distribution<float>(-WINDOW_HEIGHT / 2 + 50, WINDOW_HEIGHT / 2 - 50)(m_rng);
                            } while (std::abs(newPos.x) < 100.0f && std::abs(newPos.y) < 100.0f); // Keep away from center

                            Vec2 newVel;
                            newVel.x = std::uniform_real_distribution<float>(ASTEROID_MIN_VEL_X, ASTEROID_MAX_VEL_X)(m_rng);
                            newVel.y = std::uniform_real_distribution<float>(ASTEROID_MIN_VEL_Y, ASTEROID_MAX_VEL_Y)(m_rng);

                            Vec2 newScale;
                            newScale.x = std::uniform_real_distribution<float>(ASTEROID_MIN_SCALE_X, ASTEROID_MAX_SCALE_X)(m_rng);
                            newScale.y = std::uniform_real_distribution<float>(ASTEROID_MIN_SCALE_Y, ASTEROID_MAX_SCALE_Y)(m_rng);

                            CreateGameObject(TYPE_ASTEROID, 0, newPos, newVel, 0.0f, newScale);

                            break;
                        }
                    }
                    else if (obj.type == TYPE_BULLET) {
                        // Bullet-asteroid collision
                        ServerPlayer* player = FindPlayer(obj.ownerID);
                        if (player) {
                            // Increase player score
                            player->score += 100;

                            // Create explosion effect
                            Vec2 explosionPos = asteroid.position;
                            Vec2 explosionVel = { 0.0f, 0.0f };
                            Vec2 explosionScale = { 50.0f, 50.0f };
                            CreateGameObject(TYPE_EXPLOSION, 0, explosionPos, explosionVel, 0.0f, explosionScale);

                            // Destroy the bullet and asteroid
                            DestroyGameObject(obj.id);
                            DestroyGameObject(asteroid.id);

                            // Create 1 or 2 new asteroids
                            int numNewAsteroids = 1 + (std::uniform_int_distribution<int>(0, 1)(m_rng));

                            for (int i = 0; i < numNewAsteroids; ++i) {
                                Vec2 newPos;
                                do {
                                    newPos.x = std::uniform_real_distribution<float>(-WINDOW_WIDTH / 2 + 50, WINDOW_WIDTH / 2 - 50)(m_rng);
                                    newPos.y = std::uniform_real_distribution<float>(-WINDOW_HEIGHT / 2 + 50, WINDOW_HEIGHT / 2 - 50)(m_rng);
                                } while (std::abs(newPos.x) < 100.0f && std::abs(newPos.y) < 100.0f); // Keep away from center

                                Vec2 newVel;
                                newVel.x = std::uniform_real_distribution<float>(ASTEROID_MIN_VEL_X, ASTEROID_MAX_VEL_X)(m_rng);
                                newVel.y = std::uniform_real_distribution<float>(ASTEROID_MIN_VEL_Y, ASTEROID_MAX_VEL_Y)(m_rng);

                                Vec2 newScale;
                                newScale.x = std::uniform_real_distribution<float>(ASTEROID_MIN_SCALE_X, ASTEROID_MAX_SCALE_X)(m_rng);
                                newScale.y = std::uniform_real_distribution<float>(ASTEROID_MIN_SCALE_Y, ASTEROID_MAX_SCALE_Y)(m_rng);

                                CreateGameObject(TYPE_ASTEROID, 0, newPos, newVel, 0.0f, newScale);
                            }

                            break;
                        }
                    }
                }
            }
        }
    }
}

// Process input for a player
void StandaloneServer::ProcessPlayerInput(ServerPlayer& player, uint8_t inputFlags, float direction) {
    // Lock to prevent race condition with game loop thread
    std::lock_guard<std::mutex> lock(m_gameStateMutex);

    if (player.ship == nullptr || !player.ship->isActive) {
        return;
    }

    // VALIDATION: Add check for ship position before processing input
    //if (std::isnan(player.ship->position.x) || std::isinf(player.ship->position.x) ||
    //    std::isnan(player.ship->position.y) || std::isinf(player.ship->position.y) ||
    //    fabs(player.ship->position.x) > 10000.0f || fabs(player.ship->position.y) > 10000.0f) {
    //    std::cout << "WARNING: Ship position corrupted before processing input: ("
    //        << player.ship->position.x << "," << player.ship->position.y
    //        << ") - resetting position" << std::endl;

    //    // Reset to center of screen
    //    player.ship->position.x = 0.0f;
    //    player.ship->position.y = 0.0f;
    //    player.ship->velocity.x = 0.0f;
    //    player.ship->velocity.y = 0.0f;
    //}

    // Update ship direction
    player.ship->direction = direction;

    // Apply rotation
    if (inputFlags & INPUT_LEFT) {
        player.ship->direction += SHIP_ROT_SPEED * GAME_UPDATE_RATE;
        player.ship->direction = static_cast<float>(Wrap(player.ship->direction, -M_PI, M_PI));
    }

    if (inputFlags & INPUT_RIGHT) {
        player.ship->direction -= SHIP_ROT_SPEED * GAME_UPDATE_RATE;
        player.ship->direction = static_cast<float>(Wrap(player.ship->direction, -M_PI, M_PI));
    }

    // Apply acceleration
    if (inputFlags & INPUT_UP) {
        float velNewX = cosf(direction) * SHIP_ACCEL_FORWARD * GAME_UPDATE_RATE + player.ship->velocity.x; // Assuming 50fps fixed timestep
        float velNewY = sinf(direction) * SHIP_ACCEL_FORWARD * GAME_UPDATE_RATE + player.ship->velocity.y;

        player.ship->velocity.x = velNewX * 0.99f;
        player.ship->velocity.y = velNewY * 0.99f;
    }

    if (inputFlags & INPUT_DOWN) {
        float velNewX = -cosf(direction) * SHIP_ACCEL_BACKWARD * GAME_UPDATE_RATE + player.ship->velocity.x;
        float velNewY = -sinf(direction) * SHIP_ACCEL_BACKWARD * GAME_UPDATE_RATE + player.ship->velocity.y;

        player.ship->velocity.x = velNewX * 0.99f;
        player.ship->velocity.y = velNewY * 0.99f;
    }

    // Limit velocity
    /*float velMagnitude = sqrtf(player.ship->velocity.x * player.ship->velocity.x +
        player.ship->velocity.y * player.ship->velocity.y);

    if (velMagnitude > MAX_VELOCITY) {
        float scale = MAX_VELOCITY / velMagnitude;
        player.ship->velocity.x *= scale;
        player.ship->velocity.y *= scale;
    }*/

    // Handle firing
    if (inputFlags & INPUT_FIRE) {
        // Get the position of the ship - PREVENT OVERFLOW
        Vec2 bulletPos = player.ship->position;

        // Calculate the bullet's initial position at the tip of the ship
        // FIXED: Use proper scaling and prevent floating point errors
        bulletPos.x += cosf(player.ship->direction) * (SHIP_SCALE_X * 0.5f);
        bulletPos.y += sinf(player.ship->direction) * (SHIP_SCALE_Y * 0.5f);

        // Debug - print the exact position values before sending
        //std::cout << "Creating bullet at position (" << bulletPos.x << "," << bulletPos.y << ")" << std::endl;

        // Set bullet velocity based on ship's direction and add to ship velocity
        Vec2 bulletVel;
        bulletVel.x = player.ship->velocity.x + cosf(player.ship->direction) * BULLET_SPEED;
        bulletVel.y = player.ship->velocity.y + sinf(player.ship->direction) * BULLET_SPEED;

        Vec2 bulletScale = { BULLET_SCALE_X, BULLET_SCALE_Y };

        // Create the bullet object with proper orientation
        ServerGameObj* bullet = CreateGameObject(TYPE_BULLET, player.id, bulletPos, bulletVel, player.ship->direction, bulletScale);

        if (bullet) {
            // VERIFY VALUES before sending
            // Check for unreasonable position values that might cause synchronization issues
            bool positionError = false;
            if (std::isnan(bullet->position.x) || std::isnan(bullet->position.y) ||
                std::isinf(bullet->position.x) || std::isinf(bullet->position.y) ||
                fabs(bullet->position.x) > 10000.0f || fabs(bullet->position.y) > 10000.0f) {
                // Position values are unreasonable - fix them
                bullet->position = player.ship->position;
                positionError = true;
                std::cout << "WARNING: Fixed invalid bullet position" << std::endl;
            }

            // Send individual object create message for the bullet to all players IMMEDIATELY
            ObjectCreateMessage createMessage(ConvertToNetworkObject(*bullet));

            for (auto& pair : m_players) {
                if (pair.second.isConnected) {
                    m_socket.SendMessage(&createMessage, pair.second.endpoint);
                }
            }

            /*std::cout << "Player " << static_cast<int>(player.id)
                << " fired bullet ID " << bullet->id
                << " at position (" << bullet->position.x << "," << bullet->position.y << ")"
                << " with velocity (" << bullet->velocity.x << "," << bullet->velocity.y << ")"
                << (positionError ? " (corrected position)" : "")
                << std::endl;*/
        }
    }
}

// Create a game object
ServerGameObj* StandaloneServer::CreateGameObject(uint8_t type, uint8_t ownerID, const Vec2& position,
    const Vec2& velocity, float direction, const Vec2& scale) {

    // Validate position before creating the object
    Vec2 validatedPosition = position;
    Vec2 validatedVelocity = velocity;

    // Check for invalid values
    /*if (std::isnan(position.x) || std::isinf(position.x) || fabs(position.x) > 10000.0f) {
        validatedPosition.x = 0.0f;
        std::cout << "WARNING: CreateGameObject received invalid X position: " << position.x << std::endl;
    }

    if (std::isnan(position.y) || std::isinf(position.y) || fabs(position.y) > 10000.0f) {
        validatedPosition.y = 0.0f;
        std::cout << "WARNING: CreateGameObject received invalid Y position: " << position.y << std::endl;
    }

    if (std::isnan(velocity.x) || std::isinf(velocity.x) || fabs(velocity.x) > 10000.0f) {
        validatedVelocity.x = 0.0f;
        std::cout << "WARNING: CreateGameObject received invalid X velocity: " << velocity.x << std::endl;
    }

    if (std::isnan(velocity.y) || std::isinf(velocity.y) || fabs(velocity.y) > 10000.0f) {
        validatedVelocity.y = 0.0f;
        std::cout << "WARNING: CreateGameObject received invalid Y velocity: " << velocity.y << std::endl;
    }*/

    // Find an inactive object or create a new one
    ServerGameObj* newObj = nullptr;

    for (auto& obj : m_gameObjects) {
        if (!obj.isActive) {
            newObj = &obj;
            break;
        }
    }

    if (!newObj) {
        m_gameObjects.push_back(ServerGameObj());
        newObj = &m_gameObjects.back();
        newObj->boundingBox = new AABB();
    }

    // Initialize the object
    newObj->id = m_nextObjectID++;
    newObj->type = type;
    newObj->ownerID = ownerID;
    newObj->position = position;
    newObj->velocity = velocity;
    newObj->direction = direction;
    newObj->scale = scale;
    newObj->isActive = true;

    // Initialize the bounding box
    if (newObj->boundingBox) {
        newObj->boundingBox->min.x = (-BOUNDING_RECT_SIZE / 2.0f) * scale.x + position.x;
        newObj->boundingBox->min.y = (-BOUNDING_RECT_SIZE / 2.0f) * scale.y + position.y;
        newObj->boundingBox->max.x = (BOUNDING_RECT_SIZE / 2.0f) * scale.x + position.x;
        newObj->boundingBox->max.y = (BOUNDING_RECT_SIZE / 2.0f) * scale.y + position.y;
    }

    // For explosions, send object create message to all clients
    if (type == TYPE_EXPLOSION) {
        ObjectCreateMessage createMessage(ConvertToNetworkObject(*newObj));

        for (auto& pair : m_players) {
            if (pair.second.isConnected) {
                m_socket.SendMessage(&createMessage, pair.second.endpoint);
            }
        }
    }

    return newObj;
}

// Destroy a game object
void StandaloneServer::DestroyGameObject(uint32_t objectID) {
    // Find the object
    for (auto& obj : m_gameObjects) {
        if (obj.isActive && obj.id == objectID) {
            // Mark the object as inactive
            obj.isActive = false;

            // For important objects like asteroids, we need to wait for acknowledgments
            if (obj.type == TYPE_ASTEROID) {
                // Create a destroy event
                ServerDestroyEvent event;
                event.objectID = objectID;
                event.sequenceNumber = m_currentSequence++;

                // Initialize acknowledgments for all connected players
                for (const auto& pair : m_players) {
                    if (pair.second.isConnected) {
                        event.playerAcks[pair.first] = false;
                    }
                }

                // Add to pending destroy events
                m_pendingDestroyEvents[objectID] = event;

                // Send destroy message to all clients
                ObjectDestroyMessage destroyMessage(objectID, event.sequenceNumber);

                for (auto& pair : m_players) {
                    if (pair.second.isConnected) {
                        m_socket.SendMessage(&destroyMessage, pair.second.endpoint);
                    }
                }
            }

            break;
        }
    }
}

// Convert server object to network object
NetworkGameObject StandaloneServer::ConvertToNetworkObject(const ServerGameObj& obj) {
    NetworkGameObject netObj;

    netObj.id = obj.id;
    netObj.type = obj.type;
    netObj.ownerID = obj.ownerID;

    // Convert from Vec2 to NetVector2
    netObj.position.x = obj.position.x;
    netObj.position.y = obj.position.y;

    netObj.velocity.x = obj.velocity.x;
    netObj.velocity.y = obj.velocity.y;

    netObj.direction = obj.direction;
    netObj.scale.x = obj.scale.x;
    netObj.scale.y = obj.scale.y;

    return netObj;
}

// Check if all acknowledgments have been received for a destroy event
bool StandaloneServer::AreAllAcksReceived(const ServerDestroyEvent& event) {
    for (const auto& pair : event.playerAcks) {
        if (!pair.second) {
            return false;
        }
    }

    return true;
}

// Check for timeouts
void StandaloneServer::CheckTimeouts() {
    // Only check for timeouts if the game has already started
    if (!m_gameStarted) {
        // Skip timeout checks during the waiting phase
        return;
    }

    auto currentTime = std::chrono::steady_clock::now();

    for (auto& pair : m_players) {
        if (pair.second.isConnected) {
            float timeSinceLastHeard = std::chrono::duration<float>(currentTime - pair.second.lastHeardTime).count();

            // First warning at 15 seconds of inactivity
            if (timeSinceLastHeard > 15.0f && !pair.second.timeoutWarningIssued) {
                std::cout << "WARNING: Player " << static_cast<int>(pair.first) << " inactive for 15 seconds" << std::endl;
                pair.second.timeoutWarningIssued = true;

                // Send a ping to try to wake up the client
                PingMessage ping(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count()));
                m_socket.SendMessage(&ping, pair.second.endpoint);
            }

            // Only disconnect after the full timeout period
            if (timeSinceLastHeard > TIMEOUT_DURATION) {
                std::cout << "Player " << static_cast<int>(pair.first) << " timed out after "
                    << TIMEOUT_DURATION << " seconds of inactivity" << std::endl;

                // Mark the player as disconnected
                pair.second.isConnected = false;

                // Destroy the player's ship
                if (pair.second.ship) {
                    DestroyGameObject(pair.second.ship->id);
                    pair.second.ship = nullptr;
                }

                // Notify other players about the disconnection
                DisconnectMessage disconnectMsg(pair.first);
                for (auto& otherPair : m_players) {
                    if (otherPair.first != pair.first && otherPair.second.isConnected) {
                        m_socket.SendMessage(&disconnectMsg, otherPair.second.endpoint);
                    }
                }
            }
        }
    }
}

// Find an active player by ID
ServerPlayer* StandaloneServer::FindPlayer(uint8_t playerID) {
    auto it = m_players.find(playerID);

    if (it != m_players.end() && it->second.isConnected) {
        // Check ship position right when the player is found
        //if (it->second.ship) {
        //    if (std::isnan(it->second.ship->position.x) || std::isinf(it->second.ship->position.x) ||
        //        std::isnan(it->second.ship->position.y) || std::isinf(it->second.ship->position.y) ||
        //        fabs(it->second.ship->position.x) > 10000.0f || fabs(it->second.ship->position.y) > 10000.0f) {

        //        std::cout << "WARNING: Ship position corrupt in FindPlayer: ("
        //            << it->second.ship->position.x << "," << it->second.ship->position.y
        //            << ") - player ID " << (int)playerID << std::endl;

        //        // Reset ship position
        //        it->second.ship->position.x = 0.0f;
        //        it->second.ship->position.y = 0.0f;
        //        it->second.ship->velocity.x = 0.0f;
        //        it->second.ship->velocity.y = 0.0f;
        //    }
        //}
        return &it->second;
    }

    return nullptr;
}

// Find a free player slot
uint8_t StandaloneServer::FindFreePlayerSlot() {
    for (uint8_t i = 0; i < MAX_PLAYERS; ++i) {
        auto it = m_players.find(i);

        if (it == m_players.end() || !it->second.isConnected) {
            return i;
        }
    }

    return MAX_PLAYERS; // No free slots
}

// Check if the game should start
bool StandaloneServer::ShouldStartGame() {
    // If a game is already in progress, don't start a new one
    if (m_gameStarted) {
        return false;
    }

    // Count connected players
    int connectedPlayers = 0;
    
    for (const auto& pair : m_players) {
        if (pair.second.isConnected) {
            connectedPlayers++;
        }
    }

    // Start the game if we have exactly 4 players
    return connectedPlayers == 2;
}

// Start the game
void StandaloneServer::StartGame() {
    m_gameStarted = true;
    m_gameStartTime = std::chrono::steady_clock::now();

    // Initialize the game state
    InitializeGame();

    // Send game start message to all clients
    GameStartMessage startMessage(m_randomSeed);

    for (auto& pair : m_players) {
        if (pair.second.isConnected) {
            m_socket.SendMessage(&startMessage, pair.second.endpoint);
        }
    }

    std::cout << "Game started with " << m_players.size() << " players" << std::endl;
}

// Check if game is over
bool StandaloneServer::IsGameOver() {
    // Check if any player has reached the score threshold
    for (const auto& pair : m_players) {
        if (pair.second.isConnected && pair.second.score >= WIN_SCORE_THRESHOLD) {
            return true;
        }
    }

    // Count active players with lives remaining
    int activePlayers = 0;
    for (const auto& pair : m_players) {
        if (pair.second.isConnected && pair.second.lives > 0) {
            activePlayers++;
        }
    }

    // Game is over if only one or zero players are left
    return activePlayers <= 1;
}

// Handle game over
void StandaloneServer::HandleGameOver() {
    // Find the winner (player with the highest score among those still alive or first to reach threshold)
    uint8_t winnerID = 0;
    uint32_t highestScore = 0;

    for (const auto& pair : m_players) {
        if (pair.second.isConnected && pair.second.score > highestScore) {
            winnerID = pair.first;
            highestScore = pair.second.score;
        }
    }

    // Send game end message to all clients
    GameEndMessage endMessage(winnerID);

    for (auto& pair : m_players) {
        if (pair.second.isConnected) {
            m_socket.SendMessage(&endMessage, pair.second.endpoint);
        }
    }

    std::cout << "Game over! Player " << static_cast<int>(winnerID) << " wins with score " << highestScore << std::endl;

    // Don't reset game state yet - wait for the timer
    m_gameRestartTimer = GAME_RESTART_DELAY;

    // Keep game in "ended" state but don't reset everything yet
    // Objects should freeze but network connections maintained
}