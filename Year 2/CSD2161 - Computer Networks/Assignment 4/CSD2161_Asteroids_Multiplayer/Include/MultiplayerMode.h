/******************************************************************************/
/*!
\file		MultiplayerMode.h
\author 	Tham Kang Ting
\par    	email: kangting.t\@digipen.edu
\date   	March 29, 2025
\brief		This header file declares the MultiplayerMode class.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef MULTIPLAYER_MODE_H_
#define MULTIPLAYER_MODE_H_

#include "Collision.h"
#include "GameMode.h"
#include "../Network/UDPSocket.h"
#include "../Network/NetworkMessage.h"
#include <map>
#include <string>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <vector>
#include "AEEngine.h"

// Forward declarations
struct MPGameObjInst;
struct MPGameObj;

// Structure to hold multiplayer player data
struct MultiplayerPlayer {
    uint8_t id;
    std::string name;
    bool isConnected;
    MPGameObjInst* shipInstance;
    uint32_t score;
    uint8_t lives;

    MultiplayerPlayer() :
        id(0), name(""), isConnected(false), shipInstance(nullptr), score(0), lives(3) {
    }
};

// Game object structure for multiplayer
struct MPGameObj {
    unsigned long type;
    AEGfxVertexList* pMesh;
    AEGfxTexture* pTex;
};

// Game object instance structure for multiplayer
struct MPGameObjInst {
    uint32_t networkID;
    MPGameObj* pObject;
    unsigned long flag;
    AEVec2 scale;
    AEVec2 posCurr;
    AEVec2 posPrev;
    AEVec2 velCurr;
    float dirCurr;
    AABB boundingBox;
    AEMtx33 transform;
    float cooldown;
    uint8_t ownerID;
    
};

// MultiplayerMode class
class MultiplayerMode : public GameMode {
public:
    MultiplayerMode(const std::string& serverAddress);
    ~MultiplayerMode();

    // Initialize the game mode
    bool Initialize() override;

    // Update the game mode
    void Update() override;

    // Draw the game mode
    void Draw() override;

    // Free resources
    void Free() override;

    // Unload resources
    void Unload() override;

private:
    // Network communication
    UDPSocket m_socket;
    NetworkEndpoint m_serverEndpoint;
    uint8_t m_playerID;
    bool m_isConnected;
    uint32_t m_currentSequence;
    uint32_t m_lastReceivedSequence;

    // Player data
    std::map<uint8_t, MultiplayerPlayer> m_players;

    // Game objects for multiplayer
    std::vector<MPGameObj> m_gameObjList;
    unsigned long m_gameObjNum;

    std::vector<MPGameObjInst> m_gameObjInstList;
    unsigned long m_gameObjInstNum;

    // Object tracking
    std::unordered_map<uint32_t, MPGameObjInst*> m_networkObjects;

    // Synchronization
    std::chrono::steady_clock::time_point m_lastUpdateTime;
    float m_networkDelay;
    bool m_gameStarted;
    uint32_t m_randomSeed;
    float m_gameRestartTimer;       // Timer for game restart countdown
    bool m_showRestartMessage;      // Flag to show restart message

    // Background rendering
    AEGfxVertexList* m_backgroundMesh;
    AEGfxTexture* m_background;
    AEMtx33 m_backgroundScale;

    // Game state
    unsigned long m_score;
    long m_shipLives;
    bool m_onValueChange;
    bool m_gameOver;

    // Audio
    AEAudio m_accelerationSound;
    AEAudio m_bulletSound;
    AEAudio m_explosion;
    AEAudioGroup m_soundEffect;

    // Event acknowledgment
    struct DestroyEvent {
        uint32_t objectID;
        uint32_t sequenceNumber;
        std::map<uint8_t, bool> playerAcks;
    };
    std::map<uint32_t, DestroyEvent> m_pendingDestroyEvents;

    // Initialize game objects
    void InitializeGameObjects();

    // Create game object instances
    MPGameObjInst* CreateGameObjInst(unsigned long type, AEVec2* scale, AEVec2* pPos, AEVec2* pVel, float dir, uint32_t networkID = 0, uint8_t ownerID = 0);

    // Destroy game object instance
    void DestroyGameObjInst(MPGameObjInst* pInst);

    // Process network messages
    void OnMessageReceived(const NetworkMessage* message, const NetworkEndpoint& sender);

    // Send player input to server
    void SendPlayerInput();

    // Handle different message types
    void HandleJoinResponse(const JoinResponseMessage* message);
    void HandleGameStart(const GameStartMessage* message);
    void HandleGameStateUpdate(const GameStateUpdateMessage* message);
    void HandleObjectCreate(const ObjectCreateMessage* message);
    void HandleObjectDestroy(const ObjectDestroyMessage* message);
    void HandleScoreUpdate(const ScoreUpdateMessage* message);
    void HandleGameEnd(const GameEndMessage* message);

    // Create or update a game object from network data
    MPGameObjInst* CreateOrUpdateGameObject(const NetworkGameObject& netObj);

    // Send acknowledgment for object destruction
    void SendObjectDestroyAck(uint32_t objectID, uint32_t sequenceNumber);

    // Measure and update network delay
    void UpdateNetworkDelay();

    // Check if all acknowledgments have been received for a destroy event
    bool AreAllAcksReceived(const DestroyEvent& event);

    // Render UI elements
    void DrawUI();

    void HandlePlayerConnectionUpdate(uint8_t playerID, bool connected);

    void HandleDisconnect(const DisconnectMessage* message);
};

#endif // MULTIPLAYER_MODE_H_