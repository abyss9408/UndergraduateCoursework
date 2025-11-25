/******************************************************************************/
/*!
\file		MultiplayerMode.cpp
\author 	Tham Kang Ting
\par    	email: kangting.t\@digipen.edu
\date   	March 29, 2025
\brief		This source file implements the MultiplayerMode class.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "MultiplayerMode.h"
#include "main.h"
#include <iostream>
#include <random>
#include <chrono>
#include <set>

// Constants for game objects
const unsigned int GAME_OBJ_NUM_MAX = 32;
const unsigned int GAME_OBJ_INST_NUM_MAX = 2048;

const unsigned int SHIP_INITIAL_NUM = 3;
const float SHIP_SCALE_X = 16.0f;
const float SHIP_SCALE_Y = 16.0f;
const float BULLET_SCALE_X = 20.0f;
const float BULLET_SCALE_Y = 3.0f;
const float ASTEROID_MIN_SCALE_X = 10.0f;
const float ASTEROID_MAX_SCALE_X = 60.0f;
const float ASTEROID_MIN_SCALE_Y = 10.0f;
const float ASTEROID_MAX_SCALE_Y = 60.0f;

const float SHIP_ACCEL_FORWARD = 100.0f;
const float SHIP_ACCEL_BACKWARD = 100.0f;
const float SHIP_ROT_SPEED = (2.0f * PI);
const float BOUNDING_RECT_SIZE = 1.0f;

// Game object types
enum TYPE {
    TYPE_SHIP = 0,
    TYPE_BULLET,
    TYPE_WALL,
    TYPE_ASTEROID,
    TYPE_EXPLOSION,
    TYPE_NUM
};

// Object flags
const unsigned long FLAG_ACTIVE = 0x00000001;

// Network constants
const unsigned short SERVER_PORT = 27015;
const unsigned short CLIENT_PORT = 27016;
const float NETWORK_UPDATE_RATE = 0.05f; // 20 updates per second
const float PING_INTERVAL = 1.0f; // 1 ping per second

const uint32_t WIN_SCORE_THRESHOLD = 5000;
const float GAME_RESTART_DELAY = 5.0f; // 5 seconds until game restart

// Constructor
MultiplayerMode::MultiplayerMode(const std::string& serverAddress)
    : m_playerID(0), m_isConnected(false), m_currentSequence(0), m_lastReceivedSequence(0),
    m_gameObjNum(0), m_gameObjInstNum(0),
    m_networkDelay(0.0f), m_gameStarted(false), m_randomSeed(0),
    m_backgroundMesh(nullptr), m_background(nullptr),
    m_score(0), m_shipLives(SHIP_INITIAL_NUM), m_onValueChange(true), m_gameOver(false),
    m_gameRestartTimer(0.0f), m_showRestartMessage(false) {

    // Set server endpoint
    m_serverEndpoint.address = serverAddress;
    m_serverEndpoint.port = SERVER_PORT;

    // Initialize game object arrays
    m_gameObjList.resize(GAME_OBJ_NUM_MAX);
    m_gameObjInstList.resize(GAME_OBJ_INST_NUM_MAX);
}

// Destructor
MultiplayerMode::~MultiplayerMode() {
    // Clean up
    if (m_isConnected) {
        // Send disconnect message
        DisconnectMessage disconnect(m_playerID);
        m_socket.SendMessage(&disconnect, m_serverEndpoint);
    }
}

// Initialize the game mode
bool MultiplayerMode::Initialize() {
    // Initialize network socket
    if (!m_socket.Initialize()) {
        std::cerr << "Failed to initialize UDP socket" << std::endl;
        return false;
    }

    // Try to bind to CLIENT_PORT first
    bool bindSuccess = m_socket.Bind(CLIENT_PORT);

    // If binding to CLIENT_PORT fails, let the OS assign a random available port
    if (!bindSuccess) {
        std::cout << "Port " << CLIENT_PORT << " in use, trying dynamic port allocation..." << std::endl;
        bindSuccess = m_socket.Bind(0); // Port 0 means let the OS choose a free port
    }

    if (!bindSuccess) {
        std::cerr << "Failed to bind UDP socket to any port" << std::endl;
        return false;
    }

    // Log the port we're using
    std::cout << "Client bound to port " << m_socket.GetLocalPort() << std::endl;

    // Set message callback
    m_socket.SetMessageCallback([this](const NetworkMessage* message, const NetworkEndpoint& sender) {
        this->OnMessageReceived(message, sender);
    });

    // Start receiving messages
    if (!m_socket.StartReceiving()) {
        std::cerr << "Failed to start receiving messages" << std::endl;
        return false;
    }

    // Send join request to server
    JoinRequestMessage joinRequest("Player");
    if (!m_socket.SendMessage(&joinRequest, m_serverEndpoint)) {
        std::cerr << "Failed to send join request" << std::endl;
        return false;
    }

    // Initialize game objects
    InitializeGameObjects();

    // Set initial time
    m_lastUpdateTime = std::chrono::steady_clock::now();

    std::cout << "Connecting to server at " << m_serverEndpoint.address << ":" << m_serverEndpoint.port << std::endl;

    return true;
}

// Update the game mode
void MultiplayerMode::Update() {
    auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - m_lastUpdateTime).count();
    m_lastUpdateTime = currentTime;

    // If the game hasn't started yet, just return
    if (!m_gameStarted) {
        return;
    }

    // Send player input to server
    SendPlayerInput();

    // Update all game object instances
    for (unsigned long i = 0; i < GAME_OBJ_INST_NUM_MAX; i++) {
        MPGameObjInst* pInst = &m_gameObjInstList[i];

        // Skip non-active object
        if ((pInst->flag & FLAG_ACTIVE) == 0)
            continue;

        // Save previous position
        pInst->posPrev.x = pInst->posCurr.x;
        pInst->posPrev.y = pInst->posCurr.y;

        // Only run client-side physics for objects we own (our ship)
        // Server is authoritative for all other objects
        bool isOurShip = (pInst->pObject && pInst->pObject->type == TYPE_SHIP && pInst->ownerID == m_playerID);

        if (isOurShip) {
            // Client-side prediction: apply input locally for immediate feedback (mirrors server logic)

            // Get current input state
            bool upPressed = AEInputCheckCurr(AEVK_UP);
            bool downPressed = AEInputCheckCurr(AEVK_DOWN);
            bool leftPressed = AEInputCheckCurr(AEVK_LEFT);
            bool rightPressed = AEInputCheckCurr(AEVK_RIGHT);

            // Apply rotation (same logic as server in ProcessPlayerInput)
            if (leftPressed) {
                pInst->dirCurr += SHIP_ROT_SPEED * g_dt;
                // Wrap angle to [-PI, PI]
                while (pInst->dirCurr > PI) pInst->dirCurr -= 2.0f * PI;
                while (pInst->dirCurr < -PI) pInst->dirCurr += 2.0f * PI;
            }

            if (rightPressed) {
                pInst->dirCurr -= SHIP_ROT_SPEED * g_dt;
                // Wrap angle to [-PI, PI]
                while (pInst->dirCurr > PI) pInst->dirCurr -= 2.0f * PI;
                while (pInst->dirCurr < -PI) pInst->dirCurr += 2.0f * PI;
            }

            // Apply acceleration (same logic as server in ProcessPlayerInput)
            if (upPressed) {
                float velNewX = cosf(pInst->dirCurr) * SHIP_ACCEL_FORWARD * g_dt + pInst->velCurr.x;
                float velNewY = sinf(pInst->dirCurr) * SHIP_ACCEL_FORWARD * g_dt + pInst->velCurr.y;

                pInst->velCurr.x = velNewX * 0.99f;
                pInst->velCurr.y = velNewY * 0.99f;
            }

            if (downPressed) {
                float velNewX = -cosf(pInst->dirCurr) * SHIP_ACCEL_BACKWARD * g_dt + pInst->velCurr.x;
                float velNewY = -sinf(pInst->dirCurr) * SHIP_ACCEL_BACKWARD * g_dt + pInst->velCurr.y;

                pInst->velCurr.x = velNewX * 0.99f;
                pInst->velCurr.y = velNewY * 0.99f;
            }

            // Update position based on velocity (now includes locally-applied input)
            pInst->posCurr.x = pInst->posPrev.x + pInst->velCurr.x * g_dt;
            pInst->posCurr.y = pInst->posPrev.y + pInst->velCurr.y * g_dt;
        }
        // For all other objects, trust the server's position (set via network updates)

        // Check for bullets going out of bounds
        if (pInst->pObject && pInst->pObject->type == TYPE_BULLET) {
            // Destroy bullets that go off screen
            if (pInst->posCurr.x < AEGfxGetWinMinX() || pInst->posCurr.x > AEGfxGetWinMaxX() ||
                pInst->posCurr.y < AEGfxGetWinMinY() || pInst->posCurr.y > AEGfxGetWinMaxY()) {
                // Mark the bullet for destruction
                pInst->cooldown = 0.0f;
            }

            // Decrement bullet lifetime
            pInst->cooldown -= g_dt;
            if (pInst->cooldown <= 0.0f) {
                DestroyGameObjInst(pInst);
                continue; // Skip the rest of loop for this bullet
            }
        }

        // Update bounding box
        pInst->boundingBox.min.x = (-BOUNDING_RECT_SIZE / 2.0f) * pInst->scale.x + pInst->posPrev.x;
        pInst->boundingBox.min.y = (-BOUNDING_RECT_SIZE / 2.0f) * pInst->scale.y + pInst->posPrev.y;
        pInst->boundingBox.max.x = (BOUNDING_RECT_SIZE / 2.0f) * pInst->scale.x + pInst->posPrev.x;
        pInst->boundingBox.max.y = (BOUNDING_RECT_SIZE / 2.0f) * pInst->scale.y + pInst->posPrev.y;

        // Calculate transformation matrix
        AEMtx33 trans, rot, scale;

        // Compute the scaling matrix
        AEMtx33Scale(&scale, pInst->scale.x, pInst->scale.y);

        // Compute the rotation matrix
        AEMtx33Rot(&rot, pInst->dirCurr);

        // Compute the translation matrix
        AEMtx33Trans(&trans, pInst->posCurr.x, pInst->posCurr.y);

        // Concatenate the matrices
        AEMtx33Concat(&pInst->transform, &rot, &scale);
        AEMtx33Concat(&pInst->transform, &trans, &pInst->transform);

        // Handle explosions cooldown
        if (pInst->pObject && pInst->pObject->type == TYPE_EXPLOSION) {
            pInst->cooldown -= g_dt;
            if (pInst->cooldown <= 0.0f) {
                DestroyGameObjInst(pInst);
            }
        }
    }

    // Regular ping to measure network delay
    static float timeSinceLastPing = 0.0f;
    timeSinceLastPing += deltaTime;

    if (timeSinceLastPing >= 5.0f) {
        UpdateNetworkDelay();
        timeSinceLastPing = 0.0f;
    }

    // Also send a regular "keepalive" ping even if there's no input
    static float timeSinceLastKeepalive = 0.0f;
    timeSinceLastKeepalive += deltaTime;

    // Send a keepalive ping every 8 seconds even if there's no input
    if (timeSinceLastKeepalive >= 8.0f) {
        // Send an empty input message to keep the connection active
        uint8_t emptyInputFlags = 0;
        float currentDirection = 0.0f;

        // Find our ship instance to get current direction
        auto it = m_players.find(m_playerID);
        if (it != m_players.end() && it->second.shipInstance) {
            currentDirection = it->second.shipInstance->dirCurr;
        }

        uint32_t timestamp = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

        // Send keepalive input message
        PlayerInputMessage keepaliveMsg(m_playerID, emptyInputFlags, currentDirection, timestamp);
        m_socket.SendMessage(&keepaliveMsg, m_serverEndpoint);

        timeSinceLastKeepalive = 0.0f;
    }

    if (m_gameOver && m_showRestartMessage) {
        m_gameRestartTimer -= deltaTime;
        if (m_gameRestartTimer <= 0.0f) {
            // Game should be restarted by the server shortly
            m_showRestartMessage = false;

            // Clear existing game objects to prepare for reset
            // Note: Don't destroy network connection
            for (auto& pInst : m_gameObjInstList) {
                if (pInst.flag & FLAG_ACTIVE) {
                    DestroyGameObjInst(&pInst);
                }
            }
            m_networkObjects.clear();

            // Reset local game state vars, but don't reset m_gameOver
            // The server will send a new GameStart message to reset everything
            m_score = 0;
            m_shipLives = SHIP_INITIAL_NUM;
            m_onValueChange = true;
        }
    }
}

// Draw the game mode
void MultiplayerMode::Draw() {
    // Draw background
    AEGfxSetRenderMode(AE_GFX_RM_TEXTURE);
    AEGfxTextureSet(m_background, 0.0f, 0.0f);

    AEGfxSetColorToMultiply(1.0f, 1.0f, 1.0f, 1.0f);
    AEGfxSetBlendMode(AE_GFX_BM_BLEND);
    AEGfxSetTransparency(1.0f);

    AEGfxSetTransform(m_backgroundScale.m);
    AEGfxMeshDraw(m_backgroundMesh, AE_GFX_MDM_TRIANGLES);

    // Count objects by type for debugging
    int shipCount = 0;
    int bulletCount = 0;
    int asteroidCount = 0;
    int explosionCount = 0;

    // Draw all object instances
    for (unsigned long i = 0; i < GAME_OBJ_INST_NUM_MAX; i++) {
        MPGameObjInst* pInst = &m_gameObjInstList[i];

        // Skip non-active object
        if ((pInst->flag & FLAG_ACTIVE) == 0 || !pInst->pObject)
            continue;

        // Count by type
        if (pInst->pObject->type == TYPE_SHIP) shipCount++;
        else if (pInst->pObject->type == TYPE_BULLET) bulletCount++;
        else if (pInst->pObject->type == TYPE_ASTEROID) asteroidCount++;
        else if (pInst->pObject->type == TYPE_EXPLOSION) explosionCount++;

        // Draw the object
        AEGfxTextureSet(pInst->pObject->pTex, 0.0f, 0.0f);
        AEGfxSetTransform(pInst->transform.m);
        AEGfxMeshDraw(pInst->pObject->pMesh, AE_GFX_MDM_TRIANGLES);
    }

    // Show object counts for debugging
    /*char strBuffer[1024];
    sprintf_s(strBuffer, "Ships: %d, Bullets: %d, Asteroids: %d, Explosions: %d",
        shipCount, bulletCount, asteroidCount, explosionCount);
    printf("%s\n", strBuffer);*/

    // Draw UI elements
    DrawUI();
}

// Free resources
void MultiplayerMode::Free() {
    // Clean up game objects
    for (auto& pInst : m_gameObjInstList) {
        if (pInst.flag & FLAG_ACTIVE) {
            DestroyGameObjInst(&pInst);
        }
    }
    m_networkObjects.clear();
}

// Unload resources
void MultiplayerMode::Unload() {
    // Stop receiving network messages
    m_socket.StopReceiving();

    // Free all mesh data
    for (unsigned long i = 0; i < m_gameObjNum; i++) {
        MPGameObj* pObj = &m_gameObjList[i];
        if (pObj->pMesh) {
            AEGfxMeshFree(pObj->pMesh);
        }
        if (pObj->pTex) {
            AEGfxTextureUnload(pObj->pTex);
        }
    }

    // Free background and resources
    if (m_backgroundMesh) {
        AEGfxMeshFree(m_backgroundMesh);
        m_backgroundMesh = nullptr;
    }

    if (m_background) {
        AEGfxTextureUnload(m_background);
        m_background = nullptr;
    }

    // Unload audio
    AEAudioUnloadAudio(m_accelerationSound);
    AEAudioUnloadAudio(m_bulletSound);
    AEAudioUnloadAudio(m_explosion);
    AEAudioUnloadAudioGroup(m_soundEffect);
}

// Initialize game objects
void MultiplayerMode::InitializeGameObjects() {
    // Clear game object arrays
    m_gameObjNum = 0;
    for (auto& obj : m_gameObjList) {
        obj.pMesh = nullptr;
        obj.pTex = nullptr;
        obj.type = 0;
    }

    m_gameObjInstNum = 0;
    for (auto& inst : m_gameObjInstList) {
        inst.flag = 0;
        inst.pObject = nullptr;
    }

    // Create background
    AEGfxMeshStart();
    AEGfxTriAdd(
        -0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 1.0f,
        0.5f, 0.5f, 0xFFFFFFFF, 1.0f, 0.0f,
        -0.5f, 0.5f, 0xFFFFFFFF, 0.0f, 0.0f);
    AEGfxTriAdd(
        -0.5f, -0.5f, 0xFFFFFFFF, 0.0f, 1.0f,
        0.5f, -0.5f, 0xFFFFFFFF, 1.0f, 1.0f,
        0.5f, 0.5f, 0xFFFFFFFF, 1.0f, 0.0f);

    m_backgroundMesh = AEGfxMeshEnd();
    m_background = AEGfxTextureLoad("../Resources/Textures/Background.png");

    // Scale the background
    m_backgroundScale = { 0 };
    AEMtx33Scale(&m_backgroundScale, static_cast<float>(AEGfxGetWindowWidth()),
        static_cast<float>(AEGfxGetWindowHeight()));

    // Create ship shape
    MPGameObj* pObj = &m_gameObjList[m_gameObjNum++];
    pObj->type = TYPE_SHIP;

    AEGfxMeshStart();
    AEGfxTriAdd(
        -0.5f, 0.5f, 0xFFFF0000, 0.0f, 1.0f,
        -0.5f, -0.5f, 0xFFFF0000, 1.0f, 1.0f,
        0.5f, 0.0f, 0xFFFFFFFF, 0.5f, 0.0f);

    pObj->pMesh = AEGfxMeshEnd();
    pObj->pTex = AEGfxTextureLoad("../Resources/Textures/Ship.png");

    // Create bullet shape
    pObj = &m_gameObjList[m_gameObjNum++];
    pObj->type = TYPE_BULLET;

    AEGfxMeshStart();
    AEGfxTriAdd(
        -0.5f, -0.5f, 0xFFFFFF00, 0.0f, 1.0f,
        0.5f, 0.5f, 0xFFFFFF00, 1.0f, 0.0f,
        -0.5f, 0.5f, 0xFFFFFF00, 0.0f, 0.0f);
    AEGfxTriAdd(
        -0.5f, -0.5f, 0xFFFFFF00, 0.0f, 1.0f,
        0.5f, -0.5f, 0xFFFFFF00, 1.0f, 1.0f,
        0.5f, 0.5f, 0xFFFFFF00, 1.0f, 0.0f);

    pObj->pMesh = AEGfxMeshEnd();
    pObj->pTex = AEGfxTextureLoad("../Resources/Textures/Bullet.png");

    // Create wall shape
    pObj = &m_gameObjList[m_gameObjNum++];
    pObj->type = TYPE_WALL;

    AEGfxMeshStart();
    AEGfxTriAdd(
        -0.5f, -0.5f, 0x6600FF00, 0.0f, 1.0f,
        0.5f, 0.5f, 0x6600FF00, 1.0f, 0.0f,
        -0.5f, 0.5f, 0x6600FF00, 0.0f, 0.0f);
    AEGfxTriAdd(
        -0.5f, -0.5f, 0x6600FF00, 0.0f, 1.0f,
        0.5f, -0.5f, 0x6600FF00, 1.0f, 1.0f,
        0.5f, 0.5f, 0x6600FF00, 1.0f, 0.0f);

    pObj->pMesh = AEGfxMeshEnd();
    pObj->pTex = AEGfxTextureLoad("../Resources/Textures/Wall.png");

    // Create asteroid shape
    pObj = &m_gameObjList[m_gameObjNum++];
    pObj->type = TYPE_ASTEROID;

    AEGfxMeshStart();
    AEGfxTriAdd(
        -0.5f, -0.5f, 0x66FFFFFF, 0.0f, 1.0f,
        0.5f, 0.5f, 0x66FFFFFF, 1.0f, 0.0f,
        -0.5f, 0.5f, 0x66FFFFFF, 0.0f, 0.0f);
    AEGfxTriAdd(
        -0.5f, -0.5f, 0x66FFFFFF, 0.0f, 1.0f,
        0.5f, -0.5f, 0x66FFFFFF, 1.0f, 1.0f,
        0.5f, 0.5f, 0x66FFFFFF, 1.0f, 0.0f);

    pObj->pMesh = AEGfxMeshEnd();
    pObj->pTex = AEGfxTextureLoad("../Resources/Textures/Asteroid.png");

    // Create explosion shape
    pObj = &m_gameObjList[m_gameObjNum++];
    pObj->type = TYPE_EXPLOSION;

    AEGfxMeshStart();
    AEGfxTriAdd(
        -0.5f, -0.5f, 0x66FFFFFF, 0.0f, 1.0f,
        0.5f, 0.5f, 0x66FFFFFF, 1.0f, 0.0f,
        -0.5f, 0.5f, 0x66FFFFFF, 0.0f, 0.0f);
    AEGfxTriAdd(
        -0.5f, -0.5f, 0x66FFFFFF, 0.0f, 1.0f,
        0.5f, -0.5f, 0x66FFFFFF, 1.0f, 1.0f,
        0.5f, 0.5f, 0x66FFFFFF, 1.0f, 0.0f);

    pObj->pMesh = AEGfxMeshEnd();
    pObj->pTex = AEGfxTextureLoad("../Resources/Textures/Explosion.png");

    // Load sounds
    m_accelerationSound = AEAudioLoadSound("../Resources/Sounds/SpaceEngine.ogg");
    m_bulletSound = AEAudioLoadSound("../Resources/Sounds/LaserRetro.ogg");
    m_explosion = AEAudioLoadSound("../Resources/Sounds/Explosion.ogg");
    m_soundEffect = AEAudioCreateGroup();
}

// Create a game object instance
MPGameObjInst* MultiplayerMode::CreateGameObjInst(unsigned long type, AEVec2* scale, AEVec2* pPos, AEVec2* pVel, float dir, uint32_t networkID, uint8_t ownerID) {
    AEVec2 zero;
    AEVec2Zero(&zero);

    // Make sure the type is valid
    if (type >= m_gameObjNum)
        return nullptr;

    // Find a free slot in the instance array
    for (unsigned long i = 0; i < GAME_OBJ_INST_NUM_MAX; i++) {
        MPGameObjInst* pInst = &m_gameObjInstList[i];

        // Check if slot is free
        if (pInst->flag == 0) {
            // Initialize the instance
            pInst->pObject = &m_gameObjList[type];
            pInst->flag = FLAG_ACTIVE;
            pInst->scale = scale ? *scale : zero;
            pInst->posCurr = pPos ? *pPos : zero;
            pInst->velCurr = pVel ? *pVel : zero;
            pInst->dirCurr = dir;
            pInst->networkID = networkID;
            pInst->ownerID = ownerID;
            pInst->cooldown = 1.0f;

            // Return the new instance
            return pInst;
        }
    }

    // No free slots
    return nullptr;
}

// Destroy a game object instance
void MultiplayerMode::DestroyGameObjInst(MPGameObjInst* pInst) {
    // Check if the instance is already destroyed
    if (!pInst || pInst->flag == 0)
        return;

    // Remove from network objects map if it has a network ID
    if (pInst->networkID > 0) {
        m_networkObjects.erase(pInst->networkID);
    }

    // Clear the flag
    pInst->flag = 0;
}

// Process network messages
void MultiplayerMode::OnMessageReceived(const NetworkMessage* message, const NetworkEndpoint& sender) {
    // Only process messages from the server
    if (sender.address != m_serverEndpoint.address || sender.port != m_serverEndpoint.port) {
        return;
    }

    // Handle different message types
    switch (message->GetType()) {
    case MessageType::JOIN_RESPONSE:
        HandleJoinResponse(static_cast<const JoinResponseMessage*>(message));
        break;

    case MessageType::GAME_START:
        HandleGameStart(static_cast<const GameStartMessage*>(message));
        break;

    case MessageType::GAME_STATE_UPDATE:
        HandleGameStateUpdate(static_cast<const GameStateUpdateMessage*>(message));
        break;

    case MessageType::OBJECT_CREATE:
        HandleObjectCreate(static_cast<const ObjectCreateMessage*>(message));
        break;

    case MessageType::OBJECT_DESTROY:
        HandleObjectDestroy(static_cast<const ObjectDestroyMessage*>(message));
        break;

    case MessageType::SCORE_UPDATE:
        HandleScoreUpdate(static_cast<const ScoreUpdateMessage*>(message));
        break;

    case MessageType::GAME_END:
        HandleGameEnd(static_cast<const GameEndMessage*>(message));
        break;

    case MessageType::DISCONNECT:
        HandleDisconnect(static_cast<const DisconnectMessage*>(message));
        break;

    default:
        break;
    }
}

// Send player input to server
void MultiplayerMode::SendPlayerInput() {
    if (!m_isConnected || !m_gameStarted) {
        return;
    }

    // Get current state of all input keys
    bool upPressed = AEInputCheckCurr(AEVK_UP);
    bool downPressed = AEInputCheckCurr(AEVK_DOWN);
    bool leftPressed = AEInputCheckCurr(AEVK_LEFT);
    bool rightPressed = AEInputCheckCurr(AEVK_RIGHT);
    bool fireTriggered = AEInputCheckTriggered(AEVK_SPACE);

    // DIAGNOSTIC: Log when SPACE is pressed
    if (fireTriggered) {
        auto now = std::chrono::high_resolution_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::cout << "[CLIENT] FIRE TRIGGERED at time " << timestamp << "ms" << std::endl;
    }

    // Compose input flags (send even if no keys pressed to inform server of state)
    uint8_t inputFlags = 0;
    if (upPressed) inputFlags |= PlayerInputMessage::INPUT_UP;
    if (downPressed) inputFlags |= PlayerInputMessage::INPUT_DOWN;
    if (leftPressed) inputFlags |= PlayerInputMessage::INPUT_LEFT;
    if (rightPressed) inputFlags |= PlayerInputMessage::INPUT_RIGHT;
    if (fireTriggered) inputFlags |= PlayerInputMessage::INPUT_FIRE;

    // Get current direction
    float direction = 0.0f;

    // Find our ship instance
    auto it = m_players.find(m_playerID);
    if (it != m_players.end() && it->second.shipInstance) {
        direction = it->second.shipInstance->dirCurr;
    }

    // Get current timestamp
    uint32_t timestamp = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());

    // Send combined input message every tick
    PlayerInputMessage inputMessage(m_playerID, inputFlags, direction, timestamp);
    m_socket.SendMessage(&inputMessage, m_serverEndpoint);

    // DIAGNOSTIC: Log when FIRE message is sent
    if (inputFlags & PlayerInputMessage::INPUT_FIRE) {
        auto now = std::chrono::high_resolution_clock::now();
        auto sendTime = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::cout << "[CLIENT] FIRE MESSAGE SENT at time " << sendTime << "ms" << std::endl;
    }

    // Debug log
    /*if (fireTriggered) {
        std::cout << "Sending fire input to server" << std::endl;
    }*/
}

// Handle join response message
void MultiplayerMode::HandleJoinResponse(const JoinResponseMessage* message) {
    if (message->IsAccepted()) {
        m_playerID = message->GetPlayerID();
        m_isConnected = true;

        // Add ourselves to the player list
        MultiplayerPlayer player;
        player.id = m_playerID;
        player.name = "Player " + std::to_string(m_playerID);
        player.isConnected = true;
        player.shipInstance = nullptr;
        player.score = 0;
        player.lives = SHIP_INITIAL_NUM;

        m_players[m_playerID] = player;

        std::cout << "Connected to server as Player " << static_cast<int>(m_playerID) << std::endl;

        // Note: We don't need to do anything special here for joining ongoing games
        // The server will send us GameStart, then the full GameStateUpdate
    }
    else {
        std::cerr << "Server rejected connection request" << std::endl;
    }
}

// Handle game start message
void MultiplayerMode::HandleGameStart(const GameStartMessage* message) {
    m_randomSeed = message->GetRandomSeed();
    m_gameStarted = true;

    // Reset game state
    m_score = 0;
    m_shipLives = SHIP_INITIAL_NUM;
    m_onValueChange = true;
    m_gameOver = false;
    m_showRestartMessage = false;
    m_gameRestartTimer = 0.0f;

    // Clear existing game objects on restart
    for (auto& pInst : m_gameObjInstList) {
        if (pInst.flag & FLAG_ACTIVE) {
            DestroyGameObjInst(&pInst);
        }
    }
    m_networkObjects.clear();

    //std::cout << "Game starting/restarting with seed " << m_randomSeed << std::endl;
}

// Handle game state update message
void MultiplayerMode::HandleGameStateUpdate(const GameStateUpdateMessage* message) {
    // Validate sequence number to reject out-of-order packets
    uint32_t incomingSequence = message->GetSequenceNumber();
    if (incomingSequence <= m_lastReceivedSequence && m_lastReceivedSequence != 0) {
        // This is an old packet, ignore it
        return;
    }
    m_lastReceivedSequence = incomingSequence;

    // First, track all players we see in this update
    std::set<uint8_t> activePlayers;

    // Update all objects from the message
    for (const auto& netObj : message->GetObjects()) {
        // If this is a ship, note the player as active
        if (netObj.type == TYPE_SHIP) {
            activePlayers.insert(netObj.ownerID);

            // Make sure this player exists in our player map
            if (m_players.find(netObj.ownerID) == m_players.end()) {
                // New player detected - add them
                HandlePlayerConnectionUpdate(netObj.ownerID, true);
            }
        }

        // Update the object as usual
        CreateOrUpdateGameObject(netObj);
    }

    // For all players in our map, check if they appear in the active list
    for (auto& pair : m_players) {
        if (pair.second.isConnected && activePlayers.find(pair.first) == activePlayers.end() && pair.first != m_playerID) {
            // This player had a ship before but doesn't anymore - might be disconnected
            // Don't disconnect ourselves though - we might just have died
            if (pair.second.shipInstance) {
                pair.second.shipInstance = nullptr;
            }
        }
    }
}

// Handle object create message
void MultiplayerMode::HandleObjectCreate(const ObjectCreateMessage* message) {
    const NetworkGameObject& netObj = message->GetTheObject();

    // Log object creation for debugging
    /*std::cout << "Received object create: ID=" << netObj.id
        << ", Type=" << static_cast<int>(netObj.type)
        << ", Owner=" << static_cast<int>(netObj.ownerID)
        << ", Pos=(" << netObj.position.x << "," << netObj.position.y << ")"
        << ", Dir=" << netObj.direction
        << std::endl;*/

    // Create or update the object
    MPGameObjInst* pInst = CreateOrUpdateGameObject(netObj);

    // Handle bullets specially
    if (netObj.type == TYPE_BULLET) {
        // Play bullet sound
        AEAudioPlay(m_bulletSound, m_soundEffect, 0.6f, 1.0f, 0);

        // Make sure the bullet is active
        if (pInst) {
            pInst->flag |= FLAG_ACTIVE;

            // Reset cooldown for bullet lifetime
            pInst->cooldown = 3.0f; // Bullets should last 3 seconds

            //std::cout << "Bullet created with ID " << pInst->networkID << std::endl;
        }
    }

    // Send acknowledgment for the object creation
    // This step is optional but can help with synchronization
    if (netObj.type == TYPE_BULLET || netObj.type == TYPE_ASTEROID) {
        ObjectDestroyAckMessage ackMessage(netObj.id, m_currentSequence++, m_playerID);
        m_socket.SendMessage(&ackMessage, m_serverEndpoint);
    }
}

// Handle object destroy message
void MultiplayerMode::HandleObjectDestroy(const ObjectDestroyMessage* message) {
    uint32_t objectID = message->GetObjectID();
    uint32_t sequenceNumber = message->GetSequenceNumber();

    // Find the object
    auto it = m_networkObjects.find(objectID);
    if (it != m_networkObjects.end()) {
        // Check if this is a player ship
        if (it->second->pObject && it->second->pObject->type == TYPE_SHIP) {
            uint8_t ownerID = it->second->ownerID;

            // Update player map to indicate ship is destroyed
            auto playerIt = m_players.find(ownerID);
            if (playerIt != m_players.end()) {
                // Remove ship association but don't disconnect player
                playerIt->second.shipInstance = nullptr;
            }
        }

        // Destroy the object
        DestroyGameObjInst(it->second);

        // Send acknowledgment
        SendObjectDestroyAck(objectID, sequenceNumber);
    }
}

// Handle score update message
void MultiplayerMode::HandleScoreUpdate(const ScoreUpdateMessage* message) {
    // Update scores for all players
    for (uint8_t i = 0; i < 4; i++) {
        uint32_t score = message->GetScore(i);

        // Only update if there's a non-zero score (indicating an active player)
        if (score > 0 || m_players.find(i) != m_players.end()) {
            // Create player entry if it doesn't exist
            if (m_players.find(i) == m_players.end()) {
                MultiplayerPlayer player;
                player.id = i;
                player.name = "Player " + std::to_string(i);
                player.isConnected = true;
                player.shipInstance = nullptr;
                player.score = score;
                player.lives = SHIP_INITIAL_NUM;

                m_players[i] = player;
            }
            else {
                m_players[i].score = score;
            }
        }
    }

    // Update our score
    m_score = message->GetScore(m_playerID);
    m_onValueChange = true;
}

// Handle game end message
void MultiplayerMode::HandleGameEnd(const GameEndMessage* message) {
    uint8_t winnerID = message->GetWinnerID();

    std::cout << "Game over! Player " << static_cast<int>(winnerID) << " wins!" << std::endl;

    // Set game over flag
    m_gameOver = true;
    m_onValueChange = true;

    // Start restart timer
    m_gameRestartTimer = GAME_RESTART_DELAY;
    m_showRestartMessage = true;

    // Find winning player and display their score
    uint32_t winningScore = 0;
    auto it = m_players.find(winnerID);
    if (it != m_players.end()) {
        winningScore = it->second.score;
    }

    std::cout << "Player " << static_cast<int>(winnerID) << " won with score: " << winningScore << std::endl;
    std::cout << "Game will restart in " << GAME_RESTART_DELAY << " seconds..." << std::endl;
}

// Create or update a game object from network data
MPGameObjInst* MultiplayerMode::CreateOrUpdateGameObject(const NetworkGameObject& netObj) {
    // Check for unreasonable values before creating
    //if (netObj.type == TYPE_BULLET) {
    //    // Log bullet creation
    //    std::cout << "Received bullet object: ID=" << netObj.id
    //        << ", Position=(" << netObj.position.x << "," << netObj.position.y << ")"
    //        << ", Velocity=(" << netObj.velocity.x << "," << netObj.velocity.y << ")"
    //        << ", Direction=" << netObj.direction
    //        << std::endl;
    //}

    // Check if the object already exists
    auto it = m_networkObjects.find(netObj.id);

    if (it != m_networkObjects.end()) {
        // Update existing object
        MPGameObjInst* pInst = it->second;

        // For non-owned objects, use interpolation to smooth position updates
        // Owned objects (our ship) use direct updates for immediate response
        bool isOwnedByUs = (netObj.ownerID == m_playerID && netObj.type == TYPE_SHIP);

        if (isOwnedByUs) {
            // Server reconciliation for our own ship
            // Calculate position difference for error correction
            float dx = netObj.position.x - pInst->posCurr.x;
            float dy = netObj.position.y - pInst->posCurr.y;
            float distSq = dx * dx + dy * dy;

            // If difference is large (>100 units squared), snap to server position (wrap-around or teleport)
            // Otherwise, smoothly blend toward server position for correction
            const float snapThreshold = 10000.0f; // 100 units squared
            const float correctionAlpha = 0.2f;   // Blend 20% toward server position per frame

            if (distSq > snapThreshold) {
                // Large correction - snap immediately (e.g., wrap-around at screen edge)
                pInst->posCurr.x = netObj.position.x;
                pInst->posCurr.y = netObj.position.y;
            } else {
                // Small correction - blend smoothly to hide prediction errors
                pInst->posCurr.x += dx * correctionAlpha;
                pInst->posCurr.y += dy * correctionAlpha;
            }

            // Always trust server velocity (handles collisions, server-side events)
            pInst->velCurr.x = netObj.velocity.x;
            pInst->velCurr.y = netObj.velocity.y;
            pInst->dirCurr = netObj.direction;
        } else {
            // For other objects, use exponential smoothing for position (alpha = 0.3 means blend 30% toward new position)
            // This smooths out the jitter from network updates
            const float alpha = 0.3f;
            pInst->posCurr.x = pInst->posCurr.x * (1.0f - alpha) + netObj.position.x * alpha;
            pInst->posCurr.y = pInst->posCurr.y * (1.0f - alpha) + netObj.position.y * alpha;

            // Update velocity directly (less visible jitter)
            pInst->velCurr.x = netObj.velocity.x;
            pInst->velCurr.y = netObj.velocity.y;
            pInst->dirCurr = netObj.direction;
        }

        // Update the transform for proper rendering
        AEMtx33 trans, rot, scale;

        // Compute the scaling matrix
        AEMtx33Scale(&scale, pInst->scale.x, pInst->scale.y);

        // Compute the rotation matrix
        AEMtx33Rot(&rot, pInst->dirCurr);

        // Compute the translation matrix
        AEMtx33Trans(&trans, pInst->posCurr.x, pInst->posCurr.y);

        // Concatenate the matrices
        AEMtx33Concat(&pInst->transform, &rot, &scale);
        AEMtx33Concat(&pInst->transform, &trans, &pInst->transform);

        return pInst;
    }
    else {
        // Create new object
        AEVec2 scale, pos, vel;

        // Convert from NetVector2 to AEVec2
        scale.x = netObj.scale.x;
        scale.y = netObj.scale.y;

        pos.x = netObj.position.x;
        pos.y = netObj.position.y;

        vel.x = netObj.velocity.x;
        vel.y = netObj.velocity.y;

        // Create instance with the correct orientation/direction
        MPGameObjInst* pInst = CreateGameObjInst(netObj.type, &scale, &pos, &vel, netObj.direction, netObj.id, netObj.ownerID);

        if (pInst) {
            // Add to network objects map
            m_networkObjects[netObj.id] = pInst;

            // Ship specific handling
            if (netObj.type == TYPE_SHIP) {
                auto playerIt = m_players.find(netObj.ownerID);
                if (playerIt == m_players.end()) {
                    // New player detected
                    MultiplayerPlayer player;
                    player.id = netObj.ownerID;
                    player.name = "Player " + std::to_string(netObj.ownerID);
                    player.isConnected = true;
                    player.shipInstance = pInst;
                    player.score = 0;
                    player.lives = SHIP_INITIAL_NUM;

                    m_players[netObj.ownerID] = player;
                    std::cout << "New player ship detected for Player " << static_cast<int>(netObj.ownerID) << std::endl;
                    m_onValueChange = true;
                }
                else {
                    // Update existing player's ship reference
                    playerIt->second.shipInstance = pInst;
                    playerIt->second.isConnected = true;
                }
            }

            // Enhanced bullet handling
            if (netObj.type == TYPE_BULLET) {
                // Make sure transform is updated correctly
                AEMtx33 trans, rot, scale;

                // For bullets, create proper rotation based on velocity direction
                // This is crucial for visual orientation
                if (pInst->velCurr.x != 0.0f || pInst->velCurr.y != 0.0f) {
                    // Calculate direction from velocity
                    pInst->dirCurr = atan2f(pInst->velCurr.y, pInst->velCurr.x);
                }

                // Compute scaling matrix for bullet
                AEMtx33Scale(&scale, pInst->scale.x, pInst->scale.y);

                // Compute rotation matrix for proper orientation
                AEMtx33Rot(&rot, pInst->dirCurr);

                // Compute translation matrix
                AEMtx33Trans(&trans, pInst->posCurr.x, pInst->posCurr.y);

                // Concatenate matrices
                AEMtx33Concat(&pInst->transform, &rot, &scale);
                AEMtx33Concat(&pInst->transform, &trans, &pInst->transform);

                // Play bullet sound
                AEAudioPlay(m_bulletSound, m_soundEffect, 0.3f, 1.0f, 0);

                /*std::cout << "Created bullet instance with ID " << netObj.id
                    << " at (" << pos.x << "," << pos.y << ")"
                    << " with direction " << pInst->dirCurr
                    << std::endl;*/
            }

            // Play sound for explosions
            if (netObj.type == TYPE_EXPLOSION) {
                AEAudioPlay(m_explosion, m_soundEffect, 1.0f, 1.0f, 0);
            }
        }

        return pInst;
    }
}

// Send acknowledgment for object destruction
void MultiplayerMode::SendObjectDestroyAck(uint32_t objectID, uint32_t sequenceNumber) {
    ObjectDestroyAckMessage ackMessage(objectID, sequenceNumber, m_playerID);
    m_socket.SendMessage(&ackMessage, m_serverEndpoint);
}

// Measure and update network delay
void MultiplayerMode::UpdateNetworkDelay() {
    // Get current timestamp
    uint64_t timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());

    // Send ping message
    PingMessage pingMessage(timestamp);
    m_socket.SendMessage(&pingMessage, m_serverEndpoint);
}

// Check if all acknowledgments have been received for a destroy event
bool MultiplayerMode::AreAllAcksReceived(const DestroyEvent& event) {
    for (const auto& pair : event.playerAcks) {
        if (!pair.second) {
            return false;
        }
    }

    return true;
}

// Draw UI elements
void MultiplayerMode::DrawUI() {
    char strBuffer[1024];

    // Set blend mode before calling AEGfxPrint as required
    AEGfxSetBlendMode(AE_GFX_BM_BLEND);

    // Starting position for player scores (top-left)
    float xPosNorm = 0.55f; // Near the left edge
    float yPosNorm = 0.0f; // Near the top
    float lineHeightNorm = 0.1f;

    // Sort players by score for leaderboard display
    std::vector<std::pair<uint8_t, uint32_t>> playerScores;
    for (const auto& pair : m_players) {
        playerScores.push_back({ pair.first, pair.second.score });
    }

    // Sort by score (highest first)
    std::sort(playerScores.begin(), playerScores.end(),
        [](const std::pair<uint8_t, uint32_t>& a, const std::pair<uint8_t, uint32_t>& b) {
        return a.second > b.second;
    });

    // Draw player scores in sorted order
    for (const auto& score : playerScores) {
        uint8_t playerId = score.first;
        uint32_t playerScore = score.second;

        // Highlight our player's score with different color
        float r, g, b;
        if (playerId == m_playerID) {
            // Yellow for our player
            r = 1.0f; g = 1.0f; b = 0.0f;
        }
        else {
            // White for other players
            r = 1.0f; g = 1.0f; b = 1.0f;
        }

        // Set the string buffer for this player's score
        sprintf_s(strBuffer, "Player %d: %d", playerId, playerScore);
        AEGfxPrint(g_font, strBuffer, xPosNorm, yPosNorm, 1.0f, r, g, b, 1.0f);

        yPosNorm -= lineHeightNorm;
    }

    // Display network delay (top-right)
    sprintf_s(strBuffer, "Ping: %.0f ms", m_networkDelay);
    AEGfxPrint(g_font, strBuffer, 0.75f, 0.95f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    // Display win threshold (top-left)
    sprintf_s(strBuffer, "Win at: %d pts", WIN_SCORE_THRESHOLD);
    AEGfxPrint(g_font, strBuffer, -0.75f, -0.9f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);

    // Display game over message if needed
    if (m_gameOver) {
        // Find the winner
        uint8_t winnerID = 0;
        uint32_t highestScore = 0;

        for (const auto& pair : m_players) {
            if (pair.second.score > highestScore) {
                winnerID = pair.first;
                highestScore = pair.second.score;
            }
        }

        // Display winner message in the center of the screen
        sprintf_s(strBuffer, "PLAYER %d WINS!", winnerID);
        // Red color for the winner announcement
        AEGfxPrint(g_font, strBuffer, -0.15f, 0.0f, 1.5f, 1.0f, 0.0f, 0.0f, 1.0f);

        // Display score
        sprintf_s(strBuffer, "Score: %d", highestScore);
        AEGfxPrint(g_font, strBuffer, -0.1f, 0.1f, 1.2f, 1.0f, 1.0f, 1.0f, 1.0f);

        // Display restart message
        if (m_showRestartMessage) {
            sprintf_s(strBuffer, "Restarting in %.1f seconds...", m_gameRestartTimer);
            AEGfxPrint(g_font, strBuffer, -0.25f, 0.2f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f);
        }
    }

    if (m_onValueChange) {
        //// Log to console
        //printf("My Score: %d, Lives: %d\n", m_score, m_shipLives);

        //// Display scores for all players in console
        //for (const auto& pair : m_players) {
        //    printf("Player %d Score: %d\n", pair.first, pair.second.score);
        //}

        m_onValueChange = false;
    }
}

void MultiplayerMode::HandlePlayerConnectionUpdate(uint8_t playerID, bool connected) {
    auto it = m_players.find(playerID);

    if (connected) {
        // Player connected
        if (it == m_players.end()) {
            // New player - add to map
            MultiplayerPlayer player;
            player.id = playerID;
            player.name = "Player " + std::to_string(playerID);
            player.isConnected = true;
            player.shipInstance = nullptr;
            player.score = 0;
            player.lives = SHIP_INITIAL_NUM;

            m_players[playerID] = player;
            std::cout << "Player " << static_cast<int>(playerID) << " connected" << std::endl;
        }
        else {
            // Existing player - update connection status
            it->second.isConnected = true;
        }
    }
    else {
        // Player disconnected
        if (it != m_players.end()) {
            it->second.isConnected = false;
            if (it->second.shipInstance) {
                // Set ship instance to nullptr when player disconnects
                it->second.shipInstance = nullptr;
            }
            std::cout << "Player " << static_cast<int>(playerID) << " disconnected" << std::endl;
        }
    }
    m_onValueChange = true;
}

// In MultiplayerMode.cpp, add handler implementation:
void MultiplayerMode::HandleDisconnect(const DisconnectMessage* message) {
    uint8_t playerID = message->GetPlayerID();

    // Handle server (ID 0) or player disconnection
    if (playerID == 0) {
        // Server is shutting down
        m_isConnected = false;
        m_gameStarted = false;
        std::cout << "Server disconnected" << std::endl;
    }
    else {
        // Player disconnected
        HandlePlayerConnectionUpdate(playerID, false);
    }
}