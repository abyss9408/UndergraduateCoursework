/******************************************************************************/
/*!
\file		UDPSocket.h
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	March 29, 2025
\brief		This header file declares the UDPSocket class for network communication.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#pragma once

// Prevent winsock.h from being included by windows.h
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Include Windows headers first
#include <WinSock2.h>
#include <ws2tcpip.h>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include "NetworkMessage.h"

#pragma comment(lib, "Ws2_32.lib")

// Represents a network endpoint (IP + port)
struct NetworkEndpoint {
    std::string address;
    unsigned short port;

    // For use as map key
    bool operator<(const NetworkEndpoint& other) const {
        if (address != other.address)
            return address < other.address;
        return port < other.port;
    }

    bool operator==(const NetworkEndpoint& other) const {
        return address == other.address && port == other.port;
    }
};

// Callback for received messages
using MessageCallback = std::function<void(const NetworkMessage*, const NetworkEndpoint&)>;

// UDP Socket class for sending and receiving messages
class UDPSocket {
public:
    UDPSocket();
    ~UDPSocket();

    // Initialize the socket
    bool Initialize();

    // Bind to a specific port
    bool Bind(unsigned short port);

    // Set callback for received messages
    void SetMessageCallback(MessageCallback callback);

    // Start a background thread for receiving messages
    bool StartReceiving();

    // Stop the receiving thread
    void StopReceiving();

    // Send a message to a specific endpoint
    bool SendMessage(const NetworkMessage* message, const NetworkEndpoint& endpoint);

    // Broadcast a message to the local network
    bool BroadcastMessage(const NetworkMessage* message, unsigned short port);

    // Get the local address
    std::string GetLocalAddress() const;

    // Check if the socket is valid
    bool IsValid() const;

    // Get the local port that the socket is bound to
    unsigned short GetLocalPort() const;

private:
    // Socket handle
    SOCKET m_socket;

    // Callback for received messages
    MessageCallback m_callback;

    // Thread for receiving messages
    std::thread m_receiveThread;

    // Flag to control the receive thread
    std::atomic<bool> m_isReceiving;

    // Queue for received messages
    std::queue<std::pair<std::vector<uint8_t>, NetworkEndpoint>> m_messageQueue;

    // Mutex for the message queue
    std::mutex m_queueMutex;

    // Thread function for receiving messages
    void ReceiveThreadFunc();

    // Process a received message
    void ProcessReceivedMessage(const std::vector<uint8_t>& buffer, const NetworkEndpoint& sender);
};