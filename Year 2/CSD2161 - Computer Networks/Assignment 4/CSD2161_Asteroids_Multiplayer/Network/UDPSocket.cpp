/******************************************************************************/
/*!
\file		UDPSocket.cpp
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	March 29, 2025
\brief		This source file implements the UDPSocket class for network communication.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#include "UDPSocket.h"
#include <iostream>

// Static initialization of Winsock
namespace {
    class WinsockInitializer {
    public:
        WinsockInitializer() {
            WSADATA wsaData;
            int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
            if (result != 0) {
                std::cerr << "WSAStartup failed: " << result << std::endl;
            }
        }

        ~WinsockInitializer() {
            WSACleanup();
        }
    };

    static WinsockInitializer winsockInitializer;
}

UDPSocket::UDPSocket() : m_socket(INVALID_SOCKET), m_isReceiving(false) {
}

UDPSocket::~UDPSocket() {
    StopReceiving();

    if (m_socket != INVALID_SOCKET) {
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
    }
}

bool UDPSocket::Initialize() {
    // Create UDP socket
    m_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_socket == INVALID_SOCKET) {
        std::cerr << "socket failed: " << WSAGetLastError() << std::endl;
        return false;
    }

    // Set socket options for broadcast
    BOOL bOptVal = TRUE;
    int result = setsockopt(m_socket, SOL_SOCKET, SO_BROADCAST, (char*)&bOptVal, sizeof(bOptVal));
    if (result == SOCKET_ERROR) {
        std::cerr << "setsockopt failed: " << WSAGetLastError() << std::endl;
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
        return false;
    }

    // Set non-blocking mode
    u_long nonBlocking = 1;
    result = ioctlsocket(m_socket, FIONBIO, &nonBlocking);
    if (result == SOCKET_ERROR) {
        std::cerr << "ioctlsocket failed: " << WSAGetLastError() << std::endl;
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
        return false;
    }

    return true;
}

bool UDPSocket::Bind(unsigned short port) {
    if (m_socket == INVALID_SOCKET) {
        return false;
    }

    sockaddr_in localAddr;
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = INADDR_ANY;
    localAddr.sin_port = htons(port);

    int result = bind(m_socket, (sockaddr*)&localAddr, sizeof(localAddr));
    if (result == SOCKET_ERROR) {
        std::cerr << "bind failed: " << WSAGetLastError() << std::endl;
        return false;
    }

    return true;
}

void UDPSocket::SetMessageCallback(MessageCallback callback) {
    m_callback = callback;
}

bool UDPSocket::StartReceiving() {
    if (m_socket == INVALID_SOCKET || m_isReceiving) {
        return false;
    }

    m_isReceiving = true;
    m_receiveThread = std::thread(&UDPSocket::ReceiveThreadFunc, this);

    return true;
}

void UDPSocket::StopReceiving() {
    if (m_isReceiving) {
        m_isReceiving = false;

        if (m_receiveThread.joinable()) {
            m_receiveThread.join();
        }
    }
}

bool UDPSocket::SendMessage(const NetworkMessage* message, const NetworkEndpoint& endpoint) {
    if (m_socket == INVALID_SOCKET || !message) {
        return false;
    }

    std::vector<uint8_t> buffer = message->Serialize();
    if (buffer.empty()) {
        return false;
    }

    sockaddr_in destAddr;
    destAddr.sin_family = AF_INET;
    destAddr.sin_port = htons(endpoint.port);

    // Convert the address string to network format
    inet_pton(AF_INET, endpoint.address.c_str(), &destAddr.sin_addr);

    int result = sendto(m_socket, (const char*)buffer.data(), static_cast<int>(buffer.size()), 0,
        (sockaddr*)&destAddr, sizeof(destAddr));

    if (result == SOCKET_ERROR) {
        std::cerr << "sendto failed: " << WSAGetLastError() << std::endl;
        return false;
    }

    return true;
}

bool UDPSocket::BroadcastMessage(const NetworkMessage* message, unsigned short port) {
    if (m_socket == INVALID_SOCKET || !message) {
        return false;
    }

    std::vector<uint8_t> buffer = message->Serialize();
    if (buffer.empty()) {
        return false;
    }

    sockaddr_in destAddr;
    destAddr.sin_family = AF_INET;
    destAddr.sin_port = htons(port);
    destAddr.sin_addr.s_addr = INADDR_BROADCAST;

    int result = sendto(m_socket, (const char*)buffer.data(), static_cast<int>(buffer.size()), 0,
        (sockaddr*)&destAddr, sizeof(destAddr));

    if (result == SOCKET_ERROR) {
        std::cerr << "broadcast failed: " << WSAGetLastError() << std::endl;
        return false;
    }

    return true;
}

std::string UDPSocket::GetLocalAddress() const {
    char hostName[256];
    if (gethostname(hostName, sizeof(hostName)) == SOCKET_ERROR) {
        std::cerr << "gethostname failed: " << WSAGetLastError() << std::endl;
        return "127.0.0.1";
    }

    struct addrinfo hints;
    struct addrinfo* result = NULL;

    ZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;

    if (getaddrinfo(hostName, NULL, &hints, &result) != 0) {
        std::cerr << "getaddrinfo failed: " << WSAGetLastError() << std::endl;
        return "127.0.0.1";
    }

    char ipAddress[INET_ADDRSTRLEN];
    sockaddr_in* addr = (sockaddr_in*)result->ai_addr;
    inet_ntop(AF_INET, &addr->sin_addr, ipAddress, sizeof(ipAddress));

    freeaddrinfo(result);

    return std::string(ipAddress);
}

bool UDPSocket::IsValid() const {
    return m_socket != INVALID_SOCKET;
}

void UDPSocket::ReceiveThreadFunc() {
    // Buffer for receiving data
    std::vector<uint8_t> buffer(MAX_MESSAGE_SIZE);

    while (m_isReceiving) {
        sockaddr_in senderAddr;
        int senderAddrSize = sizeof(senderAddr);

        // Receive data
        int result = recvfrom(m_socket, (char*)buffer.data(), static_cast<int>(buffer.size()), 0,
            (sockaddr*)&senderAddr, &senderAddrSize);

        if (result > 0) {
            // Create a copy of the received data
            std::vector<uint8_t> receivedData(buffer.begin(), buffer.begin() + result);

            // Get sender information
            char senderIP[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &senderAddr.sin_addr, senderIP, sizeof(senderIP));

            NetworkEndpoint sender;
            sender.address = senderIP;
            sender.port = ntohs(senderAddr.sin_port);

            // Process the message
            ProcessReceivedMessage(receivedData, sender);
        }
        else if (result == SOCKET_ERROR) {
            int error = WSAGetLastError();
            if (error != WSAEWOULDBLOCK && error != WSAECONNRESET) {
                std::cerr << "recvfrom failed: " << error << std::endl;
                break;
            }
        }

        // Sleep a little to avoid high CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void UDPSocket::ProcessReceivedMessage(const std::vector<uint8_t>& buffer, const NetworkEndpoint& sender) {
    if (m_callback) {
        // Deserialize the message
        NetworkMessage* message = NetworkMessage::Deserialize(buffer);

        if (message) {
            // Call the callback
            m_callback(message, sender);

            // Cleanup
            delete message;
        }
    }
}

unsigned short UDPSocket::GetLocalPort() const {
    if (m_socket == INVALID_SOCKET) {
        return 0;
    }

    sockaddr_in addr;
    int addrLen = sizeof(addr);

    if (getsockname(m_socket, (sockaddr*)&addr, &addrLen) == SOCKET_ERROR) {
        std::cerr << "getsockname failed: " << WSAGetLastError() << std::endl;
        return 0;
    }

    return ntohs(addr.sin_port);
}