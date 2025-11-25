/* Start Header
*****************************************************************/
/*!
\file server.cpp
\author Bryan Ang Wei Ze (bryanweize.ang\digipen.edu)
\par Assignment 2
\date 23 Feb 2025
\brief
This file implements the server file which will be used to implement a
echo server.
Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

/*******************************************************************************
 * A simple TCP/IP server application
 ******************************************************************************/

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include "Windows.h"		// Entire Win32 API...
#include "winsock2.h"		// ...or Winsock alone
#include "ws2tcpip.h"		// getaddrinfo()

 // Tell the Visual Studio linker to include the following library in linking.
 // Alternatively, we could add this file to the linker command-line parameters,
 // but including it in the source code simplifies the configuration.
#pragma comment(lib, "Ws2_32.lib")
#include <cstdio>
#include <iostream>			   // cout, cerr
#include <string>			     // string
#include <vector>
#include <mutex>
#include <thread>
#include <queue>
#include <list>
#include <map>
#include <condition_variable>
#include <optional>


// Mutex for stdout synchronization
static std::mutex stdoutMutex;

// TaskQueue implementation
template <typename TItem, typename TAction, typename TOnDisconnect>
class TaskQueue{
public:
    TaskQueue(size_t workerCount, size_t slotCount, TAction& action, TOnDisconnect& disconnect) :
        _slotCount{ slotCount },
        _itemCount{ 0 },
        _onDisconnect{ disconnect },
        _stay{ true }
    {
        for (size_t i = 0; i < workerCount; ++i) {
            _workers.emplace_back(&work, std::ref(*this), std::ref(action));
        }
    }

    ~TaskQueue() {
        disconnect();
        for (std::thread& worker : _workers) {
            worker.join();
        }
    }

    void produce(TItem item) {
        // Wait for an available slot
        {
            std::unique_lock<std::mutex> slotCountLock{ _slotCountMutex };
            _producers.wait(slotCountLock, [&]() { return _slotCount > 0; });
            --_slotCount;
        }

        // Add item to buffer
        {
            std::lock_guard<std::mutex> bufferLock{ _bufferMutex };
            _buffer.push(item);
        }

        // Notify consumers
        {
            std::lock_guard<std::mutex> itemCountLock(_itemCountMutex);
            ++_itemCount;
            _consumers.notify_one();
        }
    }

    std::optional<TItem> consume() {
        std::optional<TItem> result = std::nullopt;

        // Wait for available item
        {
            std::unique_lock<std::mutex> itemCountLock(_itemCountMutex);
            _consumers.wait(itemCountLock, [&]() { return (_itemCount > 0) || (!_stay); });
            if (_itemCount == 0) {
                _consumers.notify_one();
                return result;
            }
            --_itemCount;
        }

        // Get item from buffer
        {
            std::lock_guard<std::mutex> bufferLock(_bufferMutex);
            result = _buffer.front();
            _buffer.pop();
        }

        // Notify producers
        {
            std::lock_guard<std::mutex> slotCountLock(_slotCountMutex);
            ++_slotCount;
            _producers.notify_one();
        }

        return result;
    }

    TaskQueue() = delete;
    TaskQueue(const TaskQueue&) = delete;
    TaskQueue(TaskQueue&&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;
    TaskQueue& operator=(TaskQueue&&) = delete;

private:
    static void work(TaskQueue<TItem, TAction, TOnDisconnect>& tq, TAction& action) {
        while (true) {
            {
                std::lock_guard<std::mutex> usersLock{ stdoutMutex };
                std::cout << "Thread [" << std::this_thread::get_id()
                    << "] is waiting for a task." << std::endl;
            }

            std::optional<TItem> item = tq.consume();
            if (!item) {
                break;
            }

            {
                std::lock_guard<std::mutex> usersLock{ stdoutMutex };
                std::cout << "Thread [" << std::this_thread::get_id()
                    << "] is executing a task." << std::endl;
            }

            if (!action(*item)) {
                tq.disconnect();
            }
        }

        {
            std::lock_guard<std::mutex> usersLock{ stdoutMutex };
            std::cout << "Thread [" << std::this_thread::get_id()
                << "] is exiting." << std::endl;
        }
    }

    void disconnect() {
        _stay = false;
        _onDisconnect();
    }

    // Worker threads
    std::vector<std::thread> _workers;

    // Buffer for items
    std::mutex _bufferMutex;
    std::queue<TItem> _buffer;

    // Slot management
    std::mutex _slotCountMutex;
    size_t _slotCount;
    std::condition_variable _producers;

    // Item management
    std::mutex _itemCountMutex;
    size_t _itemCount;
    std::condition_variable _consumers;

    volatile bool _stay;
    TOnDisconnect& _onDisconnect;
};

// Command IDs as specified in requirements
enum CMDID {
    UNKNOWN = (unsigned char)0x0,
    REQ_QUIT = (unsigned char)0x1,
    REQ_ECHO = (unsigned char)0x2,
    RSP_ECHO = (unsigned char)0x3,
    REQ_LISTUSERS = (unsigned char)0x4,
    RSP_LISTUSERS = (unsigned char)0x5,
    CMD_TEST = (unsigned char)0x20,
    ECHO_ERROR = (unsigned char)0x30
};

// Structure to hold client information
struct ClientInfo {
    SOCKET socket;
    uint32_t ip;
    uint16_t port;
};

// Global client tracking
static std::map<SOCKET, ClientInfo> g_clients;
static std::mutex g_clientsMutex;

// Helper function to send data reliably
bool sendData(SOCKET socket, const char* data, int length) {
    int totalSent = 0;
    while (totalSent < length) {
        int result = send(socket, data + totalSent, length - totalSent, 0);
        if (result == SOCKET_ERROR) {
            if (WSAGetLastError() == WSAEWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            return false;
        }
        totalSent += result;
    }
    return true;
}

// Helper function to find client by IP and port
SOCKET findClientByAddress(uint32_t targetIP, uint16_t targetPort) {
    std::lock_guard<std::mutex> lock(g_clientsMutex);

    char targetIPStr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &targetIP, targetIPStr, INET_ADDRSTRLEN);

    /*{
        std::lock_guard<std::mutex> printLock(stdoutMutex);
        std::cout << "\n===== DEBUG: SEARCHING FOR CLIENT =====" << std::endl;
        std::cout << "Target IP: " << targetIPStr << " (" << std::hex << targetIP << ")" << std::endl;
        std::cout << "Target Port: " << std::dec << ntohs(targetPort) << " (" << std::hex << targetPort << ")" << std::endl;
    }*/

    for (const auto& pair : g_clients) {
        char storedIPStr[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &pair.second.ip, storedIPStr, INET_ADDRSTRLEN);

        /*{
            std::lock_guard<std::mutex> printLock(stdoutMutex);
            std::cout << "\nComparing with stored client:" << std::endl;
            std::cout << "Stored IP: " << storedIPStr << " (" << std::hex << pair.second.ip << ")" << std::endl;
            std::cout << "Stored Port: " << std::dec << ntohs(pair.second.port)
                << " (" << std::hex << pair.second.port << ")" << std::endl;
        }*/

        if (pair.second.ip == targetIP && pair.second.port == targetPort) {
            /*std::lock_guard<std::mutex> printLock(stdoutMutex);
            std::cout << "===> MATCH FOUND! <==" << std::endl;
            std::cout << "=====================================" << std::endl;*/
            return pair.first;
        }
    }

    /*{
        std::lock_guard<std::mutex> printLock(stdoutMutex);
        std::cout << "===> NO MATCH FOUND <==" << std::endl;
        std::cout << "=====================================" << std::endl;
    }*/
    return INVALID_SOCKET;
}

// Function to handle REQ_ECHO command
bool handleEchoRequest(SOCKET sourceSocket, const char* message, int length) {
    if (length < 11) return false; // Minimum message size check

    // Extract destination info
    uint32_t destIP;
    uint16_t destPort;
    memcpy(&destIP, message + 1, 4);    // Skip command ID
    memcpy(&destPort, message + 5, 2);

    char destIPStr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &destIP, destIPStr, INET_ADDRSTRLEN);
    /*{
        std::lock_guard<std::mutex> lock(stdoutMutex);
        std::cout << "\n===== DEBUG: ECHO REQUEST =====" << std::endl;
        std::cout << "Target IP: " << destIPStr << " (" << std::hex << destIP << ")" << std::endl;
        std::cout << "Target Port: " << std::dec << ntohs(destPort)
            << " (" << std::hex << destPort << ")" << std::dec << std::endl;
    }*/

    // Find destination client
    SOCKET destSocket = findClientByAddress(destIP, destPort);
    if (destSocket == INVALID_SOCKET) {
        // Send error response if destination not found
        /*{
            std::lock_guard<std::mutex> lock(stdoutMutex);
            std::cout << "Sending ECHO_ERROR response" << std::endl;
        }*/
        char errorMsg = ECHO_ERROR;
        return sendData(sourceSocket, &errorMsg, 1);
    }

    // Get source client info
    ClientInfo sourceInfo;
    {
        std::lock_guard<std::mutex> lock(g_clientsMutex);
        sourceInfo = g_clients[sourceSocket];
    }

    // Print received message
    {
        std::lock_guard<std::mutex> lock(stdoutMutex);
        std::cout << "==========RECV START==========" << std::endl;
        std::cout << destIPStr << ":" << ntohs(destPort) << std::endl;

        uint32_t textLen;
        memcpy(&textLen, message + 7, 4);
        textLen = ntohl(textLen);
        std::cout.write(message + 11, textLen);
        std::cout << std::endl;
        std::cout << "==========RECV END==========" << std::endl;
    }

    // Forward message with source info
    std::vector<char> forwardMsg(length);
    forwardMsg[0] = REQ_ECHO;
    memcpy(&forwardMsg[1], &sourceInfo.ip, 4);
    memcpy(&forwardMsg[5], &sourceInfo.port, 2);
    memcpy(&forwardMsg[7], message + 7, length - 7);

    if (!sendData(destSocket, forwardMsg.data(), length)) {
        return false;
    }

    // If this is a self-echo (source and destination are the same),
    // we need to handle the response immediately
    if (sourceSocket == destSocket) {
        /*{
            std::lock_guard<std::mutex> lock(stdoutMutex);
            std::cout << "\n===== DEBUG: SELF ECHO DETECTED =====" << std::endl;
            std::cout << "Sending immediate RSP_ECHO" << std::endl;
        }*/
        // Create echo response
        std::vector<char> responseMsg(length);
        responseMsg[0] = RSP_ECHO;
        memcpy(&responseMsg[1], &sourceInfo.ip, 4);
        memcpy(&responseMsg[5], &sourceInfo.port, 2);
        memcpy(&responseMsg[7], message + 7, length - 7);

        return sendData(sourceSocket, responseMsg.data(), length);
    }

    return true;
}

// Function to handle RSP_ECHO command
bool handleEchoResponse(SOCKET sourceSocket, const char* message, int length) {
    if (length < 11) return false;

    uint32_t destIP;
    uint16_t destPort;
    memcpy(&destIP, message + 1, 4);
    memcpy(&destPort, message + 5, 2);

    /*{
        std::lock_guard<std::mutex> lock(stdoutMutex);
        char destIPStr[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &destIP, destIPStr, INET_ADDRSTRLEN);
        std::cout << "\n===== DEBUG: ECHO RESPONSE =====" << std::endl;
        std::cout << "Target IP: " << destIPStr << " (" << std::hex << destIP << ")" << std::endl;
        std::cout << "Target Port: " << std::dec << ntohs(destPort)
            << " (" << std::hex << destPort << ")" << std::dec << std::endl;
    }*/

    // Find destination client
    SOCKET destSocket = findClientByAddress(destIP, destPort);
    if (destSocket == INVALID_SOCKET) return false;

    // Get source client info
    ClientInfo sourceInfo;
    {
        std::lock_guard<std::mutex> lock(g_clientsMutex);
        sourceInfo = g_clients[sourceSocket];
    }

    // Create response message
    std::vector<char> responseMsg(length);
    responseMsg[0] = RSP_ECHO;
    memcpy(&responseMsg[1], &sourceInfo.ip, 4);
    memcpy(&responseMsg[5], &sourceInfo.port, 2);
    memcpy(&responseMsg[7], message + 7, length - 7);

    return sendData(destSocket, responseMsg.data(), length);
}

// Function to handle REQ_LISTUSERS command
bool handleListUsers(SOCKET socket) {
    std::vector<char> response;
    response.push_back(RSP_LISTUSERS);

    /*{
        std::lock_guard<std::mutex> lock(stdoutMutex);
        std::cout << "\n===== DEBUG: LIST USERS REQUEST =====" << std::endl;
        std::cout << "Number of clients: " << g_clients.size() << std::endl;
    }*/

    std::lock_guard<std::mutex> lock(g_clientsMutex);
    uint16_t userCount = htons((uint16_t)g_clients.size());

    // Add user count
    response.insert(response.end(),
        reinterpret_cast<char*>(&userCount),
        reinterpret_cast<char*>(&userCount) + sizeof(userCount));

    // Add each user's info
    for (const auto& client : g_clients) {
        // Add IP and port
        response.insert(response.end(),
            reinterpret_cast<const char*>(&client.second.ip),
            reinterpret_cast<const char*>(&client.second.ip) + 4);
        response.insert(response.end(),
            reinterpret_cast<const char*>(&client.second.port),
            reinterpret_cast<const char*>(&client.second.port) + 2);
    }

    return sendData(socket, response.data(), static_cast<int>(response.size()));
}

// Main client processing function
bool processClient(SOCKET clientSocket) {
    char recvBuf[0x10000] = { 0 }; // 64KB buffer
    int bytesReceived = 0;
    bool isFirstByte = true;
    unsigned char cmdId = UNKNOWN;
    int expectedLength = 0;

    // Set socket to non-blocking mode
    u_long mode = 1;
    ioctlsocket(clientSocket, FIONBIO, &mode);

    while (true) {
        if (isFirstByte) {
            bytesReceived = 0;
        }

        int result = recv(clientSocket, recvBuf + bytesReceived,
            sizeof(recvBuf) - bytesReceived, 0);

        if (result == SOCKET_ERROR) {
            if (WSAGetLastError() == WSAEWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            break;
        }

        if (result == 0) {
            break;
        }

        bytesReceived += result;

        // Process first byte to determine command
        if (isFirstByte && bytesReceived >= 1) {
            cmdId = (unsigned char)recvBuf[0];
            isFirstByte = false;

            /*{
                std::lock_guard<std::mutex> lock(stdoutMutex);
                std::cout << "\n===== DEBUG: RECEIVED COMMAND =====" << std::endl;
                std::cout << "Command ID: 0x" << std::hex << (int)cmdId << std::dec << std::endl;
            }*/

            switch (cmdId) {
            case REQ_QUIT:
                return false;
            case REQ_ECHO:
            case RSP_ECHO:
                expectedLength = 11; // Command + IP + Port + Length
                if (bytesReceived >= 11) {
                    uint32_t textLen;
                    memcpy(&textLen, recvBuf + 7, 4);
                    textLen = ntohl(textLen);
                    expectedLength += textLen;
                }
                break;
            case REQ_LISTUSERS:
                expectedLength = 1;
                break;
            default:
                {
                    std::lock_guard<std::mutex> lock(stdoutMutex);
                    std::cout << "Error: invalid command" << std::endl;
                }
                return false;
            }
        }

        // Process complete message
        if (bytesReceived >= expectedLength) {
            bool success = true;
            switch (cmdId) {
            case REQ_ECHO:
                success = handleEchoRequest(clientSocket, recvBuf, expectedLength);
                break;
            case RSP_ECHO:
                success = handleEchoResponse(clientSocket, recvBuf, expectedLength);
                break;
            case REQ_LISTUSERS:
                success = handleListUsers(clientSocket);
                break;
            }

            if (!success) {
                return false;
            }

            // Reset for next message
            isFirstByte = true;
            cmdId = UNKNOWN;
            bytesReceived = 0;
            expectedLength = 0;
            memset(recvBuf, 0, sizeof(recvBuf));
        }
    }

    return true;
}

int main() {
    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup() failed." << std::endl;
        return 1;
    }

    // Get port number
    std::string portStr;
    {
        std::lock_guard<std::mutex> lock(stdoutMutex);
        std::cout << "Server Port Number: ";
    }
    std::getline(std::cin, portStr);

    // Setup address info
    addrinfo hints = { 0 };
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    // Get local host name
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    // Get address info
    addrinfo* result = nullptr;
    if (getaddrinfo(hostname, portStr.c_str(), &hints, &result) != 0) {
        std::cerr << "getaddrinfo() failed." << std::endl;
        WSACleanup();
        return 1;
    }

    // Create socket
    SOCKET listenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (listenSocket == INVALID_SOCKET) {
        std::cerr << "socket() failed." << std::endl;
        freeaddrinfo(result);
        WSACleanup();
        return 1;
    }

    // Bind socket
    if (bind(listenSocket, result->ai_addr, (int)result->ai_addrlen) == SOCKET_ERROR) {
        std::cerr << "bind() failed." << std::endl;
        closesocket(listenSocket);
        freeaddrinfo(result);
        WSACleanup();
        return 1;
    }

    // Display server info
    {
        std::lock_guard<std::mutex> lock(stdoutMutex);
        char ipStr[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &((sockaddr_in*)result->ai_addr)->sin_addr, ipStr, INET_ADDRSTRLEN);
        std::cout << "\nServer IP Address: " << ipStr << std::endl;
        std::cout << "Server Port Number: " << portStr << std::endl;
    }

    freeaddrinfo(result);

    // Listen for connections
    if (listen(listenSocket, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "listen() failed." << std::endl;
        closesocket(listenSocket);
        WSACleanup();
        return 1;
    }

    // Setup task queue
    auto processClientAction = [](SOCKET s) -> bool {
        
        bool result = processClient(s);
        {
            std::lock_guard<std::mutex> lock(g_clientsMutex);
            g_clients.erase(s);

            /*std::lock_guard<std::mutex> printLock(stdoutMutex);
            std::cout << "\n===== DEBUG: CLIENT DISCONNECTED =====" << std::endl;
            std::cout << "Socket: " << s << std::endl;
            std::cout << "Remaining clients: " << g_clients.size() << std::endl;*/
        }
        closesocket(s);
        return result;
    };

    auto onDisconnect = []() {
        std::lock_guard<std::mutex> lock(stdoutMutex);
        std::cout << "Client disconnected" << std::endl;
    };

    TaskQueue<SOCKET, decltype(processClientAction), decltype(onDisconnect)>
        taskQueue(10, 20, processClientAction, onDisconnect);

    // Main server loop
    while (true) {
        sockaddr_in clientAddr;
        int clientAddrLen = sizeof(clientAddr);

        // Accept new connection
        SOCKET clientSocket = accept(listenSocket, (sockaddr*)&clientAddr, &clientAddrLen);
        if (clientSocket == INVALID_SOCKET) {
            break;
        }

        // Store client information
        ClientInfo client;
        client.socket = clientSocket;
        client.ip = clientAddr.sin_addr.s_addr;
        client.port = clientAddr.sin_port;

        /*{
            std::lock_guard<std::mutex> lock(g_clientsMutex);
            g_clients[clientSocket] = client;
        }*/

        // Print client connection info
        {
            std::lock_guard<std::mutex> lock(g_clientsMutex);
            g_clients[clientSocket] = client;

            std::lock_guard<std::mutex> printLock(stdoutMutex);
            char clientIP[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &clientAddr.sin_addr, clientIP, INET_ADDRSTRLEN);
            //std::cout << "\n===== DEBUG: NEW CLIENT CONNECTED =====" << std::endl;
            std::cout << "\nClient IP Address: " << clientIP
                << " (0x" << std::hex << ntohl(client.ip) << ")" << std::endl;
            std::cout << "Client Port Number: " << std::dec << ntohs(clientAddr.sin_port)
                << " (0x" << std::hex << ntohs(client.port) << ")" << std::dec << std::endl;
            //std::cout << "Socket: " << clientSocket << std::endl;
            //std::cout << "Total clients: " << g_clients.size() << std::endl;
            //std::cout << "=======================================" << std::endl;
        }

        // Add client socket to task queue
        taskQueue.produce(clientSocket);
    }

    // Cleanup
    closesocket(listenSocket);
    WSACleanup();
    return 0;
}