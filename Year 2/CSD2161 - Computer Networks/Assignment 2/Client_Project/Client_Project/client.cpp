/* Start Header
*****************************************************************/
/*!
\file client.cpp
\author Bryan Ang Wei Ze (bryanweize.ang\digipen.edu)
\par Assignment 2
\date 23 Feb 2025
\brief
This file implements the client file which will be used to implement a 
echo client.
Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/


/*******************************************************************************
 * A simple TCP/IP client application
 ******************************************************************************/

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif


#include "Windows.h"		// Entire Win32 API...
#include "winsock2.h"		// ...or Winsock alone
#include "ws2tcpip.h"		// getaddrinfo()

// Tell the Visual Studio linker to include the following library in linking.
// Alternatively, we could add this file to the linker command-line parameters,
// but including it in the source code simplifies the configuration.
#pragma comment(lib, "ws2_32.lib")

#include <iostream>			// cout, cerr
#include <string>			// string
#include <thread>
#include <mutex>
#include <csignal>
#include <vector>
#include <sstream>
#include <atomic>
#include <iomanip>

// Global stdout mutex for synchronized printing
static std::mutex stdoutMutex;
static std::atomic<bool> isRunning{ true };

// Message Command IDs
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

// Function to process incoming messages
void receiveMessages(SOCKET socket) {
    char buffer[0x10000] = { 0 }; // 64KB buffer
    int bytesReceived = 0;
    bool isFirstByte = true;
    unsigned char cmdId = UNKNOWN;
    int expectedLength = 0;

    while (isRunning) {
        if (isFirstByte) {
            bytesReceived = 0;
        }

        int result = recv(socket, buffer + bytesReceived,
            sizeof(buffer) - bytesReceived, 0);

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
            cmdId = (unsigned char)buffer[0];
            isFirstByte = false;

            /*{
                std::lock_guard<std::mutex> lock(stdoutMutex);
                std::cout << "\n===== DEBUG: RECEIVED COMMAND =====" << std::endl;
                std::cout << "Command ID: 0x" << std::hex << (int)cmdId << std::dec << std::endl;
            }*/

            switch (cmdId) {
            case REQ_ECHO:
            case RSP_ECHO:
                expectedLength = 11; // Command + IP + Port + Length
                if (bytesReceived >= 11) {
                    uint32_t textLen;
                    memcpy(&textLen, buffer + 7, 4);
                    textLen = ntohl(textLen);
                    expectedLength += textLen;
                }
                break;
            case RSP_LISTUSERS:
                if (bytesReceived >= 3) {
                    uint16_t numUsers;
                    memcpy(&numUsers, buffer + 1, 2);
                    numUsers = ntohs(numUsers);
                    expectedLength = 3 + (numUsers * 6); // Command + NumUsers + (IP+Port)*NumUsers
                }
                else {
                    expectedLength = 3; // Wait for header
                }
                break;
            case ECHO_ERROR:
                expectedLength = 1;
                break;
            default:
                isFirstByte = true;
                bytesReceived = 0;
                continue;
            }
        }

        // Process complete message
        if (bytesReceived >= expectedLength) {
            std::lock_guard<std::mutex> lock(stdoutMutex);
            std::cout << "==========RECV START==========" << std::endl;

            switch (cmdId) {
            case REQ_ECHO:
            case RSP_ECHO: {
                uint32_t ip;
                uint16_t port;
                uint32_t textLen;

                memcpy(&ip, buffer + 1, 4);
                memcpy(&port, buffer + 5, 2);
                memcpy(&textLen, buffer + 7, 4);
                textLen = ntohl(textLen);

                // Print source info
                char ipStr[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &ip, ipStr, INET_ADDRSTRLEN);

                /*std::cout << "\n===== DEBUG: ECHO MESSAGE DETAILS =====" << std::endl;
                std::cout << "Source IP: " << ipStr << " (0x" << std::hex << ntohl(ip) << ")" << std::endl;
                std::cout << "Source Port: " << std::dec << ntohs(port)
                    << " (0x" << std::hex << ntohs(port) << ")" << std::dec << std::endl;
                std::cout << "Message Length: " << textLen << std::endl;*/

                std::cout << ipStr << ":" << ntohs(port) << std::endl;
                std::cout.write(buffer + 11, textLen);
                std::cout << std::endl;
                break;
            }

            case RSP_LISTUSERS: {
                uint16_t numUsers;
                memcpy(&numUsers, buffer + 1, 2);
                numUsers = ntohs(numUsers);

                //std::cout << "\n===== DEBUG: USER LIST RESPONSE =====" << std::endl;
                //std::cout << "Number of users: " << numUsers << std::endl;

                std::cout << "Users:" << std::endl;
                int offset = 3;

                for (int i = 0; i < numUsers; ++i) {
                    uint32_t userIP;
                    uint16_t userPort;
                    memcpy(&userIP, buffer + offset, 4);
                    memcpy(&userPort, buffer + offset + 4, 2);

                    char ipStr[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &userIP, ipStr, INET_ADDRSTRLEN);
                    std::cout << ipStr << ":" << ntohs(userPort) << std::endl;

                    /*std::cout << "User " << i + 1 << " details:" << std::endl;
                    std::cout << "IP: 0x" << std::hex << ntohl(userIP) << std::endl;
                    std::cout << "Port: 0x" << ntohs(userPort) << std::dec << std::endl;*/

                    offset += 6;
                }
                break;
            }

            case ECHO_ERROR:
                std::cout << "Echo error" << std::endl;
                break;
            }

            std::cout << "==========RECV END==========" << std::endl;

            // Reset for next message
            isFirstByte = true;
            cmdId = UNKNOWN;
            bytesReceived = 0;
            expectedLength = 0;
            memset(buffer, 0, sizeof(buffer));
        }
    }

    std::lock_guard<std::mutex> lock(stdoutMutex);
    std::cout << "disconnection..." << std::endl;
}

int main() {
    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed." << std::endl;
        return 1;
    }

    // Create socket
    SOCKET clientSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (clientSocket == INVALID_SOCKET) {
        std::cerr << "Socket creation failed." << std::endl;
        WSACleanup();
        return 1;
    }

    // Get server details
    std::string serverIP;
    std::string portStr;

    {
        std::lock_guard<std::mutex> lock(stdoutMutex);
        std::cout << "Server IP Address: ";
    }
    std::getline(std::cin, serverIP);

    {
        std::lock_guard<std::mutex> lock(stdoutMutex);
        std::cout << "Server Port Number: ";
    }
    std::getline(std::cin, portStr);

    // Connect to server
    sockaddr_in serverAddr = { 0 };
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(std::stoi(portStr));
    inet_pton(AF_INET, serverIP.c_str(), &serverAddr.sin_addr);

    // Store our local address info for later use
    sockaddr_in localAddr = { 0 };
    int localAddrLen = sizeof(localAddr);

    if (connect(clientSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        std::cerr << "Connection failed." << std::endl;
        closesocket(clientSocket);
        WSACleanup();
        return 1;
    }

    // Get our local address after connection
    /*if (getsockname(clientSocket, (sockaddr*)&localAddr, &localAddrLen) == 0) {
        std::lock_guard<std::mutex> lock(stdoutMutex);
        char localIP[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &localAddr.sin_addr, localIP, INET_ADDRSTRLEN);
        std::cout << "\n===== DEBUG: LOCAL CONNECTION INFO =====" << std::endl;
        std::cout << "Local IP: " << localIP << " (0x" << std::hex << ntohl(localAddr.sin_addr.s_addr) << ")" << std::endl;
        std::cout << "Local Port: " << std::dec << ntohs(localAddr.sin_port)
            << " (0x" << std::hex << ntohs(localAddr.sin_port) << ")" << std::dec << std::endl;
    }*/

    // Set socket to non-blocking mode
    u_long mode = 1;
    ioctlsocket(clientSocket, FIONBIO, &mode);

    // Start receive thread
    std::thread recvThread(receiveMessages, clientSocket);

    // Main input loop
    std::string input;
    while (std::getline(std::cin, input) && isRunning) {
#ifdef DEBUG_ASSIGNTMENT2_TEST
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(5000ms);
#endif

        if (input.empty()) continue;

        if (input[0] == '/') {
            switch (input[1]) {
            case 't': {  // Test command
                if (input.length() < 3) continue;
                // Skip "/t " (3 characters) and then parse the hex string
                std::string hexStr = input.substr(3);
                // Remove any whitespace from the hex string
                hexStr.erase(std::remove_if(hexStr.begin(), hexStr.end(), ::isspace), hexStr.end());

                // Check if we have valid hex string length (must be even)
                if (hexStr.empty() || hexStr.length() % 2 != 0) {
                    std::cerr << "Invalid hex string length\n";
                    continue;
                }

                // Parse each pair of hex digits into bytes
                std::vector<char> data;
                for (size_t i = 0; i < hexStr.length(); i += 2) {
                    // Get two hex digits
                    std::string byteStr = hexStr.substr(i, 2);
                    // Convert to byte value
                    char byte = static_cast<char>(std::stoi(byteStr, nullptr, 16));
                    data.push_back(byte);
                }

                // Send the raw bytes directly to server
                if (!sendData(clientSocket, data.data(), static_cast<int>(data.size()))) {
                    std::cerr << "Failed to send test data\n";
                    continue;
                }
                break;
            }

            case 'q': {  // Quit command
                /*{
                    std::lock_guard<std::mutex> lock(stdoutMutex);
                    std::cout << "\n===== DEBUG: SENDING QUIT COMMAND =====" << std::endl;
                }*/
                char cmd = REQ_QUIT;
                sendData(clientSocket, &cmd, 1);
                isRunning = false;
                break;
            }

            case 'l': {  // List users command
                /*{
                    std::lock_guard<std::mutex> lock(stdoutMutex);
                    std::cout << "\n===== DEBUG: SENDING LIST USERS COMMAND =====" << std::endl;
                }*/
                char cmd = REQ_LISTUSERS;
                sendData(clientSocket, &cmd, 1);
                break;
            }

            case 'e': {  // Echo command
                if (input.length() < 4) continue;

                std::string params = input.substr(3);
                size_t colonPos = params.find(':');
                size_t spacePos = params.find(' ', colonPos);

                if (colonPos == std::string::npos || spacePos == std::string::npos)
                    continue;

                std::string ipStr = params.substr(0, colonPos).c_str();  // Remove leading space if any
                if (ipStr[0] == ' ') ipStr = ipStr.substr(1);  // Remove leading space if present
                
                std::string portStr = params.substr(colonPos + 1, spacePos - colonPos - 1);
                std::string message = params.substr(spacePos + 1);

                // Debug print the raw IP string before conversion
                /*{
                    std::lock_guard<std::mutex> lock(stdoutMutex);
                    std::cout << "\n===== DEBUG: PARSING IP =====" << std::endl;
                    std::cout << "Raw IP string: '" << ipStr << "'" << std::endl;
                }*/

                uint32_t ip;
                if (inet_pton(AF_INET, ipStr.c_str(), &ip) != 1) {
                    std::cout << "Invalid IP address format" << std::endl;
                    continue;
                }

                uint16_t port = htons(static_cast<uint16_t>(std::stoi(portStr)));
                uint32_t msgLen = htonl(static_cast<u_long>(message.length()));

                {
                    std::lock_guard<std::mutex> lock(stdoutMutex);
                    std::cout << "==========RECV START==========" << std::endl;
                    char parsedIp[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &ip, parsedIp, INET_ADDRSTRLEN);
                    std::cout << parsedIp << ":" << ntohs(port) << std::endl;
                    std::cout << message << std::endl;
                    std::cout << "==========RECV END==========" << std::endl;
                }

                // Create message
                std::vector<char> msgData(11 + message.length());
                msgData[0] = REQ_ECHO;
                memcpy(&msgData[1], &ip, 4);
                memcpy(&msgData[5], &port, 2);
                memcpy(&msgData[7], &msgLen, 4);
                memcpy(&msgData[11], message.c_str(), message.length());

                sendData(clientSocket, msgData.data(), static_cast<int>(msgData.size()));
                break;
            }
            }
        }

#ifdef DEBUG_ASSIGNTMENT2_TEST
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(5000ms);
#endif
    }

    // Cleanup
    shutdown(clientSocket, SD_SEND);
    closesocket(clientSocket);

    if (recvThread.joinable())
        recvThread.join();

    WSACleanup();
    return 0;
}