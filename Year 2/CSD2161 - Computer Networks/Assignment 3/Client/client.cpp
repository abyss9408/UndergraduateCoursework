/* Start Header
*****************************************************************/
/*!
\file client.cpp
\author: Bryan Ang Wei Ze, Tham Kang Ting, Low Yue Jun
\par: bryanweize.ang\@digipen.edu ,kangting.t\@digipen.edu, yuejun.low\@digipen.edu
\date 9 March 2025
\brief
This file implements the client for a file downloading system over UDP.
Based on the echo client from Assignment 3.
Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "Windows.h"
#include "winsock2.h"
#include "ws2tcpip.h"

#pragma comment(lib, "ws2_32.lib")

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <atomic>
#include <mutex>
#include <thread>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <map>
#include "../config.h"

// Globals
std::mutex printMutex;
std::atomic<bool> isRunning{ true };
std::string downloadPath;
std::string serverIPAddress;
uint16_t serverTCPPort = 0;
uint16_t serverUDPPort = 0;
uint16_t clientUDPPort = 0;
SOCKET udpSocket = INVALID_SOCKET;
std::map<std::string, std::string> downloadedFiles; // Maps session ID to filename

// Structure to track download sessions
struct DownloadSession
{
    uint32_t sessionId;
    std::string filename;
    uint32_t fileSize;
    uint32_t bytesReceived;
    std::ofstream fileStream;
    bool completed;
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point lastActivity;
    std::map<uint32_t, bool> receivedChunks; // Track which chunks have been received
};

std::map<uint32_t, DownloadSession> activeSessions;
std::mutex sessionsLock;

// Helper function for logging with timestamps
void LogMessage(const std::string &message)
{
    std::lock_guard<std::mutex> lock(printMutex);
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm tm_buf;
    localtime_s(&tm_buf, &now_time_t);

    std::cout << "["
        << std::put_time(&tm_buf, "%H:%M:%S")
        << "] " << message << std::endl;
}

// Helper function to format file size
std::string FormatFileSize(uint32_t sizeInBytes)
{
    if (sizeInBytes < 1024)
    {
        return std::to_string(sizeInBytes) + " bytes";
    }
    else if (sizeInBytes < 1024 * 1024)
    {
        return std::to_string(sizeInBytes / 1024) + " KB";
    }
    else
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << (sizeInBytes / (1024.0 * 1024.0)) << " MB";
        return ss.str();
    }
}

// Helper function to format download speed
std::string FormatSpeed(double bytesPerSecond)
{
    if (bytesPerSecond < 1024)
    {
        return std::to_string(static_cast<int>(bytesPerSecond)) + " B/s";
    }
    else if (bytesPerSecond < 1024 * 1024)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << (bytesPerSecond / 1024.0) << " KB/s";
        return ss.str();
    }
    else
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << (bytesPerSecond / (1024.0 * 1024.0)) << " MB/s";
        return ss.str();
    }
}

// Helper function to format elapsed time
std::string FormatTime(int64_t milliseconds)
{
    int seconds = static_cast<int>(milliseconds / 1000);
    int minutes = seconds / 60;
    seconds %= 60;

    std::stringstream ss;
    if (minutes > 0)
    {
        ss << minutes << "m ";
    }
    ss << seconds << "s";
    return ss.str();
}

// Helper function to parse IP:Port string
bool ParseIPPort(const std::string &ipPort, std::string &ip, uint16_t &port)
{
    size_t colonPos = ipPort.find(':');
    if (colonPos == std::string::npos) return false;

    ip = ipPort.substr(0, colonPos);
    try
    {
        port = static_cast<uint16_t>(std::stoul(ipPort.substr(colonPos + 1)));
        return true;
    }
    catch (...)
    {
        return false;
    }
}

// Helper function to format IP:Port
std::string FormatIPPort(const std::string &ip, uint16_t port)
{
    return ip + ":" + std::to_string(port);
}

// Helper function to send file listing request
void SendListFilesRequest(SOCKET sock)
{
    char cmd = REQ_LISTFILES;
    int result = send(sock, &cmd, 1, 0);
    if (result == SOCKET_ERROR)
    {
        LogMessage("Failed to send list files request: " + std::to_string(WSAGetLastError()));
    }
    else
    {
        LogMessage("Sent file listing request");
    }
}

// Helper function to create download request
std::vector<char> CreateDownloadRequest(const std::string &clientIP, uint16_t clientPort, const std::string &filename)
{
    std::vector<char> buffer;
    buffer.push_back(REQ_DOWNLOAD);

    // Add client IP address (4 bytes)
    in_addr addr;
    inet_pton(AF_INET, clientIP.c_str(), &addr);
    uint32_t netIP = addr.s_addr;
    buffer.insert(buffer.end(),
        reinterpret_cast<char *>(&netIP),
        reinterpret_cast<char *>(&netIP) + 4);

    // Add client port (2 bytes)
    uint16_t netPort = htons(clientPort);
    buffer.insert(buffer.end(),
        reinterpret_cast<char *>(&netPort),
        reinterpret_cast<char *>(&netPort) + 2);

    // Add filename length (4 bytes)
    uint32_t filenameLen = static_cast<uint32_t>(filename.length());
    uint32_t netLen = htonl(filenameLen);
    buffer.insert(buffer.end(),
        reinterpret_cast<char *>(&netLen),
        reinterpret_cast<char *>(&netLen) + 4);

    // Add filename
    buffer.insert(buffer.end(), filename.begin(), filename.end());

    return buffer;
}

// Helper function to send download request
bool SendDownloadRequest(SOCKET sock, const std::string &clientIP, uint16_t clientPort, const std::string &filename)
{
    auto buffer = CreateDownloadRequest(clientIP, clientPort, filename);
    int result = send(sock, buffer.data(), static_cast<int>(buffer.size()), 0);

    if (result == SOCKET_ERROR)
    {
        LogMessage("Failed to send download request: " + std::to_string(WSAGetLastError()));
        return false;
    }

    LogMessage("Sent download request for file: " + filename);
    return true;
}

// Helper function to create and initialize UDP socket
bool InitializeUDPSocket()
{
    udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSocket == INVALID_SOCKET)
    {
        LogMessage("Failed to create UDP socket. Error: " + std::to_string(WSAGetLastError()));
        return false;
    }

    // Set up the local address structure for binding
    sockaddr_in localAddr;
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = htons(clientUDPPort);

    // Bind the UDP socket to the specified client port
    if (bind(udpSocket, (sockaddr *)&localAddr, sizeof(localAddr)) == SOCKET_ERROR)
    {
        LogMessage("Failed to bind UDP socket. Error: " + std::to_string(WSAGetLastError()));
        closesocket(udpSocket);
        udpSocket = INVALID_SOCKET;
        return false;
    }

    // Set socket to non-blocking mode
    u_long mode = 1;
    ioctlsocket(udpSocket, FIONBIO, &mode);

    // Increase send buffer size
    int sendBufSize = 1024 * 1024;  // 1 MB
    if (setsockopt(udpSocket, SOL_SOCKET, SO_SNDBUF, (char*)&sendBufSize, sizeof(sendBufSize)) == SOCKET_ERROR) {
        LogMessage("Warning: Failed to set send buffer size: " + std::to_string(WSAGetLastError()));
        // Continue anyway - buffer size optimization is not critical
    }

    // Increase receive buffer size
    int recvBufSize = 1024 * 1024;  // 1 MB
    if (setsockopt(udpSocket, SOL_SOCKET, SO_RCVBUF, (char*)&recvBufSize, sizeof(recvBufSize)) == SOCKET_ERROR) {
        LogMessage("Warning: Failed to set receive buffer size: " + std::to_string(WSAGetLastError()));
        // Continue anyway - buffer size optimization is not critical
    }

    LogMessage("UDP socket initialized on port " + std::to_string(clientUDPPort));
    return true;
}

// Helper function to create an ACK packet
std::vector<char> CreateAckPacket(uint32_t sessionId, uint32_t seqNum)
{
    std::vector<char> packet(9); // 1 byte for flags + 4 bytes for session ID + 4 bytes for ACK number

    uint8_t flags = UDP_ACK;
    uint32_t netSessionId = htonl(sessionId);
    uint32_t netAckNum = htonl(seqNum);

    memcpy(packet.data(), &flags, 1);
    memcpy(packet.data() + 1, &netSessionId, 4);
    memcpy(packet.data() + 5, &netAckNum, 4);

    return packet;
}

// Helper function to send an ACK packet
void SendAck(uint32_t sessionId, uint32_t seqNum)
{
    auto packet = CreateAckPacket(sessionId, seqNum);

    // Set up the server address
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    inet_pton(AF_INET, serverIPAddress.c_str(), &(serverAddr.sin_addr));
    serverAddr.sin_port = htons(serverUDPPort);

    // Send the ACK
    int result = sendto(udpSocket, packet.data(), static_cast<int>(packet.size()), 0,
        (sockaddr *)&serverAddr, sizeof(serverAddr));

    if (result == SOCKET_ERROR)
    {
        LogMessage("Failed to send ACK: " + std::to_string(WSAGetLastError()));
    }
}

// Helper function to create a new download session
DownloadSession CreateDownloadSession(uint32_t sessionId, const std::string &filename, uint32_t fileSize)
{
    DownloadSession session;
    session.sessionId = sessionId;
    session.filename = filename;
    session.fileSize = fileSize;
    session.bytesReceived = 0;
    session.completed = false;
    session.startTime = std::chrono::steady_clock::now();
    session.lastActivity = session.startTime;

    return session;
}

// Helper function to process file list response
void ProcessFileListResponse(const std::vector<char> &buffer, size_t bufferSize)
{
    if (bufferSize < 7)
    {
        LogMessage("Invalid file list response: too short");
        return;
    }

    uint16_t numFiles;
    memcpy(&numFiles, buffer.data() + 1, 2);
    numFiles = ntohs(numFiles);

    uint32_t listLength;
    memcpy(&listLength, buffer.data() + 3, 4);
    listLength = ntohl(listLength);

    if (bufferSize < 7 + listLength)
    {
        LogMessage("Invalid file list response: incomplete data");
        return;
    }

    LogMessage("==========FILES AVAILABLE==========");
    LogMessage("Number of files: " + std::to_string(numFiles));

    int offset = 7;
    for (uint16_t i = 0; i < numFiles; ++i)
    {
        if (offset + 4 > bufferSize)
        {
            LogMessage("Invalid file list entry: incomplete length data");
            break;
        }

        // Get filename length
        uint32_t filenameLen;
        memcpy(&filenameLen, buffer.data() + offset, 4);
        filenameLen = ntohl(filenameLen);
        offset += 4;

        if (offset + filenameLen > bufferSize)
        {
            LogMessage("Invalid file list entry: incomplete filename data");
            break;
        }

        // Get filename
        std::string filename(buffer.data() + offset, filenameLen);
        offset += filenameLen;

        LogMessage(std::to_string(i + 1) + ". " + filename);
    }
    LogMessage("==================================");
}

// Helper function to process download response
bool ProcessDownloadResponse(const std::vector<char> &buffer, size_t bufferSize)
{
    if (bufferSize < 15)
    {
        LogMessage("Invalid download response: too short");
        return false;
    }

    // Parse server IP address
    in_addr serverAddr;
    memcpy(&serverAddr.s_addr, buffer.data() + 1, 4);
    char serverIP[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &serverAddr, serverIP, INET_ADDRSTRLEN);

    // Parse server port
    uint16_t serverPort;
    memcpy(&serverPort, buffer.data() + 5, 2);
    serverPort = ntohs(serverPort);

    // Parse session ID
    uint32_t sessionId;
    memcpy(&sessionId, buffer.data() + 7, 4);
    sessionId = ntohl(sessionId);

    // Parse file length
    uint32_t fileLength;
    memcpy(&fileLength, buffer.data() + 11, 4);
    fileLength = ntohl(fileLength);

    // Get the requested filename from our records
    std::string requestedFilename = "download_" + std::to_string(sessionId);

    // Try to find the original filename from pending downloads
    {
        std::lock_guard<std::mutex> lock(printMutex);

        // Look for any pending download and use it for this session
        std::string pendingKey;
        for (const auto& entry : downloadedFiles)
        {
            if (entry.first.find("pending_") == 0)
            {
                requestedFilename = entry.second;
                pendingKey = entry.first;
                break;
            }
        }

        // Map this sessionId to the filename and remove the pending entry
        if (!pendingKey.empty())
        {
            downloadedFiles[std::to_string(sessionId)] = requestedFilename;
            downloadedFiles.erase(pendingKey);
        }
    }

    // Initialize a new download session
    std::lock_guard<std::mutex> lock(sessionsLock);
    DownloadSession newSession = CreateDownloadSession(sessionId, requestedFilename, fileLength);

    // Create the output file
    std::filesystem::path outputFilePath = downloadPath;
    outputFilePath /= requestedFilename;

    newSession.fileStream.open(outputFilePath.string(), std::ios::binary);
    if (!newSession.fileStream)
    {
        LogMessage("Failed to create output file: " + outputFilePath.string());
        return false;
    }

    // Pre-allocate the file space
    newSession.fileStream.seekp(fileLength - 1);
    newSession.fileStream.put(0);
    newSession.fileStream.seekp(0);

    // Add session to active sessions - use std::move since std::ofstream is not copyable
    activeSessions.emplace(sessionId, std::move(newSession));

    LogMessage("Starting download from " + std::string(serverIP) + ":" + std::to_string(serverPort));
    LogMessage("Session ID: " + std::to_string(sessionId));
    LogMessage("File size: " + FormatFileSize(fileLength));
    LogMessage("Output file: " + outputFilePath.string());

    return true;
}

// Helper function to process UDP data packet
void ProcessDataPacket(const std::vector<char> &buffer, int bytesReceived)
{
    if (bytesReceived < 13)
    {
        LogMessage("Invalid data packet: too short");
        return;
    }

    uint32_t sessionId, fileOffset, dataLength;

    // Extract header information
    memcpy(&sessionId, buffer.data() + 1, 4);
    sessionId = ntohl(sessionId);

    memcpy(&fileOffset, buffer.data() + 5, 4);
    fileOffset = ntohl(fileOffset);

    memcpy(&dataLength, buffer.data() + 9, 4);
    dataLength = ntohl(dataLength);

    if (bytesReceived < 13 + dataLength)
    {
        LogMessage("Invalid data packet: incomplete data");
        return;
    }

    // Extract sequence number for ACK (using file offset)
    uint32_t seqNum = fileOffset / DATA_PACKET_SIZE;

    // Process the data in the session
    std::lock_guard<std::mutex> lock(sessionsLock);
    auto it = activeSessions.find(sessionId);
    if (it == activeSessions.end())
    {
        LogMessage("Received data for unknown session: " + std::to_string(sessionId));
        return;
    }

    DownloadSession &session = it->second;

    // Check if this is a duplicate packet
    if (session.receivedChunks.find(seqNum) != session.receivedChunks.end() &&
        session.receivedChunks[seqNum])
    {
        // Already received this chunk, just acknowledge it again
        SendAck(sessionId, seqNum);
        return;
    }

    // Update activity timestamp
    session.lastActivity = std::chrono::steady_clock::now();

    // Write data to file at correct position
    session.fileStream.seekp(fileOffset);
    session.fileStream.write(buffer.data() + 13, dataLength);

    // Mark this chunk as received
    session.receivedChunks[seqNum] = true;

    // Update bytes received
    session.bytesReceived += dataLength;

    // Calculate and display progress
    double progress = static_cast<double>(session.bytesReceived) / session.fileSize * 100.0;

    // Calculate speed
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        session.lastActivity - session.startTime).count();
    double speedBps = (duration > 0) ?
        (session.bytesReceived * 1000.0 / duration) : 0;

    // Only log progress periodically to avoid console spam
    static std::map<uint32_t, int> lastProgressPercent;
    int currentProgressPercent = static_cast<int>(progress);
    if (lastProgressPercent.find(sessionId) == lastProgressPercent.end() ||
        currentProgressPercent - lastProgressPercent[sessionId] >= 5 ||
        currentProgressPercent == 100)
    {

        lastProgressPercent[sessionId] = currentProgressPercent;

        LogMessage("Session " + std::to_string(sessionId) + ": " +
            std::to_string(currentProgressPercent) + "% complete, " +
            FormatSpeed(speedBps));
    }

    // Check if download is complete
    if (session.bytesReceived >= session.fileSize)
    {
        session.completed = true;
        session.fileStream.close();

        // Calculate download time and speed
        auto endTime = std::chrono::steady_clock::now();
        auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - session.startTime).count();
        double speedBps = (session.fileSize * 1000.0) / durationMs;

        LogMessage("Download completed: " + session.filename);
        LogMessage("Size: " + FormatFileSize(session.fileSize));
        LogMessage("Time: " + FormatTime(durationMs));
        LogMessage("Speed: " + FormatSpeed(speedBps));
    }

    // Send ACK for this packet
    SendAck(sessionId, seqNum);
}

// UDP Receiver thread function
void UDPReceiverThread()
{
    std::vector<char> buffer(MAX_PACKET_SIZE + 64);  // Extra space for headers
    sockaddr_in senderAddr;
    int senderAddrLen = sizeof(senderAddr);

    LogMessage("UDP receiver thread started");

    while (isRunning)
    {
        // Receive data from UDP socket
        int bytesReceived = recvfrom(udpSocket, buffer.data(), static_cast<int>(buffer.size()), 0,
            (sockaddr *)&senderAddr, &senderAddrLen);

        if (bytesReceived == SOCKET_ERROR)
        {
            if (WSAGetLastError() != WSAEWOULDBLOCK)
            {
                LogMessage("UDP receive error: " + std::to_string(WSAGetLastError()));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (bytesReceived < 1)
        {
            continue;
        }

        // Parse the packet header
        uint8_t flags = static_cast<uint8_t>(buffer[0]);

        // Check if it's a data packet (LSB should be 0)
        if ((flags & 0x01) == UDP_DATA)
        {
            ProcessDataPacket(buffer, bytesReceived);
        }
    }

    LogMessage("UDP receiver thread stopped");
}

// Session monitor thread to detect stalled downloads
void SessionMonitorThread()
{
    LogMessage("Session monitor thread started");

    while (isRunning)
    {
        std::this_thread::sleep_for(std::chrono::seconds(5));

        std::lock_guard<std::mutex> lock(sessionsLock);
        auto now = std::chrono::steady_clock::now();

        for (auto &pair : activeSessions)
        {
            if (pair.second.completed) continue;

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - pair.second.lastActivity).count();

            if (elapsed > 30000)
            { // 30 seconds inactivity
                LogMessage("Warning: Session " + std::to_string(pair.first) +
                    " inactive for " + std::to_string(elapsed / 1000) + " seconds");
            }
        }
    }

    LogMessage("Session monitor thread stopped");
}

// TCP Receiver thread function
void TCPReceiverThread(SOCKET sock)
{
    std::vector<char> buffer(TCP_BUFFER_SIZE);
    std::vector<char> messageBuffer;  // For handling partial messages

    LogMessage("TCP receiver thread started");

    while (isRunning)
    {
        int bytesReceived = recv(sock, buffer.data(), static_cast<int>(buffer.size()), 0);
        if (bytesReceived <= 0)
        {
            if (WSAGetLastError() != WSAEWOULDBLOCK)
            {
                LogMessage("TCP connection error: " + std::to_string(WSAGetLastError()));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Add received data to message buffer
        messageBuffer.insert(messageBuffer.end(), buffer.data(), buffer.data() + bytesReceived);

        // Process complete messages
        while (messageBuffer.size() >= 1)
        {
            uint8_t cmdId = static_cast<uint8_t>(messageBuffer[0]);
            int messageSize = 0;

            switch (cmdId)
            {
            case RSP_LISTFILES:
                ProcessFileListResponse(messageBuffer, messageBuffer.size());
                // Calculate message size
                if (messageBuffer.size() >= 7)
                {
                    uint32_t listLength;
                    memcpy(&listLength, messageBuffer.data() + 3, 4);
                    listLength = ntohl(listLength);
                    messageSize = 7 + listLength;
                }
                break;

            case RSP_DOWNLOAD:
                ProcessDownloadResponse(messageBuffer, messageBuffer.size());
                messageSize = 15; // Fixed size for download response
                break;

            case DOWNLOAD_ERROR:
                LogMessage("==========ERROR==========");
                LogMessage("Download error: File not found or inaccessible");
                LogMessage("==========================");
                messageSize = 1;
                break;

            default:
                LogMessage("Unknown command received: " + std::to_string(cmdId));
                messageSize = 1;
                break;
            }

            if (messageSize == 0 || messageSize > messageBuffer.size())
            {
                break;  // Incomplete message
            }

            // Remove processed message from buffer
            messageBuffer.erase(messageBuffer.begin(), messageBuffer.begin() + messageSize);
        }
    }

    LogMessage("TCP receiver thread stopped");
}

// Helper function to display usage instructions
void DisplayUsage()
{
    std::cout << "\nAvailable commands:" << std::endl;
    std::cout << "/l - List available files" << std::endl;
    std::cout << "/d IP:Port filename - Download a file" << std::endl;
    std::cout << "/q - Quit" << std::endl;
}

// Helper function to initialize Winsock
bool InitializeWinsock(WSADATA &wsaData)
{
    int errorCode = WSAStartup(MAKEWORD(WINSOCK_VERSION, WINSOCK_SUBVERSION), &wsaData);
    if (NO_ERROR != errorCode)
    {
        std::cerr << "WSAStartup() failed with error: " << errorCode << std::endl;
        return false;
    }
    return true;
}

// Helper function to resolve server address
bool ResolveServerAddress(addrinfo **result)
{
    addrinfo hints{};
    SecureZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    int errorCode = getaddrinfo(serverIPAddress.c_str(), std::to_string(serverTCPPort).c_str(), &hints, result);
    if ((NO_ERROR != errorCode) || (nullptr == *result))
    {
        std::cerr << "getaddrinfo() failed with error: " << errorCode << std::endl;
        return false;
    }
    return true;
}

// Helper function to create TCP socket
SOCKET CreateTCPSocket(addrinfo *info)
{
    SOCKET clientSocket = socket(info->ai_family, info->ai_socktype, info->ai_protocol);
    if (INVALID_SOCKET == clientSocket)
    {
        std::cerr << "socket() failed with error: " << WSAGetLastError() << std::endl;
        return INVALID_SOCKET;
    }
    return clientSocket;
}

// Helper function to connect to server
bool ConnectToServer(SOCKET clientSocket, addrinfo *info)
{
    int errorCode = connect(clientSocket, info->ai_addr, static_cast<int>(info->ai_addrlen));
    if (SOCKET_ERROR == errorCode)
    {
        std::cerr << "connect() failed with error: " << WSAGetLastError() << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    // Get Server IP Address
    std::cout << "Server IP Address: ";
    std::getline(std::cin, serverIPAddress);

    // Get Server TCP Port Number
    std::string portStr;
    std::cout << "Server TCP Port Number: ";
    std::getline(std::cin, portStr);
    serverTCPPort = static_cast<uint16_t>(std::stoul(portStr));

    // Get Server UDP Port Number
    std::cout << "Server UDP Port Number: ";
    std::getline(std::cin, portStr);
    serverUDPPort = static_cast<uint16_t>(std::stoul(portStr));

    // Get Client UDP Port Number
    std::cout << "Client UDP Port Number: ";
    std::getline(std::cin, portStr);
    clientUDPPort = static_cast<uint16_t>(std::stoul(portStr));

    // Get Path to store downloads
    std::cout << "Path: ";
    std::getline(std::cin, downloadPath);

    // Ensure the download directory exists
    std::filesystem::create_directories(downloadPath);

    // Initialize Winsock
    WSADATA wsaData{};
    if (!InitializeWinsock(wsaData))
    {
        return RETURN_CODE_1;
    }

    // Resolve server address
    addrinfo *info = nullptr;
    if (!ResolveServerAddress(&info))
    {
        WSACleanup();
        return RETURN_CODE_2;
    }

    // Create TCP socket
    SOCKET clientSocket = CreateTCPSocket(info);
    if (clientSocket == INVALID_SOCKET)
    {
        freeaddrinfo(info);
        WSACleanup();
        return RETURN_CODE_2;
    }

    // Connect to server
    if (!ConnectToServer(clientSocket, info))
    {
        freeaddrinfo(info);
        closesocket(clientSocket);
        WSACleanup();
        return RETURN_CODE_3;
    }

    freeaddrinfo(info);

    // Set socket to non-blocking mode
    u_long mode = 1;
    ioctlsocket(clientSocket, FIONBIO, &mode);

    // Initialize UDP socket
    if (!InitializeUDPSocket())
    {
        std::cerr << "Failed to initialize UDP socket." << std::endl;
        closesocket(clientSocket);
        WSACleanup();
        return RETURN_CODE_4;
    }

    // Start threads
    std::thread tcpReceiver(TCPReceiverThread, clientSocket);
    std::thread udpReceiver(UDPReceiverThread);
    std::thread sessionMonitor(SessionMonitorThread);

    // Display usage instructions
    DisplayUsage();

    // Main loop for handling input
    std::string input;
    while (isRunning && std::getline(std::cin, input))
    {
        if (input.empty()) continue;

        if (input == "/q")
        {
            char quitCmd = REQ_QUIT;
            send(clientSocket, &quitCmd, 1, 0);
            LogMessage("Disconnecting from server...");
            break;
        }
        else if (input == "/l")
        {
            SendListFilesRequest(clientSocket);
        }
        else if (input.substr(0, 2) == "/d")
        {
            if (input.size() < 3)
            {
                LogMessage("Invalid download command. Use: /d IP:Port filename");
                continue;
            }
            std::istringstream iss(input.substr(3));
            std::string ipPort, filename;

            if (iss >> ipPort >> filename)
            {
                std::string clientIP;
                uint16_t clientPort;

                if (ParseIPPort(ipPort, clientIP, clientPort))
                {
                    // Store the filename for this download request
                    {
                        std::lock_guard<std::mutex> lock(printMutex);
                        std::string sessionKey = "pending_" + filename;
                        downloadedFiles[sessionKey] = filename;
                    }

                    SendDownloadRequest(clientSocket, clientIP, clientPort, filename);
                }
                else
                {
                    LogMessage("Invalid IP:Port format. Use: /d IP:Port filename");
                }
            }
            else
            {
                LogMessage("Invalid download command. Use: /d IP:Port filename");
            }
        }
        else if (input == "/help")
        {
            DisplayUsage();
        }
        else
        {
            LogMessage("Unknown command. Type /help for available commands.");
        }
    }

    // Clean up
    isRunning = false;
    shutdown(clientSocket, SD_SEND);
    closesocket(clientSocket);

    if (udpSocket != INVALID_SOCKET)
    {
        closesocket(udpSocket);
    }

    if (tcpReceiver.joinable())
    {
        tcpReceiver.join();
    }

    if (udpReceiver.joinable())
    {
        udpReceiver.join();
    }

    if (sessionMonitor.joinable())
    {
        sessionMonitor.join();
    }

    // Close any open file handles
    {
        std::lock_guard<std::mutex> lock(sessionsLock);
        for (auto &session : activeSessions)
        {
            if (session.second.fileStream.is_open())
            {
                session.second.fileStream.close();
            }
        }
    }

    WSACleanup();
    return 0;
}