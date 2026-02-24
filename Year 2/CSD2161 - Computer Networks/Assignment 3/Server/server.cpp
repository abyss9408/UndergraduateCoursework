/* Start Header
*****************************************************************/
/*!
\file server.cpp
\author: Bryan Ang Wei Ze, Tham Kang Ting, Low Yue Jun
\par: bryanweize.ang\@digipen.edu ,kangting.t\@digipen.edu, yuejun.low\@digipen.edu
\date 9 March 2025
\brief
This file implements the server for a file downloading system over UDP.
Based on the echo server from Assignment 3.
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
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <thread>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "taskqueue.h"

// -----------------------------------------------------------------------
// Protocol constants (wire format — must match between client and server)
// -----------------------------------------------------------------------
#define WINSOCK_VERSION    2
#define WINSOCK_SUBVERSION 2
#define MAX_STR_LEN        1000
#define RETURN_CODE_1      1
#define RETURN_CODE_2      2
#define RETURN_CODE_3      3
#define RETURN_CODE_4      4

constexpr uint8_t UNKNOWN        = 0x0;
constexpr uint8_t REQ_QUIT       = 0x1;
constexpr uint8_t REQ_DOWNLOAD   = 0x2;
constexpr uint8_t RSP_DOWNLOAD   = 0x3;
constexpr uint8_t REQ_LISTFILES  = 0x4;
constexpr uint8_t RSP_LISTFILES  = 0x5;
constexpr uint8_t CMD_TEST       = 0x20;
constexpr uint8_t DOWNLOAD_ERROR = 0x30;

constexpr uint8_t  UDP_DATA        = 0x0;
constexpr uint8_t  UDP_ACK         = 0x1;
constexpr uint32_t UDP_HEADER_SIZE = 13;

// -----------------------------------------------------------------------
// Runtime-tunable parameters — defaults; overridden by LoadConfig()
// -----------------------------------------------------------------------
uint32_t PROTOCOL_TYPE               = 1;
uint32_t MAX_PACKET_SIZE             = 1472;
uint32_t DATA_PACKET_SIZE            = 1459;  // 1472 - 13; recalculated after load
uint32_t TCP_BUFFER_SIZE             = 4096;
uint32_t FILE_BUFFER_SIZE            = 4096;
uint32_t WINDOW_SIZE                 = 256;
uint32_t ACK_TIMEOUT_MS              = 250;
uint32_t MAX_RETRIES                 = 5;
uint32_t MAX_CLIENTS                 = 10;
uint32_t THREAD_POOL_SIZE            = 10;
uint32_t MAX_SESSION_IDLE_MS         = 30000;
uint32_t SESSION_MONITOR_INTERVAL_MS = 5000;
uint32_t PROGRESS_UPDATE_PERCENT     = 5;
uint32_t MAX_FILE_SIZE               = 200u * 1024u * 1024u;
uint32_t MAX_FILES                   = 256;

// Reads config.cfg from the executable's directory, falling back to the
// current working directory. Call once at startup before using config values.
bool LoadConfig(const std::string& path = "")
{
    std::string actualPath = path;
    if (actualPath.empty())
    {
        char exePath[MAX_PATH] = {};
        GetModuleFileNameA(nullptr, exePath, MAX_PATH);
        std::string exeDir(exePath);
        size_t pos = exeDir.find_last_of("/\\");
        actualPath = (pos != std::string::npos ? exeDir.substr(0, pos + 1) : "") + "config.cfg";
    }

    std::ifstream cfgFile(actualPath);
    if (!cfgFile.is_open())
    {
        cfgFile.open("config.cfg");
        if (cfgFile.is_open())
            actualPath = "config.cfg";
    }

    if (!cfgFile.is_open())
    {
        std::cout << "[Config] config.cfg not found, using built-in defaults.\n";
        return false;
    }

    auto trim = [](const std::string& s) -> std::string {
        const char* ws = " \t\r\n";
        size_t start = s.find_first_not_of(ws);
        if (start == std::string::npos) return {};
        size_t end = s.find_last_not_of(ws);
        return s.substr(start, end - start + 1);
    };

    std::string line;
    while (std::getline(cfgFile, line))
    {
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos)
            line = line.substr(0, commentPos);

        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue;

        std::string key    = trim(line.substr(0, eqPos));
        std::string valStr = trim(line.substr(eqPos + 1));
        if (key.empty() || valStr.empty()) continue;

        uint32_t value = 0;
        try { value = static_cast<uint32_t>(std::stoul(valStr)); }
        catch (...) { continue; }

        if      (key == "PROTOCOL_TYPE")               PROTOCOL_TYPE = value;
        else if (key == "MAX_PACKET_SIZE")             MAX_PACKET_SIZE = value;
        else if (key == "TCP_BUFFER_SIZE")             TCP_BUFFER_SIZE = value;
        else if (key == "FILE_BUFFER_SIZE")            FILE_BUFFER_SIZE = value;
        else if (key == "WINDOW_SIZE")                 WINDOW_SIZE = value;
        else if (key == "ACK_TIMEOUT_MS")              ACK_TIMEOUT_MS = value;
        else if (key == "MAX_RETRIES")                 MAX_RETRIES = value;
        else if (key == "MAX_CLIENTS")                 MAX_CLIENTS = value;
        else if (key == "THREAD_POOL_SIZE")            THREAD_POOL_SIZE = value;
        else if (key == "MAX_SESSION_IDLE_MS")         MAX_SESSION_IDLE_MS = value;
        else if (key == "SESSION_MONITOR_INTERVAL_MS") SESSION_MONITOR_INTERVAL_MS = value;
        else if (key == "PROGRESS_UPDATE_PERCENT")     PROGRESS_UPDATE_PERCENT = value;
        else if (key == "MAX_FILE_SIZE")               MAX_FILE_SIZE = value;
        else if (key == "MAX_FILES")                   MAX_FILES = value;
    }

    DATA_PACKET_SIZE = MAX_PACKET_SIZE - UDP_HEADER_SIZE;
    std::cout << "[Config] Loaded configuration from '" << actualPath << "'.\n";
    return true;
}

// Structure to store client information
struct ClientInfo {
    std::string ip;
    uint16_t port;
    SOCKET socket;
};

// Structure for download session
struct DownloadSession {
    uint32_t sessionId;
    std::string filename;
    uint32_t fileSize;
    std::string clientIP;
    uint16_t clientPort;
    std::ifstream fileStream;
    uint32_t nextOffset;
    uint32_t windowStart;  // For sliding window protocol
    uint32_t windowSize;   // For sliding window protocol
    std::map<uint32_t, bool> acked;  // Track which packets have been ACKed
    std::atomic<bool> isActive;
    std::thread transferThread;
    std::chrono::steady_clock::time_point lastActivity;
    uint32_t retryCount;  // Count of retransmissions
};

// Global variables
std::map<SOCKET, ClientInfo> g_clients;
std::mutex g_clients_mutex;
SOCKET g_listenerSocket = INVALID_SOCKET;
SOCKET g_udpSocket = INVALID_SOCKET;
std::string g_filePath; // Path to store downloadable files
uint16_t g_udpPort = 0;
std::map<uint32_t, DownloadSession> g_downloadSessions;
std::mutex g_sessionsLock;
std::atomic<uint32_t> g_nextSessionId{ 1 };
std::atomic<bool> g_serverRunning{ true };

// Forward declarations
bool ProcessClient(SOCKET clientSocket);
void DisconnectServer();
void RemoveClient(SOCKET clientSocket);
void UDPReceiverThread();
void SessionMonitorThread();
void TransferFile(uint32_t sessionId);

// Helper function for logging with timestamps
void LogMessage(const std::string& message) {
    std::lock_guard<std::mutex> lock(_stdoutMutex);
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm tm_buf;
    localtime_s(&tm_buf, &now_time_t);

    std::cout << "["
        << std::put_time(&tm_buf, "%H:%M:%S")
        << "] " << message << std::endl;
}

// Helper function to format IP:Port
std::string FormatIPPort(const std::string& ip, uint16_t port) {
    return ip + ":" + std::to_string(port);
}

// Helper function to send error response
void SendDownloadError(SOCKET clientSocket) {
    char errorMsg = DOWNLOAD_ERROR;
    send(clientSocket, &errorMsg, 1, 0);
    LogMessage("Sent download error response");
}

// Helper function to get list of available files
std::vector<std::string> GetAvailableFiles() {
    std::vector<std::string> files;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(g_filePath)) {
            if (entry.is_regular_file()) {
                files.push_back(entry.path().filename().string());
            }
        }
    }
    catch (const std::exception& e) {
        LogMessage("Error listing files: " + std::string(e.what()));
    }
    return files;
}

// Helper function to send file listing response
void SendListFilesResponse(SOCKET clientSocket) {
    std::vector<std::string> files = GetAvailableFiles();
    LogMessage("Sending file list with " + std::to_string(files.size()) + " files");

    std::vector<char> response;
    response.push_back(RSP_LISTFILES);

    // Number of files (2 bytes)
    uint16_t numFiles = static_cast<uint16_t>(files.size());
    uint16_t netNumFiles = htons(numFiles);
    response.insert(response.end(),
        reinterpret_cast<char*>(&netNumFiles),
        reinterpret_cast<char*>(&netNumFiles) + 2);

    // Calculate total length of file list data
    uint32_t totalLength = 0;
    for (const auto& file : files) {
        totalLength += 4; // Filename length (4 bytes)
        totalLength += file.length(); // Filename
    }

    // Add total length of file list (4 bytes)
    uint32_t netTotalLength = htonl(totalLength);
    response.insert(response.end(),
        reinterpret_cast<char*>(&netTotalLength),
        reinterpret_cast<char*>(&netTotalLength) + 4);

    // Add each file's info
    for (const auto& file : files) {
        // Filename length (4 bytes)
        uint32_t filenameLen = static_cast<uint32_t>(file.length());
        uint32_t netFilenameLen = htonl(filenameLen);
        response.insert(response.end(),
            reinterpret_cast<char*>(&netFilenameLen),
            reinterpret_cast<char*>(&netFilenameLen) + 4);

        // Filename
        response.insert(response.end(), file.begin(), file.end());
    }

    send(clientSocket, response.data(), static_cast<int>(response.size()), 0);
}

// Helper function to parse client IP and port from buffer
bool ParseClientIPPort(const std::vector<char>& buffer, std::string& clientIP, uint16_t& clientPort) {
    if (buffer.size() < 7) return false; // Need at least 7 bytes for IP and port

    in_addr clientAddr;
    memcpy(&clientAddr.s_addr, buffer.data() + 1, 4);
    char ipStr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &clientAddr, ipStr, INET_ADDRSTRLEN);
    clientIP = ipStr;

    memcpy(&clientPort, buffer.data() + 5, 2);
    clientPort = ntohs(clientPort);

    return true;
}

// Helper function to parse filename from buffer
bool ParseFilename(const std::vector<char>& buffer, std::string& filename) {
    if (buffer.size() < 11) return false; // Minimum header size

    uint32_t filenameLen;
    memcpy(&filenameLen, buffer.data() + 7, 4);
    filenameLen = ntohl(filenameLen);

    if (buffer.size() < 11 + filenameLen) return false;

    filename = std::string(buffer.data() + 11, filenameLen);
    return true;
}

// Helper function to check if file exists and get size
bool GetFileInfo(const std::string& filename, uint32_t& fileSize) {
    std::filesystem::path filePath = std::filesystem::path(g_filePath) / filename;
    if (!std::filesystem::exists(filePath)) {
        return false;
    }

    fileSize = static_cast<uint32_t>(std::filesystem::file_size(filePath));
    return true;
}

// Helper function to create download response message
std::vector<char> CreateDownloadResponse(uint32_t sessionId, uint32_t fileSize) {
    std::vector<char> response;
    response.push_back(RSP_DOWNLOAD);

    // Get server's IP address
    char localIP[MAX_STR_LEN];
    gethostname(localIP, MAX_STR_LEN);
    addrinfo hints{}, * info = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    getaddrinfo(localIP, nullptr, &hints, &info);
    sockaddr_in* sockaddr_ipv4 = reinterpret_cast<sockaddr_in*>(info->ai_addr);
    uint32_t serverIP = sockaddr_ipv4->sin_addr.s_addr;
    response.insert(response.end(),
        reinterpret_cast<char*>(&serverIP),
        reinterpret_cast<char*>(&serverIP) + 4);
    freeaddrinfo(info);

    // Server UDP port (2 bytes)
    uint16_t netPort = htons(g_udpPort);
    response.insert(response.end(),
        reinterpret_cast<char*>(&netPort),
        reinterpret_cast<char*>(&netPort) + 2);

    // Session ID (4 bytes)
    uint32_t netSessionId = htonl(sessionId);
    response.insert(response.end(),
        reinterpret_cast<char*>(&netSessionId),
        reinterpret_cast<char*>(&netSessionId) + 4);

    // File length (4 bytes)
    uint32_t netFileSize = htonl(fileSize);
    response.insert(response.end(),
        reinterpret_cast<char*>(&netFileSize),
        reinterpret_cast<char*>(&netFileSize) + 4);

    return response;
}

// Helper function to handle download request
bool HandleDownloadRequest(SOCKET clientSocket, const std::vector<char>& buffer) {
    // Parse client IP and port
    std::string clientIP;
    uint16_t clientPort;
    if (!ParseClientIPPort(buffer, clientIP, clientPort)) {
        LogMessage("Failed to parse client IP and port");
        SendDownloadError(clientSocket);
        return true;
    }

    // Parse filename
    std::string filename;
    if (!ParseFilename(buffer, filename)) {
        LogMessage("Failed to parse filename");
        SendDownloadError(clientSocket);
        return true;
    }

    LogMessage("Download request from " + FormatIPPort(clientIP, clientPort) +
        " for file: " + filename);

    // Check if file exists and get size
    uint32_t fileSize;
    if (!GetFileInfo(filename, fileSize)) {
        LogMessage("File not found: " + filename);
        SendDownloadError(clientSocket);
        return true;
    }

    // Create a new session ID
    uint32_t sessionId = g_nextSessionId++;

    // Create and send the response
    auto response = CreateDownloadResponse(sessionId, fileSize);
    send(clientSocket, response.data(), static_cast<int>(response.size()), 0);

    // Setup the download session
    DownloadSession& session = g_downloadSessions[sessionId];

    session.sessionId = sessionId;
    session.filename = filename;
    session.fileSize = fileSize;
    session.clientIP = clientIP;
    session.clientPort = clientPort;
    session.nextOffset = 0;
    session.windowStart = 0;
    session.windowSize = WINDOW_SIZE;
    session.isActive.store(true);
    session.lastActivity = std::chrono::steady_clock::now();
    session.retryCount = 0;

    // Open the file for reading
    std::filesystem::path filePath = std::filesystem::path(g_filePath) / filename;
    session.fileStream.open(filePath, std::ios::binary);
    if (!session.fileStream) {
        LogMessage("Failed to open file: " + filename);
        SendDownloadError(clientSocket);
        return true;
    }

    LogMessage("Created download session " + std::to_string(sessionId) +
        " for file: " + filename + " (" + std::to_string(fileSize) + " bytes)");

    // Store the session and start the transfer thread
    auto RunFileTransferSession = [](uint32_t sessionId) {
        TransferFile(sessionId);
    };

    // Now start the thread
    session.transferThread = std::thread(RunFileTransferSession, sessionId);

    return true;
}

// Helper function to create a UDP data packet
std::vector<char> CreateDataPacket(uint32_t sessionId, uint32_t offset,
    const char* data, uint32_t dataSize) {
    std::vector<char> packet(13 + dataSize); // Header + data

    uint8_t flags = UDP_DATA;
    uint32_t netSessionId = htonl(sessionId);
    uint32_t netOffset = htonl(offset);
    uint32_t netDataSize = htonl(dataSize);

    int packetPos = 0;
    memcpy(packet.data() + packetPos, &flags, 1);
    packetPos += 1;

    memcpy(packet.data() + packetPos, &netSessionId, 4);
    packetPos += 4;

    memcpy(packet.data() + packetPos, &netOffset, 4);
    packetPos += 4;

    memcpy(packet.data() + packetPos, &netDataSize, 4);
    packetPos += 4;

    memcpy(packet.data() + packetPos, data, dataSize);

    return packet;
}

// Helper function to send a data packet
bool SendDataPacket(const std::string& clientIP, uint16_t clientPort,
    const std::vector<char>& packet) {
    sockaddr_in clientAddr;
    clientAddr.sin_family = AF_INET;
    inet_pton(AF_INET, clientIP.c_str(), &clientAddr.sin_addr);
    clientAddr.sin_port = htons(clientPort);

    int bytesSent = sendto(g_udpSocket, packet.data(), static_cast<int>(packet.size()), 0,
        (sockaddr*)&clientAddr, sizeof(clientAddr));

    return bytesSent != SOCKET_ERROR;
}

// Function to transfer a file - called in a separate thread
void TransferFile(uint32_t sessionId) {
    DownloadSession* session = nullptr;

    // Get session information
    // NOTE: The session pointer is safe to use after the lock is released because:
    // 1. Sessions are never erased from g_downloadSessions (only marked inactive)
    // 2. std::map doesn't invalidate pointers/references on insertions
    // However, mutable fields (acked, windowStart, nextOffset) are accessed concurrently
    // from this thread and the ACK receiver thread without additional synchronization.
    {
        std::lock_guard<std::mutex> lock(g_sessionsLock);
        auto it = g_downloadSessions.find(sessionId);
        if (it == g_downloadSessions.end()) {
            LogMessage("Session " + std::to_string(sessionId) + " not found");
            return;
        }
        session = &(it->second);
    }

    LogMessage("Starting file transfer for session " + std::to_string(sessionId));

    // Buffer for file data
    std::vector<char> fileDataBuffer(session->windowSize * DATA_PACKET_SIZE);

    // Sliding window protocol (Go-Back-N)
    while (session->isActive.load() && session->nextOffset < session->fileSize) {
        bool anyPacketSent = false;

        // Send all packets in the current window
        for (uint32_t seqNum = session->windowStart;
            seqNum < session->windowStart + session->windowSize &&
            seqNum * DATA_PACKET_SIZE < session->fileSize;
            ++seqNum) {

            // Check if this packet has already been ACKed
            if (session->acked.find(seqNum) != session->acked.end() && session->acked[seqNum]) {
                continue;
            }

            // Calculate offset for this sequence number
            uint32_t offset = seqNum * DATA_PACKET_SIZE;

            // Don't go beyond the file size
            if (offset >= session->fileSize) {
                break;
            }

            // Position the file pointer
            session->fileStream.seekg(offset);

            // Read data from file
            uint32_t dataSize = min(DATA_PACKET_SIZE, session->fileSize - offset);
            session->fileStream.read(fileDataBuffer.data(), dataSize);

            // Create and send the packet
            auto packet = CreateDataPacket(session->sessionId, offset, fileDataBuffer.data(), dataSize);
            bool sent = SendDataPacket(session->clientIP, session->clientPort, packet);

            if (sent) {
                anyPacketSent = true;
                LogMessage("Sent packet: Session " + std::to_string(session->sessionId) +
                    ", Seq " + std::to_string(seqNum) +
                    ", Offset " + std::to_string(offset) +
                    ", Size " + std::to_string(dataSize));

                // Update last activity time
                session->lastActivity = std::chrono::steady_clock::now();
            }
            else {
                LogMessage("Failed to send packet: Session " + std::to_string(session->sessionId) +
                    ", Seq " + std::to_string(seqNum));
            }
        }

        // If nothing was sent (all packets in window were already ACKed), move window forward
        if (!anyPacketSent && session->windowStart * DATA_PACKET_SIZE < session->fileSize) {
            session->windowStart++;
            continue;
        }

        // Wait for ACKs with timeout
        std::this_thread::sleep_for(std::chrono::milliseconds(ACK_TIMEOUT_MS));

        // Check if we have a timeout
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - session->lastActivity).count();

        if (elapsed > ACK_TIMEOUT_MS) {
            // Increment retry count
            session->retryCount++;

            if (session->retryCount > MAX_RETRIES) {
                LogMessage("Maximum retries limit reached for session " + std::to_string(sessionId) +
                    ", waiting 5 more seconds for any final ACKs");

                // Instead of breaking, wait a bit longer for final ACKs
                auto finalWaitStart = std::chrono::steady_clock::now();
                while (std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - finalWaitStart).count() < 5) {

                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                    // Check if we've received any new ACKs during this waiting period
                    auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - session->lastActivity).count() < 100) {
                        // Reset retry count if we got a new ACK
                        LogMessage("Received new ACKs during final wait, continuing transfer");
                        session->retryCount = 0;
                        break;
                    }
                }

                // If no new ACKs during grace period, now we can break
                if (session->retryCount > MAX_RETRIES) {
                    LogMessage("No new ACKs during final wait, ending transfer");
                    break;
                }
            }

            LogMessage("Timeout for session " + std::to_string(sessionId) +
                ", retry " + std::to_string(session->retryCount) +
                "/" + std::to_string(MAX_RETRIES));

            // Reset the window to force retransmission
            for (auto& ack : session->acked) {
                if (ack.first >= session->windowStart) {
                    ack.second = false;
                }
            }
        }
        else {
            // Check if we can advance the window
            bool foundUnackedPacket = false;

            for (uint32_t seqNum = session->windowStart;
                seqNum < session->windowStart + session->windowSize;
                ++seqNum) {

                if (session->acked.find(seqNum) == session->acked.end() || !session->acked[seqNum]) {
                    session->windowStart = seqNum;
                    foundUnackedPacket = true;
                    break;
                }
            }

            // If all packets in the window are acked, move window forward
            if (!foundUnackedPacket) {
                session->windowStart += session->windowSize;

                // Update next offset
                uint32_t newNextOffset = session->windowStart * DATA_PACKET_SIZE;
                if (newNextOffset > session->nextOffset) {
                    session->nextOffset = min(newNextOffset, session->fileSize);

                    // Log progress
                    float progress = static_cast<float>(session->nextOffset) / session->fileSize * 100.0f;
                    LogMessage("Session " + std::to_string(sessionId) +
                        " progress: " + std::to_string(static_cast<int>(progress)) + "%");
                }
            }
        }

        // Check if we're done with the file
        if (session->nextOffset >= session->fileSize) {
            break;
        }
    }

    // Check if last chunk was ACKed, which indicates completion
    uint32_t totalChunks = (session->fileSize + DATA_PACKET_SIZE - 1) / DATA_PACKET_SIZE;
    uint32_t lastChunk = totalChunks - 1;

    if (session->acked.find(lastChunk) != session->acked.end() && session->acked[lastChunk]) {
        LogMessage("Last chunk (Seq " + std::to_string(lastChunk) + ") is ACKed - transfer is complete");
        session->nextOffset = session->fileSize;
    }

    // Close the session
    {
        std::lock_guard<std::mutex> lock(g_sessionsLock);
        auto it = g_downloadSessions.find(sessionId);
        if (it != g_downloadSessions.end()) {
            it->second.fileStream.close();
            it->second.isActive.store(false);

            if (it->second.nextOffset >= it->second.fileSize) {
                LogMessage("File transfer completed successfully: Session " + std::to_string(sessionId));
            }
            else {
                LogMessage("File transfer incomplete: Session " + std::to_string(sessionId) +
                    ", transferred " + std::to_string(it->second.nextOffset) +
                    "/" + std::to_string(it->second.fileSize) + " bytes");
            }
        }
    }
}

// Helper function to process ACK packet
void ProcessAckPacket(const std::vector<char>& buffer, int bytesReceived) {
    if (bytesReceived < 9) {
        return; // Not enough data for ACK packet
    }

    uint32_t sessionId, ackNum;

    // Extract header information
    memcpy(&sessionId, buffer.data() + 1, 4);
    sessionId = ntohl(sessionId);

    memcpy(&ackNum, buffer.data() + 5, 4);
    ackNum = ntohl(ackNum);

    // Update the session's ACK status
    std::lock_guard<std::mutex> lock(g_sessionsLock);
    auto it = g_downloadSessions.find(sessionId);
    if (it != g_downloadSessions.end()) {
        it->second.acked[ackNum] = true;
        it->second.lastActivity = std::chrono::steady_clock::now();
        it->second.retryCount = 0; // Reset retry count on successful ACK

        LogMessage("Received ACK: Session " + std::to_string(sessionId) +
            ", Seq " + std::to_string(ackNum));
    }
}

// Process client requests
bool ProcessClient(SOCKET clientSocket) {
    const size_t BUFFER_SIZE = TCP_BUFFER_SIZE;
    char buffer[BUFFER_SIZE];
    bool stay = true;

    // Enable non-blocking mode
    u_long mode = 1;
    ioctlsocket(clientSocket, FIONBIO, &mode);

    while (stay && g_serverRunning) {
        int bytesReceived = recv(clientSocket, buffer, BUFFER_SIZE, 0);

        if (bytesReceived == SOCKET_ERROR) {
            if (WSAGetLastError() == WSAEWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            LogMessage("Connection error on socket " + std::to_string(clientSocket));
            break;
        }

        if (bytesReceived == 0) {
            LogMessage("Client disconnected gracefully");
            break; // Connection closed
        }

        // Process received data
        std::vector<char> msgData(buffer, buffer + bytesReceived);
        uint8_t cmdId = static_cast<uint8_t>(msgData[0]);

        switch (cmdId) {
        case REQ_QUIT:
            LogMessage("Received quit request");
            stay = false;
            break;

        case REQ_DOWNLOAD:
            HandleDownloadRequest(clientSocket, msgData);
            break;

        case REQ_LISTFILES:
            LogMessage("Received list files request");
            SendListFilesResponse(clientSocket);
            break;

        default:
            LogMessage("Unknown command received: " + std::to_string(cmdId));
            break;
        }
    }

    RemoveClient(clientSocket);
    return true;
}

// UDP Receiver thread for ACKs
void UDPReceiverThread() {
    std::vector<char> buffer(FILE_BUFFER_SIZE);
    sockaddr_in senderAddr;
    int senderAddrLen = sizeof(senderAddr);

    LogMessage("UDP receiver thread started");

    while (g_serverRunning && g_udpSocket != INVALID_SOCKET) {
        int bytesReceived = recvfrom(g_udpSocket, buffer.data(), static_cast<int>(buffer.size()), 0,
            (sockaddr*)&senderAddr, &senderAddrLen);

        if (bytesReceived == SOCKET_ERROR) {
            if (WSAGetLastError() != WSAEWOULDBLOCK) {
                LogMessage("UDP receive error: " + std::to_string(WSAGetLastError()));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Process UDP packet
        if (bytesReceived >= 1) {
            uint8_t flags = static_cast<uint8_t>(buffer[0]);

            // Check if it's an ACK packet (LSB should be 1)
            if ((flags & 0x01) == UDP_ACK) {
                ProcessAckPacket(buffer, bytesReceived);
            }
        }
    }

    LogMessage("UDP receiver thread stopped");
}

// Thread to monitor sessions and clean up inactive ones
void SessionMonitorThread() {
    LogMessage("Session monitor thread started");

    while (g_serverRunning) {
        std::this_thread::sleep_for(std::chrono::milliseconds(SESSION_MONITOR_INTERVAL_MS));

        std::vector<uint32_t> sessionsToRemove;

        {
            std::lock_guard<std::mutex> lock(g_sessionsLock);
            auto now = std::chrono::steady_clock::now();

            for (const auto& pair : g_downloadSessions) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - pair.second.lastActivity).count();

                if (elapsed > MAX_SESSION_IDLE_MS && pair.second.isActive.load()) {
                    LogMessage("Session " + std::to_string(pair.first) + " timed out after " +
                        std::to_string(elapsed) + "ms of inactivity");
                    sessionsToRemove.push_back(pair.first);
                }
            }
        }

        // Clean up timed out sessions
        for (uint32_t sessionId : sessionsToRemove) {
            std::lock_guard<std::mutex> lock(g_sessionsLock);
            auto it = g_downloadSessions.find(sessionId);
            if (it != g_downloadSessions.end()) {
                it->second.isActive.store(false);

                if (it->second.transferThread.joinable()) {
                    it->second.transferThread.detach(); // Detach the thread to let it finish
                }

                LogMessage("Cleaned up inactive session " + std::to_string(sessionId));
            }
        }
    }

    LogMessage("Session monitor thread stopped");
}

// Initialize UDP socket for file transfers
bool InitializeUDPSocket() {
    g_udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_udpSocket == INVALID_SOCKET) {
        LogMessage("Failed to create UDP socket. Error: " + std::to_string(WSAGetLastError()));
        return false;
    }

    // Set up the local address structure for binding
    sockaddr_in localAddr;
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    localAddr.sin_port = htons(g_udpPort);

    // Bind the UDP socket to the specified port
    if (bind(g_udpSocket, (sockaddr*)&localAddr, sizeof(localAddr)) == SOCKET_ERROR) {
        LogMessage("Failed to bind UDP socket. Error: " + std::to_string(WSAGetLastError()));
        closesocket(g_udpSocket);
        g_udpSocket = INVALID_SOCKET;
        return false;
    }

    // Set socket to non-blocking mode
    u_long mode = 1;
    ioctlsocket(g_udpSocket, FIONBIO, &mode);

    // Increase send buffer size
    int sendBufSize = 1024 * 1024;  // 1 MB
    if (setsockopt(g_udpSocket, SOL_SOCKET, SO_SNDBUF, (char*)&sendBufSize, sizeof(sendBufSize)) == SOCKET_ERROR) {
        LogMessage("Warning: Failed to set send buffer size: " + std::to_string(WSAGetLastError()));
        // Continue anyway - buffer size optimization is not critical
    }

    // Increase receive buffer size
    int recvBufSize = 1024 * 1024;  // 1 MB
    if (setsockopt(g_udpSocket, SOL_SOCKET, SO_RCVBUF, (char*)&recvBufSize, sizeof(recvBufSize)) == SOCKET_ERROR) {
        LogMessage("Warning: Failed to set receive buffer size: " + std::to_string(WSAGetLastError()));
        // Continue anyway - buffer size optimization is not critical
    }

    LogMessage("UDP socket initialized on port " + std::to_string(g_udpPort));
    return true;
}

void RemoveClient(SOCKET clientSocket) {
    std::lock_guard<std::mutex> lock(g_clients_mutex);

    auto it = g_clients.find(clientSocket);
    if (it != g_clients.end()) {
        LogMessage("Removing client " + FormatIPPort(it->second.ip, it->second.port));
        g_clients.erase(it);
    }

    shutdown(clientSocket, SD_BOTH);
    closesocket(clientSocket);
}

void DisconnectServer() {
    LogMessage("Disconnecting server...");
    g_serverRunning = false;

    if (g_listenerSocket != INVALID_SOCKET) {
        shutdown(g_listenerSocket, SD_BOTH);
        closesocket(g_listenerSocket);
        g_listenerSocket = INVALID_SOCKET;
    }

    if (g_udpSocket != INVALID_SOCKET) {
        shutdown(g_udpSocket, SD_BOTH);
        closesocket(g_udpSocket);
        g_udpSocket = INVALID_SOCKET;
    }

    // Stop and clean up all active sessions
    std::lock_guard<std::mutex> lock(g_sessionsLock);
    for (auto& session : g_downloadSessions) {
        session.second.isActive.store(false);
        if (session.second.transferThread.joinable()) {
            session.second.transferThread.join();
        }
        session.second.fileStream.close();
    }
    g_downloadSessions.clear();

    LogMessage("Server disconnected");
}

int main() {
    LoadConfig();

    // Get TCP Port Number
    std::string tcpPortStr;
    std::cout << "Server TCP port Number: ";
    std::getline(std::cin, tcpPortStr);
    uint16_t tcpPort = static_cast<uint16_t>(std::stoul(tcpPortStr));

    // Get UDP Port Number
    std::string udpPortStr;
    std::cout << "Server UDP Port Number: ";
    std::getline(std::cin, udpPortStr);
    g_udpPort = static_cast<uint16_t>(std::stoul(udpPortStr));

    // Get Path for downloadable files
    std::cout << "Path: ";
    std::getline(std::cin, g_filePath);

    // Ensure the directory exists
    std::filesystem::create_directories(g_filePath);

    // Initialize Winsock
    WSADATA wsaData{};
    int errorCode = WSAStartup(MAKEWORD(WINSOCK_VERSION, WINSOCK_SUBVERSION), &wsaData);
    if (NO_ERROR != errorCode) {
        std::cerr << "WSAStartup() failed with error: " << errorCode << std::endl;
        return errorCode;
    }

    // Get local host information
    char host[MAX_STR_LEN];
    gethostname(host, MAX_STR_LEN);

    // Setup TCP listener
    addrinfo hints{};
    SecureZeroMemory(&hints, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* info = nullptr;
    errorCode = getaddrinfo(host, tcpPortStr.c_str(), &hints, &info);
    if ((NO_ERROR != errorCode) || (nullptr == info)) {
        std::cerr << "getaddrinfo() failed with error: " << errorCode << std::endl;
        WSACleanup();
        return errorCode;
    }

    // Create TCP listener socket
    g_listenerSocket = socket(hints.ai_family, hints.ai_socktype, hints.ai_protocol);
    if (INVALID_SOCKET == g_listenerSocket) {
        std::cerr << "socket() failed with error: " << WSAGetLastError() << std::endl;
        freeaddrinfo(info);
        WSACleanup();
        return RETURN_CODE_1;
    }

    // Bind TCP socket
    errorCode = bind(g_listenerSocket, info->ai_addr, static_cast<int>(info->ai_addrlen));
    if (NO_ERROR != errorCode) {
        std::cerr << "bind() failed with error: " << WSAGetLastError() << std::endl;
        closesocket(g_listenerSocket);
        g_listenerSocket = INVALID_SOCKET;
    }

    freeaddrinfo(info);

    if (INVALID_SOCKET == g_listenerSocket) {
        std::cerr << "TCP socket setup failed" << std::endl;
        WSACleanup();
        return RETURN_CODE_2;
    }

    // Initialize UDP socket
    if (!InitializeUDPSocket()) {
        std::cerr << "Failed to initialize UDP socket" << std::endl;
        closesocket(g_listenerSocket);
        WSACleanup();
        return RETURN_CODE_3;
    }



    // Start UDP receiver thread for ACKs
    std::thread udpReceiver(UDPReceiverThread);

    // Start session monitor thread
    std::thread sessionMonitor(SessionMonitorThread);

    // Start TCP listener
    errorCode = listen(g_listenerSocket, SOMAXCONN);
    if (NO_ERROR != errorCode) {
        std::cerr << "listen() failed with error: " << WSAGetLastError() << std::endl;
        DisconnectServer();
        if (udpReceiver.joinable()) udpReceiver.join();
        if (sessionMonitor.joinable()) sessionMonitor.join();
        WSACleanup();
        return RETURN_CODE_3;
    }

    // Print server information
    char serverIPAddr[MAX_STR_LEN];
    struct sockaddr_in serverAddress;
    int addrLen = sizeof(serverAddress);
    getsockname(g_listenerSocket, (sockaddr*)&serverAddress, &addrLen);
    inet_ntop(AF_INET, &(serverAddress.sin_addr), serverIPAddr, INET_ADDRSTRLEN);

    std::cout << std::endl;
    std::cout << "Server IP Address: " << serverIPAddr << std::endl;
    std::cout << "Server TCP Port Number: " << tcpPort << std::endl;
    std::cout << "Server UDP Port Number: " << g_udpPort << std::endl;

    LogMessage("Server started");

    // Create task queue with global function pointers
    void (*disconnectFn)() = DisconnectServer;
    bool (*processClientFn)(SOCKET) = ProcessClient;

    TaskQueue<SOCKET, bool(*)(SOCKET), void(*)()>
        taskQueue(10, 20, processClientFn, disconnectFn);

    LogMessage("Task queue initialized with 10 worker threads");

    // Accept client connections
    while (g_serverRunning && g_listenerSocket != INVALID_SOCKET) {
        sockaddr_in clientAddr;
        int clientAddrLen = sizeof(clientAddr);

        SOCKET clientSocket = accept(g_listenerSocket, (sockaddr*)&clientAddr, &clientAddrLen);
        if (clientSocket == INVALID_SOCKET) {
            if (WSAGetLastError() != WSAEWOULDBLOCK) {
                LogMessage("accept() failed with error: " + std::to_string(WSAGetLastError()));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Store client information
        ClientInfo client;
        char clientIP[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(clientAddr.sin_addr), clientIP, INET_ADDRSTRLEN);
        client.ip = clientIP;
        client.port = ntohs(clientAddr.sin_port);
        client.socket = clientSocket;

        LogMessage("New client connection from " + FormatIPPort(client.ip, client.port));

        {
            std::lock_guard<std::mutex> lock(g_clients_mutex);
            g_clients[clientSocket] = client;
        }

        taskQueue.produce(clientSocket);
    }

    // Clean up
    LogMessage("Server shutting down...");
    DisconnectServer();

    if (udpReceiver.joinable()) {
        udpReceiver.join();
    }

    if (sessionMonitor.joinable()) {
        sessionMonitor.join();
    }

    // Wait for all download sessions to complete
    {
        std::lock_guard<std::mutex> lock(g_sessionsLock);
        for (auto& session : g_downloadSessions) {
            if (session.second.transferThread.joinable()) {
                session.second.transferThread.join();
            }
        }
    }

    WSACleanup();
    LogMessage("Server shutdown complete");
    return 0;
}