/* Start Header
*****************************************************************/
/*!
\file config.h
\author Tham Kang Ting (kangting.t)
\par
\date 9 March 2025
\brief
This file defines configuration parameters for the UDP file transfer system.
Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

// Network constants
#define WINSOCK_VERSION     2
#define WINSOCK_SUBVERSION  2
#define MAX_STR_LEN         1000
#define RETURN_CODE_1       1
#define RETURN_CODE_2       2
#define RETURN_CODE_3       3
#define RETURN_CODE_4       4

// Protocol selection
// 0 = Stop-and-Wait
// 1 = Go-Back-N (current implementation)
// 2 = Selective Repeat
#define PROTOCOL_TYPE 1

// Message format constants
constexpr uint8_t UNKNOWN = 0x0;
constexpr uint8_t REQ_QUIT = 0x1;
constexpr uint8_t REQ_DOWNLOAD = 0x2;
constexpr uint8_t RSP_DOWNLOAD = 0x3;
constexpr uint8_t REQ_LISTFILES = 0x4;
constexpr uint8_t RSP_LISTFILES = 0x5;
constexpr uint8_t CMD_TEST = 0x20;
constexpr uint8_t DOWNLOAD_ERROR = 0x30;

// UDP Packet constants
constexpr uint8_t UDP_DATA = 0x0;  // LSB 0 means data packet
constexpr uint8_t UDP_ACK = 0x1;   // LSB 1 means ACK packet

// Network parameters
constexpr uint32_t MAX_PACKET_SIZE = 1472;  // Maximum UDP packet size (optimal for 1500 MTU: 1500 - 20 IP - 8 UDP = 1472)
constexpr uint32_t UDP_HEADER_SIZE = 13;    // Size of our UDP header (1+4+4+4)
constexpr uint32_t DATA_PACKET_SIZE = MAX_PACKET_SIZE - UDP_HEADER_SIZE;
constexpr uint32_t TCP_BUFFER_SIZE = 4096;  // TCP buffer size
constexpr uint32_t FILE_BUFFER_SIZE = 4096; // File read/write buffer size

// Go-Back-N protocol parameters
constexpr uint32_t WINDOW_SIZE = 256;        // Window size for Go-Back-N (aggressive for maximum throughput)
constexpr uint32_t ACK_TIMEOUT_MS = 250;    // Timeout for ACK in milliseconds (optimized for LAN)
constexpr uint32_t MAX_RETRIES = 5;         // Maximum number of retransmissions

// Server parameters
constexpr uint32_t MAX_CLIENTS = 10;        // Maximum number of simultaneous clients
constexpr uint32_t THREAD_POOL_SIZE = 10;   // Number of threads in the server pool
constexpr uint32_t MAX_SESSION_IDLE_MS = 30000; // Session timeout after 30 seconds of inactivity
constexpr uint32_t SESSION_MONITOR_INTERVAL_MS = 5000; // Session check interval

// Client parameters
constexpr uint32_t PROGRESS_UPDATE_PERCENT = 5; // Update progress display every 5%

// File transfer limitations (as specified in requirements)
constexpr uint32_t MAX_FILE_SIZE = 200 * 1024 * 1024; // 200 MB maximum file size
constexpr uint32_t MAX_FILES = 256;         // Maximum number of downloadable files

#endif // CONFIG_H
