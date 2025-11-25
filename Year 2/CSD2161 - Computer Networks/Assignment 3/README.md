/* Start Header
*****************************************************************/
/*! \file: README.md
\author: Bryan Ang Wei Ze, Tham Kang Ting, Low Yue Jun
\par: bryanweize.ang\@digipen.edu ,kangting.t\@digipen.edu, yuejun.low\@digipen.edu
\date: 14/03/2025 (DD/MM/YYYY)
\brief Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of thie file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited. */

/* End Header
*****************************************************************/


# UDP File Downloading System

## Overview

This project implements a robust file downloading system over a combination of TCP and UDP. The system allows clients to connect to a server, view available files, and download them using a reliable Go-Back-N protocol over UDP. TCP is used for control messages, and UDP for the actual file transfers with reliability mechanisms that handle network disruptions.

## Features

- **Multiple Client Support**: Server simultaneously handles multiple client connections and file transfers
- **File Listing**: Clients can request and view available files on the server
- **Reliable Go-Back-N Protocol**: Ensures file integrity even on unreliable networks
- **Comprehensive Logging**: Detailed timestamp-based logging for easier debugging
- **Progress Tracking**: Real-time download progress, speed, and ETA display
- **Session Monitoring**: Automatic detection and recovery for stalled downloads
- **Error Handling**: Robust error detection and recovery mechanisms
- **Configurable Parameters**: Easily adjustable protocol parameters in config.h

## Protocol Implementation

The system implements a reliable data transfer protocol with the following features:

### Go-Back-N Protocol

- **Sliding Window Mechanism**: Configurable window size (default: 5 packets)
- **Sequence Numbering**: Each packet has a unique sequence number for tracking
- **Cumulative Acknowledgements**: Client acknowledges packets by sequence number
- **Timeout and Retransmission**: Configurable timeout with automatic retransmission
- **Retry Limiting**: Maximum retry attempts to prevent infinite retransmissions
- **Flow Control**: Window advancement based on acknowledgment reception

### Message Formats

#### TCP Messages

1. **REQ_LISTFILES (0x4)**: Request for available files list
2. **RSP_LISTFILES (0x5)**: Server response with file list
3. **REQ_DOWNLOAD (0x2)**: Request to download a file via UDP
4. **RSP_DOWNLOAD (0x3)**: Server response with download session information
5. **REQ_QUIT (0x1)**: Request to disconnect
6. **DOWNLOAD_ERROR (0x30)**: Error response for failed download requests

#### UDP Packets

1. **Data Packet (LSB=0)**:
   - Flags (1 byte): LSB 0 indicates data packet
   - Session ID (4 bytes): Identifies the download session
   - File Offset (4 bytes): Position in the file
   - Data Length (4 bytes): Length of the data in this packet
   - File Data (variable): Actual file data

2. **ACK Packet (LSB=1)**:
   - Flags (1 byte): LSB 1 indicates ACK packet
   - Session ID (4 bytes): Identifies the download session
   - ACK Number (4 bytes): Sequence number being acknowledged

## Configuration

Key parameters can be adjusted in `config.h`:

```cpp
// Protocol selection
// 0 = Stop-and-Wait
// 1 = Go-Back-N
// 2 = Selective Repeat
#define PROTOCOL_TYPE 1

// Network parameters
constexpr uint32_t MAX_PACKET_SIZE = 1024;  // Maximum UDP packet size
constexpr uint32_t UDP_HEADER_SIZE = 13;    // Size of our UDP header
constexpr uint32_t DATA_PACKET_SIZE = MAX_PACKET_SIZE - UDP_HEADER_SIZE;

// Protocol parameters
constexpr uint32_t WINDOW_SIZE = 5;         // Window size for Go-Back-N
constexpr uint32_t ACK_TIMEOUT_MS = 500;    // Timeout for ACK in milliseconds
constexpr uint32_t MAX_RETRIES = 5;         // Maximum number of retransmissions

// Session parameters
constexpr uint32_t MAX_SESSION_IDLE_MS = 30000; // Session timeout after 30 seconds
```

## Compilation

To compile the project using Visual Studio:

1. Open the solution file in Visual Studio
2. Select Release configuration
3. Build the solution

## Usage

### Server

1. Run the server executable
2. You will be prompted for:
   - TCP port number
   - UDP port number for file transfers
   - Path to the directory containing files for download
3. The server will display its IP address and port information
4. The server will then wait for client connections

### Client

1. Run the client executable
2. You will be prompted for:
   - Server's IP address
   - Server's TCP port number
   - Server's UDP port number
   - Client's UDP port number to use for receiving files
   - Local path to store downloaded files
3. After connecting successfully, you can use the available commands

### Client Commands

- `/l`: List available files on the server
- `/d IP:Port filename`: Download a file (e.g., `/d 192.168.1.5:8000 example.txt`)
- `/q`: Quit and disconnect from the server
- `/help`: Display available commands

## Testing

The system has been thoroughly tested with various scenarios:

- **Regular Operation**: Normal file transfers without network disruptions
- **Network Disruptions**: Using the Ethernet Break Circuit to simulate connection issues
- **Multiple Clients**: Simultaneous downloads from multiple clients
- **Large Files**: Transfers of files up to 200MB
- **Edge Cases**: Server/client disconnections, file not found, and other error scenarios

## Implementation Notes

- **Multithreading**: The server uses the TaskQueue system for TCP connections and separate threads for file transfers and UDP communications
- **Client Multithreading**: The client employs multiple threads for UI, TCP communication, and UDP file reception
- **Socket Optimization**: Socket buffer sizes are increased for better performance
- **File Handling**: Files are efficiently accessed with proper buffer management
- **Session Management**: Each download has a unique session ID with comprehensive state tracking
- **Timeout Handling**: Graceful timeout recovery with multiple retry attempts
- **Progress Reporting**: Detailed download progress with speed and time estimation

## Limitations

- Maximum file size: 200 MB
- Maximum number of concurrent downloadable files: 256
- Network performance depends on window size and acknowledgment timeout values
