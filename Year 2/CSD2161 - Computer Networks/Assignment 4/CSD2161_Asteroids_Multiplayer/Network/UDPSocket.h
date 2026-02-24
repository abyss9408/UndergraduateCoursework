/******************************************************************************/
/*!
\file   UDPSocket.h
\brief  WinSock2 UDP socket wrapper
*/
/******************************************************************************/
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#include <cstdint>
#include <string>
#include <vector>

class UDPSocket
{
public:
    UDPSocket();
    ~UDPSocket();

    // Disable copy
    UDPSocket(const UDPSocket&)            = delete;
    UDPSocket& operator=(const UDPSocket&) = delete;

    // Bind to a local port (pass empty string or "0.0.0.0" for INADDR_ANY)
    bool Bind(const std::string& ip, uint16_t port);

    // Send to a destination specified by string
    bool SendTo(const std::vector<uint8_t>& data,
                const std::string& ip, uint16_t port);

    // Send to a destination specified by sockaddr_in (efficient, no DNS)
    bool SendTo(const uint8_t* data, size_t len, const sockaddr_in& addr);

    // Receive a datagram; returns bytes received (>0), 0 on timeout/nothing, <0 on error
    int  RecvFrom(std::vector<uint8_t>& outData, sockaddr_in& fromAddr);

    // Set receive timeout in milliseconds (0 = blocking)
    void SetRecvTimeout(int timeoutMs);

    void Close();
    bool IsValid() const { return _socket != INVALID_SOCKET; }

    // Must be called once at program start / end
    static bool InitWinsock();
    static void ShutdownWinsock();

private:
    SOCKET _socket = INVALID_SOCKET;
};
