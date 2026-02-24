/******************************************************************************/
/*!
\file   UDPSocket.cpp
\brief  WinSock2 UDP socket wrapper implementation
*/
/******************************************************************************/
#include "UDPSocket.h"
#include <cstring>

#pragma comment(lib, "ws2_32.lib")

UDPSocket::UDPSocket()
{
    _socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
}

UDPSocket::~UDPSocket()
{
    Close();
}

bool UDPSocket::Bind(const std::string& ip, uint16_t port)
{
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);

    if (ip.empty() || ip == "0.0.0.0")
        addr.sin_addr.s_addr = INADDR_ANY;
    else
        inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

    return bind(_socket, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0;
}

bool UDPSocket::SendTo(const std::vector<uint8_t>& data,
                       const std::string& ip, uint16_t port)
{
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

    int sent = sendto(_socket,
                      reinterpret_cast<const char*>(data.data()),
                      static_cast<int>(data.size()),
                      0,
                      reinterpret_cast<const sockaddr*>(&addr),
                      sizeof(addr));
    return sent != SOCKET_ERROR;
}

bool UDPSocket::SendTo(const uint8_t* data, size_t len, const sockaddr_in& addr)
{
    int sent = sendto(_socket,
                      reinterpret_cast<const char*>(data),
                      static_cast<int>(len),
                      0,
                      reinterpret_cast<const sockaddr*>(&addr),
                      sizeof(addr));
    return sent != SOCKET_ERROR;
}

int UDPSocket::RecvFrom(std::vector<uint8_t>& outData, sockaddr_in& fromAddr)
{
    const int BUF_SIZE = 4096;
    outData.resize(BUF_SIZE);

    int fromLen  = sizeof(fromAddr);
    int received = recvfrom(_socket,
                            reinterpret_cast<char*>(outData.data()),
                            BUF_SIZE,
                            0,
                            reinterpret_cast<sockaddr*>(&fromAddr),
                            &fromLen);
    if (received > 0)
        outData.resize(static_cast<size_t>(received));
    else
        outData.clear();

    return received;
}

void UDPSocket::SetRecvTimeout(int timeoutMs)
{
    DWORD tv = static_cast<DWORD>(timeoutMs);
    setsockopt(_socket, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&tv), sizeof(tv));
}

void UDPSocket::Close()
{
    if (_socket != INVALID_SOCKET)
    {
        closesocket(_socket);
        _socket = INVALID_SOCKET;
    }
}

bool UDPSocket::InitWinsock()
{
    WSADATA wsaData;
    return WSAStartup(MAKEWORD(2, 2), &wsaData) == 0;
}

void UDPSocket::ShutdownWinsock()
{
    WSACleanup();
}
