/* Start Header
*****************************************************************/
/*!
\file echoclient.cpp
\author Bryan Ang Wei Ze (bryanweize.ang\digipen.edu)
\par Assignment 1
\date 26 Jan 2025
\brief
This file implements the client file which will be used to implement a 
simple echo client.
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
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

#define WINSOCK_VERSION     2
#define WINSOCK_SUBVERSION  2
#define MAX_STR_LEN         1000
#define RETURN_CODE_1       1
#define RETURN_CODE_2       2
#define RETURN_CODE_3       3
#define RETURN_CODE_4       4

enum class CommandID : uint8_t
{
	QUIT = 1,
	ECHO = 2
};

bool SendMsg(SOCKET clientSocket, uint8_t cmdID, const std::vector<uint8_t>& data = {}, bool raw = false, uint32_t length = 0)
{
	// Send command ID
	uint8_t cmd = static_cast<uint8_t>(cmdID);
	if (send(clientSocket, reinterpret_cast<char*>(&cmd), 1, 0) <= 0)
		return false;

	// For invalid command IDs, return immediately after sending the command
	if (cmdID != static_cast<uint8_t>(CommandID::QUIT) &&
		cmdID != static_cast<uint8_t>(CommandID::ECHO))
	{
		return false;
	}

	// Send text length
	uint32_t networkLength = 0;
	if (raw)
	{
		networkLength = htonl(static_cast<uint32_t>(length));
	}
	else
	{
		networkLength = htonl(static_cast<uint32_t>(data.size()));
	}

	if (send(clientSocket, reinterpret_cast<char*>(&networkLength), 4, 0) <= 0)
		return false;

	// Send text field
	size_t totalSent = 0;
	while (totalSent < data.size())
	{
		int bytesSent = send(clientSocket,
			reinterpret_cast<const char*>(data.data() + totalSent),
			static_cast<int>(data.size() - totalSent), 0);
		if (bytesSent <= 0) return false;
		totalSent += bytesSent;
	}
	return true;
}

void ReceiveEchoResponse(SOCKET clientSocket)
{
	// Skip command ID and length fields
	uint8_t cmdID;
	uint32_t length;
	if (recv(clientSocket, reinterpret_cast<char*>(&cmdID), 1, 0) <= 0 || recv(clientSocket, reinterpret_cast<char*>(&length), 4, 0) <= 0)
	{
		return;
	}
	length = ntohl(length);

	// Receive and print text field
	std::vector<uint8_t> buffer(length);
	size_t totalReceived = 0;
	while (totalReceived < length)
	{
		int bytesReceived = recv(clientSocket,
			reinterpret_cast<char*>(buffer.data() + totalReceived),
			static_cast<int>(length - totalReceived), 0);
		if (bytesReceived <= 0) break;
		totalReceived += bytesReceived;
	}

	std::cout.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
	std::cout << std::endl;
}

void ProcessUserInput(SOCKET clientSocket)
{
	std::string input;
	while (true)
	{
		std::getline(std::cin, input);

		if (input == "/q")
		{
			SendMsg(clientSocket, static_cast<uint8_t>(CommandID::QUIT));
			break;
		}
		else if (input.substr(0, 3) == "/t ")
		{
			// Handle hex input - data is already in network byte order
			std::vector<uint8_t> hexData;
			std::istringstream iss(input); // Skip "/t" prefix

			std::string hexStr;
			iss >> hexStr;
			iss >> hexStr;

			// Extract command ID
			std::string cmdIdStr;
			uint8_t cmdId = 0;
			cmdIdStr = hexStr.substr(0, 2);
			cmdId = static_cast<uint8_t>(std::stoi(cmdIdStr, nullptr, 16));

			// Extract text length
			std::string textLengthStr;
			uint32_t textLength = 0;
			textLengthStr = hexStr.substr(2, 8);
			textLength = static_cast<uint32_t>(std::stoi(textLengthStr, nullptr, 16));

			// Remove command ID and text length from hex string
			hexStr = hexStr.substr(10);

			// Process two characters at a time
			for (size_t i = 0; i < hexStr.length(); i += 2)
			{
				if (i + 1 < hexStr.length())
				{
					// Ensure we have 2 chars to process
					std::string hexByte = hexStr.substr(i, 2);
					uint8_t byte = static_cast<uint8_t>(std::stoi(hexByte, nullptr, 16));
					hexData.push_back(byte);
				}
			}

			// Send message and check if it was successful
			if (!SendMsg(clientSocket, cmdId, hexData, true, textLength))
			{
				break;
			}

			// Only try to receive response for valid commands
			if (cmdId == static_cast<uint8_t>(CommandID::ECHO))
			{
				ReceiveEchoResponse(clientSocket);
			}
		}
		else
		{
			// Normal text message
			std::vector<uint8_t> data(input.begin(), input.end());
			if (!SendMsg(clientSocket, static_cast<uint8_t>(CommandID::ECHO), data))
			{
				break;
			}
			ReceiveEchoResponse(clientSocket);
		}
	}
}

// This program requires one extra command-line parameter: a server hostname.
int main(int argc, char** argv)
{
	constexpr uint16_t port = 2048;

	// Get IP Address
	std::string host{};
	std::cout << "Server IP Address: ";
	std::getline(std::cin, host);

	std::cout << std::endl;

	// Get Port Number
	std::string portNumber;
	std::cout << "Server Port Number: ";
	std::getline(std::cin, portNumber);

	std::cout << std::endl;

	std::string portString = portNumber;


	// -------------------------------------------------------------------------
	// Start up Winsock, asking for version 2.2.
	//
	// WSAStartup()
	// -------------------------------------------------------------------------

	// This object holds the information about the version of Winsock that we
	// are using, which is not necessarily the version that we requested.
	WSADATA wsaData{};
	SecureZeroMemory(&wsaData, sizeof(wsaData));

	// Initialize Winsock. You must call WSACleanup when you are finished.
	// As this function uses a reference counter, for each call to WSAStartup,
	// you must call WSACleanup or suffer memory issues.
	int errorCode = WSAStartup(MAKEWORD(WINSOCK_VERSION, WINSOCK_SUBVERSION), &wsaData);
	if (NO_ERROR != errorCode)
	{
		std::cerr << "WSAStartup() failed." << std::endl;
		return errorCode;
	}


	// -------------------------------------------------------------------------
	// Resolve a server host name into IP addresses (in a singly-linked list).
	//
	// getaddrinfo()
	// -------------------------------------------------------------------------

	// Object hints indicates which protocols to use to fill in the info.
	addrinfo hints{};
	SecureZeroMemory(&hints, sizeof(hints));
	hints.ai_family = AF_INET;			// IPv4
	hints.ai_socktype = SOCK_STREAM;	// Reliable delivery
	// Could be 0 to autodetect, but reliable delivery over IPv4 is always TCP.
	hints.ai_protocol = IPPROTO_TCP;	// TCP

	addrinfo* info = nullptr;
	errorCode = getaddrinfo(host.c_str(), portString.c_str(), &hints, &info);
	if ((NO_ERROR != errorCode) || (nullptr == info))
	{
		std::cerr << "getaddrinfo() failed." << std::endl;
		WSACleanup();
		return errorCode;
	}


	// -------------------------------------------------------------------------
	// Create a socket and attempt to connect to the first resolved address.
	//
	// socket()
	// connect()
	// -------------------------------------------------------------------------

	SOCKET clientSocket = socket(
		info->ai_family,
		info->ai_socktype,
		info->ai_protocol);
	if (INVALID_SOCKET == clientSocket)
	{
		std::cerr << "socket() failed." << std::endl;
		freeaddrinfo(info);
		WSACleanup();
		return RETURN_CODE_2;
	}

	errorCode = connect(
		clientSocket,
		info->ai_addr,
		static_cast<int>(info->ai_addrlen));
	if (SOCKET_ERROR == errorCode)
	{
		std::cerr << "connect() failed." << std::endl;
		freeaddrinfo(info);
		closesocket(clientSocket);
		WSACleanup();
		return RETURN_CODE_3;
	}

	ProcessUserInput(clientSocket);

	errorCode = shutdown(clientSocket, SD_SEND);
	if (SOCKET_ERROR == errorCode)
	{
		std::cerr << "shutdown() failed." << std::endl;
	}
	closesocket(clientSocket);


	// -------------------------------------------------------------------------
	// Clean-up after Winsock.
	//
	// WSACleanup()
	// -------------------------------------------------------------------------

	WSACleanup();
}
