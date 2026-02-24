/******************************************************************************/
/*!
\file   ServerMain.cpp
\brief  Entry point for the standalone Asteroids server
*/
/******************************************************************************/
#include "UDPSocket.h"
#include "StandaloneServer.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <cstdint>

// ---------------------------------------------------------------------------
// Minimal config reader (no AlphaEngine available here)
struct SrvConfig
{
    uint16_t port       = 9999;
    int      minPlayers = 2;
    int      maxPlayers = 4;
};

static SrvConfig LoadConfig(const std::string& filename = "network.cfg")
{
    SrvConfig cfg;
    std::ifstream f(filename);
    if (!f.is_open()) return cfg;

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string k = line.substr(0, pos);
        std::string v = line.substr(pos + 1);

        auto trim = [](std::string& s)
        {
            const std::string ws = " \t\r\n";
            auto a = s.find_first_not_of(ws);
            auto b = s.find_last_not_of(ws);
            s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
        };
        trim(k); trim(v);

        if      (k == "server_port") cfg.port       = static_cast<uint16_t>(std::stoi(v));
        else if (k == "min_players") cfg.minPlayers  = std::stoi(v);
        else if (k == "max_players") cfg.maxPlayers  = std::stoi(v);
    }
    return cfg;
}

// ---------------------------------------------------------------------------
int main()
{
    if (!UDPSocket::InitWinsock())
    {
        printf("[Server] WinSock init failed\n");
        return 1;
    }

    SrvConfig cfg = LoadConfig();
    printf("[Server] Config: port=%u  min=%d  max=%d\n",
           cfg.port, cfg.minPlayers, cfg.maxPlayers);

    StandaloneServer server(cfg.port, cfg.minPlayers, cfg.maxPlayers);
    server.Run();   // blocks until Ctrl-C or error

    UDPSocket::ShutdownWinsock();
    return 0;
}
