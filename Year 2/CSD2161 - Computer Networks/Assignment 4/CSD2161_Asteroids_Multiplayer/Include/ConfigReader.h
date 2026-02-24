/******************************************************************************/
/*!
\file   ConfigReader.h
\brief  Parses network.cfg into a NetworkConfig struct
*/
/******************************************************************************/
#pragma once

#include <cstdint>
#include <string>
#include <fstream>

struct NetworkConfig
{
    std::string serverIp   = "127.0.0.1";
    uint16_t    serverPort = 9999;
    std::string playerName = "Player1";
    int         minPlayers = 2;
    int         maxPlayers = 4;
};

class ConfigReader
{
public:
    static NetworkConfig Load(const std::string& filename = "network.cfg")
    {
        NetworkConfig cfg;
        std::ifstream file(filename);
        if (!file.is_open())
            return cfg;

        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty() || line[0] == '#')
                continue;

            auto pos = line.find('=');
            if (pos == std::string::npos)
                continue;

            std::string key = line.substr(0, pos);
            std::string val = line.substr(pos + 1);

            // Trim whitespace
            auto trim = [](std::string& s)
            {
                const std::string ws = " \t\r\n";
                size_t start = s.find_first_not_of(ws);
                size_t end   = s.find_last_not_of(ws);
                s = (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
            };
            trim(key);
            trim(val);

            if      (key == "server_ip")   cfg.serverIp   = val;
            else if (key == "server_port") cfg.serverPort = static_cast<uint16_t>(std::stoi(val));
            else if (key == "player_name") cfg.playerName = val;
            else if (key == "min_players") cfg.minPlayers = std::stoi(val);
            else if (key == "max_players") cfg.maxPlayers = std::stoi(val);
        }
        return cfg;
    }
};
