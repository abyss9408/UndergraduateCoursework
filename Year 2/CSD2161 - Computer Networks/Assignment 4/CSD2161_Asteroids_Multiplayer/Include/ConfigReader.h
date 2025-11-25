/******************************************************************************/
/*!
\file		ConfigReader.h
\author 	Michael Henry Lazaroo
\par    	email: m.lazaroo\@digipen.edu
\date   	March 30, 2025
\brief		This header file declares a simple configuration file reader class.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#ifndef CONFIG_READER_H_
#define CONFIG_READER_H_

#include <string>
#include <map>
#include <fstream>
#include <iostream>

class ConfigReader {
public:
    // Load configuration from a file
    bool LoadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        m_values.clear();
        std::string line;

        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // Find the equal sign
            size_t equalPos = line.find('=');
            if (equalPos != std::string::npos) {
                std::string key = line.substr(0, equalPos);
                std::string value = line.substr(equalPos + 1);

                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                m_values[key] = value;
            }
        }

        file.close();
        return true;
    }

    // Get a string value from the configuration
    std::string GetString(const std::string& key, const std::string& defaultValue) const {
        auto it = m_values.find(key);
        if (it != m_values.end()) {
            return it->second;
        }
        return defaultValue;
    }

    // Get an integer value from the configuration
    int GetInt(const std::string& key, int defaultValue) const {
        auto it = m_values.find(key);
        if (it != m_values.end()) {
            try {
                return std::stoi(it->second);
            }
            catch (const std::exception& e) {
                std::cerr << "Error converting value to int: " << e.what() << std::endl;
            }
        }
        return defaultValue;
    }

    // Set a string value in the configuration
    void SetString(const std::string& key, const std::string& value) {
        m_values[key] = value;
    }

    // Save configuration to a file
    bool SaveToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        for (const auto& pair : m_values) {
            file << pair.first << " = " << pair.second << std::endl;
        }

        file.close();
        return true;
    }

private:
    std::map<std::string, std::string> m_values;
};

#endif // CONFIG_READER_H_