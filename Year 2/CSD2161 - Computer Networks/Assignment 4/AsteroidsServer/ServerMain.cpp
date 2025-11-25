/******************************************************************************/
/*!
\file		ServerMain.cpp
\author 	Low Yue Jun
\par    	email: yuejun.low\@digipen.edu
\date   	March 29, 2025
\brief		This source file contains the entry point for the standalone server.

Copyright (C) 2025 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#include "StandaloneServer.h"
#include <iostream>
#include <conio.h>

int main(int argc, char* argv[]) {
    std::cout << "Spaceships Multiplayer Server" << std::endl;
    std::cout << "------------------------------" << std::endl;

    // Create the server
    StandaloneServer server;

    // Initialize the server
    if (!server.Initialize()) {
        std::cerr << "Failed to initialize server" << std::endl;
        std::cout << "Press any key to exit..." << std::endl;
        _getch();
        return 1;
    }

    std::cout << "Server initialized and ready for connections" << std::endl;
    std::cout << "Press 'Q' at any time to shut down the server" << std::endl;

    // Start a separate thread for the server
    std::thread serverThread([&server]() {
        server.Run();
    });

    // Wait for 'Q' keypress
    bool running = true;
    while (running) {
        if (_kbhit()) {
            int key = _getch();
            if (key == 'q' || key == 'Q') {
                running = false;
            }
        }

        // Sleep to reduce CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Shutdown the server
    std::cout << "Shutting down server..." << std::endl;
    server.Shutdown();

    // Wait for the server thread to finish
    if (serverThread.joinable()) {
        serverThread.join();
    }

    std::cout << "Server shut down" << std::endl;
    std::cout << "Press any key to exit..." << std::endl;
    _getch();

    return 0;
}