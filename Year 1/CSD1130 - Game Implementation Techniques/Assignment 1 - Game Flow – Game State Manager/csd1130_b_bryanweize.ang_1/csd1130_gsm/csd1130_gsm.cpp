/* Start Header ************************************************************************/
/*!
\file csd1130_gsm.cpp
\author Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par bryanweize.ang\@digipen.edu
\date Jan 22, 2024
\brief This source file contains the 'main' function. Program execution begins and ends there.
Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents
without the prior written consent of DigiPen Institute of
Technology is prohibited.
*/
/* End Header **************************************************************************/

#include "pch.h"
#include "System.h"
#include "GameStateManager.h"
#include "Input.h"

int main()
{
    //Systems initialize
    SystemInitialize();
    
    //GSM initialize
    GSM_Initialize(GS_LEVEL1);

    while (current != GS_QUIT)
    {
        //game state is not restarted
        if (current != GS_RESTART)
        {
            //GSM update
            GSM_Update();

            //load the current game state assets
            fpLoad();
        }
        else
        { 
            next = previous;
            current = previous;
        }

        //initialize the current game state
        fpInitialize();

        //the game loop
        while(current == next)
        {
            //handle user input
            Input_Handle();

            //update the current game state
            fpUpdate();

            //draw the current game state
            fpDraw();
        }

        //free and reset the current game state
        fpFree();

        // changing of current game state
        if (next != GS_RESTART)
        {
            //unload the current game state assets
            fpUnload();
        }

        previous = current;
        current = next;
    }

    //Systems exit (terminate)
    SystemTerminate();

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
