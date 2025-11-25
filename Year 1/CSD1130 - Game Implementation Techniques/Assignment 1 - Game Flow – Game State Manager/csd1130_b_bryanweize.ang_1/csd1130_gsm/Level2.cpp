/* Start Header ************************************************************************/
/*!
\file Level2.cpp
\author Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par bryanweize.ang\@digipen.edu
\date Jan 22, 2024
\brief This source file implements the Load, Initialize, Update, Draw, Free and Unload
functions of Level2.
Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents
without the prior written consent of DigiPen Institute of
Technology is prohibited.
*/
/* End Header **************************************************************************/
#include "pch.h"
#include "Level2.h"
#include <fstream>
#include "GameStateManager.h"

int Level2_Counter{ 0 }, Level2_Lives{ 0 };

void Level2_Load()
{
	std::cout << "Level2:Load\n";

	//initialize Level2_Lives using txt file
	std::ifstream read_lives;
	read_lives.open("Level2_Lives.txt");
	read_lives >> Level2_Lives;
	read_lives.close();
}

void Level2_Initialize()
{
	std::cout << "Level2:Initialize\n";

	//initialize Level2_Counter using txt file
	std::ifstream read_counter;
	read_counter.open("Level2_Counter.txt");
	read_counter >> Level2_Counter;
	read_counter.close();
}

void Level2_Update()
{
	std::cout << "Level2:Update\n";
	Level2_Counter--;

	if (Level2_Counter == 0)
	{
		Level2_Lives--;

		// quit the application when lives is decremented to 0
		if (Level2_Lives == 0)
		{
			next = GS_QUIT;
		}
		else //restart the game state when lives is decremented to 1
		{
			next = GS_RESTART;
		}
	}

}

void Level2_Draw()
{
	std::cout << "Level2:Draw\n";
}

void Level2_Free()
{
	std::cout << "Level2:Free\n";
}

void Level2_Unload()
{
	std::cout << "Level2:Unload\n";
}