/* Start Header ************************************************************************/
/*!
\file Level1.cpp
\author Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par bryanweize.ang\@digipen.edu
\date Jan 22, 2024
\brief This source file implements the Load, Initialize, Update, Draw, Free and Unload
functions of Level1.
Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents
without the prior written consent of DigiPen Institute of
Technology is prohibited.
*/
/* End Header **************************************************************************/
#include "pch.h"
#include "Level1.h"
#include "GameStateManager.h"
#include <fstream>

int Level1_Counter{ 0 };


void Level1_Load()
{
	std::cout << "Level1:Load\n";

	//initialize Level1_Counter using txt file
	std::ifstream read_counter;
	read_counter.open("Level1_Counter.txt");
	read_counter >> Level1_Counter;
	read_counter.close();
}

void Level1_Initialize()
{
	std::cout << "Level1:Initialize\n";
}


void Level1_Update()
{
	std::cout << "Level1:Update\n";
	Level1_Counter--;
	if (Level1_Counter == 0)
	{
		next = GS_LEVEL2;
	}
}

void Level1_Draw()
{
	std::cout << "Level1:Draw\n";
}

void Level1_Free()
{
	std::cout << "Level1:Free\n";
}

void Level1_Unload()
{
	std::cout << "Level1:Unload\n";
}