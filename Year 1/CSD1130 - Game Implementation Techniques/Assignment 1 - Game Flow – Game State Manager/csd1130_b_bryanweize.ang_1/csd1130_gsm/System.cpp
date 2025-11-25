/* Start Header ************************************************************************/
/*!
\file System.cpp
\author Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par bryanweize.ang\@digipen.edu
\date Jan 22, 2024
\brief This source file implements the Intialize and Terminate functions of the System
Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents
without the prior written consent of DigiPen Institute of
Technology is prohibited.
*/
/* End Header **************************************************************************/
#include "pch.h"
#include <iostream>

void SystemInitialize()
{
	std::cout << "System:Initialize\n";
}

void SystemTerminate()
{
	std::cout << "System:Exit\n";
}