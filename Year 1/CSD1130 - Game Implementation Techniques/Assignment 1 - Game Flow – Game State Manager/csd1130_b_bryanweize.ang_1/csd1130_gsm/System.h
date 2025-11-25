/* Start Header ************************************************************************/
/*!
\file System.h
\author Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par bryanweize.ang\@digipen.edu
\date Jan 22, 2024
\brief This header file declares the Intialize and Terminate functions of the System
Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents
without the prior written consent of DigiPen Institute of
Technology is prohibited.
*/
/* End Header **************************************************************************/
#pragma once

// ----------------------------------------------------------------------------
// This function initializes the System
// It should be called once after the application is launched
// It initializes the system components and allocates memory
// ----------------------------------------------------------------------------
void SystemInitialize();

// ----------------------------------------------------------------------------
// This function terminates the System
// It should be called once after the application is exited
// It terminates the system components and frees up allocated memory
// ----------------------------------------------------------------------------
void SystemTerminate();