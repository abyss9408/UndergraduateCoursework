/* Start Header ************************************************************************/
/*!
\file GameStateManager.h
\author Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par bryanweize.ang\@digipen.edu
\date Jan 22, 2024
\brief This source file declares the Initialize and Update functions of the Game State
Manager.
Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents
without the prior written consent of DigiPen Institute of
Technology is prohibited.
*/
/* End Header **************************************************************************/
#pragma once

typedef void(*FP)(void);

extern int current, previous, next;

extern FP fpLoad, fpInitialize, fpUpdate, fpDraw, fpFree, fpUnload;

// ----------------------------------------------------------------------------
// This function initializes the Game State Manager
// It should be called once after the system is initialized
// It initializes the current, previous and next game state to the starting state
// ----------------------------------------------------------------------------
void GSM_Initialize(int startingState);

// ----------------------------------------------------------------------------
// This function updates the Game State Manager
// It should be called when the application is running
// It updates the function pointers based on the current level
// ----------------------------------------------------------------------------
void GSM_Update();