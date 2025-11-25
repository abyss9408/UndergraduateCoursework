/******************************************************************************/
/*!
\file		main.h
\author 	Elie Hosry, ehosry, 00000000
\par    	ehosry@digipen.edu
\date   	Jul 03 2023
\brief  	This file includes basic main header information

Copyright (C) 2023 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/


#ifndef CSD1130_MAIN_H_
#define CSD1130_MAIN_H_

//------------------------------------
// Globals

extern float	g_dt;
extern double	g_appTime;

// ---------------------------------------------------------------------------
// includes

#include "AEEngine.h"
#include "Math.h"

#include <iostream>
#include <fstream>
#include <string>

#include "GameStateMgr.h"
#include "GameState_Cage.h"
#include "Collision.h"
#include "Vector2D.h"
#include "Matrix3x3.h"


extern s8	fontId;

void TestMyMath();

#endif // CSD1130_MAIN_H_










