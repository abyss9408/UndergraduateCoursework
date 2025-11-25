/******************************************************************************/
/*!
\file		main.h
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 05, 2024
\brief		This header file declares globals and includes

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/


#ifndef CSD1130_MAIN_H_
#define CSD1130_MAIN_H_

//------------------------------------
// Globals

extern float	g_dt;
extern double	g_appTime;
extern signed char	g_font;
extern int		level;

// ---------------------------------------------------------------------------
// includes

#include "AEEngine.h"
#include "Math.h"
#include <fstream>
#include <string>

#include "GameStateMgr.h"
#include "GameState_Menu.h"
#include "GameState_Platform.h"
#include "Collision.h"


#endif // CSD1130_MAIN_H_










