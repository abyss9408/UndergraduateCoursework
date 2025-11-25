/* Start Header ************************************************************************/
/*!
\file Level2.h
\author Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par bryanweize.ang\@digipen.edu
\date Jan 22, 2024
\brief This header file declares the Load, Initialize, Update, Draw, Free and Unload
functions of Level2.
Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents
without the prior written consent of DigiPen Institute of
Technology is prohibited.
*/
/* End Header **************************************************************************/
#pragma once

// ----------------------------------------------------------------------------
// This function loads all necessary assets in Level2
// It should be called once before the start of the level
// It loads assets like textures, meshes and music files etc¡­
// ----------------------------------------------------------------------------
void Level2_Load();

// ----------------------------------------------------------------------------
// This function prepares Level2's data to be used for the first time
// It should be called once before the start of the level or when the level is
// restarted
// It initializes Level2's data
// ----------------------------------------------------------------------------
void Level2_Initialize();

// ----------------------------------------------------------------------------
// This function updates Level2's data
// It should be called every frame
// It updates Level2's data based on several factors like user input, time, or
// gameplay logic
// ----------------------------------------------------------------------------
void Level2_Update();

// ----------------------------------------------------------------------------
// This function draws Level2's data
// It should be called every frame
// It draws by sending Level2's data to the graphics engine component
// ----------------------------------------------------------------------------
void Level2_Draw();

// ----------------------------------------------------------------------------
// This function cleans up Level2
// It should be called once after the end of the level
// It cleans the instances and make Level2 ready to be unloaded or initialized
// again
// ----------------------------------------------------------------------------
void Level2_Free();

// ----------------------------------------------------------------------------
// This function unloads all assets in Level2
// It should be called once after the end of the level and the level is not
// being restarted
// It unloads assets like textures, meshes and music files etc¡­
// ----------------------------------------------------------------------------
void Level2_Unload();