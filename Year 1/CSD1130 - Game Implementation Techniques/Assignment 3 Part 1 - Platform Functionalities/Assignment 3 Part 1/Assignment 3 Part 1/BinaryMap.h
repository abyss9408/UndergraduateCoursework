/******************************************************************************/
/*!
\file		BinaryMap.h
\author 	DigiPen
\par    	email: digipen\@digipen.edu ... ...
\date   	February 1, 20xx
\brief		This file contains six functions declared that load and unload map data from
			text files and check/determine collisions using a Binary Collision Map.

Copyright (C) 20xx DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior 
written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#ifndef BINARY_MAP_H_
#define BINARY_MAP_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


const int	COLLISION_LEFT		= 0x00000001;	//0001
const int	COLLISION_RIGHT		= 0x00000002;	//0010
const int	COLLISION_TOP		= 0x00000004;	//0100
const int	COLLISION_BOTTOM	= 0x00000008;	//1000


enum TYPE_OBJECT
{
	TYPE_OBJECT_EMPTY,			//0
	TYPE_OBJECT_COLLISION,		//1
	TYPE_OBJECT_HERO,			//2
	TYPE_OBJECT_ENEMY1,			//3
	TYPE_OBJECT_COIN			//4
};


int		GetCellValue(int X, int Y);
int		CheckInstanceBinaryMapCollision(float PosX, float PosY, 
										float scaleX, float scaleY);
void	SnapToCell(float *Coordinate);
int		ImportMapDataFromFile(const char *FileName);
void	FreeMapData(void);
void	PrintRetrievedInformation(void);

#endif // BINARY_MAP_H_