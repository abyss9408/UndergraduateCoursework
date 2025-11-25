/******************************************************************************/
/*!
\file		BinaryMap.cpp
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 17, 2024
\brief		This file contains six functions that load and unload map data from
			text files and check/determine collisions using a Binary Collision Map.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior 
written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/


#include "BinaryMap.h"
#include <fstream>
#include <iostream>
#include <string>


/*The number of horizontal elements*/
static int BINARY_MAP_WIDTH;

/*The number of vertical elements*/
static int BINARY_MAP_HEIGHT;

/*This will contain all the data of the map, which will be retreived from a file
when the "ImportMapDataFromFile" function is called*/
static int **MapData;

/*This will contain the collision data of the binary map. It will be filled in the 
"ImportMapDataFromFile" after filling "MapData". Basically, if an array element 
in MapData is 1, it represents a collision cell, any other value is a non-collision
cell*/
static int **BinaryCollisionArray;


/******************************************************************************/
/*!
	This function opens the file name "FileName" and retrieves all the map data.
	It allocates memory for the 2 arrays: MapData & BinaryCollisionArray
	The first line in this file is the width of the map.
	The second line in this file is the height of the map.
	The remaining part of the file is a series of numbers
	Each number represents the ID (or value) of a different element in the 
	double dimensionaly array.

	Example:

	Width 5
	Height 5
	1 1 1 1 1
	1 1 1 3 1
	1 4 2 0 1
	1 0 0 0 1
	1 1 1 1 1


	After importing the above data, "MapData" and " BinaryCollisionArray" 
	should be

	1 1 1 1 1
	1 1 1 3 1
	1 4 2 0 1
	1 0 0 0 1
	1 1 1 1 1

	and

	1 1 1 1 1
	1 1 1 0 1
	1 0 0 0 1
	1 0 0 0 1
	1 1 1 1 1

	respectively.
	
	Finally, the function returns 1 if the file named "FileName" exists, 
	otherwise it returns 0
 */
/******************************************************************************/
int ImportMapDataFromFile(const char *FileName)
{
	std::ifstream ifs(FileName);

	if (!ifs.is_open())
	{
		return 0;
	}

	std::string width, height;

	// read width and height
	ifs >> width >> BINARY_MAP_WIDTH;
	ifs >> height >> BINARY_MAP_HEIGHT;

	// dynamically allocate memory for 2D MapData and BinaryCollisionArray
	MapData = new int* [BINARY_MAP_WIDTH];
	BinaryCollisionArray = new int* [BINARY_MAP_WIDTH];

	for (int i{}; i < BINARY_MAP_WIDTH; i++)
	{
		MapData[i] = new int[BINARY_MAP_HEIGHT];
		BinaryCollisionArray[i] = new int[BINARY_MAP_HEIGHT];
	}

	// read data from file into MapData and BinaryCollisionArray
	for (int i{}; i < BINARY_MAP_HEIGHT; i++)
	{
		for (int j{}; j < BINARY_MAP_WIDTH; j++)
		{
			ifs >> MapData[j][i];
			BinaryCollisionArray[j][i] = MapData[j][i];

			if (BinaryCollisionArray[j][i] != 1 && BinaryCollisionArray[j][i] != 0)
			{
				BinaryCollisionArray[j][i] = 0;
			}
		}
	}

	ifs.close();


	return 1;
}

/******************************************************************************/
/*!
	This function frees the memory that was allocated for the 2 arrays MapData 
	& BinaryCollisionArray which was allocated in the "ImportMapDataFromFile" 
	function
 */
/******************************************************************************/
void FreeMapData(void)
{
	for (int i{}; i < BINARY_MAP_WIDTH; i++)
	{
		delete[] MapData[i];
		delete[] BinaryCollisionArray[i];
	}

	delete[] MapData;
	delete[] BinaryCollisionArray;
}

/******************************************************************************/
/*!
	This function prints out the content of the 2D array “MapData”
	You must print to the console, the same information you are reading from "Exported.txt" file
	Follow exactly the same format of the file, including the print of the width and the height
	Add spaces and end lines at convenient places
 */
/******************************************************************************/
void PrintRetrievedInformation(void)
{
	std::cout << " Width " << BINARY_MAP_WIDTH << '\n';
	std::cout << "Height " << BINARY_MAP_HEIGHT << '\n';

	for (int i{}; i < BINARY_MAP_HEIGHT; ++i)
	{
		for (int j{}; j < BINARY_MAP_WIDTH; ++j)
		{
			std::cout << MapData[j][i] << ' ';
		}
		std::cout << '\n';
	}
}

/******************************************************************************/
/*!
	This function retrieves the value of the element (X;Y) in BinaryCollisionArray.
	Before retrieving the value, it should check that the supplied X and Y values
	are not out of bounds (in that case return 0)
 */
/******************************************************************************/
int GetCellValue(int X, int Y)
{
	if (X < 0 || Y < 0 || X >= BINARY_MAP_WIDTH || Y >= BINARY_MAP_HEIGHT)
	{
		return 0;
	}
	
	return BinaryCollisionArray[X][Y];
}

/******************************************************************************/
/*!
	This function snaps the value sent as parameter to the center of the cell.
	It is used when a sprite is colliding with a collision area from one 
	or more side.
	To snap the value sent by "Coordinate", find its integral part by type 
	casting it to an integer, then add 0.5 (which is half the cell's width 
	or height)
 */
/******************************************************************************/
void SnapToCell(float *Coordinate)
{
	*Coordinate = static_cast<int>(*Coordinate) + 0.5f;
}

/******************************************************************************/
/*!
	This function creates 2 hot spots on each side of the object instance, 
	and checks if each of these hot spots is in a collision area (which means 
	the cell if falls in has a value of 1).
	At the beginning of the function, a "Flag" integer should be initialized to 0.
	Each time a hot spot is in a collision area, its corresponding bit 
	in "Flag" is set to 1.
	Finally, the function returns the integer "Flag"
	The position of the object instance is received as PosX and PosY
	The size of the object instance is received as scaleX and scaleY

	Note: This function assume the object instance's size is 1 by 1 
		  (the size of 1 tile)

	Creating the hotspots:
		-Handle each side separately.
		-2 hot spots are needed for each collision side.
		-These 2 hot spots should be positioned on 1/4 above the center 
		and 1/4 below the center

	Example: Finding the hots spots on the left side of the object instance

	float x1, y1, x2, y2;

	-hotspot 1
	x1 = PosX + scaleX/2	To reach the right side
	y1 = PosY + scaleY/4	To go up 1/4 of the height
	
	-hotspot 2
	x2 = PosX + scaleX/2	To reach the right side
	y2 = PosY - scaleY/4	To go down 1/4 of the height
 */
/******************************************************************************/
int CheckInstanceBinaryMapCollision(float PosX, float PosY, 
									float scaleX, float scaleY)
{
	// collision flag;
	int FLAG{ 0b0000 };

	// collision side values
	const int COLLISION_LEFT{ 0b0001 };
	const int COLLISION_RIGHT{ 0b0010 };
	const int COLLISION_TOP{ 0b0100 };
	const int COLLISION_BOTTOM{ 0b1000 };

	// x,y coordinates for hotspots
	float x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8;

	// top side
	// left top side
	x1 = PosX - scaleX / 4;
	y1 = PosY + scaleY / 2;

	// right top side
	x2 = PosX + scaleX / 4;
	y2 = PosY + scaleY / 2;

	if (BinaryCollisionArray[static_cast<int>(x1)][static_cast<int>(y1)] ||
		BinaryCollisionArray[static_cast<int>(x2)][static_cast<int>(y2)])
	{
		FLAG |= COLLISION_TOP;
	}

	// bottom side
	// left bottom side
	x3 = PosX - scaleX / 4;
	y3 = PosY - scaleY / 2;

	// right bottom side
	x4 = PosX + scaleX / 4;
	y4 = PosY - scaleY / 2;

	if (BinaryCollisionArray[static_cast<int>(x3)][static_cast<int>(y3)] ||
		BinaryCollisionArray[static_cast<int>(x4)][static_cast<int>(y4)])
	{
		FLAG |= COLLISION_BOTTOM;
	}

	// left side
	// upper left side
	x5 = PosX - scaleX / 2;
	y5 = PosY + scaleY / 4;

	// lower left side
	x6 = PosX - scaleX / 2;
	y6 = PosY - scaleY / 4;

	if (BinaryCollisionArray[static_cast<int>(x5)][static_cast<int>(y5)] ||
		BinaryCollisionArray[static_cast<int>(x6)][static_cast<int>(y6)])
	{
		FLAG |= COLLISION_LEFT;
	}

	// right side
	// upper right side
	x7 = PosX + scaleX / 2;
	y7 = PosY + scaleY / 4;

	// lower right side
	x8 = PosX + scaleX / 2;
	y8 = PosY - scaleY / 4;

	if (BinaryCollisionArray[static_cast<int>(x7)][static_cast<int>(y7)] ||
		BinaryCollisionArray[static_cast<int>(x8)][static_cast<int>(y8)])
	{
		FLAG |= COLLISION_RIGHT;
	}

	return FLAG;
}