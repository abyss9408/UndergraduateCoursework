/******************************************************************************/
/*!
\file		main.cpp
\author 	DigiPen
\par    	email: digipen\@digipen.edu ... ...
\date   	February 1, 20xx
\brief		This is a test driver file, to check on and validate all the different 
			binary map collisions functionalities

Copyright (C) 20xx DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#include "BinaryMap.h"

#include <iostream>
#include <stdlib.h>

/******************************************************************************/
/*!

 */
 /******************************************************************************/
int main()
{
	int Flag = 0;
	float f = 0.0f;

	
	//Test here if you can read "Export.txt"
	if (!ImportMapDataFromFile("Exported.txt"))
	{
		printf("Could not import Exported.txt file\n");
		return 0;
	}




	
	//Printing the information read by ImportMapDataFromFile
	PrintRetrievedInformation();

	//Testing "GetCellValue
	//To console
	printf("\n\nTesting GetCellValue\n");
	printf("Cell (0,0) = %i\n", GetCellValue(0, 0));	//You should get 1
	printf("Cell (1,1) = %i\n", GetCellValue(1, 1));	//You should get 1	
	printf("Cell (1,2) = %i\n", GetCellValue(1, 2));	//You should get 0	
	printf("Cell (1,2) = %i\n", GetCellValue(1, 2));	//You should get 0	
	printf("Cell (3,4) = %i\n", GetCellValue(3, 4));	//You should get 1	
	printf("Cell (-1,1) = %i\n", GetCellValue(-1, 1));	//You should get 0	
	printf("Cell (1,-1) = %i\n", GetCellValue(1, -1));	//You should get 0	
	printf("Cell (5,1) = %i\n", GetCellValue(5, 1));	//You should get 0	
	printf("Cell (1,5) = %i\n", GetCellValue(1, 5));	//You should get 0

	



	//Testing "SnapToCell"
	//To console
	printf("\n\nTesting SnapToCell\n");
	f = 0.2f;
	SnapToCell(&f);
	printf("0.2 got snapped to: %f\n", f);				//You should get 0.5

	f = 2.3f;
	SnapToCell(&f);
	printf("2.3 got snapped to: %f\n", f);				//You should get 2.5

	f = 1.7f;
	SnapToCell(&f);
	printf("1.7 got snapped to: %f\n", f);				//You should get 1.5

	f = 5.4f;
	SnapToCell(&f);
	printf("5.4 got snapped to: %f\n", f);				//You should get 5.5

	f = 4.9f;
	SnapToCell(&f);
	printf("4.9 got snapped to: %f\n", f);				//You should get 4.5

		



	//Testing "CheckInstanceBinaryMapCollision"
	//To console
	printf("\n\nTesting CheckInstanceBinaryMapCollision\n");
	Flag = 0;
	Flag = CheckInstanceBinaryMapCollision(1.7f, 2.2f, 1.0f, 1.0f);
	printf("Flag for 1.7f, 2.2f, 1.0f, 1.0f is: %i\n", Flag);		//You should get 11

	Flag = 0;
	Flag = CheckInstanceBinaryMapCollision(3.4f, 1.7f, 1.0f, 1.0f);
	printf("Flag for 3.4f, 1.7f, 1.0f, 1.0f is: %i\n", Flag);		//You should get 1

	Flag = 0;
	Flag = CheckInstanceBinaryMapCollision(1.2f, 3.8f, 1.0f, 1.0f);
	printf("Flag for 1.2f, 3.8f, 1.0f, 1.0f is: %i\n", Flag);		//You should get 15
	
		



	FreeMapData();

	return 1;
}