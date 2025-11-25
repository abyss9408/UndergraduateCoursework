/******************************************************************************/
/*!
\file		Collision.cpp
\author 	Low Yue Jun
\par    	email: yuejun.low\@digipen.edu
\date   	March 30, 2025
\brief		This source file implements the CollisionIntersection_RectRect
			function.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
 /******************************************************************************/

#include "main.h"

static bool StaticCollisionIntersection_RectRect(const AABB& a, const AABB& b)
{
	if (a.max.x < b.min.x || a.min.x > b.max.x)
	{
		return false;
	}
	if (a.max.y < b.min.y || a.min.y > b.max.y)
	{
		return false;
	}
	return true;
}

/**************************************************************************/
/*!
	Dynamic collision check between two rectangles
	*/
	/**************************************************************************/
bool CollisionIntersection_RectRect(const AABB& aabb1,          //Input
	const AEVec2& vel1,         //Input 
	const AABB& aabb2,          //Input 
	const AEVec2& vel2,         //Input
	float& firstTimeOfCollision) //Output: the calculated value of tFirst, below, must be returned here
{
	UNREFERENCED_PARAMETER(aabb1);
	UNREFERENCED_PARAMETER(vel1);
	UNREFERENCED_PARAMETER(aabb2);
	UNREFERENCED_PARAMETER(vel2);
	UNREFERENCED_PARAMETER(firstTimeOfCollision);


	/*
	Implement the collision intersection over here.

	The steps are:
	Step 1: Check for static collision detection between rectangles (static: before moving).
				If the check returns no overlap, you continue with the dynamic collision test
					with the following next steps 2 to 5 (dynamic: with velocities).
				Otherwise you return collision is true, and you stop.

	Step 2: Initialize and calculate the new velocity of Vb
			tFirst = 0  //tFirst variable is commonly used for both the x-axis and y-axis
			tLast = dt  //tLast variable is commonly used for both the x-axis and y-axis

	Step 3: Working with one dimension (x-axis).
			if(Vb < 0)
				case 1
				case 4
			else if(Vb > 0)
				case 2
				case 3
			else //(Vb == 0)
				case 5

			case 6

	Step 4: Repeat step 3 on the y-axis

	Step 5: Return true: the rectangles intersect

	*/
	if (StaticCollisionIntersection_RectRect(aabb1, aabb2))
	{
		firstTimeOfCollision = 0.0f;
		return true;
	}

	AEVec2 vRel;
	float tLast;
	vRel.x = vel2.x - vel1.x;
	vRel.y = vel2.y - vel1.y;
	firstTimeOfCollision = 0.0f;
	tLast = static_cast<float>(AEFrameRateControllerGetFrameTime());

	// x-axis
	if (vRel.x < 0.0f)
	{
		// case 1
		if (aabb1.min.x > aabb2.max.x)
		{
			return false;
		}

		// case 4
		if (aabb1.max.x < aabb2.min.x)
		{
			firstTimeOfCollision = AEMax((aabb1.max.x - aabb2.min.x) / vRel.x, firstTimeOfCollision);
		}
		if (aabb1.min.x < aabb2.max.x)
		{
			tLast = AEMin((aabb1.min.x - aabb2.max.x) / vRel.x, tLast);
		}
	}
	else if (vRel.x > 0.0f)
	{
		// case 3
		if (aabb2.min.x > aabb1.max.x)
		{
			return false;
		}

		// case 2
		if (aabb2.max.x < aabb1.min.x)
		{
			firstTimeOfCollision = AEMax((aabb1.min.x - aabb2.max.x) / vRel.x, firstTimeOfCollision);
		}
		if (aabb1.max.x > aabb2.min.x)
		{
			tLast = AEMin((aabb1.max.x - aabb2.min.x) / vRel.x, tLast);
		}
	}
	else
	{
		if (aabb1.max.x < aabb2.min.x)
		{
			return false;
		}
		else if (aabb1.min.x > aabb2.max.x)
		{
			return false;
		}
	}

	if (firstTimeOfCollision > tLast)
	{
		return false;
	}

	// y-axis
	if (vRel.y < 0.0f)
	{
		// case 1
		if (aabb1.min.y > aabb2.max.y)
		{
			return false;
		}

		// case 4
		if (aabb1.max.y < aabb2.min.y)
		{
			firstTimeOfCollision = AEMax((aabb1.max.y - aabb2.min.y) / vRel.y, firstTimeOfCollision);
		}
		if (aabb1.min.y < aabb2.max.y)
		{
			tLast = AEMin((aabb1.min.y - aabb2.max.y) / vRel.y, tLast);
		}
	}
	else if (vRel.y > 0.0f)
	{
		// case 3
		if (aabb1.max.y < aabb2.min.y)
		{
			return false;
		}

		// case 2
		if (aabb1.min.y > aabb2.max.y)
		{
			firstTimeOfCollision = AEMax((aabb1.min.y - aabb2.max.y) / vRel.y, firstTimeOfCollision);
		}
		if (aabb1.max.y > aabb2.min.y)
		{
			tLast = AEMin((aabb1.max.y - aabb2.min.y) / vRel.y, tLast);
		}
	}
	else
	{
		if (aabb1.max.y < aabb2.min.y)
		{
			return false;
		}
		else if (aabb1.min.y > aabb2.max.y)
		{
			return false;
		}
	}

	if (firstTimeOfCollision > tLast)
	{
		return false;
	}

	return true;
}