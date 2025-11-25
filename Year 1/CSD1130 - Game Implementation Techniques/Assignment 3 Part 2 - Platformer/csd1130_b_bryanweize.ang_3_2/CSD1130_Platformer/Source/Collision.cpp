/******************************************************************************/
/*!
\file		Collision.cpp
\author 	Bryan Ang Wei Ze
\par    	email: bryanweize.ang\@digipen.edu
\date   	February 28, 2024
\brief		This source file implements the CollisionIntersection_RectRect
			function.

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
 */
/******************************************************************************/

#include "main.h"

/**************************************************************************/
/*!
	Static collision check between two rectangles
*/
/**************************************************************************/
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
									const AEVec2& vel2)         //Input
{
	UNREFERENCED_PARAMETER(aabb1);
	UNREFERENCED_PARAMETER(vel1);
	UNREFERENCED_PARAMETER(aabb2);
	UNREFERENCED_PARAMETER(vel2);

	if (StaticCollisionIntersection_RectRect(aabb1, aabb2))
	{
		return true;
	}

	AEVec2 vRel;
	float tLast, tFirst;
	vRel.x = vel2.x - vel1.x;
	vRel.y = vel2.y - vel1.y;
	tFirst = 0.0f;
	tLast = g_dt;

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
			tFirst = AEMax((aabb1.max.x - aabb2.min.x) / vRel.x, tFirst);
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
			tFirst = AEMax((aabb1.min.x - aabb2.max.x) / vRel.x, tFirst);
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

	if (tFirst > tLast)
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
			tFirst = AEMax((aabb1.max.y - aabb2.min.y) / vRel.y, tFirst);
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
			tFirst = AEMax((aabb1.min.y - aabb2.max.y) / vRel.y, tFirst);
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

	if (tFirst > tLast)
	{
		return false;
	}

	return true;
}