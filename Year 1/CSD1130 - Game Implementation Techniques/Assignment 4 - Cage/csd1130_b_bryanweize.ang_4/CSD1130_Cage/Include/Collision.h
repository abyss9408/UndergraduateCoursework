/******************************************************************************/
/*!
\file		Collision.h
\author 	Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par    	bryanweize.ang@digipen.edu
\date   	Mar 22 2024
\brief  	This file contains the declarations of the BuildLineSegment, 
			CollisionIntersection_CircleLineSegment, CheckMovingCircleToLineEdge
			and CollisionResponse_CircleLineSegment functions

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#ifndef CSD1130_COLLISION_H_
#define CSD1130_COLLISION_H_

#include "Vector2D.h"
#include "Matrix3x3.h"

/******************************************************************************/
/*!
 */
/******************************************************************************/
struct LineSegment
{
	CSD1130::Vec2 m_pt0;						//End point P0
	CSD1130::Vec2 m_pt1;						//End point P1
	CSD1130::Vec2 m_normal;					//normalized outward normal
};

void BuildLineSegment(LineSegment &lineSegment,								//Line segment reference - input
						const CSD1130::Vec2 &p0,									//Point P0 - input
						const CSD1130::Vec2 &p1);									//Point P1 - input

/******************************************************************************/
/*!
 */
/******************************************************************************/
struct Circle
{
	CSD1130::Vec2 m_center;
	float m_radius;
};


// INTERSECTION FUNCTIONS
int CollisionIntersection_CircleLineSegment(const Circle &circle,			//Circle data - input
	const CSD1130::Vec2 &ptEnd,													//End circle position - input
	const LineSegment &lineSeg,												//Line segment - input
	CSD1130::Vec2 &interPt,														//Intersection point - output
	CSD1130::Vec2 &normalAtCollision,												//Normal vector at collision time - output
	float &interTime,														//Intersection time ti - output
	bool & checkLineEdges);													//The last parameter is for Extra Credits: when true => check collision with line segment edges



// For Extra Credits
int CheckMovingCircleToLineEdge(bool withinBothLines,						//Flag stating that the circle is starting from between 2 imaginary line segments distant +/- Radius respectively - input
	const Circle &circle,													//Circle data - input
	const CSD1130::Vec2 &ptEnd,													//End circle position - input
	const LineSegment &lineSeg,												//Line segment - input
	CSD1130::Vec2 &interPt,														//Intersection point - output
	CSD1130::Vec2 &normalAtCollision,												//Normal vector at collision time - output
	float &interTime);														//Intersection time ti - output



// RESPONSE FUNCTIONS
void CollisionResponse_CircleLineSegment(const CSD1130::Vec2 &ptInter,				//Intersection position of the circle - input
	const CSD1130::Vec2 &normal,													//Normal vector of reflection on collision time - input
	CSD1130::Vec2 &ptEnd,															//Final position of the circle after reflection - output
	CSD1130::Vec2 &reflected);														//Normalized reflection vector direction - output




#endif // CSD1130_COLLISION_H_