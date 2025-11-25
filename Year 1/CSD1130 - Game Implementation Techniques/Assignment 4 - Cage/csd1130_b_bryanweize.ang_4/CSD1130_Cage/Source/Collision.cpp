/******************************************************************************/
/*!
\file		Collision.cpp
\author 	Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par    	bryanweize.ang@digipen.edu
\date   	Mar 22 2024
\brief  	This file contains the defintions of the BuildLineSegment, 
			CollisionIntersection_CircleLineSegment, CheckMovingCircleToLineEdge
			and CollisionResponse_CircleLineSegment functions

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#include "main.h"

/******************************************************************************/
/*!
	Build a line segment with two points and normalized outward normal
 */
/******************************************************************************/
void BuildLineSegment(LineSegment &lineSegment,
	const CSD1130::Vec2& p0,
	const CSD1130::Vec2& p1)
{
	lineSegment.m_pt0 = p0;
	lineSegment.m_pt1 = p1;

	CSD1130::Vec2 edge = p1 - p0;
	CSD1130::Vec2 normal = { edge.y,-edge.x };
	CSD1130::Vector2DNormalize(lineSegment.m_normal, normal);
}

/******************************************************************************/
/*!
	Check for collision with body of line segment
 */
/******************************************************************************/
int CollisionIntersection_CircleLineSegment(const Circle &circle,
	const CSD1130::Vec2&ptEnd,
	const LineSegment &lineSeg,
	CSD1130::Vec2 &interPt,
	CSD1130::Vec2 &normalAtCollision,
	float &interTime,
	bool & checkLineEdges)
{
	CSD1130::Vec2 P0_Prime, P1_Prime, V, V_Outward_Normal, BsP0_Prime, BsP1_Prime;
	V = ptEnd - circle.m_center;
	V_Outward_Normal = { V.y,-V.x };

	//circle is starting from inside half plane
	if (CSD1130::Vector2DDotProduct(lineSeg.m_normal, circle.m_center) - 
		CSD1130::Vector2DDotProduct(lineSeg.m_normal, lineSeg.m_pt0) <= -circle.m_radius)
	{
		//simulate imaginary line (LNS2) edge points
		P0_Prime = lineSeg.m_pt0 - circle.m_radius * lineSeg.m_normal;
		P1_Prime = lineSeg.m_pt1 - circle.m_radius * lineSeg.m_normal;
		BsP0_Prime = P0_Prime - circle.m_center;
		BsP1_Prime = P1_Prime - circle.m_center;

		if (CSD1130::Vector2DDotProduct(V_Outward_Normal, BsP0_Prime) * 
			CSD1130::Vector2DDotProduct(V_Outward_Normal, BsP1_Prime) < 0)
		{
			interTime = (CSD1130::Vector2DDotProduct(lineSeg.m_normal, lineSeg.m_pt0) -
				CSD1130::Vector2DDotProduct(lineSeg.m_normal, circle.m_center) - circle.m_radius) /
				(CSD1130::Vector2DDotProduct(lineSeg.m_normal, V));

			if (0.f <= interTime && interTime <= 1.f)
			{
				interPt = circle.m_center + (V * interTime);
				normalAtCollision = -lineSeg.m_normal;
				return 1;
			}
			else
			{
				if (checkLineEdges && CheckMovingCircleToLineEdge(false, circle, ptEnd, lineSeg, interPt, normalAtCollision, interTime))
				{
					return 1;
				}
			}
		}
	}
	//circle is starting from outside half plane
	else if (CSD1130::Vector2DDotProduct(lineSeg.m_normal, circle.m_center) - 
		CSD1130::Vector2DDotProduct(lineSeg.m_normal, lineSeg.m_pt0) >= circle.m_radius)
	{
		//simulate imaginary line (LNS2) edge points
		P0_Prime = lineSeg.m_pt0 + circle.m_radius * lineSeg.m_normal;
		P1_Prime = lineSeg.m_pt1 + circle.m_radius * lineSeg.m_normal;
		BsP0_Prime = P0_Prime - circle.m_center;
		BsP1_Prime = P1_Prime - circle.m_center;

		if (CSD1130::Vector2DDotProduct(V_Outward_Normal, BsP0_Prime) *
			CSD1130::Vector2DDotProduct(V_Outward_Normal, BsP1_Prime) < 0)
		{
			interTime = (CSD1130::Vector2DDotProduct(lineSeg.m_normal, lineSeg.m_pt0) -
				CSD1130::Vector2DDotProduct(lineSeg.m_normal, circle.m_center) + circle.m_radius) /
				(CSD1130::Vector2DDotProduct(lineSeg.m_normal, V));

			if (0.f <= interTime && interTime <= 1.f)
			{
				interPt = circle.m_center + (V * interTime);
				normalAtCollision = lineSeg.m_normal;
				return 1;
			}
			else
			{
				if (checkLineEdges && CheckMovingCircleToLineEdge(false, circle, ptEnd, lineSeg, interPt, normalAtCollision, interTime))
				{
					return 1;
				}
			}
		}
	}
	//circle is starting from between both imaginary lines
	else
	{
		if (checkLineEdges && CheckMovingCircleToLineEdge(true, circle, ptEnd, lineSeg, interPt, normalAtCollision, interTime))
		{
			return 1;
		}
	}

	//no collision
	return 0;
}

/******************************************************************************/
/*!
	Check for collision with edges of line segment
*/
/******************************************************************************/
int CheckMovingCircleToLineEdge(bool withinBothLines,
	const Circle &circle,
	const CSD1130::Vec2 &ptEnd,
	const LineSegment &lineSeg,
	CSD1130::Vec2 &interPt,
	CSD1130::Vec2 &normalAtCollision,
	float &interTime)
{
	CSD1130::Vec2 P0P1{ lineSeg.m_pt1 - lineSeg.m_pt0 };
	CSD1130::Vec2 BsP0{ lineSeg.m_pt0 - circle.m_center };
	CSD1130::Vec2 BsP1{ lineSeg.m_pt1 - circle.m_center };

	CSD1130::Vec2 V{ ptEnd - circle.m_center };

	CSD1130::Vec2 V_Normalized;
	CSD1130::Vector2DNormalize(V_Normalized, V);

	CSD1130::Vec2 M{ V.y,-V.x };
	CSD1130::Vector2DNormalize(M, M);

	float m{}, s{};

	if (withinBothLines)
	{
		//P0 side
		if (CSD1130::Vector2DDotProduct(BsP0, P0P1) > 0.f)
		{
			m = CSD1130::Vector2DDotProduct(BsP0, V_Normalized);

			//circle is facing P0
			if (m > 0.f)
			{
				float dist0 = CSD1130::Vector2DDotProduct(BsP0, M);

				if (abs(dist0) > circle.m_radius)
				{
					return 0;
				}
				else
				{
					s = sqrtf(circle.m_radius * circle.m_radius - dist0 * dist0);

					interTime = (m - s) / CSD1130::Vector2DLength(V);

					if (interTime <= 1.f)
					{
						interPt = circle.m_center + V * interTime;
						CSD1130::Vec2 P0Bi{ interPt - lineSeg.m_pt0 };
						CSD1130::Vector2DNormalize(P0Bi, P0Bi);
						normalAtCollision = P0Bi;
						return 1;
					}
				}
			}
		}
		//P1 side
		else 
		{
			m = CSD1130::Vector2DDotProduct(BsP1, V_Normalized);

			//circle is facing P1
			if (m > 0.f)
			{
				float dist1 = CSD1130::Vector2DDotProduct(BsP1, M);

				if (abs(dist1) > circle.m_radius)
				{
					return 0;
				}
				else
				{
					s = sqrtf(circle.m_radius * circle.m_radius - dist1 * dist1);

					interTime = (m - s) / CSD1130::Vector2DLength(V);

					if (interTime <= 1.f)
					{
						interPt = circle.m_center + V * interTime;
						CSD1130::Vec2 P1Bi{ interPt - lineSeg.m_pt1 };
						CSD1130::Vector2DNormalize(P1Bi, P1Bi);
						normalAtCollision = P1Bi;
						return 1;
					}
				}
			}
		}
	}
	//circle is not within both lines
	else 
	{
		bool P0Side{ false };

		float dist0 = CSD1130::Vector2DDotProduct(BsP0, M);
		float dist1 = CSD1130::Vector2DDotProduct(BsP1, M);

		float dist0_absolute = abs(dist0);
		float dist1_absolute = abs(dist1);

		// determine if circle is closer to P0 or P1 side
		if (dist0_absolute > circle.m_radius && dist1_absolute > circle.m_radius)
		{
			return 0;
		}
		else if (dist0_absolute <= circle.m_radius && dist1_absolute <= circle.m_radius)
		{
			float m0 = CSD1130::Vector2DDotProduct(BsP0, V_Normalized);
			float m1 = CSD1130::Vector2DDotProduct(BsP1, V_Normalized);

			float m0_absolute = abs(m0);
			float m1_absolute = abs(m1);

			if (m0_absolute < m1_absolute)
			{
				P0Side = true;
			}
			else
			{
				P0Side = false;
			}
		}
		else if (dist0_absolute <= circle.m_radius)
		{
			P0Side = true;
		}
		else //if(dist1_absoluteValue <= R)
		{
			P0Side = false;
		}

		//circle is closer to P0
		if (P0Side)
		{
			m = CSD1130::Vector2DDotProduct(BsP0, V_Normalized);

			//circle is moving away from P0
			if (m < 0.f)
			{
				return 0;
			}
			//circle is moving towards P0
			else 
			{
				s = sqrtf(circle.m_radius * circle.m_radius - dist0 * dist0);
				interTime = (m - s) / CSD1130::Vector2DLength(V);

				if (interTime <= 1.f)
				{
					interPt = circle.m_center + V * interTime;
					CSD1130::Vec2 P0Bi{ interPt - lineSeg.m_pt0 };
					CSD1130::Vector2DNormalize(P0Bi, P0Bi);
					normalAtCollision = P0Bi;
					return 1;
				}
			}
		}
		//circle is closer to P1
		else 
		{
			m = CSD1130::Vector2DDotProduct(BsP1, V_Normalized);

			//circle is moving away from P1
			if (m < 0.f)
			{
				return 0;
			}
			//circle is moving towards P1
			else
			{
				s = sqrtf(circle.m_radius * circle.m_radius - dist1 * dist1);
				interTime = (m - s) / CSD1130::Vector2DLength(V);

				if (interTime <= 1.f)
				{
					interPt = circle.m_center + V * interTime;
					CSD1130::Vec2 P1Bi{ interPt - lineSeg.m_pt1 };
					CSD1130::Vector2DNormalize(P1Bi, P1Bi);
					normalAtCollision = P1Bi;
					return 1;
				}
			}
		}
	}

	//no collision
	return 0;
}





/******************************************************************************/
/*!
	CollisionResponse_CircleLineSegment function
 */
/******************************************************************************/
void CollisionResponse_CircleLineSegment(const CSD1130::Vec2 &ptInter,
	const CSD1130::Vec2 &normal,
	CSD1130::Vec2 &ptEnd,
	CSD1130::Vec2 &reflected)
{
	//Reflection vector
	reflected = (ptEnd - ptInter) - 2 * CSD1130::Vector2DDotProduct(ptEnd - ptInter, normal) * normal;
	
	//Be_prime
	ptEnd = ptInter + reflected;
	CSD1130::Vector2DNormalize(reflected, reflected);
}



