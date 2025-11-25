/******************************************************************************/
/*!
\file		Vector2D.cpp
\author 	Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par    	bryanweize.ang@digipen.edu
\date   	Mar 22 2024
\brief  	This file includes the defintions of the member and non-member
			functions of Vector2D structure

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#include "Vector2D.h"
#include "main.h"

#include <math.h>


namespace CSD1130
{
	/**************************************************************************/
	/*!
		Constructor that takes in 2 float values
	 */
	/**************************************************************************/
	Vector2D::Vector2D(float _x, float _y)
	{
		UNREFERENCED_PARAMETER(_x);
		UNREFERENCED_PARAMETER(_y);

		//TODO
		x = _x;
		y = _y;
	}

	/**************************************************************************/
	/*!
		Compound addition assignment that takes in another Vec2
	 */
	/**************************************************************************/
	Vector2D& Vector2D::operator += (const Vector2D &rhs)
	{
		UNREFERENCED_PARAMETER(rhs);

		//TODO
		x += rhs.x;
		y += rhs.y;
		return *this;
	}

	/**************************************************************************/
	/*!
		Compound subtraction assignment that takes in another Vec2
	 */
	/**************************************************************************/
	Vector2D& Vector2D::operator -= (const Vector2D &rhs)
	{
		UNREFERENCED_PARAMETER(rhs);

		//TODO and FIX
		x -= rhs.x;
		y -= rhs.y;
		return *this;
	}

	/**************************************************************************/
	/*!
		Compound muliplication assignment that takes in 1 float value
	 */
	/**************************************************************************/
	Vector2D& Vector2D::operator *= (float rhs)
	{
		UNREFERENCED_PARAMETER(rhs);

		//TODO and FIX
		x *= rhs;
		y *= rhs;
		return *this;
	}

	/**************************************************************************/
	/*!
		Compound division assignment that takes in 1 float value
	 */
	/**************************************************************************/
	Vector2D& Vector2D::operator /= (float rhs)
	{
		UNREFERENCED_PARAMETER(rhs);

		//TODO and FIX
		x /= rhs;
		y /= rhs;
		return *this;
	}

	/**************************************************************************/
	/*!
		Negation operator
	 */
	/**************************************************************************/
	Vector2D Vector2D::operator-() const
	{
		//TODO and FIX

		return Vector2D(-x, -y);
	}


	// Non-member functions
	//-------------------------------------------------------------------------

	/**************************************************************************/
	/*!
		Addition operator that takes in 2 Vec2s
	 */
	/**************************************************************************/
	Vector2D operator + (const Vector2D &lhs, const Vector2D &rhs)
	{
		return Vector2D(lhs.x + rhs.x, lhs.y + rhs.y);
	}

	/**************************************************************************/
	/*!
		Subtraction operator that takes in 2 Vec2s
	 */
	/**************************************************************************/
	Vector2D operator - (const Vector2D &lhs, const Vector2D &rhs)
	{
		return Vector2D(lhs.x - rhs.x, lhs.y - rhs.y);
	}

	/**************************************************************************/
	/*!
		Multiplication operator that takes in 1 Vec2 and 1 float value
	 */
	/**************************************************************************/
	Vector2D operator * (const Vector2D &lhs, float rhs)
	{
		return Vector2D(lhs.x * rhs, lhs.y * rhs);
	}

	/**************************************************************************/
	/*!
		Multiplication operator that takes in 1 float value and 1 Vec2
	 */
	/**************************************************************************/
	Vector2D operator * (float lhs, const Vector2D &rhs)
	{
		return Vector2D(rhs * lhs);
	}

	/**************************************************************************/
	/*!
		Division operator that takes in 1 Vec2 and 1 float value
	 */
	/**************************************************************************/
	Vector2D operator / (const Vector2D &lhs, float rhs)
	{
		UNREFERENCED_PARAMETER(lhs);
		UNREFERENCED_PARAMETER(rhs);

		//TODO and FIX

		return Vector2D(lhs.x / rhs, lhs.y / rhs);
	}

	/**************************************************************************/
	/*!
		In this function, pResult will be the unit vector of pVec0
	 */
	/**************************************************************************/
	void Vector2DNormalize(Vector2D &pResult, const Vector2D &pVec0)
	{
		UNREFERENCED_PARAMETER(pResult);
		UNREFERENCED_PARAMETER(pVec0);

		//TODO and FIX
		float length{ sqrtf(pVec0.x * pVec0.x + pVec0.y * pVec0.y) };
		pResult.x = pVec0.x / length;
		pResult.y = pVec0.y / length;
	}

	/**************************************************************************/
	/*!
		This function returns the length of the vector pVec0 
	 */
	/**************************************************************************/
	float Vector2DLength(const Vector2D &pVec0)
	{
		UNREFERENCED_PARAMETER(pVec0);

		//TODO and FIX

		return sqrtf(pVec0.x * pVec0.x + pVec0.y * pVec0.y);
	}

	/**************************************************************************/
	/*!
		This function return the square of pVec0's length. Avoid the square root 
	 */
	/**************************************************************************/
	float Vector2DSquareLength(const Vector2D &pVec0)
	{
		UNREFERENCED_PARAMETER(pVec0);

		//TODO and FIX

		return pVec0.x * pVec0.x + pVec0.y * pVec0.y;
	}

	/**************************************************************************/
	/*!
		In this function, pVec0 and pVec1 are considered as 2D points.
		The distance between these 2 2D points is returned
	 */
	/**************************************************************************/
	float Vector2DDistance(const Vector2D &pVec0, const Vector2D &pVec1)
	{
		UNREFERENCED_PARAMETER(pVec0);
		UNREFERENCED_PARAMETER(pVec1);

		//TODO and FIX

		return sqrtf((pVec0.x - pVec1.x) * (pVec0.x - pVec1.x) + 
			(pVec0.y - pVec1.y) * (pVec0.y - pVec1.y));
	}

	/**************************************************************************/
	/*!
		In this function, pVec0 and pVec1 are considered as 2D points.
		The squared distance between these 2 2D points is returned. 
		Avoid the square root
	 */
	/**************************************************************************/
	float Vector2DSquareDistance(const Vector2D &pVec0, const Vector2D &pVec1)
	{
		UNREFERENCED_PARAMETER(pVec0);
		UNREFERENCED_PARAMETER(pVec1);

		//TODO and FIX

		return (pVec0.x - pVec1.x) * (pVec0.x - pVec1.x) +
			(pVec0.y - pVec1.y) * (pVec0.y - pVec1.y);
	}

	/**************************************************************************/
	/*!
		This function returns the dot product between pVec0 and pVec1
	 */
	/**************************************************************************/
	float Vector2DDotProduct(const Vector2D &pVec0, const Vector2D &pVec1)
	{
		UNREFERENCED_PARAMETER(pVec0);
		UNREFERENCED_PARAMETER(pVec1);

		//TODO and FIX

		return pVec0.x * pVec1.x + pVec0.y * pVec1.y;
	}

	/**************************************************************************/
	/*!
		This functions return the cross product magnitude 
		between pVec0 and pVec1
	 */
	/**************************************************************************/
	float Vector2DCrossProductMag(const Vector2D &pVec0, const Vector2D &pVec1)
	{
		UNREFERENCED_PARAMETER(pVec0);
		UNREFERENCED_PARAMETER(pVec1);

		//TODO and FIX

		return pVec0.x * pVec1.y - pVec0.y * pVec1.x;
	}
}