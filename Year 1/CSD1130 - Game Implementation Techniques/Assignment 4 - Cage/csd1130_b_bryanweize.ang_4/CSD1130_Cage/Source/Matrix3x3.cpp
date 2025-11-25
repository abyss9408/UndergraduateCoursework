/******************************************************************************/
/*!
\file		Matrix3x3.cpp
\author 	Bryan Ang Wei Ze, bryanweize.ang, 2301397
\par    	bryanweize.ang@digipen.edu
\date   	Mar 22 2024
\brief  	This file includes the defintions of the member and non-member
			functions of the Matrix3x3 structure

Copyright (C) 2024 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the prior
written consent of DigiPen Institute of Technology is prohibited.
*/
/******************************************************************************/

#include "Matrix3x3.h"
#include "main.h"

#include <string.h>
#include <math.h>


#define PIOVER180	0.0174532925199432f

namespace CSD1130
{
	/**************************************************************************/
	/*!
		Constructor that takes in an array of floats
	 */
	/**************************************************************************/
	Matrix3x3::Matrix3x3(const float *pArr) 
		: m00(pArr[0]), m01(pArr[1]), m02(pArr[2]), m10(pArr[3]), m11(pArr[4]), m12(pArr[5]), m20(pArr[6]), m21(pArr[7]), m22(pArr[8])
	{
		UNREFERENCED_PARAMETER(pArr);

		//TODO and FIX
	}

	/**************************************************************************/
	/*!
		Constructor that takes in 9 float values
	 */
	/**************************************************************************/
	Matrix3x3::Matrix3x3(float _00, float _01, float _02,
						 float _10, float _11, float _12,
						 float _20, float _21, float _22)
		: m00(_00), m01(_01), m02(_02), m10(_10), m11(_11), m12(_12), m20(_20), m21(_21), m22(_22)
	{
		UNREFERENCED_PARAMETER(_00);
		UNREFERENCED_PARAMETER(_01);
		UNREFERENCED_PARAMETER(_02);
		UNREFERENCED_PARAMETER(_10);
		UNREFERENCED_PARAMETER(_11);
		UNREFERENCED_PARAMETER(_12);
		UNREFERENCED_PARAMETER(_20);
		UNREFERENCED_PARAMETER(_21);
		UNREFERENCED_PARAMETER(_22);

		//TODO and FIX
	}

	/**************************************************************************/
	/*!
		Copy assignment that takes in another Mtx33
	 */
	/**************************************************************************/
	Matrix3x3& Matrix3x3::operator=(const Matrix3x3 &rhs)
	{
		UNREFERENCED_PARAMETER(rhs);

		for (size_t i = 0; i < 9; ++i)
		{
			m[i] = rhs.m[i];
		}

		return *this;
	}

	/**************************************************************************/
	/*!
		Compound multiplication assignment that takes in another Mtx33
	 */
	/**************************************************************************/
	Matrix3x3& Matrix3x3::operator *= (const Matrix3x3 &rhs)
	{
		UNREFERENCED_PARAMETER(rhs);

		*this = *this * rhs;

		return *this;
	}

	/**************************************************************************/
	/*!
		This operator multiplies the matrix lhs with another matrix rhs 
		and returns the result as a matrix
	 */
	/**************************************************************************/
	Matrix3x3 operator * (const Matrix3x3 &lhs, const Matrix3x3 &rhs)
	{
		UNREFERENCED_PARAMETER(lhs);
		UNREFERENCED_PARAMETER(rhs);

		return Matrix3x3((lhs.m00 * rhs.m00) + (lhs.m01 * rhs.m10) + (lhs.m02 * rhs.m20), 
			(lhs.m00 * rhs.m01) + (lhs.m01 * rhs.m11) + (lhs.m02 * rhs.m21), 
			(lhs.m00 * rhs.m02) + (lhs.m01 * rhs.m12) + (lhs.m02 * rhs.m22),
			(lhs.m10 * rhs.m00) + (lhs.m11 * rhs.m10) + (lhs.m12 * rhs.m20),
			(lhs.m10 * rhs.m01) + (lhs.m11 * rhs.m11) + (lhs.m12 * rhs.m21),
			(lhs.m10 * rhs.m02) + (lhs.m11 * rhs.m12) + (lhs.m12 * rhs.m22),
			(lhs.m20 * rhs.m00) + (lhs.m21 * rhs.m10) + (lhs.m22 * rhs.m20),
			(lhs.m20 * rhs.m01) + (lhs.m21 * rhs.m11) + (lhs.m22 * rhs.m21),
			(lhs.m20 * rhs.m02) + (lhs.m21 * rhs.m12) + (lhs.m22 * rhs.m22));
	}

	/**************************************************************************/
	/*!
		This function multiplies the matrix pMtx with the vector rhs 
		and returns the result as a vector
	 */
	/**************************************************************************/
	Vector2D operator * (const Matrix3x3 &lhs, const Vector2D &rhs)
	{
		UNREFERENCED_PARAMETER(lhs);
		UNREFERENCED_PARAMETER(rhs);

		return Vector2D(lhs.m00 * rhs.x + lhs.m01 * rhs.y + lhs.m02 * 1.0f, lhs.m10 * rhs.x + lhs.m11 * rhs.y + lhs.m12 * 1.0f);
	}

	/**************************************************************************/
	/*!
		This function sets the matrix pResult to the identity matrix
	 */
	/**************************************************************************/
	void Mtx33Identity(Matrix3x3 &pResult)
	{
		UNREFERENCED_PARAMETER(pResult);
		pResult.m00 = 1.0f;
		pResult.m01 = 0.0f;
		pResult.m02 = 0.0f;
		pResult.m10 = 0.0f;
		pResult.m11 = 1.0f;
		pResult.m12 = 0.0f;
		pResult.m20 = 0.0f;
		pResult.m21 = 0.0f;
		pResult.m22 = 1.0f;
	}

	/**************************************************************************/
	/*!
		This function creates a translation matrix from x & y 
		and saves it in pResult
	 */
	/**************************************************************************/
	void Mtx33Translate(Matrix3x3 &pResult, float x, float y)
	{
		UNREFERENCED_PARAMETER(pResult);
		UNREFERENCED_PARAMETER(x);
		UNREFERENCED_PARAMETER(y);

		Mtx33Identity(pResult);
		pResult.m02 = x;
		pResult.m12 = y;
	}

	/**************************************************************************/
	/*!
		This function creates a scaling matrix from x & y 
		and saves it in pResult
	 */
	/**************************************************************************/
	void Mtx33Scale(Matrix3x3 &pResult, float x, float y)
	{
		UNREFERENCED_PARAMETER(pResult);
		UNREFERENCED_PARAMETER(x);
		UNREFERENCED_PARAMETER(y);

		Mtx33Identity(pResult);
		pResult.m00 = x;
		pResult.m11 = y;
	}

	/**************************************************************************/
	/*!
		This matrix creates a rotation matrix from "angle" whose value 
		is in radian. Save the resultant matrix in pResult.
	 */
	/**************************************************************************/
	void Mtx33RotRad(Matrix3x3 &pResult, float angle)
	{
		UNREFERENCED_PARAMETER(pResult);
		UNREFERENCED_PARAMETER(angle);

		Mtx33Identity(pResult);
		pResult.m00 = cosf(angle);
		pResult.m10 = sinf(angle);
		pResult.m01 = -sinf(angle);
		pResult.m11 = cosf(angle);
	}

	/**************************************************************************/
	/*!
		This matrix creates a rotation matrix from "angle" whose value 
		is in degree. Save the resultant matrix in pResult.
	 */
	/**************************************************************************/
	void Mtx33RotDeg(Matrix3x3 &pResult, float angle)
	{
		UNREFERENCED_PARAMETER(pResult);
		UNREFERENCED_PARAMETER(angle);

		Mtx33Identity(pResult);
		pResult.m00 = cosf(angle * PIOVER180);
		pResult.m10 = sinf(angle * PIOVER180);
		pResult.m01 = -sinf(angle * PIOVER180);
		pResult.m11 = cosf(angle * PIOVER180);
	}

	/**************************************************************************/
	/*!
		This functions calculated the transpose matrix of pMtx 
		and saves it in pResult
	 */
	/**************************************************************************/
	void Mtx33Transpose(Matrix3x3 &pResult, const Matrix3x3 &pMtx)
	{
		UNREFERENCED_PARAMETER(pResult);
		UNREFERENCED_PARAMETER(pMtx);

		const Matrix3x3 mtx(pMtx);

		pResult.m00 = mtx.m00;
		pResult.m10 = mtx.m01;
		pResult.m20 = mtx.m02;
		pResult.m01 = mtx.m10;
		pResult.m11 = mtx.m11;
		pResult.m21 = mtx.m12;
		pResult.m02 = mtx.m20;
		pResult.m12 = mtx.m21;
		pResult.m22 = mtx.m22;
	}

	/**************************************************************************/
	/*!
		This function calculates the inverse matrix of pMtx and saves the 
		result in pResult. If the matrix inversion fails, pResult 
		would be set to NULL.
	*/
	/**************************************************************************/
	void Mtx33Inverse(Matrix3x3 *pResult, float *determinant, const Matrix3x3 &pMtx)
	{
		UNREFERENCED_PARAMETER(pResult);
		UNREFERENCED_PARAMETER(determinant);
		UNREFERENCED_PARAMETER(pMtx);

		*determinant = ((pMtx.m00 * pMtx.m11 * pMtx.m22) + (pMtx.m01 * pMtx.m12 * pMtx.m20) + (pMtx.m02 * pMtx.m10 * pMtx.m21)) -
			((pMtx.m20 * pMtx.m11 * pMtx.m02) + (pMtx.m21 * pMtx.m12 * pMtx.m00) + (pMtx.m22 * pMtx.m10 * pMtx.m01));

		if (*determinant == 0.0f)
		{
			pResult = NULL;
			return;
		}

		Matrix3x3 CofactorMtx;

		CofactorMtx.m00 = (pMtx.m11 * pMtx.m22) - (pMtx.m12 * pMtx.m21);
		CofactorMtx.m01 = -((pMtx.m10 * pMtx.m22) - (pMtx.m12 * pMtx.m20));
		CofactorMtx.m02 = (pMtx.m10 * pMtx.m21) - (pMtx.m11 * pMtx.m20);
		CofactorMtx.m10 = -((pMtx.m01 * pMtx.m22) - (pMtx.m02 * pMtx.m21));
		CofactorMtx.m11 = (pMtx.m00 * pMtx.m22) - (pMtx.m02 * pMtx.m20);
		CofactorMtx.m12 = -((pMtx.m00 * pMtx.m21) - (pMtx.m01 * pMtx.m20));
		CofactorMtx.m20 = (pMtx.m01 * pMtx.m12) - (pMtx.m02 * pMtx.m11);
		CofactorMtx.m21 = -((pMtx.m00 * pMtx.m12) - (pMtx.m02 * pMtx.m10));
		CofactorMtx.m22 = (pMtx.m00 * pMtx.m11) - (pMtx.m01 * pMtx.m10);

		Mtx33Transpose(CofactorMtx, CofactorMtx);

		for (size_t i = 0; i < 9; ++i)
		{
			pResult->m[i] = CofactorMtx.m[i] / *determinant;
		}
		//TODO and FIX
	}
}