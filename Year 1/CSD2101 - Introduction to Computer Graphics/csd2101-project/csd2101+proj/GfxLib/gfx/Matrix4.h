/*!
@file    Matrix4.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Matrix4.h,v 1.7 2005/02/23 23:34:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_MATRIX4_H_
#define GFX_MATRIX4_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

#include "Vector3.h"
#include "Vector4.h"
#include "Math.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxMatrix4
/*! 4x4 matrix class.

    The matrix is stored in column-major order.
*/
{
	// * operator: matrix multiplication
	friend gfxMatrix4 operator*(const gfxMatrix4& l, const gfxMatrix4& r);
	friend gfxVector4 operator*(const gfxMatrix4& l, const gfxVector4& r);

	// * operator: scalar multiplication
	friend gfxMatrix4 operator*(float l, const gfxMatrix4& r);

	public:
		// ct and dt 
		gfxMatrix4();
		// Element eij has column index i and row index j.
		// The 16 parameters are specified in row-major order but are saved
		// in memory using column-major order. For example, elements e00,
		// e01, e02, e03 are stored sequentially at memory location mMat[0],
		// mMat[1], mMat[2], and mMat[3], respectively.
		gfxMatrix4(	float e00, float e10, float e20, float e30,
								float e01, float e11, float e21, float e31,
								float e02, float e12, float e22, float e32,
								float e03, float e13, float e23, float e33);
		gfxMatrix4(const gfxMatrix4& m);
		~gfxMatrix4();
	  
		// = operator: assignment
		gfxMatrix4&		operator=(const gfxMatrix4& m);

		// () operators: component access
		float&				operator()(unsigned int column, unsigned int row);
		const float&	operator()(unsigned int column, unsigned int row) const;
	  
		// iterators
		float*				Begin();
		const float*	Begin() const;
		float*				End();
		const float*	End() const;
	  
		// reinitialization
		// Element eij has column index i and row index j.
		// The 16 parameters are specified in row-major order but are saved
		// in memory using column-major order. For example, elements e00,
		// e01, e02, e03 are stored sequentially at memory location mMat[0],
		// mMat[1], mMat[2], and mMat[3], respectively.
		void	SetTo(float e00, float e10, float e20, float e30,
								float e01, float e11, float e21, float e31,
								float e02, float e12, float e22, float e32,
								float e03, float e13, float e23, float e33);
		void	SetToZero();			// zero all 16 elements
		void	SetToIdentity();	// set 16 elements to define an identity matrix 

		// return column and row vectors ...
		gfxVector3	GetRow3(unsigned int) const;
		gfxVector4	GetRow4(unsigned int) const;
		gfxVector3	GetCol3(unsigned int) const;
		gfxVector4	GetCol4(unsigned int) const;

		// set column and row vectors
		//void				SetCol4(unsigned int, const gfxVector4&);
		//void				SetRow4(unsigned int, const gfxVector4&);

		// matrix construction
		static gfxMatrix4	BuildTranslation(float x, float y, float z);
		static gfxMatrix4	BuildTranslation(const gfxVector3& xyz);

		// Rotate by angle a degrees about axis (x, y, z)
		static gfxMatrix4	BuildRotation(float a, float x, float y, float z);
		static gfxMatrix4	BuildRotation(float angle, const gfxVector3& axis);

		// Scale about pivot point (cx, cy, cz) with scale factors (x, y, z)
		static gfxMatrix4	BuildScaling(float cx, float cy, float cz, float x, float y, float z);
		static gfxMatrix4	BuildScaling(const gfxVector3& pivot, const gfxVector3& scaleFactors);

		static gfxMatrix4	BuildLookAt(const gfxVector3& eye, const gfxVector3& at, const gfxVector3& up);
		static gfxMatrix4	BuildPerspective(float vfov, float aspect, float near, float far);
		static gfxMatrix4	BuildFrustum(float l, float r, float b, float t, float n, float f);
		static gfxMatrix4	BuildOrtho(float l, float r, float b, float t, float n, float f);
		static gfxMatrix4	BuildViewport(float x, float y, float w, float h);
		static gfxMatrix4	BuildNormalTransform(const gfxMatrix4&);

	protected:
		// data members
		float  mMat[16];  //!< Matrix elements.
		/*
		Note:
		In C/C++ and most other programming languages, 4 x 4 arrays are
		stored in row-major order. That is, row 0 is stored in lower memory addresses
		followed by row 1 and so on. Within each row, column 0 is stored at lower memory
		addresses followed by column 1 and so on.
		Consider the 16 elements of a 4 x 4 matrix M in row-major order:

				| M[0][0]		M[0][1]		M[0][2]		M[0][3]	|
				|																				|
				| M[1][0]		M[1][1]		M[1][2]		M[1][3]	|
				|																				|
				| M[2][0]		M[2][1]		M[2][2]		M[2][3]	|
				|																				|
				| M[3][0]		M[3][1]		M[3][2]		M[3][3]	|

		On the other hand, OpenGL and this framework store 4 x 4 matrix elements 
		using an array of 16 elements which are stored in memory using column-major
		order. Therefore, the elements of the 4 x 4 matrix M are stored in array
		mMat[16] as follows:

				| mMat[0]		mMat[4]		mMat[8]		mMat[12]|
				|																				|
				| mMat[1]		mMat[5]		mMat[9]		mMat[13]|
				|																				|
				| mMat[2]		mMat[6]		mMat[10]	mMat[14]|
				|																				|
				| mMat[3]		mMat[7]		mMat[11]	mMat[15]|

		*/
};

#endif  /* GFX_MATRIX4_H_ */