/*!
@file    Vector3.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Vector3.h,v 1.6 2005/02/23 23:34:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_VECTOR3_H_
#define GFX_VECTOR3_H_

/*                                                                    classes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxVector3
/*! 3-component vector class.
*/
{
	// friends - arithmetic operators
	friend gfxVector3 operator+(const gfxVector3& l, const gfxVector3& r);
	friend gfxVector3 operator-(const gfxVector3& l, const gfxVector3& r);
	friend gfxVector3 operator*(float l, const gfxVector3& r);

	// friends - dot product
	friend float operator*(const gfxVector3& l, const gfxVector3& r);

	// friends - cross product
	friend gfxVector3 operator^(const gfxVector3& l, const gfxVector3& r);

	public:
		// ct and dt 
		gfxVector3();
		gfxVector3(float xi, float yi, float zi);
		gfxVector3(const gfxVector3& v);
		~gfxVector3();

		// = operator: assignment
		gfxVector3& operator=(const gfxVector3& v);

		// [] operators: component access
		float&       operator[](unsigned int i);
		const float& operator[](unsigned int i) const;

		// arithmetic operators: vector operations
		gfxVector3& operator+=(const gfxVector3& v);
		gfxVector3& operator-=(const gfxVector3& v);

		// arithmetic operators: scalar operations
		gfxVector3& operator*=(float s);
		gfxVector3& operator/=(float s);

		// arithmetic operators: negation
		gfxVector3 operator-(void);

		// length
		float Length() const;

		// normalization
		gfxVector3 Norm() const;
		void       Normalize();

		// reinitialization
		void SetTo(float xi, float yi, float zi);
		void SetToZero();

		// delegates
		float &x;  //!< Access to X component.
		float &y;  //!< Access to Y component.
		float &z;  //!< Access to Z component.

		// special
		static const gfxVector3  ZERO;        //!< Zero vector (0,0,0).
		static const gfxVector3  UNIT_X;      //!< Unit vector in X (1,0,0).
		static const gfxVector3  UNIT_Y;      //!< Unit vector in Y (0,1,0).
		static const gfxVector3  UNIT_Z;      //!< Unit vector in Z (0,0,1).
		static const gfxVector3  UNIT_NEG_X;  //!< Negative unit vector in X (-1,0,0).
		static const gfxVector3  UNIT_NEG_Y;  //!< Negative unit vector in Y (0,-1,0).
		static const gfxVector3  UNIT_NEG_Z;  //!< Negative unit vector in Z (0,0,-1).

	private:
		// data members
		float  mVec[3];  //!< Vector elements.
};

#endif	/* GFX_VECTOR3_H_ */