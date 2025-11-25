/*!
@file    Vector4.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Vector4.h,v 1.5 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_VECTOR4_H_
#define GFX_VECTOR4_H_

/*                                                                    includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxVector4
/*! 3D homogeneous vector class.
		The implicit value of the homogeneous coordinate is 1.f.

*/
{
	// friends
	friend gfxVector4 operator+(const gfxVector4& l, const gfxVector4& r);
	friend gfxVector4 operator-(const gfxVector4& l, const gfxVector4& r);
	friend gfxVector4 operator*(const gfxVector4& l, float r);
	friend gfxVector4 operator*(float l, const gfxVector4& r);

	public:
		// ct and dt 
		gfxVector4();
		gfxVector4(float xi, float yi, float zi, float wi);
		gfxVector4(const gfxVector4& v);
		~gfxVector4();

		// = operator: assignment
		gfxVector4& operator=(const gfxVector4& v);

		// [] operators: component access
		float&       operator[](unsigned int i);
		const float& operator[](unsigned int i) const;

		// length
		float Length() const;

		// normalization
		gfxVector4 Norm() const;
		void       Normalize();

		// reinitialization
		void SetTo(float xi, float yi, float zi, float wi);
		void SetToZero();

		// delegates
		float &x;  //!< Access to X component.
		float &y;  //!< Access to Y component.
		float &z;  //!< Access to Z component.
		float &w;  //!< Access to W component.

	private:
		// data members
		float  mVec[4];  //!< Vector elements.
};


#endif  /* GFX_VECTOR4_H_ */