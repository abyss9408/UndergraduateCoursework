/*!
@file    Transform.h
@author  Prasanna Ghali       (pghali@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_TRANSFORM_H_
#define GFX_TRANSFORM_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "Vector3.h"
#include "Vector4.h"
#include "Matrix4.h"

/*                                                  transform types: enumeration
----------------------------------------------------------------------------- */

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxTransform
/*! Camera class.
*/
{
public:
	// ctors
	gfxTransform();
	// scale factors, orientation, position of model frame in world frame
	gfxTransform(const gfxVector3&, const gfxVector3*, const gfxVector3&);

	// update
	gfxMatrix4 Compute() const;
	void MakeIdentity();

	// manipulators
	void	SetScale(float sx, float sy, float sz)			{ mScale.SetTo(sx, sy, sz); }
	void	SetScale(const gfxVector3& rscale)					{ mScale	=	rscale; }
	
	void	SetRotation(const gfxVector3& xaxis,
										const gfxVector3& yaxis,
										const gfxVector3& zaxis)				{ mOrientation[0] = xaxis; 
																											mOrientation[1] = yaxis;
																											mOrientation[2] = zaxis; 
																										}
	void	SetTranslation(float tx, float ty, float tz)	{ mTranslation.SetTo(tx, ty, tz); }
	void	SetTranslation(const gfxVector3& rtranslate)	{ mTranslation	=	rtranslate; }

  // accessors
	bool							IsIdentity() const				{ return mIsIdentity; }
	bool							IsUniformScale() const		{ return mIsUniformScale; }

	const gfxVector3&	GetTranslation() const		{ return mTranslation; }
	const gfxVector3&	GetScale() const					{ return mScale; }
	const gfxVector3*	GetRotation()							{ return mOrientation; }

private:
	bool					mIsIdentity, mIsUniformScale;
	gfxVector3		mScale;						// scale factors
	gfxVector3		mOrientation[3];	// orientation of model's basis in world frame
	gfxVector3		mTranslation;			// position of model frame origin in world frame
};


#endif // GFX_TRANSFORM_H_