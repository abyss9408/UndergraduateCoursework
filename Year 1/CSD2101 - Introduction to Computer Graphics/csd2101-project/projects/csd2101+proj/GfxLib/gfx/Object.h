/*!
@file    Object.h
@author  Prasanna Ghali       (pghali@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*/

/*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_OBJECT_H_
#define GFX_OBJECT_H_

/*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "Vector3.h"
#include "Vector4.h"
#include "Matrix4.h"
#include "Sphere.h"
#include "Model.h"
#include "Transform.h"
#include "GFXInternal.h"

class gfxObject
{
public: 
	// ctors
	gfxObject(const std::string&, gfxModel*, const gfxTransform&, bool isPickable = false);

	// manipulators
	void	SetDiffuseMat(float, float, float);			//!<Set diffuse reflection coefficients.
	void	SetWorldPosition(float, float, float);	//!<Set displacement of object's local frame
																								//!<from world frame origin.
	void	SetWorldPosition(const gfxVector3&);		//!<Set displacement of object's local frame
																								//!<from world frame origin.
	// accessors
	bool								PickingEnabled() const;		//!<If returning true, object is "pickable";
																								//!<Otherwise not "pickable".
	const std::string&	GetName() const;					//!<Returns object name.
	gfxVector3					GetWorldPosition() const;	//!<Returns displacement of local frame from world frame origin.
	gfxMatrix4					GetWorldMtx() const;			//!<Returns matrix manifestation of mapping of local frame in world frame.
	gfxSphere						GetModelBVSphere() const;	//!<Returns model sphere bounding volume.
	gfxMaterial					GetDiffuseMat() const;		//!<Returns diffuse reflection coefficients.

	void Update();	// Computes new model-to-world transform and updates
									// world bound.

	void Draw(gfxGraphicsPipe *dev) const;

private:
	std::string			mName;						//!<string identifier
	gfxModel*				mPMesh;						//!<pointer to model (geometry, model BS)
	gfxTransform		mWorldTransform;	//!<model-to-world xform
	gfxMatrix4			mWorldMtx;				//!<matrix manifestation of model-to-world xform
	gfxMaterial			mDiffMat;					//!<diffuse material properties
	bool						mPicking;					//!<is object "pickable"?
};


/*                                                                   typedefs
----------------------------------------------------------------------------- */
// object list
using gfxObjectList = std::vector<gfxObject>;

#endif