/*!
@file    Sphere.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Sphere.h,v 1.2 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_SPHERE_H_
#define GFX_SPHERE_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

#include "Matrix4.h"
#include "Vector3.h"


/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxSphere
/*! Sphere class.
*/
{
  public:
    // ct and dt
    gfxSphere();
    gfxSphere(float x, float y, float z, float r);
    gfxSphere(const gfxSphere& p);
    ~gfxSphere();
    
    // = operator: assignment
    gfxSphere& operator=(const gfxSphere& p);

		gfxSphere	Transform(const gfxMatrix4& model_to_someframe_mtx) const;
    
    // properties
    gfxVector3 center;  //!< Sphere center.
    float      radius;  //!< Sphere radius.
};


/*                                                             implementation
----------------------------------------------------------------------------- */

#endif  /* GFX_SPHERE_H_ */