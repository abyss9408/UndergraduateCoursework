/*!
@file    Frustum.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Frustum.h,v 1.2 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_FRUSTUM_H_
#define GFX_FRUSTUM_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"
#include "Vertex.h"
#include "Vector3.h"
#include "Sphere.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxPlane
/*! Plane class.
*/
{
  public:
    // ct and dt
    gfxPlane(float ia = 0.f, float ib = 0.f, float ic = 0.f, float id = 0.f);
    gfxPlane(const gfxPlane& p);
    ~gfxPlane(void);
    
    // = operator: assignment
    gfxPlane& operator=(const gfxPlane& p);
    
    // delegates
    float  &a;  //!< Access to A component.
    float  &b;  //!< Access to B component.
    float  &c;  //!< Access to C component.
    float  &d;  //!< Access to D component.
  
  private:
    // data members
    float  mP[4];  //!< Plane components.
};

/*  _________________________________________________________________________ */
class gfxFrustum
/*! Frustum class.
*/
{
  public:
    // ct and dt 
    gfxFrustum();
    gfxFrustum(const gfxFrustum& q);
    ~gfxFrustum();
    
    // = operator: assignment
    gfxFrustum& operator=(const gfxFrustum& q);
    
    // delegates
    gfxPlane &l;  //!< Access to left plane.
    gfxPlane &r;  //!< Access to right plane.
    gfxPlane &b;  //!< Access to bottom plane.
    gfxPlane &t;  //!< Access to top plane.
    gfxPlane &n;  //!< Access to near plane.
    gfxPlane &f;  //!< Access to far plane.
    
  public:
    // data members
    gfxPlane  mPlanes[6];  //!< Frustum planes.
};


/*                                                             implementation
----------------------------------------------------------------------------- */

#endif  /* GFX_FRUSTUM_H_ */