/*!
@file	   Controller_Clipping.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Controller_Clipping.h,v 1.1 2005/02/19 22:10:23 sls Exp $
     $Log: Controller_Clipping.h,v $
     Revision 1.1  2005/02/19 22:10:23  sls
     Implemented Sutherland-Hodgman clipping through derived clipping controller class.

     Revision 1.1  2005/02/12 06:43:44  jmp
     Clipping controller for Sutherland-Hodgeman clipping.

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_CONTROLLER_CLIPPING_H_
#define GFX_CONTROLLER_CLIPPING_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"
#include "GraphicsPipe.h"
#include "Matrix4.h"
#include "Frustum.h"
#include "Sphere.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxController_Clipping
/*!
*/
{
  public:
    // ct and dt
             gfxController_Clipping();
    virtual ~gfxController_Clipping();
    
		// operation - compute frustum in reference frame defined by parameter proj
		virtual gfxFrustum ComputeFrustum(const gfxMatrix4&	proj) = 0;

		// operations - culling
    virtual bool Cull(const gfxSphere& bs, const gfxFrustum& f, gfxOutCode* oc) = 0;
    
    // operations - clipping
    virtual gfxVertexBuffer Clip(gfxOutCode oc, const gfxVertexBuffer& p) = 0;
};


#endif  /* GFX_CONTROLLER_CLIPPING_H_ */