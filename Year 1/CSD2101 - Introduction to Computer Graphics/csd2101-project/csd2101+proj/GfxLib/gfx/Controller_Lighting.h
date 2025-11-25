/*!
@file    Controller_Lighting.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Controller_Lighting.h,v 1.3 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_CONTROLLER_LIGHTING_H_
#define GFX_CONTROLLER_LIGHTING_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"
#include "GraphicsPipe.h"
#include "Frustum.h"
#include "Vector3.h"
#include "Vector4.h"
#include "Matrix4.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxController_Lighting
/*! Subclass gfxController_Lighting and override LightVertex() to implement
    lighting in your application. When the pipe invokes LightVertex, it will
    have filled out the world-frame fields of the vertex it provides.
*/
{
  public:
    // ct and dt
             gfxController_Lighting();
    virtual ~gfxController_Lighting();
    
    // operations - lighting
		virtual void LightVertex(gfxVertex&					vertex,
														const gfxVector3&		light,
														const gfxMaterial&	dm) = 0;
    
    // operations - shadows
    virtual std::vector<gfxVector4> GetShadowGeometry(const gfxVertexBuffer&	verts,
																											const gfxMatrix4&				modelview,
																											const gfxMatrix4&				proj,
																											const gfxPlane&					plane,
																											const gfxVector3&				light) = 0;
};


#endif  /* GFX_CONTROLLER_LIGHTING_H_ */