/*!
@file    YourCullerClipper.h
@author  Prasanna Ghali       (pghali@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#ifndef YOUR_CULLER_CLIPPER_H_
#define YOUR_CULLER_CLIPPER_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include  "../../GfxLib/gfx/GFX.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class YourClipper : public gfxController_Clipping
/*! Example clipper subclass.

    This class implements culling (by bounding spheres) and clipping.
    You need to implement both the Cull() and the Clip() functions. More details
    can be found in the comments above each function.
*/
{
  public:
    // ct and dt
             YourClipper() { }
    virtual ~YourClipper() { }

		// operation - compute view-frame frustum
		virtual gfxFrustum ComputeFrustum(gfxMatrix4 const&	projection_mtx);

    // operations - culling
    virtual bool Cull(gfxSphere const&	bounding_sphere,
                      gfxFrustum const&	frustum,
                      gfxOutCode        *ptr_outcode);
    
    // operations - clipping
    virtual gfxVertexBuffer Clip(gfxOutCode outcode, gfxVertexBuffer const& vertex_buffer);
};

#endif  /* YOUR_CULLER_CLIPPER_H_ */
