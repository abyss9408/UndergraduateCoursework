/*!
@file    YourRasterizer.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: YourRasterizer.h,v 1.13 2005/03/15 23:34:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#ifndef YOUR_RASTERIZER_H_
#define YOUR_RASTERIZER_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include  "../../GfxLib/gfx/GFX.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class YourRasterizer : public gfxController_Rasterization
/*! Example rasterizer subclass.
    
    An instance of this subclass can be created and used to override the
    reference rasterizer in a specific pipe by calling the UseRasterizer()
    method of gfxGraphicsPipe. 
    
    You can set the render mode of a pipe by calling SetRenderMode() and
    passing either GFX_POINTS, GFX_WIREFRAME, or GFX_FILLED. The appropriate
    function from this rasterizer subclass will be invoked for each triangle.
    
    Within each function, dev points to the pipe to which the triangle should
    be rasterized. You can get a pointer to the framebuffer and its width and
    height from there. Each vertex contains x_d and y_d fields (X and Y screen
    position) as well as an z_d field (depth value), r, g, b color fields, and
		u, v texture coordinate fields.
    
    The pipe framebuffer is an array of unsigned ints in ARGB format. The
    examples below should be enough to get you started.
*/
{
  public:
    // ct and dt
             YourRasterizer() { }
    virtual ~YourRasterizer() { }

    // operations
    virtual void DrawPoint(gfxGraphicsPipe*, gfxVertex const&);
		virtual void DrawLine(gfxGraphicsPipe*, gfxVertex const&, gfxVertex const&);
		virtual void DrawWireframe(gfxGraphicsPipe*, gfxVertex const&, gfxVertex const&, gfxVertex const&);
		virtual void DrawFilled(gfxGraphicsPipe*, gfxVertex const&, gfxVertex const&, gfxVertex const&);
};

#endif  /* YOUR_RASTERIZER_H_ */
