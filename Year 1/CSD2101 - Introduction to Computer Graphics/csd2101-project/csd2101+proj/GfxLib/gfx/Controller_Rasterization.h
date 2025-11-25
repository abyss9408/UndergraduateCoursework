/*!
@file    Controller_Rasterization.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Controller_Rasterization.h,v 1.4 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_CONTROLLER_RASTERIZATION_H_
#define GFX_CONTROLLER_RASTERIZATION_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"
#include "GraphicsPipe.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxController_Rasterization
/*! Subclass gfxController_Rasterization and override DrawFilled() to implement
    your own rasterization. DrawPoint(), DrawLine() and DrawWireframe() are not used,
		you must implement them but they don't need to do anything. Yet.
    
    After transformation and lighting, the pipe will invoke DrawFilled() for
    each triangle rendered. All fields of all input vertices will be filled out.
*/
{
  public:
    // ct and dt
             gfxController_Rasterization();
    virtual ~gfxController_Rasterization();
    
    // operations
		virtual void DrawPoint(	gfxGraphicsPipe*	dev,
														const gfxVertex&	v0) = 0;
		virtual void DrawLine(gfxGraphicsPipe*	dev,
													const gfxVertex&	v0,
													const gfxVertex&	v1) = 0;
    virtual void DrawWireframe(	gfxGraphicsPipe*	dev,
																const gfxVertex&	v0,
																const gfxVertex&	v1,
																const gfxVertex&	v2) = 0;
    virtual void DrawFilled(gfxGraphicsPipe*	dev,
														const gfxVertex&	v0,
														const gfxVertex&	v1,
														const gfxVertex&	v2) = 0;
};


#endif  /* GFX_CONTROLLER_RASTERIZATION_H_ */