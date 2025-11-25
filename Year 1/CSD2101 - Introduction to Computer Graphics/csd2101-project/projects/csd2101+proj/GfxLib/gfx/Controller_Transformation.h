/*!
@file     Controller_Transformation.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Controller_Transformation.h,v 1.4 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_CONTROLLER_TRANSFORMATION_H_
#define GFX_CONTROLLER_TRANSFORMATION_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"
#include "GraphicsPipe.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxController_Transformation
/*! Subclass gfxController_Transformation and implement TransformBuffer().
    TransformBuffer() will be passed the world, view and projection matrices, as
    well as viewport bounding information and a pointer to a list of vertices
    to transform. You should transform each vertex and store the results in that
    vertex's x_d and y_d fields.
*/
{
  public:
    // ct and dt
             gfxController_Transformation();
    virtual ~gfxController_Transformation();

    // operations
		virtual void TransformBuffer(	const gfxGraphicsPipe*	dev,
																	const gfxMatrix4&				modelview,
																	const gfxMatrix4&				project,
																	gfxVertexBuffer*				vb) = 0;
};


#endif  // GFX_CONTROLLER_TRANSFORMATION_H_
