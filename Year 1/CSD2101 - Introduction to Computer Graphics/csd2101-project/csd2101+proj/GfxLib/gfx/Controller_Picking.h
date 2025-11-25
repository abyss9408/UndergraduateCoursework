/*!
@file     Controller_Picking.h
@author  Prasanna Ghali       (pghali@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_CONTROLLER_PICKING_H_
#define GFX_CONTROLLER_PICKING_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"
#include "GraphicsPipe.h"
#include "Object.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

// forward declarations
class gfxCamera;
class gfxVector3;

/*  _________________________________________________________________________ */
class gfxController_Picking
/*! Subclass gfxController_Picking and implement its virtual methods to override
    the default picking behavior with your own.
*/
{
  public:
    // ct and dt
             gfxController_Picking();
    virtual ~gfxController_Picking();

    // operations - picking
    virtual int Pick(unsigned int					xd,
										unsigned int					yd,
										const gfxObjectList&	scene,
										const gfxCamera*			cam) = 0;

};


#endif  // GFX_CONTROLLER_TRANSFORMATION_H_
