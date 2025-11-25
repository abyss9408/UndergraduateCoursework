/*!
@file    PolyLineGenerator.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: PolyLineGenerator.h,v 1.2 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_MESHGENERATOR_POINT_H_
#define GFX_MESHGENERATOR_POINT_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

// forward declarations
class gfxModel;

/*  _________________________________________________________________________ */
class gfxMeshGenerator_Point
/*! 
*/
{ 
  public:
    // importer
    static gfxModel GenerateCircleBoundary(float radius, float res);
  
  private:
    // disabled
     gfxMeshGenerator_Point(); 
    ~gfxMeshGenerator_Point();
};


#endif  /* GFX_MESHGENERATOR_POINT_H_ */