/*!
@file    MeshGenerator_Plane.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: MeshGenerator_Plane.h,v 1.2 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_MESHGENERATOR_PLANE_H_
#define GFX_MESHGENERATOR_PLANE_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

// forward declarations
class gfxModel;

/*  _________________________________________________________________________ */
class gfxMeshGenerator_Plane
/*! 
*/
{ 
  public:
    // importer
    static gfxModel Generate(const WCHAR* tex, int res, float size);
  
  private:
    // disabled
     gfxMeshGenerator_Plane(); 
    ~gfxMeshGenerator_Plane();
};


#endif  /* GFX_MESHGENERATOR_PLANE_H_ */