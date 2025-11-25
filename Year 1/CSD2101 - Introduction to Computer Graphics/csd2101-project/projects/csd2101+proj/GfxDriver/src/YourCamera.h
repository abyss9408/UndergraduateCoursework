/*!
@file    Camera.h
@author  Josh Petrie       (jmp, jpetrie@digipen.edu)
         Scott Smith       (sls, ssmith@digipen.edu)
         Patrick Laukaitis (pjl, plaukait@digipen.edu)

CVS: $Id: Rasterizer.h,v 1.13 2005/03/15 23:34:41 jmp Exp $

All content (c) 2005 DigiPen (USA) Corporation, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#ifndef CAMERA_H_
#define CAMERA_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include  "../../GfxLib/gfx/GFX.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class YourCamera : public gfxCamera
/*! Example camera wrapper class.

    This class contains the basic methods for a spherical camera interface that
    wraps the quaternion-based camera used internally by the device. All
    methods are empty and the class has no data other than a pointer to the
    actual camera that is being wrapped. You can manipulate the actual camera
    with the Update() method, which takes the eye position and target position.
    You don't need to deal with support camera roll (e.g., altering up vector).
    
    You are expected to store or compute, as appropriate, the camera's position,
    and view vector using spherical coordinates.
*/
{
  public:
    // ct and dt
             YourCamera() { }
    virtual ~YourCamera() { }
    
    // displacement
    virtual void Move(float x, float y, float z);
    virtual void Move(gfxVector3 const& p);
    
    // updates
    virtual void UpdateSphericalFromPoints();
    virtual void UpdatePointsFromSpherical();
};

#endif  /* CAMERA_H_ */