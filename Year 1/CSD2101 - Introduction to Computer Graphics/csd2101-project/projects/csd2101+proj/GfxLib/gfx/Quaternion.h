/*!
@file    Quaternion.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Quaternion.h,v 1.2 2005/02/16 21:15:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_QUATERNION_H_
#define GFX_QUATERNION_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

#include "Vector3.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxQuaternion
/*! Quaternion class.
*/
{
  // friends
  friend gfxQuaternion operator~(const gfxQuaternion& q);
  friend gfxQuaternion operator*(const gfxQuaternion& l, const gfxQuaternion& r);
  friend gfxVector3    operator*(const gfxQuaternion& l, const gfxVector3& r);
  
  public:
    // ct and dt 
    gfxQuaternion();
    gfxQuaternion(float si, float xi, float yi, float zi);
    gfxQuaternion(const float* q);
    gfxQuaternion(const gfxVector3& axis, float angle);
    gfxQuaternion(const gfxQuaternion& q);
    ~gfxQuaternion();
    
    // = operator: assignment
    gfxQuaternion& operator=(const gfxQuaternion& q);
    
    // [] operators: component access
    float&       operator[](unsigned int i);
    const float& operator[](unsigned int i) const;

    // reinitialization
    void SetToIdentity();

    // delegates
    float &s;  //!< Access to S component.
    float &x;  //!< Access to X component.
    float &y;  //!< Access to Y component.
    float &z;  //!< Access to Z component.
    
  protected:
    // data members
    float  mQuat[4];  //!< gfxQuaternion elements.
};


/*                                                             implementation
----------------------------------------------------------------------------- */


#endif  /* GFX_QUATERNION_H_ */