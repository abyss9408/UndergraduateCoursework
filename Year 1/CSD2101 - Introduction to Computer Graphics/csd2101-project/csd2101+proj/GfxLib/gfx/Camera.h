/*!
@file    Camera.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Camera.h,v 1.3 2005/02/22 04:10:51 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_CAMERA_H_
#define GFX_CAMERA_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"
#include "Vector3.h"
#include "Vector4.h"
#include "Matrix4.h"
#include "Quaternion.h"
#include "Frustum.h"
#include "Math.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
class gfxCamera
/*! Camera class.
*/
{ 
  public:
     // ct and dt
             gfxCamera();
    virtual ~gfxCamera();
    
    // accessors
    float				GetAzimuth() const  { return (mAzimuth); }
    float				GetLatitude() const { return (mLatitude); }
    float				GetRadius() const   { return (mRadius); }
    float				GetFOV() const      { return (mVertFOV); }
    float				GetAspect() const   { return (mAspect); }
    float				GetNearDist() const { return (mDistNear); }
    float				GetFarDist() const  { return (mDistFar); }
    
    // manipulators
    void				SetAzimuth(float azimuth) { mAzimuth		= fclampw(azimuth, 0.f, TWO_PI); UpdatePointsFromSpherical(); }
    void				SetLatitude(float lat)    { mLatitude	= fclamp(lat, -HALF_PI, HALF_PI); UpdatePointsFromSpherical(); }
    void				SetRadius(float radius)   { mRadius		= radius; }
    void				SetFOV(float fov)         { mVertFOV		= fov; }
    void				SetAspect(float aspect)   { mAspect		= aspect; }
    void				SetNearDist(float n)      { mDistNear	= n; }
    void				SetFarDist(float f)       { mDistFar		= f; }

   // camera position
    gfxVector3	GetPosition() const										{ return (mFrom); }
    void				SetPosition(const gfxVector3& p)				{ mFrom = p; UpdateSphericalFromPoints(); }
    void				SetPosition(float x, float y, float z) { mFrom.SetTo(x,y,z); UpdateSphericalFromPoints(); }
    
    // target
    gfxVector3	GetTarget() const									{ return (mAt); }
    void				SetTarget(const gfxVector3& t)			{ mAt = t; UpdateSphericalFromPoints(); }
    void				SetTarget(float x,float y,float z)	{ mAt.SetTo(x,y,z); UpdateSphericalFromPoints(); }
    
    // up vector
    gfxVector3	GetUp() const											{ return (mUp); }
    void				SetUp(const gfxVector3& u)					{ mUp = u; }
    void				SetUp(float x, float y, float z)		{ mUp.SetTo(x, y, z); }

		// viewport
    void				SetViewport(float x,float z,float w,float h) { mVPX = x; mVPY = z; mVPW = w; mVPH = h; }
    size_t			GetViewportWidth() const											{ return (static_cast<size_t>(mVPW)); }
    size_t			GetViewportHeight() const										{ return (static_cast<size_t>(mVPH)); }

    // View, projection, and NDC-to-viewport transformation operators.
		void				SetLookAtMatrix();
		void				SetPerspMatrix(float fovy, float aspect, float near, float far);
		void				SetPerspMatrix(float left, float right, float bottom, float top, float near, float far);
		void				SetOrthoMatrix(float left, float right, float bottom, float top, float near, float far);
		void				SetViewportMatrix(float x, float y, float width, float height);
		
		// Compute camera's orientation (in terms of basis vectors) in world frame
		void				GetBasisVectors(gfxVector3* bx, gfxVector3* by, gfxVector3* bz);

		gfxMatrix4	GetLookAtMatrix() const;
    gfxMatrix4	GetProjectionMatrix() const;
    gfxMatrix4	GetViewportMatrix() const;

		// update camera and target positions along side, up, and view axes of camera.
    virtual void	Move(const gfxVector3& p);
    virtual void	Move(float x, float y, float z);
    virtual void	UpdatePointsFromSpherical();

   // update camera's spherical coordinates from new camera and target positions
    virtual void	UpdateSphericalFromPoints();

protected:

		gfxVector3	mFrom;			//!< Camera position.
    gfxVector3	mAt;				//!< Camera target.
		gfxVector3	mUp;				//!< Camera up orientation
   
		// Spherical coordinates
    float				mRadius;		//!< Distance between camera at and target positions.
														//!< Here, we are specifying the radius of a sphere
														//!< centered at mFrom and mAt is a point on the surface
														//!< of the sphere.
    float				mLatitude;	//!< Latitude angle alpha [-pi/2..pi/2] where 
														//!< (pi/2 - alpha) radians is the angle between the view-vector
														//!< and up axis (0, 1, 0). When alpha is pi/2, the view-vector
														//!< is oriented along (0, 1, 0) and when alpha is -pi/2, the
														//!< view-vector is oriented along (0, -1, 0).
														//!< See class notes for pictures.
    float				mAzimuth;		//!< Azimuth angle beta [0..2pi] where beta is the angle between
														//!< the view vector projected onto the ZX-plane and z-axis (0, 0, 1).
														//!< beta increases from 0 to 2pi radians in counter-clockwise direction
														//!< when viewed from a point on up axis looking down along negative up axis.
														//!< See class presentations for pictures.

		// Make copies of view, projection, and viewport transforms.
		// These are useful for other applications such as culling,
		// intersections, picking, ...
		gfxMatrix4	mLookAtMtx;
		gfxMatrix4	mProjMtx;
		gfxMatrix4	mViewportMtx;

    float				mVertFOV;		//!< Camera's vertical FOV.
														//!< Since this is a user-specified value, unit of
														//!  measurement is degrees and not radians.
    float				mAspect;		//!< Camera's aspect ratio.
    float				mDistNear;	//!< (Positive) Distance to near clip plane from view point.
    float				mDistFar;		//!< (Positive) Distance to far clip plane from view point.
    
    float				mVPW, mVPH;	//!< Width and height of camera's viewport.
    float				mVPX, mVPY;	//!< Top-left corner of camera's viewport within window device
														//!< allocated to graphics pipe.
};


#endif  /* GFX_CAMERA_H_ */