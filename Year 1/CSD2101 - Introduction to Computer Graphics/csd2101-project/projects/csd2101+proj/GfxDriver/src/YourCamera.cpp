/*!
@file		YourCamera.cpp
@author		Prasanna Ghali		(pghali@digipen.edu)
@co-author  Bryan Ang Wei Ze    (bryanweize.ang@digipen.edu)

CVS: $Id: Camera.cpp,v 1.13 2005/03/15 23:34:41 pghali Exp $

All content (c) 2005 DigiPen (USA) Corporation, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "YourCamera.h"

/*                                                                  functions
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
void YourCamera::
Move(gfxVector3 const& p)
/*! Displace the camera along its basis vectors.

    @param p -->  Displacement vector such that p.x, p.y, and p.z are
									displacements along the camera's side, up, and viewing
									basis vectors.
*/
{
	/*
		Displace the camera so that it is always following the planar floor but
		can look up or down. This behavior is unlike a flight simulator where the
		camera moves in the direction of the view vector.
		Check out the sample application to make sure you understand the basic mechanics
		of the required camera.
	*/
	// compuate camera right and view vector
	gfxVector3 view{ mAt - mFrom };
	view.Normalize();

	gfxVector3 right{ view ^ mUp };
	right.Normalize();

	// update camera position along i and k basis vecs
	mFrom += gfxVector3(view.x * p.z, 0.f, view.z * p.z) + gfxVector3(right.x * p.x, 0.f, right.z * p.x);

	// add vertical displacement to y-coord
	mFrom += p.y * mUp;

	// update camera target
	mAt = mFrom + view;
}

/*  _________________________________________________________________________ */
void YourCamera::
Move(float x, float y, float z)
/*! Displace the camera along its basis vectors.

    @param x -->  Displacement along camera's side (X) axis.
    @param y -->  Displacement along camera's up (Y) axis.
    @param z -->  Displacement along camera's view (-Z) axis.
*/
{
	/*
		Displace the camera such it is always following the planar floor but
		can look up or down. This behavior is unlike a flight simulator where the
		camera moves in the direction of the view vector.
		Check out the sample application to make sure you understand the basic mechanics
		of the required camera.
	*/
	Move(gfxVector3(x, y, z));
}

/*  _________________________________________________________________________ */
void YourCamera::
UpdateSphericalFromPoints()
/*! Updates camera's spherical coordinates using updated
		camera position mFrom and camera target mAt.
*/
{
	gfxVector3 view{ mAt - mFrom };

	mRadius = sqrt(view.x * view.x + view.y * view.y + view.z * view.z);
	mLatitude = asin(view.y / mRadius);

	if (view.z != 0.f)
	{
		if (view.x >= 0.f)
		{
			mAzimuth = atan2(view.x, view.z);
		}
		else
		{
			mAzimuth = 2 * PI + atan2(view.x, view.z);
		}
	}
	else
	{
		if (view.x > 0.f)
		{
			mAzimuth = PI / 2;
		}
		else if (view.x < 0.f)
		{
			mAzimuth = 3 * PI / 2;
		}
	}
}

/*  _________________________________________________________________________ */
void YourCamera::
UpdatePointsFromSpherical()
/*! Updates camera's target position mAt using camera's
		updated spherical coordinates.
*/
{
	gfxVector3 view;

	view.x = mRadius * cos(mLatitude) * sin(mAzimuth);
	view.y = mRadius * sin(mLatitude);
	view.z = mRadius * cos(mLatitude) * cos(mAzimuth);

	mAt = mFrom + view;
}
