/*!
@file		YourBoundingSphere.cpp
@author		Prasanna Ghali		(pghali@digipen.edu)
@co-author  Bryan Ang Wei Ze    (bryanweize.ang@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#include  "../../GfxLib/gfx/GFX.h"

/*  _________________________________________________________________________ */
gfxSphere gfxModel::
ComputeModelBVSphere(std::vector<gfxVector3> const& verts)
/*! Compute model's bounding sphere (in model frame) using Ritter's method.

    @param verts	-->	Model-frame vertices of scene object. This function will be
                      called once for each model as it is loaded. You should compute
                      the bounding sphere containing all the input vertices
                      stored in the verts vector using Ritter's method.
                      See GfxLib/Model.h, GfxLib/Sphere.h, GfxLib/Vertex.h for more
                      information regarding gfxModel, gfxSphere, and gfxVertex class
                      declarations.

    @return
    Object of type gfxSphere defining the bounding sphere containing the model using
    Ritter's method.
*/
{
    gfxVector3 min_x, max_x, min_y, max_y, min_z, max_z, center;
    float dist_x{}, dist_y{}, dist_z{}, max_dist{}, radius{};
    min_x = max_x = min_y = max_y = min_z = max_z = verts[0];

    // find the extreme points along each axis
    for (const gfxVector3& vert : verts)
    {
        if (vert.x < min_x.x) min_x = vert;
        if (vert.x > max_x.x) max_x = vert;
        if (vert.y < min_y.y) min_y = vert;
        if (vert.y > max_y.y) max_y = vert;
        if (vert.z < min_z.z) min_z = vert;
        if (vert.z > max_z.z) max_z = vert;
    }

    /* calculate max distance between two points, with that then calculate the midpoint(center)
    between those points and half the distance(radius) */
    dist_x = sqrtf((max_x.x - min_x.x) * (max_x.x - min_x.x) +
        (max_x.y - min_x.y) * (max_x.y - min_x.y) +
        (max_x.z - min_x.z) * (max_x.z - min_x.z));
    dist_y = sqrtf((max_y.x - min_y.x) * (max_x.x - min_y.x) +
        (max_y.y - min_y.y) * (max_y.y - min_y.y) +
        (max_y.z - min_y.z) * (max_y.z - min_y.z));
    dist_z = sqrtf((max_z.x - min_z.x) * (max_x.x - min_x.x) +
        (max_x.y - min_x.y) * (max_x.y - min_x.y) +
        (max_x.z - min_x.z) * (max_x.z - min_x.z));
    max_dist = std::max({ dist_x , dist_y, dist_z });

    if (max_dist == dist_x)
    {
        center = { (max_x.x + min_x.x) / 2,(max_x.y + min_x.y) / 2,(max_x.z + min_x.z) / 2 };
    }
    else if (max_dist == dist_y)
    {
        center = { (max_y.x + min_y.x) / 2,(max_y.y + min_y.y) / 2,(max_y.z + min_y.z) / 2 };
    }
    else
    {
        center = { (max_z.x + min_z.x) / 2,(max_z.y + min_z.y) / 2,(max_z.z + min_z.z) / 2 };
    }
    radius = max_dist * 0.5f;
    

    for (const gfxVector3& vert : verts)
    {
        gfxVector3 center_to_vert{ vert - center };

        // grow sphere if current vert is outside the current sphere
        if (center_to_vert * center_to_vert > radius * radius)
        {
            gfxVector3 u{ center_to_vert.x / center_to_vert.Length(),
                center_to_vert.y / center_to_vert.Length(),
                center_to_vert.z / center_to_vert.Length() };
            gfxVector3 vert_prime{ center - radius * u };

            // compute new center and radius of new sphere
            center = { (vert_prime.x + vert.x) / 2,
                (vert_prime.y + vert.y) / 2,
                (vert_prime.z + vert.z) / 2 };
            radius = (vert - center).Length();
        }
    }

    return (gfxSphere(center.x, center.y, center.z, radius));
}

/*  _________________________________________________________________________ */
gfxSphere gfxSphere::
Transform(gfxMatrix4 const& xform) const
/*! Transform a bounding sphere to a destination reference frame using
    the matrix manifestation of the transform from model to destination
    frame. Note that the only valid affine transforms for this assignment are:
    scale (if any), followed by rotation (if any), followed by translation.

    @param xform -->  The matrix manifestation of transform from model to
                      destination frame.

    @return
    The model bounding sphere transformed to destination frame.
*/
{
    gfxVector4 new_center{ xform * gfxVector4(center.x, center.y, center.z, 1.f) };

    // extract squared scaling factors
    float squared_scale_x{ xform(0,0) * xform(0,0) + xform(0,1) * xform(0,1) + xform(0,2) * xform(0,2) },
        squared_scale_y{ xform(1,0) * xform(1,0) + xform(1,1) * xform(1,1) + xform(1,2) * xform(1,2) },
        squared_scale_z{ xform(2,0) * xform(2,0) + xform(2,1) * xform(2,1) + xform(2,2) * xform(2,2) };

    return (gfxSphere(new_center.x, new_center.y, new_center.z,
        sqrtf(std::max({ squared_scale_x, squared_scale_y, squared_scale_z })) * radius));
}
