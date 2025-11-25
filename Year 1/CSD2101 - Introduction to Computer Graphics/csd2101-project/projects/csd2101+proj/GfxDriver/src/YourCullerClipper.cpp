/*!
@file		YourCullerClipper.cpp
@author		Prasanna Ghali		(pghali@digipen.edu)
@co-author  Bryan Ang Wei Ze    (bryanweize.ang@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "YourCullerClipper.h"

/*                                                                  functions
----------------------------------------------------------------------------- */
/*  _________________________________________________________________________ */
gfxFrustum YourClipper::
ComputeFrustum(gfxMatrix4 const& perspective_mtx)
/*! Get view frame frustum plane equations

  @param perspective_mtx	--> Matrix manifestation of perspective (or, orthographic)
  transform.

  @return	--> gfxFrustum
  Plane equations of the six surfaces that specify the view volume in view frame.
*/
{
	gfxVector4 r0{ perspective_mtx.GetRow4(0) },
		r1{ perspective_mtx.GetRow4(1) },
		r2{ perspective_mtx.GetRow4(2) },
		r3{ perspective_mtx.GetRow4(3) };

	// view frustum plane equations in view frame (non-normalized)
	gfxFrustum view_frame_frustum;

	// left
	view_frame_frustum.l.a = -r0.x - r3.x;
	view_frame_frustum.l.b = -r0.y - r3.y;
	view_frame_frustum.l.c = -r0.z - r3.z;
	view_frame_frustum.l.d = -r0.w - r3.w;

	// right
	view_frame_frustum.r.a = r0.x - r3.x;
	view_frame_frustum.r.b = r0.y - r3.y;
	view_frame_frustum.r.c = r0.z - r3.z;
	view_frame_frustum.r.d = r0.w - r3.w;

	// bottom
	view_frame_frustum.b.a = -r1.x - r3.x;
	view_frame_frustum.b.b = -r1.y - r3.y;
	view_frame_frustum.b.c = -r1.z - r3.z;
	view_frame_frustum.b.d = -r1.w - r3.w;

	// top
	view_frame_frustum.t.a = r1.x - r3.x;
	view_frame_frustum.t.b = r1.y - r3.y;
	view_frame_frustum.t.c = r1.z - r3.z;
	view_frame_frustum.t.d = r1.w - r3.w;

	// near
	view_frame_frustum.n.a = -r2.x - r3.x;
	view_frame_frustum.n.b = -r2.y - r3.y;
	view_frame_frustum.n.c = -r2.z - r3.z;
	view_frame_frustum.n.d = -r2.w - r3.w;

	// far
	view_frame_frustum.f.a = r2.x - r3.x;
	view_frame_frustum.f.b = r2.y - r3.y;
	view_frame_frustum.f.c = r2.z - r3.z;
	view_frame_frustum.f.d = r2.w - r3.w;

	// compute lengths of computed view frustum plane normals
	float length_l, length_r, length_b, length_t, length_n, length_f;
	length_l = sqrtf(view_frame_frustum.l.a * view_frame_frustum.l.a + view_frame_frustum.l.b * view_frame_frustum.l.b + view_frame_frustum.l.c * view_frame_frustum.l.c);
	length_r = sqrtf(view_frame_frustum.r.a * view_frame_frustum.r.a + view_frame_frustum.r.b * view_frame_frustum.r.b + view_frame_frustum.r.c * view_frame_frustum.r.c);
	length_b = sqrtf(view_frame_frustum.b.a * view_frame_frustum.b.a + view_frame_frustum.b.b * view_frame_frustum.b.b + view_frame_frustum.b.c * view_frame_frustum.b.c);
	length_t = sqrtf(view_frame_frustum.t.a * view_frame_frustum.t.a + view_frame_frustum.t.b * view_frame_frustum.t.b + view_frame_frustum.t.c * view_frame_frustum.t.c);
	length_n = sqrtf(view_frame_frustum.n.a * view_frame_frustum.n.a + view_frame_frustum.n.b * view_frame_frustum.n.b + view_frame_frustum.n.c * view_frame_frustum.n.c);
	length_f = sqrtf(view_frame_frustum.f.a * view_frame_frustum.f.a + view_frame_frustum.f.b * view_frame_frustum.f.b + view_frame_frustum.f.c * view_frame_frustum.f.c);

	// normalize the computed view frustum plane normals
	// left
	view_frame_frustum.l.a /= length_l;
	view_frame_frustum.l.b /= length_l;
	view_frame_frustum.l.c /= length_l;
	view_frame_frustum.l.d /= length_l;

	// right
	view_frame_frustum.r.a /= length_r;
	view_frame_frustum.r.b /= length_r;
	view_frame_frustum.r.c /= length_r;
	view_frame_frustum.r.d /= length_r;

	// bottom
	view_frame_frustum.b.a /= length_b;
	view_frame_frustum.b.b /= length_b;
	view_frame_frustum.b.c /= length_b;
	view_frame_frustum.b.d /= length_b;

	// top
	view_frame_frustum.t.a /= length_t;
	view_frame_frustum.t.b /= length_t;
	view_frame_frustum.t.c /= length_t;
	view_frame_frustum.t.d /= length_t;

	// near
	view_frame_frustum.n.a /= length_n;
	view_frame_frustum.n.b /= length_n;
	view_frame_frustum.n.c /= length_n;
	view_frame_frustum.n.d /= length_n;

	// far
	view_frame_frustum.f.a /= length_f;
	view_frame_frustum.f.b /= length_f;
	view_frame_frustum.f.c /= length_f;
	view_frame_frustum.f.d /= length_f;

	return view_frame_frustum;
}

/*  _________________________________________________________________________ */
bool YourClipper::
Cull(gfxSphere const& bounding_sphere, gfxFrustum const& frustum, gfxOutCode *ptr_outcode)
/*! Performing culling.

@param bs		--> View-frame definition of the bounding sphere of object
which is being tested for inclusion, exclusion, or
intersection with view frustum.
@param f		--> View-frame frustum plane equations.
@param oc		--> Six-bit flag specifying the frustum planes intersected
by bounding sphere of object. A given bit of the outcode
is set if the sphere crosses the appropriate plane for that
outcode bit - otherwise the bit is cleared.

@return
True if the vertices bounded by the sphere should be culled.
False otherwise.

If the return value is false, the outcode oc indicates which planes
the sphere intersects with. A given bit of the outcode is set if the
sphere crosses the appropriate plane for that outcode bit.
*/
{
	// for now, assume bounding sphere completely inside frustum
	*ptr_outcode = 0;
	float dist{};

	// loop through all 6 planes (left -> right -> bottom -> top -> near -> far)
	for (size_t i{}; i < 6; ++i)
	{
		// compute distance of sphere from current plane
		dist = frustum.mPlanes[i].a * bounding_sphere.center.x + frustum.mPlanes[i].b * bounding_sphere.center.y + frustum.mPlanes[i].c * bounding_sphere.center.z + frustum.mPlanes[i].d;
		
		// bounding sphere is completely outside plane
		if (dist > bounding_sphere.radius)
		{
			return true;
		}
		// bounding sphere is straddling plane
		else if (-bounding_sphere.radius < dist && dist < bounding_sphere.radius)
		{
			*ptr_outcode |= static_cast<gfxOutCode>(1) << i;
		}
	}

	// bounding sphere is either completely inside or straddling planes
	return false;
}

/*  _________________________________________________________________________ */
gfxVertexBuffer YourClipper::
Clip(gfxOutCode outcode, gfxVertexBuffer const& vertex_buffer)
/*!
Perform clipping.

@param outcode	--> Outcode of view-frame of bounding sphere of object specifying
the view-frame frustum planes that the sphere is straddling.

@param vertex_buffer	--> The input vertex buffer contains three points
forming a triangle.
Each vertex has x_c, y_c, z_c, and w_c fields that describe the position
of the vertex in clip space. Additionally, each vertex contains an array
of floats (bs[6]) that contains the boundary condition for each clip plane,
and an outcode value specifying which planes the vertex is inside.

The gfxClipPlane enum in GraphicsPipe.h contains the indices into bs
array. gfxClipCode contains bit values for each clip plane code.

If an object's bounding sphere could not be trivially accepted nor rejected,
it is reasonable to expect that the object is straddling only a
subset of the six frustum planes. This means that the object's triangles
need not be clipped against all the six frustum planes but only against
the subset of planes that the bounding sphere is straddling.
Furthermore, even if the bounding sphere is straddling a subset of planes,
the triangles themselves can be trivially accepted or rejected.
To implement the above two insights, use argument outcode - the object bounding
sphere's outcode which was previously returned by Cull().

Notes:
When computing clip frame intersection points, ensure that all necessary information
required to project and rasterize the vertex is computed using linear interpolation
between the inside and outside vertices. This includes:
clip frame coordinates: (c_x, c_y, c_z, c_w),
texture coordinates: (u, v),
vertex color coordinates: (r, g, b, a),
boundary conditions: bc[GFX_CPLEFT], bc[GFX_CPRIGHT], ...
outcode: oc

As explained in class, consistency in computing computing the parameter t
using t = 0 for inside point and t = 1 for outside point helps in preventing
tears and other artifacts.

Although the input primitive is a triangle, after clipping, the output
primitive may be a convex polygon with more than 3 vertices. In that
case, you must produce as output an ordered list of clipped vertices
that form triangles when taken in groups of three.

@return None
*/
{
	// triangle is outside frustum
	if (vertex_buffer[0].oc & vertex_buffer[1].oc & vertex_buffer[2].oc)
	{
		return gfxVertexBuffer();
	}
	// triangle is inside frustum
	else if (!(vertex_buffer[0].oc | vertex_buffer[1].oc | vertex_buffer[2].oc))
	{
		return vertex_buffer;
	}
	
	// initial polygon before clipping
	gfxVertexBuffer result{ vertex_buffer };

	// plane clip codes
	std::vector<gfxClipCode> plane_clip_codes{ GFX_CCLEFT, GFX_CCRIGHT, GFX_CCBOTTOM, GFX_CCTOP, GFX_CCNEAR, GFX_CCFAR };

	for (size_t current_plane{}; current_plane < plane_clip_codes.size(); ++current_plane)
	{
		gfxClipCode clip_code{ plane_clip_codes[current_plane] };
		gfxVertexBuffer current;
		size_t num_vertices{ result.size() };

		// clip only against relevant planes
		if (outcode & clip_code)
		{
			for (size_t j{}; j < num_vertices; ++j)
			{
				const gfxVertex& curr{ result[j] };
				const gfxVertex& next{ result[(j + 1) % num_vertices] };

				// function object that calculates intersection time and interpolates vertices
				auto interpolate_vertex = [&](const gfxVertex& v0, const gfxVertex& v1) -> gfxVertex
				{
					float t{ v1.bc[current_plane] / (v1.bc[current_plane] - v0.bc[current_plane]) };
					gfxVertex out_vtx;

					// interpolate clip coordinates
					out_vtx.x_c = t * v0.x_c + (1.f - t) * v1.x_c;
					out_vtx.y_c = t * v0.y_c + (1.f - t) * v1.y_c;
					out_vtx.z_c = t * v0.z_c + (1.f - t) * v1.z_c;
					out_vtx.w_c = t * v0.w_c + (1.f - t) * v1.w_c;

					// interpolate texture coordinates
					out_vtx.s = t * v0.s + (1.f - t) * v1.s;
					out_vtx.t = t * v0.t + (1.f - t) * v1.t;

					// interpolate vertex color coordinates
					out_vtx.r = t * v0.r + (1.f - t) * v1.r;
					out_vtx.g = t * v0.g + (1.f - t) * v1.g;
					out_vtx.b = t * v0.b + (1.f - t) * v1.b;
					out_vtx.a = t * v0.a + (1.f - t) * v1.a;

					// calculate new boundary conditions
					out_vtx.bc[GFX_CPLEFT] = -out_vtx.x_c - out_vtx.w_c;
					out_vtx.bc[GFX_CPRIGHT] = out_vtx.x_c - out_vtx.w_c;
					out_vtx.bc[GFX_CPBOTTOM] = -out_vtx.y_c - out_vtx.w_c;
					out_vtx.bc[GFX_CPTOP] = out_vtx.y_c - out_vtx.w_c;
					out_vtx.bc[GFX_CPNEAR] = -out_vtx.z_c - out_vtx.w_c;
					out_vtx.bc[GFX_CPFAR] = out_vtx.z_c - out_vtx.w_c;

					// calculate outcode of intersection point
					out_vtx.oc = 0;
					for (size_t i{}; i < plane_clip_codes.size(); ++i)
					{
						if (out_vtx.bc[i] > 0.f)
						{
							out_vtx.oc |= static_cast<gfxOutCode>(1) << i;
						}
					}

					return out_vtx;
				};

				// both vertices are inside
				if (!(curr.oc & clip_code) && !(next.oc & clip_code))
				{
					current.push_back(next);
				}
				// current vertex is inside and next vertex is outside
				else if (!(curr.oc & clip_code) && (next.oc & clip_code))
				{
					current.push_back(interpolate_vertex(curr, next));
				}
				// current vertex is outside and next is inside
				else if ((curr.oc & clip_code) && !(next.oc & clip_code))
				{
					current.push_back(interpolate_vertex(next, curr));
					current.push_back(next);
				}
			}

			// update output polygon vertices of after clipping against current plane
			result = current;
		}
	}

	if (result.size() <= 3)
	{
		return result;
	}

	// triangulate output polygon if it has more than 3 vertices
	gfxVertexBuffer tri_fans;
	for (size_t i{}; i < result.size() - 2; ++i)
	{
		tri_fans.push_back(result[0]);
		tri_fans.push_back(result[i + 1]);
		tri_fans.push_back(result[i + 2]);
	}

	return tri_fans;
}