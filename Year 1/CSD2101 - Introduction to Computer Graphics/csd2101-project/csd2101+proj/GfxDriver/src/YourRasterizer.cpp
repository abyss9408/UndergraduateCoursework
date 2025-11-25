/*!
@file		YourRasterizer.cpp
@author		Prasanna Ghali		(pghali@digipen.edu)
@co-author  Bryan Ang Wei Ze    (bryanweize.ang@digipen.edu)

CVS: $Id: YourRasterizer.cpp,v 1.13 2005/03/15 23:34:41 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "YourRasterizer.h"

// utility functions that help to compute triangles attributes such as area, edge equations and bounding box
float triangle_area(const gfxVertex& v0, const gfxVertex& v1, const gfxVertex& v2)
{
	return 0.5f * ((v1.x_d - v0.x_d) * (v2.y_d - v0.y_d) - (v2.x_d - v0.x_d) * (v1.y_d - v0.y_d));
}

float triangle_area(const gfxVertex& v0, const gfxVertex& v1, const gfxVector3& v2)
{
	return 0.5f * ((v1.x_d - v0.x_d) * (v2.y - v0.y_d) - (v2.x - v0.x_d) * (v1.y_d - v0.y_d));
}

gfxVector3 calculate_edge(gfxVertex const& v1, gfxVertex const& v2)
{
	return gfxVector3{ v1.y_d - v2.y_d, v2.x_d - v1.x_d, v1.x_d * v2.y_d - v2.x_d * v1.y_d };
}

void calculate_bounding_box(gfxVertex const& v0, gfxVertex const& v1, gfxVertex const& v2, 
	int width, int height, int& x_min, int& x_max, int& y_min, int& y_max)
{
	x_min = static_cast<int>(std::min({ v0.x_d, v1.x_d, v2.x_d }));
	x_max = static_cast<int>(std::ceil(std::max({ v0.x_d, v1.x_d, v2.x_d })));
	y_min = static_cast<int>(std::min({ v0.y_d, v1.y_d, v2.y_d }));
	y_max = static_cast<int>(std::ceil(std::max({ v0.y_d, v1.y_d, v2.y_d })));

	x_min = std::max(0, x_min);
	y_min = std::max(0, y_min);
	x_max = std::min(width, x_max);
	y_max = std::min(height, y_max);
}

bool is_top_left(gfxVector3 const& edge)
{
	return edge.x > 0.f || (edge.x == 0.f && edge.y < 0.f);
}

float initial_evaluation(gfxVector3 const& edge, int x, int y)
{
	return edge.x * (x + 0.5f) + edge.y * (y + 0.5f) + edge.z;
}

/*                                                                  functions
----------------------------------------------------------------------------- */
/*  _________________________________________________________________________ */
void YourRasterizer::
DrawFilled(gfxGraphicsPipe *dev, gfxVertex const&	v0, 
           gfxVertex const&	v1, gfxVertex const&	v2)
/*! Render a triangle in filled mode using 3 device-frame points.
  @param dev <->  Pointer to the pipe to render to.
  @param v0  -->  First vertex.
  @param v1  -->  Second vertex.
  @param v2  -->  Third vertex.

READ THIS FUNCTION HEADER CAREFULLY:

Begin by reading the declaration of type gfxVertex (in Vertex.h) so that
you're aware of the data encapsulated by a gfxVertex object.

-------------------------------------------------------------------------------
Triangles can be rendered in one of three ways:
1) Texture mapped (but not lit)
2) Lit (but not texture mapped)
3) Lit and texture mapped

To reduce computational overheads, this software pipe doesn't implement both
lighting and texture mapping on objects - it is one or the other.
The current rendering state of the triangle (or object) is determined by
the following call:

  unsigned int  *tex = dev->GetCurrentTexture();
    
If the function returns 0 (a null pointer), there is no texture state bound
to the currently-drawing triangle. This means that the object is lit but not 
texture mapped. Otherwise if the function returns an address, the object 
is texture mapped but not lit. The texture image is buffered in memory as
a sequence of 32-bit unsigned integral values in xxRRGGBB format which
can be directly written to the colorbuffer without any transformations.
The width and height of the texture image can be determined by the calls:

 unsigned int  tW = dev->GetCurrentTextureWidth();
 unsigned int  tH = dev->GetCurrentTextureHeight();

NOTE: Assume that GL_REPEAT is the default wrapping mode for both s and t
texture coordinates.
-------------------------------------------------------------------------------

Testing depth buffer state of the pipe:
Users can choose to turn on or off the depth buffering algorithm. The current
state of the pipe can be determined by the following call:
    
  bool          wdepth    = dev->DepthTestEnabled();

If DepthTestEnabled() returns 0, depth buffering is disabled, otherwise
it is enabled.
-------------------------------------------------------------------------------

Which interpolation scheme is to be used by the rasterizer to interpolate
vertex color and texture coordinates?
Since users can set the graphics pipe's to render scenes using linear or
hyperbolic interpolation, your rasterizer must ideally implement both these
interpolation schemes. However, since hyperbolic interpolation was not covered
in the lectures, you're only required to implement linear interpolation of 
vertex attributes using the barycentric interpolation scheme discussed in lectures.
The current interpolation state of the graphics pipe
can be determined by the call:

  gfxInterpMode im = dev->GetInterpMode();

Enumeration type gfxInterpMode is declared in GraphicsPipe.h with
enumeration constants: GFX_LINEAR_INTERP or GFX_HYPERBOLIC_INTERP.
-------------------------------------------------------------------------------

How is the front colorbuffer accessed?
The dimensions of the colorbuffer are specified by the user and have values
determined by the read-only sWindowWidth and sWindowHeight variables defined
at file-scope in main.cpp. The calls

  unsigned int  *frameBuf = dev->GetFrameBuffer();
  size_t        w         = dev->GetWidth();
  size_t        h         = dev->GetHeight();

returns the address of the first element, the width, and the height of the
of the colorbuffer. The colorbuffer is a linear array of 32-bit unsigned 
integral values. The pixel form is xxRRGGBB - the most significant 8 bits 
are unused. To write to this colorbuffer, you must convert the normalized 
color components in range [0.f, 1.f] from the interpolator into values in 
range [0, 255] and pack the components into a 32-bit integral value with 
format xxRRGGBB.
-------------------------------------------------------------------------------

How is the depthbuffer accessed?
The depthbuffer's dimensions are exactly the same as the colorbuffer. The call

  float         *depthBuf = dev->GetDepthBuffer();

returns the address of the first element of the depthbuffer. As can be seen the
depthbuffer stores depthvalues in the range [0.f, 1.f]. The depthbuffer
has the same dimensions as the colorbuffer.
-------------------------------------------------------------------------------

What are the coordinate conventions of the colorbuffer, depthbuffer and
texels in the texture image?
Memory for these buffer is allocated by Windows which is also used to blit
the contents of the colorbuffer to the display device's video memory.
Windows defines the colorbuffer with the convention that the origin is at
the upper-left corner. This means that assigning the value 0x00ff0000 to
frameBuf[0] will paint the upper-left corner with red color. Reading a
depth value from the first location of the depthbuffer provides the depth
value of the upper-left corner. Simiarly, reading a texel from the first
location of the texture image will return the texel associated with the
upper-left corner. However, the graphics pipe simulates OpenGL and therefore 
generates device coordinates with the bottom-left corner as the origin. All of 
this means that it is your responsibility to map the OpenGL device (or viewport 
or window) coordinates from the bottom-left corner to upper-left corner.
*/
{
	int width{ static_cast<int>(dev->GetWidth()) }, height{ static_cast<int>(dev->GetHeight()) };
	unsigned int* frame_buffer{ dev->GetFrameBuffer() };

	float signed_area{ triangle_area(v0, v1, v2) };

	if (signed_area <= 0.f)
	{
		return;
	}

	gfxVector3 Edge0{ calculate_edge(v1, v2) },
		Edge1{ calculate_edge(v2, v0) },
		Edge2{ calculate_edge(v0, v1) };

	int x_min, x_max, y_min, y_max;
	calculate_bounding_box(v0, v1, v2, width, height, x_min, x_max, y_min, y_max);

	bool Edge0_tl{ is_top_left(Edge0) },
		Edge1_tl{ is_top_left(Edge1) },
		Edge2_tl{ is_top_left(Edge2) };

	// start values for vertical spans
	float Eval0{ initial_evaluation(Edge0, x_min, y_min) },
		Eval1{ initial_evaluation(Edge1, x_min, y_min) },
		Eval2{ initial_evaluation(Edge2, x_min, y_min) };

	// depth buffer
	float* depth_buffer{ dev->GetDepthBuffer() };

	for (int y{ y_min }; y < y_max; ++y)
	{
		// start values for horizontal spans
		float hEval0{ Eval0 }, hEval1{ Eval1 }, hEval2{ Eval2 };

		for (int x{ x_min }; x < x_max; ++x)
		{
			// centre of current fragment is inside triangle or on a top-left edge
			if ((hEval0 > 0.f || (hEval0 == 0.f && Edge0_tl)) &&
				(hEval1 > 0.f || (hEval1 == 0.f && Edge1_tl)) &&
				(hEval2 > 0.f || (hEval2 == 0.f && Edge2_tl)))
			{
				gfxVertex current_frag;
				current_frag.x_d = static_cast<float>(x);
				current_frag.y_d = static_cast<float>(y);

				// only compute barycentric coordinates if centre of current pixel is inside triangle or on top-left edges
				float w0{ triangle_area(v1, v2, gfxVector3(x + 0.5f, y + 0.5f, 1.f)) / signed_area },
					w1{ triangle_area(v2, v0, gfxVector3(x + 0.5f, y + 0.5f, 1.f)) / signed_area },
					w2{ 1.f - w0 - w1 };

				float depth{ w0 * v0.z_d + w1 * v1.z_d + w2 * v2.z_d };

				// depth of current frag is greater than corresponding depth stored in depth buffer
				if (dev->DepthTestEnabled() && depth > depth_buffer[y * width + x])
				{
					// incrementally update hEvals
					hEval0 += Edge0.x;
					hEval1 += Edge1.x;
					hEval2 += Edge2.x;
					continue;
				}

				// update depth buffer
				depth_buffer[y * width + x] = depth;

				// triangle is textured mapped
				if (dev->GetCurrentTexture())
				{
					int texture_width{ static_cast<int>(dev->GetCurrentTextureWidth()) },
						texture_height{ static_cast<int>(dev->GetCurrentTextureHeight()) }, 
						texel_s{}, texel_t{};

					// interpolate texture coords
					float interpolated_texture_s{ w0 * v0.s + w1 * v1.s + w2 * v2.s },
						interpolated_texture_t{ w0 * v0.t + w1 * v1.t + w2 * v2.t };

					texel_s = static_cast<int>((interpolated_texture_s - std::floor(interpolated_texture_s)) * texture_width);
					texel_t = static_cast<int>((interpolated_texture_t - std::floor(interpolated_texture_t)) * texture_height);

					// calculate index of texel mapped to
					int index{ texel_t * texture_width + texel_s };

					// update framebuffer
					unsigned int clr{ dev->GetCurrentTexture()[index] };
					frame_buffer[(height - 1 - y) * width + x] = clr;
				}
				// triangle is lit
				else
				{
					current_frag.r = w0 * v0.r + w1 * v1.r + w2 * v2.r;
					current_frag.g = w0 * v0.g + w1 * v1.g + w2 * v2.g;
					current_frag.b = w0 * v0.b + w1 * v1.b + w2 * v2.b;

					DrawPoint(dev, current_frag);
				}
			}

			// incrementally update hEvals
			hEval0 += Edge0.x;
			hEval1 += Edge1.x;
			hEval2 += Edge2.x;
		}

		// incrementally update Evals
		Eval0 += Edge0.y;
		Eval1 += Edge1.y;
		Eval2 += Edge2.y;
	}
}


/*  _________________________________________________________________________ */
void YourRasterizer::
DrawPoint(gfxGraphicsPipe *dev, gfxVertex const&	v0)
/*! Render a device frame point.

	@param dev -->  Pointer to the pipe to render to.
	@param v0  -->  Vertex with device frame coordinates previously computed.

	The framebuffer is a linear array of 32-bit pixels.
	fb[y * width + x] will access a pixel at (x, y). The
	pixel format is xxRRGGBB (thus the shifting that's
	going on to pack the color components into a single
	32-bit pixel value.
*/
{
	int x{ static_cast<int>(v0.x_d) }, y{ static_cast<int>(v0.y_d) };
	size_t width{ dev->GetWidth() }, height{ dev->GetHeight() };

	if (x < 0 || x >= width || y < 0 || y >= height)
	{
		return;
	}

	unsigned int color
	{
		(static_cast<unsigned int>(v0.r * 255) << 16) + 
		(static_cast<unsigned int>(v0.g * 255) << 8) + 
		(static_cast<unsigned int>(v0.b * 255))
	};
	dev->GetFrameBuffer()[(height - 1 - y) * width + x] = color;
}

/*  _________________________________________________________________________ */
void YourRasterizer::
DrawLine(gfxGraphicsPipe *dev, gfxVertex const&	v0, gfxVertex const&	v1)
/*! Render a line segment between 2 device frame points.

    @param dev -->  Pointer to the pipe to render to.
    @param v0  -->  First vertex.
    @param v1  -->  Second vertex.
*/
{
	int dx = static_cast<int>(v1.x_d) - static_cast<int>(v0.x_d), 
		dy = static_cast<int>(v1.y_d) - static_cast<int>(v0.y_d);
	gfxVertex v;
	v.x_d = v0.x_d;
	v.y_d = v0.y_d;
	// octant 0
	if (dx > 0 && dy > 0 && abs(dx) >= abs(dy))
	{
		int d = 2 * dy - dx, de = 2 * dy, dne = 2 * dy - 2 * dx;
		DrawPoint(dev, v);
		while (--dx)
		{
			if (d > 0)
			{
				d += dne;
				++v.y_d;
			}
			else
			{
				d += de;
			}
			++v.x_d;
			DrawPoint(dev, v);
		}
	}
	// octant 1
	else if (dx > 0 && dy > 0 && abs(dx) < abs(dy))
	{
		int d = 2 * dx - dy, dn = 2 * dx, dne = 2 * dx - 2 * dy;
		DrawPoint(dev, v);
		while (--dy)
		{
			if (d > 0)
			{
				d += dne;
				++v.x_d;
			}
			else
			{
				d += dn;
			}
			++v.y_d;
			DrawPoint(dev, v);
		}
	}
	// octant 2
	else if (dx < 0 && dy > 0 && abs(dx) < abs(dy))
	{
		int d = -2 * dx - dy, dn = -2 * dx, dnw = -2 * dx - 2 * dy;
		DrawPoint(dev, v);
		while (--dy)
		{
			if (d > 0)
			{
				d += dnw;
				--v.x_d;
			}
			else
			{
				d += dn;
			}
			++v.y_d;
			DrawPoint(dev, v);
		}
	}
	// octant 3
	else if (dx < 0 && dy > 0 && abs(dx) >= abs(dy))
	{
		int d = -2 * dy - dx, dw = -2 * dy, dnw = -2 * dy - 2 * dx;
		DrawPoint(dev, v);
		while (++dx)
		{
			if (d < 0)
			{
				d += dnw;
				++v.y_d;
			}
			else
			{
				d += dw;
			}
			--v.x_d;
			DrawPoint(dev, v);
		}
	}
	// octant 4
	else if (dx < 0 && dy < 0 && abs(dx) >= abs(dy))
	{
		int d = 2 * dy - dx, dw = 2 * dy, dsw = 2 * dy - 2 * dx;
		DrawPoint(dev, v);
		while (++dx)
		{
			if (d < 0)
			{
				d += dsw;
				--v.y_d;
			}
			else
			{
				d += dw;
			}
			--v.x_d;
			DrawPoint(dev, v);
		}
	}
	// octant 5
	else if (dx < 0 && dy < 0 && abs(dx) < abs(dy))
	{
		int d = 2 * dx - dy, ds = 2 * dx, dsw = 2 * dx - 2 * dy;
		DrawPoint(dev, v);
		while (++dy)
		{
			if (d < 0)
			{
				d += dsw;
				--v.x_d;
			}
			else
			{
				d += ds;
			}
			--v.y_d;
			DrawPoint(dev, v);
		}
	}
	// octant 6
	else if (dx > 0 && dy < 0 && abs(dx) < abs(dy))
	{
		int d = -2 * dx - dy, ds = -2 * dx, dse = -2 * dx - 2 * dy;
		DrawPoint(dev, v);
		while (++dy)
		{
			if (d < 0)
			{
				d += dse;
				++v.x_d;
			}
			else
			{
				d += ds;
			}
			--v.y_d;
			DrawPoint(dev, v);
		}
	}
	// octant 7
	else if (dx > 0 && dy < 0 && abs(dx) >= abs(dy))
	{
		int d = -2 * dy - dx, de = -2 * dy, dse = -2 * dy - 2 * dx;
		DrawPoint(dev, v);
		while (--dx)
		{
			if (d > 0)
			{
				d += dse;
				--v.y_d;
			}
			else
			{
				d += de;
			}
			++v.x_d;
			DrawPoint(dev, v);
		}
	}
	// vertical lines going up
	else if (dx == 0 && dy > 0)
	{
		DrawPoint(dev, v);
		while (--dy)
		{
			++v.y_d;
			DrawPoint(dev, v);
		}
	}
	// vertical lines going down
	else if (dx == 0 && dy < 0)
	{
		DrawPoint(dev, v);
		while (++dy)
		{
			--v.y_d;
			DrawPoint(dev, v);
		}
	}
	// horizontal lines going left
	else if (dx < 0 && dy == 0)
	{
		DrawPoint(dev, v);
		while (++dx)
		{
			--v.x_d;
			DrawPoint(dev, v);
		}
	}
	// horizontal lines going right
	else if (dx > 0 && dy == 0)
	{
		DrawPoint(dev, v);
		while (--dx)
		{
			++v.x_d;
			DrawPoint(dev, v);
		}
	}
}

/*  _________________________________________________________________________ */
void YourRasterizer::
DrawWireframe(gfxGraphicsPipe* dev,
              gfxVertex const& v0,
              gfxVertex const& v1,
              gfxVertex const& v2)
/*! Render a triangle in wireframe mode using 3 device-frame points.

    @param dev -->  Pointer to the pipe to render to.
    @param v0  -->  First vertex.
    @param v1  -->  Second vertex.
    @param v2  -->  Third vertex.
*/
{
	float signed_area{ triangle_area(v0, v1, v2) };

	if (signed_area <= 0.f)
	{
		return;
	}

	DrawLine(dev, v0, v1);
	DrawLine(dev, v1, v2);
	DrawLine(dev, v2, v0);
}
