/*!
@file    GraphicsPipe.inl
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: GraphicsPipe.inl,v 1.1 2005/02/22 04:28:29 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                  functions
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
HWND gfxGraphicsPipe::
GetHWND() const
/*! Get the pipe's window handle.

    @return
    The HWND of the pipe's window.
*/
{
  return (mWindow);
}

/*  _________________________________________________________________________ */
HDC gfxGraphicsPipe::
GetHDC() const
/*! Get the pipe's frame buffer DC.

    @return
    The HDC of the pipe's frame buffer.
*/
{
  return (mBitmapDC);
}

/*  _________________________________________________________________________ */
size_t gfxGraphicsPipe::
GetWidth() const
/*! Get the pipe's window width.

    @return
    The width, in pixels, of the pipe window.
*/
{
  return (mWindW);
}

/*  _________________________________________________________________________ */
size_t gfxGraphicsPipe::
GetHeight() const
/*! Get the pipe's window height.

    @return
    The height, in pixels, of the pipe window.
*/
{
  return (mWindH);
}

/*  _________________________________________________________________________ */
size_t gfxGraphicsPipe::
GetCullCount() const
/*! Get the pipe's cull count.

    @return
    The cull count.
*/
{
  return (mCullCount);
}

/*  _________________________________________________________________________ */
unsigned int* gfxGraphicsPipe::
GetFrameBuffer()
/*! Get the frame buffer pointer.

    @return
    A pointer to the frame buffer.
*/
{
  return (mSurface);
}

void gfxGraphicsPipe::
SetMatrixMode(gfxMatrixMode m)
/*! Set the pipe's matrix stack for subsequent matrix operations.
		Two values are accepted: GFX_MODELVIEW or GFX_PROJECTION
    @param m -->  GFX_MODELVIEW or GFX_PROJECTION
*/
{
  mMatrixMode = m;
}


/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
SetPrimitiveType(gfxPrimitive p)
/*! Set the pipe's current render primitive type.

    @param p -->  The new render primitive type.
*/
{
  mPrimType = p;
}


/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
SetRenderMode(gfxRenderMode m)
/*! Set the pipe's rendering mode.

    @param m -->  The new rendering mode.
*/
{
  mDrawMode = m;
}

/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
SetShadeMode(gfxShadeMode s)
/*! Set the pipe's shading mode.

    @param s -->  The new shading mode.
*/
{
  mShadeMode = s;
}

/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
SetRenderColor(float r, float g, float b, float a)
/*! Set the pipe's render mode.

    @param r -->  Red component of color in range [0.0, 1.0].
    @param g -->  Green component of color in range [0.0, 1.0].
    @param b -->  Blue component of color in range [0.0, 1.0].
    @param a -->  Alpha component of color in range [0.0, 1.0].
*/
{
  mRenderColor.x = r;
  mRenderColor.y = g;
  mRenderColor.z = b;
  mRenderColor.w = a;
}

/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
SetRenderColor(gfxVector4 clr) 
/*! Set the pipe's render mode.

    @param clr -->  (r, g, b, a) components each in range [0.0, 1.0].
*/
{
  mRenderColor = clr;
}

/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
SetInterpMode(gfxInterpMode h)
/*! Set the interpolation mode to be used in triangle rasterizer.

    @param h -->  One of GFX_LINEAR_INTERP or GFX_HYPERBOLIC_INTERP
*/
{
  mInterpMode = h;
}

/*  _________________________________________________________________________ */
gfxInterpMode gfxGraphicsPipe::
GetInterpMode()
/*! Return the interpolation mode to be used in triangle rasterizer.

    @return -->  One of GFX_LINEAR_INTERP or GFX_HYPERBOLIC_INTERP
*/
{
  return mInterpMode;
}


/*  _________________________________________________________________________ */
gfxMatrixMode gfxGraphicsPipe::
GetMatrixMode()
/*! Return the pipe's current matrix stack.
		Two possible values are: GFX_MODELVIEW and GFX_PROJECTION
    @return -->  GFX_MODELVIEW or GFX_PROJECTION
*/
{
  return (mMatrixMode);
}

/*  _________________________________________________________________________ */
gfxPrimitive gfxGraphicsPipe::
GetPrimitiveType()
/*! Get the pipe's current render primitive type.

    @return The current render primitive type.
*/
{
  return (mPrimType);
}

/*  _________________________________________________________________________ */
gfxVector4 gfxGraphicsPipe::
GetRenderColor()
/*! Set the pipe's rendering color when flat shading is specified

    @param clr -->  The new rendering color.
*/
{
  return (mRenderColor);
}
  
/*  _________________________________________________________________________ */
float* gfxGraphicsPipe::
GetDepthBuffer()
/*! Get the depth buffer pointer.

    @return
    A pointer to the depth buffer.
*/
{
  return (mDepthBuffer);
}

/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
UseTransformer(gfxController_Transformation* c)
/*! Change the transformation controller used by the pipe.

    @param c -->  Pointer to the new controller. If 0, the pipe will use the
                  internal controller.
*/
{
  if(c == 0)
    mControllers.transformer = nGetInternalTransformer();
  else
    mControllers.transformer = c;
}

/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
UseRasterizer(gfxController_Rasterization* c)
/*! Change the rasterization controller used by the pipe.

    @param c -->  Pointer to the new controller. If 0, the pipe will use the
                  internal controller.
*/
{
  if(c == 0)
    mControllers.rasterizer = nGetInternalRasterizer();
  else
    mControllers.rasterizer = c;
}

/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
UseLighter(gfxController_Lighting* c)
/*! Change the lighting controller used by the pipe.

    @param c -->  Pointer to the new controller. If 0, the pipe will use the
                  internal controller.
*/
{
  if(c == 0)
    mControllers.lighter = nGetInternalLighter();
  else
    mControllers.lighter = c;
}

/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
UseClipper(gfxController_Clipping* c)
/*! Change the clipping controller used by the pipe.

    @param c -->  Pointer to the new controller. If 0, the pipe will use the
                  internal controller.
*/
{
  if(c == 0)
    mControllers.clipper = nGetInternalClipper();
  else
    mControllers.clipper = c;
}

/*  _________________________________________________________________________ */
void gfxGraphicsPipe::
UsePicker(gfxController_Picking* c)
/*! Change the picking controller used by the pipe.

    @param c -->  Pointer to the new controller. If 0, the pipe will use the
                  internal controller.
*/
{
  if(c == 0)
    mControllers.picker = nGetInternalPicker();
  else
    mControllers.picker = c;
}

/*  _________________________________________________________________________ */
gfxController_Clipping* gfxGraphicsPipe::
GetClipper() const
/*! Get the current clipper.
*/
{
  return (mControllers.clipper);
}

/*  _________________________________________________________________________ */
gfxController_Picking* gfxGraphicsPipe::
GetPicker() const
/*! Get the pickter.
*/
{
  return (mControllers.picker);
}

