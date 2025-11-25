/*!
@file    GraphicsPipe.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: GraphicsPipe.h,v 1.16 2005/02/22 04:10:51 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_GRAPHICSPIPE_H_
#define GFX_GRAPHICSPIPE_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

#include "Vector3.h"
#include "Matrix4.h"
#include "Sphere.h"
#include "Frustum.h"

/*                                                                   typedefs
----------------------------------------------------------------------------- */

// event handler
typedef LRESULT (CALLBACK *gfxEventHandler)(HWND wind,UINT msg,WPARAM wp,LPARAM lp);

// matrix stack
using gfxMatrixStack = std::vector<gfxMatrix4>;

// material properties for diffuse light per color component
struct gfxMaterial
{
	float kr;
	float kg;
	float kb;
};

// primitives
enum gfxPrimitive
{
	GFX_POINT,
	GFX_LINE,
	GFX_TRIANGLE,
	GFX_INDEXED_TRIANGLE
};

// draw modes
enum gfxRenderMode
//! Constants for rendering mode.
{
  GFX_POINTS,     //!< Render points.
  GFX_WIREFRAME,  //!< Render wireframe triangles.
  GFX_FILLED      //!< Render filled triangles.
};

// shading modes
enum gfxShadeMode
//! Constants for shading mode
{
	GFX_FLAT,
	GFX_SMOOTH
};

// interpolation modes
enum gfxInterpMode
//! Constants for shading mode
{
  GFX_LINEAR_INTERP,
  GFX_HYPERBOLIC_INTERP
};

// matrix stacks
enum gfxMatrixMode
{
	GFX_MODELVIEW		= 0,
	GFX_PROJECTION		= 1,
	GFX_MATRIXSTACKS
};

// clip code
enum gfxClipCode
{
  GFX_CCLEFT   = 1,
  GFX_CCRIGHT  = 2,
  GFX_CCBOTTOM = 4,
  GFX_CCTOP    = 8,
  GFX_CCNEAR   = 16,
  GFX_CCFAR    = 32
};

// clip plane
enum gfxClipPlane
{
  GFX_CPLEFT   = 0,
  GFX_CPRIGHT  = 1,
  GFX_CPBOTTOM = 2,
  GFX_CPTOP    = 3,
  GFX_CPNEAR   = 4,
  GFX_CPFAR    = 5
};


/*                                                                    classes
----------------------------------------------------------------------------- */

// forward declarations
class gfxController_Transformation;
class gfxController_Rasterization;
class gfxController_Lighting;
class gfxController_Clipping;
class gfxController_Picking;

/*  _________________________________________________________________________ */
class gfxGraphicsPipe
/*! A graphics pipe is an instance of a software renderer.

    A pipe encapsulates the underlying windowing system, as well as render
    state options and other related properties.
    
    A pipe has a set of so-called "controller" classes which represent the
    stages of the graphics pipeline (transformation, lighting, clipping, and
    so on). By default, a pipe will use internal implementations for each
    controller. However, it is possible to instruct the pipe to use custom
    controllers for each stage of the pipeline.
*/
{
  public:
    // structs
    struct Controllers
    //! Encapsulates the controllers used by the pipe.
    {
      gfxController_Transformation *transformer;  //!< Transformation controller.
      gfxController_Rasterization  *rasterizer;   //!< Rasterization controller.
      gfxController_Lighting       *lighter;      //!< Lighting controller.
      gfxController_Clipping       *clipper;      //!< Clipping controller.
      gfxController_Picking        *picker;       //!< Picking controller.
    };
    
    // ct and dt
     gfxGraphicsPipe(int x, int y, size_t w, size_t h, const std::string& name);
    ~gfxGraphicsPipe();
    
	// accessors
  inline HWND   GetHWND() const;
  inline HDC    GetHDC() const;
  inline size_t GetWidth() const;
  inline size_t GetHeight() const;
  inline size_t GetCullCount() const;
  
	// Set graphics pipe states
	inline void				SetMatrixMode(gfxMatrixMode m);		// Model-view or projection matrix stacks
	inline void				SetPrimitiveType(gfxPrimitive p);	// Primitive type to be rendered: Point, line, triangle, ...
	inline void				SetRenderMode(gfxRenderMode m);		// render mode
	inline void				SetShadeMode(gfxShadeMode s);		// shade mode
  inline void       SetInterpMode(gfxInterpMode h);   // interpolation mode: linear or hyperbolic
  inline void       SetRenderColor(float r, float g, float b, float a);
  inline void       SetRenderColor(gfxVector4 clr);
	inline void				SetAmbientMat(float r, float g, float b){ mAmbMat.kr = r; mAmbMat.kg = g; mDiffMat.kb = b; }
	inline void				SetAmbientMat(const gfxMaterial& am)	{ mAmbMat = am; }
	inline void				SetDiffuseMat(float r, float g, float b){ mDiffMat.kr = r; mDiffMat.kg = g; mDiffMat.kb = b; }
	inline void				SetDiffuseMat(const gfxMaterial& dm)	{ mDiffMat = dm; }

	inline void				EnableLighting(bool b)					{	mLighting = b; }
	inline void				EnableDepthTest(bool b)					{ mDepth = b; }
	inline void				EnableShadows(bool b)					{ mShadows = b; }

	// Retrieve graphics pipe states
	inline gfxMatrixMode	GetMatrixMode();
	inline gfxPrimitive		GetPrimitiveType();
  inline gfxVector4     GetRenderColor();
  inline gfxInterpMode  GetInterpMode();   // interpolation mode: linear or hyperbolic
  inline gfxPlane				GetShadowPlane() const		{ return (mShadowPlane); }
	inline gfxVector3			GetLightPos() const			{ return (gfxVector3(mLight.x,mLight.y,mLight.z)); }
	inline gfxMaterial		GetAmbientMat() const		{ return mAmbMat; }
	inline gfxMaterial		GetDiffuseMat() const		{ return mDiffMat; }   

	inline bool						LightingEnabled() const		{ return (mLighting); }
	inline bool						DepthTestEnabled() const	{ return (mDepth); }
	inline bool						ShadowsEnabled() const		{ return (mShadows); }

  // buffer rendering
	void	DrawPoints(gfxVertexBuffer*);
	void	DrawLines(gfxVertexBuffer*);
	void	DrawTriangles(gfxVertexBuffer*);
	void	DrawIndexedTriangles(gfxVertexBuffer*, const gfxIndexBuffer*, const gfxSphere* = 0);

  // text rendering
  void	DrawText(int, int, const std::string&); 
  
  // Matrix stack manipulation
	void	PushMatrix();                    
  void	PopMatrix();                  
  void	LoadIdentity();
  void	LoadMatrix(const gfxMatrix4&);    
  void	MultMatrix(const gfxMatrix4&); 
	void	Translate(float, float, float);
	void	Scale(float, float, float);
	void	Rotate(float, float, float, float);
	void	LookAt(	float, float, float,
								float, float, float,
								float,	float,	float);
	void	LookAt(	const gfxVector3&,
								const gfxVector3&,
								const gfxVector3&);
	void	Perspective(float, float, float, float);
	void	Frustum(float, float, float, float, float, float);
	void	Ortho(float, float, float, float, float, float);
	void	SetLightPos(float, float, float);
	void	SetLightPos(const gfxVector3&);
	void	SetShadowPlane(float, float, float, float, const gfxVector3&);
	void	SetShadowPlane(const gfxPlane&, const gfxVector3&);

	void SetViewportMatrix(float x, float y, float w, float h);	// NDC-to-viewport transform

  // rendering
  void RenderBegin();
  void RenderEnd();
  
  // buffer pointers
  inline unsigned int* GetFrameBuffer();
  inline float*        GetDepthBuffer();
  
  // event handlers
  void AssignEventHandler(UINT msg, gfxEventHandler h);
  void RemoveEventHandler(UINT msg);
  
  // controllers
  inline void UseTransformer(gfxController_Transformation* c);
  inline void UseRasterizer(gfxController_Rasterization* c);
  inline void UseLighter(gfxController_Lighting* c);
  inline void UseClipper(gfxController_Clipping* c);
  inline void UsePicker(gfxController_Picking* c);
  
  // texture
  void				SetCurrentTexture(unsigned int*	tex,
																unsigned int	w,
																unsigned int	h) { mCurTexPtr = tex; mCurTexW = w; mCurTexH = h; }
  unsigned int* GetCurrentTexture() const									{ return (mCurTexPtr); }
  unsigned int  GetCurrentTextureWidth() const						{ return (mCurTexW); }
  unsigned int  GetCurrentTextureHeight() const					  { return (mCurTexH); }
  
  // clipping
  inline gfxController_Clipping* GetClipper() const;
  // picking
  inline gfxController_Picking* GetPicker() const; // picking
  
  // shadows
  gfxController_Lighting* GetLighter() const				 { return (mControllers.lighter); }

	// view-frame frustum plane equations
	const gfxFrustum& GetFrustum() const							 { return (mFrustum); }
	void							SetFrustum(const gfxFrustum& f)	 { mFrustum = f; }

  private:
    // disabled
    gfxGraphicsPipe(const gfxGraphicsPipe& d);
    gfxGraphicsPipe& operator=(const gfxGraphicsPipe& d) = delete;
  
    // typedefs
    using EventHandlerMap     = std::map<UINT,gfxEventHandler>;
    using EventHandlerMapIter = EventHandlerMap::iterator;
    
    // setup helpers
    void nSetupWindow();
    void nSetupBitmap();
    
    // buffer rendering
		void nRenderPointBuffer(	gfxVertexBuffer*	vb);
		void nRenderLineBuffer(	gfxVertexBuffer*	vb);
    void nRenderTriangleBuffer(	gfxVertexBuffer*			vb,
																const gfxIndexBuffer*	ib = 0,
																const gfxSphere*			bs = 0);
    
    // internal controllers
    gfxController_Rasterization*  nGetInternalRasterizer();
    gfxController_Transformation* nGetInternalTransformer();
    gfxController_Lighting*       nGetInternalLighter();
    gfxController_Clipping*       nGetInternalClipper();
    gfxController_Picking*        nGetInternalPicker();
    
    // message handler
    static LRESULT CALLBACK nWindowProc(HWND wind, UINT msg, WPARAM wp, LPARAM lp);
    
    // data members
    HWND						mWindow;					//!< Window handle.
    std::string			mWindName;				//!< Window name.
    int							mWindX;						//!< Window X coordinate.
    int							mWindY;						//!< Window Y coordinate.
    size_t					mWindW;						//!< Window width.
    size_t					mWindH;						//!< Window height.
    HDC							mDC;							//!< Window device context.

    HBITMAP					mBitmap;					//!< Current frame buffer bitmap.
    HBITMAP					mBitmapOld;				//!< Old frame buffer bitmap.
    HDC							mBitmapDC;				//!< Frame buffer device.
    
    int							mBufferSz;				//!< Size (in elements) of frame and auxiliary buffers.
    unsigned int*		mSurface;					//!< Frame buffer surface.
    float*					mDepthBuffer;			//!< Depth buffer.
    
    size_t					mCullCount;				//!< Count of objects culled last render pass.
    
    gfxPlane				mShadowPlane;			//!< Plane equation of shadow receiver. Always specified in the
																			//!< frame defined by the current transformation matrix of the 
																			//!< active matrix stack.
    gfxVector3			mLight;						//!< Pipe's light position.
		gfxFrustum			mFrustum;					//!< Equations of six frustum planes in clip frame.
    
		gfxMaterial			mDiffMat;					//!< Diffuse color of primitive material
		gfxMaterial			mAmbMat;					//!< Background ambient color of primitive material
 
		EventHandlerMap	mHandlers;				//!< Event delegates.
    Controllers			mControllers;			//!< Controller delegates.
    
		
		gfxMatrixStack	mMatrixStack[GFX_MATRIXSTACKS];	//!< Matrix stacks for model-to-view frame and projection transforms.
		gfxMatrix4			mViewportMtx;			//!< Matrix manifestation of NDC-to-viewport transform
		gfxMatrixMode		mMatrixMode;			//!< Current active matrix stack. Can be either of the 2 enumeration constants:
																			//! GFX_MODELVIEW or GFX_PROJECTION.
    
    unsigned int*		mCurTexPtr;				//!< Pointer to 2D tex map data
    unsigned int		mCurTexW;					//!< Width and height of 2D tex map
    unsigned int		mCurTexH;
    
		bool						mLighting;				//!< If true, pipe will "light" incoming vertices; otherwise not.
    bool						mShadows;					//!< If true, pipe will "shadow" occluders onto receiver plane; otherwise not.
    bool						mDepth;						//!< If true, pipe will "depth buffer" pixels; otherwise not.
    
		gfxPrimitive		mPrimType;				//!< Current render primitive type.
    gfxRenderMode		mDrawMode;				//!< Current rendering mode.
		gfxShadeMode		mShadeMode;				//!< Current shading mode.
    gfxInterpMode   mInterpMode;      //!< Current triangle interpolation mode.
    gfxVector4      mRenderColor;     //!< Current primary color of graphics pipe.
    gfxVector4      mVertexColor;     //!< Current vertex color.
    gfxVector3			mVertexNormal;		//!< Current vertex normal.
    gfxVertexBuffer	mVertexStream;		//!< VB used by immediate mode commands.
};


/*                                                             implementation
----------------------------------------------------------------------------- */

#include "GraphicsPipe.inl"


#endif  /* GFX_GRAPHICSPIPE_H_ */
