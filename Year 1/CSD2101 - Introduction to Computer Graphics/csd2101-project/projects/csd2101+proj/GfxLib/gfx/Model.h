/*!
@file    Model.h
@author  Prasanna Ghali       (pghali@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_MODEL_H_
#define GFX_MODEL_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"
#include "GraphicsPipe.h"
#include "Vertex.h"
#include "Sphere.h"
#include "Matrix4.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

// forward declarations
class gfxGraphicsPipe;


/*  _________________________________________________________________________ */
class gfxModel
/*! Mesh object.
*/
{ 
  public:
      // ct and dt
     gfxModel();
     gfxModel(unsigned int numFaces, unsigned int numVerts, gfxPrimitive primitive);
     gfxModel(const gfxVertexBuffer& vb, gfxPrimitive primitive);
     gfxModel(const gfxIndexBuffer& ib, const gfxVertexBuffer& vb, gfxPrimitive primitive);
     gfxModel(const gfxModel& m);
    ~gfxModel();
    
    // operator=: assignment
    gfxModel& operator=(const gfxModel& m);
    
		// accessors
		gfxPrimitive	SetPrimitiveType(gfxPrimitive t)	{ mPrimitiveType	=	t; }
 		gfxPrimitive	GetPrimitiveType() const					{ return mPrimitiveType; }

		// buffer accessors
          gfxIndexBuffer&		GetIndexBuffer()        { return (mIdxBuffer); }
    const gfxIndexBuffer&		GetIndexBuffer() const  { return (mIdxBuffer); }
          gfxVertexBuffer&	GetVertexBuffer()       { return (mVtxBuffer); }
    const gfxVertexBuffer&	GetVertexBuffer() const { return (mVtxBuffer); }
          unsigned int*			GetTexture()            { return (mTexture); }
    const unsigned int*			GetTexture() const      { return (mTexture); }
    
    // texture
    void          SetTexture(unsigned int*	data, unsigned int w, unsigned int h);
    unsigned int  GetTextureWidth() const  { return (mTexW); }
    unsigned int  GetTextureHeight() const { return (mTexH); }

    // drawing
    //void Draw(gfxGraphicsPipe* dev, const gfxSphere* bvP = 0);
    void Draw(gfxGraphicsPipe*);
   
    // bounds
		gfxSphere GetModelBVSphere() const	{ return (mModelBVSphere); }
	private:
    static gfxSphere ComputeModelBVSphere(const std::vector< gfxVector3 > &verts);
    
  private:
    // data members
    gfxIndexBuffer  mIdxBuffer;			//!< Model index buffer.
    gfxVertexBuffer mVtxBuffer;			//!< Model vertex buffer.
    
    unsigned int*		mTexture;				//!< Texture buffer.
    unsigned int    mTexW;					//!< Texture width.
    unsigned int    mTexH;					//!< Texture height.

		gfxPrimitive		mPrimitiveType;		//!< Primtive type
    
    gfxSphere				mModelBVSphere;	//!< Model bounding sphere (model frame).
};

using gfxModelList = std::vector<gfxModel>;


#endif  /* GFX_MODEL_H_ */