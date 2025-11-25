/*!
@file     Vertex.h
@author  Prasanna Ghali       (pghali@digipen.edu)

CVS: $Id: Vertex.h,v 1.8 2005/02/22 04:10:51 jmp Exp $

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */

#ifndef GFX_VERTEX_H_
#define GFX_VERTEX_H_

/*                                                                   includes
----------------------------------------------------------------------------- */

#include "GFXInternal.h"

/*                                                                    classes
----------------------------------------------------------------------------- */

// outcode
using gfxOutCode = unsigned int;


/*  _________________________________________________________________________ */
class gfxVertex
/*!
*/
{
  public:
    // ct and dt
     gfxVertex(float xi = 0.f, float yi = 0.f, float zi = 0.f);
     gfxVertex(const gfxVertex& vtx);
    ~gfxVertex();
    
    // = operator: copy constructor
    gfxVertex& operator=(const gfxVertex& vtx);
    
    // PRE-TRANSFORM PROPERTIES

    float						x_m;  //!< Model-frame X coordinate.
    float						y_m;  //!< Model-frame Y coordinate.
    float						z_m;  //!< Model-frame Z coordinate.
    float						nx_m; //!< Model-frame normal X component.
    float						ny_m; //!< Model-frame normal Y component.
    float						nz_m; //!< Model-frame normal Z component.
    
    // vertex color
    float		        r;    //!< Vertex red component in range [0.f, 1.f].
    float		        g;    //!< Vertex green component in range [0.f, 1.f].
    float		        b;    //!< Vertex blue component in range [0.f, 1.f].
    float		        a;    //!< Vertex alpha component in range [0.f, 1.f].

    // texture coordinates
    float						s;
    float						t;
    
    // POST-TRANSFORM PROPERTIES

    float						x_v;    //!< View-frame X coordinate.
    float						y_v;    //!< View-frame Y coordinate.
    float						z_v;    //!< View-frame Z coordinate.
    float						nx_v;   //!< View-frame normal X component.
    float						ny_v;   //!< View-frame normal Y component.
    float						nz_v;   //!< View-frame normal Z component.
		
		float						x_c;		//!< Clip-frame X coordinate.
	  float						y_c;		//!< Clip-frame Y coordinate.
	  float						z_c;		//!< Clip-frame Z coordinate.
	  float						w_c;		//!< Clip-frame W coordinate.
    float						i_w;		//!< 1 over w_c

	  float						x_n;		//!< NDC X coordinate 
	  float						y_n;		//!< NDC Y coordinate 
	  float						z_n;		//!< NDC Z coordinate

    // vertex color - using only (r/w, g/w, b/w)
    float		        r_w;    //!< r over w_c.
    float		        g_w;    //!< g over w_c.
    float		        b_w;    //!< b over w_c.
    // NOTE: vertex alpha is not used to reduce computational overhead
    //float		        a_w;    //!< Vertex alpha component.
    
    // texture coordinates for hyperbolic interpolation
    float						s_w;		//!< s over w_c
    float						t_w;		//!< t over w_c

		float						x_d;    //!< Device-frame or viewport X coordinate.
    float						y_d;    //!< Device-frame or viewport Y coordinate.
    float						z_d;    //!< Device-frame or viewport Z coordinate (depth).

    // culling & clipping attributes
	  float						bc[6];  //!< Clip plane boundary conditions.
    gfxOutCode			oc;     //!< Outcode for this vertex.
};

/*                                                                   typedefs
----------------------------------------------------------------------------- */

// buffer types
using gfxVertexBuffer = std::vector<gfxVertex>;    //!< Vertex buffer type.
using gfxIndexBuffer  = std::vector<unsigned int>; //!< Index buffer type.

#endif  /* GFX_VERTEX_H_ */