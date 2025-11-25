/* !
@file    	glapp.h
@author  	pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date    	12/05/2024

This file contains the declaration of namespace GLApp that encapsulates the
functionality required to implement an OpenGL application including 
compiling, linking, and validating shader programs
setting up geometry and index buffers, 
configuring VAO to present the buffered geometry and index data to
vertex shaders,
configuring textures (in later labs),
configuring cameras (in later labs), 
and transformations (in later labs).
*//*__________________________________________________________________________*/

/*                                                                      guard
----------------------------------------------------------------------------- */
#ifndef GLAPP_H
#define GLAPP_H
#include <glhelper.h>
#include <glslshader.h>

/*                                                                   includes
----------------------------------------------------------------------------- */

struct GLApp {
  static void init();
  static void update();
  static void draw();
  static void cleanup();

  // encapsulates state required to render a geometrical model
  struct GLModel
  {
	  GLenum primitive_type; // which OpenGL primitive to be rendered?
	  GLSLShader shdr_pgm;   // which shader program?
	  GLuint vaoid;			 // handle to VAO
	  GLuint vbo_hdl;		 // handle to VBO
	  GLuint idx_elem_cnt;   // how many elements of primitive_type
							 // are to be rendered?

	  // member functions defined in glapp.cpp
	  void setup_vao();
	  void setup_shdrpgm();
	  void draw();
  };

  // data member to represent geometric model to be rendered
  static GLModel mdl;
};

#endif /* GLAPP_H */
