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
	  GLenum primitive_type;
	  GLuint primitive_cnt;
	  GLuint vaoid;
	  GLuint draw_cnt;
	  GLSLShader shdr_pgm;

	  void setup_shdrpgm(std::string const& vtx_shdr, std::string const& frag_shdr);
	  void draw();
  };

  struct GLViewport
  {
	  GLint x, y;
	  GLsizei width, height;
  };

  // data member to represent geometric model to be rendered
  static GLModel mdl;

  // container for viewports
  static std::vector<GLViewport> vps;

  // container for models
  static std::vector<GLModel> models;

  // setup VAO for GL_POINT primitives
  static GLApp::GLModel points_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frag_shdr);

  // setup VAO for GL_LINES primitives
  static GLApp::GLModel lines_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frag_shdr);

  // setup VAO for GL_TRIANGLE_FAN primitives
  static GLApp::GLModel trifans_model(GLint slices, std::string const& vtx_shdr, std::string const& frag_shdr);

  // setup VAO for GL_TRIANGLE_STRIP primitives
  static GLApp::GLModel tristrip_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frag_shdr);
};

#endif /* GLAPP_H */
