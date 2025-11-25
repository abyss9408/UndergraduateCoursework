/* !
@file    	glapp.h
@author  	pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date    	21/06/2024

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
#include <iostream>
#include <fstream>
#include <sstream>
#include <glm/gtc/type_ptr.hpp>
#include <hook_csd2101.h>

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
		GLSLShader shdr_pgm;
		GLuint vaoid;
		GLuint vbo_hdl;
		GLuint draw_cnt;

		// member functions defined in glapp.cpp
		void setup_vao();
		void setup_shdrpgm(std::string const& vtx_shdr, std::string const& frg_shdr);
		void draw();
	};

	struct Vertex
	{
		glm::vec2 position;
		glm::vec3 color;
		glm::vec2 texture;
	};

	struct GLViewport
	{
		GLint x, y;
		GLsizei width, height;
	};

	// data member to represent geometric model to be rendered
	static GLModel mdl;

	static GLint task_id;
	static GLfloat tile_size;
};

GLuint setup_texobj(std::string const& pathname);

#endif /* GLAPP_H */
