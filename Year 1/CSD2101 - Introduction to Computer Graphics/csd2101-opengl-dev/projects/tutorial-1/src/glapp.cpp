/*!
@file       glapp.cpp
@author  	pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date    	12/05/2024

This file implements functionality useful and necessary to build OpenGL
applications including use of external APIs such as GLFW to create a
window and start up an OpenGL context and to extract function pointers
to OpenGL implementations.

*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glapp.h>
#include <iostream>

GLApp::GLModel GLApp::mdl;

/*                                                   objects with file scope
----------------------------------------------------------------------------- */
const std::string my_tutorial_2_vs = {
  #include "my-tutorial-1.vert"
};

const std::string my_tutorial_2_fs = {
  #include "my-tutorial-1.frag"
};

/*  _________________________________________________________________________*/
/*! GLApp::init

This function clear the color buffer with initial RGB value, initialises viewport, setup
rectangle model vertex array object and shader program.
*/
void GLApp::init() {
	// Part 1: clear color buffer with RGBA value in glClearColor ...
	glClearColor(1.f, 0.f, 0.f, 1.f);

	// Part 2: use entire window as viewport ...
	glViewport(0, 0, GLHelper::width, GLHelper::height);

	mdl.setup_vao();
	mdl.setup_shdrpgm();
}

/*  _________________________________________________________________________*/
/*! GLApp::update

This function dynamically compute the background color and colors of each vertex of the
rectangle model when U key state is toggled to true. Otherwise, it toggles the background color
and colors of each vertex to the respective static colors.
*/
void GLApp::update() {
	static GLfloat background_red_val{ 0.f }, background_blue_val{ 1.0f };
	static GLboolean background_red_flag{ GL_TRUE }, background_blue_flag{ GL_FALSE };
	static GLboolean rect_red_flag[4]{ GL_TRUE }, rect_blue_flag[4]{ GL_FALSE };

	std::array<glm::vec3, 4> clr_vtx
	{
		glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f),
		glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f, 1.f, 1.f)
	};

	// U key state is toggled to true
	if (GLHelper::keystateU == GL_TRUE)
	{
		// dynamic background
		if (background_red_val >= 1.0f)
		{
			background_red_flag = GL_FALSE;
		}
		else if (background_red_val <= 0.0f)
		{
			background_red_flag = GL_TRUE;
		}

		if (background_blue_val >= 1.0f)
		{
			background_blue_flag = GL_FALSE;
		}
		else if (background_blue_val <= 0.0f)
		{
			background_blue_flag = GL_TRUE;
		}

		background_red_flag ? background_red_val += static_cast<GLfloat>(GLHelper::delta_time) / 4 : background_red_val -= static_cast<GLfloat>(GLHelper::delta_time) / 4;
		background_blue_flag ? background_blue_val += static_cast<GLfloat>(GLHelper::delta_time) / 4 : background_blue_val -= static_cast<GLfloat>(GLHelper::delta_time) / 4;

		glClearColor(background_red_val, 0.f, background_blue_val, 1.f);


		// rectangle model vertex color interpolation
		const glm::vec3 init_clr{ 0.f, 1.f, 1.f };
		const glm::vec3 final_clr{ 1.f, 1.f, 0.f };
		static std::array<glm::vec3, 4> interpolated_color
		{
			init_clr, glm::vec3(final_clr.r / 3, 1.f, init_clr.b / 3 * 2),
			glm::vec3(final_clr.r / 3 * 2, 1.f, init_clr.b / 3), final_clr
		};
		
		// compute interpolated color per vertex
		for (size_t i{}; i < clr_vtx.size(); ++i)
		{
			// red attribute exceeds 1.0f, start decreasing it
			if (interpolated_color[i].r >= final_clr.r)
			{
				rect_red_flag[i] = GL_FALSE;
			}
			// red attribute below 1.0f, start increasing it
			else if (interpolated_color[i].r <= init_clr.r)
			{
				rect_red_flag[i] = GL_TRUE;
			}
			// blue attribute exceeds 1.0f, start decreasing it
			if (clr_vtx[i].b >= init_clr.b)
			{
				rect_blue_flag[i] = GL_FALSE;
			}
			// blue attribute below 1.0f, start increasing it
			else if (interpolated_color[i].b <= final_clr.b)
			{
				rect_blue_flag[i] = GL_TRUE;
			}
			rect_red_flag[i] ? interpolated_color[i].r += static_cast<GLfloat>(GLHelper::delta_time) : interpolated_color[i].r -= static_cast<GLfloat>(GLHelper::delta_time);
			rect_blue_flag[i] ? interpolated_color[i].b += static_cast<GLfloat>(GLHelper::delta_time) : interpolated_color[i].b -= static_cast<GLfloat>(GLHelper::delta_time);
			clr_vtx[i] = interpolated_color[i];
		}
	}
	else // U key state is toggled to false
	{
		glClearColor(1.f, 0.f, 0.f, 1.f);

		clr_vtx = 
		{
			glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f),
			glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f, 1.f, 1.f)
		};
	}

	// overwrite rectangle VBO
	glNamedBufferSubData(mdl.vbo_hdl, sizeof(glm::vec2) * 4, sizeof(clr_vtx), clr_vtx.data());

	// update window title
	std::string title = "Tutorial 1 | Bryan Ang Wei Ze | " + std::to_string(GLHelper::fps);
	glfwSetWindowTitle(GLHelper::ptr_window, title.c_str());
}

/*  _________________________________________________________________________*/
/*! GLApp::draw

This function renders all objects to the back buffer
*/
void GLApp::draw() {
	glClear(GL_COLOR_BUFFER_BIT);
	mdl.draw();
}

void GLApp::cleanup() {
	// empty for now
}

/*  _________________________________________________________________________*/
/*! GLApp::GLModel::setup_vao

This function transfer rectangle model vertices data from client to server side
by creating vertex buffer and vertex array objects
*/
void GLApp::GLModel::setup_vao()
{
	// vertex position attributes
	std::array<glm::vec2, 4> pos_vtx
	{
		glm::vec2(0.5f, -0.5f), glm::vec2(0.5f, 0.5f),
		glm::vec2(-0.5f, 0.5f), glm::vec2(-0.5f, -0.5f)
	};

	// vertex color attributes
	std::array<glm::vec3, 4> clr_vtx
	{
		glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f),
		glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f, 1.f, 1.f)
	};

	// compute and store values to simplify VBO and VAO management
	GLsizei position_data_offset = 0;
	GLsizei position_attribute_size = sizeof(glm::vec2);
	GLsizei position_data_size = position_attribute_size * static_cast<GLsizei>(pos_vtx.size());

	GLsizei color_data_offset = position_data_size;
	GLsizei color_attribute_size = sizeof(glm::vec3);
	GLsizei color_data_size = color_attribute_size * static_cast<GLsizei>(clr_vtx.size());

	glCreateBuffers(1, &vbo_hdl);

	// nullptr means no data is transferred
	glNamedBufferStorage(vbo_hdl, position_data_size + color_data_size, nullptr, GL_DYNAMIC_STORAGE_BIT);

	/*
	+ position_data_offset			   + color_data_offset
	|								   |
	v								   v
	+-----------------------------------------------------------------------+
	|			 Vertex Data		   |			 Color Data				|
	+----------------------------------+------------------------------------+
			 position_data_size					  color_data_size
	<---------------------------------> <----------------------------------->
	*/

	glNamedBufferSubData(vbo_hdl, position_data_offset, position_data_size, pos_vtx.data());
	glNamedBufferSubData(vbo_hdl, color_data_offset, color_data_size, clr_vtx.data());

	// vaoid is data member 1 of GLApp::GLModel
	glCreateVertexArrays(1, &vaoid);

	// for vertex position array, vertex attribute index is 8
	// and vertex buffer binding point is 3
	glEnableVertexArrayAttrib(vaoid, 8);
	glVertexArrayVertexBuffer(vaoid, 3, vbo_hdl, position_data_offset, position_attribute_size);
	glVertexArrayAttribFormat(vaoid, 8, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 8, 3);

	// for vertex position array, vertex attribute index is 9
	// and vertex buffer binding point is 4
	glEnableVertexArrayAttrib(vaoid, 9);
	glVertexArrayVertexBuffer(vaoid, 4, vbo_hdl, color_data_offset, color_attribute_size);
	glVertexArrayAttribFormat(vaoid, 9, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 9, 4);

	primitive_type = GL_TRIANGLES;

	std::array<GLushort, 6> idx_vtx
	{
		0, 1, 2, // 1st triangle's position and color coordinates are stored
				 // in indices 0, 1, 2 of vertex attribute arrays in VBO
		2, 3, 0	 // 2nd triangle's position and color coordinates are stored
				 // in indices 2, 3, 0 of vertex attribute arrays in VBO
	};

	idx_elem_cnt = static_cast<GLuint>(idx_vtx.size());

	// transfer topology information from CPU to GPU
	GLuint ebo_hdl;
	glCreateBuffers(1, &ebo_hdl);
	glNamedBufferStorage(ebo_hdl, sizeof(GLushort) * idx_elem_cnt, reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(vaoid, ebo_hdl);
	glBindVertexArray(0);
}

/*  _________________________________________________________________________*/
/*! GLApp::setup_shdrpgm

This function compiles the source codes of the vertex and fragment shaders, links
the shader objects to create a shader program for the rectangle model as well as
validating the shader program.
*/
void GLApp::GLModel::setup_shdrpgm()
{
	if (!shdr_pgm.CompileShaderFromString(GL_VERTEX_SHADER, my_tutorial_2_vs))
	{
		std::cout << "Vertex shader failed to compile: ";
		std::cout << shdr_pgm.GetLog() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (!shdr_pgm.CompileShaderFromString(GL_FRAGMENT_SHADER, my_tutorial_2_fs))
	{
		std::cout << "Fragment shader failed to compile: ";
		std::cout << shdr_pgm.GetLog() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (!shdr_pgm.Link())
	{
		std::cout << "Shader program failed to link!" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (!shdr_pgm.Validate())
	{
		std::cout << "Shader program failed to validate!" << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

/*  _________________________________________________________________________*/
/*! GLApp::GLModel::draw

This function renders the rectangle model to the back buffer
*/
void GLApp::GLModel::draw()
{
	// there are many shader programs initialized - here we're saying
	// which specific shader program should be used to render geometry
	shdr_pgm.Use();

	// there are many models, each with their own initialized VAO object
	// here, we're saying which VAO's state should be used to set up pipe
	glBindVertexArray(vaoid);

	// here, we're saying what primitive is to be rendered and how many
	// such primitives exist.
	// the graphics driver knows where to get the indices because the VAO
	// containing this state information has been made current ...
	glDrawElements(primitive_type, idx_elem_cnt, GL_UNSIGNED_SHORT, NULL);

	// after completing the rendering, we tell the driver that VAO
	// vaoid and current shader program are no longer current
	glBindVertexArray(0);
	shdr_pgm.UnUse();
}