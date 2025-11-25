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

Bonus task is implemented in GLApp::tristrip_model lines 363-379 and
GLApp::GLModel::draw lines 502-503
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glapp.h>
#include <iostream>
#include <sstream>
#include <iomanip>

GLApp::GLModel GLApp::mdl;
std::vector<GLApp::GLViewport> GLApp::vps;
std::vector<GLApp::GLModel> GLApp::models;
const GLushort restart_index{ 0xFFFF };

/*                                                   objects with file scope
----------------------------------------------------------------------------- */
const std::string my_tutorial_2_vs = {
  #include "my-tutorial-2.vert"
};

const std::string my_tutorial_2_fs = {
  #include "my-tutorial-2.frag"
};

/*  _________________________________________________________________________*/
/*! GLApp::init

This function clear the color buffer with initial RGB value, initialises viewport, setup
rectangle model vertex array object and shader program.
*/
void GLApp::init() {
	// Part 1: clear color buffer with RGBA value in glClearColor ...
	glClearColor(1.f, 1.f, 1.f, 1.f);

	// Part 2: create viewport
	GLint w{ GLHelper::width }, h{ GLHelper::height };
	GLint hs{ h / 3 }, ws{ hs };
	vps.push_back({ 0, 0, w - ws, h });
	vps.push_back({ w - ws, 2 * hs, ws, hs });
	vps.push_back({ w - ws, hs, ws, hs });
	vps.push_back({ w - ws, 0, ws, hs });

	// Part 3: create different geometries and insert them into
	// repository container GLApp::models ...
	GLApp::models.emplace_back(GLApp::points_model(20, 20, my_tutorial_2_vs, my_tutorial_2_fs));
	GLApp::models.emplace_back(GLApp::lines_model(40, 40, my_tutorial_2_vs, my_tutorial_2_fs));
	GLApp::models.emplace_back(GLApp::trifans_model(50, my_tutorial_2_vs, my_tutorial_2_fs));
	GLApp::models.emplace_back(GLApp::tristrip_model(10, 15, my_tutorial_2_vs, my_tutorial_2_fs));
}

/*  _________________________________________________________________________*/
/*! GLApp::update

This function dynamically compute the background color and colors of each vertex of the
rectangle model when U key state is toggled to true. Otherwise, it toggles the background color
and colors of each vertex to the respective static colors.
*/
void GLApp::update() {
	std::stringstream sstr;

	sstr << "Tutorial 2 | Bryan Ang Wei Ze | POINTS : " << models[0].primitive_cnt << ", " << models[0].draw_cnt <<
		" | LINES : " << models[1].primitive_cnt << ", " << models[1].draw_cnt <<
		" | FAN : " << models[2].primitive_cnt << ", " << models[2].draw_cnt <<
		" | STRIP : " << models[3].primitive_cnt << ", " << models[3].draw_cnt <<
		" | " << std::fixed << std::setprecision(2) << GLHelper::fps;
	glfwSetWindowTitle(GLHelper::ptr_window, sstr.str().c_str());

	GLint w{ GLHelper::width }, h{ GLHelper::height };
	static GLint old_w{}, old_h{};
	// update viewport settings in vps only if window's dimension change
	if (w != old_w || h != old_h)
	{
		GLint hs{ h / 3 }, ws{ hs };
		GLApp::vps[0] = { 0,0,w - ws,h };
		GLApp::vps[1] = { w - ws, 2 * hs,ws,hs };
		GLApp::vps[2] = { w - ws,hs,ws,hs };
		GLApp::vps[3] = { w - ws, 0 ,ws,hs };
		old_w = w;
		old_h = h;
	}
}

/*  _________________________________________________________________________*/
/*! GLApp::draw

This function renders all objects to the back buffer
*/
void GLApp::draw() {
	glClear(GL_COLOR_BUFFER_BIT);

	// dynamically switching viewports
	// models[0] = points_model
	// models[1] = lines_model
	// models[2] = trifans_model
	// models[3] = tristrips_model
	if (GLHelper::keystate1)
	{
		for (size_t i{ 0 }; i < vps.size(); ++i)
		{
			glViewport(vps[i].x, vps[i].y, vps[i].width, vps[i].height);
			GLApp::models[i].draw();
		}
	}
	else if (GLHelper::keystate2)
	{
		glViewport(vps[0].x, vps[0].y, vps[0].width, vps[0].height);
		GLApp::models[1].draw();
		glViewport(vps[1].x, vps[1].y, vps[1].width, vps[1].height);
		GLApp::models[0].draw();
		glViewport(vps[2].x, vps[2].y, vps[2].width, vps[2].height);
		GLApp::models[2].draw();
		glViewport(vps[3].x, vps[3].y, vps[3].width, vps[3].height);
		GLApp::models[3].draw();
	}
	else if (GLHelper::keystate3)
	{
		glViewport(vps[0].x, vps[0].y, vps[0].width, vps[0].height);
		GLApp::models[2].draw();
		glViewport(vps[1].x, vps[1].y, vps[1].width, vps[1].height);
		GLApp::models[0].draw();
		glViewport(vps[2].x, vps[2].y, vps[2].width, vps[2].height);
		GLApp::models[1].draw();
		glViewport(vps[3].x, vps[3].y, vps[3].width, vps[3].height);
		GLApp::models[3].draw();
	}
	else if (GLHelper::keystate4)
	{
		glViewport(vps[0].x, vps[0].y, vps[0].width, vps[0].height);
		GLApp::models[3].draw();
		glViewport(vps[1].x, vps[1].y, vps[1].width, vps[1].height);
		GLApp::models[0].draw();
		glViewport(vps[2].x, vps[2].y, vps[2].width, vps[2].height);
		GLApp::models[1].draw();
		glViewport(vps[3].x, vps[3].y, vps[3].width, vps[3].height);
		GLApp::models[2].draw();
	}
}

void GLApp::cleanup() {
	glInvalidateBufferData(mdl.vaoid);
	glDeleteBuffers(1, &mdl.vaoid);
}


GLApp::GLModel GLApp::points_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frag_shdr)
{
	std::vector<glm::vec2> pos_vtx;
	pos_vtx.reserve((slices + 1) * (stacks + 1));

	// j is the stacks(rows) index and i is the slices(columns) index
	for (GLint i{ 0 }; i < stacks + 1; ++i)
	{
		for (GLint j{ 0 }; j < slices + 1; ++j)
		{
			pos_vtx.emplace_back(glm::vec2(
				((1.f - (-1.f)) / slices * j) - 1.0f,
				((1.f - (-1.f)) / stacks * i) - 1.0f));
		}
	}

	// create vertex buffer object
	GLuint vbo_hdl;
	glCreateBuffers(1, &vbo_hdl);
	glNamedBufferStorage(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data(), GL_DYNAMIC_STORAGE_BIT);

	// create vertex array object
	// both attribute and bind indexes are 0
	GLApp::GLModel mdl;
	glCreateVertexArrays(1, &mdl.vaoid);
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 0, vbo_hdl, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 0);
	glBindVertexArray(0);

	// initialize model specs
	mdl.primitive_type = GL_POINTS;
	mdl.setup_shdrpgm(vtx_shdr, frag_shdr);
	mdl.draw_cnt = static_cast<GLuint>(pos_vtx.size());
	mdl.primitive_cnt = mdl.draw_cnt;
	return mdl;
}

GLApp::GLModel GLApp::lines_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frag_shdr)
{
	std::vector<glm::vec2> pos_vtx;
	pos_vtx.reserve((slices + 1) * 2 + (stacks + 1) * 2);
	GLfloat const u{ 2.f / static_cast<GLfloat>(slices) };
	GLfloat const v{ 2.f / static_cast<GLfloat>(stacks) };

	// compute and store endpoints for (slices+1) set of vertical lines
	// for each x from -1 to 1
	// start endpoint is (x, -1) and end endpoint is (x, 1)
	for (GLint col{ 0 }; col <= slices; ++col)
	{
		GLfloat x{ u * static_cast<GLfloat>(col) - 1.f };
		pos_vtx.emplace_back(glm::vec2(x, -1.f)); // bottom end point of line
		pos_vtx.emplace_back(glm::vec2(x, 1.f)); // top end point of line
	}

	// compute and store endpoints for (stacks+1) set of horizontal lines
	// for each y from -1 to 1
	// start endpoint is (-1, y) and end endpoint is (1, y)
	for (GLint row{ 0 }; row <= stacks; ++row)
	{
		GLfloat y{ v * static_cast<GLfloat>(row) - 1.f };
		pos_vtx.emplace_back(glm::vec2(-1.f, y)); // left end point of line
		pos_vtx.emplace_back(glm::vec2(1.f, y)); // right end point of line
	}

	// create vertex buffer object
	GLuint vbo_hdl;
	glCreateBuffers(1, &vbo_hdl);
	glNamedBufferStorage(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data(), GL_DYNAMIC_STORAGE_BIT);

	// create vertex array object
	// both attribute and bind indexes are 0
	GLApp::GLModel mdl;
	glCreateVertexArrays(1, &mdl.vaoid);
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 0, vbo_hdl, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 0);
	glBindVertexArray(0);

	// initialize model specs
	mdl.primitive_type = GL_LINES;
	mdl.setup_shdrpgm(vtx_shdr, frag_shdr);
	mdl.draw_cnt = 2 * (slices + 1) + 2 * (stacks + 1); // number of vertices
	mdl.primitive_cnt = mdl.draw_cnt / 2; // number of primitives
	return mdl;
}

GLApp::GLModel GLApp::trifans_model(GLint slices, std::string const& vtx_shdr, std::string const& frag_shdr)
{
	// Generate the (slices+2) count of vertices required to
	// render a triangle fan parameterization of a circle with unit
	// radius and centered at (0, 0)
	std::vector<glm::vec2> pos_vtx;
	GLfloat angle_deg{ 360.f / slices };

	pos_vtx.reserve(slices + 2);
	// pivot vertex
	pos_vtx.emplace_back(glm::vec2(0.f, 0.f));

	// compute parameterized points
	for (GLint pt{ 1 }; pt < slices + 2; ++pt)
	{
		pos_vtx.emplace_back(glm::vec2(cosf((pt - 1) * glm::radians(angle_deg)), sinf((pt - 1) * glm::radians(angle_deg))));
	}

	// Compute (slices+2) count of vertex color coordinates.
	std::vector<glm::vec3> clr_vtx;
	clr_vtx.reserve(slices + 2);
	for (GLint clr{ 0 }; clr < slices + 2; ++clr)
	{
		clr_vtx.emplace_back(glm::vec3(
			static_cast<GLfloat>(rand()) / static_cast<GLfloat>(RAND_MAX),
			static_cast<GLfloat>(rand()) / static_cast<GLfloat>(RAND_MAX),
			static_cast<GLfloat>(rand()) / static_cast<GLfloat>(RAND_MAX)));
	}

	// Generate a VAO handle to encapsulate the VBO(s) and state of triangle fan mesh
	GLuint vbo_hdl;
	glCreateBuffers(1, &vbo_hdl);
	glNamedBufferStorage(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo_hdl, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	GLApp::GLModel mdl;

	// vertex position attribute index is 0 and binding index is 2
	glCreateVertexArrays(1, &mdl.vaoid);
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 2, vbo_hdl, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 2);

	// vertex color attribute index is 1 and binding index is 3
	glEnableVertexArrayAttrib(mdl.vaoid, 1);
	glVertexArrayVertexBuffer(mdl.vaoid, 3, vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(mdl.vaoid, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 1, 3);
	glBindVertexArray(0);

	// initialize model specs
	mdl.primitive_type = GL_TRIANGLE_FAN;
	mdl.setup_shdrpgm(vtx_shdr, frag_shdr);
	mdl.draw_cnt = slices + 2; // number of vertices
	mdl.primitive_cnt = slices; // number of primitives
	return mdl;
}

GLApp::GLModel GLApp::tristrip_model(GLint slices, GLint stacks, std::string const& vtx_shdr, std::string const& frag_shdr)
{
	// Compute and store a "stack" of uniformly "sliced" points
	std::vector<glm::vec2> pos_vtx;
	pos_vtx.reserve((slices + 1) * (stacks + 1));

	// j is the stacks(rows) index and i is the slices(columns) index
	for (GLint i{ 0 }; i < stacks + 1; ++i)
	{
		for (GLint j{ 0 }; j < slices + 1; ++j)
		{
			pos_vtx.emplace_back(glm::vec2(
				((1.f - (-1.f)) / slices * j) - 1.0f, 
				((1.f - (-1.f)) / stacks * i) - 1.0f));
		}
	}

	// Compute (slices + 1) * (stacks + 1) count of vertex color coordinates.
	std::vector<glm::vec3> clr_vtx;
	clr_vtx.reserve((slices + 1) * (stacks + 1));
	for (GLint clr{ 0 }; clr < (slices + 1) * (stacks + 1); ++clr)
	{
		clr_vtx.emplace_back(glm::vec3(
			static_cast<GLfloat>(rand()) / static_cast<GLfloat>(RAND_MAX),
			static_cast<GLfloat>(rand()) / static_cast<GLfloat>(RAND_MAX),
			static_cast<GLfloat>(rand()) / static_cast<GLfloat>(RAND_MAX)));
	}

	// Generate a VAO handle to encapsulate the VBO(s) and state of triangle strip mesh
	GLuint vbo_hdl;
	glCreateBuffers(1, &vbo_hdl);
	glNamedBufferStorage(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo_hdl, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	GLApp::GLModel mdl;

	// vertex position attribute index is 0 and binding index is 2
	glCreateVertexArrays(1, &mdl.vaoid);
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 2, vbo_hdl, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 2);

	// vertex color attribute index is 1 and binding index is 3
	glEnableVertexArrayAttrib(mdl.vaoid, 1);
	glVertexArrayVertexBuffer(mdl.vaoid, 3, vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(mdl.vaoid, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 1, 3);

	// index array
	std::vector<GLushort> idx_vtx;
	idx_vtx.reserve(((slices + 1) * 2) * stacks + ((stacks - 1) * 2));

	// calculate indices
	for (GLint i{ 0 }; i < stacks; ++i)
	{
		for (GLint j{ 0 }; j <= slices; ++j)
		{
			idx_vtx.push_back(slices * (i + 1) + (i + 1) + j);
			//std::cout << slices * (i + 1) + (i + 1) + j << ", ";
			idx_vtx.push_back(slices * i + i + j);
			//std::cout << slices * i + i + j << ", ";
		}

		// add restart index to the end of first to second last stack indices
		if (i < stacks - 1)
		{
			idx_vtx.push_back(restart_index);
			//std::cout << restart_index << ",\n";
		}
	}

	// generate index buffer
	GLuint idx_elem_cnt = static_cast<GLuint>(idx_vtx.size());
	GLuint ebo_hdl;
	glCreateBuffers(1, &ebo_hdl);
	glNamedBufferStorage(ebo_hdl, sizeof(GLushort) * idx_elem_cnt, reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(mdl.vaoid, ebo_hdl);
	glBindVertexArray(0);

	// initialize model specs
	mdl.primitive_type = GL_TRIANGLE_STRIP;
	mdl.setup_shdrpgm(vtx_shdr, frag_shdr);
	mdl.draw_cnt = idx_elem_cnt;
	mdl.primitive_cnt = slices * stacks * 2;
	return mdl;
}

/*  _________________________________________________________________________*/
/*! GLApp::setup_shdrpgm

This function compiles the source codes of the vertex and fragment shaders, links
the shader objects to create a shader program for the rectangle model as well as
validating the shader program.
*/
void GLApp::GLModel::setup_shdrpgm(std::string const& vtx_shdr, std::string const& frag_shdr)
{
	if (!shdr_pgm.CompileShaderFromString(GL_VERTEX_SHADER, vtx_shdr))
	{
		std::cout << "Vertex shader failed to compile: ";
		std::cout << shdr_pgm.GetLog() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (!shdr_pgm.CompileShaderFromString(GL_FRAGMENT_SHADER, frag_shdr))
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
	switch (primitive_type)
	{
	case GL_POINTS:
		glPointSize(10.f);
		glVertexAttrib3f(1, 1.f, 0.f, 0.f); // red color for points
		glDrawArrays(primitive_type, 0, draw_cnt);
		glPointSize(1.f);
		break;
	case GL_LINES:
		glLineWidth(3.f);
		glVertexAttrib3f(1, 0.f, 0.0f, 1.f); // blue color for points
		glDrawArrays(primitive_type, 0, draw_cnt);
		glLineWidth(1.f);
		break;
	case GL_TRIANGLE_FAN:
		glDrawArrays(primitive_type, 0, draw_cnt);
		break;
	case GL_TRIANGLE_STRIP:
		glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
		glPrimitiveRestartIndex(restart_index);
		glDrawElements(primitive_type, draw_cnt, GL_UNSIGNED_SHORT, NULL);
	}

	// after completing the rendering, we tell the driver that VAO
	// vaoid and current shader program are no longer current
	glBindVertexArray(0);
	shdr_pgm.UnUse();
}