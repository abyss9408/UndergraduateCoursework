/*!
@file       glapp.cpp
@author  	pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date    	30/05/2024

This file implements functionality useful and necessary to build OpenGL
applications including use of external APIs such as GLFW to create a
window and start up an OpenGL context and to extract function pointers
to OpenGL implementations.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glapp.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <glm/gtc/type_ptr.hpp>



/*                                                   objects with file scope
----------------------------------------------------------------------------- */
const std::string my_tutorial_3_vs = {
  #include "my-tutorial-3.vert"
};

const std::string my_tutorial_3_fs = {
  #include "my-tutorial-3.frag"
};

const std::string my_red_fs = {
  #include "my-red.frag"
};

const std::string my_green_fs = {
  #include "my-green.frag"
};

const std::string my_blue_fs = {
  #include "my-blue.frag"
};

const std::string my_black_fs = {
  #include "my-black.frag"
};

const std::string my_pulse_fs = {
  #include "my-pulse.frag"
};

// viewport properties
GLApp::GLViewport vp{ 0, 0, GLHelper::width,GLHelper::height };

// containers
std::vector<GLApp::GLModel> GLApp::models;
std::vector<GLSLShader> GLApp::shdrpgms;
std::list<GLApp::GLObject> GLApp::objects;

// spawn object flag
GLboolean spawn_obj{ GL_TRUE };

// static data members of GLApp
GLint GLApp::num_box_obj{};
GLint GLApp::num_mys_obj{};
GLuint GLApp::current_shdr_pgm{};

// random number generator
std::random_device rd;
std::default_random_engine gen(rd());

/*  _________________________________________________________________________*/
/*! GLApp::init

This function clear the color buffer with initial RGB value, initialises viewport, setup
rectangle model vertex array object and shader program.
*/
void GLApp::init() {
	// Part 1: clear color buffer with RGBA value in glClearColor ...
	glClearColor(1.f, 1.f, 1.f, 1.f);

	// Part 2: create viewport
	glViewport(vp.x, vp.y, vp.width, vp.height);

	// container of pairs of std::strings with each pair encapsulating
	// a vertex shader and a fragment shader
	GLApp::VPSS shdr_strs{
	std::make_pair(my_tutorial_3_vs, my_tutorial_3_fs),
	std::make_pair(my_tutorial_3_vs, my_red_fs),
	std::make_pair(my_tutorial_3_vs, my_green_fs),
	std::make_pair(my_tutorial_3_vs, my_blue_fs),
	std::make_pair(my_tutorial_3_vs, my_black_fs),
	std::make_pair(my_tutorial_3_vs, my_pulse_fs)
	};

	//create shared shader programs
	GLApp::init_shdrpgms_cont(shdr_strs);

	//initialize geometric models
	GLApp::init_models_cont();
}

/*  _________________________________________________________________________*/
/*! GLApp::update

This function dynamically compute the background color and colors of each vertex of the
rectangle model when U key state is toggled to true. Otherwise, it toggles the background color
and colors of each vertex to the respective static colors.
*/
void GLApp::update() {
	GLint w{ GLHelper::width }, h{ GLHelper::height };
	static GLint old_w{}, old_h{};
	// update viewport settings if window's dimension change
	if (w != old_w || h != old_h)
	{
		vp = { 0,0,w,h };
		glViewport(vp.x, vp.y, vp.width, vp.height);
		old_w = w;
		old_h = h;
	}

	// update polygon rasterization mode using glPolygonMode
	if (GLHelper::keystateP)
	{
		GLint polygonMode;
		glGetIntegerv(GL_POLYGON_MODE, &polygonMode);

		switch (polygonMode)
		{
		case GL_FILL:
			polygonMode = GL_LINE;
			break;
		case GL_LINE:
			polygonMode = GL_POINT;
			break;
		default:
			polygonMode = GL_FILL;
			break;
		}
		glPolygonMode(GL_FRONT_AND_BACK, polygonMode);
		GLHelper::keystateP = GL_FALSE;
	}

	// update selected shader program
	if (GLHelper::keystateS)
	{
		switch (current_shdr_pgm)
		{
		case 0:
			// change to red
			current_shdr_pgm = 1;
			break;
		case 1:
			// change to green
			current_shdr_pgm = 2;
			break;
		case 2:
			// change to blue
			current_shdr_pgm = 3;
			break;
		case 3:
			// change to black
			current_shdr_pgm = 4;
			break;
		case 4:
			// change to pulse
			current_shdr_pgm = 5;
			break;
		default:
			// change to back to default
			current_shdr_pgm = 0;
			break;
		}
		GLHelper::keystateS = GL_FALSE;
	}

	if (GLHelper::buttonstateLeft)
	{
		// if maximum object limit is not reached, spawn new object(s)
		if (objects.size() == 0) // first object
		{
			GLApp::GLObject first_obj;
			first_obj.init();
			objects.emplace_back(first_obj);
		}
		else if (objects.size() > 0 && objects.size() < 32768 && spawn_obj)
		{
			size_t num_of_objects_to_spawn{ objects.size()};
			for (size_t i{}; i < num_of_objects_to_spawn; ++i)
			{
				GLApp::GLObject obj;
				obj.init();
				objects.emplace_back(obj);
			}
		}
		else if (objects.size() == 32768) // maxinum object limit is reached
		{
			spawn_obj = GL_FALSE;
		}

		if (GL_FALSE == spawn_obj)
		{
			// kill oldest objects until there is only 1 object left
			if (objects.size() != 1)
			{
				size_t num_of_objects_to_kill{ objects.size() / 2 };
				for (size_t i{}; i < num_of_objects_to_kill; ++i)
				{
					// if oldest object is box, decrement the box object counter. Otherwise, decrement the mystery object counter
					objects.front().mdl_ref == 0 ? --GLApp::num_box_obj : --GLApp::num_mys_obj;

					// destroy oldest object
					objects.pop_front();
				}
				
				// make left clicks spawn objects again
				if (objects.size() == 1)
				{
					spawn_obj = GL_TRUE;
				}
			}
		}

		GLHelper::buttonstateLeft = GL_FALSE;
	}

	for (auto &obj : objects)
	{
		obj.update(GLHelper::delta_time);
	}
}

/*  _________________________________________________________________________*/
/*! GLApp::draw

This function renders all objects to the back buffer
*/
void GLApp::draw() {
	// Part 1: Write window title
	std::stringstream sstr;
	std::string shdr_pgm_name;

	switch (current_shdr_pgm)
	{
	case 1:
		shdr_pgm_name = "RED";
		break;
	case 2:
		shdr_pgm_name = "GREEN";
		break;
	case 3:
		shdr_pgm_name = "BLUE";
		break;
	case 4:
		shdr_pgm_name = "BLACK";
		break;
	case 5:
		shdr_pgm_name = "PULSE";
		break;
	default:
		shdr_pgm_name = "DEFAULT";
		break;
	}

	sstr << "Tutorial 3 | Bryan Ang Wei Ze | Obj: " << objects.size() <<
		" | Box: " << GLApp::num_box_obj <<
		" | Mystery: " << GLApp::num_mys_obj <<
		" | " << std::fixed << std::setprecision(2) << GLHelper::fps <<
		" | Current Shader: " << shdr_pgm_name;
	glfwSetWindowTitle(GLHelper::ptr_window, sstr.str().c_str());

	// Clear back buffer of color buffer
	glClear(GL_COLOR_BUFFER_BIT);

	GLint polygonMode;
	glGetIntegerv(GL_POLYGON_MODE, &polygonMode);

	if (GL_POINT == polygonMode)
	{
		glPointSize(10.f);
	}
	else if (GL_LINE == polygonMode)
	{
		glLineWidth(15.f);
	}

	// Render each object in container GLApp::objects
	for (auto const& obj : GLApp::objects) {
		obj.draw(); // call member function GLObject::draw()
	}

	glPointSize(1.f);
	glLineWidth(1.f);
}

void GLApp::cleanup() {

	for (auto &mdl : models)
	{
		glInvalidateBufferData(mdl.vaoid);
		glDeleteBuffers(1, &mdl.vaoid);
	}
}

/*  _________________________________________________________________________*/
/*! GLApp::setup_shdrpgm

This function compiles the source codes of the vertex and fragment shaders, links
the shader objects to create a shader program for the rectangle model as well as
validating the shader program.
*/
void GLApp::init_shdrpgms_cont(GLApp::VPSS const& vpss)
{
	for (auto const& x : vpss)
	{
		GLSLShader shdr_pgm;

		if (!shdr_pgm.CompileShaderFromString(GL_VERTEX_SHADER, x.first))
		{
			std::cout << "Vertex shader failed to compile: ";
			std::cout << shdr_pgm.GetLog() << std::endl;
			std::exit(EXIT_FAILURE);
		}

		if (!shdr_pgm.CompileShaderFromString(GL_FRAGMENT_SHADER, x.second))
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

		// insert shader program into container
		GLApp::shdrpgms.emplace_back(shdr_pgm);
	}
}

void GLApp::init_models_cont()
{
	GLApp::models.emplace_back(GLApp::box_model());
	GLApp::models.emplace_back(GLApp::mystery_model());
}

GLApp::GLModel GLApp::box_model()
{
	std::uniform_real_distribution<GLfloat> urdf(0.0f, std::nextafter(1.0f, std::numeric_limits<GLfloat>::max()));

	// vertex position attributes
	std::array<glm::vec2, 4> pos_vtx
	{
		glm::vec2(0.5f, -0.5f), glm::vec2(0.5f, 0.5f),
		glm::vec2(-0.5f, 0.5f), glm::vec2(-0.5f, -0.5f)
	};

	// vertex color attributes
	std::array<glm::vec3, 4> clr_vtx
	{
		glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)),
		glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen))
	};

	// Generate a VAO handle to encapsulate the VBO(s) and state of triangle strip mesh
	GLuint vbo_hdl;
	glCreateBuffers(1, &vbo_hdl);
	glNamedBufferStorage(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo_hdl, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	// vaoid is data member 1 of GLApp::GLModel
	GLApp::GLModel mdl;
	glCreateVertexArrays(1, &mdl.vaoid);

	// vertex position attribute index is 0 and binding index is 2
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 2, vbo_hdl, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 2);

	// vertex color attribute index is 1 and binding index is 3
	glEnableVertexArrayAttrib(mdl.vaoid, 1);
	glVertexArrayVertexBuffer(mdl.vaoid, 3, vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(mdl.vaoid, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 1, 3);

	std::array<GLushort, 6> idx_vtx
	{
		0, 1, 2,
		2, 3, 0
	};

	// generate index buffer
	GLuint ebo_hdl;
	glCreateBuffers(1, &ebo_hdl);
	glNamedBufferStorage(ebo_hdl, sizeof(GLushort) * static_cast<GLuint>(idx_vtx.size()), reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(mdl.vaoid, ebo_hdl);
	glBindVertexArray(0);

	mdl.primitive_type = GL_TRIANGLES;
	mdl.primitive_cnt = 2;
	mdl.draw_cnt = static_cast<GLuint>(idx_vtx.size());
	return mdl;
}

GLApp::GLModel GLApp::mystery_model()
{
	std::uniform_real_distribution<GLfloat> urdf(0.0f, std::nextafter(1.0f, std::numeric_limits<GLfloat>::max()));

	// vertex position attributes
	std::array<glm::vec2, 24> pos_vtx
	{
		glm::vec2(0.5f, -0.5f), glm::vec2(0.5f, 0.5f), glm::vec2(-0.5f, 0.5f), glm::vec2(-0.5f, -0.5f), // centre
		glm::vec2(1.5f, -0.5f), glm::vec2(1.5f, 0.5f), glm::vec2(1.5f, -1.0f), glm::vec2(1.5f, 1.0f), glm::vec2(2.5f, 0.0f), // right
		glm::vec2(0.5f, 1.5f), glm::vec2(-0.5f, 1.5f), glm::vec2(1.0f, 1.5f), glm::vec2(-1.0f, 1.5f), glm::vec2(0.0f, 2.5f), // top
		glm::vec2(-1.5f, 0.5f), glm::vec2(-1.5f, -0.5f), glm::vec2(-1.5f, 1.0f), glm::vec2(-1.5f, -1.0f), glm::vec2(-2.5f, 0.0f), // left
		glm::vec2(-0.5f, -1.5f), glm::vec2(0.5f, -1.5f), glm::vec2(-1.0f, -1.5f), glm::vec2(1.0f, -1.5f), glm::vec2(0.0f, -2.5f) // bottom
	};

	// vertex color attributes
	std::array<glm::vec3, 24> clr_vtx
	{
		glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), // centre
		glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), // right
		glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), // top
		glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), // left
		glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)), glm::vec3(urdf(gen), urdf(gen), urdf(gen)) // bottom
	};

	// Generate a VAO handle to encapsulate the VBO(s) and state of triangle strip mesh
	GLuint vbo_hdl;
	glCreateBuffers(1, &vbo_hdl);
	glNamedBufferStorage(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size() + sizeof(glm::vec3) * clr_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo_hdl, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());
	glNamedBufferSubData(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3) * clr_vtx.size(), clr_vtx.data());

	// vaoid is data member 1 of GLApp::GLModel
	GLApp::GLModel mdl;
	glCreateVertexArrays(1, &mdl.vaoid);

	// vertex position attribute index is 0 and binding index is 2
	glEnableVertexArrayAttrib(mdl.vaoid, 0);
	glVertexArrayVertexBuffer(mdl.vaoid, 2, vbo_hdl, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(mdl.vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 0, 2);

	// vertex color attribute index is 1 and binding index is 3
	glEnableVertexArrayAttrib(mdl.vaoid, 1);
	glVertexArrayVertexBuffer(mdl.vaoid, 3, vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), sizeof(glm::vec3));
	glVertexArrayAttribFormat(mdl.vaoid, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(mdl.vaoid, 1, 3);

	std::array<GLushort, 42> idx_vtx
	{
		// centre
		0, 1, 2,
		2, 3, 0,

		// right
		4, 5, 1,
		1, 0, 4,
		8, 7, 6,

		// top
		1, 9, 10,
		10, 2, 1,
		13, 12, 11,

		// left
		3, 2, 14,
		14, 15, 3,
		18, 17, 16,

		// bottom
		20, 0, 3,
		3, 19, 20,
		23, 22, 21
	};

	// generate index buffer
	GLuint ebo_hdl;
	glCreateBuffers(1, &ebo_hdl);
	glNamedBufferStorage(ebo_hdl, sizeof(GLushort) * static_cast<GLuint>(idx_vtx.size()), reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(mdl.vaoid, ebo_hdl);
	glBindVertexArray(0);

	mdl.primitive_type = GL_TRIANGLES;
	mdl.primitive_cnt = 14;
	mdl.draw_cnt = static_cast<GLuint>(idx_vtx.size());
	return mdl;
}

void GLApp::GLObject::init()
{
	std::uniform_real_distribution<GLfloat> urdf(-1.0f, std::nextafter(1.0f, std::numeric_limits<GLfloat>::max()));
	std::uniform_int_distribution<> urdi(0, 1);
	GLfloat const world_half_extent{ 5000.0f };
	
	// scale object width and height between 40 and 500 units
	scaling = glm::vec2(urdf(gen) * 175.f + 225.f,
		urdf(gen) * 175.f + 225.f);

	// position object in game world such that its x and y coordinates
	// are in range [-5,000, +5,000) ...
	position = glm::vec2(urdf(gen) * world_half_extent,
		urdf(gen) * world_half_extent);

	angle_disp = urdf(gen) * 360.f;
	angle_speed = urdf(gen) * 30.f;

	shd_ref = current_shdr_pgm;
	mdl_ref = urdi(gen);

	mdl_ref == 0 ? ++GLApp::num_box_obj : ++GLApp::num_mys_obj;
}

void GLApp::GLObject::update(GLdouble delta_time)
{
	glm::mat3 ndc_scl, scl, rot, trans;
	// compute current orientation
	angle_disp += angle_speed * static_cast<GLfloat>(delta_time);

	// compute scale matrix
	scl = glm::mat3(
		scaling.x, 0.f, 0.f,
		0.f, scaling.y, 0.f,
		0.f, 0.f, 1.f
	);

	// compute rotation matrix
	rot = glm::transpose(glm::mat3(
		cosf(glm::radians(angle_disp)), -sinf(glm::radians(angle_disp)), 0.f,
		sinf(glm::radians(angle_disp)), cosf(glm::radians(angle_disp)), 0.f,
		0.f, 0.f, 1.f
	));

	// compute translation matrix
	trans = glm::transpose(glm::mat3(
		1.f, 0.f, position.x,
		0.f, 1.f, position.y,
		0.f, 0.f, 1.f
	));

	ndc_scl = glm::mat3(
		1.f / 5000, 0.f, 0.f,
		0.f, 1.f / 5000, 0.f,
		0.f, 0.f, 1.f
	);

	mdl_to_ndc_xform = ndc_scl * trans * rot * scl;
}

void GLApp::GLObject::draw() const
{
	GLApp::shdrpgms[shd_ref].Use();
	glBindVertexArray(GLApp::models[mdl_ref].vaoid);

	GLint uniform_var_loc1 = glGetUniformLocation(GLApp::shdrpgms[shd_ref].GetHandle(),
		"uModel_to_NDC");

	// if referenced shader program is pulse
	if (5 == shd_ref)
	{
		GLfloat currentTime = static_cast<GLfloat>(glfwGetTime());

		GLint uniform_time_loc1 = glGetUniformLocation(GLApp::shdrpgms[shd_ref].GetHandle(),
			"uTime");

		if (uniform_time_loc1 >= 0)
		{
			glUniform1f(uniform_time_loc1, currentTime);
		}
		else
		{
			std::cout << "Uniform variable doesn't exist!!!\n";
			std::exit(EXIT_FAILURE);
		}
	}

	if (uniform_var_loc1 >= 0)
	{
		glUniformMatrix3fv(uniform_var_loc1, 1, GL_FALSE,
			glm::value_ptr(GLApp::GLObject::mdl_to_ndc_xform));
	}
	else
	{
		std::cout << "Uniform variable doesn't exist!!!\n";
		std::exit(EXIT_FAILURE);
	}

	glDrawElements(models[mdl_ref].primitive_type, models[mdl_ref].draw_cnt, GL_UNSIGNED_SHORT, NULL);
	glBindVertexArray(0);
	GLApp::shdrpgms[shd_ref].UnUse();
}