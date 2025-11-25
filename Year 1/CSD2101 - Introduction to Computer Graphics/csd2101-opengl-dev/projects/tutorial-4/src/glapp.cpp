/*!
@file       glapp.cpp
@author  	pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date    	05/06/2024

This file implements functionality useful and necessary to build OpenGL
applications including use of external APIs such as GLFW to create a
window and start up an OpenGL context and to extract function pointers
to OpenGL implementations.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glapp.h>

/*                                                   objects with file scope
----------------------------------------------------------------------------- */

// viewport properties
GLApp::GLViewport vp{ 0, 0, GLHelper::width,GLHelper::height };

// containers
std::map<std::string, GLApp::GLModel> GLApp::models;
std::map<std::string, GLSLShader> GLApp::shdrpgms;
std::map<std::string, GLApp::GLObject> GLApp::objects;

// camera
GLApp::Camera2D GLApp::camera2d;

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

	// Part 3: parse scene file /scenes/tutorial-4.scn
	// and store repositories of models of type GLModel in container
	// GLApp::models, store shader programs of type GLSLShader in
	// container GLApp::shdrpgms, and store repositories of objects of
	// type GLObject in container GLApp::objects
	GLApp::init_scene("../scenes/tutorial-4.scn");

	GLApp::camera2d.init(GLHelper::ptr_window, &GLApp::objects.at("Camera"));
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

	GLApp::camera2d.update(GLHelper::ptr_window);

	for (auto &obj : objects)
	{
		if (obj.first != "Camera")
		{
			obj.second.update(GLHelper::delta_time);
		}
	}
}

/*  _________________________________________________________________________*/
/*! GLApp::draw

This function renders all objects to the back buffer
*/
void GLApp::draw() {
	// Part 1: Write window title
	std::stringstream sstr;

	sstr << "Tutorial 4 | Bryan Ang Wei Ze | Camera Position (" << std::fixed << std::setprecision(2) << camera2d.pgo->position.x << ", " << camera2d.pgo->position.y <<
		") | Orientation: " << std::setprecision(0) << camera2d.pgo->orientation.x << " degrees" <<
		" | Window height: " << camera2d.height <<
		" | FPS: " << std::setprecision(2) << GLHelper::fps;
	glfwSetWindowTitle(GLHelper::ptr_window, sstr.str().c_str());

	// Clear back buffer of color buffer
	glClear(GL_COLOR_BUFFER_BIT);

	for (const auto& obj : objects)
	{
		if (obj.first != "Camera")
		{
			obj.second.draw();
		}
	}

	objects["Camera"].draw();
}

/*  _________________________________________________________________________*/
/*! GLApp::cleanup

This function returns buffer to GPU
*/
void GLApp::cleanup()
{
	for (auto &mdl : models)
	{
		mdl.second.release();
	}
}

/*  _________________________________________________________________________ */
/*! GLApp::GLModel::init

@param std::string mesh_filename
File name of mesh/model

Reads mesh/model data from file, create VAO to transfer mesh/model vertices positions and indices data to GPU
*/
void GLApp::GLModel::init(std::string mesh_filename)
{
	std::vector<glm::vec2> pos_vtx; 
	std::vector<GLushort> idx_vtx;
	primitive_cnt = 0;
	std::ifstream ifs{ mesh_filename, std::ios::in };
	if (!ifs) {
		std::cout << "ERROR: Unable to open mesh file: "
			<< mesh_filename << "\n";
		exit(EXIT_FAILURE);
	}
	ifs.seekg(0, std::ios::beg);

	std::string line;
	getline(ifs, line); // first line is name of mesh
	GLchar prefix;
	std::string mesh_name;
	std::istringstream line_sstm{ line };
	line_sstm >> prefix >> mesh_name;

	// read mesh vertices and indices
	while (getline(ifs, line))
	{
		std::istringstream line_sstm{ line };
		glm::vec2 pos(0.f, 0.f);
		GLushort idx;
		line_sstm >> prefix;

		if (prefix == 'v')
		{
			line_sstm >> pos.x >> pos.y;
			pos_vtx.emplace_back(glm::vec2(pos.x, pos.y));
		}

		if (prefix == 'f' || prefix == 't')
		{
			while (line_sstm >> idx)
			{
				idx_vtx.emplace_back(idx);
			}

			++primitive_cnt;
		}
	}

	// Generate a VAO handle to encapsulate the VBO(s) and state of mesh
	GLuint vbo_hdl;
	glCreateBuffers(1, &vbo_hdl);
	glNamedBufferStorage(vbo_hdl, sizeof(glm::vec2) * pos_vtx.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
	glNamedBufferSubData(vbo_hdl, 0, sizeof(glm::vec2) * pos_vtx.size(), pos_vtx.data());

	// vaoid is data member 1 of GLApp::GLModel
	glCreateVertexArrays(1, &vaoid);

	// vertex position attribute index is 0 and binding index is 2
	glEnableVertexArrayAttrib(vaoid, 0);
	glVertexArrayVertexBuffer(vaoid, 2, vbo_hdl, 0, sizeof(glm::vec2));
	glVertexArrayAttribFormat(vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 0, 2);

	// generate index buffer
	GLuint ebo_hdl;
	glCreateBuffers(1, &ebo_hdl);
	glNamedBufferStorage(ebo_hdl, sizeof(GLushort) * static_cast<GLuint>(idx_vtx.size()), reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(vaoid, ebo_hdl);
	glBindVertexArray(0);

	draw_cnt = static_cast<GLuint>(idx_vtx.size());
	if (prefix == 'f')
	{
		primitive_type = GL_TRIANGLE_FAN;
	}
	else
	{
		primitive_type = GL_TRIANGLES;
	}
}

/*  _________________________________________________________________________ */
/*! GLApp::GLModel::release

Returns allocated buffers to GPU
*/
void GLApp::GLModel::release()
{
	glInvalidateBufferData(vaoid);
	glDeleteBuffers(1, &vaoid);
}

/*  _________________________________________________________________________ */
/*! GLApp::GLObject::update

@param GLdouble delta_time
Frame time

Updates an object's orientation and transformation matrices
*/
void GLApp::GLObject::update(GLdouble delta_time)
{
	glm::mat3 scl, rot, trans;

	// compute current orientation
	orientation.x += orientation.y * static_cast<GLfloat>(delta_time);

	// compute scale matrix
	scl = glm::transpose(glm::mat3(
		scaling.x, 0.f, 0.f,
		0.f, scaling.y, 0.f,
		0.f, 0.f, 1.f
	));

	// compute rotation matrix
	rot = glm::transpose(glm::mat3(
		cosf(glm::radians(orientation.x)), -sinf(glm::radians(orientation.x)), 0.f,
		sinf(glm::radians(orientation.x)), cosf(glm::radians(orientation.x)), 0.f,
		0.f, 0.f, 1.f
	));

	// compute translation matrix
	trans = glm::transpose(glm::mat3(
		1.f, 0.f, position.x,
		0.f, 1.f, position.y,
		0.f, 0.f, 1.f
	));

	// compute model transformation matrix
	mdl_xform = trans * rot * scl;

	// compute model to ndc transformation matrix
	mdl_to_ndc_xform = camera2d.world_to_ndc_xform * mdl_xform;
}

/*  _________________________________________________________________________ */
/*! GLApp::GLObject::draw

Renders an object
*/
void GLApp::GLObject::draw() const
{
	// load shader program in use by object
	shd_ref->second.Use();

	// bind VAO of object's model
	glBindVertexArray(mdl_ref->second.vaoid);

	// copy object's color to fragment shader uniform variable uColor
	GLint uniform_color_loc1 = glGetUniformLocation(shd_ref->second.GetHandle(),
		"uColor");

	if (uniform_color_loc1 >= 0)
	{
		glUniform3fv(uniform_color_loc1, 1, glm::value_ptr(GLApp::GLObject::color));
	}
	else
	{
		std::cout << "Uniform variable doesn't exist!!!\n";
		std::exit(EXIT_FAILURE);
	}

	/* copy object's model-to-NDC matrix to vertex shader's
	uniform variable uModelToNDC */
	GLint uniform_model_to_ndc_loc1 = glGetUniformLocation(shd_ref->second.GetHandle(),
		"uModel_to_NDC");

	if (uniform_model_to_ndc_loc1 >= 0)
	{
		glUniformMatrix3fv(uniform_model_to_ndc_loc1, 1, GL_FALSE,
			glm::value_ptr(GLApp::GLObject::mdl_to_ndc_xform));
	}
	else
	{
		std::cout << "Uniform variable doesn't exist!!!\n";
		std::exit(EXIT_FAILURE);
	}

	// render object using indexed draw
	glDrawElements(mdl_ref->second.primitive_type, mdl_ref->second.draw_cnt, GL_UNSIGNED_SHORT, NULL);

	// unbind VAO
	glBindVertexArray(0);

	// unload shader program
	shd_ref->second.UnUse();
}

/*  _________________________________________________________________________ */
/*! GLApp::insert_shdrpgm

@param std::string shdr_pgm_name
Name of shader program

@param std::string vtx_shdr
File path of vertex shader

@param std::string frg_shdr
File path of fragment shader

Compiles and links shader files to create shader program which is then inserted to GLApp::shdrpgms
*/
void GLApp::insert_shdrpgm(std::string shdr_pgm_name, std::string vtx_shdr, std::string frg_shdr)
{
	std::map<std::string, GLSLShader>::iterator it =
		GLApp::shdrpgms.find(shdr_pgm_name);
	if (it != GLApp::shdrpgms.end()) return;
	std::vector<std::pair<GLenum, std::string>> shdr_files{
	std::make_pair(GL_VERTEX_SHADER, vtx_shdr),
	std::make_pair(GL_FRAGMENT_SHADER, frg_shdr)
	};

	GLSLShader shdr_pgm;

	// Automation hook. [!WARNING!] Do not alter/remove this!
	AUTOMATION_HOOK_SHADER(shdr_pgm, shdr_files);

	if (GL_FALSE == shdr_pgm.CompileLinkValidate(shdr_files)) {
		std::cout << "Unable to compile/link/validate shader programs\n";
		std::cout << shdr_pgm.GetLog() << "\n";
		std::exit(EXIT_FAILURE);
	}
	// add compiled, linked, and validated shader program to
	// std::map container GLApp::shdrpgms
	GLApp::shdrpgms[shdr_pgm_name] = shdr_pgm;
}

/*  _________________________________________________________________________ */
/*! GLApp::init_scene

@param std::string scene_filename
File path of scene file

Reads objects data of scene from file, creates and insert objects into
GLApp::objects container
*/
void GLApp::init_scene(std::string scene_filename)
{
	std::ifstream ifs{ scene_filename, std::ios::in };
	if (!ifs) {
		std::cout << "ERROR: Unable to open scene file: "
			<< scene_filename << "\n";
		exit(EXIT_FAILURE);
	}
	ifs.seekg(0, std::ios::beg);

	std::string line;
	getline(ifs, line); // first line is count of objects in scene
	std::istringstream line_sstm{ line };
	int obj_cnt;
	line_sstm >> obj_cnt; // read count of objects in scene
	while (obj_cnt--) // read each object's parameters
	{
		GLObject new_obj;

		// read model name
		getline(ifs, line);
		std::istringstream line_model_name{ line };
		std::string model_name;
		line_model_name >> model_name;

		if (models.find(model_name) == models.end())
		{
			models.insert({ model_name, GLModel()});
			models[model_name].init("../meshes/" + model_name + ".msh");
		}

		// read object's name
		getline(ifs, line);
		std::istringstream line_object_name{ line };
		std::string object_name;
		line_object_name >> object_name;

		// read shader program name and the directories of shaders
		getline(ifs, line);
		std::istringstream line_shdr_pgm{ line };
		std::string shdr_pgm_name, vrt_shdr, frag_shdr;
		line_shdr_pgm >> shdr_pgm_name >> vrt_shdr >> frag_shdr;

		if (shdrpgms.find(shdr_pgm_name) == shdrpgms.end())
		{
			insert_shdrpgm(shdr_pgm_name, vrt_shdr, frag_shdr);
		}

		// read object's rgb values
		getline(ifs, line);
		std::istringstream line_rgb_vals{ line };
		glm::vec3 rgb_vals;
		line_rgb_vals >> rgb_vals.r >> rgb_vals.g >> rgb_vals.b;

		// read object's scaling factors
		getline(ifs, line);
		std::istringstream line_scl_factors{ line };
		glm::vec2 scl_factors;
		line_scl_factors >> scl_factors.x >> scl_factors.y;

		// read object's orientation factors
		getline(ifs, line);
		std::istringstream line_orient_factors{ line };
		glm::vec2 orient_factors;
		line_orient_factors >> orient_factors.x >> orient_factors.y;

		// read object's position
		getline(ifs, line);
		std::istringstream line_position{ line };
		glm::vec2 position;
		line_position >> position.x >> position.y;

		// assign object's attributes
		new_obj.orientation = orient_factors;
		new_obj.scaling = scl_factors;
		new_obj.position = position;
		new_obj.color = rgb_vals;
		new_obj.mdl_ref = models.find(model_name);
		new_obj.shd_ref = shdrpgms.find(shdr_pgm_name);

		// insert object into container
		objects.insert({ object_name, new_obj });
	}
}

/*  _________________________________________________________________________ */
/*! GLApp::Camera2D::init

@param GLFWwindow pWindow
Pointer to GLFWwindow object

@param GLApp::GLObject ptr
Pointer to scene object to embed camera in

Initializes specifications of 2D camera
*/
void GLApp::Camera2D::init(GLFWwindow* pWindow, GLApp::GLObject* ptr)
{
	// embed camera
	pgo = ptr;

	GLsizei fb_width, fb_height;
	glfwGetFramebufferSize(pWindow, &fb_width, &fb_height);
	ar = static_cast<GLfloat>(fb_width) / fb_height;

	// compuate up vector
	up = glm::transpose(glm::mat3(
		cosf(glm::radians(pgo->orientation.x)), -sinf(glm::radians(pgo->orientation.x)), 0.f,
		sinf(glm::radians(pgo->orientation.x)), cosf(glm::radians(pgo->orientation.x)), 0.f,
		0.f, 0.f, 1.f
	)) * glm::vec3(0.f, 1.f, 0.f);

	// compuate right vector
	right = glm::transpose(glm::mat2(
		cosf(glm::radians(-90.f)), -sinf(glm::radians(-90.f)),
		sinf(glm::radians(-90.f)), cosf(glm::radians(-90.f))
	)) * up;

	// initializes to free camera
	view_xform = glm::transpose(glm::mat3(
		1.f, 0.f, -pgo->position.x,
		0.f, 1.f, -pgo->position.y,
		0.f, 0.f, 1.f
	));

	// compute camera window to ndc transformation matrix
	camwin_to_ndc_xform = glm::mat3(
		2.f / (ar * height), 0.f, 0.f,
		0.f, 2.f / height, 0.f,
		0.f, 0.f, 1.f
	);

	// compute world to ndc transformation matrix
	world_to_ndc_xform = camwin_to_ndc_xform * view_xform;
}

/*  _________________________________________________________________________ */
/*! GLApp::Camera2D::update

@param GLFWwindow pWindow
Pointer to GLFWwindow object

Updates camera specifications based on user input
*/
void GLApp::Camera2D::update(GLFWwindow* pWindow)
{
	// update flags
	camtype_flag = (GLHelper::keystateV ? GL_TRUE : GL_FALSE);
	zoom_flag = (GLHelper::keystateZ ? GL_TRUE : GL_FALSE);
	left_turn_flag = (GLHelper::keystateH ? GL_TRUE : GL_FALSE);
	right_turn_flag = (GLHelper::keystateK ? GL_TRUE : GL_FALSE);
	move_flag = (GLHelper::keystateU ? GL_TRUE : GL_FALSE);

	// update aspect ratio
	GLsizei fb_width, fb_height;
	glfwGetFramebufferSize(pWindow, &fb_width, &fb_height);
	ar = static_cast<GLfloat>(fb_width) / fb_height;

	// update orientation
	if (left_turn_flag)
	{
		pgo->orientation.x += pgo->orientation.y;
	}
	if (right_turn_flag)
	{
		pgo->orientation.x -= pgo->orientation.y;
	}

	// update up and right vectors
	if (left_turn_flag || right_turn_flag)
	{
		// update up vector
		up = glm::transpose(glm::mat3(
			cosf(glm::radians(pgo->orientation.x)), -sinf(glm::radians(pgo->orientation.x)), 0.f,
			sinf(glm::radians(pgo->orientation.x)), cosf(glm::radians(pgo->orientation.x)), 0.f,
			0.f, 0.f, 1.f
		)) * glm::vec3(0.f, 1.f, 0.f);

		// update right vector
		right = glm::transpose(glm::mat2(
			cosf(glm::radians(-90.f)), -sinf(glm::radians(-90.f)),
			sinf(glm::radians(-90.f)), cosf(glm::radians(-90.f))
		)) * up;
	}

	// update position
	if (move_flag)
	{
		pgo->position += linear_speed * up;
	}

	// zoom effect
	if (height == max_height)
	{
		height_chg_dir = -1;
	}
	else if (height == min_height)
	{
		height_chg_dir = 1;
	}
	if (zoom_flag)
	{
		height += height_chg_dir * height_chg_val;
	}

	// switch between free and first person camera
	if (camtype_flag)
	{
		view_xform = glm::transpose(glm::mat3(
			right.x, right.y, glm::dot(-right, pgo->position),
			up.x, up.y, glm::dot(-up, pgo->position),
			0.f, 0.f, 1.f
		));
	}
	else
	{
		view_xform = glm::transpose(glm::mat3(
			1.f, 0.f, -pgo->position.x,
			0.f, 1.f, -pgo->position.y,
			0.f, 0.f, 1.f
		));
	}

	// update transformation matrics
	// compute camera to ndc transformation matrix
	camwin_to_ndc_xform = glm::transpose(glm::mat3(
		2.f / (ar * height), 0.f, 0.f,
		0.f, 2.f / height, 0.f,
		0.f, 0.f, 1.f
	));

	// compute world to ndc transformation matrix
	world_to_ndc_xform = camwin_to_ndc_xform * view_xform;

	glm::mat3 scl, rot, trans;

	// compute camera object scale matrix
	scl = glm::transpose(glm::mat3(
		pgo->scaling.x, 0.f, 0.f,
		0.f, pgo->scaling.y, 0.f,
		0.f, 0.f, 1.f
	));

	// compute camera object rotation matrix
	rot = glm::transpose(glm::mat3(
		cosf(glm::radians(pgo->orientation.x)), -sinf(glm::radians(pgo->orientation.x)), 0.f,
		sinf(glm::radians(pgo->orientation.x)), cosf(glm::radians(pgo->orientation.x)), 0.f,
		0.f, 0.f, 1.f
	));

	// compute camera object translation matrix
	trans = glm::transpose(glm::mat3(
		1.f, 0.f, pgo->position.x,
		0.f, 1.f, pgo->position.y,
		0.f, 0.f, 1.f
	));

	// compute camera model transformation matrix
	pgo->mdl_xform = trans * rot * scl;

	// compute camera model to ndc transformation matrix
	pgo->mdl_to_ndc_xform = world_to_ndc_xform * pgo->mdl_xform;
}