/*!
@file       glapp.cpp
@author  	pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date    	21/06/2024

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

GLApp::GLModel GLApp::mdl;

GLint GLApp::task_id{};
GLfloat GLApp::tile_size{ 16.0f };
GLboolean tile_size_flag{ GL_TRUE };
GLboolean modulate_flag{ GL_FALSE };
GLboolean rotation_flag{ GL_FALSE };
GLfloat PI{ glm::pi<float>() };
GLfloat animation_time_elapsed{ 0.0f };
GLuint tex_obj{};

/*  _________________________________________________________________________*/
/*! GLApp::init

This function clear the color buffer with initial RGB value, initialises viewport, setup
rectangle model vertex array object and shader program.
*/
void GLApp::init() {
	// Part 1: clear color buffer with RGBA value in glClearColor ...
	glClearColor(1.f, 1.f, 1.f, 1.f);

	// Part 2: use entire window as viewport ...
	glViewport(0, 0, GLHelper::width, GLHelper::height);

	mdl.setup_vao();
	mdl.setup_shdrpgm("../projects/tutorial-5/shaders/my-tutorial-5.vert","../projects/tutorial-5/shaders/my-tutorial-5.frag");
	tex_obj = setup_texobj("../images/duck-rgba-256.tex");

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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
	
	// update task_id
	if (GLHelper::keystateT)
	{
		switch (task_id)
		{
		case 0:
			task_id = 1;
			break;
		case 1:
			// reset tile size and animation time elapsed
			animation_time_elapsed = 0.0f;
			tile_size = 16.0f;
			task_id = 2;
			break;
		case 2:
			task_id = 3;
			break;
		case 3:
			task_id = 4;
			break;
		case 4:
			task_id = 5;
			break;
		case 5:
			task_id = 6;
			break;
		default:
			task_id = 0;
		}

		GLHelper::keystateT = GL_FALSE;
	}

	// update modulate flag
	if (GLHelper::keystateM)
	{
		modulate_flag = ~modulate_flag;
		GLHelper::keystateM = GL_FALSE;
	}

	// enable/disable blending
	if (GLHelper::keystateA)
	{
		glIsEnabled(GL_BLEND) ? glDisable(GL_BLEND) : glEnable(GL_BLEND);
		GLHelper::keystateA = GL_FALSE;
	}

	// update rotation flag
	if (GLHelper::keystateR)
	{
		rotation_flag = ~rotation_flag;
		GLHelper::keystateR = GL_FALSE;
	}

	// update animated checkerboard tile size
	if (task_id == 2)
	{
		animation_time_elapsed += static_cast<GLfloat>(GLHelper::delta_time);

		if (tile_size < 16.0f && !tile_size_flag)
		{
			tile_size_flag = GL_TRUE;
		}
		else if (tile_size > 256.0f && tile_size_flag)
		{
			tile_size_flag = GL_FALSE;
		}
		
		tile_size = 16.0f + ((sinf((PI * (animation_time_elapsed / 30)) - (PI / 2)) + 1.0f) / 2) * (256.0f - 16.0f);
	}
}

/*  _________________________________________________________________________*/
/*! GLApp::draw

This function renders all objects to the back buffer
*/
void GLApp::draw() {
	std::stringstream sstr;
	std::string task_name, alpha_blend_status, modulate_status, rotation_status;

	// update task name
	switch (task_id)
	{
	case 0:
		task_name = "Task 0: Paint Color";
		break;
	case 1:
		task_name = "Task 1: Fixed-Size Checkerboard";
		break;
	case 2:
		task_name = "Task 2: Animated Checkerboard";
		break;
	case 3:
		task_name = "Task 3: Texture Mapping";
		break;
	case 4:
		task_name = "Task 4: Repeating";
		break;
	case 5:
		task_name = "Task 5: Mirroring";
		break;
	case 6:
		task_name = "Task 6: Clamping";
	}

	// update statuses of alpha blending, modulation and rotation
	alpha_blend_status = glIsEnabled(GL_BLEND) ? "ON" : "OFF";
	modulate_status = modulate_flag ? "ON" : "OFF";
	rotation_status = rotation_flag ? "ON" : "OFF";

	// update window title
	sstr << "Tutorial 5 | Bryan Ang Wei Ze | " << task_name <<
		" | Alpha Blend: " << alpha_blend_status << 
		" | Modulate: " << modulate_status <<
		" | Rotation: " << rotation_status;

	glfwSetWindowTitle(GLHelper::ptr_window, sstr.str().c_str());

	// Clear back buffer of color buffer
	glClear(GL_COLOR_BUFFER_BIT);
	mdl.draw();
}

/*  _________________________________________________________________________*/
/*! GLApp::cleanup

This function returns buffer to GPU and delete textures
*/
void GLApp::cleanup()
{
	glInvalidateBufferData(mdl.vbo_hdl);
	glDeleteBuffers(1, &mdl.vbo_hdl);
	glDeleteVertexArrays(1, &mdl.vaoid);
	glDeleteTextures(1, &tex_obj);
}

/*  _________________________________________________________________________*/
/*! GLApp::GLModel::setup_vao

This function transfer rectangle model vertices data from client to server side
by creating vertex buffer and vertex array objects
*/
void GLApp::GLModel::setup_vao()
{
	// create an array of vertices
	std::array<Vertex, 4> vertices
	{
		glm::vec2(-1.f, 1.f), glm::vec3(1.f, 0.f, 1.f), glm::vec2(0.f, 1.f),
		glm::vec2(-1.f, -1.f), glm::vec3(1.f, 0.f, 0.f), glm::vec2(0.f, 0.f),
		glm::vec2(1.f, 1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec2(1.f, 1.f),
		glm::vec2(1.f, -1.f), glm::vec3(0.f, 1.f, 0.f), glm::vec2(1.f, 0.f)
	};

	glCreateBuffers(1, &vbo_hdl);

	// transfer vertices data to buffer
	glNamedBufferStorage(vbo_hdl, sizeof(vertices), vertices.data(), GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT);

	// vaoid is data member 1 of GLApp::GLModel
	glCreateVertexArrays(1, &vaoid);

	// for vertex position, attribute index is 0
	// and vertex buffer binding point is 3
	glEnableVertexArrayAttrib(vaoid, 0);
	glVertexArrayVertexBuffer(vaoid, 3, vbo_hdl, offsetof(Vertex, position), sizeof(Vertex));
	glVertexArrayAttribFormat(vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 0, 3);

	// for vertex color, attribute index is 1
	// and vertex buffer binding point is 4
	glEnableVertexArrayAttrib(vaoid, 1);
	glVertexArrayVertexBuffer(vaoid, 4, vbo_hdl, offsetof(Vertex, color), sizeof(Vertex));
	glVertexArrayAttribFormat(vaoid, 1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 1, 4);

	// for vertex texture, attribute index is 2
	// and vertex buffer binding point is 5
	glEnableVertexArrayAttrib(vaoid, 2);
	glVertexArrayVertexBuffer(vaoid, 5, vbo_hdl, offsetof(Vertex, texture), sizeof(Vertex));
	glVertexArrayAttribFormat(vaoid, 2, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 2, 5);

	primitive_type = GL_TRIANGLE_STRIP;
	draw_cnt = static_cast<GLuint>(vertices.size());
	glBindVertexArray(0);
}

/*  _________________________________________________________________________*/
/*! GLApp::setup_shdrpgm

This function compiles the source codes of the vertex and fragment shaders, links
the shader objects to create a shader program for the rectangle model as well as
validating the shader program.
*/
void GLApp::GLModel::setup_shdrpgm(std::string const& vtx_shdr, std::string const& frg_shdr)
{
	std::vector<std::pair<GLenum, std::string>> shdr_files{
	std::make_pair(GL_VERTEX_SHADER, vtx_shdr),
	std::make_pair(GL_FRAGMENT_SHADER, frg_shdr)
	};

	// Automation hook. [!WARNING!] Do not alter/remove this!
	AUTOMATION_HOOK_SHADER(shdr_pgm, shdr_files);

	if (GL_FALSE == shdr_pgm.CompileLinkValidate(shdr_files)) {
		std::cout << "Unable to compile/link/validate shader programs\n";
		std::cout << shdr_pgm.GetLog() << "\n";
		std::exit(EXIT_FAILURE);
	}
}

/*  _________________________________________________________________________*/
/*! GLApp::GLModel::draw

This function renders the rectangle model to the back buffer
*/
void GLApp::GLModel::draw()
{
	glBindTextureUnit(6, tex_obj);

	// update texture parameters
	switch (task_id)
	{
	case 3:
		glTextureParameteri(tex_obj, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(tex_obj, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		break;
	case 4:
		glTextureParameteri(tex_obj, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTextureParameteri(tex_obj, GL_TEXTURE_WRAP_T, GL_REPEAT);
		break;
	case 5:
		glTextureParameteri(tex_obj, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
		glTextureParameteri(tex_obj, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		break;
	case 6:
		glTextureParameteri(tex_obj, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(tex_obj, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	// compute rotation matrix
	GLdouble theta{ glfwGetTime() };
	GLfloat cos_theta{ static_cast<GLfloat>(std::cos(theta)) },
		sin_theta{ static_cast<GLfloat>(std::sin(theta)) };
	glm::mat2 rot_mtx{ cos_theta, sin_theta, -sin_theta, cos_theta };

	shdr_pgm.Use();
	glBindVertexArray(vaoid);

	// copy task id to fragment shader uniform variable uTask
	shdr_pgm.SetUniform("uTask", task_id);

	// tell fragment shader sampler2D uTex2d will use texture image unit 6
	shdr_pgm.SetUniform("uTex2d", 6);

	// send rotation matrix as uniform variable to vertex shader
	shdr_pgm.SetUniform("uRotMtx", rot_mtx);

	// copy rotation flag to vertex shader uniform variable uRotate
	shdr_pgm.SetUniform("uRotate", rotation_flag);

	// copy tile size to fragment shader uniform variable uTask
	shdr_pgm.SetUniform("uTileSize", tile_size);

	// copy modulate flag to fragment shader uniform variable uModulate
	shdr_pgm.SetUniform("uModulate", modulate_flag);

	// copy normalized mouse position to vertex shader uniform variable uMcn
	shdr_pgm.SetUniform("uMcn", GLHelper::mcn);

	glDrawArrays(primitive_type, 0, draw_cnt);

	// after completing the rendering, we tell the driver that VAO
	// vaoid and current shader program are no longer current
	glBindVertexArray(0);
	shdr_pgm.UnUse();
}

GLuint setup_texobj(std::string const& pathname)
{
	GLuint width{ 256 }, height{ 256 }, bytes_per_texel{ 4 };
	GLubyte* ptr_texels = new GLubyte[width * height * bytes_per_texel];
	
	std::ifstream file(pathname, std::ios::binary);
	if (!file.is_open())
	{
		std::cout << "Failed to open texture file\n";
		std::exit(EXIT_FAILURE);
	}

	file.read(reinterpret_cast<char*>(ptr_texels), width * height * bytes_per_texel);

	GLuint texobj_hdl;

	// define and initialize a handle to texture object that will
	// encapsulate two-dimensional textures
	glCreateTextures(GL_TEXTURE_2D, 1, &texobj_hdl);

	// allocate GPU storage for texture image data loaded from file
	glTextureStorage2D(texobj_hdl, 1, GL_RGBA8, width, height);

	// copy image data from client memory to GPU texture buffer memory
	glTextureSubImage2D(texobj_hdl, 0, 0, 0, width, height,
		GL_RGBA, GL_UNSIGNED_BYTE, ptr_texels);

	// client memory not required since image is buffered in GPU memory
	delete[] ptr_texels;

	// nothing more to do - return handle to texture object
	return texobj_hdl;
}