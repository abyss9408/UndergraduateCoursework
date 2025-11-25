/*!
@file       glpbo.cpp
@author     pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date       29/06/2024

This file simulates GPU rasterizer.

*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glpbo.h>
#include <glhelper.h>

GLsizei GLPbo::width;
GLsizei GLPbo::height;
GLsizei GLPbo::pixel_cnt;
GLsizei GLPbo::byte_cnt;
GLPbo::Color* GLPbo::ptr_to_pbo;
GLuint GLPbo::vaoid;
GLuint GLPbo::elem_cnt;
GLuint GLPbo::pboid;
GLuint GLPbo::texid;
GLSLShader GLPbo::shdr_pgm;
GLPbo::Color GLPbo::clear_clr;


std::string my_tutorial_6_vs
{
	R"( #version 450 core

	layout (location=0) in vec2 aVertexPosition;
	layout (location=1) in vec2 aVertexTexture;
	
	layout (location=0) out vec2 vTexture;
	
	void main()
	{
		gl_Position = vec4(aVertexPosition, 0.0, 1.0);
		vTexture = aVertexTexture;
	})"
};

std::string my_tutorial_6_fs
{
	R"(#version 450 core
	
	layout (location=0) in vec2 aVertexTexture;
	layout (location=0) out vec4 fFragColor;

	uniform sampler2D uTex2d;
	
	void main()
	{
		fFragColor = texture(uTex2d, aVertexTexture);
	})"
};

double map_range(double s, double a1, double a2, double b1, double b2)
{
	return b1 + (((s - a1)*(b2 - b1)) / (a2 - a1));
}

void GLPbo::set_clear_color(GLPbo::Color clr)
{
	clear_clr = clr;
}

void GLPbo::set_clear_color(GLubyte r, GLubyte g, GLubyte b, GLubyte a)
{
	clear_clr.r = r;
	clear_clr.g = g;
	clear_clr.b = b;
	clear_clr.a = a;
}

void GLPbo::clear_color_buffer()
{
	std::fill(ptr_to_pbo, ptr_to_pbo + pixel_cnt, clear_clr);
}

void GLPbo::init(GLsizei w, GLsizei h)
{
	width = w;
	height = h;
	pixel_cnt = width * height;
	byte_cnt = pixel_cnt * 4;
	set_clear_color(255, 255, 255);

	// create texture object
	glCreateTextures(GL_TEXTURE_2D, 1, &texid);
	glTextureStorage2D(texid, 1, GL_RGBA8, width, height);

	// create pixel buffer object
	glCreateBuffers(1, &pboid);
	glNamedBufferStorage(pboid, byte_cnt, nullptr, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT);

	setup_quad_vao();
	setup_shdrpgm();
}

void GLPbo::setup_quad_vao()
{
	// create an array of vertices
	std::array<Vertex, 4> vertices
	{
		glm::vec2(1.f, -1.f), glm::vec2(1.f, 0.f),
		glm::vec2(1.f, 1.f), glm::vec2(1.f, 1.f),
		glm::vec2(-1.f, 1.f), glm::vec2(0.f, 1.f),
		glm::vec2(-1.f, -1.f), glm::vec2(0.f, 0.f)
	};

	GLuint vbo_hdl;
	glCreateBuffers(1, &vbo_hdl);

	// transfer vertices data to buffer
	glNamedBufferStorage(vbo_hdl, sizeof(vertices), vertices.data(), GL_DYNAMIC_STORAGE_BIT);

	// vaoid is data member 1 of GLApp::GLModel
	glCreateVertexArrays(1, &vaoid);

	// for vertex position, attribute index is 0
	// and vertex buffer binding point is 3
	glEnableVertexArrayAttrib(vaoid, 0);
	glVertexArrayVertexBuffer(vaoid, 3, vbo_hdl, offsetof(Vertex, position), sizeof(Vertex));
	glVertexArrayAttribFormat(vaoid, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 0, 3);

	// for vertex texture, attribute index is 1
	// and vertex buffer binding point is 4
	glEnableVertexArrayAttrib(vaoid, 1);
	glVertexArrayVertexBuffer(vaoid, 4, vbo_hdl, offsetof(Vertex, texture), sizeof(Vertex));
	glVertexArrayAttribFormat(vaoid, 1, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(vaoid, 1, 4);

	std::array<GLushort, 6> idx_vtx
	{
		0, 1, 2, // 1st triangle's position and color coordinates are stored
		// in indices 0, 1, 2 of vertex attribute arrays in VBO
		2, 3, 0	 // 2nd triangle's position and color coordinates are stored
		// in indices 2, 3, 0 of vertex attribute arrays in VBO
	};

	elem_cnt = static_cast<GLuint>(idx_vtx.size());

	// transfer topology information from CPU to GPU
	GLuint ebo_hdl;
	glCreateBuffers(1, &ebo_hdl);
	glNamedBufferStorage(ebo_hdl, sizeof(GLushort) * elem_cnt, reinterpret_cast<GLvoid*>(idx_vtx.data()), GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayElementBuffer(vaoid, ebo_hdl);
	glBindVertexArray(0);
}

void GLPbo::setup_shdrpgm()
{
	if (!shdr_pgm.CompileShaderFromString(GL_VERTEX_SHADER, my_tutorial_6_vs))
	{
		std::cout << "Vertex shader failed to compile: ";
		std::cout << shdr_pgm.GetLog() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (!shdr_pgm.CompileShaderFromString(GL_FRAGMENT_SHADER, my_tutorial_6_fs))
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

void GLPbo::emulate()
{
	set_clear_color(static_cast<GLubyte>(map_range(cos(glfwGetTime()), -1.0, 1.0, 0.0, 255.0)),
		static_cast<GLubyte>(map_range(sin(glfwGetTime()), -1.0, 1.0, 0.0, 255.0)),
		static_cast<GLubyte>(map_range(cos(glfwGetTime()), -1.0, 1.0, 0.0, 255.0)));

	if (GLHelper::keystateR)
	{
		set_clear_color(255, 0, 0);
	}
	else if (GLHelper::keystateG)
	{
		set_clear_color(0, 255, 0);
	}
	else if (GLHelper::keystateB)
	{
		set_clear_color(0, 0, 255);
	}

	ptr_to_pbo = static_cast<Color*>(glMapNamedBuffer(pboid, GL_WRITE_ONLY));

	clear_color_buffer();

	glUnmapNamedBuffer(pboid);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboid);

	// copy image data from client memory to GPU texture buffer memory
	glTextureSubImage2D(texid, 0, 0, 0, width, height,
		GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	std::stringstream sstr;

	// update window title
	sstr << "Tutorial 6 | Bryan Ang Wei Ze" <<
		" | PBO Size: " << width << " x " << height <<
		" | FPS: " << std::fixed << std::setprecision(2) << GLHelper::fps;

	glfwSetWindowTitle(GLHelper::ptr_window, sstr.str().c_str());
}

void GLPbo::draw_fullwindow_quad()
{
	shdr_pgm.Use();
	
	// tell fragment shader sampler2D uTex2d will use texture image unit 2
	shdr_pgm.SetUniform("uTex2d", 2);

	glBindTextureUnit(2, texid);
	glBindVertexArray(vaoid);
	glDrawElements(GL_TRIANGLES, elem_cnt, GL_UNSIGNED_SHORT, NULL);

	glBindTextureUnit(2, 0);
	glBindVertexArray(0);
	shdr_pgm.UnUse();
}

void GLPbo::cleanup()
{
	glInvalidateBufferData(pboid);
	glDeleteBuffers(1, &pboid);
	glDeleteVertexArrays(1, &vaoid);
	glDeleteTextures(1, &texid);
}