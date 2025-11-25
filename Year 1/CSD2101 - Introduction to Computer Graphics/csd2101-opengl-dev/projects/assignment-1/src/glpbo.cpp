/*!
@file       glpbo.cpp
@author     pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date       05/07/2024

This file simulates GPU rasterizer.
To switch between DPML and my parser, press O

*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glpbo.h>
#include <glhelper.h>

// file globals
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

// flags
GLboolean rotation_flag{ GL_FALSE };
GLboolean painter_flag{ GL_FALSE };
GLboolean my_parser_flag{ GL_FALSE };

// model container
std::vector<GLPbo::Model> GLPbo::models;

// model id to keep track of which model to render
GLint model_id{};

// random number generator
std::random_device rd;
std::default_random_engine gen(rd());

// painter mode
GLPbo::Drawing GLPbo::draw;

std::string my_assignment_1_vs
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

std::string my_assignment_1_fs
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

GLfloat map_range(GLfloat x, GLfloat in_min, GLfloat in_max, GLfloat out_min, GLfloat out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

GLfloat triangle_area(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
{
	return 0.5f * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y));
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

void GLPbo::set_pixel(GLint x, GLint y, Color clr)
{
	if (x >= 0 && x < width && y >= 0 && y < height)
	{
		ptr_to_pbo[x + width * y] = clr;
	}
}

void GLPbo::viewport_xform()
{
	glm::mat3 ndc_to_viewporrt_xform
	{
		width / 2.f, 0.f, 0.f,
		0.f, height / 2.f, 0.f,
		width / 2.f, height / 2.f, 1.f
	};

	for (Model& mdl : models)
	{
		glm::mat3 rot
		{
			cosf(glm::radians(mdl.angle_disp)), sinf(glm::radians(mdl.angle_disp)), 0.f,
			-sinf(glm::radians(mdl.angle_disp)), cosf(glm::radians(mdl.angle_disp)), 0.f,
			0.f, 0.f, 1.f
		};

		for (size_t i{}; i < mdl.pm.size(); ++i)
		{
			mdl.pm[i].z = 1.f;
			mdl.pd[i] = ndc_to_viewporrt_xform * rot * mdl.pm[i];
			mdl.pd[i].z = 0.f;
		}
	}
}

void GLPbo::init(GLsizei w, GLsizei h)
{
	width = w;
	height = h;
	pixel_cnt = width * height;
	byte_cnt = pixel_cnt * 4;
	set_clear_color(255, 255, 255);

	glEnable(GL_SCISSOR_TEST);

	// create texture object
	glCreateTextures(GL_TEXTURE_2D, 1, &texid);
	glTextureStorage2D(texid, 1, GL_RGBA8, width, height);

	// create pixel buffer object
	glCreateBuffers(1, &pboid);
	glNamedBufferStorage(pboid, byte_cnt, nullptr, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT);

	setup_quad_vao();
	setup_shdrpgm();

	int id{};
	std::string model_name;
	std::ifstream scene_file{"../scenes/ass-1.scn", std::ios::in };

	while (scene_file >> model_name)
	{
		models.emplace_back(Model());
		if (!MyObjParser::parse_obj_mesh("../meshes/" + model_name + ".obj", models[id].pm, models[id].nml, models[id].tex, models[id].tri, true, true))
		{
			std::cout << "Failed to open mesh file" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		++id;
	}

	// randomly generate colors for every model triangle edges and filled triangles
	// map vertex normals from range [-1, 1] to [0, 1]
	std::uniform_int_distribution<> urdi(0, 255);
	std::uniform_real_distribution<GLfloat> urdf(0.0f, std::nextafter(1.0f, std::numeric_limits<GLfloat>::max()));

	for (Model &mdl : models)
	{
		for (size_t i{}; i < mdl.tri.size() / 3; ++i)
		{
			mdl.edge_clr.emplace_back(Color(urdi(gen), urdi(gen), urdi(gen)));
			mdl.tri_clr.emplace_back(glm::vec3(urdf(gen), urdf(gen), urdf(gen)));
		}

		// set pd size same as pm for each model
		for (size_t i{}; i < mdl.pm.size(); ++i)
		{
			mdl.pd.emplace_back(glm::vec3());
		}

		for (glm::vec3 &vn : mdl.nml)
		{
			vn = glm::vec3(map_range(vn.x, -1.f, 1.f, 0.f, 1.f), 
				map_range(vn.y, -1.f, 1.f, 0.f, 1.f), 
				map_range(vn.z, -1.f, 1.f, 0.f, 1.f));
		}
	}

	// transform ndc position coordinates to viewport coordinates
	viewport_xform();
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
	if (!shdr_pgm.CompileShaderFromString(GL_VERTEX_SHADER, my_assignment_1_vs))
	{
		std::cout << "Vertex shader failed to compile: ";
		std::cout << shdr_pgm.GetLog() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (!shdr_pgm.CompileShaderFromString(GL_FRAGMENT_SHADER, my_assignment_1_fs))
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
	GLint culled{};
	const GLfloat angle_speed{ 75.f };

	// user inputs
	if (GLHelper::keystateM)
	{
		model_id == 4 ? model_id = 0 : ++model_id;
		rotation_flag = GL_FALSE;
		GLHelper::keystateM = GL_FALSE;
	}

	if (GLHelper::keystateW)
	{
		switch (models[model_id].render_mode)
		{
		case Model::RenderMode::WIREFRAME:
			models[model_id].render_mode = Model::RenderMode::WIREFRAME_COLOR;
			break;
		case Model::RenderMode::WIREFRAME_COLOR:
			models[model_id].render_mode = Model::RenderMode::FACETED;
			break;
		case Model::RenderMode::FACETED:
			models[model_id].render_mode = Model::RenderMode::SHADED;
			break;
		default:
			models[model_id].render_mode = Model::RenderMode::WIREFRAME;
		}
		GLHelper::keystateW = GL_FALSE;
	}

	if (GLHelper::keystateR)
	{
		rotation_flag = ~rotation_flag;
		GLHelper::keystateR = GL_FALSE;
	}

	if (GLHelper::keystateP)
	{
		painter_flag = ~painter_flag;
		if (painter_flag)
		{
			// set pbo buffer to yellow
			set_clear_color(255, 255, 0); 
			clear_color_buffer();
		}
		else
		{
			// reset pbo buffer back to white
			set_clear_color(255, 255, 255);
		}
		GLHelper::keystateP = GL_FALSE;
	}

	if (GLHelper::keystateO)
	{
		my_parser_flag = ~my_parser_flag;

		// clear all model positions, vertex normals, texture coords and indices parsed by previous parser
		for (auto& mdl : models)
		{
			mdl.pm.clear();
			mdl.nml.clear();
			mdl.tex.clear();
			mdl.tri.clear();
			mdl.pd.clear();
		}

		int id{};
		std::string model_name;
		std::ifstream scene_file{ "../scenes/ass-1.scn", std::ios::in };

		while (scene_file >> model_name)
		{
			if (my_parser_flag)
			{
				if (!MyObjParser::parse_obj_mesh("../meshes/" + model_name + ".obj", models[id].pm, models[id].nml, models[id].tex, models[id].tri, true, true))
				{
					std::cout << "Failed to open mesh file" << std::endl;
					std::exit(EXIT_FAILURE);
				}
			}
			else
			{
				if (!DPML::parse_obj_mesh("../meshes/" + model_name + ".obj", models[id].pm, models[id].nml, models[id].tex, models[id].tri, true, true))
				{
					std::cout << "Failed to open mesh file" << std::endl;
					std::exit(EXIT_FAILURE);
				}
			}
			++id;
		}

		// re-map vertex normals and resize viewport coordinates container
		for (Model& mdl : models)
		{
			mdl.pd.resize(mdl.pm.size());
			for (glm::vec3& vn : mdl.nml)
			{
				vn = glm::vec3(map_range(vn.x, -1.f, 1.f, 0.f, 1.f),
					map_range(vn.y, -1.f, 1.f, 0.f, 1.f),
					map_range(vn.z, -1.f, 1.f, 0.f, 1.f));
			}
		}

		// transform new ndc position coordinates to viewport coordinates
		viewport_xform();

		GLHelper::keystateO = GL_FALSE;
	}
	
	// rotate current model anti-clockwise
	if (rotation_flag)
	{
		models[model_id].angle_disp += angle_speed * static_cast<GLfloat>(GLHelper::delta_time);\

		// update viewport coordinates 
		viewport_xform();
	}

	

	// rendering 
	ptr_to_pbo = static_cast<Color*>(glMapNamedBuffer(pboid, GL_WRITE_ONLY));
	
	// painter mode
	if (painter_flag)
	{
		painter_mode();
	}
	// static mode
	else
	{
		clear_color_buffer();
		GLfloat signed_area{};
		
		switch (models[model_id].render_mode)
		{
		case Model::RenderMode::FACETED:
			for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
			{
				if (!render_triangle(models[model_id].pd[models[model_id].tri[i]],
					models[model_id].pd[models[model_id].tri[i + 1]],
					models[model_id].pd[models[model_id].tri[i + 2]],
					models[model_id].tri_clr[i / 3]))
				{
					++culled;
				}
			}
			break;
		case Model::RenderMode::SHADED:
			for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
			{
				if (!render_triangle(models[model_id].pd[models[model_id].tri[i]],
					models[model_id].pd[models[model_id].tri[i + 1]],
					models[model_id].pd[models[model_id].tri[i + 2]],
					models[model_id].nml[models[model_id].tri[i]],
					models[model_id].nml[models[model_id].tri[i + 1]],
					models[model_id].nml[models[model_id].tri[i + 2]]))
				{
					++culled;
				}
			}
			break;
		default:
			for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
			{
				signed_area = triangle_area(models[model_id].pd[models[model_id].tri[i]],
					models[model_id].pd[models[model_id].tri[i + 1]],
					models[model_id].pd[models[model_id].tri[i + 2]]);

				if (signed_area > 0.f)
				{
					switch (models[model_id].render_mode)
					{
					case Model::RenderMode::WIREFRAME:
						render_linebresenham(static_cast<GLint>(models[model_id].pd[models[model_id].tri[i]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i]].y),
							static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 1]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 1]].y), Color(0, 0, 0));
						render_linebresenham(static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 1]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 1]].y),
							static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 2]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 2]].y), Color(0, 0, 0));
						render_linebresenham(static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 2]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 2]].y),
							static_cast<GLint>(models[model_id].pd[models[model_id].tri[i]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i]].y), Color(0, 0, 0));
						break;
					case Model::RenderMode::WIREFRAME_COLOR:
						render_linebresenham(static_cast<GLint>(models[model_id].pd[models[model_id].tri[i]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i]].y),
							static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 1]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 1]].y), models[model_id].edge_clr[i / 3]);
						render_linebresenham(static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 1]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 1]].y),
							static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 2]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 2]].y), models[model_id].edge_clr[i / 3]);
						render_linebresenham(static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 2]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i + 2]].y),
							static_cast<GLint>(models[model_id].pd[models[model_id].tri[i]].x), static_cast<GLint>(models[model_id].pd[models[model_id].tri[i]].y), models[model_id].edge_clr[i / 3]);
						break;
					}
				}
				else
				{
					++culled;
				}
			}
		}
	}
	
	glUnmapNamedBuffer(pboid);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboid);

	// copy image data from client memory to GPU texture buffer memory
	glTextureSubImage2D(texid, 0, 0, 0, width, height,
		GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// update window title
	std::stringstream sstr;
	std::string mode, model_name, parser;

	// update render mode name
	switch (models[model_id].render_mode)
	{
	case Model::RenderMode::WIREFRAME_COLOR:
		mode = "Wireframe Color";
		break;
	case Model::RenderMode::FACETED:
		mode = "Faceted";
		break;
	case Model::RenderMode::SHADED:
		mode = "Shaded";
		break;
	default:
		mode = "Wireframe";
	}

	// update model name
	switch (model_id)
	{
	case 1:
		model_name = "Suzzane";
		break;
	case 2:
		model_name = "Ogre";
		break;
	case 3:
		model_name = "Head";
		break;
	case 4:
		model_name = "Teapot";
		break;
	default:
		model_name = "Cube";
	}

	// update parser name
	parser = my_parser_flag ? "MyObjParser" : "DPML";
	
	if (painter_flag)
	{
		sstr << "Assignment 1 | Bryan Ang Wei Ze" <<
			" | Mode: Painter" <<
			" | Parser: " << parser <<
			" | FPS: " << std::fixed << std::setprecision(2) << GLHelper::fps;
	}
	else
	{
		sstr << "Assignment 1 | Bryan Ang Wei Ze" <<
			" | Mode: " << mode <<
			" | Model: " << model_name <<
			" | Vtx: " << models[model_id].pm.size() <<
			" | Tri: " << models[model_id].tri.size() / 3 <<
			" | Culled: " << culled <<
			" | Parser: " << parser <<
			" | FPS: " << std::fixed << std::setprecision(2) << GLHelper::fps;
	}
	
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

// line rasterizer using bresenham algorithm
void GLPbo::render_linebresenham(GLint px0, GLint py0, 
	GLint px1, GLint py1, const Color& draw_clr)
{
	glScissor(0, 0, width, height);
	GLint dx = px1 - px0, dy = py1 - py0;

	// octant 0
	if (dx > 0 && dy > 0 && abs(dx) >= abs(dy))
	{
		GLint d = 2 * dy - dx, de = 2 * dy, dne = 2 * dy - 2 * dx;
		set_pixel(px0, py0, draw_clr);
		while (--dx)
		{
			if (d > 0)
			{
				d += dne;
				++py0;
			}
			else
			{
				d += de;
			}
			++px0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// octant 1
	else if (dx > 0 && dy > 0 && abs(dx) < abs(dy))
	{
		GLint d = 2 * dx - dy, dn = 2 * dx, dne = 2 * dx - 2 * dy;
		set_pixel(px0, py0, draw_clr);
		while (--dy)
		{
			if (d > 0)
			{
				d += dne;
				++px0;
			}
			else
			{
				d += dn;
			}
			++py0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// octant 2
	else if (dx < 0 && dy > 0 && abs(dx) < abs(dy))
	{
		GLint d = -2 * dx - dy, dn = -2 * dx, dnw = -2 * dx - 2 * dy;
		set_pixel(px0, py0, draw_clr);
		while (--dy)
		{
			if (d > 0)
			{
				d += dnw;
				--px0;
			}
			else
			{
				d += dn;
			}
			++py0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// octant 3
	else if (dx < 0 && dy > 0 && abs(dx) >= abs(dy))
	{
		GLint d = -2 * dy - dx, dw = -2 * dy, dnw = -2 * dy - 2 * dx;
		set_pixel(px0, py0, draw_clr);
		while (++dx)
		{
			if (d < 0)
			{
				d += dnw;
				++py0;
			}
			else
			{
				d += dw;
			}
			--px0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// octant 4
	else if (dx < 0 && dy < 0 && abs(dx) >= abs(dy))
	{
		GLint d = 2 * dy - dx, dw = 2 * dy, dsw = 2 * dy - 2 * dx;
		set_pixel(px0, py0, draw_clr);
		while (++dx)
		{
			if (d < 0)
			{
				d += dsw;
				--py0;
			}
			else
			{
				d += dw;
			}
			--px0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// octant 5
	else if (dx < 0 && dy < 0 && abs(dx) < abs(dy))
	{
		GLint d = 2 * dx - dy, ds = 2 * dx, dsw = 2 * dx - 2 * dy;
		set_pixel(px0, py0, draw_clr);
		while (++dy)
		{
			if (d < 0)
			{
				d += dsw;
				--px0;
			}
			else
			{
				d += ds;
			}
			--py0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// octant 6
	else if (dx > 0 && dy < 0 && abs(dx) < abs(dy))
	{
		GLint d = -2 * dx - dy, ds = -2 * dx, dse = -2 * dx - 2 * dy;
		set_pixel(px0, py0, draw_clr);
		while (++dy)
		{
			if (d < 0)
			{
				d += dse;
				++px0;
			}
			else
			{
				d += ds;
			}
			--py0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// octant 7
	else if (dx > 0 && dy < 0 && abs(dx) >= abs(dy))
	{
		GLint d = -2 * dy - dx, de = -2 * dy, dse = -2 * dy - 2 * dx;
		set_pixel(px0, py0, draw_clr);
		while (--dx)
		{
			if (d > 0)
			{
				d += dse;
				--py0;
			}
			else
			{
				d += de;
			}
			++px0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// vertical lines going up
	else if (dx == 0 && dy > 0)
	{
		set_pixel(px0, py0, draw_clr);
		while (--dy)
		{
			++py0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// vertical lines going down
	else if (dx == 0 && dy < 0)
	{
		set_pixel(px0, py0, draw_clr);
		while (++dy)
		{
			--py0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// horizontal lines going left
	else if (dx < 0 && dy == 0)
	{
		set_pixel(px0, py0, draw_clr);
		while (++dx)
		{
			--px0;
			set_pixel(px0, py0, draw_clr);
		}
	}
	// horizontal lines going right
	else if (dx > 0 && dy == 0)
	{
		set_pixel(px0, py0, draw_clr);
		while (--dx)
		{
			++px0;
			set_pixel(px0, py0, draw_clr);
		}
	}
}

// flat shaded triangle rasterizer
bool GLPbo::render_triangle(glm::vec3 const& p0, glm::vec3 const& p1, 
	glm::vec3 const& p2, glm::vec3 const& clr)
{
	GLfloat signed_area{ triangle_area(p0, p1, p2) };
	
	if (signed_area <= 0.f)
	{
		return false;
	}
	
	// edge equations
	glm::vec3 Edge0{ p1.y - p2.y, p2.x - p1.x, p1.x * p2.y - p2.x * p1.y },
		Edge1{ p2.y - p0.y, p0.x - p2.x, p2.x * p0.y - p0.x * p2.y },
		Edge2{ p0.y - p1.y, p1.x - p0.x, p0.x * p1.y - p1.x * p0.y };

	// calculate bounding box of triangle
	GLint x_min{ static_cast<GLint>(std::min({p0.x, p1.x, p2.x})) },
		x_max{ static_cast<GLint>(std::ceil(std::max({p0.x, p1.x, p2.x}))) },
		y_min{ static_cast<GLint>(std::min({p0.y, p1.y, p2.y})) },
		y_max{ static_cast<GLint>(std::ceil(std::max({p0.y, p1.y, p2.y}))) };

	// pre compute top-left boolean constants for each edge
	bool Edge0_tl{ Edge0.x > 0.f || (Edge0.x == 0.f && Edge0.y < 0.f) ? true : false },
		Edge1_tl{ Edge1.x > 0.f || (Edge1.x == 0.f && Edge1.y < 0.f) ? true : false },
		Edge2_tl{ Edge2.x > 0.f || (Edge2.x == 0.f && Edge2.y < 0.f) ? true : false };

	// evaluate initial pixel
	const GLfloat init_Eval0{ Edge0.x * (x_min + 0.5f) + Edge0.y * (y_min + 0.5f) + Edge0.z },
		init_Eval1{ Edge1.x * (x_min + 0.5f) + Edge1.y * (y_min + 0.5f) + Edge1.z },
		init_Eval2{ Edge2.x * (x_min + 0.5f) + Edge2.y * (y_min + 0.5f) + Edge2.z };

	// start values for vertical spans
	GLfloat Eval0{ init_Eval0 },
		Eval1{ init_Eval1 },
		Eval2{ init_Eval2 };

	for (GLint y{ y_min }; y < y_max; ++y)
	{
		// start values for horizontal spans
		GLfloat hEval0{ Eval0 }, hEval1{ Eval1 }, hEval2{ Eval2 };

		for (GLint x{ x_min }; x < x_max; ++x)
		{
			// centre of current frag is inside triangle or on a top-left edge
			if ((hEval0 > 0.f || (hEval0 == 0.f && Edge0_tl)) &&
				(hEval1 > 0.f || (hEval1 == 0.f && Edge1_tl)) &&
				(hEval2 > 0.f || (hEval2 == 0.f && Edge2_tl)))
			{
				set_pixel(x, y, Color(static_cast<GLubyte>(clr.r * 255), 
					static_cast<GLubyte>(clr.g * 255), 
					static_cast<GLubyte>(clr.b * 255)));
			}

			// incrementally update hEvals
			hEval0 = Eval0 + ((x - x_min + 1) * Edge0.x);
			hEval1 = Eval1 + ((x - x_min + 1) * Edge1.x);
			hEval2 = Eval2 + ((x - x_min + 1) * Edge2.x);
		}

		// incrementally update Evals
		Eval0 = init_Eval0 + ((y - y_min + 1) * Edge0.y);
		Eval1 = init_Eval1 + ((y - y_min + 1) * Edge1.y);
		Eval2 = init_Eval2 + ((y - y_min + 1) * Edge2.y);
	}

	return true;
}

// smooth shaded triangle rasterizer
bool GLPbo::render_triangle(glm::vec3 const& p0, glm::vec3 const& p1,
	glm::vec3 const& p2, glm::vec3 const& c0,
	glm::vec3 const& c1, glm::vec3 const& c2)
{
	GLfloat signed_area{ triangle_area(p0, p1, p2) };

	if (signed_area <= 0.f)
	{
		return false;
	}

	// edge equations
	glm::vec3 Edge0{ p1.y - p2.y, p2.x - p1.x, p1.x * p2.y - p2.x * p1.y },
		Edge1{ p2.y - p0.y, p0.x - p2.x, p2.x * p0.y - p0.x * p2.y },
		Edge2{ p0.y - p1.y, p1.x - p0.x, p0.x * p1.y - p1.x * p0.y };

	// calculate bounding box of triangle
	GLint x_min{ static_cast<GLint>(std::min({p0.x, p1.x, p2.x})) },
		x_max{ static_cast<GLint>(std::ceil(std::max({p0.x, p1.x, p2.x}))) },
		y_min{ static_cast<GLint>(std::min({p0.y, p1.y, p2.y})) },
		y_max{ static_cast<GLint>(std::ceil(std::max({p0.y, p1.y, p2.y}))) };

	// pre compute top-left boolean constants for each edge
	bool Edge0_tl{ Edge0.x > 0.f || (Edge0.x == 0.f && Edge0.y < 0.f) ? true : false },
		Edge1_tl{ Edge1.x > 0.f || (Edge1.x == 0.f && Edge1.y < 0.f) ? true : false },
		Edge2_tl{ Edge2.x > 0.f || (Edge2.x == 0.f && Edge2.y < 0.f) ? true : false };

	// evaluate initial pixel
	const GLfloat init_Eval0{ Edge0.x * (x_min + 0.5f) + Edge0.y * (y_min + 0.5f) + Edge0.z },
		init_Eval1{ Edge1.x * (x_min + 0.5f) + Edge1.y * (y_min + 0.5f) + Edge1.z },
		init_Eval2{ Edge2.x * (x_min + 0.5f) + Edge2.y * (y_min + 0.5f) + Edge2.z };

	// start values for vertical spans
	GLfloat Eval0{ init_Eval0 },
		Eval1{ init_Eval1 },
		Eval2{ init_Eval2 };

	for (GLint y{ y_min }; y < y_max; ++y)
	{
		// start values for horizontal spans
		GLfloat hEval0{ Eval0 }, hEval1{ Eval1 }, hEval2{ Eval2 };
		
		for (GLint x{ x_min }; x < x_max; ++x)
		{
			if ((hEval0 > 0.f || (hEval0 == 0.f && Edge0_tl)) &&
				(hEval1 > 0.f || (hEval1 == 0.f && Edge1_tl)) &&
				(hEval2 > 0.f || (hEval2 == 0.f && Edge2_tl)))
			{
				// only compute barycentric coordinates if centre of current pixel is inside triangle or on top-left edges
				GLfloat w0{ triangle_area(p1, p2, glm::vec3(x + 0.5f, y + 0.5f, 0.f)) / signed_area },
					w1{ triangle_area(p2, p0, glm::vec3(x + 0.5f, y + 0.5f, 0.f)) / signed_area },
					w2{ 1.f - w0 - w1 };
				
				glm::vec3 interpolated_color{ w0 * c0 + w1 * c1 + w2 * c2 };

				set_pixel(x, y, Color(static_cast<GLubyte>(interpolated_color.r * 255), 
					static_cast<GLubyte>(interpolated_color.g * 255),
					static_cast<GLubyte>(interpolated_color.b * 255)));
			}

			// incrementally update hEvals
			hEval0 = Eval0 + ((x - x_min + 1) * Edge0.x);
			hEval1 = Eval1 + ((x - x_min + 1) * Edge1.x);
			hEval2 = Eval2 + ((x - x_min + 1) * Edge2.x);
		}

		// incrementally update Evals
		Eval0 = init_Eval0 + ((y - y_min + 1) * Edge0.y);
		Eval1 = init_Eval1 + ((y - y_min + 1) * Edge1.y);
		Eval2 = init_Eval2 + ((y - y_min + 1) * Edge2.y);
	}

	return true;
}

// painter mode
void GLPbo::painter_mode()
{
	if (GLHelper::mousestateLeft)
	{
		// set end point
		double xpos, ypos;
		glfwGetCursorPos(GLHelper::ptr_window, &xpos, &ypos);
		draw.x1 = static_cast<GLint>(xpos);
		draw.y1 = GLHelper::height - static_cast<GLint>(ypos) - 1;

		// render line
		GLPbo::render_linebresenham(GLPbo::draw.x0, GLPbo::draw.y0,
			GLPbo::draw.x1, GLPbo::draw.y1, GLPbo::Color(255, 0, 0)); 
		
		// update start point
		draw.x0 = draw.x1;
		draw.y0 = draw.y1;
	}
}