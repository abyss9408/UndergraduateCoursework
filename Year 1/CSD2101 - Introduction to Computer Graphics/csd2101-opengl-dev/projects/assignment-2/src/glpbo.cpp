/*!
@file       glpbo.cpp
@author     pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date       11/07/2024

This file simulates GPU rasterizer.

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
std::vector<GLfloat> GLPbo::depthBuffer;
GLuint GLPbo::vaoid;
GLuint GLPbo::elem_cnt;
GLuint GLPbo::pboid;
GLuint GLPbo::texid;
GLSLShader GLPbo::shdr_pgm;
GLPbo::Color GLPbo::clear_clr;
GLPbo::PointLight GLPbo::light_source;
GLPbo::Texture GLPbo::ogre_texture;

namespace CORE10
{
	const GLfloat angle_speed{ 60.f };
	const GLfloat light_source_angle_speed{ 0.2f };
	const std::string model_file_name[2]{ "ogre_ptn.obj", "cube_ptn.obj" };
	const std::string texture_file_name{ "ogre.tex" };
	const glm::vec3 camera_pos{ 0.f,0.f,10.f };
	const glm::vec3 camera_target{ 0.f,0.f,0.f };
	const GLfloat plane_b{ -1.5f };
	const GLfloat plane_t{ 1.5f };
	const GLfloat plane_n{ 8.f };
	const GLfloat plane_f{ 12.f };
	const glm::vec3 light_pos{ 0.f,0.f,10.f };
	const glm::vec3 light_intensity{ 1.f,1.f,1.f };
}

// model container
std::vector<GLPbo::Model> GLPbo::models;

// model id to keep track of which model to render
GLint model_id{};

// flags
GLboolean rotation_y_axisflag{ GL_FALSE };
GLboolean rotation_1_1_0_flag{ GL_FALSE };
GLboolean rotation_1_1_1_flag{ GL_FALSE };
GLboolean light_source_rotation_flag{ GL_FALSE };

std::string my_assignment_2_vs
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

std::string my_assignment_2_fs
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

glm::vec3 calculate_lighting_faceted(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& light_pos, const glm::vec3& intensity)
{
	// calculate triangle centeroid
	glm::vec3 centeroid{ (p0 + p1 + p2) / 3.0f };

	// calculate normalized centeroid to light source pos vector
	glm::vec3 lightDir = glm::normalize(light_pos - centeroid);

	// calculate normalized triangle face normal
	glm::vec3 normal = glm::normalize(glm::cross(
		p1 - p0,
		p2 - p0
	));

	// calculate the diffuse component
	float diff = std::max(glm::dot(normal, lightDir), 0.0f);

	// combine the color based on light properties
	return diff * intensity;
}

glm::vec3 calculate_lighting_smooth(const glm::vec3& pos, const glm::vec3& nml, const glm::vec3& light_pos, const glm::vec3& intensity)
{
	// calculate normalized vertex position to light source pos vector
	glm::vec3 lightDir = glm::normalize(light_pos - pos);

	// calculate the diffuse component
	float diff = std::max(glm::dot(nml, lightDir), 0.0f);

	// combine the color based on light properties
	return diff * intensity;
}

GLfloat map_range(GLfloat x, GLfloat in_min, GLfloat in_max, GLfloat out_min, GLfloat out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

GLfloat triangle_area(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
{
	return 0.5f * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y));
}

glm::vec3 calculate_edge(glm::vec3 const& p1, glm::vec3 const& p2)
{
	return glm::vec3{ p1.y - p2.y, p2.x - p1.x, p1.x * p2.y - p2.x * p1.y };
}

void calculate_bounding_box(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2,
	GLint& x_min, GLint& x_max, GLint& y_min, GLint& y_max)
{
	x_min = static_cast<GLint>(std::min({ p0.x, p1.x, p2.x }));
	x_max = static_cast<GLint>(std::ceil(std::max({ p0.x, p1.x, p2.x })));
	y_min = static_cast<GLint>(std::min({ p0.y, p1.y, p2.y }));
	y_max = static_cast<GLint>(std::ceil(std::max({ p0.y, p1.y, p2.y })));
}

bool is_top_left(glm::vec3 const& edge)
{
	return edge.x > 0.f || (edge.x == 0.f && edge.y < 0.f);
}

GLfloat initial_evaluation(glm::vec3 const& edge, GLint x, GLint y)
{
	return edge.x * (x + 0.5f) + edge.y * (y + 0.5f) + edge.z;
}

// frame buffer functions
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

void GLPbo::set_pixel_depth(GLint x, GLint y, GLfloat depth)
{
	if (x >= 0 && x < width && y >= 0 && y < height)
	{
		depthBuffer[x + width * y] = std::min(depthBuffer[x + width * y], depth);
	}
}

void GLPbo::model_to_viewport_xform()
{
	glm::mat4 model_scale
	{
		2.f, 0.f, 0.f, 0.f,
		0.f, 2.f, 0.f, 0.f,
		0.f, 0.f, 2.f, 0.f,
		0.f, 0.f, 0.f, 1.f
	};

	glm::mat4 view_xform
	{
		glm::lookAt(CORE10::camera_pos, CORE10::camera_target, glm::vec3(0.f,1.f,0.f))
	};

	glm::mat4 projection_xform
	{
		glm::ortho(static_cast<GLfloat>(CORE10::plane_b * (width / height)), 
			static_cast<GLfloat>(CORE10::plane_t * (width / height)),
			CORE10::plane_b, 
			CORE10::plane_t, 
			CORE10::plane_n, 
			CORE10::plane_f)
	};

	glm::mat4 viewport_transform
	{
		static_cast<float>(width) / 2, 0.f, 0.f, 0.f,
		0.f, static_cast<float>(height) / 2, 0.f, 0.f,
		0.f, 0.f, 0.5f, 0.f,
		static_cast<float>(width) / 2, static_cast<float>(height) / 2, 0.5f, 1.f
	};

	glm::mat4 world_to_viewport_xform
	{
		viewport_transform * projection_xform * view_xform
	};

	for (Model& mdl : models)
	{
		for (size_t i{}; i < mdl.pos_model.size(); ++i)
		{
			GLfloat disp_rad{ glm::radians(mdl.angle_disp) };
			glm::mat4 model_rot{ 1.f };
			
			// rotate about vector <1, 1, 0>
			if (rotation_1_1_0_flag)
			{
				model_rot = glm::rotate(model_rot, disp_rad, glm::vec3(1.f, 1.f, 0.f));
			}
			// rotate about vector <1, 1, 1>
			else if (rotation_1_1_1_flag)
			{
				model_rot = glm::rotate(model_rot, disp_rad, glm::vec3(1.f, 1.f, 1.f));
			}
			// rotate about y-axis
			else
			{
				model_rot =
				{
					cosf(disp_rad), 0.f, -sinf(disp_rad), 0.f,
					0.f, 1.f, 0.f, 0.f,
					sinf(disp_rad), 0.f, cosf(disp_rad), 0.f,
					0.f, 0.f, 0.f, 1.f
				};
			}

			glm::mat4 model_xform
			{
				model_rot * model_scale
			};

			mdl.pos_world[i] = model_xform * glm::vec4(mdl.pos_model[i], 1.0f);
			mdl.nml_xform[i] = model_rot * glm::vec4(mdl.nml[i], 1.0f);
			mdl.pos_viewport[i] = world_to_viewport_xform * glm::vec4(mdl.pos_world[i], 1.0f);
		}
	}
}

// GLPbo member functions
void GLPbo::init(GLsizei w, GLsizei h)
{
	width = w;
	height = h;
	pixel_cnt = width * height;
	byte_cnt = pixel_cnt * 4;
	set_clear_color(0, 0, 0);

	light_source.position = CORE10::light_pos;
	light_source.intensity = CORE10::light_intensity;

	// create texture object
	glCreateTextures(GL_TEXTURE_2D, 1, &texid);
	glTextureStorage2D(texid, 1, GL_RGBA8, width, height);

	// create pixel buffer object
	glCreateBuffers(1, &pboid);
	glNamedBufferStorage(pboid, byte_cnt, nullptr, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT);
	setup_quad_vao();
	setup_shdrpgm();

	// read meshes data
	models.resize(2);
	if (!DPML::parse_obj_mesh("../meshes/" + CORE10::model_file_name[0], models[0].pos_model, models[0].nml, models[0].tex, models[0].tri, true, true))
	{
		std::cout << "Failed to open mesh file " + CORE10::model_file_name[0] << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (!DPML::parse_obj_mesh("../meshes/" + CORE10::model_file_name[1], models[1].pos_model, models[1].nml, models[1].tex, models[1].tri, true, true))
	{
		std::cout << "Failed to open mesh file " + CORE10::model_file_name[1] << std::endl;
		std::exit(EXIT_FAILURE);
	}

	models[1].angle_disp = 30.f;

	for (Model& mdl : models)
	{
		mdl.pos_world.resize(mdl.pos_model.size(), glm::vec3());
		mdl.nml_xform.resize(mdl.nml.size(), glm::vec3());
		mdl.pos_viewport.resize(mdl.pos_model.size(), glm::vec3());
	}

	// transform model coordinates to viewport coordinates
	model_to_viewport_xform();

	// load texture from file
	std::ifstream file("../images/ogre.tex", std::ios::binary);
	if (!file.is_open())
	{
		std::cerr << "Failed to open texture file: " << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// read width, height and bytes per texel from header
	file.read(reinterpret_cast<char*>(&ogre_texture.width), sizeof(int));
	file.read(reinterpret_cast<char*>(&ogre_texture.height), sizeof(int));
	file.read(reinterpret_cast<char*>(&ogre_texture.bytes_per_texel), sizeof(int));

	GLint data_size{ ogre_texture.width * ogre_texture.height * ogre_texture.bytes_per_texel };
	ogre_texture.texels.resize(data_size);

	// read texture data
	file.read(reinterpret_cast<char*>(ogre_texture.texels.data()), data_size);

	file.close();
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
	if (!shdr_pgm.CompileShaderFromString(GL_VERTEX_SHADER, my_assignment_2_vs))
	{
		std::cout << "Vertex shader failed to compile: ";
		std::cout << shdr_pgm.GetLog() << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (!shdr_pgm.CompileShaderFromString(GL_FRAGMENT_SHADER, my_assignment_2_fs))
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
	depthBuffer.assign(pixel_cnt, std::numeric_limits<float>::infinity());

	// toggle between models
	if (GLHelper::keystateM)
	{
		model_id == 1 ? model_id = 0 : ++model_id;
		rotation_y_axisflag = GL_FALSE;
		rotation_1_1_0_flag = GL_FALSE;
		rotation_1_1_1_flag = GL_FALSE;
		GLHelper::keystateM = GL_FALSE;
	}

	// toggle between render modes for the current model
	if (GLHelper::keystateW)
	{
		switch (models[model_id].render_mode)
		{
		case Model::RenderMode::WIREFRAME:
			models[model_id].render_mode = Model::RenderMode::SHADOW_MAP;
			break;
		case Model::RenderMode::SHADOW_MAP:
			models[model_id].render_mode = Model::RenderMode::FACETED;
			break;
		case Model::RenderMode::FACETED:
			models[model_id].render_mode = Model::RenderMode::SHADED;
			break;
		case Model::RenderMode::SHADED:
			models[model_id].render_mode = Model::RenderMode::TEXTURED;
			break;
		case Model::RenderMode::TEXTURED:
			models[model_id].render_mode = Model::RenderMode::TEXTURED_FACETED;
			break;
		case Model::RenderMode::TEXTURED_FACETED:
			models[model_id].render_mode = Model::RenderMode::TEXTURED_SHADED;
			break;
		default:
			models[model_id].render_mode = Model::RenderMode::WIREFRAME;
		}
		GLHelper::keystateW = GL_FALSE;
	}

	// toggle model rotation about y-axis
	if (GLHelper::keystateR)
	{
		rotation_y_axisflag = ~rotation_y_axisflag;
		rotation_1_1_0_flag = GL_FALSE;
		rotation_1_1_1_flag = GL_FALSE;
		GLHelper::keystateR = GL_FALSE;
	}

	// toggle model rotation about vector <1, 1, 0>
	if (GLHelper::keystateX)
	{
		rotation_y_axisflag = GL_FALSE;
		rotation_1_1_0_flag = ~rotation_1_1_0_flag;
		rotation_1_1_1_flag = GL_FALSE;
		GLHelper::keystateX = GL_FALSE;
	}

	// toggle model rotation about vector <1, 1, 1>
	if (GLHelper::keystateZ)
	{
		rotation_y_axisflag = GL_FALSE;
		rotation_1_1_0_flag = GL_FALSE;
		rotation_1_1_1_flag = ~rotation_1_1_1_flag;
		GLHelper::keystateZ = GL_FALSE;
	}

	// toggle light source rotation about y-axis
	if (GLHelper::keystateL)
	{
		light_source_rotation_flag = ~light_source_rotation_flag;
		GLHelper::keystateL = GL_FALSE;
	}

	// rotate current model anti-clockwise about y axis, vector <1, 1, 0> or vector <1, 1, 1>
	if (rotation_y_axisflag || rotation_1_1_0_flag || rotation_1_1_1_flag)
	{
		models[model_id].angle_disp += CORE10::angle_speed * static_cast<GLfloat>(GLHelper::delta_time);

		// update viewport coordinates 
		model_to_viewport_xform();
	}

	// rotate light source anti-clockwise about y axis
	if (light_source_rotation_flag)
	{
		static GLfloat angle{ static_cast<GLfloat>(glfwGetTime()) * glm::radians(1.f) };

		// calculate rotation matrix
		glm::mat4 rot
		{
			cosf(angle), 0.f, -sinf(angle), 0.f,
			0.f, 1.f, 0.f, 0.f,
			sinf(angle), 0.f, cosf(angle), 0.f,
			0.f, 0.f, 0.f, 1.f
		};

		// update light source position
		light_source.position = glm::vec3(rot * glm::vec4(light_source.position, 1.f));
	}

	// rendering 
	ptr_to_pbo = static_cast<Color*>(glMapNamedBuffer(pboid, GL_WRITE_ONLY));
	clear_color_buffer();
	GLfloat signed_area{};

	switch (models[model_id].render_mode)
	{
	case Model::RenderMode::SHADOW_MAP:
		for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
		{
			if (!render_triangle_shadow_map(models[model_id].pos_viewport[models[model_id].tri[i]],
				models[model_id].pos_viewport[models[model_id].tri[i + 1]],
				models[model_id].pos_viewport[models[model_id].tri[i + 2]]))
			{
				++culled;
			}
		}
		break;
	case Model::RenderMode::FACETED:
		for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
		{
			// calculate lighting for current triangle
			glm::vec3 lighting{ calculate_lighting_faceted(models[model_id].pos_world[models[model_id].tri[i]],
				models[model_id].pos_world[models[model_id].tri[i + 1]],
				models[model_id].pos_world[models[model_id].tri[i + 2]], light_source.position, light_source.intensity) };

			if (!render_triangle_faceted(models[model_id].pos_viewport[models[model_id].tri[i]],
				models[model_id].pos_viewport[models[model_id].tri[i + 1]],
				models[model_id].pos_viewport[models[model_id].tri[i + 2]], lighting))
			{
				++culled;
			}
		}
		break;
	case Model::RenderMode::SHADED:
		for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
		{
			// calculate lighting for each vertex using the respective vertex normals
			glm::vec3 l0{ calculate_lighting_smooth(models[model_id].pos_world[models[model_id].tri[i]],
				models[model_id].nml_xform[models[model_id].tri[i]], light_source.position, light_source.intensity) };
			glm::vec3 l1{ calculate_lighting_smooth(models[model_id].pos_world[models[model_id].tri[i + 1]],
				models[model_id].nml_xform[models[model_id].tri[i + 1]], light_source.position, light_source.intensity) };
			glm::vec3 l2{ calculate_lighting_smooth(models[model_id].pos_world[models[model_id].tri[i + 2]],
				models[model_id].nml_xform[models[model_id].tri[i + 2]], light_source.position, light_source.intensity) };

			if (!render_triangle_smooth(models[model_id].pos_viewport[models[model_id].tri[i]],
				models[model_id].pos_viewport[models[model_id].tri[i + 1]],
				models[model_id].pos_viewport[models[model_id].tri[i + 2]], l0, l1, l2))
			{
				++culled;
			}
		}
		break;
	case Model::RenderMode::TEXTURED:
		for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
		{
			if (!render_triangle_texture_map(models[model_id].pos_viewport[models[model_id].tri[i]],
				models[model_id].pos_viewport[models[model_id].tri[i + 1]],
				models[model_id].pos_viewport[models[model_id].tri[i + 2]],
				models[model_id].tex[models[model_id].tri[i]],
				models[model_id].tex[models[model_id].tri[i + 1]],
				models[model_id].tex[models[model_id].tri[i + 2]]))
			{
				++culled;
			}
		}
		break;
	case Model::RenderMode::TEXTURED_FACETED:
		for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
		{
			// calculate lighting for current triangle
			glm::vec3 lighting{ calculate_lighting_faceted(models[model_id].pos_world[models[model_id].tri[i]],
				models[model_id].pos_world[models[model_id].tri[i + 1]],
				models[model_id].pos_world[models[model_id].tri[i + 2]], light_source.position, light_source.intensity) };

			if (!render_triangle_texture_faceted(models[model_id].pos_viewport[models[model_id].tri[i]],
				models[model_id].pos_viewport[models[model_id].tri[i + 1]],
				models[model_id].pos_viewport[models[model_id].tri[i + 2]],
				lighting,
				models[model_id].tex[models[model_id].tri[i]],
				models[model_id].tex[models[model_id].tri[i + 1]],
				models[model_id].tex[models[model_id].tri[i + 2]]))
			{
				++culled;
			}
		}
		break;
	case Model::RenderMode::TEXTURED_SHADED:
		for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
		{
			// calculate lighting for each vertex using the respective vertex normals
			glm::vec3 l0{ calculate_lighting_smooth(models[model_id].pos_world[models[model_id].tri[i]],
				models[model_id].nml_xform[models[model_id].tri[i]], light_source.position, light_source.intensity) };
			glm::vec3 l1{ calculate_lighting_smooth(models[model_id].pos_world[models[model_id].tri[i + 1]],
				models[model_id].nml_xform[models[model_id].tri[i + 1]], light_source.position, light_source.intensity) };
			glm::vec3 l2{ calculate_lighting_smooth(models[model_id].pos_world[models[model_id].tri[i + 2]],
				models[model_id].nml_xform[models[model_id].tri[i + 2]], light_source.position, light_source.intensity) };

			if (!render_triangle_texture_shaded(models[model_id].pos_viewport[models[model_id].tri[i]],
				models[model_id].pos_viewport[models[model_id].tri[i + 1]],
				models[model_id].pos_viewport[models[model_id].tri[i + 2]],
				l0, l1, l2,
				models[model_id].tex[models[model_id].tri[i]],
				models[model_id].tex[models[model_id].tri[i + 1]],
				models[model_id].tex[models[model_id].tri[i + 2]]))
			{
				++culled;
			}
		}
		break;
	default:
		for (size_t i = 0; i < models[model_id].tri.size(); i += 3)
		{
			signed_area = triangle_area(models[model_id].pos_viewport[models[model_id].tri[i]],
				models[model_id].pos_viewport[models[model_id].tri[i + 1]],
				models[model_id].pos_viewport[models[model_id].tri[i + 2]]);
			if (signed_area > 0.f)
			{
				render_linebresenham(static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i]].x), static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i]].y),
					static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i + 1]].x), static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i + 1]].y), Color(0, 0, 255));
				render_linebresenham(static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i + 1]].x), static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i + 1]].y),
					static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i + 2]].x), static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i + 2]].y), Color(0, 0, 255));
				render_linebresenham(static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i + 2]].x), static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i + 2]].y),
					static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i]].x), static_cast<GLint>(models[model_id].pos_viewport[models[model_id].tri[i]].y), Color(0, 0, 255));
			}
			else
			{
				++culled;
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
	std::string mode, model_name;

	// update render mode name
	switch (models[model_id].render_mode)
	{
	case Model::RenderMode::SHADOW_MAP:
		mode = "Shadow Map";
		break;
	case Model::RenderMode::FACETED:
		mode = "Faceted";
		break;
	case Model::RenderMode::SHADED:
		mode = "Shaded";
		break;
	case Model::RenderMode::TEXTURED:
		mode = "Textured";
		break;
	case Model::RenderMode::TEXTURED_FACETED:
		mode = "Textured/Faceted";
		break;
	case Model::RenderMode::TEXTURED_SHADED:
		mode = "Textured/Shaded";
		break;
	default:
		mode = "Wireframe";
	}

	// update model name
	model_name = model_id ? "Cube" : "ogre_ptn";

	sstr << "A2 | Bryan Ang Wei Ze" <<
		" | Model: " << model_name <<
		" | Mode: " << mode <<
		" | Vertices: " << models[model_id].pos_model.size() <<
		" | Triangles: " << models[model_id].tri.size() / 3 <<
		" | Culled: " << culled <<
		" | FPS: " << std::fixed << std::setprecision(2) << GLHelper::fps;

	glfwSetWindowTitle(GLHelper::ptr_window, sstr.str().c_str());
}

void GLPbo::draw_fullwindow_quad()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

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

// general triangle rasterizer
template <typename PixelFunc>
static bool GLPbo::render_triangle(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2, PixelFunc&& set_pixel_func)
{
	GLfloat signed_area{ triangle_area(p0, p1, p2) };

	if (signed_area <= 0.f)
	{
		return false;
	}

	glm::vec3 Edge0{ calculate_edge(p1, p2) },
		Edge1{ calculate_edge(p2, p0) },
		Edge2{ calculate_edge(p0, p1) };

	GLint x_min, x_max, y_min, y_max;
	calculate_bounding_box(p0, p1, p2, x_min, x_max, y_min, y_max);

	bool Edge0_tl{ is_top_left(Edge0) },
		Edge1_tl{ is_top_left(Edge1) },
		Edge2_tl{ is_top_left(Edge2) };

	// start values for vertical spans
	GLfloat Eval0{ initial_evaluation(Edge0, x_min, y_min) },
		Eval1{ initial_evaluation(Edge1, x_min, y_min) },
		Eval2{ initial_evaluation(Edge2, x_min, y_min) };

	for (GLint y{ y_min }; y < y_max; ++y)
	{
		// start values for horizontal spans
		GLfloat hEval0{ Eval0 }, hEval1{ Eval1 }, hEval2{ Eval2 };

		for (GLint x{ x_min }; x < x_max; ++x)
		{
			// centre of current fragment is inside triangle or on a top-left edge
			if ((hEval0 > 0.f || (hEval0 == 0.f && Edge0_tl)) &&
				(hEval1 > 0.f || (hEval1 == 0.f && Edge1_tl)) &&
				(hEval2 > 0.f || (hEval2 == 0.f && Edge2_tl)))
			{
				// only compute barycentric coordinates if centre of current pixel is inside triangle or on top-left edges
				GLfloat w0{ triangle_area(p1, p2, glm::vec3(x + 0.5f, y + 0.5f, 0.f)) / signed_area },
					w1{ triangle_area(p2, p0, glm::vec3(x + 0.5f, y + 0.5f, 0.f)) / signed_area },
					w2{ 1.f - w0 - w1 };

				GLfloat depth{ w0 * p0.z + w1 * p1.z + w2 * p2.z };
				if (x >= 0 && x < width && y >= 0 && y < height)
				{
					if (depth < depthBuffer[x + width * y])
					{
						set_pixel_func(x, y, depth, w0, w1, w2);
					}
				}
			}

			// incrementally update hEvals
			hEval0 += Edge0.x;
			hEval1 += Edge1.x;
			hEval2 += Edge2.x;
		}

		// incrementally update Evals
		Eval0 += Edge0.y;
		Eval1 += Edge1.y;
		Eval2 += Edge2.y;
	}

	return true;
}

// shadow map triangle rasterizer
bool GLPbo::render_triangle_shadow_map(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2)
{
	auto set_pixel_func = [&](GLint x, GLint y, GLfloat depth, GLfloat, GLfloat, GLfloat) -> void
	{
		set_pixel(x, y, Color(static_cast<GLubyte>(depth * 255),
			static_cast<GLubyte>(depth * 255),
			static_cast<GLubyte>(depth * 255)));

		// update depth of current fragment in depth buffer
		set_pixel_depth(x, y, depth);
	};

	return render_triangle(p0, p1, p2, set_pixel_func);
}

// faceted shaded triangle rasterizer
bool GLPbo::render_triangle_faceted(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2,
	glm::vec3 const& lighting)
{
	auto set_pixel_func = [&](GLint x, GLint y, GLfloat depth, GLfloat, GLfloat, GLfloat) -> void
	{
		set_pixel(x, y, Color(static_cast<GLubyte>(lighting.r * 255),
			static_cast<GLubyte>(lighting.g * 255),
			static_cast<GLubyte>(lighting.b * 255)));

		// update depth of current fragment in depth buffer
		set_pixel_depth(x, y, depth);
	};

	return render_triangle(p0, p1, p2, set_pixel_func);
}

bool GLPbo::render_triangle_smooth(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2,
	glm::vec3 const& l0, glm::vec3 const& l1, glm::vec3 const& l2)
{
	auto set_pixel_func = [&](GLint x, GLint y, GLfloat depth, GLfloat w0, GLfloat w1, GLfloat w2) -> void
	{
		glm::vec3 smooth_lighting{ w0 * l0 + w1 * l1 + w2 * l2 };
		set_pixel(x, y, Color(static_cast<GLubyte>(smooth_lighting.r * 255),
			static_cast<GLubyte>(smooth_lighting.g * 255),
			static_cast<GLubyte>(smooth_lighting.b * 255)));

		// update depth of current fragment in depth buffer
		set_pixel_depth(x, y, depth);
	};

	return render_triangle(p0, p1, p2, set_pixel_func);
}

bool GLPbo::render_triangle_texture_map(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2,
	glm::vec2 const& t0, glm::vec2 const& t1, glm::vec2 const& t2)
{
	auto set_pixel_func = [&](GLint x, GLint y, GLfloat depth, GLfloat w0, GLfloat w1, GLfloat w2) -> void
	{
		// interpolate texture coords
		glm::vec2 interpolated_tex_coords{ w0 * t0 + w1 * t1 + w2 * t2 };

		// map interpolated texture coords to texel coords
		glm::ivec2 texel_coords
		{
			static_cast<GLint>(std::floor(ogre_texture.width * glm::clamp(interpolated_tex_coords.x, 0.f, 1.f)) - 0.5f),
			static_cast<GLint>(std::floor(ogre_texture.height * glm::clamp(interpolated_tex_coords.y, 0.f, 1.f)) - 0.5f)
		};

		// calculate index of texel mapped to
		GLint index = (texel_coords.y * ogre_texture.width + texel_coords.x) * ogre_texture.bytes_per_texel;
		set_pixel(x, y, Color(ogre_texture.texels[index],
			ogre_texture.texels[index + 1],
			ogre_texture.texels[index + 2]));

		// update depth of current fragment in depth buffer
		set_pixel_depth(x, y, depth);
	};

	return render_triangle(p0, p1, p2, set_pixel_func);
}

bool GLPbo::render_triangle_texture_faceted(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2,
	glm::vec3 const& lighting, glm::vec2 const& t0, glm::vec2 const& t1, glm::vec2 const& t2)
{
	auto set_pixel_func = [&](GLint x, GLint y, GLfloat depth, GLfloat w0, GLfloat w1, GLfloat w2) -> void
	{
		// interpolate texture coords
		glm::vec2 interpolated_tex_coords{ w0 * t0 + w1 * t1 + w2 * t2 };

		// map interpolated texture coords to texel coords
		glm::ivec2 texel_coords
		{
			static_cast<GLint>(std::floor(ogre_texture.width * glm::clamp(interpolated_tex_coords.x, 0.f, 1.f)) - 0.5f),
			static_cast<GLint>(std::floor(ogre_texture.height * glm::clamp(interpolated_tex_coords.y, 0.f, 1.f)) - 0.5f)
		};

		// calculate index of texel mapped to
		GLint index = (texel_coords.y * ogre_texture.width + texel_coords.x) * ogre_texture.bytes_per_texel;

		// modulate texel colors and faceted lighting colors
		set_pixel(x, y, Color(static_cast<GLubyte>(ogre_texture.texels[index] * lighting.r),
			static_cast<GLubyte>(ogre_texture.texels[index + 1] * lighting.g),
			static_cast<GLubyte>(ogre_texture.texels[index + 2] * lighting.b)));

		// update depth of current fragment in depth buffer
		set_pixel_depth(x, y, depth);
	};

	return render_triangle(p0, p1, p2, set_pixel_func);
}

bool GLPbo::render_triangle_texture_shaded(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2,
	glm::vec3 const& l0, glm::vec3 const& l1, glm::vec3 const& l2,
	glm::vec2 const& t0, glm::vec2 const& t1, glm::vec2 const& t2)
{
	auto set_pixel_func = [&](GLint x, GLint y, GLfloat depth, GLfloat w0, GLfloat w1, GLfloat w2) -> void
	{
		// interpolate texture coords and lighting colors
		glm::vec2 interpolated_tex_coords{ w0 * t0 + w1 * t1 + w2 * t2 };
		glm::vec3 smooth_lighting{ w0 * l0 + w1 * l1 + w2 * l2 };

		// map interpolated texture coords to texel coords
		glm::ivec2 texel_coords
		{
			static_cast<GLint>(std::floor(ogre_texture.width * glm::clamp(interpolated_tex_coords.x, 0.f, 1.f)) - 0.5f),
			static_cast<GLint>(std::floor(ogre_texture.height * glm::clamp(interpolated_tex_coords.y, 0.f, 1.f)) - 0.5f)
		};

		// calculate index of texel mapped to
		GLint index = (texel_coords.y * ogre_texture.width + texel_coords.x) * ogre_texture.bytes_per_texel;

		// modulate texel colors and smooth lighting colors
		set_pixel(x, y, Color(static_cast<GLubyte>(ogre_texture.texels[index] * smooth_lighting.r),
			static_cast<GLubyte>(ogre_texture.texels[index + 1] * smooth_lighting.g),
			static_cast<GLubyte>(ogre_texture.texels[index + 2] * smooth_lighting.b)));

		// update depth of current fragment in depth buffer
		set_pixel_depth(x, y, depth);
	};

	return render_triangle(p0, p1, p2, set_pixel_func);
}