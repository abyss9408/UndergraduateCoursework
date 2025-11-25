/* !
@file    	glapp.h
@author  	pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date    	05/06/2024

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
#include <iomanip>
#include <random>
#include <glm/gtc/type_ptr.hpp>
#include <map>
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
	  GLuint primitive_cnt;
	  GLuint vaoid;
	  GLuint draw_cnt;

	  void init(std::string);
	  void release();
  };

  struct GLViewport
  {
	  GLint x, y;
	  GLsizei width, height;
  };

  // function to insert shader program into container GLApp::shdrpgms ...
  static void insert_shdrpgm(std::string, std::string, std::string);
  // function to parse scene file ...
  static void init_scene(std::string);

  struct GLObject
  {
	  // orientation.x is angle_disp and
	  // orientation.y is angle_speed
	  // both values specified in degrees
	  glm::vec2 orientation;

	  // translation and scaling
	  glm::vec2 scaling;
	  glm::vec2 position;
	  glm::mat3 mdl_to_ndc_xform;

	  // which model and which shader
	  std::map<std::string, GLApp::GLModel>::iterator mdl_ref;
	  std::map<std::string, GLSLShader>::iterator shd_ref;

	  glm::vec3 color;
	  glm::mat3 mdl_xform;

	  void draw() const;
	  void update(GLdouble delta_time);
  };

  // containers
  static std::map<std::string, GLSLShader> shdrpgms;
  static std::map<std::string, GLModel> models;
  static std::map<std::string, GLObject> objects;

  struct Camera2D
  {
	  GLObject* pgo;
	  glm::vec2 right, up;
	  glm::mat3 view_xform, camwin_to_ndc_xform, world_to_ndc_xform;

	  GLint height{ 1000 };
	  GLfloat ar;

	  // window change parameters ...
	  GLint min_height{ 500 }, max_height{ 2000 };
	  // height is increasing if 1 and decreasing if -1
	  GLint height_chg_dir{ 1 };
	  // increments by which window height is changed per Z key press
	  GLint height_chg_val{ 5 };

	  // camera's speed when button U is pressed
	  GLfloat linear_speed{ 2.f };

	  // keyboard button press flags
	  GLboolean camtype_flag{ GL_FALSE }; // button V
	  GLboolean zoom_flag{ GL_FALSE }; // button Z
	  GLboolean left_turn_flag{ GL_FALSE }; // button H
	  GLboolean right_turn_flag{ GL_FALSE }; // button K
	  GLboolean move_flag{ GL_FALSE }; // button U

	  void init(GLFWwindow*, GLObject* ptr);
	  void update(GLFWwindow*);
  };

  static Camera2D camera2d;
};

#endif /* GLAPP_H */
