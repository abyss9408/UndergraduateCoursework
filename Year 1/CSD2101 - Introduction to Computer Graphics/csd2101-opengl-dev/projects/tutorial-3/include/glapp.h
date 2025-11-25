/* !
@file    	glapp.h
@author  	pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date    	30/05/2024

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
  };

  struct GLViewport
  {
	  GLint x, y;
	  GLsizei width, height;
  };

  // container for models
  static std::vector<GLModel> models;

  static GLApp::GLModel box_model();
  static GLApp::GLModel mystery_model();

  static void init_models_cont();

  static std::vector<GLSLShader> shdrpgms;

  using VPSS = std::vector<std::pair<std::string, std::string>>;
  static void init_shdrpgms_cont(GLApp::VPSS const&);

  struct GLObject
  {
	  // angular speed and angular displacement are with respect to
	  // X-axis and together represent the orientation of an object
	  GLfloat angle_speed, angle_disp;

	  // translation and scaling
	  glm::vec2 scaling;
	  glm::vec2 position;

	  // matrix to map geometry from model to world to NDC coordinates
	  glm::mat3 mdl_to_ndc_xform;

	  // which model and which shader
	  GLuint mdl_ref;
	  GLuint shd_ref;

	  // function to initialize object's state
	  void init();

	  // function to render object's model (specified by index mdl_ref)
	  // uses model transformation matrix mdl_to_ndc_xform matrix
	  // and shader program specified by index shd_ref ...
	  void draw() const;

	  // function to update angle_disp and then compute mdl_to_ndc_xform
	  void update(GLdouble delta_time);
  };

  static std::list<GLApp::GLObject> objects;
  static GLuint current_shdr_pgm;
  static GLint num_box_obj;
  static GLint num_mys_obj;
};

#endif /* GLAPP_H */
