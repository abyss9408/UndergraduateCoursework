/*!
@file    	  glhelper.cpp
@author  	  pghali@digipen.edu
@co-author	parminder.singh@digipen.edu
@co-author	bryanweize.ang@digipen.edu
@date    	12/05/2024

This file implements functionality useful and necessary to build OpenGL
applications including use of external APIs such as GLFW to create a
window and start up an OpenGL context and use GLEW to extract function 
pointers to OpenGL implementations.

*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */
#include <glhelper.h>
#include <iostream>
#include <hook_csd2101.h>

/*                                                   objects with file scope
----------------------------------------------------------------------------- */
// static data members declared in GLHelper
GLint GLHelper::width;
GLint GLHelper::height;
GLdouble GLHelper::fps;
GLdouble GLHelper::delta_time;
std::string GLHelper::title;
GLFWwindow* GLHelper::ptr_window;
GLboolean GLHelper::keystate1{ GL_TRUE };
GLboolean GLHelper::keystate2{ GL_FALSE };
GLboolean GLHelper::keystate3{ GL_FALSE };
GLboolean GLHelper::keystate4{ GL_FALSE };

/*  _________________________________________________________________________ */
/*! init

@param GLint width
@param GLint height
Dimensions of window requested by program

@param std::string title_str
String printed to window's title bar

@return bool
true if OpenGL context and GLEW were successfully initialized.
false otherwise.

Uses GLFW to create OpenGL context. GLFW's initialization follows from here:
http://www.glfw.org/docs/latest/quick.html
a window of size width x height pixels
and its associated OpenGL context that matches a core profile that is 
compatible with OpenGL 4.5 and doesn't support "old" OpenGL, has 32-bit RGBA,
double-buffered color buffer, 24-bit depth buffer and 8-bit stencil buffer 
with each buffer of size width x height pixels
*/
bool GLHelper::init(GLint width, GLint height, std::string title) {
  GLHelper::width = width;
  GLHelper::height = height;
  GLHelper::title = title;

  // Part 1
  if (!glfwInit()) {
    std::cout << "GLFW init has failed - abort program!!!" << std::endl;
    return false;
  }

  // In case a GLFW function fails, an error is reported to callback function
  glfwSetErrorCallback(GLHelper::error_cb);

  // Before asking GLFW to create an OpenGL context, we specify the minimum constraints
  // in that context:
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);
  glfwWindowHint(GLFW_RED_BITS, 8); glfwWindowHint(GLFW_GREEN_BITS, 8);
  glfwWindowHint(GLFW_BLUE_BITS, 8); glfwWindowHint(GLFW_ALPHA_BITS, 8);

  GLHelper::ptr_window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
  if (!GLHelper::ptr_window) {
    std::cerr << "GLFW unable to create OpenGL context - abort program\n";
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(GLHelper::ptr_window);
  
  setup_event_callbacks();

  // this is the default setting ...
  glfwSetInputMode(GLHelper::ptr_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

  // Part 2: Initialize entry points to OpenGL functions and extensions
  GLenum err = glewInit();
  if (GLEW_OK != err) {
    std::cerr << "Unable to initialize GLEW - error: "
      << glewGetErrorString(err) << " abort program" << std::endl;
    return false;
  }
  if (GLEW_VERSION_4_5) {
    std::cout << "Using glew version: " << glewGetString(GLEW_VERSION) << std::endl;
    std::cout << "Driver supports OpenGL 4.5\n" << std::endl;
  } else {
    std::cerr << "Warning: The driver may lack full compatibility with OpenGL 4.5, potentially limiting access to advanced features." << std::endl;
  }

  return true;
}

void GLHelper::setup_event_callbacks() {
    AUTOMATION_HOOK_EVENTS(); // Automation hook. [!WARNING!] Do not alter/remove this! 
    
    glfwSetFramebufferSizeCallback(GLHelper::ptr_window, GLHelper::fbsize_cb);
    glfwSetKeyCallback(GLHelper::ptr_window, GLHelper::key_cb);
    glfwSetMouseButtonCallback(GLHelper::ptr_window, GLHelper::mousebutton_cb);
    glfwSetCursorPosCallback(GLHelper::ptr_window, GLHelper::mousepos_cb);
    glfwSetScrollCallback(GLHelper::ptr_window, GLHelper::mousescroll_cb);
}

/*  _________________________________________________________________________ */
/*! cleanup

@param none

@return none

For now, there are no resources allocated by the application program.
The only task is to have GLFW return resources back to the system and
gracefully terminate.
*/
void GLHelper::cleanup() {
  // Part 1
  glfwTerminate();
}

/*  _________________________________________________________________________*/
/*! key_cb

@param GLFWwindow*
Handle to window that is receiving event

@param int
the keyboard key that was pressed or released

@parm int
Platform-specific scancode of the key

@parm int
GLFW_PRESS, GLFW_REPEAT or GLFW_RELEASE
action will be GLFW_KEY_UNKNOWN if GLFW lacks a key token for it,
for example E-mail and Play keys.

@parm int
bit-field describing which modifier keys (shift, alt, control)
were held down

@return none

This function is called when keyboard buttons are pressed.
When the ESC key is pressed, the close flag of the window is set.
*/
void GLHelper::key_cb(GLFWwindow *pwin, int key, int scancode, int action, int mod) {
  if (GLFW_PRESS == action) {
#ifdef _DEBUG
    std::cout << "Key pressed" << std::endl;
#endif
  } else if (GLFW_REPEAT == action) {
#ifdef _DEBUG
    std::cout << "Key repeatedly pressed" << std::endl;
#endif
  } else if (GLFW_RELEASE == action) {
#ifdef _DEBUG
    std::cout << "Key released" << std::endl;
#endif
  }

  if (GLFW_PRESS == action)
  {
      switch (key)
      {
      case GLFW_KEY_1:
          keystate1 = GL_TRUE;
          keystate2 = GL_FALSE;
          keystate3 = GL_FALSE;
          keystate4 = GL_FALSE;
          break;
      case GLFW_KEY_2:
          keystate1 = GL_FALSE;
          keystate2 = GL_TRUE;
          keystate3 = GL_FALSE;
          keystate4 = GL_FALSE;
          break;
      case GLFW_KEY_3:
          keystate1 = GL_FALSE;
          keystate2 = GL_FALSE;
          keystate3 = GL_TRUE;
          keystate4 = GL_FALSE;
          break;
      case GLFW_KEY_4:
          keystate1 = GL_FALSE;
          keystate2 = GL_FALSE;
          keystate3 = GL_FALSE;
          keystate4 = GL_TRUE;
          break;
      case GLFW_KEY_ESCAPE:
          glfwSetWindowShouldClose(pwin, GLFW_TRUE);
          break;
      }
  }
}

/*  _________________________________________________________________________*/
/*! mousebutton_cb

@param GLFWwindow*
Handle to window that is receiving event

@param int
the mouse button that was pressed or released
GLFW_MOUSE_BUTTON_LEFT and GLFW_MOUSE_BUTTON_RIGHT specifying left and right
mouse buttons are most useful

@parm int
action is either GLFW_PRESS or GLFW_RELEASE

@parm int
bit-field describing which modifier keys (shift, alt, control)
were held down

@return none

This function is called when mouse buttons are pressed.
*/
void GLHelper::mousebutton_cb(GLFWwindow *pwin, int button, int action, int mod) {
  switch (button) {
  case GLFW_MOUSE_BUTTON_LEFT:
#ifdef _DEBUG
    std::cout << "Left mouse button ";
#endif
    break;
  case GLFW_MOUSE_BUTTON_RIGHT:
#ifdef _DEBUG
    std::cout << "Right mouse button ";
#endif
    break;
  }
  switch (action) {
  case GLFW_PRESS:
#ifdef _DEBUG
    std::cout << "pressed!!!" << std::endl;
#endif
    break;
  case GLFW_RELEASE:
#ifdef _DEBUG
    std::cout << "released!!!" << std::endl;
#endif
    break;
  }
}

/*  _________________________________________________________________________*/
/*! mousepos_cb

@param GLFWwindow*
Handle to window that is receiving event

@param double
new cursor x-coordinate, relative to the left edge of the client area

@param double
new cursor y-coordinate, relative to the top edge of the client area

@return none

This functions receives the cursor position, measured in screen coordinates but
relative to the top-left corner of the window client area.
*/
void GLHelper::mousepos_cb(GLFWwindow *pwin, double xpos, double ypos) {
#ifdef _DEBUG
  std::cout << "Mouse cursor position: (" << xpos << ", " << ypos << ")" << std::endl;
#endif
}

/*  _________________________________________________________________________*/
/*! mousescroll_cb

@param GLFWwindow*
Handle to window that is receiving event

@param double
Scroll offset along X-axis

@param double
Scroll offset along Y-axis

@return none

This function is called when the user scrolls, whether with a mouse wheel or
touchpad gesture. Although the function receives 2D scroll offsets, a simple
mouse scroll wheel, being vertical, provides offsets only along the Y-axis.
*/
void GLHelper::mousescroll_cb(GLFWwindow *pwin, double xoffset, double yoffset) {
#ifdef _DEBUG
  std::cout << "Mouse scroll wheel offset: ("
    << xoffset << ", " << yoffset << ")" << std::endl;
#endif
}

/*  _________________________________________________________________________ */
/*! error_cb

@param int
GLFW error code

@parm char const*
Human-readable description of the code

@return none

The error callback receives a human-readable description of the error and
(when possible) its cause.
*/
void GLHelper::error_cb(int error, char const* description) {
#ifdef _DEBUG
  std::cerr << "GLFW error: " << description << std::endl;
#endif
}

/*  _________________________________________________________________________ */
/*! fbsize_cb

@param GLFWwindow*
Handle to window that is being resized

@parm int
Width in pixels of new window size

@parm int
Height in pixels of new window size

@return none

This function is called when the window is resized - it receives the new size
of the window in pixels.
*/
void GLHelper::fbsize_cb(GLFWwindow *ptr_win, int width, int height) {
#ifdef _DEBUG
  std::cout << "fbsize_cb getting called!!!" << std::endl;
#endif
  GLHelper::width = width;
  GLHelper::height = height;
}

/*  _________________________________________________________________________*/
/*! update_time

@param double
fps_calc_interval: the interval (in seconds) at which fps is to be
calculated

This function must be called once per game loop. It uses GLFW's time functions
to compute:
1. the interval in seconds between each frame
2. the frames per second every "fps_calc_interval" seconds
*/
void GLHelper::update_time(double fps_calc_interval) {
  // get elapsed time (in seconds) between previous and current frames
  static double prev_time = glfwGetTime();
  double curr_time = glfwGetTime();
  delta_time = curr_time - prev_time;
  prev_time = curr_time;

  // fps calculations
  static double count = 0.0; // number of game loop iterations
  static double start_time = glfwGetTime();
  // get elapsed time since very beginning (in seconds) ...
  double elapsed_time = curr_time - start_time;

  ++count;

  // update fps at least every 10 seconds ...
  fps_calc_interval = (fps_calc_interval < 0.0) ? 0.0 : fps_calc_interval;
  fps_calc_interval = (fps_calc_interval > 10.0) ? 10.0 : fps_calc_interval;
  if (elapsed_time > fps_calc_interval) {
    GLHelper::fps = count / elapsed_time;
    start_time = curr_time;
    count = 0.0;
  }
}

/*  _________________________________________________________________________*/
/*! print_specs

This function prints useful information about your OpenGL context and graphics 
hardware to standard output
*/
void GLHelper::print_specs() {
    GLint major_ver{}, minor_ver{}, current_context{}, max_vertex, max_indices{}, max_tex{},
        max_gen_vertex_attri{}, max_vertex_buffer_bindings{};
    GLint max_view_dims[2];

    // get info about OpenGL context and graphics hardware
    GLubyte const* str_ven = glGetString(GL_VENDOR);
    GLubyte const* str_ren = glGetString(GL_RENDERER);
    GLubyte const* str_ver = glGetString(GL_VERSION);
    GLubyte const* str_shader_ver = glGetString(GL_SHADING_LANGUAGE_VERSION);
    glGetIntegerv(GL_MAJOR_VERSION, &major_ver);
    glGetIntegerv(GL_MINOR_VERSION, &minor_ver);
    glGetIntegerv(GL_DOUBLEBUFFER, &current_context);
    glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &max_vertex);
    glGetIntegerv(GL_MAX_ELEMENTS_INDICES, &max_indices);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_tex);
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, &max_view_dims[0]);
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &max_gen_vertex_attri);
    glGetIntegerv(GL_MAX_VERTEX_ATTRIB_BINDINGS, &max_vertex_buffer_bindings);

    // print info to standard output
    std::cout << "GPU Vendor: " << str_ven << '\n';
    std::cout << "GL Renderer: " << str_ren << '\n';
    std::cout << "GL Version: " << str_ver << '\n';
    std::cout << "GL Shader Version: " << str_shader_ver << '\n';
    std::cout << "GL Major Version: " << major_ver << '\n';
    std::cout << "GL Minor Version: " << minor_ver << '\n';
    std::cout << (current_context ? "Current OpenGL Context is double-buffered\n" : "Current OpenGL Context is not double-buffered\n");
    std::cout << "Maximum Vertex Count: " << max_vertex << '\n';
    std::cout << "Maximum Indices Count: " << max_indices << '\n';
    std::cout << "GL Maximum texture size: " << max_tex << '\n';
    std::cout << "Maximum Viewport Dimensions: " << max_view_dims[0] << " x " << max_view_dims[1] << '\n';
    std::cout << "Maximum generic vertex attributes: " << max_gen_vertex_attri << '\n';
    std::cout << "Maximum vertex buffer bindings: " << max_vertex_buffer_bindings << '\n';
}