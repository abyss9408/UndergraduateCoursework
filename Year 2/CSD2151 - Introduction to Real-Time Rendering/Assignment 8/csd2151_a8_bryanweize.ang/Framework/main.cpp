/*!*****************************************************************************
\file main.cpp
\author Vadim Surov (vsurov\@digipen.edu)
\co-author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 11
\date 03/23/2025 (MM/DD/YYYY)
\brief This file has definitions of the main function
       along with global variables and sub-functions used in main.
*******************************************************************************/

#include <iostream> // For standard input/output operations
#include "common/debug.h" // Custom debugging utilities
#include "common/cg.h" // Common computer graphics utilities
#include <GLFW/glfw3.h> // For OpenGL and windowing

// Global variable for the scene object
cg::Scene* pScene = nullptr;

// Camera mode
cg::CameraType cameraMode = cg::CameraType::ORBITING;

// Mouse position tracking
double prevX = 0.0, prevY = 0.0;
bool firstMouse = true;

// Flag to determine if mouse button is pressed
bool mousePressed = false;

#define ANINSTANCE(x, z)\
         { MATERIAL_OBJECT, glm::translate(glm::mat4(1.0f), {x, 0.0f, z}) }

#define INSTANCES(x)\
                   ANINSTANCE(x, -20.0f-5.0f),\
                   ANINSTANCE(x, -18.0f-5.0f),\
                   ANINSTANCE(x, -16.0f-5.0f),\
                   ANINSTANCE(x, -14.0f-5.0f),\
                   ANINSTANCE(x, -12.0f-5.0f),\
                   ANINSTANCE(x, -10.0f-5.0f),\
                   ANINSTANCE(x,  -8.0f-5.0f),\
                   ANINSTANCE(x,  -6.0f-5.0f),\
                   ANINSTANCE(x,  -4.0f-5.0f),\
                   ANINSTANCE(x,  -2.0f-5.0f),\
                   ANINSTANCE(x,   0.0f-5.0f),\
                   ANINSTANCE(x,   2.0f-5.0f),\
                   ANINSTANCE(x,   4.0f-5.0f),\
                   ANINSTANCE(x,   6.0f-5.0f),\
                   ANINSTANCE(x,   8.0f-5.0f),\
                   ANINSTANCE(x,  10.0f-5.0f),\
                   ANINSTANCE(x,  12.0f-5.0f),\
                   ANINSTANCE(x,  14.0f-5.0f),\
                   ANINSTANCE(x,  16.0f-5.0f),\
                   ANINSTANCE(x,  18.0f-5.0f),\
                   ANINSTANCE(x,  20.0f-5.0f)

/*
   This function serves as the callback parameter for
      the glfwSetKeyCallback function, used in the main function.

   This function serves as the callback parameter for
      the glfwSetKeyCallback function, used in the main function
*/
void keyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
    // Close the app on ESC key pressed event
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(pWindow, GL_TRUE);

        // Switch between camera control types (O for orbiting, W for walking)
        if (key == GLFW_KEY_O)
        {
            cameraMode = cg::CameraType::ORBITING;
            pScene->camerasSetType(cameraMode);
        }
        else if (key == GLFW_KEY_W)
        {
            cameraMode = cg::CameraType::WALKING;
            pScene->camerasSetType(cameraMode);
        }
    }
}

/*
   This function serves as the callback parameter for
       the glfwSetMouseButtonCallback function, used in the main function
*/
void mouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods)
{
    // Reset first mouse flag when mouse button is pressed
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            mousePressed = true;
        }
        else if (action == GLFW_RELEASE)
        {
            mousePressed = false;
        }
    }
}

/*
   This function serves as the callback parameter for
       the glfwSetCursorPosCallback function, used in the main function
*/
void cursorPosCallback(GLFWwindow* pWindow, double xpos, double ypos)
{
    if (firstMouse)
    {
        prevX = xpos;
        prevY = ypos;
        firstMouse = false;
    }

    double xoffset = xpos - prevX;
    double yoffset = ypos - prevY;

    prevX = xpos;
    prevY = ypos;

    if (mousePressed)
    {
        // Update camera based on cursor movement
        pScene->camerasOnCursor(xoffset, yoffset, &pScene->shader);
    }
}

/*
   This function serves as the callback parameter for
       the glfwSetScrollCallback function, used in the main function
*/
void scrollCallback(GLFWwindow* pWindow, double xoffset, double yoffset)
{
    pScene->camerasOnScroll(xoffset, yoffset, &pScene->shader);
}

/*
   This function serves as the callback parameter for
       the glfwSetWindowSizeCallback function, used in the main function
*/
void sizeCallback(GLFWwindow* pWindow, int width, int height)
{
    // Update scene passes with new dimensions
    pScene->resize(width, height);
}

/*
    Main function: entry point of the program
*/
int main(int argc, char** argv)
{
    // Initialize GLFW library
    if (!glfwInit())
        exit(EXIT_FAILURE);

    // Debug context setup for OpenGL if in debug mode
#if defined( _DEBUG )
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

    // Configure OpenGL context properties
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

    // Create a window with an OpenGL context
    GLFWwindow* pWindow = glfwCreateWindow(WIDTH, HEIGHT, "OgresAssignment", NULL, NULL);
    if (!pWindow)
    {
        std::cerr << "Unable to create OpenGL context." << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(pWindow);

    // Load OpenGL functions using GLEW
    glewExperimental = GL_TRUE; // Enable experimental features for core profile
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;

        // Close pWindow and terminate GLFW
        glfwTerminate();

        // Exit program
        std::exit(EXIT_FAILURE);
    }

    // Debug information setup for OpenGL
#if defined( _DEBUG )
    dumpGLInfo();
#endif

    std::cout << std::endl;
    std::cout << "A computer graphics framework." << std::endl;
    std::cout << std::endl;

    // More setup and start of the debuging output
#if defined( _DEBUG )
    glDebugMessageCallback(debugCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 0,
        GL_DEBUG_SEVERITY_NOTIFICATION, -1, "Start debugging");
#endif

    // Set GLFW callback functions
    glfwSetKeyCallback(pWindow, keyCallback);
    glfwSetMouseButtonCallback(pWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(pWindow, cursorPosCallback);
    glfwSetScrollCallback(pWindow, scrollCallback);
    glfwSetWindowSizeCallback(pWindow, sizeCallback);

    try
    {
        // Define the scene and its properties
        cg::Scene scene =
        {
            // Vertex shader
            {
                #include "OgresAssignment.vert.glsl"
            },

            // Fragment shader
            {
                #include "OgresAssignment.frag.glsl"
            },

            // Rendering passes
            {
                // Pass 0
                {
                    // Rendering target
                    { cg::FBO::Type::Deferred, WIDTH, HEIGHT },

                    // Viewport
                    { 0, 0, WIDTH, HEIGHT },

                    // Color to clear buffers
                    BLACK,

                    // Depth test
                    ENABLE,

                    // Objects
                    {
                        {
                            "bs_ears.obj",
                            {
                                INSTANCES(-10.0f),
                                INSTANCES(-8.0f),
                                INSTANCES(-6.0f),
                                INSTANCES(-4.0f),
                                INSTANCES(-2.0f),
                                INSTANCES(0.0f),
                                INSTANCES(2.0f),
                                INSTANCES(4.0f),
                                INSTANCES(6.0f),
                                INSTANCES(8.0f),
                                INSTANCES(10.0f),
                            }
                        }
                    },

                    // The camera
                    {
                        { {0.0f, 4.0f, 5.0f} }
                    },

                    // Lights
                    {
                        DEFAULT
                    },

                    // Textures
                    {
                        { { "ogre_diffuse.png" },   GL_TEXTURE3 }, // Channel 3
                        { { "ogre_normalmap.png" }, GL_TEXTURE4 }  // Channel 4 
                    },

                    // Setup optional uniforms in the shader
                    NULL
                },
                
                // Pass 1
                {
                    // Rendering target
                    DEFAULT,

                    // Viewport
                    { 0, 0, WIDTH, HEIGHT },

                    // Background color
                    { },

                    // Depth buffer
                    DISABLE,

                    // Objects
                    {
                        {
                            QUAD
                        }
                    },

                    // The camera
                    {

                    },

                    // Lights
                    {

                    },

                    // Textures
                    {

                    },

                    // Setup uniforms in the shader
                    NULL
                }
            }
        };

        // Provides a global access through pScene pointer
        pScene = &scene; 

        // Main rendering loop
        while (!glfwWindowShouldClose(pWindow))
        {
            checkForOpenGLError(__FILE__, __LINE__);

            scene.render();

            glfwSwapBuffers(pWindow);
            glfwPollEvents();
        }

    }
    catch (std::exception const& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;

        // Close pWindow and terminate GLFW
        glfwTerminate();

        // Exit program
        std::exit(EXIT_FAILURE);
    }

    // End of the debugging output
#if defined( _DEBUG )
    glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 1,
        GL_DEBUG_SEVERITY_NOTIFICATION, -1, "End debug");
#endif

    // Close pWindow and terminate GLFW
    glfwTerminate();

    // Exit program
    return EXIT_SUCCESS;
}