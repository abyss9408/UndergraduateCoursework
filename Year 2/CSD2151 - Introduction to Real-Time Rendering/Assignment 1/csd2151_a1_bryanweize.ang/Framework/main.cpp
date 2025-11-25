/*!*****************************************************************************
\file main.cpp
\author Vadim Surov (vsurov\@digipen.edu)
\co-author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151/CSD2150/CS250
\par Assignment: 1
\date 01/06/2025 (MM/DD/YYYY)
\brief This file has definitions of the main function
       along with global variables and sub-functions used in main.
*******************************************************************************/

#include <iostream> // For standard input/output operations
#include "common/debug.h" // Custom debugging utilities
#include "common/cg.h" // Common computer graphics utilities
#include <GLFW/glfw3.h> // For OpenGL and windowing
#include <random>

// Global variable for the scene object
cg::Scene* pScene = nullptr;

float red = 0.0f;
float green = 0.0f;
float blue = 0.0f;

// random number generator
std::random_device rd;
std::default_random_engine gen(rd());
std::uniform_real_distribution<GLfloat> urdf(0.0f, std::nextafter(1.0f, std::numeric_limits<GLfloat>::max()));

/*
   This function serves as the callback parameter for 
      glfwSetKeyCallback function used in the main function
*/
void keyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
    // Close the app on ESC key pressed event
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(pWindow, GL_TRUE);
        }
        else if (key == GLFW_KEY_W)
        {
            pScene->passes[0].clearColor = { 1.0f, 1.0f, 1.0f, 0.0f };
        }
		else if (key == GLFW_KEY_SPACE)
		{
			pScene->passes[0].clearColor = { urdf(gen), urdf(gen), urdf(gen), 0.0f};
		}
    }
}

/*
   This function serves as the callback parameter for
      glfwSetMouseButtonCallback function used in the main function
*/
void mouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods)
{
    // Placeholder for mouse button actions
}

/*
   This function serves as the callback parameter for
      glfwSetCursorPosCallback function used in the main function
*/
void cursorPosCallback(GLFWwindow* pWindow, double xpos, double ypos)
{
    // Placeholder for cursor movement actions
	red = static_cast<float>(xpos) / WIDTH;
	green = static_cast<float>(ypos) / HEIGHT;

	pScene->passes[0].clearColor = { red, green, blue, 0.0f };
}

/*
   This function serves as the callback parameter for
      glfwSetScrollCallback function used in the main function
*/
void scrollCallback(GLFWwindow* pWindow, double xoffset, double yoffset)
{
    // Placeholder for scroll actions
    blue += static_cast<float>(yoffset) * 0.1f;
    
    // Clamp blue value between 0 and 1
	blue = std::max(0.0f, std::min(1.0f, blue));
    
    pScene->passes[0].clearColor = { red, green, blue, 0.0f };
}

/*
   This function serves as the callback parameter for
      glfwSetWindowSizeCallback function used in the main function
*/
void sizeCallback(GLFWwindow* pWindow, int width, int height)
{
    // Placeholder for window resize actions
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
    GLFWwindow* pWindow = glfwCreateWindow(WIDTH, HEIGHT, "Empty", NULL, NULL);
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
	std::cout << "The red color component varies based on the horizontal movement of the mouse from the left side of"
        "the window(where its value is 0.0) to the right side(where its value is 1.0)." << std::endl;
    std::cout << "The green color component varies based on the vertical movement of the mouse from the left side of"
        "the window(where its value is 0.0) to the right side(where its value is 1.0)." << std::endl;
    std::cout << "The blue color component varies based on the mouse wheel's rotation. It increases when scrolling"
        "forward and decreases when scrolling backward." << std::endl;
	std::cout << "Press 'W' to set the background color to white." << std::endl;
	std::cout << "Press 'Space' to set the background color to a random color." << std::endl;
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
                #include "Framework.vert.glsl"
            },

            // Fragment shader
            {
                #include "Framework.frag.glsl"
            },

            // Rendering passes
            {
                // Pass 0
                {
                    // Rendering target
                    DEFAULT,

                    // Viewport
                    { 0, 0, WIDTH, HEIGHT },

                    // Color to clear buffers
                    { },

                    // Depth test
                    ENABLE,

                    // Objects
                    {

                    },

                    // The camera
                    {
                        DEFAULT
                    },

                    // Lights
                    {
                        DEFAULT
                    },

                    // Textures
                    {

                    },

                    // Setup optional uniforms in the shader
                    [](cg::Program& shader)
                    {

                    }
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