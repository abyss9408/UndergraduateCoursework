/*!*****************************************************************************
\file main.cpp
\author Vadim Surov (vsurov\@digipen.edu)
\co-author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 4.2
\date 02/02/2025 (MM/DD/YYYY)
\brief This file has definitions of the main function
       along with global variables and sub-functions used in main.
*******************************************************************************/

#include <iostream> // For standard input/output operations
#include "common/debug.h" // Custom debugging utilities
#include "common/cg.h" // Common computer graphics utilities
#include <GLFW/glfw3.h> // For OpenGL and windowing

// Global variable for the scene object
cg::Scene* pScene = nullptr;

// Global variable for the screen size
glm::ivec2 screen = { WIDTH, HEIGHT };

// Global variables for interactions
bool lbutton_down = false;
bool mode_shift = false;
bool mode_alt = false;

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
        {
            glfwSetWindowShouldClose(pWindow, GL_TRUE);
        }
        else if (key == GLFW_KEY_TAB)
        {
            int n = 1;
            for (; n < pScene->passes[0].objects.size(); ++n)
            {
                if (pScene->passes[0].objects[n].isVisible)
                {
					break;
                }
            }

			pScene->passes[0].objects[n].isVisible = false;
            pScene->passes[0].objects[(n + 1 == pScene->passes[0].objects.size()) ? 1 : n + 1].isVisible = true;
        }
    }
    
	mode_shift = (mods == GLFW_MOD_SHIFT);
	mode_alt = (mods == GLFW_MOD_ALT);
}

/*
   This function serves as the callback parameter for
       the glfwSetMouseButtonCallback function, used in the main function
*/
void mouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
		if (action == GLFW_PRESS)
		{
			lbutton_down = true;
		}
		else if (action == GLFW_RELEASE)
		{
			lbutton_down = false;
		}
    }
}

/*
   This function serves as the callback parameter for
       the glfwSetCursorPosCallback function, used in the main function
*/
void cursorPosCallback(GLFWwindow* pWindow, double xpos, double ypos)
{
    if (!lbutton_down)
    {
        return;
    }

    if (mode_shift)
    {
        static double oldxpos = xpos;
		static double oldypos = ypos;
		pScene->lightsOnCursor(xpos - oldxpos, ypos - oldypos, &pScene->shader);
		oldxpos = xpos;
		oldypos = ypos;
	}
	else
	{
        static double oldxpos = xpos;
        static double oldypos = ypos;
        pScene->camerasOnCursor(xpos - oldxpos, ypos - oldypos, &pScene->shader);
        oldxpos = xpos;
        oldypos = ypos;
    }
}

/*
   This function serves as the callback parameter for
       the glfwSetScrollCallback function, used in the main function
*/
void scrollCallback(GLFWwindow* pWindow, double xoffset, double yoffset)
{
    if (mode_shift)
    {
        pScene->lightsOnScroll(xoffset, yoffset, &pScene->shader);
    }
    else
    {
        pScene->camerasOnScroll(xoffset, yoffset, &pScene->shader);
    }
}

/*
   This function serves as the callback parameter for
       the glfwSetWindowSizeCallback function, used in the main function
*/
void sizeCallback(GLFWwindow* pWindow, int width, int height)
{
    pScene->resize(width, height);
    screen = { width, height };
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
    GLFWwindow* pWindow = glfwCreateWindow(WIDTH, HEIGHT, "BlinnPhong", NULL, NULL);
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
    std::cout << "Interactions:" << std::endl;
    std::cout << "- Press the Tab key to change the object." << std::endl;
    std::cout << "- The orbiting around the object is controlled by the mouse left button and wheel." << std::endl;
    std::cout << "- The light is controlled the same way when shift is pressed." << std::endl;
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
                #include "BlinnPhong.vert.glsl"
            },

            // Fragment shader
            {
                #include "BlinnPhong.frag.glsl"
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
                    BLACK,

                    // Depth test
                    ENABLE,

                    // Objects
                    {
                        {
                            PLANE,
                            MATERIAL_PLANE,
                            glm::translate(glm::mat4(1.0f), {0.0f, -1.1f, 0.0f}),
                            VISIBLE
                        },
                        {
                            TORUS,
                            MATERIAL_OBJECT,
                            NOTRANSFORM,
                            VISIBLE
                        },
                        {
                            CUBE,
                            MATERIAL_OBJECT,
                            NOTRANSFORM,
                            INVISIBLE
                        },
                        {
                            SPHERE,
                            MATERIAL_OBJECT,
                            NOTRANSFORM,
                            INVISIBLE
                        },
                        {
                            TEAPOT,
                            MATERIAL_OBJECT,
                            NOTRANSFORM,
                            INVISIBLE
                        }
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