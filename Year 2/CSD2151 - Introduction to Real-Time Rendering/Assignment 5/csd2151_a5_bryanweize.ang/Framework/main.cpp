/*!*****************************************************************************
\file main.cpp
\author Vadim Surov (vsurov\@digipen.edu)
\co-author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 5.2
\date 02/09/2025 (MM/DD/YYYY)
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
int currentObject = 0; // Track current selected object

/*
   This function serves as the callback parameter for
      the glfwSetKeyCallback function, used in the main function.

   This function serves as the callback parameter for
      the glfwSetKeyCallback function, used in the main function
*/
void keyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(pWindow, GL_TRUE);
        }
        else if (key == GLFW_KEY_TAB) {
            currentObject = (currentObject + 1) % pScene->passes[1].objects.size();
            std::cout << "Selected object " << currentObject + 1 << std::endl;
        }
        else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9) {
            float value = float(key - GLFW_KEY_0);
            auto& material = pScene->passes[1].objects[currentObject].instances[0].material;

            if (mods & GLFW_MOD_ALT) {
                float eta = 0.9f + value / 90.0f;
                material.params["material.eta"] = { eta };
            }
            else {
                float reflection = value / 9.0f;
                material.params["material.reflectionFactor"] = { reflection };
            }
        }
    }
    mode_shift = (mods & GLFW_MOD_SHIFT);
    mode_alt = (mods & GLFW_MOD_ALT);
}

/*
   This function serves as the callback parameter for
       the glfwSetMouseButtonCallback function, used in the main function
*/
void mouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        lbutton_down = (action == GLFW_PRESS);
    }
}

/*
   This function serves as the callback parameter for
       the glfwSetCursorPosCallback function, used in the main function
*/
void cursorPosCallback(GLFWwindow* pWindow, double xpos, double ypos)
{
    static double oldxpos = xpos;
    static double oldypos = ypos;

    if (lbutton_down)
    {
        pScene->camerasOnCursor(xpos - oldxpos, ypos - oldypos, &pScene->shader);
    }

    oldxpos = xpos;
    oldypos = ypos;
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
    GLFWwindow* pWindow = glfwCreateWindow(WIDTH, HEIGHT, "Environment Mapping", NULL, NULL);
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
    std::cout << "1. Tab - Switch between objects (sphere, cube, torus) to change its reflection/refraction factors" << std::endl;
    std::cout << "2. 0-9 - Adjust reflection factor (0.0 to 1.0)" << std::endl;
    std::cout << "3. Alt + 0-9 - Adjust refraction ratio (0.9 to 1.0)" << std::endl;
    std::cout << "4. Left mouse drag - Orbit camera" << std::endl;
    std::cout << "5. Mouse wheel - Zoom camera" << std::endl;
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
                #include "EnvironmentMap.vert.glsl"
            },

            // Fragment shader
            {
                #include "EnvironmentMap.frag.glsl"
            },

            // Rendering passes
            {
				// Pass 0 (to render background)
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
                            SKYBOX
                        }
                    },

                    // The camera
                    {
                        {{ 3.0f, 2.0f, 0.0f}}
                    },

                    // Lights
                    {
                        
                    },

                    // Textures
                    {

                    },

                    // Setup optional uniforms in the shader
                    NULL
                },
                // Pass 1 (to render the object)
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
                        {
                            SPHERE,
                            {
                                { "material.color", { 1.0f, 1.0f, 1.0f, 1.0f } },
                                { "material.reflectionFactor", { 0.5f } },
                                { "material.eta", { 0.94f } }
                            }
                        },
						{
                            CUBE,
                            {
                                { "material.color", { 1.0f, 1.0f, 1.0f, 1.0f } },
                                { "material.reflectionFactor", { 0.5f } },
                                { "material.eta", { 0.94f } }
                            },
                            glm::translate(glm::mat4(1.0f), {0.0f, 0.0f, 4.0f})
                        },
                        {
                            TORUS,
                            {
                                { "material.color", { 1.0f, 1.0f, 1.0f, 1.0f } },
                                { "material.reflectionFactor", { 0.5f } },
                                { "material.eta", { 0.94f } }
                            },
                            glm::translate(glm::mat4(1.0f), {0.0f, 0.0f, -4.0f})
                        }
                    },

                    // The camera
                    {
                        { { 3.0f, 2.0f, 0.0f } }
                    },

                    // Lights
                    {

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