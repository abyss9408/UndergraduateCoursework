/*!*****************************************************************************
\file main.cpp
\author Vadim Surov (vsurov\@digipen.edu)
\co-author Bryan Ang Wei Ze (bryanweize.ang\digipen.edu)
\par Course: CSD2151
\par Assignment: 8
\date 03/02/2025 (MM/DD/YYYY)
\brief This file has definitions of the main function
       along with global variables and sub-functions used in main.
*******************************************************************************/

#include <iostream> // For standard input/output operations
#include "common/debug.h" // Custom debugging utilities
#include "common/cg.h" // Common computer graphics utilities
#include <GLFW/glfw3.h> // For OpenGL and windowing

// Global variable for the scene object
cg::Scene* pScene = nullptr;
bool lightControl = false; // When true, mouse controls light instead of camera
cg::CameraType cameraType = cg::CameraType::ORBITING; // Default camera mode

// Mouse tracking variables
double lastX = 0.0;
double lastY = 0.0;
bool firstMouse = true;
bool leftMousePressed = false;

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
            glfwSetWindowShouldClose(pWindow, GLFW_TRUE);
        }
        // Toggle light control with left shift key
        else if (key == GLFW_KEY_LEFT_SHIFT)
        {
            lightControl = true;
        }
        // Switch camera type with W (walking) and O (orbiting) keys
        else if (key == GLFW_KEY_W)
        {
            cameraType = cg::CameraType::WALKING;
            pScene->camerasSetType(cameraType);
        }
        else if (key == GLFW_KEY_O)
        {
            cameraType = cg::CameraType::ORBITING;
            pScene->camerasSetType(cameraType);
        }
    }
	else if (action == GLFW_RELEASE)
    {
        if (key == GLFW_KEY_LEFT_SHIFT)
        {
            lightControl = false;
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
            leftMousePressed = true;
            firstMouse = true; // Reset first mouse flag
        }
        else if (action == GLFW_RELEASE)
        {
            leftMousePressed = false;
        }
    }
}

/*
   This function serves as the callback parameter for
       the glfwSetCursorPosCallback function, used in the main function
*/
void cursorPosCallback(GLFWwindow* pWindow, double xpos, double ypos)
{
    if (!leftMousePressed) // Only update when left button is pressed
        return;

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
        return;
    }

    double xoffset = xpos - lastX;
    double yoffset = ypos - lastY;

    lastX = xpos;
    lastY = ypos;

    // Apply sensitivity adjustment
    xoffset *= 0.1;
    yoffset *= 0.1;

    // Update either camera or light based on current control mode
    if (lightControl)
    {
        pScene->lightsOnCursor(xoffset, yoffset, &pScene->shader);
    }
    else
    {
        pScene->camerasOnCursor(xoffset, yoffset, &pScene->shader);
    }
}

/*
   This function serves as the callback parameter for
       the glfwSetScrollCallback function, used in the main function
*/
void scrollCallback(GLFWwindow* pWindow, double xoffset, double yoffset)
{
    // Update either camera or light zoom based on current control mode
    if (lightControl)
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
    GLFWwindow* pWindow = glfwCreateWindow(WIDTH, HEIGHT, "NightSky", NULL, NULL);
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
	std::cout << "- Press W or O to switch between the camera movement types (orbiting or walking)" << std::endl;
	std::cout << "- Hold left shift key to control the light instead of the camera" << std::endl;
	std::cout << "- Use the left mouse button or scroll wheel to control the camera or light" << std::endl;
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
        cg::Scene::Texture texture =
        {
            "night_negx.png", "night_posx.png",
            "night_posy.png", "night_negy.png",
            "night_posz.png", "night_negz.png"
        };

        // Define camera with initial position and FOV
        cg::Camera camera = { cameraType, { 0.0f, 1.0f, 5.0f }, { 0.0f, 0.0f, 0.0f }, 75.0f };

        // Define the scene and its properties
        cg::Scene scene =
        {
            // Vertex shader
            {
                #include "NightSkyAssignment.vert.glsl"
            },

            // Fragment shader
            {
                #include "NightSkyAssignment.frag.glsl"
            },

            // Rendering passes
            {
                // Pass 0 (render the skybox)
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
                            SKYBOX
                        }
                    },

                    // The camera
                    {
                        camera
                    },

                    // Lights
                    {
                        
                    },

                    // Textures
                    {
                        texture
                    },

                    // Setup optional uniforms in the shader
                    NULL
                },
                
                // Pass 1 (render the objects)
                {
                    // Rendering target
                    DEFAULT,

                    // Viewport
                    { 0, 0, WIDTH, HEIGHT },

                    // Color to clear buffers
                    { },

                    // Depth test
                    { },

                    // Objects
                    {
                        {
                            PLANE,
                            MATERIAL_CHECKERBOARD,
                            glm::translate(glm::scale(glm::mat4(1.0f), {10.0f, 1.0f, 10.0f}), {0.0f, -1.01f, 0.0f})
                        },
                        {
                            SPHERE,
                            MATERIAL_PBR_PHONG,
                            glm::translate(glm::mat4(1.0f), {-3.0f, 0.0f, 0.0f})
                        },
                        {
                            TEAPOT,
                            MATERIAL_CARTOON
                        },
                        {
                            CUBE,
                            MATERIAL_DISCARD,
                            glm::translate(glm::mat4(1.0f), {3.0f, 0.0f, 0.0f})
                        }
                    },

                    // The camera
                    {
                        camera
                    },

                    // Lights
                    {
                        { {0.0f, 5.0f, 2.0f} }
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