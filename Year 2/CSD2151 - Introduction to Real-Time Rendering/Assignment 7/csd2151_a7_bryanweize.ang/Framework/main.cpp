/*!*****************************************************************************
\file main.cpp
\author Vadim Surov (vsurov\@digipen.edu)
\co-author Bryan Ang Wei Ze (bryanweize.ang\@digipen.edu)
\par Course: CSD2151
\par Assignment: 9
\date 03/09/2025 (MM/DD/YYYY)
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

// Current object index - to switch between objects
int currentObjectIndex = 0;
const int numObjects = 4;

// FBOs for multi-pass rendering
cg::FBO renderFBO1, renderFBO2, renderFBO3;

// Texture objects for FBOs
GLuint fboTex1 = 0, fboTex2 = 0, fboTex3 = 0;

// Filter options
int filterOption = 4;

/*
   This function serves as the callback parameter for
      the glfwSetKeyCallback function, used in the main function.

   This function serves as the callback parameter for
      the glfwSetKeyCallback function, used in the main function
*/
void keyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
    // Close the app on ESC key pressed event
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(pWindow, GL_TRUE);

        // Switch between camera control types (O for orbiting, W for walking)
        if (key == GLFW_KEY_O) {
            cameraMode = cg::CameraType::ORBITING;
            pScene->camerasSetType(cameraMode);
            std::cout << "Camera mode: Orbiting" << std::endl;
        }
        else if (key == GLFW_KEY_W) {
            cameraMode = cg::CameraType::WALKING;
            pScene->camerasSetType(cameraMode);
            std::cout << "Camera mode: Walking" << std::endl;
        }

        // Switch between objects using TAB key
        if (key == GLFW_KEY_TAB)
        {
            currentObjectIndex = (currentObjectIndex + 1) % numObjects;
            // Toggle visibility of objects
            for (int i = 0; i < numObjects; i++)
            {
                pScene->passes[0].objects[i].isVisible = (i == currentObjectIndex);
            }
            std::cout << "Switched to Object " << (currentObjectIndex + 1) << std::endl;
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
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
        }
        else if (action == GLFW_RELEASE) {
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
    if (firstMouse) {
        prevX = xpos;
        prevY = ypos;
        firstMouse = false;
    }

    double xoffset = xpos - prevX;
    double yoffset = ypos - prevY;

    prevX = xpos;
    prevY = ypos;

    if (mousePressed) {
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
    // Resize all FBOs and update viewport
    renderFBO1.resize(width, height);
    renderFBO2.resize(width, height);
    renderFBO3.resize(width, height);

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
    GLFWwindow* pWindow = glfwCreateWindow(WIDTH, HEIGHT, "Framework", NULL, NULL);
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
        // Initialize FBOs for multi-pass rendering
        renderFBO1 = cg::FBO(cg::FBO::Texture, WIDTH, HEIGHT, &fboTex1);
        renderFBO2 = cg::FBO(cg::FBO::Texture, WIDTH, HEIGHT, &fboTex2);
        renderFBO3 = cg::FBO(cg::FBO::Texture, WIDTH, HEIGHT, &fboTex3);

        // Define the scene and its properties
        cg::Scene scene =
        {
            // Vertex shader
            {
                #include "FiltersAssignment.vert.glsl"
            },

            // Fragment shader
            {
                #include "FiltersAssignment.frag.glsl"
            },

            // Rendering passes
            {
                // Pass 0: Blinn-Phong illumination
                {
                    // Rendering target (first FBO)
                    renderFBO1,

                    // Viewport
                    { 0, 0, WIDTH, HEIGHT },

                    // Color to clear buffers
                    BLACK,

                    // Depth test
                    ENABLE,

                    // Objects - Four different meshes to switch between
                    {
                        // Object 1: Teapot (initially visible)
                        {
                            TEAPOT,
                            MATERIAL_OBJECT,
					        NOTRANSFORM,
					        VISIBLE
                        },

                        // Object 2: Torus (initially invisible)
                        {
                            TORUS,
                            MATERIAL_OBJECT,
                            NOTRANSFORM,
                            INVISIBLE
                        },

                        // Object 3: Sphere (initially invisible)
                        {
                            SPHERE,
                            MATERIAL_OBJECT,
                            NOTRANSFORM,
                            INVISIBLE
                        },

                        // Object 4: Cube (initially invisible)
                        {
                            CUBE,
                            MATERIAL_OBJECT,
                            NOTRANSFORM,
                            INVISIBLE
                        }
                    },

            // The camera (starts in orbiting mode)
            {
                DEFAULT
            },

            // Lights
            {
                DEFAULT
            },

            // No textures needed for this pass
            { },

            // Setup optional uniforms
            [](cg::Program& shader)
            {
            // Set the Pass number to 0 for Blinn-Phong pass
            shader.setUniform("Pass", 0);
        }
    },

            // Pass 1: Horizontal blur filter
            {
                // Rendering target (second FBO)
                renderFBO2,

                // Viewport
                { 0, 0, WIDTH, HEIGHT },

                // Color to clear buffers
                { },

                // No depth test needed for post-processing
                DISABLE,

                // Object - full screen quad
                {
                    {
                        QUAD
                    }
                },

            // No camera needed for post-processing
            {
                
            },

            // No lights needed for post-processing
            {
                
            },

            // The texture from the first pass
            {
                { fboTex1, GL_TEXTURE0, GL_TEXTURE_2D }
            },

            // Setup uniforms for vertical blur (using Texture for the input)
            [](cg::Program& shader)
            {
                shader.setUniform("Pass", 1);
                shader.setUniform("Texture", 0);
            }
        },

            // Pass 2: Vertical blur filter
            {
                // Rendering target (third FBO)
                renderFBO3,

                // Viewport
                { 0, 0, WIDTH, HEIGHT },

                // Color to clear buffers
                { },

                // No depth test needed for post-processing
                DISABLE,

                // Object - full screen quad
                {
                    {
                        QUAD
                    }
                },

            // No camera needed for post-processing
            {
                
            },

            // No lights needed for post-processing
            {
                
            },

            // The texture from the second pass
            {
                { fboTex2, GL_TEXTURE0, GL_TEXTURE_2D }
            },

            // Setup uniforms for horizontal blur
            [](cg::Program& shader)
            {
                shader.setUniform("Pass", 2);
                shader.setUniform("Texture", 0);
            }
        },

            // Pass 3: Edge detection filter (final pass to screen)
            {
                // Rendering target (default framebuffer)
                DEFAULT,

                // Viewport
                { 0, 0, WIDTH, HEIGHT },

                // Color to clear buffers
                { },

                // No depth test needed for post-processing
                DISABLE,

                // Object - full screen quad
                {
                    {
                        QUAD
                    }
                },

            // No camera needed for post-processing
            {
                
            },

            // No lights needed for post-processing
            {
                
            },

            // The texture from the third pass
            {
                { fboTex3, GL_TEXTURE0, GL_TEXTURE_2D }
            },

            // Setup uniforms for edge detection
            [](cg::Program& shader)
            {
                shader.setUniform("Pass", 3);
                shader.setUniform("Texture", 0);
                shader.setUniform("Option", 4); // Option 4: Apply both blur and edge detection
                shader.setUniform("EdgeThreshold", 0.03f); // Edge detection threshold
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