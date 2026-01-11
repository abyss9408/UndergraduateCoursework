////////////////////////////////////////////////////////////////////////
// The scene class contains all the parameters needed to define and
// draw a simple scene, including:
//   * Geometry
//   * Light parameters
//   * Material properties
//   * viewport size parameters
//   * Viewing transformation values
//   * others ...
//
// Some of these parameters are set when the scene is built, and
// others are set by the framework in response to user mouse/keyboard
// intewractions.  All of them can be used to draw the scene.

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <random>

#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>
using namespace gl;

#include <glu.h>                // For gluErrorString

#include "framework.h"
#include "shapes.h"
#include "object.h"
#include "texture.h"
#include "transform.h"

#if 0
std::random_device randomSeed;
std::mt19937_64 RG(randomSeed());
#else
std::string seed = "A constant seed.";
std::seed_seq constSeed (seed.begin(),seed.end());
std::mt19937_64 RG(constSeed);
#endif

std::uniform_real_distribution<> myrandom(0.0, 1.0);
std::uniform_real_distribution<> BALrandom(-1.0, 1.0);
// Call myrandom(RG) to get a uniformly distributed random number in [0,1].

const float rad = M_PI/180.0f;    // Convert degrees to radians

mat4 Identity;

const float grndSize = 60.0;    // Island radius;  Minimum about 20;  Maximum 1000 or so
const float grndOctaves = 4.0;  // Number of levels of detail to compute
const float grndFreq = 0.03;    // Number of hills per (approx) 50m
const float grndPersistence = 0.03; // Terrain roughness: Slight:0.01  rough:0.05
const float grndLow = -3.0;         // Lowest extent below sea level
const float grndHigh = 5.0;        // Highest extent above sea level

////////////////////////////////////////////////////////////////////////
// This macro makes it easy to sprinkle checks for OpenGL errors
// throughout your code.  Most OpenGL calls can record errors, and a
// careful programmer will check the error status *often*, perhaps as
// often as after every OpenGL call.  At the very least, once per
// refresh will tell you if something is going wrong.
#define CHECKERROR {GLenum err = glGetError(); if (err != GL_NO_ERROR) { fprintf(stderr, "OpenGL error (at line scene.cpp:%d): %s\n", __LINE__, gluErrorString(err)); exit(-1);} }

// Create an RGB color from human friendly parameters: hue, saturation, value
vec3 HSV2RGB(const float h, const float s, const float v)
{
    if (s == 0.0)
        return vec3(v,v,v);

    int i = (int)(h*6.0) % 6;
    float f = (h*6.0f) - i;
    float p = v*(1.0f - s);
    float q = v*(1.0f - s*f);
    float t = v*(1.0f - s*(1.0f-f));
    if      (i == 0)  return vec3(v,t,p);
    else if (i == 1)  return vec3(q,v,p);
    else if (i == 2)  return vec3(p,v,t);
    else if (i == 3)  return vec3(p,q,v);
    else if (i == 4)  return vec3(t,p,v);
    else   /*i == 5*/ return vec3(v,p,q);
}

////////////////////////////////////////////////////////////////////////
// InitializeScene is called once during setup to create all the
// textures, shape VAOs, and shader programs as well as setting a
// number of other parameters.
int total(int p, int e, int g) { return (4*p*p)*(2*e+1)*(2*e+1) + 2*g*g; }
void Scene::InitializeScene()
{
    while (total(polyCount,ellipsoidCount, grndCount) <= GOAL) {
        polyCount++;
        
        if (total(polyCount,ellipsoidCount+1, grndCount) <= GOAL) {
            ellipsoidCount += 1; }
        if (total(polyCount,ellipsoidCount, grndCount+10) <= GOAL) {
            grndCount += 10; }
        //printf("%d %d %d: %d\n", polyCount, ellipsoidCount, grndCount,
        //       total(polyCount,ellipsoidCount, grndCount));
        }
    
    glEnable(GL_DEPTH_TEST);
    CHECKERROR;

    // Set initial light parameters
    lightSpin = 150.0;
    lightTilt = 5.0;
    lightDist = 100.0;
    
    w_down = false;
    s_down = false;
    a_down = false;
    d_down = false;
    nav = true;
    spin = 0.0;
    tilt = 15.33;
    eye = vec3(0.0, -5.0, 1.5);
    speed = 300.0/30.0;
    last_time = glfwGetTime();
    tr = vec3(0.0, 0.0, 25.0);
   
    ry = 0.4;
    front = 0.5;
    back = 5000.0;

    objectRoot = new Object(NULL, nullId);

    CHECKERROR;
    // Create the lighting shader program from source code files.
    lightingProgram = new ShaderProgram();
    lightingProgram->AddShader("lightingPhong.vert", GL_VERTEX_SHADER);
    lightingProgram->AddShader("lightingPhong.frag", GL_FRAGMENT_SHADER);

    glBindAttribLocation(lightingProgram->programId, 0, "vertex");
    glBindAttribLocation(lightingProgram->programId, 1, "vertexNormal");
    glBindAttribLocation(lightingProgram->programId, 2, "vertexTexture");
    glBindAttribLocation(lightingProgram->programId, 3, "vertexTangent");
    lightingProgram->LinkProgram();
    
    // Create all the Polygon shapes
    proceduralground = new ProceduralGround(grndSize, grndCount,
                                     grndOctaves, grndFreq, grndPersistence,
                                     grndLow, grndHigh);
    
    Shape* SpherePolygons = new SphereObj(polyCount);
    Shape* SeaPolygons = new PlaneObj(2000.0, 1);
    Shape* GroundPolygons = proceduralground;

    CHECKERROR;
    // Various colors used in the subsequent models
    vec3 woodColor(87.0/255.0, 51.0/255.0, 35.0/255.0);
    vec3 brickColor(134.0/255.0, 60.0/255.0, 56.0/255.0);
    vec3 floorColor(6*16/255.0, 5.5*16/255.0, 3*16/255.0);
    vec3 brassColor(0.5, 0.5, 0.1);
    vec3 grassColor(62.0/255.0, 102.0/255.0, 38.0/255.0);
    vec3 waterColor(0.3, 0.3, 1.0);

    vec3 black(0.0, 0.0, 0.0);
    vec3 brightSpec(0.5, 0.5, 0.5);
    vec3 polishedSpec(0.3, 0.3, 0.3);
 
    // Creates all the models from which the scene is composed.  Each
    // is created with a polygon shape (possibly NULL), a
    // transformation, and the surface lighting parameters Kd, Ks, and
    // alpha.

    //sky        = new Object(SpherePolygons, skyId, black, black, 0);
    ground     = new Object(GroundPolygons, groundId, grassColor, black, 1);
    sea        = new Object(SeaPolygons, seaId, waterColor, brightSpec, 120);
    //objects    = new Object(NULL, nullId);

    for (int i=-ellipsoidCount;  i<=ellipsoidCount;  i++) {
        for (int j=-ellipsoidCount;  j<=ellipsoidCount;  j++) {
            vec3 hue = HSV2RGB(myrandom(RG), 1.0, 1.0);
            float r = 0.5*(0.5+2*myrandom(RG));
            float h = 2*myrandom(RG);
            vec3 s(r, r, h);
            int v = (i==0 && j==0) ? 0 : 1;
            vec3 t(5*i + v*2*BALrandom(RG),
                        5*j + v*2*BALrandom(RG),
                        0.0);
            t[2] = proceduralground->HeightAt(t[0], t[1]) + s[2]*0.9;
            Object* ellipsoid = new Object(SpherePolygons, 4, hue, vec3(0.5), 120);
            objectRoot->add(ellipsoid, Translate(t)*Scale(s)); } }

    // Scene is composed of some scene objects, and the sky, ground, and sea
    objectRoot->add(ground);
    objectRoot->add(sea); 

    CHECKERROR;

    // At this point, the scene is built and all the objects have been
    // sent to the graphics card as Vertex Array Objects. Let's count
    // the number of triangles.
    triangleCount = 0;
    for (INSTANCE instance : objectRoot->instances) {
        Object* object = instance.first;
        std::vector<ivec3>& Tri = object->shape->Tri; // The object's list of triangles
        triangleCount += Tri.size();        }

    // Time to invoke the student code.
    studentCode = new StudentCode();

    for (auto inst : objectRoot->instances) {
        Object* object =  inst.first;  // The object ...
        mat4& iTr =  inst.second; // ... and its instance transformation
        std::vector<vec4>& Pnt = object->shape->Pnt;
        std::vector<ivec3>& Tri = object->shape->Tri;
        std::vector<vec3>& Nrm = object->shape->Nrm;
        studentCode->ObjectTriangles(Pnt, Nrm, Tri, iTr, object); }
    studentCode->EndOfTriangles();

    fsq = new PlaneObj(1.0, 1);

    CHECKERROR;
    raytraceProgram = new ShaderProgram();
    raytraceProgram->AddShader("rt.vert", GL_VERTEX_SHADER);
    raytraceProgram->AddShader("rt.frag", GL_FRAGMENT_SHADER);
    raytraceProgram->LinkProgram();

}

void Scene::DrawGUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (ImGui::BeginMainMenuBar()) {
        ImGui::Text(" %.1f FPS", ImGui::GetIO().Framerate);
        ImGui::Text(" %d triangles", triangleCount);
        if (ImGui::BeginMenu("Debug")) {
            studentCode->DrawGui(); // Draw the student's GUI under menu item "Debug".
            ImGui::EndMenu(); } 
        ImGui::EndMainMenuBar(); } 

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Scene::BuildTransforms()
{
    
    // Work out the eye position as the user move it with the WASD keys.
    float now = glfwGetTime();
    float stepDist = (now-last_time)*speed;
    last_time = now;
    vec3 stepDir;
    if (w_down)
        stepDir =  vec3(sin(spin*rad), cos(spin*rad), 0.0);
    if (s_down)
        stepDir = -vec3(sin(spin*rad), cos(spin*rad), 0.0);
    if (d_down)
        stepDir =  vec3(cos(spin*rad), -sin(spin*rad), 0.0);
    if (a_down)
        stepDir = -vec3(cos(spin*rad), -sin(spin*rad), 0.0);

    if (studentCode->IsStepLegal(eye-vec3(0,0,0.5), stepDir, stepDist)) {
        eye += stepDist*stepDir;
        eye[2] = proceduralground->HeightAt(eye[0], eye[1]) + 1.5; // Set the height of the eye
        }

    CHECKERROR;

    if (nav)
        WorldView = Rotate(0, tilt-90)*Rotate(2, spin) *Translate(-eye[0], -eye[1], -eye[2]);
    else
        WorldView = Translate(tr[0], tr[1], -tr[2]) *Rotate(0, tilt-90)*Rotate(2, spin);
    WorldProj = Perspective((ry*width)/height, ry, front, (mode==0) ? 1000 : back);

}

void Scene::TraceScene()
{
    glfwGetFramebufferSize(window, &width, &height);
    
    // Calculate the light's position from lightSpin, lightTilt, lightDist
    lightPos = vec3(lightDist*cos(lightSpin*rad)*sin(lightTilt*rad),
                         lightDist*sin(lightSpin*rad)*sin(lightTilt*rad), 
                         lightDist*cos(lightTilt*rad));
    
    BuildTransforms();

    // The lighting algorithm needs the inverse of the WorldView matrix
    viewInverse = inverse(WorldView);
    projInverse = inverse(WorldProj);
    studentCode->SetTransforms(viewInverse, projInverse);
    
    if (raysize != width*height || rtTextureId==0) {
        CHECKERROR;
        //delete[] rayimage;
        raysize = width*height;
        rayimage = new float[3*raysize];
        
        if (!rtTextureId)
            glGenTextures(1, &rtTextureId);   // Get an integer id for this texture from OpenGL
        glBindTexture(GL_TEXTURE_2D, rtTextureId);
        glTexImage2D(GL_TEXTURE_2D, 0, (GLint)GL_RGB32F, width, height, 0, 
                     GL_RGB, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        CHECKERROR;  }
            
    raytraceProgram->UseShader();
    int programId = raytraceProgram->programId;

    CHECKERROR;
    // Set the viewport, and clear the screen
    glViewport(0, 0, width, height);
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);

    studentCode->GenerateRayTracedImage(width, height, rayimage, lightPos);

    CHECKERROR;
    // Bind the texture in unit 0, update its contents, and draw the quad to display it.
    glActiveTexture((gl::GLenum)((int)GL_TEXTURE0));
    glBindTexture(GL_TEXTURE_2D, rtTextureId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, rayimage);
    
    int loc = glGetUniformLocation(programId, "colorBuf");
    glUniform1i(loc, 0);

    // Draw the FSQ
    fsq->DrawVAO();

    // Unbind the texture
    glBindTexture(GL_TEXTURE_2D, 0);

    // Turn off the shader
    raytraceProgram->UnuseShader();

    CHECKERROR;
}

////////////////////////////////////////////////////////////////////////
// Procedure DrawScene is called whenever the scene needs to be
// drawn. (Which is often: 60 times per second or more is common.)
void Scene::DrawScene()
{
    // Set the viewport
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    CHECKERROR;
    // Calculate the light's position from lightSpin, lightTilt, lightDist
    lightPos = vec3(lightDist*cos(lightSpin*rad)*sin(lightTilt*rad),
                         lightDist*sin(lightSpin*rad)*sin(lightTilt*rad), 
                         lightDist*cos(lightTilt*rad));

    BuildTransforms();

    // The lighting algorithm needs the inverse of the WorldView matrix
    viewInverse = inverse(WorldView);
    projInverse = inverse(WorldProj);
    studentCode->SetTransforms(viewInverse, projInverse);
    
    CHECKERROR;
    int loc, programId;

    ////////////////////////////////////////////////////////////////////////////////
    // Lighting pass
    ////////////////////////////////////////////////////////////////////////////////
    
    // Choose the lighting shader
    lightingProgram->UseShader();
    programId = lightingProgram->programId;

    // Set the viewport, and clear the screen
    glViewport(0, 0, width, height);
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);


    loc = glGetUniformLocation(programId, "WorldProj");
    glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(WorldProj));
    loc = glGetUniformLocation(programId, "WorldView");
    glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(WorldView));
    loc = glGetUniformLocation(programId, "WorldInverse");
    glUniformMatrix4fv(loc, 1, GL_FALSE, Pntr(viewInverse));
    loc = glGetUniformLocation(programId, "lightPos");
    glUniform3fv(loc, 1, &(lightPos[0]));   
    loc = glGetUniformLocation(programId, "mode");
    glUniform1i(loc, mode);
    CHECKERROR;

    // Draw all objects
    CHECKERROR;

    glPolygonOffset(1.0, 1.0);
    glEnable(GL_POLYGON_OFFSET_FILL);
    objectRoot->Draw(lightingProgram, Identity);
    glDisable(GL_POLYGON_OFFSET_FILL);

    CHECKERROR;

    studentCode->DrawDebug(programId);
    
    // Turn off the shader
    lightingProgram->UnuseShader();

    ////////////////////////////////////////////////////////////////////////////////
    // End of Lighting pass
    ////////////////////////////////////////////////////////////////////////////////
}

void Scene::DestroyScene()
{
    studentCode->Destroy();
}
