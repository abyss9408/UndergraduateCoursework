/*!
@file    main.cpp
@author  Prasanna Ghali       (pghali@digipen.edu)
         Parminder Singh   (parminder.singh@digipen.edu)

All content (c) 2002 DigiPen Institute of Technology, all rights reserved.
*//*__________________________________________________________________________*/

/*                                                                   includes
----------------------------------------------------------------------------- */

//@todo Uncomment each of the following three lines as you progress through
// the problems [rasterization, first-person camera, view-frustum culling, frustum
// clipping] assigned for the final project. You can complete these tasks in any
// order. My implementations of 3D picking and shadows [not assigned but that
// work in the sample] are hidden from you. However, they rely on auxiliary
// functions gfxModel::ComputeModelBVSphere() and gfxSphere::Transform() - both
// these functions are defined in file YourBoundingSphere.cpp.
// You must implement these functions first for all tasks except the
// first-person camera and rasterizer to work correctly!!!
#define YOUR_RASTERIZER
#define YOUR_CAMERA
#define YOUR_CULLER_CLIPPER

#include  "../../GfxLib/gfx/GFX.h"
#include "YourRasterizer.h"
#include "YourCullerClipper.h"
#include "YourCamera.h"
#include <cstdlib>
#include <sstream>
#include <string>
#include <iostream>
#include <chrono>
#include <thread> // For std::this_thread::sleep_for

/*                                                                  constants
----------------------------------------------------------------------------- */

// default window parameters
int constexpr			sWindowWidth = 1280;
int constexpr			sWindowHeight = 780;
size_t constexpr	sWindowX = 0;
size_t constexpr	sWindowY = 0;

// default viewport parameters
float constexpr		sViewportWidth = static_cast<float>(sWindowWidth);
float constexpr	  sViewportHeight = static_cast<float>(sWindowHeight);
float const				sViewportX = 0.f;
float const				sViewportY = 0.f;

// default camera parameters
gfxVector3 const	sCameraPosition(0.f, 5.f, 20.f);
gfxVector3 const	sCameraTarget(0.f, 5.f, -1.f);
gfxVector3 const	sCameraUp(0.f, 1.f, 0.f);

gfxPlane const		sReceiverPlane(0.f, 1.f, 0.f, 0.f);

// default perspective projection parameters
float const				sFOVY = 45.f; // in degrees
float const				sAspectRatio = sViewportWidth / sViewportHeight;
float const				sNearDist = 1.f;
float const				sFarDist = 1000.f;

// camera displacements
float const				sCamStep = 0.5f;
float const				sCamOrientationDamper = 1.f / 100.f;

// frame rate
float const				sMaxFps = 60.f;	// throttled game fps

/*                                                    objects with file scope
----------------------------------------------------------------------------- */

//gfxCamera* sPCam = 0; // pointer to either built-in or YourCamera

// your implementations
#ifdef YOUR_RASTERIZER
YourRasterizer* gYourRasterizer = 0;
#endif

#ifdef YOUR_CAMERA
gfxCamera* sPCam = new YourCamera();
#else
gfxCamera* sPCam = new gfxCamera();
#endif

#ifdef YOUR_CULLER_CLIPPER
YourClipper* sYourClipper = 0;
#endif

// mouse
int               sMX = 0;
int               sMY = 0;

// toggles
bool              sMoveF = false;
bool              sMoveB = false;
bool              sMouseLook = false;
bool              sWireframe = false;
gfxInterpMode     sInterpMode = GFX_LINEAR_INTERP; //GFX_HYPERBOLIC_INTERP;

// steps
float							sAngle = PI;
float const				sMeshSpinStep = 0.01f;
int								sPickID = -1;
int								sReceiverIdx;
int								sLightIdx;

// device
gfxGraphicsPipe* sSWPipe = 0; // software graphics pipe
gfxObjectList     sScene;      // object list

// models
gfxModel* sCubeModelP = 0;;
gfxModel* sSphereHRMP = 0;
gfxModel* sSphereLRMP = 0;
gfxModel* sGndModelP = 0;

/*                                                  functions with file scope
----------------------------------------------------------------------------- */
static void InitScene();
static void DrawScene(std::stringstream&);
static void UpdateScene();

/*                                                                  functions
----------------------------------------------------------------------------- */

/*  _________________________________________________________________________ */
void UpdateScene()
/*!
        Read user input and updates camera orientation and position
*/
{
    POINT		cursorPos;
    ::GetCursorPos(&cursorPos);
    if (sMouseLook) {
        // Using changes in mouse position between previous and current frame,
        // measure in radians, changes in latitudinal and azimuth angles.
        float  deltaAlpha = static_cast<float>(cursorPos.x - sMX) * sCamOrientationDamper;
        float  deltaBeta = static_cast<float>(cursorPos.y - sMY) * sCamOrientationDamper;

        // Orient.
        sPCam->SetAzimuth(sPCam->GetAzimuth() - deltaAlpha);
        sPCam->SetLatitude(sPCam->GetLatitude() - deltaBeta);

        // Move.
        if (sMoveF) {
            sPCam->Move(0.f, 0.f, sCamStep);
        }
        if (sMoveB) {
            sPCam->Move(0.f, 0.f, -sCamStep);
        }
    }
    else if (sPickID >= 0 && sPickID < static_cast<int>(sScene.size())) {
        gfxVector3 basis(1.f, 0.f, 0.f);
        gfxVector3  basis_x(1.f, 0.f, 0.f);
        gfxVector3  basis_y(0.f, 1.f, 0.f);
        gfxVector3  basis_z(0.f, 0.f, 1.f);
        gfxVector3  pos = sScene[sPickID].GetWorldPosition();

        // Get basis vectors for the current camera.
        sPCam->GetBasisVectors(&basis_x, &basis_y, &basis_z);

        // This allows us to move the object in world space intuitively.
        // Remember to invert basis Y because it increases upwards and
        // screen coordinates increase downward.
        pos += (static_cast<float>(cursorPos.x - sMX) / 50.f) * basis_x;
        pos += (static_cast<float>(cursorPos.y - sMY) / 50.f) * -basis_y;
        sScene[sPickID].SetWorldPosition(pos.x, ((pos.y <= 1.75f) ? 2.f : pos.y), pos.z);
    }

    for (size_t i = 0; i < sScene.size(); ++i) {
        if (i == sLightIdx) {
            sAngle += 0.05f;
            gfxVector3 pos(static_cast<float>(5.0 * sin(sAngle)),
                sScene[sLightIdx].GetWorldPosition().y,
                static_cast<float>(5.0 * cos(sAngle)) - 5.f);
            sScene[i].SetWorldPosition(pos);
        }

        sScene[i].Update();
    }

    // Update cursor position.
    sMX = cursorPos.x;
    sMY = cursorPos.y;
}

/*  _________________________________________________________________________ */
void DrawScene(std::stringstream& fmt)
/*!
        Sets render states: depth buffer, lighting, shadows, ...
        Passes vertex and triangle lists to transform engine and
        subsequently to pixel engine

    @param fmt -->  Any text to write to viewport.
*/
{
    // set up lookat, perspective, and NDC-to-viewport transforms
    sPCam->SetLookAtMatrix();
    sSWPipe->SetViewportMatrix(sViewportX, sViewportY, sViewportWidth, sViewportHeight);

    sSWPipe->SetMatrixMode(GFX_PROJECTION);
    sSWPipe->LoadIdentity();
    sSWPipe->MultMatrix(sPCam->GetProjectionMatrix());	// CTM for projection stack is perspective transform

    // compute view-frame frustum plane equations for culling
    sSWPipe->SetFrustum(sSWPipe->GetClipper()->ComputeFrustum(sPCam->GetProjectionMatrix()));

    // enable lighting, gourard shading, depth buffering, ...
    sSWPipe->EnableLighting(false);
    sSWPipe->EnableShadows(false);
    sSWPipe->EnableDepthTest(true);
    sSWPipe->SetShadeMode(GFX_SMOOTH);
    sSWPipe->SetRenderMode(sWireframe ? GFX_WIREFRAME : GFX_FILLED);
    sSWPipe->SetRenderColor(0.f, 0.f, 0.f, 1.f); // use black color for wireframe
    sSWPipe->SetInterpMode(sInterpMode);

    sSWPipe->SetMatrixMode(GFX_MODELVIEW);
    sSWPipe->LoadMatrix(sPCam->GetLookAtMatrix());	// CTM for model-view stack is view matrix
    sSWPipe->SetLightPos(sScene[sLightIdx].GetWorldPosition());
    sSWPipe->SetShadowPlane(sReceiverPlane, sPCam->GetPosition());
    // Start processing primitives rendering.
    sSWPipe->RenderBegin();
    // first, render receiver as usual (with depth buffering, lighting (if lit))
    sScene[sReceiverIdx].Draw(sSWPipe);
    // second, render occluders' shadows above receiver by disabling depth buffering, 
  // lighting, texture mapping, with linear interpolation mode
    sSWPipe->EnableLighting(false);
    sSWPipe->EnableShadows(true);
    sSWPipe->EnableDepthTest(false);
    sSWPipe->SetInterpMode(GFX_LINEAR_INTERP);
    for (int i = 0; i < static_cast<int>(sScene.size()); ++i) {
        if (sScene[i].GetName() == "occluder") {
            sScene[i].Draw(sSWPipe);
        }
    }

    // finally, render rest of scene with lighting and depth buffering enabled,
    // while shadowing is disabled with the user selected interpolation mode 
    sSWPipe->SetInterpMode(sInterpMode);
    sSWPipe->EnableLighting(true);
    sSWPipe->EnableShadows(false);
    sSWPipe->EnableDepthTest(true);
    for (int i = 0; i < static_cast<int>(sScene.size()); ++i) {
        if (i != sReceiverIdx) {
            sScene[i].Draw(sSWPipe);
        }
    }

    sSWPipe->DrawText(0, 0, fmt.str()); // Draw the text.     

    sSWPipe->RenderEnd(); // Finish rendering and present the frame.
}

/*  _________________________________________________________________________ */
void InitScene()
/*!
        Initializes viewing parameters
        Loads models from IFS files.
        Instantiates objects from models into scene
*/
{
    POINT	cursor_pos;
    ::GetCursorPos(&cursor_pos); // Get initial cursor position.
    sMX = cursor_pos.x;
    sMY = cursor_pos.y;

    // Load meshes. The second parameter specifies whether or
    // not to generate normals. Since IFS models have no colors
    // or textures, doing so is the only way we can get nice models.
    sCubeModelP = new gfxModel(gfxMeshImporter_IFS::Import("../content/cube.ifs", true));
    sSphereHRMP = new gfxModel(gfxMeshImporter_IFS::Import("../content/sphere_1010.ifs", true));
    sSphereLRMP = new gfxModel(gfxMeshImporter_IFS::Import("../content/sphere_0505.ifs", true));
    sGndModelP = new gfxModel(gfxMeshGenerator_Plane::Generate(L"../content/floor.png", 20, 200.0f));

    // create scene ...
    gfxVector3 u(1.f, 1.f, 1.f);
    gfxVector3 up(0.f, 1.f, 0.f);
    gfxVector3 w(u ^ up);
    gfxVector3 v(w ^ u);
    u.Normalize();
    v.Normalize();
    w.Normalize();
    gfxVector3 orient[3] = { u, v, w };
    gfxVector3 noRotation[3] = { gfxVector3(1.f, 0.f, 0.f),
                                                                gfxVector3(0.f, 1.f, 0.f),
                                                                gfxVector3(0.f, 0.f, 1.f) };

    // light source
    gfxObject	lightObj("light",
        sCubeModelP,
        gfxTransform(gfxVector3(1.f, 1.f, 1.f),
            noRotation,
            gfxVector3(5.f * sin(sAngle), 10.f, -5.f + 5.f * cos(sAngle))),
        false);
    lightObj.SetDiffuseMat(1.f, 0.f, 0.f);
    sLightIdx = static_cast<unsigned int>(sScene.size());
    sScene.push_back(lightObj);

    // occluders
    gfxObject	occObj1("occluder",
        sSphereLRMP,
        gfxTransform(gfxVector3(1.5f, 1.5f, 1.5f),
            noRotation,
            gfxVector3(0.f, 3.f, -5.f)),
        true);
    occObj1.SetDiffuseMat(0.f, 1.f, 0.f);
    sScene.push_back(occObj1);

    gfxObject	occObj2("occluder",
        sSphereLRMP,
        gfxTransform(gfxVector3(1.5f, 1.5f, 1.5f),
            noRotation,
            gfxVector3(-4.5f, 3.f, -5.f)),
        true);
    occObj2.SetDiffuseMat(0.f, 1.f, 1.f);
    sScene.push_back(occObj2);

    gfxObject	occObj3("occluder",
        sSphereLRMP,
        gfxTransform(gfxVector3(1.5f, 1.5f, 1.5f),
            noRotation,
            gfxVector3(4.5f, 3.f, -5.f)),
        true);
    occObj3.SetDiffuseMat(1.f, 0.f, 0.f);
    sScene.push_back(occObj3);

    gfxObject	occObj4("occluder",
        sCubeModelP,
        gfxTransform(gfxVector3(2.f, 1.f, 1.f),
            orient,
            gfxVector3(-8.5f, 3.f, -5.f)),
        true);
    occObj4.SetDiffuseMat(1.f, 0.f, 0.f);
    sScene.push_back(occObj4);


    gfxVector3	pillarScale(2.f, 2.f, 2.f);
    gfxVector3	leftPillarPos(-20.f, 1.f, 0.f);
    gfxVector3	midPillarPos(0.f, 1.f, 0.f);
    gfxVector3	rightPillarPos(20.f, 1.f, 0.f);
    gfxVector3	ballScale(2.f, 2.f, 2.f);

    // next, set up parallelepiped and spherical pillars - from back to front
    for (int z = -40, idx = 0; z <= 40; z += 8, ++idx) {
        // left parallelepiped pillar
        leftPillarPos.z = static_cast<float>(z);
        gfxObject leftPillar("occluder",
            sCubeModelP,
            gfxTransform(pillarScale, noRotation, leftPillarPos),
            true
        );
        leftPillar.SetDiffuseMat(1.f, 0.f, 0.f);
        sScene.push_back(leftPillar);

        // left spherical pillar
        float	disp = -45.f + ((idx % 2) ? 0.f : 10.f);
        gfxObject leftBall("occluder",
            sSphereLRMP,
            gfxTransform(ballScale, noRotation, gfxVector3(disp, 1.75f, static_cast<float>(z))),
            true
        );
        leftBall.SetDiffuseMat(0.f, 0.f, 1.f);
        sScene.push_back(leftBall);

        midPillarPos.z = static_cast<float>(z);
        gfxObject midPillar("occluder",
            sCubeModelP,
            gfxTransform(pillarScale, noRotation, midPillarPos),
            true
        );
        midPillar.SetDiffuseMat(0.f, 1.f, 0.f);
        sScene.push_back(midPillar);

        rightPillarPos.z = static_cast<float>(z);
        gfxObject	rightPillar("occluder",
            sCubeModelP,
            gfxTransform(pillarScale, noRotation, rightPillarPos),
            true
        );
        rightPillar.SetDiffuseMat(0.f, 0.f, 1.f);
        sScene.push_back(rightPillar);

        disp = 45.f + ((idx % 2) ? 10.f : 0.f);
        gfxObject	rightBall("occluder",
            sSphereLRMP,
            gfxTransform(ballScale, noRotation, gfxVector3(disp, 1.75f, static_cast<float>(z))),
            true
        );
        rightBall.SetDiffuseMat(1.f, 0.f, 0.f);
        sScene.push_back(rightBall);
    }

    // finally, set up receiver plane
    gfxObject ground("receiver",
        sGndModelP,
        gfxTransform(gfxVector3(1.f, 1.f, 1.f), noRotation, gfxVector3(0.f, 0.f, 0.f))
    );
    sReceiverIdx = static_cast<unsigned int>(sScene.size());
    sScene.push_back(ground);
}

/*  _________________________________________________________________________ */
LRESULT CALLBACK MyKeyHandler(HWND wind, UINT msg, WPARAM wp, LPARAM lp)
/*! Keyboard input handler.

    @param wind -->  The window handle.
    @param msg  -->  The message ID.
    @param wp   -->  Message payload one.
    @param lp   -->  Message payload two.

    @return
    Zero if the message was handled, non-zero otherwise.
*/
{
    // Stupid-proofing.
    assert(msg == WM_KEYDOWN);
    switch (wp) {
        // E and D zoom the camera in or out.
    case 'E': sPCam->Move(0.f, 0.f, sCamStep); break;
    case 'D': sPCam->Move(0.f, 0.f, -sCamStep); break;

        // S and F pan the camera right or left.
    case 'S': sPCam->Move(sCamStep, 0.f, 0.f); break;
    case 'F': sPCam->Move(-sCamStep, 0.f, 0.f); break;

        // Q and A pan the camera up or down.
    case 'Q': sPCam->Move(0.f, sCamStep, 0.f); break;
    case 'A': sPCam->Move(0.f, -sCamStep, 0.f); break;

        // W and R yaw the camera.
    case 'W': sPCam->SetAzimuth(sPCam->GetAzimuth() + sCamStep / 3.f); break;
    case 'R': sPCam->SetAzimuth(sPCam->GetAzimuth() - sCamStep / 3.f); break;

        // T and G pitch the camera..
    case 'T': sPCam->SetLatitude(sPCam->GetLatitude() + sCamStep / 3.f); break;
    case 'G': sPCam->SetLatitude(sPCam->GetLatitude() - sCamStep / 3.f); break;

        // Space toggles mouselook.
    case ' ': sMouseLook = !sMouseLook; sPickID = -1; break;

        // V toggles wireframe.
    case 'V': sWireframe = !sWireframe; break;

        // H toggles interpolation mode between hyperbolic and linear
    case 'H':
        sInterpMode = (sInterpMode == GFX_LINEAR_INTERP) ? GFX_HYPERBOLIC_INTERP : GFX_LINEAR_INTERP;
        break;

        // Escape resets the camera.
    case VK_ESCAPE: {
        sPCam->SetPosition(sCameraPosition);
        sPCam->SetTarget(sCameraTarget);
    }
                  break;
    default: {
        // Do nothing.
    }
    }

    return (0);
}

/*  _________________________________________________________________________ */
LRESULT CALLBACK MyMouseHandler(HWND wind, UINT msg, WPARAM wp, LPARAM lp)
/*! Mouse input handler.

    @param wind -->  The window handle.
    @param msg  -->  The message ID.
    @param wp   -->  Message payload one.
    @param lp   -->  Message payload two.

    @return
    Zero if the message was handled, non-zero otherwise.
*/
{
    switch (msg) {
    case WM_LBUTTONDOWN: {
        ::SetCapture(wind);
        sMoveF = true;
        sMoveB = false;

        if (!sMouseLook) {
            sPickID = sSWPipe->GetPicker()->Pick(LOWORD(lp), HIWORD(lp), sScene, sPCam);
        }
    }
                       break;
    case WM_LBUTTONUP: {
        ::ReleaseCapture();
        sMoveF = false;
        sMoveB = false;
        sPickID = -1;
    }
                     break;
    case WM_RBUTTONDOWN: {
        ::SetCapture(wind);
        sMoveF = false;
        sMoveB = true;
    }
                       break;
    case WM_RBUTTONUP: {
        ::ReleaseCapture();
        sMoveF = false;
        sMoveB = false;
    }
                     break;
    default: {
        // Do nothing.
    }
    }

    return (0);
}

class FPSCounter {
public:
    void Start() {
        // Initialize FPS counter
        frameCount = 0;
        startTime = std::chrono::steady_clock::now();
    }

    double FPS() {
        return fps;
    }

    void Frame() {
        // Count each frame
        ++frameCount;

        // Calculate elapsed time since last FPS update
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();

        if (elapsedTime >= 1000) { // Update FPS every 1 second
            // Calculate FPS
            fps = frameCount / (elapsedTime / 1000.0);

            // Print FPS
            //std::cout << "FPS: " << fps << std::endl;

            // Reset counters for the next second
            frameCount = 0;
            startTime = currentTime;
        }
    }

private:
    int frameCount;
    double fps = 0;
    std::chrono::steady_clock::time_point startTime;
};

/*  _________________________________________________________________________ */
int WINAPI WinMain(HINSTANCE thisInst, HINSTANCE prevInst, LPSTR cmdLine, int cmdShow)
/*! Windows application entry point.

    @param thisInst  This application instance.
    @param prevInst  The previous application instance.
    @param cmdLine   Command-line arguments passed to the application.
    @param cmdShow   How to show the initial window.

    @return
    A result code.
*/
{
    // Load GDI+.
    ULONG_PTR                     gdi = 0;
    Gdiplus::GdiplusStartupInput  gdiInput(0, 0, 0);
    Gdiplus::GdiplusStartup(&gdi, &gdiInput, 0);

    // Initialize the cycle-resolution timer.
    // This can be useful for profiling sections of your code.
    // It can't throw, so we're safe calling it outside the try block.
    gfxCycleTimer::Init();

    // Create custom implementation objects.
#ifdef YOUR_RASTERIZER
    gYourRasterizer = new YourRasterizer();
#endif
#ifdef YOUR_CULLER_CLIPPER
    sYourClipper = new YourClipper();
#endif

    // initialize camera
    //sPCam = new gfxCamera();
    sPCam->SetPosition(sCameraPosition);
    sPCam->SetTarget(sCameraTarget);
    sPCam->SetUp(sCameraUp);
    sPCam->SetAspect(sAspectRatio);
    sPCam->SetFOV(sFOVY);
    sPCam->SetNearDist(sNearDist);
    sPCam->SetFarDist(sFarDist);
    sPCam->SetViewport(sViewportX, sViewportY, sViewportWidth, sViewportHeight);
    sPCam->SetPerspMatrix(sFOVY, sAspectRatio, sNearDist, sFarDist);

    // Create and initialize graphics pipe
    sSWPipe = new gfxGraphicsPipe(sWindowX, sWindowY,
        sWindowWidth, sWindowHeight,
        "CSD2101 AY2324 T3 Project");

    // Query the performance counter frequency. We'll use this to calculate our frame time.
    LARGE_INTEGER   freq;
    ::QueryPerformanceFrequency(&freq);

    try {
        // Hook up the controller overrides.
        // Here we tell the device to use our controllers instead of the default.
        // Pass 0 to reset to the default.
        // define graphics models, objects, and viewing parameters
#ifdef YOUR_RASTERIZER
        sSWPipe->UseRasterizer(gYourRasterizer);
#else
        sSWPipe->UseRasterizer(0);
#endif

#ifdef YOUR_CULLER_CLIPPER
        sSWPipe->UseClipper(sYourClipper);
#else
        sSWPipe->UseClipper(0);
#endif

        sSWPipe->UseLighter(0);
        sSWPipe->UsePicker(0);

        // Install event handlers.
        sSWPipe->AssignEventHandler(WM_KEYDOWN, MyKeyHandler);
        sSWPipe->AssignEventHandler(WM_LBUTTONDOWN, MyMouseHandler);
        sSWPipe->AssignEventHandler(WM_LBUTTONUP, MyMouseHandler);
        sSWPipe->AssignEventHandler(WM_RBUTTONDOWN, MyMouseHandler);
        sSWPipe->AssignEventHandler(WM_RBUTTONUP, MyMouseHandler);
        sSWPipe->AssignEventHandler(WM_MOUSEMOVE, MyMouseHandler);

        InitScene();
        std::cout << "Set up complete\n";
        std::stringstream tmp_ss("");
        DrawScene(tmp_ss);

        // throttle game to constant frame rate of 60 frames/second
        LONGLONG	countsPerSec = 0;
        // tick count per second
        ::QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
        // tick count per frame 
        DWORD			countsPerFrame = static_cast<DWORD>(countsPerSec / sMaxFps);
        LONGLONG	countsToWaitForEndOfFrame = 0;
        // tick count at start of frame
        ::QueryPerformanceCounter((LARGE_INTEGER*)&countsToWaitForEndOfFrame);
        LONGLONG	countsAfterEndofTask = 0;
        float			currFps = 60.f;

        FPSCounter fpsCounter; // currFPS is buggy, using FPSCounter class to report FPS on window's title
        fpsCounter.Start();

        bool	bQuitMsg = false;
        while (!bQuitMsg) {
            std::stringstream  fmt;
            fmt.precision(2);

            static bool bNewFrame = true;
            MSG					msg;
            if (::PeekMessage(&msg, 0, 0, 0, PM_REMOVE)) { // If a quit message, break the loop
                if (msg.message == WM_QUIT) {
                    bQuitMsg = true;
                }
                else {
                    ::TranslateMessage(&msg);
                    ::DispatchMessage(&msg);
                }
            }

            if (bNewFrame) {
                UpdateScene();
                bNewFrame = false;
            }

            // Get performance counter value at end of task (frame)
            ::QueryPerformanceCounter((LARGE_INTEGER*)&countsAfterEndofTask);
            if (countsAfterEndofTask > countsToWaitForEndOfFrame) {
                static int sCtr = 1;

                fmt << "Mouselook " << (sMouseLook ? "ON" : "OFF") << " Object Cnt: " << sScene.size()
                    << "; Culled: " << sSWPipe->GetCullCount() << "; FPS: " << fpsCounter.FPS() << "; sPickID: " << sPickID;
                DrawScene(fmt);

                // set counter for next frame
                countsToWaitForEndOfFrame += countsPerFrame;

                // if current frame takes too much time, get ahead
                if (countsToWaitForEndOfFrame < countsAfterEndofTask) {
                    countsToWaitForEndOfFrame = countsAfterEndofTask + countsPerFrame;
                }

                fpsCounter.Frame();
                if (sCtr == 30) {
                    currFps = static_cast<float>(-countsPerSec) /
                        static_cast<float>(countsAfterEndofTask - countsToWaitForEndOfFrame);
                    sCtr = 0;
                }
                else {
                    ++sCtr;
                }

                bNewFrame = true;
            }

            // For timing, we calculate frame time, not FPS.
            // FPS is a non-linear measure and not really a
            // good performance metric. For more information
            // see http://www.mvps.org/directx/articles/fps_versus_frame_time.htm
        }	// end while()
    }
    catch (std::exception const& e) {
        ::MessageBox(0, e.what(), "Error", MB_OK);
    }

    // Exit.
    Gdiplus::GdiplusShutdown(gdi);
    ::ReleaseCapture();

#ifdef YOUR_CULLER_CLIPPER
    delete sYourClipper;
#endif
#ifdef YOUR_RASTERIZER
    delete gYourRasterizer;
#endif

    delete	sPCam;
    delete	sCubeModelP;
    delete	sSphereLRMP;
    delete	sSphereHRMP;
    delete	sGndModelP;

    return (0);
}

int main() {
    return WinMain(GetModuleHandle(NULL), NULL, GetCommandLineA(), SW_SHOWNORMAL);
}
