/**
 * File: app.ts
 *
 * Author: Bryan Ang Wei Ze (2301397)
 *
 * Resources used:
 * Claude Code: Generate draft code snippets and explore
 * alternative approaches, which were reviewed and
 * modified before integration.
 * 
 * Declaration:
 * I affirm that this file represents my own work and understanding.
 * Except where explicitly acknowledged, it has not been copied from
 * other students or external repositories. Any use of external
 * resources, tools, or assistance (including AI-based tools) has been
 * appropriately referenced.
 *
 * When using AI tools, I affirm that I have not delegated the
 * completion of this assignment to an autonomous or agentic coding
 * system in a manner where I did not meaningfully read, reason about,
 * or engage with the source code being produced (for example, by
 * submitting code generated end-to-end by an AI agent 
 * without inspection, understanding, or modification).
 *
 * I take full responsibility for the integrity of this work under the
 * Singapore Institute of Technology Code of Conduct for Learners:
 * https://www.singaporetech.edu.sg/sitlearn/sites/sitlearn/files/2024-12/Code%20of%20Conduct%20for%20Learners.pdf
**/

import { Engine, FreeCamera, HemisphericLight, MeshBuilder, Scene, Vector3, VideoDome, WebXRDefaultExperience, SceneLoader, ActionManager, ExecuteCodeAction, PointerDragBehavior, PointLight } from "@babylonjs/core";
import { AdvancedDynamicTexture, TextBlock } from "@babylonjs/gui";
import "@babylonjs/loaders/glTF";

export class App {
  // the BabylonJS engine
  private engine: Engine;

  /**
   * Constructor to create the App object with an engine.
   * @param engine The Babylon engine to use for the application.
   */
  constructor(engine: Engine) {
    this.engine = engine;
  }

  /**
   * Create the scene.
   * @returns A promise that resolves when the application is done running.
   * @remarks This is the main entry point for the application.
   */
  async createScene() {
    // create the scene
    const scene = new Scene(this.engine);
    // print the hello message
    console.log("Hello, immersive world!");

    // Create a VideoDome for immersive background environment
    const videoDome = new VideoDome(
      "video dome",
      "video.mp4",
      {
        resolution: 32,
        autoPlay: true,
        loop: true
      },
      scene
    );

    // Creates and positions a free camera
    const camera = new FreeCamera("camera1",
        new Vector3(0, 5, -15),
        scene);
    // Targets the camera to scene origin
    camera.setTarget(Vector3.Zero());

    // Creates a light, aiming 0,1,0
    const light = new HemisphericLight("light",
        new Vector3(0, 10, 0),
        scene);

    // Dim the light
    light.intensity = 0.8;

    // Create a sphere
    const sphere = MeshBuilder.CreateSphere("sphere", {diameter: 2, segments: 64}, scene);
    sphere.position.y = 1;

    // Add interaction to the sphere using ActionManager
    sphere.actionManager = new ActionManager(scene);
    sphere.actionManager.registerAction(
      new ExecuteCodeAction(
        ActionManager.OnPickTrigger,
        () => {
          // Change sphere scale when picked
          sphere.scaling = sphere.scaling.scale(1.5);
        }
      )
    );

    // Make the sphere draggable
    const dragBehavior = new PointerDragBehavior({});
    sphere.addBehavior(dragBehavior);

    // Load the dragon model asynchronously
    SceneLoader.ImportMeshAsync(
      "",
      "https://raw.githubusercontent.com/BabylonJS/Assets/master/meshes/Georgia-Tech-Dragon/",
      "dragon.glb",
      scene
    ).then((result) => {
      // Get the root mesh (first mesh is typically the container)
      const dragon = result.meshes[0];
      dragon.name = "dragon";
      dragon.position = new Vector3(0, 0, 10);
      dragon.scaling = new Vector3(30, 30, 30);

    }).catch((error) => {
      console.error("Error loading dragon:", error);
    });

    // Create a plane (ground)
    const plane = MeshBuilder.CreatePlane("hello plane", {size: 15}, scene);
    plane.rotation.x = Math.PI / 2; // Rotate to be horizontal like a ground
    plane.position.y = 0;

    // Create ground mesh for XR teleportation
    const ground = MeshBuilder.CreateGround("ground", {width: 10, height: 10}, scene);
    ground.position.y = 0;

    // Create GUI
    const advancedTexture = AdvancedDynamicTexture.CreateFullscreenUI("hello texture", true, scene);

    // Create text block
    const textBlock = new TextBlock("hello text");
    textBlock.text = "Hello, XR!";
    textBlock.color = "white";
    textBlock.fontSize = 24;
    textBlock.top = "-100px";

    // Add text to GUI
    advancedTexture.addControl(textBlock);

    // Create distance text block for dynamic UI
    const distanceText = new TextBlock("distance text");
    distanceText.text = "d: 0.00";
    distanceText.color = "white";
    distanceText.fontSize = 20;
    distanceText.top = "100px";

    // Add distance text to GUI
    advancedTexture.addControl(distanceText);

    // Enable WebXR immersive VR mode
    const xrExperience = await WebXRDefaultExperience.CreateAsync(scene, {
      uiOptions: {
        sessionMode: "immersive-vr"
      },
      teleportationOptions: {
        timeToTeleport: 2000,
        floorMeshes: [ground]
      }
    });

    // Store XR experience in scene metadata for inspection
    if (!scene.metadata) {
      scene.metadata = {};
    }
    scene.metadata.xr = xrExperience;

    // Dynamic UI: Update distance text on every frame
    scene.onBeforeRenderObservable.add(() => {
      const dragon = scene.getMeshByName("dragon");
      if (dragon && sphere) {
        const distance = Vector3.Distance(sphere.position, dragon.position);
        distanceText.text = `d: ${distance.toFixed(2)}`;
      }
    });

    // return scene
    return scene;
  }
}
