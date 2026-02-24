/**
 * Task 3: Setting up a basic Babylon.js scene
 *
 * Purpose:
 * - Verify that the application creates a valid Babylon.js Scene
 * - Verify the presence of basic scene primitives and UI elements
 *
 * Test environment:
 * - Runs in jsdom using Babylon.js NullEngine
 * - Does NOT require WebGL, XR, video playback, or canvas rendering
 * - Global mocks are applied via test/setup.ts
 *
 * Verifies that:
 * 1. createScene() returns a Scene
 * 2. A sphere mesh named "sphere" exists
 * 3. A plane mesh named "hello plane" exists
 * 4. A GUI text "Hello, XR!" is present
 */

import { describe, test, expect } from "vitest";
import { NullEngine, Scene, Mesh } from "@babylonjs/core";
import { AdvancedDynamicTexture, TextBlock } from "@babylonjs/gui";
import { App } from "../src/app";

describe("t03: scene primitives and text", () => {
  test("scene contains basic meshes", async () => {
    const engine = new NullEngine();
    const app = new App(engine);

    const scene = await app.createScene();

    // Scene instance
    expect(scene).toBeInstanceOf(Scene);

    // Mesh checks
    const sphere = scene.getMeshByName("sphere") as Mesh;
    expect(sphere).toBeInstanceOf(Mesh);

    const plane = scene.getMeshByName("hello plane") as Mesh;
    expect(plane).toBeInstanceOf(Mesh);
  });

  test("scene contains advanced dynamic texture text", async () => {
    const engine = new NullEngine();
    const app = new App(engine);

    const scene = await app.createScene();

    // GUI texture check
    const texture = scene.getTextureByName("hello texture") as AdvancedDynamicTexture;
    expect(texture).toBeInstanceOf(AdvancedDynamicTexture);

    // TextBlock check
    const text = texture.getControlByName("hello text") as TextBlock;
    expect(text).toBeInstanceOf(TextBlock);
    expect(text.text).toBe("Hello, XR!");
  });
});
