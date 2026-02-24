/**
 * Task 7: Dynamic UI updates based on user-driven state changes
 *
 * Purpose:
 * - Ensure that a dynamic UI element updates correctly in response
 *   to changes in application or scene state
 *
 * Test environment:
 * - Runs in jsdom using Babylon.js NullEngine
 * - Does NOT rely on actual canvas rendering or WebXR support
 * - Scene rendering is not performed; instead,
 *   `onBeforeRenderObservable` is triggered manually
 * - Focuses on validating application logic and intent rather than
 *   visual output or immersive behavior
 *
 * Checks that:
 * 1. A plane mesh named "hello plane" exists
 * 2. The plane has an AdvancedDynamicTexture named "hello texture"
 * 3. The texture contains a TextBlock named "hello text"
 * 4. A sphere mesh named "sphere" exists
 * 5. A mesh named "dragon" exists
 * 6. When the sphere moves, the text updates to display the distance
 *    between the sphere and the dragon, formatted as "d: X.XX"
 */

import { describe, test, expect, vi } from "vitest";
import { Mesh, NullEngine, Vector3 } from "@babylonjs/core";
import { AdvancedDynamicTexture, TextBlock } from "@babylonjs/gui";
import { App } from "../src/app";

describe("t07: dynamic UI driven by interaction logic", () => {
  test("updates UI text based on sphere–dragon distance", async () => {
    const engine = new NullEngine();
    const app = new App(engine);
    const scene = await app.createScene();

    const plane = scene.getMeshByName("hello plane") as Mesh;
    expect(plane).toBeInstanceOf(Mesh);

    const texture = scene.getTextureByName(
      "hello texture",
    ) as AdvancedDynamicTexture;
    expect(texture).not.toBeNull();

    const text = texture.getControlByName("distance text") as TextBlock;
    expect(text).not.toBeNull();

    const sphere = scene.getMeshByName("sphere") as Mesh;
    expect(sphere).toBeInstanceOf(Mesh);

    // Wait for the "dragon" mesh to be available
    const dragon = await vi.waitUntil(
      () => scene.getMeshByName("dragon") as Mesh,
      { timeout: 15000, interval: 250 },
    );
    expect(dragon).toBeInstanceOf(Mesh);

    // --- First update ---
    sphere.position.set(1, 2, 3);
    scene.onBeforeRenderObservable.notifyObservers(scene);

    let expected = Vector3.Distance(sphere.position, dragon.position).toFixed(
      2,
    );
    expect(text.text).toBe(`d: ${expected}`);

    // --- Second update ---
    sphere.position.set(4, 1, 8);
    scene.onBeforeRenderObservable.notifyObservers(scene);

    expected = Vector3.Distance(sphere.position, dragon.position).toFixed(2);
    expect(text.text).toBe(`d: ${expected}`);
  }, 20000);
});
