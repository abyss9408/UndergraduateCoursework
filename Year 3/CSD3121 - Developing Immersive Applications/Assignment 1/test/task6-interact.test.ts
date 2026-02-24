/**
 * Task 6: Creating basic click and drag interactions in WebXR
 *
 * Test environment:
 * - Uses Babylon.js NullEngine for headless testing
 * - Asset loading and VideoDome are mocked globally (see test/setup.ts)
 * - Interaction is tested structurally, not via pointer simulation
 *
 * Verifies that:
 * 1. A model asset (expecting the dragon.glb) is loaded into the scene
 * 2. The dragon model is positioned and scaled correctly
 * 3. A pick/select interaction is registered on a sphere and can be
 *    triggered to produce a state change (i.e., scaling)
 */

import { describe, test, expect, vi } from "vitest";
import {
  ActionEvent,
  NullEngine,
  Mesh,
  ActionManager,
  Vector3,
} from "@babylonjs/core";
import { App } from "../src/app";

describe("t06: basic interactions and asset loading", () => {
  test("loads a dragon mesh that is fairly complex", async () => {
    const engine = new NullEngine();
    const app = new App(engine);
    const scene = await app.createScene();

    const dragon = await vi.waitUntil(() => scene.getNodeByName("dragon"), {
      timeout: 15000,
      interval: 250,
    });

    // Get all descendant meshes under the "dragon" node
    const meshes = dragon.getChildMeshes ? dragon.getChildMeshes(false) : [];

    // Wait until vertex buffers are actually there (avoid scene.whenReadyAsync)
    await vi.waitUntil(
      () => {
        const total = meshes.reduce((sum, m) => sum + m.getTotalVertices(), 0);
        return total > 0;
      },
      { timeout: 15000, interval: 250 },
    );

    const totalVerts = meshes.reduce((sum, m) => sum + m.getTotalVertices(), 0);

    expect(meshes.length).toBeGreaterThan(0);

    // the dragon model has 494947 vertices, so this heuristic
    // should confirm something similar is loaded
    expect(totalVerts).toBeGreaterThan(493000);

    expect(dragon).not.toBeNull();
    expect(dragon).toBeInstanceOf(Mesh);
    expect(dragon.name).toBe("dragon");

    // Heuristic: external/custom model should have substantial vertex count
    // expect((dragon as Mesh).getTotalVertices()).toBeGreaterThan(1000);
  });

  test("dragon positioned at (0,0,10) and enlarged to (30,30,30)", async () => {
    const engine = new NullEngine();
    const app = new App(engine);
    const scene = await app.createScene();

    const dragon = await vi.waitUntil(
      () => scene.getMeshByName("dragon")! as Mesh,
      { timeout: 15000, interval: 1000 },
    );

    expect(dragon.position.toString()).toEqual(
      new Vector3(0, 0, 10).toString(),
    );
    expect(dragon.scaling.toString()).toEqual(
      new Vector3(30, 30, 30).toString(),
    );
  }, 20000);

  test("registers and triggers pick interaction on sphere", async () => {
    const engine = new NullEngine();
    const app = new App(engine);
    const scene = await app.createScene();

    const sphere = scene.getMeshByName("sphere") as Mesh;
    expect(sphere).toBeInstanceOf(Mesh);

    // Ensure interaction system is attached
    expect(sphere.actionManager).toBeInstanceOf(ActionManager);

    const actionManager = sphere.actionManager;
    if (!actionManager) {
      throw new Error("Sphere does not have an ActionManager attached.");
    }

    const initialScale = sphere.scaling.clone();

    // Find a pick/select interaction
    const pickAction = actionManager.actions?.find(
      (a) => a.trigger === ActionManager.OnPickTrigger,
    );

    if (!pickAction) {
      throw new Error(
        "No pick interaction found. Make sure you register an " +
          "ActionManager.OnPickTrigger handler on the sphere.",
      );
    }

    // Trigger interaction via ActionManager (type-safe, implementation-agnostic)
    actionManager.processTrigger(
      ActionManager.OnPickTrigger,
      ActionEvent.CreateNew(sphere),
    );

    // Verify observable state change
    expect(sphere.scaling.equals(initialScale)).toBe(false);
  }, 20000);
});
