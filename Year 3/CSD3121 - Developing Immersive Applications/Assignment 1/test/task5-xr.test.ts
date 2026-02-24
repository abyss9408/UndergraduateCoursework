/**
 * Task 5: Enabling WebXR immersive VR mode
 *
 * Purpose:
 * - Verify that the application initializes Babylon.js WebXR support
 *   using the canonical `createDefaultXRExperienceAsync()` API.
 *
 * Test environment:
 * - Uses Babylon.js NullEngine for headless testing
 * - WebXR is *initialized structurally* but not actually entered
 * - No XR-capable browser or headset is required
 * - WebXR availability errors are expected and ignored in this environment
 * - VideoDome and asset loading are mocked globally for scene integrity
 *
 * Verifies that:
 * 1. `createDefaultXRExperienceAsync()` (or the equivalent helper) is called
 *    during scene creation to set up WebXR immersive VR support
 * 2. The helper is configured for **immersive VR** mode
 * 3. Teleportation options are supplied as below
 */

import { describe, test, beforeEach, vi, expect } from "vitest";
import { NullEngine, WebXRDefaultExperience } from "@babylonjs/core";
import { App } from "../src/app";

describe("t05: WebXR immersive VR support", () => {
  let xrSpy: any;
  const fakeXR = { baseExperience: {}, teleportation: {} } as any;

  beforeEach(() => {
    vi.clearAllMocks();
    // Mock the static helper to resolve immediately and return a fake experience
    xrSpy = vi
      .spyOn(WebXRDefaultExperience, "CreateAsync")
      .mockResolvedValue(fakeXR as any);
  });

  test("creates a default XR experience with the expected options and attaches it to the scene", async () => {
    const engine = new NullEngine();
    const app = new App(engine);

    const scene = await app.createScene();

    // Ensure the helper was called
    expect(xrSpy).toHaveBeenCalled();

    // Inspect the last call to check the options passed
    const lastCall = xrSpy.mock.calls[xrSpy.mock.calls.length - 1];
    const [calledScene, options] = lastCall;

    expect(calledScene).toBe(scene);

    // Check that UI options include immersive-vr session mode
    expect(options).toBeDefined();
    expect(options.uiOptions).toBeDefined();
    expect(options.uiOptions.sessionMode).toBe("immersive-vr");

    // Check teleportation options were provided and reference the ground mesh
    expect(options.teleportationOptions).toBeDefined();
    expect(options.teleportationOptions.timeToTeleport).toBe(2000);
    const floorMeshes = options.teleportationOptions.floorMeshes;
    expect(Array.isArray(floorMeshes)).toBe(true);
    expect(floorMeshes).toContain(scene.getMeshByName("ground"));

    // If the implementation exposes the XR experience via scene.metadata.xr,
    // verify it is defined. This is optional and will not fail the test if
    // the implementation chooses a different inspection mechanism.
    if (
      scene.metadata &&
      Object.prototype.hasOwnProperty.call(scene.metadata, "xr")
    ) {
      expect(scene.metadata.xr).toBeDefined();
    }
  });
});
