/**
 * Task 4: Creating a basic immersive environment with a VideoDome
 *
 * Purpose:
 * - Verify that the application creates a VideoDome as an immersive
 *   background environment.
 *
 * Test environment:
 * - Uses Babylon.js NullEngine for headless testing
 * - VideoDome is mocked globally (see test/setup.ts) to avoid
 *   HTMLMediaElement playback and WebGL dependencies
 * - This test verifies structural intent, not actual video playback
 *
 * Verifies that:
 * 1. A node named "video dome" exists in the scene
 * 2. The node has a videoTexture with the expected media source
 */

import { describe, test, expect } from "vitest";
import { NullEngine, Scene, TransformNode } from "@babylonjs/core";
import { App } from "../src/app";

describe("t04: immersive environment with VideoDome", () => {
  test('creates a VideoDome named "video dome"', async () => {
    const engine = new NullEngine();
    const app = new App(engine);
    const scene: Scene = await app.createScene();

    const dome = scene.getNodeByName("video dome");

    expect(dome).toBeInstanceOf(TransformNode);

    // VideoDome is mocked, so we only check structural intent
    expect((dome as any).videoTexture).toBeDefined();
    expect((dome as any).videoTexture.video.currentSrc).toContain("video.mp4");
  });
});
