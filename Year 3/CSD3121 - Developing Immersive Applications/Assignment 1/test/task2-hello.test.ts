/**
 * Task 2: Hello World
 *
 * Purpose:
 * - Verify that the application can be initialized with a Babylon.js engine
 * - Verify that a Scene is successfully created
 * - Verify that a basic startup message is printed to the console
 *
 * Test environment:
 * - Runs in jsdom using Babylon.js NullEngine
 * - Does NOT require WebGL, rendering, or a real canvas
 * - Global mocks are applied via test/setup.ts
 *
 * Verifies that:
 * 1. An App instance can be created with a Babylon.js engine
 * 2. createScene() returns a Scene
 * 3. "Hello, immersive world!" is printed to the console
 */

import { describe, test, expect, vi } from "vitest";
import { NullEngine, Scene } from "@babylonjs/core";
import { App } from "../src/app";

describe("t02: hello world", () => {
  test('creates a scene and prints "Hello, immersive world!"', async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    try {
      const engine = new NullEngine();
      const app = new App(engine);

      const scene = await app.createScene();

      // Scene check
      expect(scene).toBeInstanceOf(Scene);

      // Console output check
      const printedText = logSpy.mock.calls
        .map((args) => args.join(" "))
        .join(" ")
        .toLowerCase();

      expect(printedText).toContain("hello, immersive world!");
    } finally {
      logSpy.mockRestore();
    }
  });
});
