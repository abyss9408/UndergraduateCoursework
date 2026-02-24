// test/setup.ts
import { vi } from "vitest";

vi.mock("@babylonjs/core", async (importOriginal) => {
  const mod = (await importOriginal()) as typeof import("@babylonjs/core");

  return {
    ...mod,

    // -----------------------------
    // VideoDome (universal mock)
    // -----------------------------
    VideoDome: vi.fn(function (name, mediaUrl, options, scene) {
      const node = new mod.TransformNode(name, scene);
      (node as any).videoTexture = {
        video: { currentSrc: mediaUrl, play: vi.fn(), pause: vi.fn() },
      };
      Object.assign(node, options);
      return node as any;
    }),

    // -----------------------------
    // ImportMeshAsync (safe default)
    // -----------------------------
    // ImportMeshAsync: vi.fn(async (_url, scene) => {
    //   // minimal mesh so scene logic works
    //   const mesh = new mod.Mesh("mocked-asset", scene);
    //
    //   return {
    //     meshes: [mesh],
    //     particleSystems: [],
    //     skeletons: [],
    //     animationGroups: [],
    //   };
    // }),
  };
});
