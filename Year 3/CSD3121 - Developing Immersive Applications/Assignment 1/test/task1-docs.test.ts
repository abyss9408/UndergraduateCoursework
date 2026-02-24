/**
 * Task 1: Writing documentation with Markdown
 *
 * Purpose:
 * - Verify that basic project documentation is present
 * - Ensure students demonstrate familiarity with Markdown syntax
 *
 * Test environment:
 * - This test runs in a Node.js environment
 * - No Babylon.js, rendering, or browser APIs are involved
 *
 * Verifies that:
 * 1. about-me.md exists in the repository root
 * 2. The file contains at least two Markdown headings
 * 3. The file contains at least one Markdown image
 */

import { describe, test, expect } from "vitest";
import { existsSync, readFileSync } from "fs";
import { join } from "path";

describe("t01: writing documentation", () => {
  const aboutMePath = join(process.cwd(), "about-me.md");

  test("about-me.md exists in the repository root", () => {
    expect(existsSync(aboutMePath)).toBe(true);
  });

  test("about-me.md contains at least two Markdown headings", () => {
    const content = readFileSync(aboutMePath, "utf-8");

    // Match Markdown headings: # Heading, ## Heading, etc.
    const headings = content.match(/^#{1,6}\s+.+/gm) ?? [];
    expect(headings.length).toBeGreaterThanOrEqual(2);
  });

  test("about-me.md contains at least one Markdown image", () => {
    const content = readFileSync(aboutMePath, "utf-8");

    // Match Markdown image syntax, including ![](url) and ![alt](url)
    const images = content.match(/!\[[^\]]*\]\([^)]+\)/g) ?? [];
    expect(images.length).toBeGreaterThanOrEqual(1);
  });
});
