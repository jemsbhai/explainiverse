import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import {
  validatePackageManifest,
  validatePackedFiles,
} from "../scripts/check-package-boundary.mjs";

const packageJson = JSON.parse(
  readFileSync(resolve(process.cwd(), "package.json"), "utf8"),
) as Record<string, unknown>;
const allowlist = JSON.parse(
  readFileSync(resolve(process.cwd(), "npm-pack-allowlist.json"), "utf8"),
) as string[];

function changed(field: string, value: unknown): Record<string, unknown> {
  return { ...packageJson, [field]: value };
}

describe("private npm publication boundary", () => {
  it("accepts only the reviewed experimental CommonJS manifest", () => {
    expect(() => validatePackageManifest(packageJson)).not.toThrow();
  });

  it.each([
    ["private", false],
    ["description", "Publishable Explainiverse JavaScript package"],
    ["type", "module"],
    ["publishConfig", { access: "public" }],
    ["exports", { ".": "./dist/index.js" }],
    ["files", ["dist"]],
    ["explainiverseCapability", { publicationReady: true }],
  ])("rejects a changed %s field", (field, value) => {
    expect(() => validatePackageManifest(changed(field, value))).toThrow();
  });

  it.each([
    ["module", "./dist/index.mjs"],
    ["browser", "./dist/index.browser.js"],
    ["unpkg", "./dist/index.js"],
    ["jsdelivr", "./dist/index.js"],
  ])("rejects the publication-oriented %s field", (field, value) => {
    expect(() => validatePackageManifest(changed(field, value))).toThrow(
      new RegExp(`publication field: ${field}$`),
    );
  });

  it("rejects publication and installation lifecycle hooks", () => {
    const scripts = {
      ...(packageJson.scripts as Record<string, string>),
      publish: "npm test",
    };
    expect(() => validatePackageManifest(changed("scripts", scripts))).toThrow(
      /lifecycle script/,
    );
  });

  it("rejects missing or extra tarball entries", () => {
    expect(() => validatePackedFiles(allowlist, allowlist)).not.toThrow();
    expect(() => validatePackedFiles(allowlist.slice(1), allowlist)).toThrow(
      /missing=/,
    );
    expect(() =>
      validatePackedFiles([...allowlist, "secret.env"], allowlist),
    ).toThrow(/extra=/);
  });
});
