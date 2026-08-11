import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const EXPECTED_EXPORTS = {
  ".": {
    types: "./dist/index.d.ts",
    require: "./dist/index.js",
    default: "./dist/index.js",
  },
  "./core": {
    types: "./dist/core/index.d.ts",
    require: "./dist/core/index.js",
    default: "./dist/core/index.js",
  },
  "./visualizer": {
    types: "./dist/visualizer/index.d.ts",
    require: "./dist/visualizer/index.js",
    default: "./dist/visualizer/index.js",
  },
};
const EXPECTED_FILES = [
  "dist/index.js",
  "dist/index.d.ts",
  "dist/core",
  "dist/visualizer",
  "README.md",
];
const FORBIDDEN_LIFECYCLE_SCRIPTS = [
  "prepublish",
  "prepublishOnly",
  "publish",
  "postpublish",
  "prepare",
  "preinstall",
  "install",
  "postinstall",
  "postpack",
];
const FORBIDDEN_PUBLICATION_FIELDS = ["module", "browser", "unpkg", "jsdelivr"];

function canonical(value) {
  return JSON.stringify(value);
}

export function validatePackageManifest(packageJson) {
  const capability = {
    claimStatus: "experimental",
    wireSchema: "explainiverse.explanation.v1",
    algorithmParityWithPython: false,
    publicationReady: false,
  };
  const required = {
    name: "explainiverse-js-experimental",
    version: "0.0.0-experimental",
    private: true,
    description:
      "Private experimental TypeScript core contracts and attribution visualizer for Explainiverse",
    type: "commonjs",
    main: "./dist/index.js",
    types: "./dist/index.d.ts",
    license: "UNLICENSED",
  };
  for (const [field, expected] of Object.entries(required)) {
    if (packageJson[field] !== expected) {
      throw new Error(`package ${field} must remain ${canonical(expected)}`);
    }
  }
  if (
    canonical(packageJson.explainiverseCapability) !== canonical(capability)
  ) {
    throw new Error("experimental capability boundary changed");
  }
  if (canonical(packageJson.exports) !== canonical(EXPECTED_EXPORTS)) {
    throw new Error(
      "package exports changed outside the reviewed CommonJS surface",
    );
  }
  if (canonical(packageJson.files) !== canonical(EXPECTED_FILES)) {
    throw new Error("package files allowlist changed outside review");
  }
  if (Object.hasOwn(packageJson, "publishConfig")) {
    throw new Error(
      "private experimental package must not define publishConfig",
    );
  }
  for (const field of FORBIDDEN_PUBLICATION_FIELDS) {
    if (Object.hasOwn(packageJson, field)) {
      throw new Error(
        `private experimental package must not define publication field: ${field}`,
      );
    }
  }
  if (packageJson.scripts?.prepack !== "npm run build") {
    throw new Error("prepack must remain the reviewed build-only hook");
  }
  for (const script of FORBIDDEN_LIFECYCLE_SCRIPTS) {
    if (Object.hasOwn(packageJson.scripts ?? {}, script)) {
      throw new Error(
        `forbidden publication/install lifecycle script: ${script}`,
      );
    }
  }
}

export function validatePackedFiles(actualFiles, expectedFiles) {
  const actual = [...actualFiles].sort();
  const expected = [...expectedFiles].sort();
  if (canonical(actual) !== canonical(expected)) {
    const missing = expected.filter((entry) => !actual.includes(entry));
    const extra = actual.filter((entry) => !expected.includes(entry));
    throw new Error(
      `npm tarball differs from reviewed allowlist: missing=${canonical(missing)} extra=${canonical(extra)}`,
    );
  }
}

function main() {
  const packageJson = JSON.parse(
    readFileSync(resolve(ROOT, "package.json"), "utf8"),
  );
  const allowlist = JSON.parse(
    readFileSync(resolve(ROOT, "npm-pack-allowlist.json"), "utf8"),
  );
  validatePackageManifest(packageJson);
  const npm = process.platform === "win32" ? "npm.cmd" : "npm";
  const result = spawnSync(
    npm,
    ["pack", "--dry-run", "--json", "--ignore-scripts"],
    { cwd: ROOT, encoding: "utf8" },
  );
  if (result.status !== 0) {
    throw new Error(`npm pack failed: ${result.stderr || result.stdout}`);
  }
  const records = JSON.parse(result.stdout);
  if (
    !Array.isArray(records) ||
    records.length !== 1 ||
    !Array.isArray(records[0].files)
  ) {
    throw new Error("npm pack returned an unexpected inventory shape");
  }
  const packedFiles = records[0].files.map((entry) => entry.path);
  validatePackedFiles(packedFiles, allowlist);
  if (records[0].entryCount !== allowlist.length) {
    throw new Error(
      "npm pack entryCount does not match the reviewed allowlist",
    );
  }
  process.stdout.write(
    `verified private npm boundary and ${allowlist.length} packed files\n`,
  );
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  main();
}
