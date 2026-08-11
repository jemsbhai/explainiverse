import { gzipSync } from 'node:zlib';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

function requirePositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive safe integer`);
  }
  return value;
}

function filesUnder(root) {
  const files = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = resolve(root, entry.name);
    if (entry.isSymbolicLink()) {
      throw new Error(`bundle output must not contain symbolic links: ${path}`);
    }
    if (entry.isDirectory()) {
      files.push(...filesUnder(path));
    } else if (entry.isFile()) {
      files.push(path);
    }
  }
  return files.sort();
}

export function checkBundleBudget(policyPath, outputDirectory) {
  const policy = JSON.parse(readFileSync(policyPath, 'utf8'));
  if (policy.schemaVersion !== 1) {
    throw new Error('bundle budget must use schema version 1');
  }
  const maxTotal = requirePositiveInteger(policy.maxTotalOutputBytes, 'maxTotalOutputBytes');
  const maxSingleJs = requirePositiveInteger(
    policy.maxSingleJavaScriptBytes,
    'maxSingleJavaScriptBytes',
  );
  const maxGzipJs = requirePositiveInteger(
    policy.maxTotalJavaScriptGzipBytes,
    'maxTotalJavaScriptGzipBytes',
  );

  const files = filesUnder(outputDirectory);
  const javascript = files.filter((path) => path.endsWith('.js'));
  if (javascript.length === 0) {
    throw new Error('bundle output contains no JavaScript asset');
  }
  const totalBytes = files.reduce((total, path) => total + statSync(path).size, 0);
  const oversized = javascript.filter((path) => statSync(path).size > maxSingleJs);
  const totalJavaScriptGzipBytes = javascript.reduce(
    (total, path) => total + gzipSync(readFileSync(path), { level: 9 }).byteLength,
    0,
  );
  const violations = [];
  if (totalBytes > maxTotal) {
    violations.push(`total output ${totalBytes} exceeds ${maxTotal} bytes`);
  }
  if (oversized.length > 0) {
    violations.push(`oversized JavaScript assets: ${oversized.join(', ')}`);
  }
  if (totalJavaScriptGzipBytes > maxGzipJs) {
    violations.push(
      `gzipped JavaScript ${totalJavaScriptGzipBytes} exceeds ${maxGzipJs} bytes`,
    );
  }
  if (violations.length > 0) {
    throw new Error(violations.join('; '));
  }
  return {
    schemaVersion: 1,
    fileCount: files.length,
    javascriptAssetCount: javascript.length,
    totalBytes,
    totalJavaScriptGzipBytes,
  };
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const policyPath = resolve(process.argv[2] ?? 'bundle-budget.json');
  const outputDirectory = resolve(process.argv[3] ?? 'dist/demo');
  try {
    console.log(JSON.stringify(checkBundleBudget(policyPath, outputDirectory), null, 2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 2;
  }
}
