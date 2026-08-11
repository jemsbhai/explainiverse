import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const SCRIPT = resolve('scripts/check-bundle-budget.mjs');

function fixture(maximum: number) {
  const root = mkdtempSync(join(tmpdir(), 'explainiverse-bundle-'));
  const output = join(root, 'demo');
  const policy = join(root, 'policy.json');
  return { root, output, policy, maximum };
}

describe('demo bundle budget', () => {
  it('accepts a production output within every byte ceiling', () => {
    const { output, policy } = fixture(1024);
    execFileSync(process.execPath, [
      '-e',
      `require('node:fs').mkdirSync(${JSON.stringify(output)}, { recursive: true })`,
    ]);
    writeFileSync(join(output, 'app.js'), 'console.log("synthetic");\n');
    writeFileSync(
      policy,
      JSON.stringify({
        schemaVersion: 1,
        maxTotalOutputBytes: 1024,
        maxSingleJavaScriptBytes: 1024,
        maxTotalJavaScriptGzipBytes: 1024,
      }),
    );

    const result = execFileSync(process.execPath, [SCRIPT, policy, output], { encoding: 'utf8' });
    expect(JSON.parse(result)).toMatchObject({ javascriptAssetCount: 1, schemaVersion: 1 });
  });

  it('fails closed when JavaScript exceeds the reviewed ceiling', () => {
    const { output, policy } = fixture(8);
    execFileSync(process.execPath, [
      '-e',
      `require('node:fs').mkdirSync(${JSON.stringify(output)}, { recursive: true })`,
    ]);
    writeFileSync(join(output, 'app.js'), '0123456789');
    writeFileSync(
      policy,
      JSON.stringify({
        schemaVersion: 1,
        maxTotalOutputBytes: 8,
        maxSingleJavaScriptBytes: 8,
        maxTotalJavaScriptGzipBytes: 8,
      }),
    );

    const result = spawnSync(process.execPath, [SCRIPT, policy, output], { encoding: 'utf8' });
    expect(result.status).toBe(2);
    expect(result.stderr).toMatch(/exceeds|oversized/);
  });
});
