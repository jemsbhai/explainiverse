import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import * as publicApi from '../src';

describe('public TypeScript surface', () => {
  it('exports only audited core and visualizer values', () => {
    expect(publicApi.Explanation).toBeTypeOf('function');
    expect(publicApi.BaseExplainer).toBeTypeOf('function');
    expect(publicApi.ExplainerRegistry).toBeTypeOf('function');
    expect(publicApi.ExplanationVisualizer).toBeTypeOf('function');
    expect('explainers' in publicApi).toBe(false);
    expect('metrics' in publicApi).toBe(false);
  });

  it('remains private and records the unimplemented publication boundaries', () => {
    const packageJson = JSON.parse(
      readFileSync(resolve(process.cwd(), 'package.json'), 'utf8'),
    ) as {
      private: boolean;
      type: string;
      explainiverseCapability: {
        claimStatus: string;
        wireSchema: string;
        algorithmParityWithPython: boolean;
        publicationReady: boolean;
      };
    };

    expect(packageJson.private).toBe(true);
    expect(packageJson.type).toBe('commonjs');
    expect(packageJson.explainiverseCapability).toEqual({
      claimStatus: 'experimental',
      wireSchema: 'explainiverse.explanation.v1',
      algorithmParityWithPython: false,
      publicationReady: false,
    });
  });
});
