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
});
