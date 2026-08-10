import { describe, expect, it } from 'vitest';

import { Explanation } from '../../src/core/explanation';

describe('Explanation', () => {
  it('validates, copies, and exposes its constructor values', () => {
    const data = { feature_attributions: { a: 0.5, b: 0.3 } };
    const featureNames = ['a', 'b'];
    const metadata = { audit: { status: 'experimental' } };
    const explanation = new Explanation(
      'TestExplainer',
      'cat',
      data,
      featureNames,
      metadata,
    );

    data.feature_attributions.a = 99;
    featureNames[0] = 'mutated';
    metadata.audit.status = 'mutated';

    expect(explanation.explainerName).toBe('TestExplainer');
    expect(explanation.targetClass).toBe('cat');
    expect(explanation.getAttributions()).toEqual({ a: 0.5, b: 0.3 });
    expect(explanation.featureNames).toEqual(['a', 'b']);
    expect(explanation.metadata).toEqual({ audit: { status: 'experimental' } });
  });

  it('returns top features by magnitude or signed value without mutating payload order', () => {
    const explanation = new Explanation('Test', 'cat', {
      feature_attributions: { a: 0.1, b: 0.8, c: -0.5 },
    });

    expect(explanation.getTopFeatures(2)).toEqual([
      ['b', 0.8],
      ['c', -0.5],
    ]);
    expect(explanation.getTopFeatures(2, false)).toEqual([
      ['b', 0.8],
      ['a', 0.1],
    ]);
    expect(Object.keys(explanation.getAttributions() ?? {})).toEqual(['a', 'b', 'c']);
  });

  it('returns empty attribution results when the field is absent', () => {
    const explanation = new Explanation('Test', 'cat', {});
    expect(explanation.getAttributions()).toBeUndefined();
    expect(explanation.getTopFeatures()).toEqual([]);
  });

  it.each([
    () => new Explanation('', 'cat', {}),
    () => new Explanation('Test', '   ', {}),
    () => new Explanation('Test', 'cat', [] as unknown as Record<string, unknown>),
    () =>
      new Explanation(
        'Test',
        'cat',
        {},
        ['valid', 3] as unknown as readonly string[],
      ),
    () =>
      new Explanation(
        'Test',
        'cat',
        {},
        undefined,
        null as unknown as Record<string, unknown>,
      ),
  ])('rejects malformed constructor input', (construct) => {
    expect(construct).toThrow();
  });

  it.each([
    [{ feature_attributions: [] }, TypeError],
    [{ feature_attributions: { a: true } }, TypeError],
    [{ feature_attributions: { a: Number.NaN } }, RangeError],
    [{ feature_attributions: { a: Number.POSITIVE_INFINITY } }, RangeError],
    [{ feature_attributions: { '': 1 } }, RangeError],
  ])('rejects malformed attribution payload %#', (data, errorType) => {
    const explanation = new Explanation('Test', 'cat', data);
    expect(() => explanation.getAttributions()).toThrow(errorType);
  });

  it.each([0, -1, 1.5, Number.NaN])('rejects invalid top-k value %s', (k) => {
    const explanation = new Explanation('Test', 'cat', {
      feature_attributions: { a: 1 },
    });
    expect(() => explanation.getTopFeatures(k)).toThrow(RangeError);
  });

  it('rejects a non-boolean ranking mode at runtime', () => {
    const explanation = new Explanation('Test', 'cat', {
      feature_attributions: { a: 1 },
    });
    expect(() => explanation.getTopFeatures(1, 'yes' as unknown as boolean)).toThrow(
      TypeError,
    );
  });

  it('resolves feature indices without inventing a match', () => {
    const explanation = new Explanation('Test', 'cat', {}, ['a', 'b']);
    expect(explanation.getFeatureIndex('b')).toBe(1);
    expect(explanation.getFeatureIndex('missing')).toBeUndefined();
    expect(new Explanation('Test', 'cat', {}).getFeatureIndex('a')).toBeUndefined();
  });

  it('round-trips a detached snake-case wire payload', () => {
    const original = new Explanation(
      'Test',
      'cat',
      { feature_attributions: { a: 1 } },
      ['a'],
      { source: 'test' },
    );
    const payload = original.toObject();
    const restored = Explanation.fromObject(payload);

    payload.explanation_data.feature_attributions = { changed: 2 };
    payload.feature_names?.push('changed');

    expect(restored.toObject()).toEqual({
      explainer_name: 'Test',
      target_class: 'cat',
      explanation_data: { feature_attributions: { a: 1 } },
      feature_names: ['a'],
      metadata: { source: 'test' },
    });
  });

  it.each([
    null,
    {},
    { explainer_name: 'Test', target_class: 'cat' },
    {
      explainer_name: 'Test',
      target_class: 'cat',
      explanation_data: {},
      feature_names: 'not-an-array',
    },
  ])('rejects malformed wire payload %#', (payload) => {
    expect(() => Explanation.fromObject(payload)).toThrow();
  });

  it('rejects values that cannot be structurally cloned', () => {
    expect(
      () => new Explanation('Test', 'cat', { callback: () => 'not serializable' }),
    ).toThrow(TypeError);
  });
});
