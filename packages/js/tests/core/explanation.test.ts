import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  Explanation,
  type ExplanationRecord,
  type ExplanationValue,
} from '../../src/core/explanation';

describe('Explanation', () => {
  it('emits the shared Python/JavaScript wire fixture exactly', () => {
    const fixturePath = resolve(process.cwd(), 'tests/fixtures/explanation-wire.json');
    const fixture = JSON.parse(readFileSync(fixturePath, 'utf8')) as unknown;
    expect(Explanation.fromObject(fixture).toObject()).toEqual(fixture);
  });

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
    () => new Explanation('Test', 'cat', [] as unknown as ExplanationRecord),
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
        null as unknown as ExplanationRecord,
      ),
  ])('rejects malformed constructor input', (construct) => {
    expect(construct).toThrow();
  });

  it('rejects duplicate feature names to match the Python container contract', () => {
    expect(() => new Explanation('Test', 'cat', {}, ['a', 'a'])).toThrow(/unique/);
  });

  it.each([
    [{ payload: new Map([['a', 1]]) }, TypeError],
    [{ payload: new Set([1, 2]) }, TypeError],
    [{ payload: new Date('2026-01-01T00:00:00Z') }, TypeError],
    [{ payload: BigInt(1) }, TypeError],
    [{ payload: undefined }, TypeError],
    [{ payload: Number.NaN }, RangeError],
    [{ payload: Number.POSITIVE_INFINITY }, RangeError],
    [{ payload: -0 }, RangeError],
  ])('rejects non-wire explanation value %#', (data, errorType) => {
    expect(() =>
      new Explanation(
        'Test',
        'cat',
        data as unknown as Record<string, never>,
      ),
    ).toThrow(errorType);
  });

  it('rejects cyclic wire values instead of deferring failure to JSON transport', () => {
    const cyclic: Record<string, unknown> = {};
    cyclic.self = cyclic;
    expect(() =>
      new Explanation(
        'Test',
        'cat',
        cyclic as unknown as Record<string, never>,
      ),
    ).toThrow(/cyclic/);
  });

  it('preserves a nested __proto__ JSON key as inert own data', () => {
    const data = JSON.parse('{"nested":{"__proto__":{"polluted":true}}}') as unknown;
    const transported = new Explanation(
      'Test',
      'cat',
      data as ExplanationRecord,
    ).toObject();
    const nested = transported.explanation_data.nested as ExplanationRecord;

    expect(Object.hasOwn(nested, '__proto__')).toBe(true);
    expect(JSON.parse(JSON.stringify(transported)).explanation_data).toEqual(data);
    expect(({} as { polluted?: boolean }).polluted).toBeUndefined();
  });

  it('rejects sparse, decorated, subclassed, and accessor-backed arrays', () => {
    const sparse = new Array<unknown>(1);
    const decorated = [1] as unknown[] & { extra?: number };
    decorated.extra = 2;
    const symbolDecorated = [1];
    Object.defineProperty(symbolDecorated, Symbol('extra'), { value: 2 });
    class WireArray extends Array<number> {}
    const accessor = [1];
    let getterCalls = 0;
    Object.defineProperty(accessor, '0', {
      enumerable: true,
      configurable: true,
      get: () => {
        getterCalls += 1;
        return 1;
      },
    });

    for (const candidate of [
      sparse,
      decorated,
      symbolDecorated,
      new WireArray(1),
      accessor,
    ]) {
      expect(
        () =>
          new Explanation('Test', 'cat', {
            candidate: candidate as unknown as ExplanationValue,
          }),
      ).toThrow(TypeError);
    }
    expect(getterCalls).toBe(0);
  });

  it('rejects enumerable accessors without invoking them', () => {
    let getterCalls = 0;
    const accessor: Record<string, unknown> = {};
    Object.defineProperty(accessor, 'value', {
      enumerable: true,
      get: () => {
        getterCalls += 1;
        return 1;
      },
    });

    expect(
      () =>
        new Explanation('Test', 'cat', {
          accessor: accessor as unknown as ExplanationValue,
        }),
    ).toThrow(/accessor/);
    expect(getterCalls).toBe(0);
  });

  it.each([
    [{ feature_attributions: [] }, TypeError],
    [{ feature_attributions: { a: true } }, TypeError],
    [{ feature_attributions: { a: Number.NaN } }, RangeError],
    [{ feature_attributions: { a: Number.POSITIVE_INFINITY } }, RangeError],
    [{ feature_attributions: { '': 1 } }, RangeError],
  ])('rejects malformed attribution payload %#', (data, errorType) => {
    expect(() => new Explanation('Test', 'cat', data).getAttributions()).toThrow(
      errorType,
    );
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

  it('survives an exact JSON transport round trip', () => {
    const original = new Explanation(
      'Test',
      'cat',
      {
        feature_attributions: { a: 1, b: -0.25 },
        diagnostics: [true, null, 'finite', 3.5],
      },
      ['a', 'b'],
      { nested: { source: 'javascript' } },
    );

    const transported = JSON.parse(JSON.stringify(original.toObject())) as unknown;
    expect(Explanation.fromObject(transported).toObject()).toEqual(original.toObject());
  });

  it('rejects lossy JSON integers outside the JavaScript safe-integer range', () => {
    const lossyPayload = JSON.parse(
      '{"explainer_name":"Test","target_class":"cat","explanation_data":{"count":9007199254740993},"feature_names":null,"metadata":{}}',
    ) as unknown;
    expect(() => Explanation.fromObject(lossyPayload)).toThrow(/safe-integer range/);

    for (const count of [Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER]) {
      const explanation = new Explanation('Test', 'cat', { count });
      expect(explanation.toObject().explanation_data.count).toBe(count);
    }
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

  it('requires exactly the five declared top-level wire fields', () => {
    const complete = {
      explainer_name: 'Test',
      target_class: 'cat',
      explanation_data: {},
      feature_names: null,
      metadata: {},
    };
    for (const missing of Object.keys(complete)) {
      const candidate = { ...complete } as Record<string, unknown>;
      delete candidate[missing];
      expect(() => Explanation.fromObject(candidate)).toThrow(/missing required field/);
    }

    expect(() => Explanation.fromObject({ ...complete, unexpected: true })).toThrow(
      /unknown field/,
    );
    const protoKey = JSON.parse(
      '{"explainer_name":"Test","target_class":"cat","explanation_data":{},"feature_names":null,"metadata":{},"__proto__":{"polluted":true}}',
    ) as unknown;
    expect(() => Explanation.fromObject(protoKey)).toThrow(/unknown field/);
  });

  it('rejects a class instance masquerading as a top-level wire record', () => {
    class Payload {
      public readonly explainer_name = 'Test';
      public readonly target_class = 'cat';
      public readonly explanation_data = {};
    }

    expect(() => Explanation.fromObject(new Payload())).toThrow(TypeError);
  });

  it('rejects top-level wire accessors without invoking them', () => {
    let getterCalls = 0;
    const payload: Record<string, unknown> = {
      target_class: 'cat',
      explanation_data: {},
    };
    Object.defineProperty(payload, 'explainer_name', {
      enumerable: true,
      get: () => {
        getterCalls += 1;
        return 'Test';
      },
    });

    expect(() => Explanation.fromObject(payload)).toThrow(/accessor/);
    expect(getterCalls).toBe(0);
  });

  it('rejects decorated, subclassed, and accessor-backed feature-name arrays', () => {
    const decorated = ['a'] as string[] & { extra?: string };
    decorated.extra = 'not wire data';
    class NameArray extends Array<string> {}
    const accessor = ['a'];
    let getterCalls = 0;
    Object.defineProperty(accessor, '0', {
      enumerable: true,
      configurable: true,
      get: () => {
        getterCalls += 1;
        return 'a';
      },
    });

    for (const names of [decorated, new NameArray('a'), accessor]) {
      expect(
        () =>
          new Explanation(
            'Test',
            'cat',
            {},
            names as unknown as readonly string[],
          ),
      ).toThrow(TypeError);
    }
    expect(getterCalls).toBe(0);
  });

  it('rejects callable values that cannot cross the JSON wire contract', () => {
    expect(
      () =>
        new Explanation('Test', 'cat', {
          callback: (() => 'not serializable') as unknown as ExplanationValue,
        }),
    ).toThrow(TypeError);
  });
});
