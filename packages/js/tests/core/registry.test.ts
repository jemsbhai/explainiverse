import { beforeEach, describe, expect, it } from 'vitest';

import { BaseExplainer } from '../../src/core/explainer';
import { Explanation } from '../../src/core/explanation';
import {
  ExplainerRegistry,
  type ExplainerConstructor,
  type ExplainerMeta,
  type FilterCriteria,
  type ModelType,
} from '../../src/core/registry';

class MockExplainer extends BaseExplainer {
  public explain(_instance: unknown): Promise<Explanation> {
    return Promise.resolve(new Explanation('Mock', 'test', {}));
  }
}

class BatchMockExplainer extends MockExplainer {
  public explainBatch(instances: readonly unknown[]): Promise<Explanation[]> {
    return Promise.all(instances.map((instance) => this.explain(instance)));
  }
}

function localMeta(overrides: Partial<ExplainerMeta> = {}): ExplainerMeta {
  return {
    scope: 'local',
    model_types: ['any'],
    data_types: ['tabular'],
    ...overrides,
  };
}

describe('ExplainerRegistry', () => {
  let registry: ExplainerRegistry;

  beforeEach(() => {
    registry = new ExplainerRegistry();
  });

  it('registers immutable normalized metadata with conservative defaults', () => {
    const modelTypes: ModelType[] = ['any'];
    registry.register('mock', MockExplainer, localMeta({ model_types: modelTypes }));
    modelTypes[0] = 'tree';

    const entry = registry.get('mock');
    expect(entry.class).toBe(MockExplainer);
    expect(entry.meta.model_types).toEqual(['any']);
    expect(entry.meta.claim_status).toBe('unverified');
    expect(entry.meta.claim_scope).toMatch(/not completed an accuracy audit/i);
    expect(entry.meta.requires_training_data).toBe(false);
    expect(entry.meta.supports_batching).toBe(false);
    expect(() => (entry.meta.model_types as ModelType[]).push('tree')).toThrow(TypeError);
  });

  it('requires explicit scope for verified and quarantined claims', () => {
    expect(() =>
      registry.register(
        'verified',
        MockExplainer,
        localMeta({ claim_status: 'verified' }),
      ),
    ).toThrow(/claim_scope/);
    expect(() =>
      registry.register(
        'quarantined',
        MockExplainer,
        localMeta({ claim_status: 'quarantined' }),
      ),
    ).toThrow(/claim_scope/);

    registry.register(
      'scoped',
      MockExplainer,
      localMeta({ claim_status: 'verified', claim_scope: 'Test-only behavior.' }),
    );
    expect(registry.get('scoped').meta.claim_scope).toBe('Test-only behavior.');
  });

  it.each([
    ['', RangeError],
    ['   ', RangeError],
    [' padded', RangeError],
    [3, TypeError],
  ])('rejects malformed registry name %#', (name, errorType) => {
    expect(() =>
      registry.register(name as string, MockExplainer, localMeta()),
    ).toThrow(errorType);
  });

  it('handles prototype-like names without object-key collisions', () => {
    registry.register('__proto__', MockExplainer, localMeta());
    registry.register('constructor', MockExplainer, localMeta());
    expect(registry.listExplainers()).toEqual(['__proto__', 'constructor']);
  });

  it('rejects classes that do not implement explain', () => {
    class NotAnExplainer {}
    expect(() =>
      registry.register(
        'bad',
        NotAnExplainer as unknown as ExplainerConstructor,
        localMeta(),
      ),
    ).toThrow(TypeError);
  });

  it.each([
    { scope: 'unknown' },
    { model_types: [] },
    { model_types: ['tree', 'tree'] },
    { data_types: ['unknown'] },
    { task_types: ['unknown'] },
    { description: '   ' },
    { supports_batching: 'yes' },
    { claim_status: 'complete' },
  ])('rejects malformed metadata %#', (override) => {
    expect(() =>
      registry.register(
        'bad',
        MockExplainer,
        localMeta(override as unknown as Partial<ExplainerMeta>),
      ),
    ).toThrow();
  });

  it('rejects false batching claims and accepts a matching public method', () => {
    expect(() =>
      registry.register(
        'bad_batch',
        MockExplainer,
        localMeta({ supports_batching: true }),
      ),
    ).toThrow(/explainBatch/);

    registry.register(
      'batch',
      BatchMockExplainer,
      localMeta({ supports_batching: true }),
    );
    expect(registry.get('batch').meta.supports_batching).toBe(true);
  });

  it('rejects duplicate names unless override is explicitly true', () => {
    registry.register('mock', MockExplainer, localMeta({ description: 'first' }));
    expect(() => registry.register('mock', MockExplainer, localMeta())).toThrow(
      /already registered/,
    );
    expect(() =>
      registry.register(
        'mock',
        MockExplainer,
        localMeta(),
        'yes' as unknown as boolean,
      ),
    ).toThrow(TypeError);

    registry.register(
      'mock',
      MockExplainer,
      localMeta({ description: 'replacement' }),
      true,
    );
    expect(registry.get('mock').meta.description).toBe('replacement');
  });

  it('lists in registration order and creates with caller-supplied arguments', () => {
    registry.register('mock1', MockExplainer, localMeta());
    registry.register('mock2', MockExplainer, localMeta());

    expect(registry.listExplainers()).toEqual(['mock1', 'mock2']);
    const model = { predict: () => [] };
    const explainer = registry.create('mock1', model);
    expect(explainer).toBeInstanceOf(MockExplainer);
    expect(explainer.model).toBe(model);
  });

  it('throws a descriptive error for an unknown explainer', () => {
    expect(() => registry.get('unknown')).toThrow("Explainer 'unknown' is not registered");
  });

  it('filters exact metadata while treating model type any as a wildcard', () => {
    registry.register(
      'local_tree',
      MockExplainer,
      localMeta({
        model_types: ['tree'],
        task_types: ['classification'],
      }),
    );
    registry.register(
      'global_any',
      MockExplainer,
      {
        scope: 'global',
        model_types: ['any'],
        data_types: ['tabular'],
        task_types: ['regression'],
      },
    );
    registry.register(
      'image',
      MockExplainer,
      localMeta({ model_types: ['neural'], data_types: ['image'] }),
    );

    expect(registry.filter({ scope: 'local' })).toEqual(['local_tree', 'image']);
    expect(registry.filter({ data_type: 'image' })).toEqual(['image']);
    expect(registry.filter({ model_type: 'tree' })).toEqual([
      'local_tree',
      'global_any',
    ]);
    expect(registry.filter({ task_type: 'regression' })).toEqual(['global_any']);
    expect(registry.filter()).toEqual(['local_tree', 'global_any', 'image']);
  });

  it('rejects invalid or misspelled filter criteria instead of silently ignoring them', () => {
    expect(() => registry.filter(null as unknown as FilterCriteria)).toThrow(TypeError);
    expect(() =>
      registry.filter({ modelType: 'tree' } as unknown as FilterCriteria),
    ).toThrow(/unknown filter criteria/);
    expect(() =>
      registry.filter({ data_type: 'audio' } as unknown as FilterCriteria),
    ).toThrow(RangeError);
  });
});
