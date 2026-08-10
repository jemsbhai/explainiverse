import { describe, expect, it, vi } from 'vitest';

import { BaseExplainer } from '../../src/core/explainer';
import { Explanation } from '../../src/core/explanation';

type Model = { predict: ReturnType<typeof vi.fn> };

class TestExplainer extends BaseExplainer<Model> {
  public explain(_instance: unknown): Promise<Explanation> {
    return Promise.resolve(
      new Explanation('TestExplainer', 'test_class', { feature_attributions: {} }),
    );
  }
}

describe('BaseExplainer', () => {
  it('retains the supplied model by reference and delegates explanation semantics', async () => {
    const model: Model = { predict: vi.fn() };
    const explainer = new TestExplainer(model);

    expect(explainer.model).toBe(model);
    await expect(explainer.explain({})).resolves.toBeInstanceOf(Explanation);
  });
});
