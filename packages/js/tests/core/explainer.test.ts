import { describe, it, expect, vi } from 'vitest';
// @ts-ignore
import { BaseExplainer } from '../../src/core/explainer';
// @ts-ignore
import { Explanation } from '../../src/core/explanation';

// Concrete implementation for testing
class TestExplainer extends BaseExplainer {
    async explain(instance: any): Promise<Explanation> {
        return new Explanation(
            'TestExplainer',
            'test_class',
            { feature_attributions: {} }
        );
    }
}

describe('BaseExplainer', () => {
    it('should be extended by concrete classes', async () => {
        const model = { predict: vi.fn() };
        const explainer = new TestExplainer(model);

        expect(explainer.model).toBe(model);

        const explanation = await explainer.explain({});
        expect(explanation).toBeInstanceOf(Explanation);
        expect(explanation.explainerName).toBe('TestExplainer');
    });
});
