import { describe, it, expect } from 'vitest';
// @ts-ignore
import { Explanation } from '../../src/core/explanation';

describe('Explanation', () => {
    it('should initialize with correct properties', () => {
        const data = { feature_attributions: { a: 0.5, b: 0.3 } };
        const explainerName = 'TestExplainer';
        const targetClass = 'cat';
        const featureNames = ['a', 'b'];
        const metadata = { duration: 100 };

        // @ts-ignore
        const explanation = new Explanation(
            explainerName,
            targetClass,
            data,
            featureNames,
            metadata
        );

        expect(explanation.explainerName).toBe(explainerName);
        expect(explanation.targetClass).toBe(targetClass);
        expect(explanation.explanationData).toEqual(data);
        expect(explanation.featureNames).toEqual(featureNames);
        expect(explanation.metadata).toEqual(metadata);
    });

    it('should return top features correctly', () => {
        const data = { feature_attributions: { a: 0.1, b: 0.8, c: -0.5 } };
        // @ts-ignore
        const explanation = new Explanation('Test', 'cat', data, ['a', 'b', 'c']);

        // @ts-ignore
        const top = explanation.getTopFeatures(2);
        expect(top).toEqual([['b', 0.8], ['c', -0.5]]);

        // @ts-ignore
        const topRaw = explanation.getTopFeatures(2, false);
        expect(topRaw).toEqual([['b', 0.8], ['a', 0.1]]);
    });
});
