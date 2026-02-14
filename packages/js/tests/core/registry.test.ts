import { describe, it, expect, beforeEach } from 'vitest';
// @ts-ignore
import { ExplainerRegistry, ExplainerMeta } from '../../src/core/registry';
// @ts-ignore
import { BaseExplainer } from '../../src/core/explainer';
// @ts-ignore
import { Explanation } from '../../src/core/explanation';

class MockExplainer extends BaseExplainer {
    async explain(instance: any): Promise<Explanation> {
        return new Explanation('Mock', 'test', {});
    }
}

describe('ExplainerRegistry', () => {
    let registry: any;

    beforeEach(() => {
        registry = new ExplainerRegistry();
    });

    it('should register and retrieve an explainer', () => {
        const meta = {
            scope: 'local',
            model_types: ['any'],
            data_types: ['tabular'],
            description: 'Test explainer'
        };

        registry.register('mock', MockExplainer, meta);

        expect(registry.get('mock')).toEqual({
            class: MockExplainer,
            meta: meta
        });
    });

    it('should list registered explainers', () => {
        const meta = { scope: 'local', model_types: ['any'], data_types: ['tabular'] };
        registry.register('mock1', MockExplainer, meta);
        registry.register('mock2', MockExplainer, meta);

        expect(registry.listExplainers()).toEqual(['mock1', 'mock2']);
    });

    it('should create an explainer instance', () => {
        const meta = { scope: 'local', model_types: ['any'], data_types: ['tabular'] };
        registry.register('mock', MockExplainer, meta);

        const explainer = registry.create('mock', 'some_model');
        expect(explainer).toBeInstanceOf(MockExplainer);
        expect(explainer.model).toBe('some_model');
    });

    it('should throw error when getting unregistered explainer', () => {
        expect(() => registry.get('unknown')).toThrow();
    });

    it('should filter explainers by criteria', () => {
        const meta1 = { scope: 'local', model_types: ['tree'], data_types: ['tabular'] };
        const meta2 = { scope: 'global', model_types: ['any'], data_types: ['tabular'] };
        const meta3 = { scope: 'local', model_types: ['neural'], data_types: ['image'] };

        registry.register('local1', MockExplainer, meta1);
        registry.register('global1', MockExplainer, meta2);
        registry.register('image1', MockExplainer, meta3);

        const local = registry.filter({ scope: 'local' });
        expect(local).toContain('local1');
        expect(local).toContain('image1');
        expect(local.length).toBe(2);

        expect(registry.filter({ data_type: 'image' })).toEqual(['image1']);
        expect(registry.filter({ model_type: 'tree' })).toEqual(['local1']);
    });
});
