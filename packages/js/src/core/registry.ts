import { BaseExplainer } from './explainer';

export type ExplainerMeta = {
    scope: string; // 'local' | 'global'
    model_types: string[];
    data_types: string[];
    task_types?: string[];
    description?: string;
    paper_reference?: string;
    complexity?: string;
    requires_training_data?: boolean;
    supports_batching?: boolean;
};

type RegistryEntry = {
    class: new (...args: any[]) => BaseExplainer;
    meta: ExplainerMeta;
};

export class ExplainerRegistry {
    private registry: Record<string, RegistryEntry> = {};

    register(
        name: string,
        explainerClass: new (...args: any[]) => BaseExplainer,
        meta: ExplainerMeta,
        override: boolean = false
    ) {
        if (this.registry[name] && !override) {
            throw new Error(`Explainer '${name}' is already registered.`);
        }
        this.registry[name] = { class: explainerClass, meta };
    }

    get(name: string): RegistryEntry {
        if (!this.registry[name]) {
            throw new Error(`Explainer '${name}' is not registered.`);
        }
        return this.registry[name];
    }

    listExplainers(): string[] {
        return Object.keys(this.registry);
    }

    create(name: string, ...args: any[]): BaseExplainer {
        const entry = this.get(name);
        return new entry.class(...args);
    }

    filter(criteria: {
        scope?: string;
        model_type?: string;
        data_type?: string;
        task_type?: string;
    }): string[] {
        return Object.keys(this.registry).filter((name) => {
            const meta = this.registry[name].meta;
            if (criteria.scope && meta.scope !== criteria.scope) return false;

            if (criteria.model_type) {
                if (!meta.model_types.includes('any') && !meta.model_types.includes(criteria.model_type)) {
                    return false;
                }
            }

            if (criteria.data_type && !meta.data_types.includes(criteria.data_type)) return false;

            if (criteria.task_type) {
                if (!meta.task_types || !meta.task_types.includes(criteria.task_type)) {
                    return false;
                }
            }

            return true;
        });
    }
}
