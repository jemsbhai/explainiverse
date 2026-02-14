import { Explanation } from './explanation';

/**
 * Abstract base class for all explainers.
 */
export abstract class BaseExplainer {
    /**
     * The model being explained.
     */
    public model: any;

    constructor(model: any) {
        this.model = model;
    }

    /**
     * Generate an explanation for a single input instance.
     * @param instance The input to explain.
     * @param args Optional method-specific parameters.
     */
    abstract explain(instance: any, ...args: any[]): Promise<Explanation>;
}
