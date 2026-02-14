/**
 * Unified container for explanation results.
 */
export class Explanation {
    /**
     * Name of the explainer that generated this explanation
     */
    public explainerName: string;

    /**
     * The class/output being explained
     */
    public targetClass: string;

    /**
     * Dictionary containing explanation details (e.g., feature_attributions)
     */
    public explanationData: Record<string, any>;

    /**
     * Optional list of feature names
     */
    public featureNames?: string[];

    /**
     * Optional additional metadata
     */
    public metadata: Record<string, any>;

    constructor(
        explainerName: string,
        targetClass: string,
        explanationData: Record<string, any>,
        featureNames?: string[],
        metadata?: Record<string, any>
    ) {
        this.explainerName = explainerName;
        this.targetClass = targetClass;
        this.explanationData = explanationData;
        this.featureNames = featureNames;
        this.metadata = metadata || {};
    }

    /**
     * Get feature attributions if available.
     */
    public getAttributions(): Record<string, number> | undefined {
        return this.explanationData.feature_attributions;
    }

    /**
     * Get the top-k most important features.
     */
    public getTopFeatures(k: number = 5, absolute: boolean = true): [string, number][] {
        const attributions = this.getAttributions();
        if (!attributions) {
            return [];
        }

        const entries = Object.entries(attributions);

        entries.sort((a, b) => {
            const valA = absolute ? Math.abs(a[1]) : a[1];
            const valB = absolute ? Math.abs(b[1]) : b[1];
            return valB - valA;
        });

        return entries.slice(0, k);
    }
}
