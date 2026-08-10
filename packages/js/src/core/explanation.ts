/** Values carried by an explanation payload. */
export type ExplanationRecord = Record<string, unknown>;

/** Snake-case wire representation compatible with the Python container. */
export interface ExplanationPayload {
  explainer_name: string;
  target_class: string;
  explanation_data: ExplanationRecord;
  feature_names: string[] | null;
  metadata: ExplanationRecord;
}

function isRecord(value: unknown): value is ExplanationRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function cloneRecord(value: ExplanationRecord, fieldName: string): ExplanationRecord {
  try {
    return structuredClone(value);
  } catch (error: unknown) {
    throw new TypeError(
      `${fieldName} must contain structured-clone-compatible values`,
      { cause: error },
    );
  }
}

function requireNonEmptyString(value: unknown, fieldName: string): asserts value is string {
  if (typeof value !== 'string') {
    throw new TypeError(`${fieldName} must be a string`);
  }
  if (value.trim().length === 0) {
    throw new RangeError(`${fieldName} must be non-empty`);
  }
}

/** Validated container for one explanation result. */
export class Explanation {
  public readonly explainerName: string;
  public readonly targetClass: string;
  public readonly explanationData: ExplanationRecord;
  public readonly featureNames?: readonly string[];
  public readonly metadata: ExplanationRecord;

  public constructor(
    explainerName: string,
    targetClass: string,
    explanationData: ExplanationRecord,
    featureNames?: readonly string[],
    metadata?: ExplanationRecord,
  ) {
    requireNonEmptyString(explainerName, 'explainerName');
    requireNonEmptyString(targetClass, 'targetClass');
    if (!isRecord(explanationData)) {
      throw new TypeError('explanationData must be a non-null record');
    }
    if (featureNames !== undefined) {
      if (!Array.isArray(featureNames)) {
        throw new TypeError('featureNames must be an array of strings or undefined');
      }
      for (const featureName of featureNames) {
        requireNonEmptyString(featureName, 'featureNames entries');
      }
    }
    if (metadata !== undefined && !isRecord(metadata)) {
      throw new TypeError('metadata must be a non-null record or undefined');
    }

    this.explainerName = explainerName;
    this.targetClass = targetClass;
    this.explanationData = cloneRecord(explanationData, 'explanationData');
    this.featureNames =
      featureNames === undefined ? undefined : Object.freeze([...featureNames]);
    this.metadata = cloneRecord(metadata ?? {}, 'metadata');
  }

  /** Return a validated copy of feature attributions, when present. */
  public getAttributions(): Record<string, number> | undefined {
    const attributions = this.explanationData.feature_attributions;
    if (attributions === undefined) {
      return undefined;
    }
    if (!isRecord(attributions)) {
      throw new TypeError('feature_attributions must be a record when present');
    }

    const entries: [string, number][] = [];
    for (const [featureName, value] of Object.entries(attributions)) {
      requireNonEmptyString(featureName, 'attribution keys');
      if (typeof value !== 'number') {
        throw new TypeError('attribution values must be numeric scalars');
      }
      if (!Number.isFinite(value)) {
        throw new RangeError('attribution values must be finite');
      }
      entries.push([featureName, value]);
    }
    return Object.fromEntries(entries);
  }

  /** Return the top-k attributions, ranked by magnitude by default. */
  public getTopFeatures(k: number = 5, absolute: boolean = true): [string, number][] {
    if (!Number.isInteger(k) || k <= 0) {
      throw new RangeError('k must be a positive integer');
    }
    if (typeof absolute !== 'boolean') {
      throw new TypeError('absolute must be a boolean');
    }

    const entries = Object.entries(this.getAttributions() ?? {});
    entries.sort((left, right) => {
      const leftValue = absolute ? Math.abs(left[1]) : left[1];
      const rightValue = absolute ? Math.abs(right[1]) : right[1];
      return rightValue - leftValue;
    });
    return entries.slice(0, k);
  }

  /** Return the feature index, or undefined when names are absent or unmatched. */
  public getFeatureIndex(featureName: string): number | undefined {
    const index = this.featureNames?.indexOf(featureName) ?? -1;
    return index < 0 ? undefined : index;
  }

  /** Return a detached snake-case payload suitable for Python interoperability. */
  public toObject(): ExplanationPayload {
    return {
      explainer_name: this.explainerName,
      target_class: this.targetClass,
      explanation_data: cloneRecord(this.explanationData, 'explanationData'),
      feature_names: this.featureNames === undefined ? null : [...this.featureNames],
      metadata: cloneRecord(this.metadata, 'metadata'),
    };
  }

  /** Construct an Explanation from an untrusted snake-case payload. */
  public static fromObject(payload: unknown): Explanation {
    if (!isRecord(payload)) {
      throw new TypeError('payload must be a non-null record');
    }
    for (const field of ['explainer_name', 'target_class', 'explanation_data']) {
      if (!Object.hasOwn(payload, field)) {
        throw new TypeError(`payload is missing required field: ${field}`);
      }
    }

    const featureNames = payload.feature_names;
    return new Explanation(
      payload.explainer_name as string,
      payload.target_class as string,
      payload.explanation_data as ExplanationRecord,
      featureNames === null ? undefined : (featureNames as readonly string[] | undefined),
      payload.metadata as ExplanationRecord | undefined,
    );
  }
}
