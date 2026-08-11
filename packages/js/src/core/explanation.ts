/** JSON/Python-wire scalar values carried by an explanation payload. */
export type ExplanationPrimitive = string | number | boolean | null;

/** Recursive values that survive an exact JSON transport round trip. */
export type ExplanationValue =
  | ExplanationPrimitive
  | ExplanationValue[]
  | { [key: string]: ExplanationValue };

/** String-keyed JSON/Python-wire object carried by an explanation payload. */
export type ExplanationRecord = Record<string, ExplanationValue>;

/** Stable identifier for the first opt-in cross-language wire schema. */
export const EXPLANATION_WIRE_SCHEMA_VERSION = 'explainiverse.explanation.v1';

/** Snake-case wire representation compatible with the Python container. */
export interface ExplanationPayload {
  explainer_name: string;
  target_class: string;
  explanation_data: ExplanationRecord;
  feature_names: string[] | null;
  metadata: ExplanationRecord;
}

/** Versioned snake-case payload used by explicit wire producer/consumer APIs. */
export interface VersionedExplanationPayload extends ExplanationPayload {
  schema_version: typeof EXPLANATION_WIRE_SCHEMA_VERSION;
}

function isRecord(value: unknown): value is ExplanationRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function cloneWireValue(
  value: unknown,
  fieldName: string,
  activeObjects: WeakSet<object>,
): ExplanationValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${fieldName} must contain only finite numbers`);
    }
    if (Object.is(value, -0)) {
      throw new RangeError(
        `${fieldName} must not contain negative zero because JSON transport erases its sign`,
      );
    }
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
      throw new RangeError(
        `${fieldName} integer must be within JavaScript's safe-integer range for exact transport`,
      );
    }
    return value;
  }
  if (typeof value !== 'object') {
    throw new TypeError(`${fieldName} must contain only JSON-compatible values`);
  }
  if (activeObjects.has(value)) {
    throw new TypeError(`${fieldName} must not contain cyclic references`);
  }

  activeObjects.add(value);
  try {
    if (Array.isArray(value)) {
      if (Object.getPrototypeOf(value) !== Array.prototype) {
        throw new TypeError(`${fieldName} must contain only ordinary arrays`);
      }
      if (Object.getOwnPropertySymbols(value).length > 0) {
        throw new TypeError(`${fieldName} arrays must not contain symbol properties`);
      }
      const propertyNames = Object.getOwnPropertyNames(value);
      if (propertyNames.length !== value.length + 1) {
        throw new TypeError(
          `${fieldName} arrays must be dense and must not contain extra properties`,
        );
      }

      const cloned: ExplanationValue[] = [];
      for (let index = 0; index < value.length; index += 1) {
        const key = String(index);
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (descriptor === undefined) {
          throw new TypeError(`${fieldName} arrays must not contain sparse holes`);
        }
        if (!('value' in descriptor)) {
          throw new TypeError(`${fieldName}[${index}] must not be an accessor property`);
        }
        cloned.push(
          cloneWireValue(descriptor.value, `${fieldName}[${index}]`, activeObjects),
        );
      }
      return cloned;
    }

    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError(
        `${fieldName} must contain only arrays and plain string-keyed objects`,
      );
    }
    if (Object.getOwnPropertySymbols(value).length > 0) {
      throw new TypeError(`${fieldName} must not contain symbol-keyed properties`);
    }
    const propertyNames = Object.getOwnPropertyNames(value);
    const enumerableNames = Object.keys(value);
    if (propertyNames.length !== enumerableNames.length) {
      throw new TypeError(`${fieldName} must not contain non-enumerable properties`);
    }

    const cloned: ExplanationRecord = {};
    for (const key of enumerableNames) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (descriptor === undefined || !('value' in descriptor)) {
        throw new TypeError(`${fieldName}.${key} must not be an accessor property`);
      }
      const clonedValue = cloneWireValue(
        descriptor.value,
        `${fieldName}.${key}`,
        activeObjects,
      );
      // Assignment to the special key "__proto__" mutates an ordinary
      // object's prototype. Define every wire property explicitly so that
      // arbitrary JSON keys remain own data properties.
      Object.defineProperty(cloned, key, {
        value: clonedValue,
        enumerable: true,
        configurable: true,
        writable: true,
      });
    }
    return cloned;
  } finally {
    activeObjects.delete(value);
  }
}

function cloneRecord(value: ExplanationRecord, fieldName: string): ExplanationRecord {
  const cloned = cloneWireValue(value, fieldName, new WeakSet<object>());
  if (!isRecord(cloned)) {
    throw new TypeError(`${fieldName} must be a non-null record`);
  }
  return cloned;
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
    let clonedFeatureNames: ExplanationValue[] | undefined;
    if (featureNames !== undefined) {
      const clonedNames = cloneWireValue(
        featureNames,
        'featureNames',
        new WeakSet<object>(),
      );
      if (!Array.isArray(clonedNames)) {
        throw new TypeError('featureNames must be an array of strings or undefined');
      }
      for (const featureName of clonedNames) {
        requireNonEmptyString(featureName, 'featureNames entries');
      }
      if (new Set(clonedNames).size !== clonedNames.length) {
        throw new RangeError('featureNames must contain unique names');
      }
      clonedFeatureNames = clonedNames;
    }
    if (metadata !== undefined && !isRecord(metadata)) {
      throw new TypeError('metadata must be a non-null record or undefined');
    }

    this.explainerName = explainerName;
    this.targetClass = targetClass;
    this.explanationData = cloneRecord(explanationData, 'explanationData');
    this.featureNames =
      clonedFeatureNames === undefined
        ? undefined
        : Object.freeze(clonedFeatureNames as string[]);
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

  /** Return the explicit versioned Python/JavaScript wire representation. */
  public toWireObject(): VersionedExplanationPayload {
    return {
      schema_version: EXPLANATION_WIRE_SCHEMA_VERSION,
      ...this.toObject(),
    };
  }

  /** Construct an Explanation from an untrusted snake-case payload. */
  public static fromObject(payload: unknown): Explanation {
    const clonedPayload = cloneWireValue(payload, 'payload', new WeakSet<object>());
    if (!isRecord(clonedPayload)) {
      throw new TypeError('payload must be a non-null record');
    }
    const schemaFields = [
      'explainer_name',
      'target_class',
      'explanation_data',
      'feature_names',
      'metadata',
    ] as const;
    const schemaFieldSet = new Set<string>(schemaFields);
    const unknownFields = Object.keys(clonedPayload).filter(
      (field) => !schemaFieldSet.has(field),
    );
    if (unknownFields.length > 0) {
      throw new TypeError(`payload contains unknown field(s): ${unknownFields.join(', ')}`);
    }
    for (const field of schemaFields) {
      if (!Object.hasOwn(clonedPayload, field)) {
        throw new TypeError(`payload is missing required field: ${field}`);
      }
    }

    const featureNames = clonedPayload.feature_names;
    return new Explanation(
      clonedPayload.explainer_name as string,
      clonedPayload.target_class as string,
      clonedPayload.explanation_data as ExplanationRecord,
      featureNames === null ? undefined : (featureNames as readonly string[]),
      clonedPayload.metadata as ExplanationRecord,
    );
  }

  /** Construct an Explanation from an untrusted versioned wire payload. */
  public static fromWireObject(payload: unknown): Explanation {
    const clonedPayload = cloneWireValue(payload, 'payload', new WeakSet<object>());
    if (!isRecord(clonedPayload)) {
      throw new TypeError('wire payload must be a non-null record');
    }
    const schemaFields = [
      'schema_version',
      'explainer_name',
      'target_class',
      'explanation_data',
      'feature_names',
      'metadata',
    ] as const;
    const schemaFieldSet = new Set<string>(schemaFields);
    const unknownFields = Object.keys(clonedPayload).filter(
      (field) => !schemaFieldSet.has(field),
    );
    if (unknownFields.length > 0) {
      throw new TypeError(
        `wire payload contains unknown field(s): ${unknownFields.join(', ')}`,
      );
    }
    for (const field of schemaFields) {
      if (!Object.hasOwn(clonedPayload, field)) {
        throw new TypeError(`wire payload is missing required field: ${field}`);
      }
    }
    if (clonedPayload.schema_version !== EXPLANATION_WIRE_SCHEMA_VERSION) {
      throw new RangeError(
        `unsupported Explanation schema_version; expected ${EXPLANATION_WIRE_SCHEMA_VERSION}`,
      );
    }
    return Explanation.fromObject({
      explainer_name: clonedPayload.explainer_name,
      target_class: clonedPayload.target_class,
      explanation_data: clonedPayload.explanation_data,
      feature_names: clonedPayload.feature_names,
      metadata: clonedPayload.metadata,
    });
  }
}
