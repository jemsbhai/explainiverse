import { BaseExplainer } from './explainer';

const SCOPES = ['local', 'global'] as const;
const MODEL_TYPES = ['any', 'tree', 'linear', 'neural', 'ensemble'] as const;
const DATA_TYPES = ['tabular', 'image', 'text', 'time_series'] as const;
const TASK_TYPES = ['classification', 'regression'] as const;
const CLAIM_STATUSES = ['verified', 'quarantined', 'unverified'] as const;

export type ExplainerScope = (typeof SCOPES)[number];
export type ModelType = (typeof MODEL_TYPES)[number];
export type DataType = (typeof DATA_TYPES)[number];
export type TaskType = (typeof TASK_TYPES)[number];
export type ClaimStatus = (typeof CLAIM_STATUSES)[number];

export interface ExplainerMeta {
  readonly scope: ExplainerScope;
  readonly model_types: readonly ModelType[];
  readonly data_types: readonly DataType[];
  readonly task_types?: readonly TaskType[];
  readonly description?: string;
  readonly paper_reference?: string;
  readonly complexity?: string;
  readonly requires_training_data?: boolean;
  readonly supports_batching?: boolean;
  readonly claim_status?: ClaimStatus;
  readonly claim_scope?: string;
}

export interface ResolvedExplainerMeta extends ExplainerMeta {
  readonly requires_training_data: boolean;
  readonly supports_batching: boolean;
  readonly claim_status: ClaimStatus;
  readonly claim_scope: string;
}

export type ExplainerConstructor = new (...args: never[]) => BaseExplainer;

export interface RegistryEntry {
  readonly class: ExplainerConstructor;
  readonly meta: ResolvedExplainerMeta;
}

export interface FilterCriteria {
  readonly scope?: ExplainerScope;
  readonly model_type?: ModelType;
  readonly data_type?: DataType;
  readonly task_type?: TaskType;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireRegistryName(name: unknown): asserts name is string {
  if (typeof name !== 'string') {
    throw new TypeError('explainer name must be a string');
  }
  if (name.trim().length === 0 || name !== name.trim()) {
    throw new RangeError('explainer name must be non-empty without surrounding whitespace');
  }
}

function validateCategory<T extends string>(
  fieldName: string,
  value: unknown,
  allowed: readonly T[],
): T {
  if (typeof value !== 'string') {
    throw new TypeError(`${fieldName} must be a string`);
  }
  if (!allowed.includes(value as T)) {
    throw new RangeError(`${fieldName} must be one of: ${allowed.join(', ')}`);
  }
  return value as T;
}

function validateCategoryList<T extends string>(
  fieldName: string,
  value: unknown,
  allowed: readonly T[],
): readonly T[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new TypeError(`${fieldName} must be a non-empty array`);
  }
  const validated = value.map((item) => validateCategory(fieldName, item, allowed));
  if (new Set(validated).size !== validated.length) {
    throw new RangeError(`${fieldName} must not contain duplicates`);
  }
  return Object.freeze(validated);
}

function validateOptionalString(fieldName: string, value: unknown): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'string') {
    throw new TypeError(`${fieldName} must be a string when provided`);
  }
  if (value.trim().length === 0) {
    throw new RangeError(`${fieldName} must be non-empty when provided`);
  }
  return value;
}

function validateOptionalBoolean(fieldName: string, value: unknown): boolean | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'boolean') {
    throw new TypeError(`${fieldName} must be a boolean when provided`);
  }
  return value;
}

function normalizeMeta(meta: ExplainerMeta): ResolvedExplainerMeta {
  if (!isRecord(meta)) {
    throw new TypeError('meta must be a non-null record');
  }

  const scope = validateCategory('scope', meta.scope, SCOPES);
  const modelTypes = validateCategoryList('model_types', meta.model_types, MODEL_TYPES);
  const dataTypes = validateCategoryList('data_types', meta.data_types, DATA_TYPES);
  const taskTypes =
    meta.task_types === undefined
      ? undefined
      : validateCategoryList('task_types', meta.task_types, TASK_TYPES);
  const description = validateOptionalString('description', meta.description);
  const paperReference = validateOptionalString('paper_reference', meta.paper_reference);
  const complexity = validateOptionalString('complexity', meta.complexity);
  const requiresTrainingData = validateOptionalBoolean(
    'requires_training_data',
    meta.requires_training_data,
  );
  const supportsBatching = validateOptionalBoolean(
    'supports_batching',
    meta.supports_batching,
  );
  const claimStatus =
    meta.claim_status === undefined
      ? 'unverified'
      : validateCategory('claim_status', meta.claim_status, CLAIM_STATUSES);
  const providedClaimScope = validateOptionalString('claim_scope', meta.claim_scope);
  if (claimStatus !== 'unverified' && providedClaimScope === undefined) {
    throw new RangeError(`${claimStatus} metadata requires an explicit claim_scope`);
  }

  return Object.freeze({
    scope,
    model_types: modelTypes,
    data_types: dataTypes,
    ...(taskTypes === undefined ? {} : { task_types: taskTypes }),
    ...(description === undefined ? {} : { description }),
    ...(paperReference === undefined ? {} : { paper_reference: paperReference }),
    ...(complexity === undefined ? {} : { complexity }),
    requires_training_data: requiresTrainingData ?? false,
    supports_batching: supportsBatching ?? false,
    claim_status: claimStatus,
    claim_scope:
      providedClaimScope ?? 'Implementation has not completed an accuracy audit.',
  });
}

function validateCriteria(criteria: FilterCriteria): void {
  if (!isRecord(criteria)) {
    throw new TypeError('criteria must be a non-null record');
  }
  const allowedKeys = new Set(['scope', 'model_type', 'data_type', 'task_type']);
  const unknownKeys = Object.keys(criteria).filter((key) => !allowedKeys.has(key));
  if (unknownKeys.length > 0) {
    throw new RangeError(`unknown filter criteria: ${unknownKeys.join(', ')}`);
  }
  if (criteria.scope !== undefined) {
    validateCategory('scope', criteria.scope, SCOPES);
  }
  if (criteria.model_type !== undefined) {
    validateCategory('model_type', criteria.model_type, MODEL_TYPES);
  }
  if (criteria.data_type !== undefined) {
    validateCategory('data_type', criteria.data_type, DATA_TYPES);
  }
  if (criteria.task_type !== undefined) {
    validateCategory('task_type', criteria.task_type, TASK_TYPES);
  }
}

/** Validated, insertion-ordered registry for experimental explainer classes. */
export class ExplainerRegistry {
  private readonly registry = new Map<string, RegistryEntry>();

  public register(
    name: string,
    explainerClass: ExplainerConstructor,
    meta: ExplainerMeta,
    override: boolean = false,
  ): void {
    requireRegistryName(name);
    if (
      typeof explainerClass !== 'function' ||
      typeof explainerClass.prototype?.explain !== 'function'
    ) {
      throw new TypeError('explainerClass must define an explain() method');
    }
    if (typeof override !== 'boolean') {
      throw new TypeError('override must be a boolean');
    }
    if (this.registry.has(name) && !override) {
      throw new Error(`Explainer '${name}' is already registered.`);
    }

    const normalizedMeta = normalizeMeta(meta);
    if (
      normalizedMeta.supports_batching &&
      typeof explainerClass.prototype?.explainBatch !== 'function'
    ) {
      throw new RangeError(
        'supports_batching=true requires an explainBatch() method on explainerClass',
      );
    }
    const entry = Object.freeze({ class: explainerClass, meta: normalizedMeta });
    this.registry.set(name, entry);
  }

  public get(name: string): RegistryEntry {
    requireRegistryName(name);
    const entry = this.registry.get(name);
    if (entry === undefined) {
      throw new Error(`Explainer '${name}' is not registered.`);
    }
    return entry;
  }

  public listExplainers(): string[] {
    return [...this.registry.keys()];
  }

  public create(name: string, ...args: unknown[]): BaseExplainer {
    const entry = this.get(name);
    return Reflect.construct(entry.class, args) as BaseExplainer;
  }

  public filter(criteria: FilterCriteria = {}): string[] {
    validateCriteria(criteria);
    return [...this.registry.entries()]
      .filter(([, entry]) => {
        const meta = entry.meta;
        if (criteria.scope !== undefined && meta.scope !== criteria.scope) {
          return false;
        }
        if (
          criteria.model_type !== undefined &&
          !meta.model_types.includes('any') &&
          !meta.model_types.includes(criteria.model_type)
        ) {
          return false;
        }
        if (
          criteria.data_type !== undefined &&
          !meta.data_types.includes(criteria.data_type)
        ) {
          return false;
        }
        if (
          criteria.task_type !== undefined &&
          !meta.task_types?.includes(criteria.task_type)
        ) {
          return false;
        }
        return true;
      })
      .map(([name]) => name);
  }
}
