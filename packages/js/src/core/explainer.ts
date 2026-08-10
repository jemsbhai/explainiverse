import type { Explanation } from './explanation';

/**
 * Minimal asynchronous contract shared by experimental TypeScript explainers.
 *
 * This class stores the supplied model by reference. It does not validate or
 * configure the model, impose a universal explanation argument signature, or
 * provide implicit batching. Concrete implementations own those contracts.
 */
export abstract class BaseExplainer<Model = unknown> {
  public readonly model: Model;

  public constructor(model: Model) {
    this.model = model;
  }

  /** Generate an explanation using the concrete method's documented arguments. */
  public abstract explain(...args: unknown[]): Promise<Explanation>;
}
