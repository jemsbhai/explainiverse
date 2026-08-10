# Cross-framework comparison status

The former "state-of-the-art comparison" has been withdrawn from the active
documentation.

It mixed method names, compatibility aliases, paper metrics, implementation
variants, and library-defined diagnostics into raw counts. It also compared
other projects without pinned package versions, dated source snapshots, or
executable capability probes. Those numbers therefore could not support claims
such as "most comprehensive," "ahead of Quantus," or "only framework."

Explainiverse currently makes no cross-framework coverage or leadership claim.
The runtime registries and focused accuracy tests are the internal sources of
truth for implemented scope:

- `ExplainerMeta.claim_status` is one of `verified`, `quarantined`, or
  `unverified`.
- `ExplainerMeta.claim_scope` states the supported boundary of each explainer.
- Evaluation modules distinguish canonical formulas, explicit adaptations,
  historical compatibility names, and library-defined diagnostics.
- Undefined or non-identifiable cases are expected to fail explicitly rather
  than receive plausible-looking scores.

## Requirements for a future comparison

A new comparison may be published only when it includes all of the following:

1. Exact package versions, source revisions, Python version, and optional
   dependency sets for every framework.
2. A public operational definition for what counts as one explainer or one
   metric, including treatment of aliases, variants, aggregations, and task-
   specific overloads.
3. Executable import/constructor/behavior probes rather than documentation-name
   matching alone.
4. Formula-level tests for any claim of equivalence between similarly named
   methods.
5. Separate reporting of canonical implementations, adaptations, compatibility
   APIs, and descriptive diagnostics.
6. A dated, reproducible artifact that can be rerun when any compared project
   changes.

Until those conditions are met, method and metric counts are deliberately
omitted from project marketing and release claims.
