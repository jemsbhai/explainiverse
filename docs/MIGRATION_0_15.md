# Migrating from 0.14 to the 0.15 development line

This document records intentional corrective tightenings in `0.15.0.dev0`. They prevent
ambiguous or fabricated results; they are not compatibility shims that preserve every
`0.14.0` edge case.

## Adapter and prediction contracts

- Custom classifier adapters can expose `prediction_output_kind` as `probabilities`,
  `class_labels`, or `scores`; regressors use `regression_values`. Declare the marker whenever
  a one-dimensional numeric output can contain only `0` and `1`. Without it, endpoint
  probabilities and hard labels are mathematically indistinguishable and strict consumers may
  reject the output rather than guess.
- `SklearnAdapter` now declares its normalized public output kind and rejects complex results
  before any real cast. Class and feature names must be unique, nonblank strings.
- A one-column binary probability of exactly `0.5` now selects column/class 0, matching
  `argmax([1-p, p])` and the two-column path.
- `PyTorchAdapter` preserves model-aligned floating dtypes, widens bfloat16 to float32 only at
  NumPy boundaries, isolates caller inputs from in-place models, and reports failed device
  rollback explicitly. Classification with `output_activation=None` remains deliberately
  undeclared because that legacy mode may expose either raw scores or pre-normalized
  probabilities.
- Named PyTorch layers that execute more than once now fail closed. No implicit first/last
  occurrence is selected.

## Explainer results and validation

- Gradient explainers no longer narrow supported float64 inputs/baselines/noise to float32.
  Invalid booleans, seeds, methods, empty concepts, and nonfinite values fail before model work.
- Regression target selection is task-first. Display `class_names` no longer turns a
  multi-output regression into an argmax classification.
- EigenCAM no longer applies a Grad-CAM ReLU. Signed principal projections therefore change
  before min-max normalization.
- Two-dimensional Integrated Gradients inputs use an explicit implicit grayscale channel.
  Input shape is committed only after a successful explanation.
- `DeepLIFTShapExplainer` rejects inherited single-baseline and ordinary-IG comparison helpers.
  Use its background-data API; a background-expected IG comparator is not implemented.
- Third-party `BaseCAMExplainer` subclasses are not formula-verified unless they opt in with
  method-specific evidence.
- LIME `num_features`, SAGE custom loss, permutation seeds, TCAV flags/concepts, SmoothGrad
  scale percentages, and ProtoDash options now use strict boundary types. Truthy strings and
  numeric booleans are no longer coerced.

## ProtoDash mass and objective results

Canonical objective weights are normalized only when their nonnegative total is strictly
greater than the explainer's configured `epsilon`. A total less than or equal to `epsilon`,
including a nonzero near-zero total, now produces an all-zero `weights` vector rather than a
fabricated uniform distribution. Check `normalized_weights_defined` before treating weights as
a probability distribution. If MMD was requested, also check `mmd_defined`; `mmd_score` is
absent when no normalized measure exists.

ProtoDash objective diagnostics now reject nonfinite inputs and trap overflow, underflow, and
invalid intermediate arithmetic. When separately overflowing floating-point terms cancel to a
representable float64 value, an exact binary-rational fallback returns that value. A genuinely
unrepresentable objective raises `ValueError` instead of placing `NaN` or infinity in an
`Explanation` payload.

## Evaluation and suite behavior

- `LocalisationMask` owns a read-only copy. Construct a new mask instead of mutating an existing
  one.
- Public boolean modes reject non-booleans. Duplicate `k`/`top_k` requests are rejected instead
  of overwriting result keys.
- Consistency uses stable feature order at a top-k cutoff tie. Keep one tie policy fixed when
  comparing runs.
- `ExplanationSuite.compare()` uses semantic target equality, not `repr` identity.
- Metric registry entries now expose reviewed `stochasticity` values (`deterministic`,
  `conditional`, or `stochastic`). The legacy `stochastic` boolean remains a derived view.
- SSIM requires spatial axes of at least 3 by 3 and selects an explicit valid window. Smaller
  maps must use another similarity or upstream aggregation.
- Scale-invariant reductions now rescale finite inputs. Nonfinite inputs and genuinely
  unrepresentable outputs still fail explicitly.
- Exact reduction is retained through the final reported scalar. A scalar metric may therefore
  succeed when cancellation or scaling makes its aggregate representable even though an
  individual requested detail value is outside float64. In that case `return_details=True`
  fails with a detail-specific error instead of emitting `NaN`, infinity, or a rounded fiction.

## Packaging and JavaScript

- The redundant `image` extra was removed. `scikit-image>=0.20,<1.0` is a direct base
  dependency because mandatory LIME already requires it. Replace `.[image]` with the base
  installation; `.[all]` continues to add Torch and Captum.
- The optional Torch range is now conservatively capped at `<3.0`; widening to a future major
  requires an explicit compatibility gate. Isolated source builds use the exact reviewed
  `poetry-core==2.3.1` backend instead of resolving an arbitrary future backend.
- The private JavaScript `Explanation` wire accepts only a strict JSON-safe subset: finite
  numbers other than signed negative zero and integers outside JavaScript's safe-integer range,
  strings, booleans, null, dense ordinary arrays, and plain data objects. Duplicate names,
  accessors, special prototypes, symbols, cycles, sparse or decorated arrays, and host objects
  are rejected. Its top level is a closed five-field schema, so all five fields are required
  (`feature_names: null` and `metadata: {}` represent empty optional content) and unknown fields
  also fail. Python's broader `to_dict()` contract is unchanged and is not automatically a JS
  wire payload.

## Release operations

The publication workflow accepts only a signed annotated stable tag whose version matches
`pyproject.toml` and whose commit is on `main`. It builds once, runs the complete
Python/JS/tutorial gates, attaches hashes, an SBOM, and provenance, publishes through the
`pypi` environment, and then creates the GitHub Release. Branch protection, immutable-tag
rules, environment reviewers, and the PyPI Trusted Publisher are external repository/service
settings; the release checklist must verify them rather than infer them from workflow YAML.
Dispatch the workflow from the tag itself, with the same tag as its input—for example
`gh workflow run publish-pypi.yml --ref v0.15.0 -f tag=v0.15.0`. The workflow rejects a branch
dispatch even if its checkout was later pointed at the tag, because attestation provenance is
bound to the workflow ref and SHA.
If a post-publication GitHub Release step fails, rerun failed jobs while retained artifacts
exist; do not rerun the PyPI job or add an unchecked `skip-existing` path.
