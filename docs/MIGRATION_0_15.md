# Migrating from 0.14 to the 0.15 development line

This document records intentional corrective tightenings in `0.15.0.dev0`. They prevent
ambiguous or fabricated results; they are not compatibility shims that preserve every
`0.14.0` edge case.

## Adapter and prediction contracts

- Custom classifier adapters can expose `prediction_output_kind` as `probabilities`,
  `class_labels`, or `scores`; regressors use `regression_values`. Declare the marker whenever
  a one-dimensional or one-column numeric output can contain only `0` and `1`. Without it,
  endpoint probabilities and hard labels are mathematically indistinguishable and strict
  consumers reject the output rather than guess.
- `SklearnAdapter` now declares its normalized public output kind and rejects complex results
  before any real cast. Class and feature names must be unique, nonblank strings.
- A one-column binary probability of exactly `0.5` now selects column/class 0, matching
  `argmax([1-p, p])` and the two-column path.
- `PyTorchAdapter` preserves model-aligned floating dtypes, widens bfloat16 to float32 only at
  NumPy boundaries, and offers opt-in `result_format="tensor"` and `"dlpack"` results that
  preserve bfloat16. Returned tensors are detached owned clones on the adapter device; DLPack
  capsules own an equivalent clone and may be consumed once. Caller inputs remain isolated
  from in-place models.
- Meta-device models and `adapter.to("meta")` are rejected before mutation because adapter
  computation endpoints cannot materialize discarded weights. A failed fully inherited
  `nn.Module.to()` remains recoverable only when its inherited rollback succeeds. If a custom
  `to`, `_apply`, or child-traversal implementation fails, exact semantic restoration cannot be
  proven even when a rollback call returns, so the adapter is permanently poisoned and every
  later operation requires reconstruction. Bound-method identity includes the real method type,
  owner, and function; callable objects cannot spoof `__func__`. Internal `_modules`,
  `_parameters`, and `_buffers` registries must be exact built-in `dict`/`OrderedDict` objects,
  and custom `__getattribute__`/`__setattr__`/`__delattr__` registered-state dispatch is rejected.
  Construction never invokes a virtual custom `train`/`eval`, and a real initial device move is
  allowed only through canonical move traversal. Meta remains a pre-mutation rejection.
- Prediction, gradient, layer, mode, layer-listing, and device-move operations on one
  `PyTorchAdapter` now share the model-state lock and lock order used by explanations. A mode or
  device request therefore cannot interleave with a prediction or be overwritten by explanation
  cleanup.
- Classification with `output_activation=None` may now declare
  `classification_output_kind="scores"` or `"probabilities"`. Declared probabilities are
  range/simplex validated. Probability-only consumers reject an undeclared multiclass matrix
  rather than infer meaning from its values.
- Named PyTorch layers still fail closed on repeated execution by default. Callers may now
  select an explicit zero-based `occurrence` (or CAM `target_occurrence`). The adapter traces
  and pins the total call count; an out-of-range selector or a later dynamic count change fails,
  and hooks are removed on every path. CAM results use immutable evidence returned by their own
  locked layer call rather than shared last-call fields. TCAV uses the separate
  `layer_occurrence` setting and rejects CAVs learned for a different occurrence.
- Gradient explanation contexts accept explicit `model_generators`, an opt-in
  `model_state_protocol`, and a name-addressable `model_state_fingerprint`. Declared generators
  and protocol-owned state restore after success or exception; any changed fingerprint fails
  before returning an explanation. Python/NumPy RNG, parameter values/rebinding, buffer
  rebinding, and arbitrary attributes remain outside the default pure-forward ownership
  contract unless declared this way. Custom `train`/`eval` dispatch is rejected before model
  work; canonical evaluation mode is entered by direct registered-module flags. Any failed
  registered/protocol/RNG restoration or post-restore validation poisons the adapter so later
  prediction, explanation, mode, and device operations require reconstruction.

## Explainer results and validation

- Gradient explainers no longer narrow supported float64 inputs/baselines/noise to float32.
  Invalid booleans, seeds, methods, empty concepts, and nonfinite values fail before model work.
- Regression target selection is task-first. Display `class_names` no longer turns a
  multi-output regression into an argmax classification.
- EigenCAM no longer applies a Grad-CAM ReLU. Signed principal projections therefore change
  before min-max normalization.
- Quarantined EigenGradCAM and GradCAMElementWise library variants now route product overflow,
  underflow, exact-centering loss, and pre-SVD scale loss through their scaled numerical paths.
  EigenGradCAM restores every representable raw projection amplitude so normalization metadata
  describes the formula result, and fails explicitly when a nonzero exact centered value cannot
  survive one global binary64 SVD scale or a restored projection cell is unrepresentable. The
  ordinary-range formulas remain unchanged; these numerical repairs do not promote either
  variant out of quarantine.
- Integrated Gradients and CAM share explicit image layouts. `auto` is now a rank-only
  convention (HW, CHW, or NCHW), never a channel-size heuristic; channel-last callers must set
  `hwc` or `nhwc`. Results record both configured/resolved layout and channel axis. NHW batches
  use the per-instance `hw` contract, and custom channel counts are accepted without guessing.
- All public operations on one built-in explainer instance now share a per-instance re-entrant
  lock. This covers persistent RNGs and backend objects as well as IG shape, DeepSHAP background,
  TCAV concept, and LRP rule state. Concurrent IG shape inference uses one atomic
  validate/compute/commit transaction; a failed call commits nothing.
  Input shape is committed only after a successful explanation.
- TCAV's learned and random concept stores are private. Public mappings are immutable defensive
  snapshots; returned CAV vectors are bit-exact owned read-only copies, classifiers/metadata are defensive
  copies, and `learn_*`/`get_concept` never return an alias to the stored CAV. Concurrent snapshot
  reads share the explainer lock with learning/removal.
- `DeepLIFTShapExplainer` rejects inherited single-baseline and ordinary-IG comparison helpers.
  Use its background-data API; a background-expected IG comparator is not implemented.
- Captum-backed graph/operator restrictions and the mandatory 0.8.0/current parity gates are
  enumerated in [`CAPTUM_SUPPORT_MATRIX.md`](CAPTUM_SUPPORT_MATRIX.md). The LRP rule classes
  remain an explicitly documented private Captum dependency because neither supported Captum
  endpoint exports a public propagation-rule extension API. DeepLIFT/DeepSHAP now accept only an
  exact `nn.Sequential` graph (including nested exact Sequentials) or one exact documented leaf;
  arbitrary custom/FX roots, subclasses, `nn.Softmax`, untracked BatchNorm, pre-existing local or
  global execution hooks, noncanonical forward/call pipelines, and post-construction graph
  mutation fail closed. Multiclass softmax prediction-score attribution is rejected; raw/model
  score attribution remains supported. LRP also rejects state/load-state hooks or overrides.
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
- Consistency defaults to stable feature order at a top-k cutoff tie. Detail mode records tie
  incidence; `tie_policy="reject"` refuses a tie spanning the cutoff and
  `tie_policy="include_all"` includes the complete tied set. Comparisons reject mixed policies.
- `ExplanationSuite.compare()` uses semantic target equality, not `repr` identity.
- Metric registry entries now expose reviewed `stochasticity` values (`deterministic`,
  `conditional`, or `stochastic`). The legacy `stochastic` boolean remains a derived view.
- SSIM requires spatial axes of at least 3 by 3 and selects an explicit valid window. Smaller
  maps fail with explicit Pearson/cosine or caller-controlled aggregation guidance.
- Scale-invariant reductions now rescale finite inputs. Nonfinite inputs and genuinely
  unrepresentable outputs still fail explicitly.
- Exact reduction is retained through the final reported scalar. A scalar metric may therefore
  succeed when cancellation or scaling makes its aggregate representable even though an
  individual requested detail value is outside float64. Legacy `return_details=True` still
  fails with `DetailRepresentationError` instead of emitting `NaN`, infinity, or a rounded
  fiction. The affected detail-capable metrics accept
  `detail_format="scaled_decimal_v1"`, a versioned finite-JSON payload that retains ordinary
  floats. The selector name is retained for compatibility; exceptional elements use either a
  truthful `exact_decimal` string or canonical `exact_fraction` numerator/denominator strings.
  Ratios with repeating decimal expansions are never rounded and labelled exact. NumPy
  floating scalars wider than binary64 fail closed; callers must supply their exact value as
  `Decimal` or `Fraction` instead of accepting an implicit narrowing conversion.
- `run_seeded_replicates()` and `summarize_replicate_estimates()` report explicit seeds, sample
  counts, replicate estimates, a scoped Student-t interval where at least two independent
  streams exist, and an order-dependent convergence diagnostic. A fused high-precision path
  forms Student-t endpoints before a positive subnormal standard error can round to zero; an
  unrepresentable standalone standard error is reported as undefined rather than fabricated.
  Estimate, evaluator-score, comparison-score, confidence-level, and convergence-tolerance
  inputs cross one lossless binary64 gate. Exactly representable Python/NumPy integers and
  floats, `Decimal`, and `Fraction` values are accepted; inexact fractions/decimals, integers
  that change under conversion, wider NumPy floating scalars, and undeclared custom `Real`
  types fail closed. A caller who intentionally accepts binary64 rounding must perform that
  rounding explicitly first and pass the resulting ordinary Python `float`. The terminal
  cumulative-mean change and any tolerance decision use exact rational arithmetic over the
  accepted binary64 estimates; rounded display means cannot fabricate a zero change or a
  convergence claim. If the exact change itself is not representable as float64, its display
  field is undefined with a reason while the exact tolerance comparison remains available.
  They never label a finite estimate a global proof. `evaluate_intervention_sensitivity()`
  evaluates named prespecified baselines/backgrounds/interventions and records deterministic
  exact-reference fingerprints. Scalar NumPy floats through binary64 include their canonical
  width and exact value; wider scalar types fail closed instead of being narrowed. Cross-report
  comparison requires the declared intervention contract, ordered names, and ordered reference
  fingerprints to match exactly. The generic `ExplanationSuite.compare()`
  `comparison_contract` remains a caller assertion and is not a substitute for this exact
  intervention-reference identity check.
- The promoted offline finite-estimator tutorial applies those APIs to one explicit class-1
  probability-drop estimand. It asserts formula/output space, reproduces fresh seeded reports
  exactly, and shows a sign change across three prespecified empirical replacements without
  selecting a universal baseline or making a global faithfulness claim.
- Fairness-related group diagnostics now carry pairwise `effect_size_defined`/reason and
  Mann-Whitney defined/reason states. Signed infinity, a tied-sample `None`, and the zero-variance
  equal-means convention remain distinct; every payload states that no fairness conclusion is
  defined by the diagnostic alone.

## Packaging and JavaScript

- The redundant `image` extra was removed. `scikit-image>=0.20,<1.0` is a direct base
  dependency because mandatory LIME already requires it. Replace `.[image]` with the base
  installation; `.[all]` continues to add Torch and Captum.
- The optional Torch range is now conservatively capped at `<3.0`; widening to a future major
  requires an explicit compatibility gate. Isolated source builds use the exact reviewed
  `poetry-core==2.3.1` backend instead of resolving an arbitrary future backend.
- Python now exposes separate `Explanation.to_wire_dict()`/`from_wire_dict()` methods; the broad
  `to_dict()`/`from_dict()` contract is unchanged. The private JavaScript `Explanation` wire
  accepts the same strict JSON-safe subset through `toWireObject()`/`fromWireObject()`: finite
  numbers other than signed negative zero and integers outside JavaScript's safe-integer range,
  strings, booleans, null, dense ordinary arrays, and plain data objects. Duplicate names,
  accessors, special prototypes, symbols, cycles, sparse or decorated arrays, and host objects
  are rejected. Its top level is the closed six-field
  `explainiverse.explanation.v1` schema (`schema_version` plus the five explanation fields), so
  missing, unknown, and unsupported-version payloads fail. Wire-v1 producers and consumers also
  require a non-empty string target and apply JavaScript's safe-integer boundary to both integer
  and integer-valued-float producers. Shared fixtures plus a real Python-to-Node-to-Python
  subprocess bridge own the producer/consumer contract.

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
