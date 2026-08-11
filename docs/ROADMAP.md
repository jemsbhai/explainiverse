# Explainiverse accuracy roadmap

This roadmap tracks evidence and claim boundaries, not a race to add methods. Historical
method/metric totals and “complete” phase labels were removed because aliases, adaptations,
variants, and descriptive statistics are not comparable scientific units.

## Release rule

No additional feature is ready to merge until every affected public claim has:

1. a primary or authoritative reference when it uses a literature name;
2. an explicit formula and target/output-space contract;
3. analytical, invariant, or official-reference evidence;
4. clear unsupported-domain failures;
5. accurate public documentation and metadata; and
6. formatting, lint, type, test, build, and package verification proportional to the change.

Passing tests supports only the tested contract. It is not evidence that an explanation is
causal, useful, fair, faithful, or appropriate for a deployment.

## Current audit state

### Explainers

- Verified within registry scopes: LIME, KernelSHAP, TreeSHAP, Integrated Gradients,
  tabular DeepLIFT/DeepSHAP/SmoothGrad/Saliency, LRP, global target-class-set TCAV,
  Grad-CAM, HiResCAM, XGradCAM, LayerCAM, EigenCAM, the paper Algorithm-1
  raw-output/channel-softmax Score-CAM variant, AblationCAM, permutation importance, PDP,
  continuous first-order ALE, marginal-imputer SAGE, continuous-numeric `anchor_tabular`
  rules with strict sequential-KL certification, and ProtoDash objective weights.
- Quarantined compatibility APIs: fixed-sample Anchors-style search, constrained
  counterfactual search under the historical `counterfactual` key, EigenGradCAM, and
  GradCAMElementWise.
- Not exposed: Grad-CAM++, because the available adapter does not supply the general
  higher-derivative computation required by that method.

The runtime `ExplainerMeta.claim_status` and `claim_scope` fields are authoritative. A grouped
roadmap entry never broadens those per-method scopes.
All PyTorch explainer scopes are currently CPU-verified only. CUDA remains outside the audited
device scope until a GPU runner is part of the release gate.

### Evaluation APIs

- Formula-audited families: core and extended faithfulness, perturbation/AOPC/ROAR,
  robustness/relative stability, agreement, complexity, localisation, randomisation,
  axiomatic checks, and fairness-related audits.
- Explicit adaptations or compatibility diagnostics remain named as such. Important examples
  are deterministic PGI/PGU replacement, tabular ERASER scores, finite-sample local Lipschitz,
  mean sensitivity, attribution IoU, threshold-count compatibility aliases, and noncanonical
  fairness diagnostics.
- Every sampled maximum or supremum is a finite estimate. Every deletion/replacement metric is
  conditional on its intervention and baseline. None is a universal explanation-quality test.
- `evaluation.default_metric_registry` inventories every public `compute_*` endpoint with
  validated level, family, audit status, claim scope, score direction, stochasticity, and
  canonical-claim metadata. Its exact-inventory check runs at import and in CI.

### Interfaces and orchestration

- Model adapters have explicit task/output contracts; classification targets remain fixed
  across perturbations.
- The generic `Explanation` container validates structure and defensive-copy dictionary
  conversion without claiming that every payload is a feature-attribution vector or directly
  JSON serializable.
- Registry discovery uses compatibility metadata only. `ExplanationSuite` additionally checks
  local scope plus constructor/method call contracts, and requires an explicit shared
  comparison contract before displaying multiple outputs as comparable. Neither interface
  ranks scientific quality or identifies a best explainer.
- Generic plotting is intentionally unavailable until a real payload-aware backend exists.

### Documentation and secondary packages

- The competitive comparison is withdrawn.
- The LIME, KernelSHAP, TreeSHAP, and finite-estimator uncertainty/intervention-sensitivity
  tutorials are offline and lock-provenanced; the CI workflow is configured to re-execute them
  cleanly. Planned tutorial rows remain planning entries rather than capability claims. The
  uncertainty tutorial's t-interval is scoped to its supplied seeded streams, and its three
  empirical replacement references demonstrate intervention dependence rather than a universal
  baseline.
- The TypeScript package is private and experimental; it must pass its own build/test/lint and
  package-content gates, but is not claimed to implement the Python method inventory.

## Next work after the accuracy gate

The order below is intentional:

1. Keep full Python, declared-version, documentation, and package smoke gates green.
2. Keep each promoted tutorial offline and executable, and require a new dated execution record
   whenever its code, the package version, or the dependency lock changes.
3. Keep machine-readable explainer and metric inventories exact as endpoints evolve.
4. Upgrade the constrained counterfactual API only after a differentiable joint DiCE objective,
   supported-model contract, and independent reference oracle are available.
5. Consider other new methods only when their primary formula, supported domain, target
   semantics, and independent test oracle are available before implementation begins.

Feature quantity is not a release objective. Narrow, falsifiable claims are.
