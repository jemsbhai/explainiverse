# Explainiverse

Explainiverse is a beta Python library for constructing and evaluating machine-learning
explanations. Its public registry records what each explainer has actually been audited to do;
it does not rank methods by quality or claim that one method is appropriate for a particular
deployment.

The latest version published on PyPI is `0.14.0`. Its distributions were uploaded with Twine,
without Trusted Publishing or published provenance. The matching annotated Git tag is unsigned
and no GitHub Release exists, so the cross-service release record is incomplete. This checkout
is the `0.15.0` stable-release candidate and still declares Python 3.10 through
3.13. It has not been tagged or published.

## Accuracy status

Registry entries use three statuses:

- `verified`: the implementation and its tests support the specific `claim_scope` recorded in
  the registry.
- `quarantined`: the code remains available for compatibility, but must not be represented as
  the canonical named method.
- `unverified`: no formula-level accuracy claim is made yet.

`verified` is deliberately narrow. It does not mean that an explanation is causal, useful,
fair, stable, or faithful for a particular model and dataset. Those conclusions require
task-specific evidence.

Use the runtime metadata as the authoritative explainer inventory:

```python
from explainiverse import default_registry

print(default_registry.summary())
print(default_registry.get_meta("lime").claim_scope)
```

Evaluation endpoints have an equivalent authoritative inventory:

```python
import explainiverse.evaluation as evaluation

metrics = evaluation.default_metric_registry
metrics.validate_inventory(evaluation.__all__)
print(metrics.summary())
print(metrics.get_meta("compute_aopc").claim_scope)
```

The metric registry distinguishes formula-verified endpoints, quarantined compatibility
aliases, explicit adaptations, score direction, stochastic estimators, and instance/batch/
dataset scope. A `verified` endpoint remains limited to its callable documentation and
`claim_scope`; it is not automatically a canonical reproduction of an entire paper protocol.

The audited registry currently groups as follows:

| Status | Registry keys | Boundary |
|---|---|---|
| Verified tabular/local | `lime`, `shap`, `treeshap`, `anchor_tabular`, `protodash` | `shap` is the KernelSHAP wrapper; TreeSHAP is limited to declared tree/output contracts; `anchor_tabular` certifies only rules whose sequential KL lower bound strictly exceeds the requested threshold under the uniform empirical joint distribution of its background rows |
| Verified gradient/local | `integrated_gradients`, `deeplift`, `deepshap`, `smoothgrad`, `saliency`, `lrp` | CPU-verified only. DeepLIFT, DeepSHAP, SmoothGrad, and Saliency are flat tabular-vector APIs; Integrated Gradients and LRP also declare narrower image contracts |
| Verified concept/global | `tcav` | CPU-verified only. Dataset-level fraction of positive directional derivatives for one target-class input set; canonical TCAV requires declared class-logit scores |
| Verified CAM | `gradcam`, `hirescam`, `xgradcam`, `layercam`, `eigencam`, `scorecam`, `ablationcam` | CPU-verified only. One compatible spatial layer and the exact target/output restrictions in each `claim_scope`; `scorecam` is specifically the paper Algorithm-1 raw-output/channel-softmax variant, not the paper's later probability-weighting convention |
| Verified global | `permutation_importance`, `partial_dependence`, `ale`, `sage` | Tabular contracts; ALE implements continuous first-order ALE, not nominal ALE |
| Quarantined compatibility APIs | `anchors`, `counterfactual`, `eigengradcam`, `gradcam_elementwise` | Fixed-sample Anchors-style search, constrained counterfactual search, and two library-defined CAM variants |

Grad-CAM++ is not exposed. The previous implementation used only first derivatives and could
not establish the general higher-derivative formula. The verified `anchor_tabular` key is the
confidence-certified continuous-numeric implementation; the historical `anchors` key remains
the fixed-sample compatibility heuristic. The historical `counterfactual` key does not claim
the DiCE optimisation algorithm.

## Evaluation scope

The evaluation namespace contains canonical formulas, explicitly named adaptations, and
library-defined diagnostics. It is an API inventory, not a count of distinct scientific
metrics and not a quality leaderboard.

| Family | Audited interpretation |
|---|---|
| Core faithfulness | Deterministic baseline-replacement PGI/PGU specialisations; tabular ERASER adaptations; Bhatt fixed-size subset correlation |
| Extended faithfulness | Insertion/deletion, sensitivity-n, infidelity, IROF, selectivity/AOPC, region perturbation, pixel flipping, ROAD, and monotonicity APIs, each under its documented perturbation and target contract |
| Robustness/stability | Sampled Yeh Max-Sensitivity; Agarwal RIS/RRS/ROS equations; Dasgupta consistency; finite-sample local diagnostics. Avg-Sensitivity is a noncanonical mean heuristic |
| Localisation | Pre-pooled scalar attribution maps with exact mask/shape contracts. Attribution IoU is library-defined |
| Randomisation | MPRT-family, random-logit, and data-randomisation sensitivity diagnostics; their values are not explanation-quality verdicts |
| Axiomatic | Bounded completeness, Nguyen Non-Sensitivity, compensated translation, and conditional symmetry checks; no global axiom proof is inferred |
| Agreement/complexity | Ranking overlap/correlation and attribution-distribution statistics; no human-interpretability conclusion is inferred |
| Fairness-related audits | Group and sensitive-feature disparity diagnostics plus exact fidelity-gap formulas; diagnostic parity is not a fairness certificate |

Undefined quantities, incompatible targets, incomplete feature mappings, unsupported model
outputs, and non-finite inputs are intended to fail explicitly rather than receive fabricated
fallback scores.

## Installation from a checkout

Base dependencies, including the image utilities required by the mandatory
LIME backend:

```bash
python -m pip install -e .
```

PyTorch/Captum extras:

```bash
python -m pip install -e ".[all]"
```

The historical `image` extra was redundant because LIME already installed
scikit-image. Scikit-image is now an explicit base dependency and the no-op
extra has been removed. Consumers that previously installed `.[image]` should
install the base package instead; `.[all]` continues to add Torch and Captum.

For development, synchronize the locked all-extras environment:

```bash
poetry sync --all-extras --with dev,tutorial --no-interaction
```

## Minimal tabular example

This example uses the verified LIME wrapper scope. Its returned coefficients describe the
fitted local surrogate; they are not causal effects.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from explainiverse import SklearnAdapter, default_registry

iris = load_iris()
feature_names = iris.feature_names
class_names = iris.target_names

model = RandomForestClassifier(n_estimators=20, random_state=0)
model.fit(iris.data, iris.target)

adapter = SklearnAdapter(
    model,
    feature_names=feature_names,
    class_names=class_names,
    task="classification",
)
explainer = default_registry.create(
    "lime",
    model=adapter,
    training_data=iris.data,
    feature_names=feature_names,
    class_names=class_names,
    random_state=0,
)

explanation = explainer.explain(iris.data[0])
print(explanation.target_class)
print(explanation.get_top_features(k=4))
```

`default_registry.filter(...)` performs metadata matching. The historical
`default_registry.recommend(...)` name only orders metadata-compatible entries; it does not
recommend the best method or predict suitability, accuracy, runtime, or explanation quality.

`BaseExplainer.explain` intentionally has a generic abstract signature because local, global,
dataset-level, and feature-oriented explainers do not share one honest input contract. Consult
the concrete class and registry scope before use.

`ExplanationSuite.run(...)` executes only local explainers, checks both constructor and method
arguments, and leaves exact array shape to each concrete contract. Required method arguments
such as ProtoDash reference data must be supplied through `explainer_call_kwargs` or
`call_kwargs_by_explainer`. `ExplanationSuite.compare()` requires the same ordered feature
identity, explained target, and explicit caller-asserted `metadata["comparison_contract"]`
across multiple outputs. Built-in explainers do not currently emit that contract. The
`allow_incommensurate=True` escape hatch is a warned descriptive display, not a mathematical
comparison.

`Explanation.to_dict()` returns a defensive-copy dictionary. Payload values keep their
original Python or NumPy types, so the result is not promised to be directly JSON serializable.

## Tutorials and other packages

The LIME, KernelSHAP, TreeSHAP, and finite-estimator uncertainty/intervention-sensitivity
notebooks under `tutorials/` are deterministic, offline teaching artifacts verified against
this checkout. Each published notebook contains a dated execution record, package version,
Python/platform record, and canonicalized `poetry.lock` digest. The repository harness
statically rejects common package-install/network access paths, adds a non-loopback Python
socket guard, executes a clean in-memory copy, and fails on stale source, output, runner,
package-tree, or lock provenance:

```bash
poetry run python scripts/execute_tutorials.py
```

The guard prevents accidental network dependence in the reviewed notebooks; it is not a
security sandbox for hostile notebook code.

These examples verify their stated API and numerical contracts; they are not research
benchmarks or evidence that an explanation is causal or suitable for deployment.

The TypeScript package under `packages/js/` is private and experimental. It is not a published
JavaScript equivalent of the Python implementation.

## Verification and contribution gate

Before adding a method or metric:

1. identify the primary formula and its assumptions;
2. declare target, score-space, perturbation, shape, and unsupported-domain contracts;
3. add analytical or official-reference tests and invariants;
4. mark adaptations and heuristics in the public name, metadata, and result payload;
5. run formatting, lint, type, test, build, and package-smoke gates appropriate to the change.

The reproducible checkout gate is:

```bash
poetry sync --all-extras --with dev,tutorial --no-interaction
poetry check --lock
poetry run python -m pip check
poetry run black --check src tests scripts
poetry run isort --check-only src tests scripts
poetry run mypy src scripts
poetry run pytest --strict-config --strict-markers --cov=explainiverse --cov-branch
poetry run python scripts/execute_tutorials.py
poetry build
```

CPU CI permits the four explicitly allowlisted CUDA skips plus the conditional Python 3.10 /
XGBoost-before-3.1 vector-intercept skip; any other skipped test fails the corresponding pytest
job. CUDA execution is outside the verified device scope until a GPU runner is part of the
release gate. Reference packages are imported explicitly before tests, and the built wheel and
source distribution are each exercised in isolated consumer environments.

See the [0.15 migration notes](docs/MIGRATION_0_15.md), the
[residual-limitations mitigation plan](docs/LIMITATION_MITIGATION_PLAN.md), and the
[accuracy roadmap](docs/ROADMAP.md) for current contracts and priorities. The former
competitive comparison was withdrawn in [SOTA_COMPARISON.md](docs/SOTA_COMPARISON.md) because
raw API counts and cross-library marketing claims were not scientifically comparable.

## License and citation

The Python distribution declares the MIT license; see `LICENSE`. The separate private
`packages/js` workspace declares `UNLICENSED` in its own package metadata and is not a
published JavaScript distribution.

If you use a particular explainer or metric, cite its primary source as well as the exact
software revision and configuration you ran. For release `0.14.0`, cite the `v0.14.0` tag (or its
commit hash) and your configuration. PyPI hosts `0.14.0` artifacts, but there is no corresponding
GitHub Release or Trusted-Publishing provenance record.
