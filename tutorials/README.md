# Explainiverse tutorials

The first three tutorials are verified, deterministic, offline examples for the
untagged `0.14.0` checkout. They were reviewed and published from the locked environment on
2026-08-10. The remaining entries on this page are curriculum plans, not implemented
capabilities.

## Reproduce the verified notebooks

From the repository root:

```bash
poetry sync --all-extras --with dev,tutorial --no-interaction
poetry run python scripts/execute_tutorials.py
```

The second command statically rejects common package-install and network-access paths, adds a
non-loopback Python socket guard, validates hashes for the notebook source, published outputs,
runner, package source tree, and canonicalized `poetry.lock`, then executes every cell from a
clean in-memory copy with errors disabled. This is an accidental-dependency guard for reviewed
notebooks, not a hostile-code security sandbox. CI runs the same command. Maintainers can
republish reviewed outputs and metadata with:

```bash
poetry run python scripts/execute_tutorials.py --write
```

The tutorial dependency group provides the execution kernel and notebook libraries. To work
interactively, select the repository's Poetry environment as the kernel in an existing
Jupyter or editor frontend; a frontend is intentionally not installed by the tutorial group.

## Verified tutorials

| # | Notebook | Model and local dataset | Contract exercised | Status |
|---|---|---|---|---|
| 01 | [LIME for tabular classification](01_lime_tabular.ipynb) | Seeded random forest; sklearn Iris | One explicit probability output, local-surrogate coefficients, repeatability, and scoped PGI/PGU perturbation diagnostics | Verified offline |
| 02 | [KernelSHAP](02_kernelshap.ipynb) | Seeded logistic pipeline; sklearn Iris | Empirical background, one explicit probability output, local additivity, and repeatability | Verified offline |
| 03 | [TreeSHAP](03_treeshap.ipynb) | Seeded random forest; sklearn Breast Cancer | Strict single-instance and batch APIs, path-dependent and interventional games, and game-specific reconstruction residuals | Verified offline |

Each notebook records its execution timestamp, Explainiverse version, Python version,
platform, source/output/package/runner/lock digests, offline guard, and deterministic seed in
`metadata.explainiverse_execution`. The examples use datasets bundled with scikit-learn, use
CPU-only deterministic estimators, and make no external requests.

## Interpretation boundaries

Successful execution establishes the stated API, shape, output-target, and numerical checks
for the recorded environment. It does not establish that:

- an explanation is causal, uniquely correct, or useful for a particular decision;
- local attributions generalize to a population;
- two methods answer the same feature-absence question;
- PGI, PGU, additivity, or repeatability is a universal quality score; or
- floating-point output is byte-identical across Python, dependency, OS, or hardware changes.

The notebooks deliberately expose background data, output-column selection, perturbation
game, and numerical tolerance where those choices affect the result.

## Planned curriculum

The following rows are plans only. Names do not imply an implementation, canonical claim,
runtime commitment, or release date.

### Explainers

| # | Proposed topic | Required scope before promotion | Status |
|---|---|---|---|
| 04 | Confidence-certified tabular Anchors | Verified `anchor_tabular` KL-LUCB scope, strict certificate, whole-row empirical conditioning, budget exhaustion, and contrast with quarantined fixed-sample `anchors` | Planned |
| 05 | Counterfactual search | Project constrained-search heuristic, explicitly not the DiCE algorithm | Planned |
| 06 | Integrated Gradients | Baseline, target score, convergence, and CPU device scope | Planned |
| 07 | CAM methods | Method-specific score space, layer, layout, and image contract | Planned |
| 08 | DeepLIFT and DeepSHAP | Supported graph subset, baseline distribution, and target score | Planned |
| 09 | Permutation importance | Held-out scoring and feature-dependence limitations | Planned |
| 10 | PDP and ALE | Marginal/conditional assumptions and continuous-feature scope | Planned |
| 11 | SAGE | Marginal imputer, loss, sampling estimator, and uncertainty | Planned |
| 12 | ProtoDash | Prototype objective, kernel, weights, and criticism limits | Planned |

### Evaluation diagnostics

| # | Proposed topic | Scope | Status |
|---|---|---|---|
| 13 | Perturbation and faithfulness | Targets, baselines, interventions, and estimands | Planned |
| 14 | Robustness | Sampled sensitivity and finite local estimates | Planned |
| 15 | Localisation | Mask, sign, threshold, and image-layout contracts | Planned |
| 16 | Complexity | Sparseness and explicitly named complexity diagnostics | Planned |
| 17 | Randomisation | MPRT, Smooth MPRT, eMPRT, random-logit, and data tests | Planned |
| 18 | Axiomatic checks | Completeness and explicitly limited proxy tests | Planned |
| 19 | Fairness-related diagnostics | Group/change diagnostics without an automatic fairness verdict | Planned |

### Capstones

| # | Proposed topic | Scope | Status |
|---|---|---|---|
| 20 | Comparing explainers | Compare only explicitly commensurate targets and semantics | Planned |
| 21 | End-to-end XAI workflow | Train, explain, diagnose, and report limitations | Planned |
| 22 | Model debugging | Negative controls and spurious-correlation investigation | Planned |
| 23 | Healthcare case study | Decision-support boundaries and domain review | Planned |
| 24 | Fairness-related audit | Scoped diagnostics without automated bias claims | Planned |
| 25 | Benchmark design | Prespecified methods, datasets, estimands, and failure rules | Planned |

## Contribution requirements

A tutorial is promoted only when it:

1. uses a local real dataset and contains no install or network cells;
2. identifies the method formula, target/output space, baselines, and unsupported domains;
3. uses fixed seeds and bounded CPU work;
4. includes assertions for its important numerical and API contracts;
5. avoids universal explanation-quality or causal claims;
6. passes `scripts/execute_tutorials.py --write` from the locked environment; and
7. is added to the default runner and CI before being marked verified.

These tutorials are released under the repository's [MIT License](../LICENSE).
