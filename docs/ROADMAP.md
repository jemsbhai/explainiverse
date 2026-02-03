# Explainiverse Development Roadmap

## Project Overview

**Explainiverse** is a unified, extensible Python framework for Explainable AI (XAI) targeting publication at top-tier ML venues (NeurIPS/ICML).

- **Location:** `E:\data\code\claudecode\explainiverse`
- **Current Version:** v0.8.3
- **Package Manager:** Poetry
- **Python:** 3.10+

---

## Current Implementation Status

### Explainers (18 total)

| Category | Explainer | Status | Version Added |
|----------|-----------|--------|---------------|
| **Local - Perturbation** | LIME | ✅ Complete | v0.1.0 |
| | KernelSHAP | ✅ Complete | v0.1.0 |
| | TreeSHAP | ✅ Complete | v0.2.0 |
| **Local - Gradient** | Integrated Gradients | ✅ Complete | v0.3.0 |
| | DeepLIFT | ✅ Complete | v0.3.0 |
| | DeepSHAP | ✅ Complete | v0.3.0 |
| | GradCAM/GradCAM++ | ✅ Complete | v0.3.0 |
| | SmoothGrad | ✅ Complete | v0.5.0 |
| | Saliency Maps | ✅ Complete | v0.6.0 |
| **Local - Decomposition** | LRP | ✅ Complete | v0.8.0 |
| **Concept-Based** | TCAV | ✅ Complete | v0.7.0 |
| **Local - Rule-Based** | Anchors | ✅ Complete | v0.1.0 |
| **Local - Counterfactual** | DiCE-style | ✅ Complete | v0.2.0 |
| **Local - Example-Based** | ProtoDash | ✅ Complete | v0.4.0 |
| **Global** | Permutation Importance | ✅ Complete | v0.1.0 |
| | PDP | ✅ Complete | v0.1.0 |
| | ALE | ✅ Complete | v0.2.0 |
| | SAGE | ✅ Complete | v0.2.0 |

### Evaluation Metrics (12 total)

| Category | Metric | Status | Version | Reference |
|----------|--------|--------|---------|-----------|
| **Faithfulness (Core)** | PGI | ✅ Complete | v0.3.0 | - |
| | PGU | ✅ Complete | v0.3.0 | - |
| | Comprehensiveness | ✅ Complete | v0.3.0 | DeYoung et al., 2020 |
| | Sufficiency | ✅ Complete | v0.3.0 | DeYoung et al., 2020 |
| | Faithfulness Correlation | ✅ Complete | v0.3.0 | - |
| **Faithfulness (Extended)** | Faithfulness Estimate | ✅ Complete | v0.8.1 | Alvarez-Melis et al., 2018 |
| | Monotonicity | ✅ Complete | v0.8.2 | Arya et al., 2019 |
| | Monotonicity-Nguyen | ✅ Complete | v0.8.3 | Nguyen et al., 2020 |
| **Stability** | RIS | ✅ Complete | v0.3.0 | Agarwal et al., 2022 |
| | ROS | ✅ Complete | v0.3.0 | Agarwal et al., 2022 |
| | Lipschitz Estimate | ✅ Complete | v0.3.0 | Alvarez-Melis & Jaakkola, 2018 |

### Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| BaseExplainer | ✅ Complete | Abstract base class |
| Explanation | ✅ Complete | Unified output container |
| ExplainerRegistry | ✅ Complete | Plugin system with metadata |
| SklearnAdapter | ✅ Complete | scikit-learn wrapper |
| PyTorchAdapter | ✅ Complete | PyTorch wrapper with gradients |
| ExplanationSuite | ✅ Complete | Multi-explainer comparison |

---

## Strategic Goal: Beat State-of-the-Art Evaluation Metrics

We are expanding evaluation metrics to definitively exceed all existing XAI frameworks:

| Framework | Current Metrics | Notes |
|-----------|----------------|-------|
| **Quantus** | 37 | Current SOTA for metrics |
| **OpenXAI** | 22 | Academic benchmark |
| **Explainiverse** | 12 → **54** | Target after Phase 1-7 |

### Master Metrics Implementation Plan (7 Phases)

| Phase | Version | Category | New Metrics | Running Total |
|-------|---------|----------|-------------|---------------|
| 1 | v0.9.0 | Faithfulness | +9 | 21 |
| 2 | v0.10.0 | Robustness | +7 | 28 |
| 3 | v0.11.0 | Localisation | +8 | 36 |
| 4 | v0.12.0 | Complexity | +4 | 40 |
| 5 | v0.13.0 | Randomisation | +5 | 45 |
| 6 | v0.14.0 | Axiomatic | +4 | 49 |
| 7 | v0.15.0 | Fairness | +4 | 53 |

---

## Phase 1: Faithfulness Metrics (In Progress)

**Target:** v0.9.0 with 12 additional faithfulness metrics

| # | Metric | Reference | Status | Version |
|---|--------|-----------|--------|---------|
| 1 | Faithfulness Estimate | Alvarez-Melis et al., 2018 | ✅ Complete | v0.8.1 |
| 2 | Monotonicity | Arya et al., 2019 | ✅ Complete | v0.8.2 |
| 3 | Monotonicity-Nguyen | Nguyen et al., 2020 | ✅ Complete | v0.8.3 |
| 4 | Pixel Flipping | Bach et al., 2015 | ⏳ Next | - |
| 5 | Region Perturbation | Samek et al., 2015 | ❌ Not started | - |
| 6 | Selectivity | Montavon et al., 2018 | ❌ Not started | - |
| 7 | Sensitivity-n | Ancona et al., 2018 | ❌ Not started | - |
| 8 | IROF | Rieger & Hansen, 2020 | ❌ Not started | - |
| 9 | Infidelity | Yeh et al., 2019 | ❌ Not started | - |
| 10 | ROAD | Rong et al., 2022 | ❌ Not started | - |
| 11 | Insertion AUC | Petsiuk et al., 2018 | ❌ Not started | - |
| 12 | Deletion AUC | Petsiuk et al., 2018 | ❌ Not started | - |

---

## Future Phases (Planned)

### Phase 2: Robustness Metrics (v0.10.0)
- Local Lipschitz Estimate
- Max-Sensitivity
- Avg-Sensitivity
- Continuity
- Consistency
- Relative Input/Output Stability extensions

### Phase 3: Localisation Metrics (v0.11.0)
- Pointing Game
- Top-K Intersection
- Relevance Mass Accuracy
- Relevance Rank Accuracy
- Attribution Localisation
- Focus
- AUC (localisation)
- Bounding Box metrics

### Phase 4: Complexity Metrics (v0.12.0)
- Sparseness
- Complexity
- Effective Complexity
- Entropy

### Phase 5: Randomisation Metrics (v0.13.0)
- Model Parameter Randomisation
- Random Logit Test
- Data Randomisation
- MPRT variants

### Phase 6: Axiomatic Metrics (v0.14.0)
- Completeness
- Non-Sensitivity
- Input Invariance
- Symmetry

### Phase 7: Fairness Metrics (v0.15.0)
- Group Fairness
- Individual Fairness
- Counterfactual Fairness
- Disparity metrics

---

## Development Workflow

### Standard Workflow (Per Feature)

1. **Implement** - Write implementation following existing patterns
2. **Test** - Comprehensive test suite (35-50+ tests per metric/explainer)
3. **Document** - Update ROADMAP.md and SOTA_COMPARISON.md
4. **Bump Version** - Update pyproject.toml and __init__.py
5. **Push** - Commit to GitHub with detailed message
6. **Publish** - `poetry publish --build` to PyPI

### Commands

```powershell
# Navigate to project
cd E:\data\code\claudecode\explainiverse

# Run specific tests
python -m pytest tests/test_<name>.py -v --tb=short

# Run all tests
python -m pytest tests/ -v

# Bump version (in pyproject.toml and src/explainiverse/__init__.py)
# Then commit and push
git add -A
git commit -m "v0.X.Y: Description of changes"
git push origin main

# Publish to PyPI
poetry publish --build
```

### Version Bump Locations

1. `pyproject.toml` → `version = "X.Y.Z"`
2. `src/explainiverse/__init__.py` → `__version__ = "X.Y.Z"`

---

## Architecture

```
explainiverse/
├── core/
│   ├── explainer.py          # BaseExplainer abstract class
│   ├── explanation.py        # Unified Explanation container
│   └── registry.py           # ExplainerRegistry with metadata
├── adapters/
│   ├── sklearn_adapter.py    # scikit-learn models
│   └── pytorch_adapter.py    # PyTorch with gradient support
├── explainers/
│   ├── attribution/          # LIME, SHAP, TreeSHAP
│   ├── gradient/             # IG, DeepLIFT, DeepSHAP, SmoothGrad, Saliency, GradCAM, LRP, TCAV
│   ├── rule_based/           # Anchors
│   ├── counterfactual/       # DiCE-style
│   ├── global_explainers/    # Permutation, PDP, ALE, SAGE
│   └── example_based/        # ProtoDash
├── evaluation/
│   ├── faithfulness.py       # Core faithfulness metrics
│   ├── faithfulness_extended.py  # Extended faithfulness (Phase 1)
│   ├── stability.py          # Stability metrics
│   ├── metrics.py            # AOPC, ROAR
│   └── _utils.py             # Shared utilities
└── engine/
    └── suite.py              # Multi-explainer comparison
```

---

## Testing Standards

- **Minimum 35-50 tests** per metric/explainer
- **Test categories:**
  - Basic functionality (creation, return types, valid ranges)
  - Different baseline types (mean, median, scalar, array, callable)
  - Batch operations
  - Multiple model types (LogisticRegression, RandomForest, GradientBoosting)
  - Multiple explainers (LIME vs SHAP comparison)
  - Edge cases (few features, many features, zero/identical attributions)
  - Semantic validation (good explanations score better than random)
  - Target class handling
- **All tests must pass before pushing**

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1.0 | - | Initial release: LIME, SHAP, Anchors, PI, PDP |
| v0.2.0 | - | TreeSHAP, Counterfactual, ALE, SAGE |
| v0.3.0 | - | Evaluation metrics, IG, DeepLIFT, DeepSHAP, GradCAM |
| v0.4.0 | - | ProtoDash |
| v0.5.0 | Jan 2025 | SmoothGrad |
| v0.6.0 | Jan 2025 | Saliency Maps |
| v0.7.0 | Jan 2025 | TCAV (Concept-Based Explanations) |
| v0.8.0 | Jan 2025 | LRP (Layer-wise Relevance Propagation) |
| v0.8.1 | Feb 2025 | Faithfulness Estimate metric |
| v0.8.2 | Feb 2025 | Monotonicity metric (Arya et al., 2019) |
| v0.8.3 | Feb 2025 | Monotonicity-Nguyen metric (Nguyen et al., 2020) |

---

## Key Files for Phase 1 Development

| Purpose | File |
|---------|------|
| Implementation | `src/explainiverse/evaluation/faithfulness_extended.py` |
| Exports | `src/explainiverse/evaluation/__init__.py` |
| Utilities | `src/explainiverse/evaluation/_utils.py` |
| Tests | `tests/test_<metric_name>.py` |

---

## References

### Core Methods
- LIME: Ribeiro et al., 2016 - "Why Should I Trust You?"
- SHAP: Lundberg & Lee, 2017 - "A Unified Approach to Interpreting Model Predictions"
- Integrated Gradients: Sundararajan et al., 2017 - "Axiomatic Attribution for Deep Networks"
- TCAV: Kim et al., 2018 - "Interpretability Beyond Feature Attribution"
- LRP: Bach et al., 2015 - "On Pixel-wise Explanations for Non-Linear Classifier Decisions"

### Evaluation Metrics (Phase 1)
- Faithfulness Estimate: Alvarez-Melis & Jaakkola, 2018 - "Towards Robust Interpretability"
- Monotonicity: Arya et al., 2019 - "One Explanation Does Not Fit All"
- Monotonicity-Nguyen: Nguyen & Martinez, 2020 - "Quantitative Evaluation of ML Explanations"
- Pixel Flipping: Bach et al., 2015
- IROF: Rieger & Hansen, 2020
- Infidelity: Yeh et al., 2019
- ROAD: Rong et al., 2022
- Insertion/Deletion: Petsiuk et al., 2018

---

*Last updated: February 2025 (v0.8.3)*
