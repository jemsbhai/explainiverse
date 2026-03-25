# Explainiverse Development Roadmap

## Project Overview

**Explainiverse** is a unified, extensible Python framework for Explainable AI (XAI) designed for rigorous research and production use.

- **Current Version:** v0.12.0
- **Repository:** [github.com/jemsbhai/explainiverse](https://github.com/jemsbhai/explainiverse)
- **PyPI:** [pypi.org/project/explainiverse](https://pypi.org/project/explainiverse/)
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

### Evaluation Metrics (55 total)

| Category | Metric | Status | Version | Reference |
|----------|--------|--------|---------|-----------|
| **Faithfulness (Core)** | PGI | ✅ Complete | v0.3.0 | — |
| | PGU | ✅ Complete | v0.3.0 | — |
| | Comprehensiveness | ✅ Complete | v0.3.0 | DeYoung et al., 2020 |
| | Sufficiency | ✅ Complete | v0.3.0 | DeYoung et al., 2020 |
| | Faithfulness Correlation | ✅ Complete | v0.3.0 | — |
| **Faithfulness (Extended)** | Faithfulness Estimate | ✅ Complete | v0.8.1 | Alvarez-Melis et al., 2018 |
| | Monotonicity | ✅ Complete | v0.8.2 | Arya et al., 2019 |
| | Monotonicity-Nguyen | ✅ Complete | v0.8.3 | Nguyen et al., 2020 |
| | Pixel Flipping | ✅ Complete | v0.8.4 | Bach et al., 2015 |
| | Region Perturbation | ✅ Complete | v0.8.5 | Samek et al., 2015 |
| | Selectivity (AOPC) | ✅ Complete | v0.8.6 | Montavon et al., 2018 |
| | Sensitivity-n | ✅ Complete | v0.8.7 | Ancona et al., 2018 |
| | IROF | ✅ Complete | v0.8.9 | Rieger & Hansen, 2020 |
| | Infidelity | ✅ Complete | v0.8.10 | Yeh et al., 2019 |
| | ROAD | ✅ Complete | v0.8.11 | Rong et al., 2022 |
| | Insertion AUC | ✅ Complete | v0.9.1 | Petsiuk et al., 2018 |
| | Deletion AUC | ✅ Complete | v0.9.1 | Petsiuk et al., 2018 |
| **Stability (legacy)** | RIS (simple) | ✅ Complete | v0.3.0 | Agarwal et al., 2022 (simplified) |
| | ROS (simple) | ✅ Complete | v0.3.0 | Agarwal et al., 2022 (simplified) |
| | Lipschitz Estimate | ✅ Complete | v0.3.0 | Alvarez-Melis & Jaakkola, 2018 |
| **Robustness** | Max-Sensitivity | ✅ Complete | v0.9.4 | Yeh et al., 2019 |
| | Avg-Sensitivity | ✅ Complete | v0.9.4 | Yeh et al., 2019 |
| | Continuity | ✅ Complete | v0.9.4 | Montavon et al., 2018 |
| | Consistency | ✅ Complete | v0.9.6 | Dasgupta et al., 2022 |
| | Relative Input Stability (RIS) | ✅ Complete | v0.9.6 | Agarwal et al., 2022 (Eq 2) |
| | Relative Representation Stability (RRS) | ✅ Complete | v0.9.6 | Agarwal et al., 2022 (Eq 3) |
| | Relative Output Stability (ROS) | ✅ Complete | v0.9.6 | Agarwal et al., 2022 (Eq 5) |
| **Agreement** | Feature Agreement | ✅ Complete | v0.9.7 | Krishna et al., 2022 |
| | Rank Agreement | ✅ Complete | v0.9.7 | Krishna et al., 2022 |
| **Complexity** | Sparseness | ✅ Complete | v0.9.5 | Chalasani et al., 2020 |
| | Complexity | ✅ Complete | v0.9.5 | Bhatt et al., 2020 |
| | Effective Complexity | ✅ Complete | v0.9.5 | Nguyen & Martinez, 2020 |
| **Localisation** | Pointing Game | ✅ Complete | v0.9.8 | Zhang et al., 2018 |
| | Attribution Localisation | ✅ Complete | v0.9.8 | Kohlbrenner et al., 2020 |
| | Top-K Intersection | ✅ Complete | v0.9.8 | Theiner et al., 2021 |
| | Relevance Mass Accuracy | ✅ Complete | v0.9.8 | Arras et al., 2022 |
| | Relevance Rank Accuracy | ✅ Complete | v0.9.8 | Arras et al., 2022 |
| | AUC (localisation) | ✅ Complete | v0.9.8 | Fawcett, 2006 |
| | Energy-Based Pointing Game | ✅ Complete | v0.9.8 | Wang et al., 2020 |
| | Focus | ✅ Complete | v0.9.8 | Arias-Duart et al., 2022 |
| | Attribution IoU | ✅ Complete | v0.9.8 | — |
| **Randomisation** | MPRT | ✅ Complete | v0.10.0 | Adebayo et al., 2018 |
| | Random Logit Test | ✅ Complete | v0.10.0 | Sixt et al., 2020 |
| | Smooth MPRT | ✅ Complete | v0.10.0 | Hedstrom et al., 2023 |
| | Efficient MPRT | ✅ Complete | v0.10.0 | Hedstrom et al., 2023 |
| | Data Randomisation Test | ✅ Complete | v0.10.0 | Adebayo et al., 2018 |
| **Axiomatic** | Completeness | ✅ Complete | v0.11.0 | Sundararajan et al., 2017 |
| | Non-Sensitivity | ✅ Complete | v0.11.0 | Nguyen & Martinez, 2020 |
| | Input Invariance | ✅ Complete | v0.11.0 | Kindermans et al., 2017 |
| | Symmetry | ✅ Complete | v0.11.0 | Sundararajan et al., 2017 |
| **Fairness** | Group Fairness | ✅ Complete | v0.12.0 | Dai et al., 2022 |
| | Individual Fairness | ✅ Complete | v0.12.0 | Dwork et al., 2012 |
| | Counterfactual Explanation Fairness | ✅ Complete | v0.12.0 | Kusner et al., 2017 |
| | Fidelity Disparity | ✅ Complete | v0.12.0 | Balagopalan et al., 2022 |
| | Attribution Parity | ✅ Complete | v0.12.0 | Avodji et al., 2019 |
| | Conditional Fairness | ✅ Complete | v0.12.0 | Hardt et al., 2016 |

### Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| BaseExplainer | ✅ Complete | Abstract base class |
| Explanation | ✅ Complete | Unified output container |
| ExplainerRegistry | ✅ Complete | Plugin system with metadata |
| FairnessMetricRegistry | ✅ Complete | Extensible fairness metric registration |
| SklearnAdapter | ✅ Complete | scikit-learn wrapper |
| PyTorchAdapter | ✅ Complete | PyTorch wrapper with gradients |
| ExplanationSuite | ✅ Complete | Multi-explainer comparison |

---

## Strategic Goal: Comprehensive Evaluation Metrics

Explainiverse provides the most comprehensive evaluation metrics suite among XAI frameworks:

| Framework | Metrics | Notes |
|-----------|---------|-------|
| **Explainiverse** | **55** | All 7 phases complete ✅ |
| **Quantus** | 37 | Previous SOTA |
| **OpenXAI** | 22 | Academic benchmark |

**Explainiverse is 49% ahead of Quantus.**

### Master Metrics Implementation Plan (7 Phases) — All Complete ✅

| Phase | Version | Category | New Metrics | Running Total | Status |
|-------|---------|----------|-------------|---------------|--------|
| 1 | v0.8.x–v0.9.1 | Faithfulness | +12 | 22 | ✅ Complete |
| 2 | v0.9.4–v0.9.7 | Robustness & Agreement | +9 | 31 | ✅ Complete |
| 3 | v0.9.8 | Localisation | +9 | 40 | ✅ Complete |
| 4 | v0.9.5 | Complexity | +3 | (included above) | ✅ Complete |
| 5 | v0.10.0 | Randomisation | +5 | 45 | ✅ Complete |
| 6 | v0.11.0 | Axiomatic | +4 | 49 | ✅ Complete |
| 7 | v0.12.0 | Fairness | +6 | 55 | ✅ Complete |

---

## Phase 1: Faithfulness Metrics (✅ Complete)

| # | Metric | Reference | Version |
|---|--------|-----------|---------|
| 1 | Faithfulness Estimate | Alvarez-Melis et al., 2018 | v0.8.1 |
| 2 | Monotonicity | Arya et al., 2019 | v0.8.2 |
| 3 | Monotonicity-Nguyen | Nguyen et al., 2020 | v0.8.3 |
| 4 | Pixel Flipping | Bach et al., 2015 | v0.8.4 |
| 5 | Region Perturbation | Samek et al., 2015 | v0.8.5 |
| 6 | Selectivity (AOPC) | Montavon et al., 2018 | v0.8.6 |
| 7 | Sensitivity-n | Ancona et al., 2018 | v0.8.7 |
| 8 | IROF | Rieger & Hansen, 2020 | v0.8.9 |
| 9 | Infidelity | Yeh et al., 2019 | v0.8.10 |
| 10 | ROAD | Rong et al., 2022 | v0.8.11 |
| 11 | Insertion AUC | Petsiuk et al., 2018 | v0.9.1 |
| 12 | Deletion AUC | Petsiuk et al., 2018 | v0.9.1 |

---

## Phase 2: Robustness & Agreement Metrics (✅ Complete)

| # | Metric | Reference | Version |
|---|--------|-----------|---------|
| 1 | Max-Sensitivity | Yeh et al., 2019 | v0.9.4 |
| 2 | Avg-Sensitivity | Yeh et al., 2019 | v0.9.4 |
| 3 | Continuity | Montavon et al., 2018 | v0.9.4 |
| 4 | Consistency | Dasgupta et al., 2022 | v0.9.6 |
| 5 | Relative Input Stability (RIS) | Agarwal et al., 2022 (Eq 2) | v0.9.6 |
| 6 | Relative Representation Stability (RRS) | Agarwal et al., 2022 (Eq 3) | v0.9.6 |
| 7 | Relative Output Stability (ROS) | Agarwal et al., 2022 (Eq 5) | v0.9.6 |
| 8 | Feature Agreement | Krishna et al., 2022 | v0.9.7 |
| 9 | Rank Agreement | Krishna et al., 2022 | v0.9.7 |

---

## Phase 3: Localisation Metrics (✅ Complete)

| # | Metric | Reference | Version |
|---|--------|-----------|---------|
| 1 | Pointing Game | Zhang et al., 2018 | v0.9.8 |
| 2 | Attribution Localisation | Kohlbrenner et al., 2020 | v0.9.8 |
| 3 | Top-K Intersection | Theiner et al., 2021 | v0.9.8 |
| 4 | Relevance Mass Accuracy | Arras et al., 2022 | v0.9.8 |
| 5 | Relevance Rank Accuracy | Arras et al., 2022 | v0.9.8 |
| 6 | AUC (localisation) | Fawcett, 2006 | v0.9.8 |
| 7 | Energy-Based Pointing Game | Wang et al., 2020 | v0.9.8 |
| 8 | Focus | Arias-Duart et al., 2022 | v0.9.8 |
| 9 | Attribution IoU | — | v0.9.8 |

---

## Phase 4: Complexity Metrics (✅ Complete)

| # | Metric | Reference | Version |
|---|--------|-----------|---------|
| 1 | Sparseness | Chalasani et al., 2020 | v0.9.5 |
| 2 | Complexity | Bhatt et al., 2020 | v0.9.5 |
| 3 | Effective Complexity | Nguyen & Martinez, 2020 | v0.9.5 |

---

## Phase 5: Randomisation Metrics (✅ Complete)

| # | Metric | Reference | Version |
|---|--------|-----------|---------|
| 1 | MPRT | Adebayo et al., 2018 | v0.10.0 |
| 2 | Random Logit Test | Sixt et al., 2020 | v0.10.0 |
| 3 | Smooth MPRT | Hedstrom et al., 2023 | v0.10.0 |
| 4 | Efficient MPRT | Hedstrom et al., 2023 | v0.10.0 |
| 5 | Data Randomisation Test | Adebayo et al., 2018 | v0.10.0 |

---

## Phase 6: Axiomatic Metrics (✅ Complete)

| # | Metric | Reference | Version |
|---|--------|-----------|---------|
| 1 | Completeness | Sundararajan et al., 2017 | v0.11.0 |
| 2 | Non-Sensitivity | Nguyen & Martinez, 2020 | v0.11.0 |
| 3 | Input Invariance | Kindermans et al., 2017 | v0.11.0 |
| 4 | Symmetry | Sundararajan et al., 2017 | v0.11.0 |

---

## Phase 7: Fairness Metrics (✅ Complete)

| # | Metric | Reference | Version |
|---|--------|-----------|---------|
| 1 | Group Fairness | Dai et al., 2022 | v0.12.0 |
| 2 | Individual Fairness | Dwork et al., 2012 | v0.12.0 |
| 3 | Counterfactual Explanation Fairness | Kusner et al., 2017 | v0.12.0 |
| 4 | Fidelity Disparity | Balagopalan et al., 2022 | v0.12.0 |
| 5 | Attribution Parity | Avodji et al., 2019 | v0.12.0 |
| 6 | Conditional Fairness | Hardt et al., 2016 | v0.12.0 |

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
│   ├── robustness.py         # Phase 2 robustness metrics
│   ├── agreement.py          # Phase 2 agreement metrics
│   ├── complexity.py         # Phase 4 complexity metrics
│   ├── localisation.py       # Phase 3 localisation metrics
│   ├── randomisation.py      # Phase 5 randomisation metrics
│   ├── axiomatic.py          # Phase 6 axiomatic metrics
│   ├── fairness.py           # Phase 7 fairness metrics + FairnessMetricRegistry
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
- **All tests must pass before release**

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1.0 | — | Initial release: LIME, SHAP, Anchors, PI, PDP |
| v0.2.0 | — | TreeSHAP, Counterfactual, ALE, SAGE |
| v0.3.0 | — | Evaluation metrics, IG, DeepLIFT, DeepSHAP, GradCAM |
| v0.4.0 | — | ProtoDash |
| v0.5.0 | Jan 2025 | SmoothGrad |
| v0.6.0 | Jan 2025 | Saliency Maps |
| v0.7.0 | Jan 2025 | TCAV (Concept-Based Explanations) |
| v0.8.0 | Jan 2025 | LRP (Layer-wise Relevance Propagation) |
| v0.8.1 | Feb 2025 | Faithfulness Estimate metric |
| v0.8.2 | Feb 2025 | Monotonicity metric (Arya et al., 2019) |
| v0.8.3 | Feb 2025 | Monotonicity-Nguyen metric (Nguyen et al., 2020) |
| v0.8.4 | Feb 2025 | Pixel Flipping metric (Bach et al., 2015) |
| v0.8.5 | Feb 2025 | Region Perturbation metric (Samek et al., 2015) |
| v0.8.6 | Feb 2025 | Selectivity (AOPC) metric (Montavon et al., 2018) |
| v0.8.7 | Feb 2025 | Sensitivity-n metric (Ancona et al., 2018) |
| v0.8.8 | Feb 2025 | Documentation updates |
| v0.8.9 | Feb 2025 | IROF metric (Rieger & Hansen, 2020) |
| v0.8.10 | Feb 2025 | Infidelity metric (Yeh et al., 2019) |
| v0.8.11 | Feb 2025 | ROAD metric (Rong et al., 2022) |
| v0.9.0 | Feb 2025 | Warning elimination, dependency audit, xgboost <4.0 support |
| v0.9.1 | Feb 2025 | Fix xgboost 3.x compatibility with SHAP TreeExplainer |
| v0.9.2 | Feb 2025 | LIME default num_features changed to all features |
| v0.9.3 | Feb 2025 | Fix LRP device mismatch, GradCAM flat input, scikit-learn >=1.6 |
| v0.9.4 | Feb 2025 | Phase 2: Max-Sensitivity, Avg-Sensitivity, Continuity |
| v0.9.5 | Feb 2025 | Phase 4: Sparseness, Complexity, Effective Complexity |
| v0.9.6 | Feb 2025 | Phase 2: Consistency, RIS, RRS, ROS (Agarwal et al., 2022) |
| v0.9.7 | Feb 2025 | Phase 2 complete: Feature Agreement, Rank Agreement |
| v0.9.8 | Feb 2025 | Phase 3 complete: 9 Localisation metrics. 40 total — exceeds Quantus (37) |
| v0.10.0 | Feb 2025 | Phase 5 complete: 5 Randomisation metrics. 45 total |
| v0.11.0 | Feb 2025 | Phase 6 complete: 4 Axiomatic metrics. 49 total, 32% ahead of Quantus |
| v0.12.0 | Mar 2025 | Phase 7 complete: 6 Fairness metrics + FairnessMetricRegistry. 55 total, 49% ahead of Quantus |

---

## References

### Core Methods
- LIME: Ribeiro et al., 2016 — "Why Should I Trust You?"
- SHAP: Lundberg & Lee, 2017 — "A Unified Approach to Interpreting Model Predictions"
- Integrated Gradients: Sundararajan et al., 2017 — "Axiomatic Attribution for Deep Networks"
- TCAV: Kim et al., 2018 — "Interpretability Beyond Feature Attribution"
- LRP: Bach et al., 2015 — "On Pixel-wise Explanations for Non-Linear Classifier Decisions"

### Evaluation Metrics
- Faithfulness Estimate: Alvarez-Melis & Jaakkola, 2018 — "Towards Robust Interpretability"
- Monotonicity: Arya et al., 2019 — "One Explanation Does Not Fit All"
- Monotonicity-Nguyen: Nguyen & Martinez, 2020 — "Quantitative Evaluation of ML Explanations"
- Pixel Flipping: Bach et al., 2015
- Region Perturbation: Samek et al., 2015 — "Evaluating the Visualization of What a Deep Neural Network has Learned"
- IROF: Rieger & Hansen, 2020
- Infidelity: Yeh et al., 2019
- ROAD: Rong et al., 2022
- Insertion/Deletion: Petsiuk et al., 2018
- Pointing Game: Zhang et al., 2018
- Attribution Localisation: Kohlbrenner et al., 2020
- Top-K Intersection: Theiner et al., 2021
- Relevance Mass/Rank Accuracy: Arras et al., 2022
- ROC-AUC: Fawcett, 2006
- Energy-Based Pointing Game (Score-CAM): Wang et al., 2020
- Focus: Arias-Duart et al., 2022
- MPRT / Data Randomisation: Adebayo et al., 2018 — "Sanity Checks for Saliency Maps"
- Random Logit Test: Sixt et al., 2020 — "When Explanations Lie"
- Smooth MPRT / Efficient MPRT: Hedstrom et al., 2023 — "Sanity Checks Revisited"
- Completeness / Symmetry: Sundararajan et al., 2017 — "Axiomatic Attribution for Deep Networks"
- Non-Sensitivity: Nguyen & Martinez, 2020 — "On Quantitative Aspects of Model Interpretability"
- Input Invariance: Kindermans et al., 2017 — "The (Un)reliability of Saliency Methods"

### Fairness Metrics
- Group Fairness: Dai et al., 2022 — "Fairness via Explanation Quality" (AIES)
- Individual Fairness: Dwork et al., 2012 — "Fairness Through Awareness" (ITCS)
- Counterfactual Fairness: Kusner et al., 2017 — "Counterfactual Fairness" (NeurIPS)
- Fidelity Disparity: Balagopalan et al., 2022 — "The Road to Explainability is Paved with Bias" (FAccT)
- Attribution Parity: Avodji et al., 2019 — "Fairwashing: the risk of rationalization" (ICML)
- Conditional Fairness: Hardt et al., 2016 — "Equality of Opportunity in Supervised Learning" (NeurIPS)

---

*Last updated: March 2025 (v0.12.0)*
