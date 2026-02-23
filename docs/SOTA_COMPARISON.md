# Explainiverse vs State-of-the-Art XAI Frameworks

## Comprehensive Comparison Analysis (February 2025)

### Major XAI Frameworks Analyzed

| Framework | Maintainer | Focus | Active |
|-----------|-----------|-------|--------|
| **Quantus** | Understandable ML | Evaluation metrics | ✅ |
| **OmniXAI** | Salesforce | Multi-modal, unified interface | ✅ |
| **Captum** | Meta (PyTorch) | Deep learning attribution | ✅ |
| **Alibi** | Seldon | Production-ready explanations | ✅ |
| **InterpretML** | Microsoft | Glass-box + black-box | ✅ |
| **AIX360** | IBM/Linux Foundation | Diverse explanation types | ✅ |
| **OpenXAI** | Harvard/Academic | Evaluation & benchmarking | ✅ |
| **SHAP** | Lundberg | Shapley-based attributions | ✅ |

---

## Feature Matrix Comparison

### 1. EXPLANATION METHODS

| Method | Explainiverse | OmniXAI | Captum | Alibi | InterpretML | AIX360 |
|--------|:-------------:|:-------:|:------:|:-----:|:-----------:|:------:|
| **Local Attribution** |
| LIME | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| KernelSHAP | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| TreeSHAP | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Integrated Gradients | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| GradCAM/GradCAM++ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| DeepLIFT | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| DeepSHAP | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Saliency Maps | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| SmoothGrad | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| LRP | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Guided Backprop | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Occlusion | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Feature Ablation | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Rule-Based** |
| Anchors | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Counterfactual** |
| DiCE-style | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| CEM (Contrastive) | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Prototype CF | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Global Methods** |
| Permutation Importance | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| PDP | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| ALE | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| SAGE | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Concept-Based** |
| TCAV | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Example-Based** |
| ProtoDash | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Influence Functions | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Glass-Box Models** |
| EBM (Explainable Boosting) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| GLRM (Rule Models) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Boolean Rules (BRCG) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### 2. DATA TYPES SUPPORTED

| Data Type | Explainiverse | OmniXAI | Captum | Alibi | InterpretML | AIX360 |
|-----------|:-------------:|:-------:|:------:|:-----:|:-----------:|:------:|
| Tabular | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Images | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Text/NLP | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Time Series | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ |

### 3. ML FRAMEWORK SUPPORT

| Framework | Explainiverse | OmniXAI | Captum | Alibi | InterpretML | AIX360 |
|-----------|:-------------:|:-------:|:------:|:-----:|:-----------:|:------:|
| Scikit-learn | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| PyTorch | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| TensorFlow | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ |
| XGBoost/LightGBM | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |

### 4. EVALUATION METRICS (Key Differentiator)

| Metric | Explainiverse | Quantus | OpenXAI | OmniXAI | Captum | Alibi |
|--------|:-------------:|:-------:|:-------:|:-------:|:------:|:-----:|
| **Faithfulness** |
| PGI | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| PGU | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Comprehensiveness | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Sufficiency | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Faithfulness Correlation | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Faithfulness Estimate | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Monotonicity (Arya) | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Monotonicity-Nguyen | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pixel Flipping | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Region Perturbation | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Selectivity | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Sensitivity-n | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| IROF | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Infidelity | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| ROAD | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Insertion/Deletion AUC | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Stability** |
| RIS | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| ROS | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Lipschitz Estimate | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Max-Sensitivity | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Avg-Sensitivity | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Continuity | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Consistency | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Relative Input Stability (RIS) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Relative Representation Stability (RRS) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Relative Output Stability (ROS) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Agreement** |
| Feature Agreement | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Rank Agreement | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Localisation** |
| Pointing Game | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Attribution Localisation | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Top-K Intersection | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Relevance Mass Accuracy | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Relevance Rank Accuracy | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| AUC (localisation) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Energy-Based Pointing Game | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Focus | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Attribution IoU | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Complexity** |
| Sparseness | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Complexity | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Effective Complexity | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Randomisation** |
| Model Param Randomisation | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Random Logit | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Axiomatic** |
| Completeness | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Non-Sensitivity | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |

**Legend:** ✅ = Implemented | ⏳ = Planned | ❌ = Not available

### 5. INFRASTRUCTURE & TOOLING

| Feature | Explainiverse | OmniXAI | Captum | Alibi | InterpretML |
|---------|:-------------:|:-------:|:------:|:-----:|:-----------:|
| GUI Dashboard | ❌ | ✅ | ✅ | ❌ | ✅ |
| Jupyter Integration | ✅ | ✅ | ✅ | ✅ | ✅ |
| Plugin Registry | ✅ | ✅ | ❌ | ❌ | ❌ |
| Explainer Filtering | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multi-Explainer Suite | ✅ | ✅ | ❌ | ❌ | ✅ |
| BentoML Deployment | ❌ | ✅ | ❌ | ❌ | ❌ |
| GPT/LLM Explainer | ❌ | ✅ | ❌ | ❌ | ❌ |

---

## Summary Statistics

| Metric | Explainiverse | Quantus | OmniXAI | Captum | OpenXAI |
|--------|:-------------:|:-------:|:-------:|:------:|:-------:|
| **Explanation Methods** | 18 | 0 | ~25 | ~20 | ~10 |
| **Evaluation Metrics** | **40** → 53 | 37 | 0 | 0 | 22 |
| **Data Types** | 2 | N/A | 4 | 4 | 1 |
| **ML Frameworks** | 2 | N/A | 3 | 1 | 1 |

---

## Explainiverse Competitive Position

### Current Strengths (v0.9.8)

| Strength | Description |
|----------|-------------|
| **Unified Registry** | Plugin architecture with rich metadata, filtering by scope/model/data type |
| **Evaluation Suite Leadership** | 40 metrics — exceeds Quantus (37), targeting 53. Most comprehensive XAI evaluation framework. |
| **Complete Gradient Family** | IG, DeepLIFT, DeepSHAP, SmoothGrad, Saliency, GradCAM, LRP |
| **LRP with Multiple Rules** | ε, γ, αβ, z⁺, composite - comprehensive propagation rules |
| **SAGE** | Global Shapley importance - rare in other frameworks |
| **ALE** | Accumulated Local Effects - only Alibi also has this |
| **Anchors** | Rule-based explanations - only Alibi has this |
| **ProtoDash** | Example-based with importance weights - only AIX360 has this |
| **Clean API** | Consistent BaseExplainer interface across all methods |
| **xgboost 3.x Support** | Compatible with xgboost 1.7–3.x via automatic SHAP compatibility patching |

### Current Implementation (v0.9.8)

**18 Explainers:**
- Local Perturbation: LIME, KernelSHAP, TreeSHAP
- Local Gradient: Integrated Gradients, DeepLIFT, DeepSHAP, SmoothGrad, Saliency Maps, GradCAM/GradCAM++
- Local Decomposition: LRP (Layer-wise Relevance Propagation)
- Concept-Based: TCAV
- Rule-Based: Anchors
- Counterfactual: DiCE-style
- Example-Based: ProtoDash
- Global: Permutation Importance, PDP, ALE, SAGE

**40 Evaluation Metrics:**
- Faithfulness (Core): PGI, PGU, Comprehensiveness, Sufficiency, Faithfulness Correlation
- Faithfulness (Extended): Faithfulness Estimate, Monotonicity, Monotonicity-Nguyen, Pixel Flipping, Region Perturbation, Selectivity (AOPC), Sensitivity-n, IROF, Infidelity, ROAD
- Insertion/Deletion: Insertion AUC, Deletion AUC (Petsiuk et al., 2018)
- Stability (legacy): RIS (simple), ROS (simple), Lipschitz Estimate
- Robustness: Max-Sensitivity, Avg-Sensitivity (Yeh et al., 2019), Continuity (Montavon et al., 2018), Consistency (Dasgupta et al., 2022), Relative Input Stability / RIS (Agarwal et al., 2022, Eq 2), Relative Representation Stability / RRS (Agarwal et al., 2022, Eq 3), Relative Output Stability / ROS (Agarwal et al., 2022, Eq 5)
- Agreement: Feature Agreement, Rank Agreement (Krishna et al., 2022)
- Complexity: Sparseness (Chalasani et al., 2020), Complexity (Bhatt et al., 2020), Effective Complexity (Nguyen & Martínez, 2020)
- Localisation: Pointing Game (Zhang et al., 2018), Attribution Localisation (Kohlbrenner et al., 2020), Top-K Intersection (Theiner et al., 2021), Relevance Mass Accuracy (Arras et al., 2022), Relevance Rank Accuracy (Arras et al., 2022), AUC (Fawcett, 2006), Energy-Based Pointing Game (Wang et al., 2020), Focus (Arias-Duart et al., 2022), Attribution IoU

---

## Strategic Position

```
                    Methods Coverage
                         ↑
                    High │  OmniXAI    Captum
                         │      
                         │  Explainiverse ←── Balanced + Growing Metrics
                         │      
                    Low  │  OpenXAI    Quantus
                         └────────────────────────────→
                         Low                      High
                              Evaluation Metrics

Current: Explainiverse at (18 methods, 40 metrics) - EXCEEDS Quantus (37)!
Target:  Explainiverse at (18 methods, 53 metrics) - Undisputed leader for metrics!
```

**Key Insight:** Explainiverse is uniquely positioned to become the **only framework** combining:
1. Comprehensive explanation methods (rivaling OmniXAI/Captum)
2. Extensive evaluation metrics (exceeding Quantus)

No other framework currently achieves both.

---

## Metrics Expansion Roadmap

### Phase 1: Faithfulness (v0.8.x → v0.9.0) - IN PROGRESS

| # | Metric | Status |
|---|--------|--------|
| 1 | Faithfulness Estimate | ✅ v0.8.1 |
| 2 | Monotonicity (Arya) | ✅ v0.8.2 |
| 3 | Monotonicity-Nguyen | ✅ v0.8.3 |
| 4 | Pixel Flipping | ✅ v0.8.4 |
| 5 | Region Perturbation | ✅ v0.8.5 |
| 6 | Selectivity (AOPC) | ✅ v0.8.6 |
| 7 | Sensitivity-n | ✅ v0.8.7 |
| 8 | IROF | ✅ v0.8.9 |
| 9 | Infidelity | ✅ v0.8.10 |
| 10 | ROAD | ✅ v0.8.11 |
| 11 | Insertion AUC | ✅ v0.9.1 |
| 12 | Deletion AUC | ✅ v0.9.1 |

### Phase 2: Robustness & Agreement - ✅ COMPLETE

| # | Metric | Status |
|---|--------|--------|
| 1 | Max-Sensitivity | ✅ v0.9.4 |
| 2 | Avg-Sensitivity | ✅ v0.9.4 |
| 3 | Continuity | ✅ v0.9.4 |
| 4 | Consistency | ✅ v0.9.6 |
| 5 | Relative Input Stability (RIS) | ✅ v0.9.6 |
| 6 | Relative Representation Stability (RRS) | ✅ v0.9.6 |
| 7 | Relative Output Stability (ROS) | ✅ v0.9.6 |
| 8 | Feature Agreement | ✅ v0.9.7 |
| 9 | Rank Agreement | ✅ v0.9.7 |

### Phase 4: Complexity (v0.9.5) - COMPLETE

| # | Metric | Status |
|---|--------|--------|
| 1 | Sparseness | ✅ v0.9.5 |
| 2 | Complexity | ✅ v0.9.5 |
| 3 | Effective Complexity | ✅ v0.9.5 |

### Future Phases

| Phase | Version | Category | New Metrics | Status |
|-------|---------|----------|-------------|--------|
| ~~2~~ | ~~v0.10.0~~ | ~~Robustness~~ | ~~+9~~ | ✅ Complete |
| ~~3~~ | ~~v0.11.0~~ | ~~Localisation~~ | ~~+9~~ | ✅ Complete |
| 5 | v0.10.0 | Randomisation | +5 | ⏳ Planned |
| 6 | v0.11.0 | Axiomatic | +4 | ⏳ Planned |
| 7 | v0.12.0 | Fairness | +4 | ⏳ Planned |

---

## Gap Analysis: Remaining Opportunities

### For Metrics Dominance (HIGH PRIORITY)

| Gap | Priority | Notes |
|-----|----------|-------|
| ~~Complete Phase 2 Robustness~~ | ✅ Done | All 9 metrics implemented |
| ~~Localisation metrics~~ | ✅ Done | All 9 metrics implemented (Phase 3) |
| Randomisation metrics | 🟡 Medium | Phase 5 (5 metrics) |

### For Methods Coverage (LOWER PRIORITY)

| Gap | Priority | Notes |
|-----|----------|-------|
| Text/NLP Support | 🟡 Medium | After metrics expansion |
| TensorFlow Adapter | 🟡 Medium | After metrics expansion |
| Influence Functions | 🟢 Low | Nice to have |

---

## References

### Frameworks
- Quantus: https://github.com/understandable-machine-intelligence-lab/Quantus
- OmniXAI: https://github.com/salesforce/OmniXAI
- Captum: https://captum.ai/
- Alibi: https://github.com/SeldonIO/alibi
- InterpretML: https://github.com/interpretml/interpret
- AIX360: https://github.com/Trusted-AI/AIX360
- OpenXAI: https://github.com/AI4LIFE-GROUP/OpenXAI

### Key Evaluation Papers
- Faithfulness Estimate: Alvarez-Melis & Jaakkola, 2018
- Monotonicity: Arya et al., 2019
- Monotonicity-Nguyen: Nguyen & Martinez, 2020
- Pixel Flipping: Bach et al., 2015
- IROF: Rieger & Hansen, 2020
- Infidelity: Yeh et al., 2019
- ROAD: Rong et al., 2022
- Insertion/Deletion: Petsiuk et al., 2018

---

*Last updated: February 2025 (v0.9.8)*
*Next review: After Phase 5 (Randomisation) begins*
