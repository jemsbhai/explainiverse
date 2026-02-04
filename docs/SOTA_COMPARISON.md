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
| IROF | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Infidelity | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Insertion/Deletion AUC | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Stability** |
| RIS | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| ROS | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Lipschitz Estimate | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Max-Sensitivity | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Localisation** |
| Pointing Game | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Top-K Intersection | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Complexity** |
| Sparseness | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Effective Complexity | ⏳ | ✅ | ❌ | ❌ | ❌ | ❌ |
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
| **Evaluation Metrics** | 15 → **54** | 37 | 0 | 0 | 22 |
| **Data Types** | 2 | N/A | 4 | 4 | 1 |
| **ML Frameworks** | 2 | N/A | 3 | 1 | 1 |

---

## Explainiverse Competitive Position

### Current Strengths (v0.8.6)

| Strength | Description |
|----------|-------------|
| **Unified Registry** | Plugin architecture with rich metadata, filtering by scope/model/data type |
| **Growing Evaluation Suite** | 12 metrics now, targeting 54 (will exceed Quantus's 37) |
| **Complete Gradient Family** | IG, DeepLIFT, DeepSHAP, SmoothGrad, Saliency, GradCAM, LRP |
| **LRP with Multiple Rules** | ε, γ, αβ, z⁺, composite - comprehensive propagation rules |
| **SAGE** | Global Shapley importance - rare in other frameworks |
| **ALE** | Accumulated Local Effects - only Alibi also has this |
| **Anchors** | Rule-based explanations - only Alibi has this |
| **ProtoDash** | Example-based with importance weights - only AIX360 has this |
| **Clean API** | Consistent BaseExplainer interface across all methods |

### Current Implementation (v0.8.6)

**18 Explainers:**
- Local Perturbation: LIME, KernelSHAP, TreeSHAP
- Local Gradient: Integrated Gradients, DeepLIFT, DeepSHAP, SmoothGrad, Saliency Maps, GradCAM/GradCAM++
- Local Decomposition: LRP (Layer-wise Relevance Propagation)
- Concept-Based: TCAV
- Rule-Based: Anchors
- Counterfactual: DiCE-style
- Example-Based: ProtoDash
- Global: Permutation Importance, PDP, ALE, SAGE

**15 Evaluation Metrics:**
- Faithfulness (Core): PGI, PGU, Comprehensiveness, Sufficiency, Faithfulness Correlation
- Faithfulness (Extended): Faithfulness Estimate, Monotonicity, Monotonicity-Nguyen, Pixel Flipping, Region Perturbation, Selectivity (AOPC)
- Stability: RIS, ROS, Lipschitz Estimate

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

Current: Explainiverse at (18 methods, 15 metrics)
Target:  Explainiverse at (18 methods, 54 metrics) - Best in class for metrics!
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
| 7 | Sensitivity-n | ❌ |
| 8 | IROF | ❌ |
| 9 | Infidelity | ❌ |
| 10 | ROAD | ❌ |
| 11 | Insertion AUC | ❌ |
| 12 | Deletion AUC | ❌ |

### Future Phases

| Phase | Version | Category | New Metrics |
|-------|---------|----------|-------------|
| 2 | v0.10.0 | Robustness | +7 |
| 3 | v0.11.0 | Localisation | +8 |
| 4 | v0.12.0 | Complexity | +4 |
| 5 | v0.13.0 | Randomisation | +5 |
| 6 | v0.14.0 | Axiomatic | +4 |
| 7 | v0.15.0 | Fairness | +4 |

---

## Gap Analysis: Remaining Opportunities

### For Metrics Dominance (HIGH PRIORITY)

| Gap | Priority | Notes |
|-----|----------|-------|
| Complete Phase 1 Faithfulness | 🔴 Critical | 6 more metrics to implement |
| Robustness metrics | 🔴 High | Phase 2 |
| Localisation metrics | 🟡 Medium | Phase 3 |
| Complexity metrics | 🟡 Medium | Phase 4 |

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

*Last updated: February 2025 (v0.8.6)*
*Next review: After Phase 1 completion*
