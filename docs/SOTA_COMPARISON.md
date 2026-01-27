# Explainiverse vs State-of-the-Art XAI Frameworks

## Comprehensive Comparison Analysis (January 2026)

### Major XAI Frameworks Analyzed

| Framework | Maintainer | Focus | Active |
|-----------|-----------|-------|--------|
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
| Guided Backprop | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| LRP | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
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

### 4. EVALUATION METRICS

| Metric Type | Explainiverse | OmniXAI | Captum | Alibi | OpenXAI | AIX360 |
|-------------|:-------------:|:-------:|:------:|:-----:|:-------:|:------:|
| **Faithfulness** |
| PGI (Prediction Gap Important) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| PGU (Prediction Gap Unimportant) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Comprehensiveness | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Sufficiency | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Faithfulness Correlation | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Stability** |
| RIS (Relative Input Stability) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| ROS (Relative Output Stability) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Lipschitz Estimate | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Other** |
| Fairness Metrics | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Ground-truth Comparison | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

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

## Explainiverse Current Strengths

### Competitive Advantages

| Strength | Description |
|----------|-------------|
| **Unified Registry** | Plugin architecture with rich metadata, filtering by scope/model/data type |
| **Evaluation Metrics** | 8 built-in metrics (most frameworks have 0) - only OpenXAI competes here |
| **SAGE** | Global Shapley importance - rare in other frameworks |
| **ALE** | Accumulated Local Effects - only Alibi also has this |
| **TreeSHAP** | Optimized exact SHAP for tree models |
| **Anchors** | Rule-based explanations - only Alibi has this |
| **ProtoDash** | Example-based with importance weights - only AIX360 has this |
| **Clean API** | Consistent BaseExplainer interface across all methods |
| **Gradient Family** | Complete set: IG, DeepLIFT, DeepSHAP, SmoothGrad, Saliency, GradCAM |

### Current Implementation (v0.7.1)

**17 Explainers:**
- Local Perturbation: LIME, KernelSHAP, TreeSHAP
- Local Gradient: Integrated Gradients, DeepLIFT, DeepSHAP, SmoothGrad, Saliency Maps, GradCAM/GradCAM++
- Concept-Based: TCAV
- Rule-Based: Anchors
- Counterfactual: DiCE-style
- Example-Based: ProtoDash
- Global: Permutation Importance, PDP, ALE, SAGE

**8 Evaluation Metrics:**
- Faithfulness: PGI, PGU, Comprehensiveness, Sufficiency, Faithfulness Correlation
- Stability: RIS, ROS, Lipschitz Estimate

---

## Gap Analysis: Remaining Opportunities

### HIGH PRIORITY (For Publication Impact)

| Gap | Competitor Has It | Priority | Notes |
|-----|-------------------|----------|-------|
| **TCAV** | Captum | ✅ Complete | Concept-based explanations - now implemented in v0.7.0 |
| **LRP** | Captum | 🔴 Critical | Layer-wise Relevance Propagation - next priority |
| **Influence Functions** | Captum | 🟡 High | Training data attribution |

### MEDIUM PRIORITY

| Gap | Competitor Has It | Priority | Notes |
|-----|-------------------|----------|-------|
| Text/NLP Support | OmniXAI, Captum, Alibi | 🟡 Medium | Token importance, attention |
| Time Series | OmniXAI, Captum | 🟡 Medium | Temporal explanations |
| TensorFlow Adapter | OmniXAI, Alibi | 🟡 Medium | Keras/TF2 support |
| CEM (Contrastive) | OmniXAI, Alibi, AIX360 | 🟡 Medium | Pertinent positives/negatives |
| Occlusion | OmniXAI, Captum | 🟢 Low | Image perturbation method |

### LOWER PRIORITY

| Gap | Competitor Has It | Priority | Notes |
|-----|-------------------|----------|-------|
| Guided Backprop | OmniXAI, Captum | 🟢 Low | Gradient filtering |
| GUI Dashboard | OmniXAI, Captum, InterpretML | 🟢 Low | Interactive visualization |
| Glass-Box (EBM) | InterpretML | 🟢 Low | Wrapper for InterpretML |
| Fairness Metrics | OpenXAI | 🟢 Low | Group disparity measures |

---

## Summary Statistics

| Metric | Explainiverse | OmniXAI | Captum | Alibi | OpenXAI |
|--------|:-------------:|:-------:|:------:|:-----:|:-------:|
| **Explanation Methods** | 17 | ~25 | ~20 | ~15 | ~10 |
| **Evaluation Metrics** | 8 | 0 | 0 | 0 | 22 |
| **Data Types** | 2 | 4 | 4 | 3 | 1 |
| **ML Frameworks** | 2 | 3 | 1 | 3 | 1 |

### Explainiverse Position

```
                    Methods Coverage
                         ↑
                    High │  OmniXAI    Captum
                         │      
                         │  Explainiverse ←── Good balance
                         │      
                    Low  │  OpenXAI
                         └────────────────────→
                         Low              High
                              Evaluation Metrics
```

**Key Insight:** Explainiverse occupies a unique position with strong evaluation metrics (rivaling OpenXAI) combined with comprehensive explanation methods (approaching OmniXAI/Captum). With TCAV implemented in v0.7.0, Explainiverse now offers concept-based explanations that only Captum previously had among major frameworks.

---

## Strategic Roadmap

### Phase 1: Concept-Based (v0.7.0) ✅ COMPLETE
- **TCAV** - Testing with Concept Activation Vectors
- High publication impact, differentiator from most frameworks

### Phase 2: Propagation Methods (v0.8.0) - NEXT
- **LRP** - Layer-wise Relevance Propagation
- Completes the gradient method family

### Phase 3: Multi-Modal (v0.9.0)
- Text/NLP support
- TensorFlow adapter

### Phase 4: Production & Polish (v1.0.0)
- Visualization dashboard
- Performance optimization
- Documentation for publication

---

## References

### Frameworks
- OmniXAI: https://github.com/salesforce/OmniXAI
- Captum: https://captum.ai/
- Alibi: https://github.com/SeldonIO/alibi
- InterpretML: https://github.com/interpretml/interpret
- AIX360: https://github.com/Trusted-AI/AIX360
- OpenXAI: https://github.com/AI4LIFE-GROUP/OpenXAI

### Key Papers
- TCAV: Kim et al., 2018 - "Interpretability Beyond Feature Attribution" (ICML)
- LRP: Bach et al., 2015 - "On Pixel-Wise Explanations" (PLOS ONE)
- Evaluation: Petsiuk et al., 2018; DeYoung et al., 2020; Agarwal et al., 2022

---

*Last updated: January 2026 (v0.7.0)*
*Next review: After LRP implementation*
