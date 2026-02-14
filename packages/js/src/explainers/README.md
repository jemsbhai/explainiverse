# Explainers

This directory contains the JavaScript/TypeScript implementations of the explainers from the Python `explainiverse` package.

## Roadmap

The following explainers are planned for porting:

### Local Explainers (Instance-Level)
1.  **LIME** (Perturbation)
2.  **KernelSHAP** (Perturbation)
3.  **TreeSHAP** (Exact for Trees)
4.  **Integrated Gradients** (Gradient)
5.  **DeepLIFT** (Gradient)
6.  **DeepSHAP** (Gradient + Shapley)
7.  **SmoothGrad** (Gradient)
8.  **Saliency Maps** (Gradient)
9.  **GradCAM / GradCAM++** (Gradient for CNNs)
10. **LRP** (Decomposition)
11. **TCAV** (Concept-Based)
12. **Anchors** (Rule-Based)
13. **Counterfactual** (Contrastive)
14. **ProtoDash** (Example-Based)

### Global Explainers (Model-Level)
15. **Permutation Importance** (Feature Importance)
16. **Partial Dependence (PDP)** (Feature Effect)
17. **ALE** (Feature Effect)
18. **SAGE** (Shapley Importance)
