# Explainiverse

[![PyPI version](https://badge.fury.io/py/explainiverse.svg)](https://badge.fury.io/py/explainiverse)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Explainiverse** is a unified, extensible Python framework for Explainable AI (XAI). It provides a standardized interface for **18 state-of-the-art explanation methods** across local, global, gradient-based, concept-based, and example-based paradigms, along with **55 evaluation metrics** across 8 categories for assessing explanation quality — **49% more metrics than Quantus**, the previous state of the art.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **18 Explainers** | LIME, KernelSHAP, TreeSHAP, Integrated Gradients, DeepLIFT, DeepSHAP, SmoothGrad, Saliency Maps, GradCAM/GradCAM++, LRP, TCAV, Anchors, Counterfactual, Permutation Importance, PDP, ALE, SAGE, ProtoDash |
| **55 Evaluation Metrics** | Faithfulness (17), Robustness (7), Localisation (9), Fairness (6), Randomisation (5), Axiomatic (4), Stability (3), Complexity (3), Agreement (2) — see detailed tables below |
| **Unified API** | Consistent `BaseExplainer` interface with standardized `Explanation` output |
| **Plugin Registry** | Filter explainers by scope, model type, data type; automatic recommendations |
| **Fairness Registry** | Extensible `FairnessMetricRegistry` with decorator-based registration |
| **Framework Support** | Adapters for scikit-learn and PyTorch (with gradient computation) |

---

## Explainer Coverage

### Local Explainers (Instance-Level)

| Method | Type | Reference |
|--------|------|-----------|
| **LIME** | Perturbation | [Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938) |
| **KernelSHAP** | Perturbation | [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874) |
| **TreeSHAP** | Exact (Trees) | [Lundberg et al., 2018](https://arxiv.org/abs/1802.03888) |
| **Integrated Gradients** | Gradient | [Sundararajan et al., 2017](https://arxiv.org/abs/1703.01365) |
| **DeepLIFT** | Gradient | [Shrikumar et al., 2017](https://arxiv.org/abs/1704.02685) |
| **DeepSHAP** | Gradient + Shapley | [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874) |
| **SmoothGrad** | Gradient | [Smilkov et al., 2017](https://arxiv.org/abs/1706.03825) |
| **Saliency Maps** | Gradient | [Simonyan et al., 2014](https://arxiv.org/abs/1312.6034) |
| **GradCAM / GradCAM++** | Gradient (CNN) | [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391) |
| **LRP** | Decomposition | [Bach et al., 2015](https://doi.org/10.1371/journal.pone.0130140) |
| **TCAV** | Concept-Based | [Kim et al., 2018](https://arxiv.org/abs/1711.11279) |
| **Anchors** | Rule-Based | [Ribeiro et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11491) |
| **Counterfactual** | Contrastive | [Mothilal et al., 2020](https://arxiv.org/abs/1905.07697) |
| **ProtoDash** | Example-Based | [Gurumoorthy et al., 2019](https://arxiv.org/abs/1707.01212) |

### Global Explainers (Model-Level)

| Method | Type | Reference |
|--------|------|-----------|
| **Permutation Importance** | Feature Importance | [Breiman, 2001](https://link.springer.com/article/10.1023/A:1010933404324) |
| **Partial Dependence (PDP)** | Feature Effect | [Friedman, 2001](https://projecteuclid.org/euclid.aos/1013203451) |
| **ALE** | Feature Effect | [Apley & Zhu, 2020](https://academic.oup.com/jrsssb/article/82/4/1059/7056085) |
| **SAGE** | Shapley Importance | [Covert et al., 2020](https://arxiv.org/abs/2004.00668) |

---

## Evaluation Metrics (55 total)

Explainiverse provides the most comprehensive evaluation metrics suite among XAI frameworks, with 55 metrics across 8 categories — 49% more than Quantus (37 metrics).

### Faithfulness (17 metrics)

| Metric | Description | Reference |
|--------|-------------|-----------|
| **PGI** | Prediction Gap on Important features | [Petsiuk et al., 2018](https://arxiv.org/abs/1806.07421) |
| **PGU** | Prediction Gap on Unimportant features | [Petsiuk et al., 2018](https://arxiv.org/abs/1806.07421) |
| **Comprehensiveness** | Drop when removing top-k features | [DeYoung et al., 2020](https://arxiv.org/abs/1911.03429) |
| **Sufficiency** | Prediction using only top-k features | [DeYoung et al., 2020](https://arxiv.org/abs/1911.03429) |
| **Faithfulness Correlation** | Correlation between attribution and impact | [Bhatt et al., 2020](https://arxiv.org/abs/2005.00631) |
| **Faithfulness Estimate** | Correlation of attributions with single-feature perturbation impact | [Alvarez-Melis & Jaakkola, 2018](https://arxiv.org/abs/1806.08049) |
| **Monotonicity** | Sequential feature addition shows monotonic prediction increase | [Arya et al., 2019](https://arxiv.org/abs/1909.03012) |
| **Monotonicity-Nguyen** | Spearman correlation between attributions and feature removal impact | [Nguyen & Martinez, 2020](https://arxiv.org/abs/2010.07455) |
| **Pixel Flipping** | AUC of prediction degradation when removing features by importance | [Bach et al., 2015](https://doi.org/10.1371/journal.pone.0130140) |
| **Region Perturbation** | AUC of prediction degradation when perturbing feature regions | [Samek et al., 2015](https://arxiv.org/abs/1509.06321) |
| **Selectivity (AOPC)** | Average prediction drop when sequentially removing features | [Montavon et al., 2018](https://doi.org/10.1016/j.dsp.2017.10.011) |
| **Sensitivity-n** | Correlation between attribution sums and prediction changes for subsets | [Ancona et al., 2018](https://arxiv.org/abs/1711.06104) |
| **IROF** | Iterative Removal of Features — area over prediction degradation curve | [Rieger & Hansen, 2020](https://arxiv.org/abs/2003.08747) |
| **Infidelity** | How well attributions predict model output changes under perturbation | [Yeh et al., 2019](https://arxiv.org/abs/1901.09392) |
| **ROAD** | RemOve And Debias — noisy linear imputation for OOD-robust evaluation | [Rong et al., 2022](https://proceedings.mlr.press/v162/rong22a.html) |
| **Insertion AUC** | AUC of prediction recovery when inserting features by importance | [Petsiuk et al., 2018](https://arxiv.org/abs/1806.07421) |
| **Deletion AUC** | AUC of prediction degradation when deleting features by importance | [Petsiuk et al., 2018](https://arxiv.org/abs/1806.07421) |

### Robustness (7 metrics)

| Metric | Description | Reference |
|--------|-------------|-----------|
| **Max-Sensitivity** | Maximum explanation change under input perturbation | [Yeh et al., 2019](https://arxiv.org/abs/1901.09392) |
| **Avg-Sensitivity** | Average explanation change under input perturbation | [Yeh et al., 2019](https://arxiv.org/abs/1901.09392) |
| **Continuity** | Lipschitz-based smoothness of explanation function | [Montavon et al., 2018](https://doi.org/10.1016/j.dsp.2017.10.011) |
| **Consistency** | Agreement of explanations across similar inputs with same prediction | [Dasgupta et al., 2022](https://arxiv.org/abs/2202.00734) |
| **Relative Input Stability (RIS)** | Normalized explanation change relative to input change | [Agarwal et al., 2022](https://arxiv.org/abs/2203.06877) |
| **Relative Representation Stability (RRS)** | Normalized explanation change relative to representation change | [Agarwal et al., 2022](https://arxiv.org/abs/2203.06877) |
| **Relative Output Stability (ROS)** | Normalized explanation change relative to output change | [Agarwal et al., 2022](https://arxiv.org/abs/2203.06877) |

### Localisation (9 metrics)

| Metric | Description | Reference |
|--------|-------------|-----------|
| **Pointing Game** | Whether max attribution falls within ground-truth region | [Zhang et al., 2018](https://arxiv.org/abs/1608.00507) |
| **Attribution Localisation** | Fraction of positive attributions inside the ground-truth region | [Kohlbrenner et al., 2020](https://arxiv.org/abs/1910.09840) |
| **Top-K Intersection** | Overlap between top-k attributed features and ground-truth features | [Theiner et al., 2021](https://arxiv.org/abs/2104.14995) |
| **Relevance Mass Accuracy** | Mass of attribution within the ground-truth region | [Arras et al., 2022](https://arxiv.org/abs/2202.07397) |
| **Relevance Rank Accuracy** | Rank accuracy of attribution within the ground-truth region | [Arras et al., 2022](https://arxiv.org/abs/2202.07397) |
| **AUC** | ROC-AUC treating attribution as classifier for ground-truth mask | [Fawcett, 2006](https://doi.org/10.1016/j.patrec.2005.10.010) |
| **Energy-Based Pointing Game** | Ratio of attribution energy inside vs total | [Wang et al., 2020](https://arxiv.org/abs/1910.01279) |
| **Focus** | Concentration of attribution mass in relevant regions | [Arias-Duart et al., 2022](https://arxiv.org/abs/2202.03482) |
| **Attribution IoU** | Intersection-over-Union between thresholded attribution and mask | — |

### Fairness (6 metrics)

| Metric | Description | Reference |
|--------|-------------|-----------|
| **Group Fairness** | Disparity of explanation quality across demographic groups | [Dai et al., 2022](https://arxiv.org/abs/2205.07277) |
| **Individual Fairness** | Similar individuals should receive similar explanations | [Dwork et al., 2012](https://arxiv.org/abs/1104.3913) |
| **Counterfactual Explanation Fairness** | Fairness of counterfactual explanations across protected groups | [Kusner et al., 2017](https://arxiv.org/abs/1703.06856) |
| **Fidelity Disparity** | Disparity in explanation fidelity across demographic groups | [Balagopalan et al., 2022](https://arxiv.org/abs/2205.03295) |
| **Attribution Parity** | Equal distribution of feature attributions across groups | [Avodji et al., 2019](https://arxiv.org/abs/1901.09749) |
| **Conditional Fairness** | Fairness of explanations conditioned on legitimate features | [Hardt et al., 2016](https://arxiv.org/abs/1610.02413) |

### Randomisation (5 metrics)

| Metric | Description | Reference |
|--------|-------------|-----------|
| **MPRT** | Model Parameter Randomisation Test — sanity check for saliency | [Adebayo et al., 2018](https://arxiv.org/abs/1810.03292) |
| **Random Logit Test** | Explanation sensitivity to output logit randomisation | [Sixt et al., 2020](https://arxiv.org/abs/1912.09818) |
| **Smooth MPRT** | Smoothed variant of MPRT for reduced variance | [Hedström et al., 2023](https://arxiv.org/abs/2301.12431) |
| **Efficient MPRT** | Computationally efficient MPRT approximation | [Hedström et al., 2023](https://arxiv.org/abs/2301.12431) |
| **Data Randomisation Test** | Explanation sensitivity to training label randomisation | [Adebayo et al., 2018](https://arxiv.org/abs/1810.03292) |

### Axiomatic (4 metrics)

| Metric | Description | Reference |
|--------|-------------|-----------|
| **Completeness** | Attributions sum to difference between output and baseline | [Sundararajan et al., 2017](https://arxiv.org/abs/1703.01365) |
| **Non-Sensitivity** | Zero attribution for features that do not influence the output | [Nguyen & Martinez, 2020](https://arxiv.org/abs/2010.07455) |
| **Input Invariance** | Attributions invariant to input transformations that preserve output | [Kindermans et al., 2017](https://arxiv.org/abs/1711.00867) |
| **Symmetry** | Symmetric features receive equal attributions | [Sundararajan et al., 2017](https://arxiv.org/abs/1703.01365) |

### Stability (3 metrics)

| Metric | Description | Reference |
|--------|-------------|-----------|
| **RIS** | Relative Input Stability (simplified) | [Agarwal et al., 2022](https://arxiv.org/abs/2203.06877) |
| **ROS** | Relative Output Stability (simplified) | [Agarwal et al., 2022](https://arxiv.org/abs/2203.06877) |
| **Lipschitz Estimate** | Local Lipschitz continuity of explanation function | [Alvarez-Melis & Jaakkola, 2018](https://arxiv.org/abs/1806.08049) |

### Complexity (3 metrics)

| Metric | Description | Reference |
|--------|-------------|-----------|
| **Sparseness** | Gini coefficient of attribution distribution | [Chalasani et al., 2020](https://arxiv.org/abs/1901.09392) |
| **Complexity** | Entropy-based complexity of the attribution | [Bhatt et al., 2020](https://arxiv.org/abs/2005.00631) |
| **Effective Complexity** | Number of features with attribution above threshold | [Nguyen & Martinez, 2020](https://arxiv.org/abs/2010.07455) |

### Agreement (2 metrics)

| Metric | Description | Reference |
|--------|-------------|-----------|
| **Feature Agreement** | Overlap in top-k features between two explanation methods | [Krishna et al., 2022](https://arxiv.org/abs/2202.01602) |
| **Rank Agreement** | Rank correlation between two explanation methods | [Krishna et al., 2022](https://arxiv.org/abs/2202.01602) |

---

## Installation

```bash
# From PyPI
pip install explainiverse

# With PyTorch support (for gradient-based methods)
pip install explainiverse[torch]

# For development
git clone https://github.com/jemsbhai/explainiverse.git
cd explainiverse
poetry install
```

---

## Quick Start

### Basic Usage with Registry

```python
from explainiverse import default_registry, SklearnAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Train a model
iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)

# Wrap with adapter
adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())

# List all available explainers
print(default_registry.list_explainers())
# ['lime', 'shap', 'treeshap', 'integrated_gradients', 'deeplift', 'deepshap',
#  'smoothgrad', 'saliency', 'gradcam', 'lrp', 'tcav', 'anchors', 'counterfactual',
#  'protodash', 'permutation_importance', 'partial_dependence', 'ale', 'sage']

# Create an explainer via registry
explainer = default_registry.create(
    "lime",
    model=adapter,
    training_data=iris.data,
    feature_names=iris.feature_names.tolist(),
    class_names=iris.target_names.tolist()
)

# Generate explanation
explanation = explainer.explain(iris.data[0])
print(explanation.explanation_data["feature_attributions"])
```

### Filter and Recommend Explainers

```python
# Filter by criteria
local_explainers = default_registry.filter(scope="local", data_type="tabular")
neural_explainers = default_registry.filter(model_type="neural")
image_explainers = default_registry.filter(data_type="image")

# Get recommendations
recommendations = default_registry.recommend(
    model_type="neural",
    data_type="tabular",
    scope_preference="local",
    max_results=5
)
```

---

## Gradient-Based Explainers (PyTorch)

### Integrated Gradients

```python
from explainiverse import PyTorchAdapter
from explainiverse.explainers.gradient import IntegratedGradientsExplainer
import torch.nn as nn

# Define and wrap model
model = nn.Sequential(
    nn.Linear(10, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 3)
)
adapter = PyTorchAdapter(model, task="classification", class_names=["A", "B", "C"])

# Create explainer
explainer = IntegratedGradientsExplainer(
    model=adapter,
    feature_names=[f"feature_{i}" for i in range(10)],
    class_names=["A", "B", "C"],
    n_steps=50,
    method="riemann_trapezoid"
)

# Explain with convergence check
explanation = explainer.explain(X[0], return_convergence_delta=True)
print(f"Attributions: {explanation.explanation_data['feature_attributions']}")
print(f"Convergence delta: {explanation.explanation_data['convergence_delta']:.6f}")
```

### Layer-wise Relevance Propagation (LRP)

```python
from explainiverse.explainers.gradient import LRPExplainer

# LRP - Decomposition-based attribution with conservation property
explainer = LRPExplainer(
    model=adapter,
    feature_names=feature_names,
    class_names=class_names,
    rule="epsilon",       # Propagation rule: epsilon, gamma, alpha_beta, z_plus, composite
    epsilon=1e-6          # Stabilization constant
)

# Basic explanation
explanation = explainer.explain(X[0], target_class=0)
print(explanation.explanation_data["feature_attributions"])

# Verify conservation property (sum of attributions ~ target output)
explanation = explainer.explain(X[0], return_convergence_delta=True)
print(f"Conservation delta: {explanation.explanation_data['convergence_delta']:.6f}")

# Compare different LRP rules
comparison = explainer.compare_rules(X[0], rules=["epsilon", "gamma", "z_plus"])
for rule, result in comparison.items():
    print(f"{rule}: top feature = {result['top_feature']}")

# Layer-wise relevance analysis
layer_result = explainer.explain_with_layer_relevances(X[0])
for layer, relevances in layer_result["layer_relevances"].items():
    print(f"{layer}: sum = {sum(relevances):.4f}")

# Composite rules: different rules for different layers
explainer_composite = LRPExplainer(
    model=adapter,
    feature_names=feature_names,
    class_names=class_names,
    rule="composite"
)
explainer_composite.set_composite_rule({
    0: "z_plus",    # Input layer: focus on what's present
    2: "epsilon",   # Middle layers: balanced
    4: "epsilon"    # Output layer
})
explanation = explainer_composite.explain(X[0])
```

**LRP Propagation Rules:**

| Rule | Description | Use Case |
|------|-------------|----------|
| `epsilon` | Adds stabilization constant | General purpose (default) |
| `gamma` | Enhances positive contributions | Image classification |
| `alpha_beta` | Separates pos/neg (alpha-beta=1) | Fine-grained control |
| `z_plus` | Only positive weights | Input layers, what's present |
| `composite` | Different rules per layer | Best practice for deep nets |

### DeepLIFT and DeepSHAP

```python
from explainiverse.explainers.gradient import DeepLIFTExplainer, DeepLIFTShapExplainer

# DeepLIFT - Fast reference-based attributions
deeplift = DeepLIFTExplainer(
    model=adapter,
    feature_names=feature_names,
    class_names=class_names,
    baseline=None  # Uses zero baseline by default
)
explanation = deeplift.explain(X[0])

# DeepSHAP - DeepLIFT averaged over background samples
deepshap = DeepLIFTShapExplainer(
    model=adapter,
    feature_names=feature_names,
    class_names=class_names,
    background_data=X_train[:100]
)
explanation = deepshap.explain(X[0])
```

### SmoothGrad

```python
from explainiverse.explainers.gradient import SmoothGradExplainer

# SmoothGrad - Noise-averaged gradients for smoother saliency
explainer = SmoothGradExplainer(
    model=adapter,
    feature_names=feature_names,
    class_names=class_names,
    n_samples=50,
    noise_scale=0.15,
    noise_type="gaussian"  # or "uniform"
)

# Standard SmoothGrad
explanation = explainer.explain(X[0], method="smoothgrad")

# SmoothGrad-Squared (sharper attributions)
explanation = explainer.explain(X[0], method="smoothgrad_squared")

# VarGrad (variance of gradients)
explanation = explainer.explain(X[0], method="vargrad")
```

### GradCAM for CNNs

```python
from explainiverse.explainers.gradient import GradCAMExplainer

# For CNN models
adapter = PyTorchAdapter(cnn_model, task="classification", class_names=class_names)

explainer = GradCAMExplainer(
    model=adapter,
    target_layer="layer4",  # Last conv layer
    class_names=class_names,
    method="gradcam++"  # or "gradcam"
)

explanation = explainer.explain(image)
heatmap = explanation.explanation_data["heatmap"]
```

---

## Evaluation Examples

### Faithfulness

```python
from explainiverse.evaluation import (
    compute_pgi, compute_pgu,
    compute_comprehensiveness, compute_sufficiency,
    compute_faithfulness_correlation
)

pgi = compute_pgi(model=adapter, instance=X[0], attributions=attributions,
                  feature_names=feature_names, top_k=3)

comp = compute_comprehensiveness(model=adapter, instance=X[0], attributions=attributions,
                                 feature_names=feature_names, top_k_values=[1, 2, 3, 5])
```

### Robustness

```python
from explainiverse.evaluation import (
    compute_max_sensitivity, compute_avg_sensitivity,
    compute_continuity, compute_consistency,
    compute_relative_input_stability
)

max_sens = compute_max_sensitivity(
    explainer=explainer, instance=X[0],
    n_perturbations=10, perturbation_scale=0.1
)
```

### Localisation

```python
from explainiverse.evaluation import (
    LocalisationMask, compute_pointing_game,
    compute_attribution_localisation, compute_attribution_iou
)

mask = LocalisationMask(mask=binary_mask, feature_names=feature_names)
pg = compute_pointing_game(attributions=attributions, mask=mask)
iou = compute_attribution_iou(attributions=attributions, mask=mask, threshold=0.5)
```

### Fairness

```python
from explainiverse.evaluation import (
    compute_group_fairness, compute_individual_fairness,
    compute_counterfactual_fairness, compute_fidelity_disparity,
    compute_attribution_parity, compute_conditional_fairness
)

gf = compute_group_fairness(
    attributions_list=attributions_list,
    sensitive_features=group_labels,
    inner_metric=None  # defaults to L1 norm
)

ind_f = compute_individual_fairness(
    attributions_list=attributions_list,
    instances=X, distance_threshold=0.5
)
```

### Randomisation

```python
from explainiverse.evaluation import compute_mprt, compute_random_logit

mprt = compute_mprt(model=pytorch_model, explainer=explainer,
                    instance=X[0], target_class=0)

rlt = compute_random_logit(model=pytorch_model, explainer=explainer,
                           instance=X[0], target_class=0)
```

### Axiomatic

```python
from explainiverse.evaluation import (
    compute_completeness, compute_non_sensitivity,
    compute_input_invariance, compute_symmetry
)

comp = compute_completeness(model=adapter, instance=X[0],
                            attributions=attributions, baseline=baseline)

sym = compute_symmetry(model=adapter, instance=X[0],
                       attributions=attributions,
                       symmetric_indices=[(0, 1)])
```

---

## Global Explainers

```python
from explainiverse.explainers import (
    PermutationImportanceExplainer,
    PartialDependenceExplainer,
    ALEExplainer,
    SAGEExplainer
)

# Permutation Importance
perm_imp = PermutationImportanceExplainer(
    model=adapter, X=X_test, y=y_test,
    feature_names=feature_names, n_repeats=10
)
explanation = perm_imp.explain()

# ALE (handles correlated features)
ale = ALEExplainer(model=adapter, X=X_train, feature_names=feature_names)
explanation = ale.explain(feature="feature_0", n_bins=20)

# SAGE (global Shapley importance)
sage = SAGEExplainer(
    model=adapter, X=X_train, y=y_train,
    feature_names=feature_names, n_permutations=512
)
explanation = sage.explain()
```

---

## Multi-Explainer Comparison

```python
from explainiverse import ExplanationSuite

suite = ExplanationSuite(
    model=adapter,
    explainer_configs=[
        ("lime", {"training_data": X_train, "feature_names": feature_names, "class_names": class_names}),
        ("shap", {"background_data": X_train[:50], "feature_names": feature_names, "class_names": class_names}),
        ("treeshap", {"feature_names": feature_names, "class_names": class_names}),
    ]
)

results = suite.run(X_test[0])
suite.compare()
```

---

## Custom Explainer Registration

Explainiverse's plugin architecture allows you to register custom explainers that integrate seamlessly with the registry's discovery, filtering, and recommendation system.

```python
from explainiverse import default_registry, BaseExplainer, Explanation
from explainiverse.core.registry import ExplainerMeta

@default_registry.register_decorator(
    name="my_explainer",
    meta=ExplainerMeta(
        scope="local",
        model_types=["any"],
        data_types=["tabular"],
        task_types=["classification", "regression"],
        description="My custom attribution method",
        paper_reference="Author et al., 2024 - 'My Method' (Conference)",
        complexity="O(n * d)",
        requires_training_data=False,
        supports_batching=True
    )
)
class MyExplainer(BaseExplainer):
    def __init__(self, model, feature_names, class_names=None, **kwargs):
        super().__init__(model)
        self.feature_names = feature_names
        self.class_names = class_names

    def explain(self, instance, target_class=None, **kwargs):
        attributions = self._compute_attributions(instance, target_class)
        return Explanation(
            explainer_name="MyExplainer",
            target_class=str(target_class or 0),
            explanation_data={"feature_attributions": attributions},
            feature_names=self.feature_names,
            metadata={"method": "my_method", "params": kwargs}
        )
```

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
│   ├── faithfulness.py       # Core faithfulness (PGI, PGU, Comprehensiveness, Sufficiency)
│   ├── faithfulness_extended.py  # 12 extended faithfulness metrics
│   ├── stability.py          # RIS, ROS, Lipschitz (simplified)
│   ├── robustness.py         # 7 robustness metrics (Phase 2)
│   ├── agreement.py          # Feature Agreement, Rank Agreement (Phase 2)
│   ├── complexity.py         # Sparseness, Complexity, Effective Complexity (Phase 4)
│   ├── localisation.py       # 9 localisation metrics (Phase 3)
│   ├── randomisation.py      # 5 randomisation metrics (Phase 5)
│   ├── axiomatic.py          # 4 axiomatic metrics (Phase 6)
│   ├── fairness.py           # 6 fairness metrics + FairnessMetricRegistry (Phase 7)
│   ├── metrics.py            # AOPC, ROAR
│   └── _utils.py             # Shared utilities
└── engine/
    └── suite.py              # Multi-explainer comparison
```

---

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=explainiverse --cov-report=html

# Run specific test file
poetry run pytest tests/test_fairness.py -v

# Run specific test class
poetry run pytest tests/test_lrp.py::TestLRPConv2d -v
```

---

## Roadmap

### Completed
- [x] Core framework (BaseExplainer, Explanation, Registry)
- [x] 18 explainers across 7 categories
- [x] 55 evaluation metrics across 8 categories
- [x] PyTorch adapter with gradient support
- [x] FairnessMetricRegistry for extensible fairness evaluation
- [x] Plugin system with decorator-based registration

### Planned
- [ ] Attention-based explanations (for Transformers)
- [ ] TensorFlow/Keras adapter
- [ ] Interactive visualization dashboard
- [ ] Explanation caching and serialization
- [ ] Distributed computation support

---

## Citation

If you use Explainiverse in your research, please cite:

```bibtex
@software{explainiverse2025,
  title = {Explainiverse: A Unified Framework for Explainable AI},
  author = {Syed, Muntaser},
  year = {2025},
  url = {https://github.com/jemsbhai/explainiverse},
  version = {0.12.0}
}
```

---

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`poetry run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Explainiverse builds upon the foundational work of many researchers in the XAI community. We thank the authors of LIME, SHAP, Integrated Gradients, DeepLIFT, LRP, GradCAM, TCAV, Anchors, DiCE, ALE, SAGE, and ProtoDash for their contributions to interpretable machine learning.
