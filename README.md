# Explainiverse

[![PyPI version](https://badge.fury.io/py/explainiverse.svg)](https://badge.fury.io/py/explainiverse)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Explainiverse** is a unified, extensible Python framework for Explainable AI (XAI). It provides a standardized interface for **18 state-of-the-art explanation methods** across local, global, gradient-based, concept-based, and example-based paradigms, along with **comprehensive evaluation metrics** for assessing explanation quality.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **18 Explainers** | LIME, KernelSHAP, TreeSHAP, Integrated Gradients, DeepLIFT, DeepSHAP, SmoothGrad, Saliency Maps, GradCAM/GradCAM++, LRP, TCAV, Anchors, Counterfactual, Permutation Importance, PDP, ALE, SAGE, ProtoDash |
| **19 Evaluation Metrics** | Faithfulness (PGI, PGU, Comprehensiveness, Sufficiency, Correlation, Faithfulness Estimate, Monotonicity, Monotonicity-Nguyen, Pixel Flipping, Region Perturbation, Selectivity, Sensitivity-n, IROF, Infidelity, ROAD) and Stability (RIS, ROS, Lipschitz) |
| **Unified API** | Consistent `BaseExplainer` interface with standardized `Explanation` output |
| **Plugin Registry** | Filter explainers by scope, model type, data type; automatic recommendations |
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

## Evaluation Metrics

Explainiverse includes a comprehensive suite of evaluation metrics based on the XAI literature:

### Faithfulness Metrics

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
| **Region Perturbation** | AUC of prediction degradation when perturbing feature regions by importance | [Samek et al., 2015](https://arxiv.org/abs/1509.06321) |
| **Selectivity (AOPC)** | Average prediction drop when sequentially removing features by importance | [Montavon et al., 2018](https://doi.org/10.1016/j.dsp.2017.10.011) |
| **Sensitivity-n** | Correlation between attribution sums and prediction changes for random feature subsets | [Ancona et al., 2018](https://arxiv.org/abs/1711.06104) |
| **IROF** | Area over curve measuring prediction degradation when iteratively removing features | [Rieger & Hansen, 2020](https://arxiv.org/abs/2003.08747) |
| **Infidelity** | Measures how well attributions predict model output changes under perturbation | [Yeh et al., 2019](https://arxiv.org/abs/1901.09392) |
| **ROAD** | RemOve And Debias - uses noisy linear imputation for out-of-distribution robust evaluation | [Rong et al., 2022](https://proceedings.mlr.press/v162/rong22a.html) |

### Stability Metrics

| Metric | Description | Reference |
|--------|-------------|-----------|
| **RIS** | Relative Input Stability | [Agarwal et al., 2022](https://arxiv.org/abs/2203.06877) |
| **ROS** | Relative Output Stability | [Agarwal et al., 2022](https://arxiv.org/abs/2203.06877) |
| **Lipschitz Estimate** | Local Lipschitz continuity | [Alvarez-Melis & Jaakkola, 2018](https://arxiv.org/abs/1806.08049) |

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
print(f"Convergence δ: {explanation.explanation_data['convergence_delta']:.6f}")
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

# Verify conservation property (sum of attributions ≈ target output)
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
| `alpha_beta` | Separates pos/neg (α-β=1) | Fine-grained control |
| `z_plus` | Only positive weights | Input layers, what's present |
| `composite` | Different rules per layer | Best practice for deep nets |

**Supported Layers:**
- Linear, Conv2d
- BatchNorm1d, BatchNorm2d
- ReLU, LeakyReLU, ELU, Tanh, Sigmoid, GELU
- MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- Flatten, Dropout

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

### Saliency Maps

```python
from explainiverse.explainers.gradient import SaliencyExplainer

# Saliency Maps - simplest and fastest gradient method
explainer = SaliencyExplainer(
    model=adapter,
    feature_names=feature_names,
    class_names=class_names,
    absolute_value=True  # Default: absolute gradient magnitudes
)

# Standard saliency (absolute gradients)
explanation = explainer.explain(X[0], method="saliency")

# Input × Gradient (gradient scaled by input values)
explanation = explainer.explain(X[0], method="input_times_gradient")

# Signed saliency (keep gradient direction)
explainer_signed = SaliencyExplainer(
    model=adapter,
    feature_names=feature_names,
    class_names=class_names,
    absolute_value=False
)
explanation = explainer_signed.explain(X[0])

# Compare all variants
variants = explainer.compute_all_variants(X[0])
print(variants["saliency_absolute"])
print(variants["saliency_signed"])
print(variants["input_times_gradient"])
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

# With absolute values
explanation = explainer.explain(X[0], absolute_value=True)
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
overlay = explainer.get_overlay(original_image, heatmap, alpha=0.5)
```

### TCAV (Concept-Based Explanations)

```python
from explainiverse.explainers.gradient import TCAVExplainer

# For neural network models with concept examples
adapter = PyTorchAdapter(model, task="classification", class_names=class_names)

# Create TCAV explainer targeting a specific layer
explainer = TCAVExplainer(
    model=adapter,
    layer_name="layer3",  # Target layer for concept analysis
    class_names=class_names
)

# Learn a concept from examples (e.g., "striped" pattern)
explainer.learn_concept(
    concept_name="striped",
    concept_examples=striped_images,      # Images with stripes
    negative_examples=random_images,      # Random images without stripes
    min_accuracy=0.6                      # Minimum CAV classifier accuracy
)

# Compute TCAV score: fraction of inputs where concept positively influences prediction
tcav_score = explainer.compute_tcav_score(
    test_inputs=test_images,
    target_class=0,           # e.g., "zebra"
    concept_name="striped"
)
print(f"TCAV score: {tcav_score:.3f}")  # >0.5 means concept positively influences class

# Statistical significance testing against random concepts
result = explainer.statistical_significance_test(
    test_inputs=test_images,
    target_class=0,
    concept_name="striped",
    n_random=10,
    negative_examples=random_images
)
print(f"p-value: {result['p_value']:.4f}, significant: {result['significant']}")

# Full explanation with multiple concepts
explanation = explainer.explain(
    test_inputs=test_images,
    target_class=0,
    run_significance_test=True
)
print(explanation.explanation_data["tcav_scores"])
```

---

## Example-Based Explanations

### ProtoDash

```python
from explainiverse.explainers.example_based import ProtoDashExplainer

explainer = ProtoDashExplainer(
    model=adapter,
    training_data=X_train,
    feature_names=feature_names,
    n_prototypes=5,
    kernel="rbf",
    gamma=0.1
)

explanation = explainer.explain(X_test[0])
print(explanation.explanation_data["prototype_indices"])
print(explanation.explanation_data["prototype_weights"])
```

---

## Evaluation Metrics

### Faithfulness Evaluation

```python
from explainiverse.evaluation import (
    compute_pgi, compute_pgu,
    compute_comprehensiveness, compute_sufficiency,
    compute_faithfulness_correlation
)

# PGI - Higher is better (important features affect predictions)
pgi = compute_pgi(
    model=adapter,
    instance=X[0],
    attributions=attributions,
    feature_names=feature_names,
    top_k=3
)

# PGU - Lower is better (unimportant features don't affect predictions)
pgu = compute_pgu(
    model=adapter,
    instance=X[0],
    attributions=attributions,
    feature_names=feature_names,
    top_k=3
)

# Comprehensiveness - Higher is better
comp = compute_comprehensiveness(
    model=adapter,
    instance=X[0],
    attributions=attributions,
    feature_names=feature_names,
    top_k_values=[1, 2, 3, 5]
)

# Sufficiency - Lower is better
suff = compute_sufficiency(
    model=adapter,
    instance=X[0],
    attributions=attributions,
    feature_names=feature_names,
    top_k_values=[1, 2, 3, 5]
)

# Faithfulness Correlation
corr = compute_faithfulness_correlation(
    model=adapter,
    instance=X[0],
    attributions=attributions,
    feature_names=feature_names
)
```

### Stability Evaluation

```python
from explainiverse.evaluation import (
    compute_ris, compute_ros, compute_lipschitz_estimate
)

# RIS - Relative Input Stability (lower is better)
ris = compute_ris(
    explainer=explainer,
    instance=X[0],
    n_perturbations=10,
    perturbation_scale=0.1
)

# ROS - Relative Output Stability (lower is better)
ros = compute_ros(
    model=adapter,
    explainer=explainer,
    instance=X[0],
    n_perturbations=10,
    perturbation_scale=0.1
)

# Lipschitz Estimate (lower is better)
lipschitz = compute_lipschitz_estimate(
    explainer=explainer,
    instance=X[0],
    n_perturbations=20,
    perturbation_scale=0.1
)
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
    model=adapter,
    X=X_test,
    y=y_test,
    feature_names=feature_names,
    n_repeats=10
)
explanation = perm_imp.explain()

# Partial Dependence Plot
pdp = PartialDependenceExplainer(
    model=adapter,
    X=X_train,
    feature_names=feature_names
)
explanation = pdp.explain(feature="feature_0", grid_resolution=50)

# ALE (handles correlated features)
ale = ALEExplainer(
    model=adapter,
    X=X_train,
    feature_names=feature_names
)
explanation = ale.explain(feature="feature_0", n_bins=20)

# SAGE (global Shapley importance)
sage = SAGEExplainer(
    model=adapter,
    X=X_train,
    y=y_train,
    feature_names=feature_names,
    n_permutations=512
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

Explainiverse's plugin architecture allows you to register your own custom explainers and have them integrate seamlessly with the registry's discovery, filtering, and recommendation system.

### Why Register Custom Explainers?

| Benefit | Description |
|---------|-------------|
| **Discoverability** | Your explainer appears in `list_explainers()` and can be filtered by criteria |
| **Rich Metadata** | Attach scope, model types, data types, paper references, and complexity info |
| **Unified API** | Create instances via `default_registry.create("my_explainer", ...)` |
| **Recommendations** | Your explainer can be recommended based on the user's use case |
| **Consistency** | Follows the same `BaseExplainer` interface as all built-in methods |

### Method 1: Decorator-Based Registration (Recommended)

The cleanest way to register a custom explainer:

```python
from explainiverse import default_registry, BaseExplainer, Explanation
from explainiverse.core.registry import ExplainerMeta

@default_registry.register_decorator(
    name="my_explainer",
    meta=ExplainerMeta(
        scope="local",                              # "local" or "global"
        model_types=["any"],                        # ["any", "tree", "linear", "neural", "ensemble"]
        data_types=["tabular"],                     # ["tabular", "image", "text", "time_series"]
        task_types=["classification", "regression"],
        description="My custom attribution method",
        paper_reference="Author et al., 2024 - 'My Method' (Conference)",
        complexity="O(n * d)",                      # Computational complexity
        requires_training_data=False,
        supports_batching=True
    )
)
class MyExplainer(BaseExplainer):
    """Custom explainer implementing your attribution method."""
    
    def __init__(self, model, feature_names, class_names=None, **kwargs):
        super().__init__(model)
        self.feature_names = feature_names
        self.class_names = class_names
    
    def explain(self, instance, target_class=None, **kwargs):
        """
        Generate explanation for a single instance.
        
        Args:
            instance: Input to explain (1D array for tabular)
            target_class: Class to explain (optional)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Explanation object with feature attributions
        """
        # Your attribution logic here
        attributions = self._compute_attributions(instance, target_class)
        
        # Return standardized Explanation object
        return Explanation(
            explainer_name="MyExplainer",
            target_class=str(target_class or 0),
            explanation_data={"feature_attributions": attributions},
            feature_names=self.feature_names,
            metadata={"method": "my_method", "params": kwargs}
        )
    
    def _compute_attributions(self, instance, target_class):
        """Compute feature attributions (implement your method here)."""
        # Example: simple gradient-like computation
        import numpy as np
        attributions = {}
        for i, name in enumerate(self.feature_names):
            attributions[name] = float(np.random.randn())  # Replace with real logic
        return attributions
```

### Method 2: Programmatic Registration

For dynamic registration or when decorators aren't suitable:

```python
from explainiverse import default_registry, BaseExplainer, Explanation
from explainiverse.core.registry import ExplainerMeta, get_default_registry

# Define your explainer class
class AnotherExplainer(BaseExplainer):
    def __init__(self, model, feature_names, **kwargs):
        super().__init__(model)
        self.feature_names = feature_names
    
    def explain(self, instance, **kwargs):
        # Implementation
        return Explanation(
            explainer_name="AnotherExplainer",
            target_class="0",
            explanation_data={"feature_attributions": {}},
            feature_names=self.feature_names
        )

# Register programmatically
registry = get_default_registry()
registry.register(
    name="another_explainer",
    explainer_class=AnotherExplainer,
    meta=ExplainerMeta(
        scope="local",
        model_types=["neural"],
        data_types=["image"],
        description="Another custom explainer"
    )
)
```

### Using Your Registered Explainer

Once registered, your explainer works like any built-in method:

```python
# Verify registration
print(default_registry.list_explainers())  # [..., 'my_explainer', ...]

# Check metadata
meta = default_registry.get_meta("my_explainer")
print(meta.description)  # "My custom attribution method"

# Create via registry
explainer = default_registry.create(
    "my_explainer",
    model=adapter,
    feature_names=feature_names,
    class_names=class_names
)

# Generate explanations
explanation = explainer.explain(X[0])
print(explanation.get_top_features(k=5))

# Your explainer is now discoverable via filtering
local_explainers = default_registry.filter(scope="local")
print("my_explainer" in local_explainers)  # True

# And included in recommendations
recommended = default_registry.recommend(
    model_type="any",
    data_type="tabular",
    scope_preference="local"
)
```

### ExplainerMeta Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `scope` | `str` | **Required.** `"local"` (instance-level) or `"global"` (model-level) |
| `model_types` | `List[str]` | Compatible models: `["any", "tree", "linear", "neural", "ensemble"]` |
| `data_types` | `List[str]` | Compatible data: `["tabular", "image", "text", "time_series"]` |
| `task_types` | `List[str]` | Compatible tasks: `["classification", "regression"]` |
| `description` | `str` | Human-readable description (shown in `summary()`) |
| `paper_reference` | `str` | Citation for the method's paper |
| `complexity` | `str` | Computational complexity (e.g., `"O(n^2)"`) |
| `requires_training_data` | `bool` | Whether `explain()` needs background/training data |
| `supports_batching` | `bool` | Whether the explainer can process batches efficiently |

### Managing Registrations

```python
from explainiverse.core.registry import get_default_registry

registry = get_default_registry()

# Override an existing registration
registry.register(
    name="my_explainer",
    explainer_class=ImprovedExplainer,
    meta=ExplainerMeta(scope="local", description="Improved version"),
    override=True  # Required to replace existing
)

# Unregister an explainer
registry.unregister("my_explainer")

# View summary of all registered explainers
print(registry.summary())
```

---

## Architecture

```
explainiverse/
├── core/
│   ├── explainer.py      # BaseExplainer abstract class
│   ├── explanation.py    # Unified Explanation container
│   └── registry.py       # ExplainerRegistry with metadata
├── adapters/
│   ├── sklearn_adapter.py
│   └── pytorch_adapter.py  # With gradient support
├── explainers/
│   ├── attribution/      # LIME, SHAP, TreeSHAP
│   ├── gradient/         # IG, DeepLIFT, DeepSHAP, SmoothGrad, Saliency, GradCAM, LRP, TCAV
│   ├── rule_based/       # Anchors
│   ├── counterfactual/   # DiCE-style
│   ├── global_explainers/  # Permutation, PDP, ALE, SAGE
│   └── example_based/    # ProtoDash
├── evaluation/
│   ├── faithfulness.py   # PGI, PGU, Comprehensiveness, Sufficiency
│   └── stability.py      # RIS, ROS, Lipschitz
└── engine/
    └── suite.py          # Multi-explainer comparison
```

---

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=explainiverse --cov-report=html

# Run specific test file
poetry run pytest tests/test_lrp.py -v

# Run specific test class
poetry run pytest tests/test_lrp.py::TestLRPConv2d -v
```

---

## Roadmap

### Completed ✅
- [x] Core framework (BaseExplainer, Explanation, Registry)
- [x] Perturbation methods: LIME, KernelSHAP, TreeSHAP
- [x] Gradient methods: Integrated Gradients, DeepLIFT, DeepSHAP, SmoothGrad, Saliency Maps, GradCAM/GradCAM++
- [x] Decomposition methods: Layer-wise Relevance Propagation (LRP) with ε, γ, αβ, z⁺, composite rules
- [x] Concept-based: TCAV (Testing with Concept Activation Vectors)
- [x] Rule-based: Anchors
- [x] Counterfactual: DiCE-style
- [x] Global: Permutation Importance, PDP, ALE, SAGE
- [x] Example-based: ProtoDash
- [x] Evaluation: Faithfulness metrics (PGI, PGU, Comprehensiveness, Sufficiency, Correlation)
- [x] Evaluation: Stability metrics (RIS, ROS, Lipschitz)
- [x] PyTorch adapter with gradient support

### In Progress 🔄
- [ ] **Evaluation metrics expansion** - Adding 42 more metrics across 7 categories to exceed Quantus (37 metrics)
  - Phase 1: Faithfulness (+12 metrics) - 10/12 complete
  - Phase 2: Robustness (+7 metrics)
  - Phase 3: Localisation (+8 metrics)
  - Phase 4: Complexity (+4 metrics)
  - Phase 5: Randomisation (+5 metrics)
  - Phase 6: Axiomatic (+4 metrics)
  - Phase 7: Fairness (+4 metrics)

### Planned 📋
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
  version = {0.8.4}
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
