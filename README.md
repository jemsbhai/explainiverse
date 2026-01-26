# Explainiverse

**Explainiverse** is a unified, extensible Python framework for Explainable AI (XAI).  
It provides a standardized interface for model-agnostic explainability with 11 state-of-the-art XAI methods, evaluation metrics, and a plugin registry for easy extensibility.

---

## Features

### 🎯 Comprehensive XAI Coverage

**Local Explainers** (instance-level explanations):
- **LIME** - Local Interpretable Model-agnostic Explanations ([Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938))
- **SHAP** - SHapley Additive exPlanations via KernelSHAP ([Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874))
- **TreeSHAP** - Exact SHAP values for tree models, 10x+ faster ([Lundberg et al., 2018](https://arxiv.org/abs/1802.03888))
- **Integrated Gradients** - Axiomatic attributions for neural networks ([Sundararajan et al., 2017](https://arxiv.org/abs/1703.01365))
- **GradCAM/GradCAM++** - Visual explanations for CNNs ([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391))
- **Anchors** - High-precision rule-based explanations ([Ribeiro et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11491))
- **Counterfactual** - DiCE-style diverse counterfactual explanations ([Mothilal et al., 2020](https://arxiv.org/abs/1905.07697))

**Global Explainers** (model-level explanations):
- **Permutation Importance** - Feature importance via performance degradation ([Breiman, 2001](https://link.springer.com/article/10.1023/A:1010933404324))
- **Partial Dependence (PDP)** - Marginal feature effects ([Friedman, 2001](https://projecteuclid.org/euclid.aos/1013203451))
- **ALE** - Accumulated Local Effects, unbiased for correlated features ([Apley & Zhu, 2020](https://academic.oup.com/jrsssb/article/82/4/1059/7056085))
- **SAGE** - Shapley Additive Global importancE ([Covert et al., 2020](https://arxiv.org/abs/2004.00668))

### 🔌 Extensible Plugin Registry
- Register custom explainers with rich metadata
- Filter by scope (local/global), model type, data type
- Automatic recommendations based on use case

### 📊 Evaluation Metrics
- **AOPC** (Area Over Perturbation Curve)
- **ROAR** (Remove And Retrain)
- Multiple baseline options and curve generation

### 🧪 Standardized Interface
- Consistent `BaseExplainer` API
- Unified `Explanation` output format
- Model adapters for sklearn and PyTorch

---

## Installation

From PyPI:

```bash
pip install explainiverse
```

With PyTorch support (for neural network explanations):

```bash
pip install explainiverse[torch]
```

For development:

```bash
git clone https://github.com/jemsbhai/explainiverse.git
cd explainiverse
poetry install
```

---

## Quick Start

### Using the Registry (Recommended)

```python
from explainiverse import default_registry, SklearnAdapter
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Train a model
iris = load_iris()
model = RandomForestClassifier().fit(iris.data, iris.target)
adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())

# List available explainers
print(default_registry.list_explainers())
# ['lime', 'shap', 'treeshap', 'integrated_gradients', 'gradcam', 'anchors', 'counterfactual', 'permutation_importance', 'partial_dependence', 'ale', 'sage']

# Create and use an explainer
explainer = default_registry.create(
    "lime",
    model=adapter,
    training_data=iris.data,
    feature_names=iris.feature_names,
    class_names=iris.target_names.tolist()
)
explanation = explainer.explain(iris.data[0])
print(explanation.explanation_data["feature_attributions"])
```

### Filter Explainers by Criteria

```python
# Find local explainers for tabular data
local_tabular = default_registry.filter(scope="local", data_type="tabular")
print(local_tabular)  # ['lime', 'shap', 'treeshap', 'integrated_gradients', 'anchors', 'counterfactual']

# Find explainers for images/CNNs
image_explainers = default_registry.filter(data_type="image")
print(image_explainers)  # ['lime', 'integrated_gradients', 'gradcam']

# Get recommendations
recommendations = default_registry.recommend(
    model_type="any",
    data_type="tabular",
    scope_preference="local"
)
```

### TreeSHAP for Tree Models (10x+ Faster)

```python
from explainiverse.explainers import TreeShapExplainer
from sklearn.ensemble import RandomForestClassifier

# Train a tree-based model
model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

# TreeSHAP works directly with the model (no adapter needed)
explainer = TreeShapExplainer(
    model=model,
    feature_names=feature_names,
    class_names=class_names
)

# Single instance explanation
explanation = explainer.explain(X_test[0])
print(explanation.explanation_data["feature_attributions"])

# Batch explanations (efficient)
explanations = explainer.explain_batch(X_test[:10])

# Feature interactions
interactions = explainer.explain_interactions(X_test[0])
print(interactions.explanation_data["interaction_matrix"])
```

### PyTorch Adapter for Neural Networks

```python
from explainiverse import PyTorchAdapter
import torch.nn as nn

# Define a PyTorch model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

# Wrap with adapter
adapter = PyTorchAdapter(
    model,
    task="classification",
    class_names=["cat", "dog", "bird"]
)

# Use with any explainer
predictions = adapter.predict(X)  # Returns numpy array

# Get gradients for attribution methods
predictions, gradients = adapter.predict_with_gradients(X)

# Access intermediate layers
activations = adapter.get_layer_output(X, layer_name="0")
```

### Integrated Gradients for Neural Networks

```python
from explainiverse.explainers import IntegratedGradientsExplainer
from explainiverse import PyTorchAdapter

# Wrap your PyTorch model
adapter = PyTorchAdapter(model, task="classification", class_names=class_names)

# Create IG explainer
explainer = IntegratedGradientsExplainer(
    model=adapter,
    feature_names=feature_names,
    class_names=class_names,
    n_steps=50  # More steps = more accurate
)

# Explain a prediction
explanation = explainer.explain(X_test[0])
print(explanation.explanation_data["feature_attributions"])

# Check convergence (sum of attributions ≈ F(x) - F(baseline))
explanation = explainer.explain(X_test[0], return_convergence_delta=True)
print(f"Convergence delta: {explanation.explanation_data['convergence_delta']}")
```

### GradCAM for CNN Visual Explanations

```python
from explainiverse.explainers import GradCAMExplainer
from explainiverse import PyTorchAdapter

# Wrap your CNN model
adapter = PyTorchAdapter(cnn_model, task="classification", class_names=class_names)

# Find the last convolutional layer
layers = adapter.list_layers()
target_layer = "layer4"  # Adjust based on your model architecture

# Create GradCAM explainer
explainer = GradCAMExplainer(
    model=adapter,
    target_layer=target_layer,
    class_names=class_names,
    method="gradcam"  # or "gradcam++" for improved version
)

# Explain an image prediction
explanation = explainer.explain(image)  # image shape: (C, H, W) or (N, C, H, W)
heatmap = explanation.explanation_data["heatmap"]

# Create overlay visualization
overlay = explainer.get_overlay(original_image, heatmap, alpha=0.5)
```

### Using Specific Explainers

```python
# Anchors - Rule-based explanations
from explainiverse.explainers import AnchorsExplainer

anchors = AnchorsExplainer(
    model=adapter,
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names
)
explanation = anchors.explain(instance)
print(explanation.explanation_data["rules"])
# ['petal length (cm) > 2.45', 'petal width (cm) <= 1.75']

# Counterfactual - What-if explanations
from explainiverse.explainers import CounterfactualExplainer

cf = CounterfactualExplainer(
    model=adapter,
    training_data=X_train,
    feature_names=feature_names
)
explanation = cf.explain(instance, num_counterfactuals=3)
print(explanation.explanation_data["changes"])

# SAGE - Global Shapley importance
from explainiverse.explainers import SAGEExplainer

sage = SAGEExplainer(
    model=adapter,
    X=X_train,
    y=y_train,
    feature_names=feature_names
)
explanation = sage.explain()
print(explanation.explanation_data["feature_attributions"])
```

### Explanation Suite (Multi-Explainer Comparison)

```python
from explainiverse import ExplanationSuite

suite = ExplanationSuite(
    model=adapter,
    explainer_configs=[
        ("lime", {"training_data": X_train, "feature_names": feature_names, "class_names": class_names}),
        ("shap", {"background_data": X_train[:50], "feature_names": feature_names, "class_names": class_names}),
    ]
)

results = suite.run(instance)
suite.compare()
```

---

## Registering Custom Explainers

```python
from explainiverse import ExplainerRegistry, ExplainerMeta, BaseExplainer

@default_registry.register_decorator(
    name="my_explainer",
    meta=ExplainerMeta(
        scope="local",
        model_types=["any"],
        data_types=["tabular"],
        description="My custom explainer",
        paper_reference="Author et al., 2024"
    )
)
class MyExplainer(BaseExplainer):
    def explain(self, instance, **kwargs):
        # Your implementation
        return Explanation(...)
```

---

## Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=explainiverse

# Run specific test file
poetry run pytest tests/test_new_explainers.py -v
```

---

## Roadmap

- [x] LIME, SHAP (KernelSHAP)
- [x] TreeSHAP (optimized for tree models) ✅
- [x] Anchors, Counterfactuals
- [x] Permutation Importance, PDP, ALE, SAGE
- [x] Explainer Registry with filtering
- [x] PyTorch Adapter ✅
- [x] Integrated Gradients ✅
- [x] GradCAM/GradCAM++ for CNNs ✅ NEW
- [ ] TensorFlow adapter
- [ ] Interactive visualization dashboard

---

## Citation

If you use Explainiverse in your research, please cite:

```bibtex
@software{explainiverse2024,
  title = {Explainiverse: A Unified Framework for Explainable AI},
  author = {Syed, Muntaser},
  year = {2024},
  url = {https://github.com/jemsbhai/explainiverse}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
