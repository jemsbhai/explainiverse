# Explainiverse

**Explainiverse** is a unified, extensible, and testable Python framework for Explainable AI (XAI).  
It offers a standardized interface for model-agnostic explainability, evaluation metrics like AOPC and ROAR, and support for multiple XAI methods out of the box.

---

## Features

- Standardized `Explainer` API (`BaseExplainer`)
- Support for:
  - Local and global feature attribution
  - Regression and classification tasks
- Integrated explainers:
  - **LIME** (tabular, local surrogate)
  - **SHAP** (KernelExplainer with multi-class, regression, cohort support)
- Evaluation metrics:
  - **AOPC** (Area Over Perturbation Curve)
  - **ROAR** (Remove And Retrain)
    - Multiple `top_k` support
    - Baseline options: `"mean"`, `"median"`, `np.ndarray`, `callable`
    - Curve generation for ROAR vs feature importance
- Explainability Suite:
  - Run and compare multiple explainers
  - Auto-suggestion based on model/task type
- Built-in support for models: `LogisticRegression`, `RandomForest`, `SVC`, `KNN`, `XGB`, `NaiveBayes`, and more

---


## Installation

From PyPI:

```bash
pip install explainiverse
```

For development use:

```bash
git clone https://github.com/jemsbhai/explainiverse.git
cd explainiverse
poetry install
```

---

## Quick Example
```python

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.engine.suite import ExplanationSuite

# Wrap your model
adapter = SklearnAdapter(your_model, class_names=["yes", "no"])

# Build the suite
suite = ExplanationSuite(
    model=adapter,
    explainer_configs=[
        ("lime", {...}),
        ("shap", {...})
    ],
    data_meta={"task": "classification"}
)

results = suite.run(instance)
suite.compare()
suite.evaluate_roar(X_train, y_train, X_test, y_test, top_k=3)
```


---

## Running Tests

All tests can be run using:

```bash
poetry run python tests/test_all.py
```

For individual component testing:

```bash
poetry run python tests/test_shap_explainer.py
poetry run python tests/test_lime_explainer.py
poetry run python tests/test_evaluation_metrics.py
```

---

## Documentation

Documentation is currently in development.  
Until then, test files (especially `test_shap_explainer.py`) demonstrate usage and structure.

---

## License

This project is licensed under the MIT License.

