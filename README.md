# Explainiverse

Explainiverse is a unified, extensible, and testable Python framework for explainable AI (XAI).  
It provides a consistent API and support for post-hoc explainers like LIME and SHAP, model adapters, and rigorous evaluation strategies.

---

## Features

- Standardized Explainer interface (`BaseExplainer`)
- Support for classification, regression, and multi-class models
- Integrated explainers:
  - LIME (Local surrogate models)
  - SHAP (KernelExplainer with per-class and global support)
- Adapter layer for scikit-learn models
- Explanation object with structured output and future extensibility for `.plot()`
- Full unit test suite covering classification, regression, global/cohort SHAP, and adapter behavior

---

## Installation

This package will soon be available on PyPI.

For development use:

```bash
git clone https://github.com/jemsbhai/explainiverse.git
cd explainiverse
poetry install
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
```

---

## Documentation

Documentation is currently in development.  
Until then, test files (especially `test_shap_explainer.py`) demonstrate usage and structure.

---

## License

This project is licensed under the MIT License.
