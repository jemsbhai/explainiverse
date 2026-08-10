"""Regression tests for fixed-output prediction changes.

When a perturbation flips the predicted class, ``compute_prediction_change``
must compare the original class column at both inputs rather than comparing two
independently selected maximum probabilities.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.evaluation._utils import compute_prediction_change, get_prediction_value


@pytest.fixture
def iris_gbc_setup():
    """Return a deterministic multiclass integration fixture."""
    iris = load_iris()
    X_train, X_test, y_train, _ = train_test_split(
        iris.data,
        iris.target,
        test_size=0.3,
        random_state=42,
        stratify=iris.target,
    )
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())
    return X_test, adapter


def test_get_prediction_value_selects_the_original_argmax_output(iris_gbc_setup):
    """Implicit classification selection returns the current argmax column."""
    X_test, adapter = iris_gbc_setup

    for instance in X_test[:10]:
        outputs = adapter.predict(instance.reshape(1, -1))[0]
        expected = outputs[int(np.argmax(outputs))]
        assert get_prediction_value(adapter, instance) == pytest.approx(expected)


def test_prediction_change_tracks_original_output_across_guaranteed_class_flip():
    """A deterministic class flip retains the original output column."""

    class FlipClassifier:
        _estimator_type = "classifier"
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            values = np.asarray(X, dtype=float)[:, 0]
            return np.array([[0.95, 0.05] if value > 0 else [0.10, 0.90] for value in values])

    model = FlipClassifier()
    original = np.array([1.0])
    perturbed = np.array([-1.0])
    original_outputs = model.predict_proba(original.reshape(1, -1))[0]
    perturbed_outputs = model.predict_proba(perturbed.reshape(1, -1))[0]

    assert np.argmax(original_outputs) == 0
    assert np.argmax(perturbed_outputs) == 1
    assert compute_prediction_change(model, original, perturbed) == pytest.approx(
        abs(original_outputs[0] - perturbed_outputs[0])
    )
    assert compute_prediction_change(model, original, perturbed) == pytest.approx(0.85)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
