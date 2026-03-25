"""
Regression test: get_prediction_value and compute_prediction_change
must track the originally predicted class, not max probability.

The np.max bug means that when removing important features causes the
model to flip to a DIFFERENT class with high confidence, PGI appears
small (because max prob barely changed), but it should be large
(because P(original class) dropped dramatically).
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    get_prediction_value,
    compute_prediction_change,
)
from explainiverse.evaluation.faithfulness import compute_pgi, compute_pgu


@pytest.fixture
def iris_gbc_setup():
    """Iris with GradientBoosting — strong enough to have high-confidence predictions."""
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)

    adapter = SklearnAdapter(model, class_names=class_names)

    return X_train, X_test, y_train, y_test, feature_names, class_names, model, adapter


class TestGetPredictionValue:
    """Tests for get_prediction_value correctness."""

    def test_returns_predicted_class_probability(self, iris_gbc_setup):
        """get_prediction_value must return P(predicted_class), not max(P).

        For the originally predicted class, these are the same. But
        the function must be consistent about WHICH class it tracks.
        """
        (X_train, X_test, y_train, y_test,
         feature_names, class_names, model, adapter) = iris_gbc_setup

        for i in range(10):
            instance = X_test[i]
            pred_val = get_prediction_value(adapter, instance)

            # Get full probability vector
            probs = adapter.predict(instance.reshape(1, -1))[0]
            predicted_class = np.argmax(probs)
            predicted_class_prob = probs[predicted_class]

            # get_prediction_value should return predicted class probability
            assert abs(pred_val - predicted_class_prob) < 1e-6, \
                f"Instance {i}: got {pred_val}, expected P(class {predicted_class}) = {predicted_class_prob}"


class TestComputePredictionChange:
    """Tests for compute_prediction_change correctness."""

    def test_detects_class_flip(self, iris_gbc_setup):
        """When perturbing features causes a class flip, the prediction
        change must be large — reflecting the drop in the ORIGINALLY
        predicted class probability, not the max probability.
        """
        (X_train, X_test, y_train, y_test,
         feature_names, class_names, model, adapter) = iris_gbc_setup

        # Find a high-confidence instance
        for i in range(len(X_test)):
            probs = adapter.predict(X_test[i:i + 1])[0]
            if np.max(probs) > 0.9:
                instance = X_test[i]
                original_class = np.argmax(probs)
                break

        # Create a heavily perturbed version using mean baseline
        baseline = np.mean(X_train, axis=0)
        perturbed = baseline.copy()  # Replace ALL features

        perturbed_probs = adapter.predict(perturbed.reshape(1, -1))[0]
        perturbed_class = np.argmax(perturbed_probs)

        change = compute_prediction_change(adapter, instance, perturbed)

        # If the class flipped and original had high confidence,
        # the change should be substantial
        original_class_prob_original = probs[original_class]
        original_class_prob_perturbed = perturbed_probs[original_class]

        expected_change = abs(original_class_prob_original - original_class_prob_perturbed)

        # The compute_prediction_change should reflect the original class drop
        # not just |max(original) - max(perturbed)|
        if original_class != perturbed_class:
            # Class flipped — change must be at least as large as the
            # drop in original class probability
            assert change >= expected_change * 0.5, \
                f"Class flipped ({original_class} -> {perturbed_class}) but " \
                f"change={change:.4f} is too small. " \
                f"P(orig_class) went from {original_class_prob_original:.4f} " \
                f"to {original_class_prob_perturbed:.4f} " \
                f"(expected_change={expected_change:.4f})"


class TestPGIvsPGU:
    """Tests that PGI > PGU for a good explanation — the fundamental
    sanity check that was failing due to the max-probability bug."""

    def test_shap_pgi_greater_than_pgu(self, iris_gbc_setup):
        """For a good explainer (SHAP), PGI should exceed PGU.

        Removing important features should change the prediction MORE
        than removing unimportant features.
        """
        (X_train, X_test, y_train, y_test,
         feature_names, class_names, model, adapter) = iris_gbc_setup

        from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer

        shap_explainer = ShapExplainer(
            model=adapter,
            background_data=X_train[:30],
            feature_names=feature_names,
            class_names=class_names
        )

        pgi_scores = []
        pgu_scores = []

        for i in range(10):
            instance = X_test[i]
            explanation = shap_explainer.explain(instance)

            pgi = compute_pgi(
                adapter, instance, explanation, k=0.5,
                baseline="mean", background_data=X_train
            )
            pgu = compute_pgu(
                adapter, instance, explanation, k=0.5,
                baseline="mean", background_data=X_train
            )
            pgi_scores.append(pgi)
            pgu_scores.append(pgu)

        mean_pgi = np.mean(pgi_scores)
        mean_pgu = np.mean(pgu_scores)

        assert mean_pgi > mean_pgu, \
            f"SHAP PGI ({mean_pgi:.4f}) should be > PGU ({mean_pgu:.4f}) " \
            f"for a good explainer. If PGU >= PGI, the prediction change " \
            f"function is not tracking the correct class."

    def test_shap_beats_random_on_pgi(self, iris_gbc_setup):
        """SHAP must have higher PGI than random attributions."""
        (X_train, X_test, y_train, y_test,
         feature_names, class_names, model, adapter) = iris_gbc_setup

        from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer

        shap_explainer = ShapExplainer(
            model=adapter,
            background_data=X_train[:30],
            feature_names=feature_names,
            class_names=class_names
        )

        shap_pgi_scores = []
        random_pgi_scores = []

        np.random.seed(42)
        for i in range(10):
            instance = X_test[i]

            # SHAP explanation
            shap_exp = shap_explainer.explain(instance)
            shap_pgi = compute_pgi(
                adapter, instance, shap_exp, k=0.5,
                baseline="mean", background_data=X_train
            )
            shap_pgi_scores.append(shap_pgi)

            # Random explanation
            random_attrs = np.random.randn(len(feature_names))
            random_exp = Explanation(
                explainer_name="Random",
                target_class="N/A",
                explanation_data={
                    "feature_attributions": dict(zip(feature_names, random_attrs))
                },
                feature_names=feature_names
            )
            random_pgi = compute_pgi(
                adapter, instance, random_exp, k=0.5,
                baseline="mean", background_data=X_train
            )
            random_pgi_scores.append(random_pgi)

        mean_shap = np.mean(shap_pgi_scores)
        mean_random = np.mean(random_pgi_scores)

        assert mean_shap > mean_random, \
            f"SHAP PGI ({mean_shap:.4f}) must beat Random PGI ({mean_random:.4f}). " \
            f"If Random wins, get_prediction_value is using max-probability " \
            f"instead of predicted-class probability."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
