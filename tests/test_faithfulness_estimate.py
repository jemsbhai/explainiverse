# tests/test_faithfulness_estimate.py
"""Contract tests for the local AIX360-style removal correlation proxy.

Tests cover:
- Basic functionality
- Different baseline types
- Subset size variations
- Batch operations
- Edge cases
- Multiple model types
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.faithfulness_extended import (
    _extract_attribution_array,
    compute_batch_faithfulness_estimate,
    compute_faithfulness_estimate,
)
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer

# =============================================================================
# Helper Function Tests
# =============================================================================


class TestExtractAttributionArray:
    """Tests for _extract_attribution_array helper."""

    def test_extract_with_feature_names(self):
        """Extraction works with feature names."""
        exp = Explanation(
            explainer_name="test",
            target_class="class_0",
            explanation_data={
                "feature_attributions": {
                    "f0": 0.5,
                    "f1": -0.3,
                    "f2": 0.8,
                }
            },
            feature_names=["f0", "f1", "f2"],
        )

        arr = _extract_attribution_array(exp, 3)

        assert len(arr) == 3
        assert arr[0] == 0.5
        assert arr[1] == -0.3
        assert arr[2] == 0.8

    def test_extract_with_index_pattern(self):
        """Extraction works with feature_N naming pattern."""
        exp = Explanation(
            explainer_name="test",
            target_class="class_0",
            explanation_data={
                "feature_attributions": {
                    "feature_0": 0.1,
                    "feature_1": 0.2,
                    "feature_2": 0.3,
                }
            },
        )

        arr = _extract_attribution_array(exp, 3)

        assert len(arr) == 3
        assert arr[0] == 0.1
        assert arr[1] == 0.2
        assert arr[2] == 0.3

    def test_extract_empty_attributions(self):
        """Raises error for empty attributions."""
        exp = Explanation(
            explainer_name="test",
            target_class="class_0",
            explanation_data={"feature_attributions": {}},
        )

        with pytest.raises(ValueError, match="No feature attributions"):
            _extract_attribution_array(exp, 3)


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestFaithfulnessEstimateBasic:
    """Basic functionality tests for Faithfulness Estimate."""

    @pytest.fixture
    def setup_iris(self):
        """Setup Iris dataset with LogisticRegression."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)

        return {
            "X": X,
            "y": y,
            "adapter": adapter,
            "feature_names": feature_names,
            "class_names": class_names,
        }

    def test_returns_float(self, setup_iris):
        """Faithfulness Estimate returns a float."""
        data = setup_iris

        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"],
        )

        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]

        score = compute_faithfulness_estimate(
            data["adapter"], instance, exp, baseline="mean", background_data=data["X"]
        )

        assert isinstance(score, float)
        print(f"\nFaithfulness Estimate: {score:.4f}")

    def test_returns_valid_correlation_range(self, setup_iris):
        """Score is in valid correlation range [-1, 1]."""
        data = setup_iris

        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"],
        )

        scores = []
        for i in range(10):
            instance = data["X"][i]
            exp = lime.explain(instance)
            exp.feature_names = data["feature_names"]

            score = compute_faithfulness_estimate(
                data["adapter"], instance, exp, baseline="mean", background_data=data["X"]
            )
            scores.append(score)

        for score in scores:
            assert -1.0 <= score <= 1.0 or score == 0.0

        print(f"\nScores range: [{min(scores):.4f}, {max(scores):.4f}]")

    def test_deterministic_with_seed(self, setup_iris):
        """Results are deterministic with seed (for subset_size > 1)."""
        data = setup_iris

        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"],
        )

        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]

        score1 = compute_faithfulness_estimate(
            data["adapter"],
            instance,
            exp,
            baseline="mean",
            background_data=data["X"],
            subset_size=2,
            n_subsets=50,
            seed=42,
        )

        score2 = compute_faithfulness_estimate(
            data["adapter"],
            instance,
            exp,
            baseline="mean",
            background_data=data["X"],
            subset_size=2,
            n_subsets=50,
            seed=42,
        )

        assert score1 == score2

    def test_single_feature_mode(self, setup_iris):
        """Single feature perturbation mode works (subset_size=1)."""
        data = setup_iris

        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"],
        )

        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]

        score = compute_faithfulness_estimate(
            data["adapter"],
            instance,
            exp,
            baseline="mean",
            background_data=data["X"],
            subset_size=1,  # Explicitly single feature
        )

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0 or score == 0.0

    def test_subset_mode(self, setup_iris):
        """Random subset perturbation mode works (subset_size > 1)."""
        data = setup_iris

        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"],
        )

        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]

        score = compute_faithfulness_estimate(
            data["adapter"],
            instance,
            exp,
            baseline="mean",
            background_data=data["X"],
            subset_size=2,
            n_subsets=100,
        )

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0 or score == 0.0


# =============================================================================
# Baseline Type Tests
# =============================================================================


class TestBaselineTypes:
    """Tests for different baseline configurations."""

    @pytest.fixture
    def setup_model(self):
        """Setup synthetic data and model."""
        X, y = make_classification(
            n_samples=100, n_features=8, n_informative=4, n_classes=2, random_state=42
        )
        class_names = ["class_0", "class_1"]
        feature_names = [f"feat_{i}" for i in range(8)]

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter, training_data=X, feature_names=feature_names, class_names=class_names
        )

        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names

        return {"X": X, "adapter": adapter, "instance": instance, "exp": exp}

    def test_mean_baseline(self, setup_model):
        """Works with 'mean' baseline."""
        data = setup_model

        score = compute_faithfulness_estimate(
            data["adapter"],
            data["instance"],
            data["exp"],
            baseline="mean",
            background_data=data["X"],
        )

        assert isinstance(score, float)
        print(f"\nMean baseline score: {score:.4f}")

    def test_median_baseline(self, setup_model):
        """Works with 'median' baseline."""
        data = setup_model

        score = compute_faithfulness_estimate(
            data["adapter"],
            data["instance"],
            data["exp"],
            baseline="median",
            background_data=data["X"],
        )

        assert isinstance(score, float)
        print(f"\nMedian baseline score: {score:.4f}")

    def test_scalar_baseline(self, setup_model):
        """Works with scalar baseline."""
        data = setup_model

        score = compute_faithfulness_estimate(
            data["adapter"], data["instance"], data["exp"], baseline=0.0
        )

        assert isinstance(score, float)
        print(f"\nScalar (0.0) baseline score: {score:.4f}")

    def test_array_baseline(self, setup_model):
        """Works with array baseline."""
        data = setup_model
        n_features = data["instance"].shape[0]
        custom_baseline = np.random.randn(n_features)

        score = compute_faithfulness_estimate(
            data["adapter"], data["instance"], data["exp"], baseline=custom_baseline
        )

        assert isinstance(score, float)
        print(f"\nArray baseline score: {score:.4f}")

    def test_callable_baseline(self, setup_model):
        """Works with callable baseline."""
        data = setup_model

        score = compute_faithfulness_estimate(
            data["adapter"],
            data["instance"],
            data["exp"],
            baseline=lambda x: np.percentile(x, 25, axis=0),
            background_data=data["X"],
        )

        assert isinstance(score, float)
        print(f"\nCallable (25th percentile) baseline score: {score:.4f}")


# =============================================================================
# Batch Operation Tests
# =============================================================================


class TestBatchOperations:
    """Tests for batch faithfulness estimate computation."""

    @pytest.fixture
    def setup_batch(self):
        """Setup data for batch testing."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter, training_data=X, feature_names=feature_names, class_names=class_names
        )

        # Generate explanations for multiple samples
        explanations = []
        for i in range(20):
            exp = lime.explain(X[i])
            exp.feature_names = feature_names
            explanations.append(exp)

        return {"X": X, "adapter": adapter, "explanations": explanations}

    def test_batch_returns_dict(self, setup_batch):
        """Batch computation returns dictionary with expected keys."""
        data = setup_batch

        result = compute_batch_faithfulness_estimate(
            data["adapter"],
            data["X"][:20],
            data["explanations"],
            baseline="mean",
            background_data=data["X"][20:],
        )

        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "n_samples" in result

        print(f"\nBatch result: {result}")

    def test_batch_max_samples(self, setup_batch):
        """max_samples parameter limits evaluation."""
        data = setup_batch

        result = compute_batch_faithfulness_estimate(
            data["adapter"],
            data["X"][:20],
            data["explanations"],
            baseline="mean",
            max_samples=5,
            background_data=data["X"][20:],
        )

        assert result["n_samples"] <= 5

    def test_batch_mean_in_range(self, setup_batch):
        """Batch mean is in valid correlation range."""
        data = setup_batch

        result = compute_batch_faithfulness_estimate(
            data["adapter"],
            data["X"][:20],
            data["explanations"],
            baseline="mean",
            background_data=data["X"][20:],
        )

        assert -1.0 <= result["mean"] <= 1.0


# =============================================================================
# Multiple Model Tests
# =============================================================================


class TestMultipleModels:
    """Tests across different model types."""

    def test_random_forest(self):
        """Works with Random Forest model."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter, training_data=X, feature_names=feature_names, class_names=class_names
        )

        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names

        score = compute_faithfulness_estimate(
            adapter, instance, exp, baseline="mean", background_data=X
        )

        assert isinstance(score, float)
        print(f"\nRandom Forest score: {score:.4f}")

    def test_gradient_boosting(self):
        """Works with Gradient Boosting model."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)

        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter, training_data=X, feature_names=feature_names, class_names=class_names
        )

        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names

        score = compute_faithfulness_estimate(
            adapter, instance, exp, baseline="mean", background_data=X
        )

        assert isinstance(score, float)
        print(f"\nGradient Boosting score: {score:.4f}")


# =============================================================================
# Explainer Comparison Tests
# =============================================================================


class TestExplainerComparison:
    """Tests comparing LIME vs SHAP explanations."""

    def test_lime_vs_shap(self):
        """Both LIME and SHAP produce valid scores."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter, training_data=X, feature_names=feature_names, class_names=class_names
        )

        shap = ShapExplainer(
            model=adapter,
            background_data=X[:30],
            feature_names=feature_names,
            class_names=class_names,
        )

        instance = X[0]

        lime_exp = lime.explain(instance)
        lime_exp.feature_names = feature_names

        shap_exp = shap.explain(instance)
        shap_exp.feature_names = feature_names

        lime_score = compute_faithfulness_estimate(
            adapter, instance, lime_exp, baseline="mean", background_data=X
        )

        shap_score = compute_faithfulness_estimate(
            adapter, instance, shap_exp, baseline="mean", background_data=X
        )

        assert isinstance(lime_score, float)
        assert isinstance(shap_score, float)

        print(f"\nLIME score: {lime_score:.4f}")
        print(f"SHAP score: {shap_score:.4f}")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_few_features(self):
        """Works with few features."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_informative=2, n_redundant=0, n_classes=2, random_state=42
        )
        class_names = ["class_0", "class_1"]
        feature_names = ["f0", "f1", "f2", "f3", "f4"]

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter, training_data=X, feature_names=feature_names, class_names=class_names
        )

        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names

        score = compute_faithfulness_estimate(
            adapter, instance, exp, baseline="mean", background_data=X
        )

        assert isinstance(score, float)

    def test_many_features(self):
        """Works with many features."""
        X, y = make_classification(
            n_samples=100, n_features=50, n_informative=20, n_classes=2, random_state=42
        )
        class_names = ["class_0", "class_1"]
        feature_names = [f"feat_{i}" for i in range(50)]

        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter, training_data=X, feature_names=feature_names, class_names=class_names
        )

        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names

        score = compute_faithfulness_estimate(
            adapter, instance, exp, baseline="mean", background_data=X
        )

        assert isinstance(score, float)
        print(f"\n50-feature score: {score:.4f}")

    def test_identical_attributions(self):
        """Handles case where all attributions are identical."""
        # Create explanation with identical attributions
        exp = Explanation(
            explainer_name="test",
            target_class="class_0",
            explanation_data={"feature_attributions": {"f0": 0.5, "f1": 0.5, "f2": 0.5, "f3": 0.5}},
            feature_names=["f0", "f1", "f2", "f3"],
        )

        X, y = make_classification(
            n_samples=50, n_features=4, n_informative=2, n_redundant=0, random_state=42
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["c0", "c1"])

        instance = X[0]

        with pytest.raises(ValueError, match="undefined for a constant input"):
            compute_faithfulness_estimate(
                adapter, instance, exp, baseline="mean", background_data=X
            )

    def test_zero_attributions(self):
        """Handles case where attributions are all zero."""
        exp = Explanation(
            explainer_name="test",
            target_class="class_0",
            explanation_data={"feature_attributions": {"f0": 0.0, "f1": 0.0, "f2": 0.0, "f3": 0.0}},
            feature_names=["f0", "f1", "f2", "f3"],
        )

        X, y = make_classification(
            n_samples=50, n_features=4, n_informative=2, n_redundant=0, random_state=42
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["c0", "c1"])

        instance = X[0]

        with pytest.raises(ValueError, match="undefined for a constant input"):
            compute_faithfulness_estimate(
                adapter, instance, exp, baseline="mean", background_data=X
            )


# =============================================================================
# Semantic Validation Tests
# =============================================================================


class TestSemanticValidation:
    """Integration smoke tests without explainer-quality claims."""

    def test_coefficient_ordering_returns_float(self):
        """A hand-selected attribution ordering returns a float."""
        # Create a simple fitted linear classifier.
        np.random.seed(42)
        n_samples = 100

        # Feature 0 has weight 1.0, feature 1 has weight 0.5, others are noise
        X = np.random.randn(n_samples, 4)
        y = (X[:, 0] * 1.0 + X[:, 1] * 0.5 > 0).astype(int)

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["c0", "c1"])

        # Supply one deterministic attribution ordering.
        exp = Explanation(
            explainer_name="test",
            target_class="c0",
            explanation_data={
                "feature_attributions": {
                    "f0": 1.0,
                    "f1": 0.5,
                    "f2": 0.1,
                    "f3": 0.1,
                }
            },
            feature_names=["f0", "f1", "f2", "f3"],
        )

        instance = X[0]

        score = compute_faithfulness_estimate(
            adapter, instance, exp, baseline="mean", background_data=X
        )

        assert isinstance(score, float)

    def test_lime_and_seeded_attributions_match_single_feature_pcc_equation(self):
        """Both attribution vectors obey the exact single-feature PCC equation."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)

        lime = LimeExplainer(
            model=adapter, training_data=X, feature_names=feature_names, class_names=class_names
        )

        instance = X[0]
        lime_exp = lime.explain(instance)
        lime_exp.feature_names = feature_names

        # Create a second seeded attribution vector; no ordering claim is made.
        np.random.seed(123)
        random_exp = Explanation(
            explainer_name="random",
            target_class=class_names[0],
            explanation_data={
                "feature_attributions": {fn: np.random.randn() for fn in feature_names}
            },
            feature_names=feature_names,
        )

        baseline = np.mean(X, axis=0)
        perturbed = np.repeat(instance.reshape(1, -1), len(feature_names), axis=0)
        for feature_index in range(len(feature_names)):
            perturbed[feature_index, feature_index] = baseline[feature_index]
        original_output = adapter.predict(instance.reshape(1, -1))[0, 0]
        prediction_drops = original_output - adapter.predict(perturbed)[:, 0]

        for explanation in [lime_exp, random_exp]:
            score = compute_faithfulness_estimate(
                adapter,
                instance,
                explanation,
                baseline="mean",
                background_data=X,
            )
            attribution_map = explanation.explanation_data["feature_attributions"]
            attribution_values = np.array([attribution_map[name] for name in feature_names])
            expected = np.corrcoef(attribution_values, prediction_drops)[0, 1]
            assert score == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
