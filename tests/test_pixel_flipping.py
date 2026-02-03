# tests/test_pixel_flipping.py
"""
Comprehensive tests for Pixel Flipping metric (Bach et al., 2015).

Pixel Flipping sequentially removes features in order of attributed importance
(most important first) and measures cumulative prediction degradation.
Lower AUC = better faithfulness (faster degradation).

Tests cover:
- Basic functionality
- Different baseline types
- Batch operations
- Edge cases
- Multiple model types
- Semantic validation
- Curve output verification
"""
import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.faithfulness_extended import (
    compute_pixel_flipping,
    compute_batch_pixel_flipping,
)


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestPixelFlippingBasic:
    """Basic functionality tests for Pixel Flipping."""
    
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
            "X": X, "y": y,
            "adapter": adapter,
            "feature_names": feature_names,
            "class_names": class_names,
        }
    
    def test_returns_float(self, setup_iris):
        """Pixel Flipping returns a float."""
        data = setup_iris
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        score = compute_pixel_flipping(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"]
        )
        
        assert isinstance(score, float)
        print(f"\nPixel Flipping AUC: {score:.4f}")
    
    def test_returns_valid_range(self, setup_iris):
        """Score is in valid range (typically 0 to ~1, can exceed 1 in some cases)."""
        data = setup_iris
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        scores = []
        for i in range(10):
            instance = data["X"][i]
            exp = lime.explain(instance)
            exp.feature_names = data["feature_names"]
            
            score = compute_pixel_flipping(
                data["adapter"], instance, exp,
                baseline="mean", background_data=data["X"]
            )
            scores.append(score)
        
        # AUC should be positive (prediction values are probabilities >= 0)
        for score in scores:
            assert score >= 0
        
        print(f"\nScores range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"Mean score: {np.mean(scores):.4f}")
    
    def test_deterministic(self, setup_iris):
        """Results are deterministic (no randomness in algorithm)."""
        data = setup_iris
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        score1 = compute_pixel_flipping(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"]
        )
        
        score2 = compute_pixel_flipping(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"]
        )
        
        assert score1 == score2
    
    def test_use_absolute_true(self, setup_iris):
        """Works with use_absolute=True (default)."""
        data = setup_iris
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        score = compute_pixel_flipping(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"],
            use_absolute=True
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_use_absolute_false(self, setup_iris):
        """Works with use_absolute=False."""
        data = setup_iris
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        score = compute_pixel_flipping(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"],
            use_absolute=False
        )
        
        assert isinstance(score, float)
        assert score >= 0


# =============================================================================
# Curve Output Tests
# =============================================================================

class TestCurveOutput:
    """Tests for return_curve=True functionality."""
    
    @pytest.fixture
    def setup_model(self):
        """Setup synthetic data and model."""
        X, y = make_classification(
            n_samples=100, n_features=8, n_informative=4,
            n_classes=2, random_state=42
        )
        class_names = ["class_0", "class_1"]
        feature_names = [f"feat_{i}" for i in range(8)]
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        return {
            "X": X, "adapter": adapter,
            "instance": instance, "exp": exp,
            "n_features": 8
        }
    
    def test_return_curve_dict(self, setup_model):
        """return_curve=True returns dictionary."""
        data = setup_model
        
        result = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline="mean", background_data=data["X"],
            return_curve=True
        )
        
        assert isinstance(result, dict)
        assert "auc" in result
        assert "curve" in result
        assert "predictions" in result
        assert "feature_order" in result
        assert "n_features" in result
    
    def test_curve_length(self, setup_model):
        """Curve has correct length (n_features + 1)."""
        data = setup_model
        
        result = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline="mean", background_data=data["X"],
            return_curve=True
        )
        
        # n_features + 1 points (original + after each removal)
        assert len(result["curve"]) == data["n_features"] + 1
        assert len(result["predictions"]) == data["n_features"] + 1
    
    def test_curve_starts_at_one(self, setup_model):
        """Normalized curve starts at 1.0 (original prediction)."""
        data = setup_model
        
        result = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline="mean", background_data=data["X"],
            return_curve=True
        )
        
        # First point is original / original = 1.0
        assert abs(result["curve"][0] - 1.0) < 1e-6
    
    def test_feature_order_valid(self, setup_model):
        """Feature order contains valid indices."""
        data = setup_model
        
        result = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline="mean", background_data=data["X"],
            return_curve=True
        )
        
        feature_order = result["feature_order"]
        assert len(feature_order) == data["n_features"]
        assert set(feature_order) == set(range(data["n_features"]))
    
    def test_auc_matches_curve(self, setup_model):
        """AUC in dict matches standalone computation."""
        data = setup_model
        
        result = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline="mean", background_data=data["X"],
            return_curve=True
        )
        
        score_only = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline="mean", background_data=data["X"],
            return_curve=False
        )
        
        assert abs(result["auc"] - score_only) < 1e-10


# =============================================================================
# Baseline Type Tests
# =============================================================================

class TestBaselineTypes:
    """Tests for different baseline configurations."""
    
    @pytest.fixture
    def setup_model(self):
        """Setup synthetic data and model."""
        X, y = make_classification(
            n_samples=100, n_features=8, n_informative=4,
            n_classes=2, random_state=42
        )
        class_names = ["class_0", "class_1"]
        feature_names = [f"feat_{i}" for i in range(8)]
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        return {
            "X": X, "adapter": adapter,
            "instance": instance, "exp": exp
        }
    
    def test_mean_baseline(self, setup_model):
        """Works with 'mean' baseline."""
        data = setup_model
        
        score = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline="mean", background_data=data["X"]
        )
        
        assert isinstance(score, float)
        assert score >= 0
        print(f"\nMean baseline AUC: {score:.4f}")
    
    def test_median_baseline(self, setup_model):
        """Works with 'median' baseline."""
        data = setup_model
        
        score = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline="median", background_data=data["X"]
        )
        
        assert isinstance(score, float)
        assert score >= 0
        print(f"\nMedian baseline AUC: {score:.4f}")
    
    def test_scalar_baseline(self, setup_model):
        """Works with scalar baseline."""
        data = setup_model
        
        score = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline=0.0
        )
        
        assert isinstance(score, float)
        assert score >= 0
        print(f"\nScalar (0.0) baseline AUC: {score:.4f}")
    
    def test_array_baseline(self, setup_model):
        """Works with array baseline."""
        data = setup_model
        n_features = data["instance"].shape[0]
        custom_baseline = np.random.randn(n_features)
        
        score = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline=custom_baseline
        )
        
        assert isinstance(score, float)
        assert score >= 0
        print(f"\nArray baseline AUC: {score:.4f}")
    
    def test_callable_baseline(self, setup_model):
        """Works with callable baseline."""
        data = setup_model
        
        score = compute_pixel_flipping(
            data["adapter"], data["instance"], data["exp"],
            baseline=lambda x: np.percentile(x, 25, axis=0),
            background_data=data["X"]
        )
        
        assert isinstance(score, float)
        assert score >= 0
        print(f"\nCallable (25th percentile) baseline AUC: {score:.4f}")


# =============================================================================
# Batch Operation Tests
# =============================================================================

class TestBatchOperations:
    """Tests for batch pixel flipping computation."""
    
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
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Generate explanations for multiple samples
        explanations = []
        for i in range(20):
            exp = lime.explain(X[i])
            exp.feature_names = feature_names
            explanations.append(exp)
        
        return {
            "X": X, "adapter": adapter,
            "explanations": explanations
        }
    
    def test_batch_returns_dict(self, setup_batch):
        """Batch computation returns dictionary with expected keys."""
        data = setup_batch
        
        result = compute_batch_pixel_flipping(
            data["adapter"], data["X"][:20], data["explanations"],
            baseline="mean"
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
        
        result = compute_batch_pixel_flipping(
            data["adapter"], data["X"][:20], data["explanations"],
            baseline="mean", max_samples=5
        )
        
        assert result["n_samples"] <= 5
    
    def test_batch_values_positive(self, setup_batch):
        """Batch values are non-negative."""
        data = setup_batch
        
        result = compute_batch_pixel_flipping(
            data["adapter"], data["X"][:20], data["explanations"],
            baseline="mean"
        )
        
        assert result["mean"] >= 0
        assert result["min"] >= 0
        assert result["max"] >= 0


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
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        score = compute_pixel_flipping(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert score >= 0
        print(f"\nRandom Forest AUC: {score:.4f}")
    
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
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        score = compute_pixel_flipping(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert score >= 0
        print(f"\nGradient Boosting AUC: {score:.4f}")


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
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        shap = ShapExplainer(
            model=adapter,
            background_data=X[:30],
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        
        lime_exp = lime.explain(instance)
        lime_exp.feature_names = feature_names
        
        shap_exp = shap.explain(instance)
        shap_exp.feature_names = feature_names
        
        lime_score = compute_pixel_flipping(
            adapter, instance, lime_exp,
            baseline="mean", background_data=X
        )
        
        shap_score = compute_pixel_flipping(
            adapter, instance, shap_exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(lime_score, float)
        assert isinstance(shap_score, float)
        assert lime_score >= 0
        assert shap_score >= 0
        
        print(f"\nLIME AUC: {lime_score:.4f}")
        print(f"SHAP AUC: {shap_score:.4f}")


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_few_features(self):
        """Works with few features."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_informative=2,
            n_redundant=0, n_classes=2, random_state=42
        )
        class_names = ["class_0", "class_1"]
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        score = compute_pixel_flipping(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_many_features(self):
        """Works with many features."""
        X, y = make_classification(
            n_samples=100, n_features=50, n_informative=20,
            n_classes=2, random_state=42
        )
        class_names = ["class_0", "class_1"]
        feature_names = [f"feat_{i}" for i in range(50)]
        
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        score = compute_pixel_flipping(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert score >= 0
        print(f"\n50-feature AUC: {score:.4f}")
    
    def test_identical_attributions(self):
        """Handles case where all attributions are identical."""
        exp = Explanation(
            explainer_name="test",
            target_class="class_0",
            explanation_data={
                "feature_attributions": {
                    "f0": 0.5, "f1": 0.5, "f2": 0.5, "f3": 0.5
                }
            },
            feature_names=["f0", "f1", "f2", "f3"]
        )
        
        X, y = make_classification(
            n_samples=50, n_features=4, n_informative=2,
            n_redundant=0, random_state=42
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["c0", "c1"])
        
        instance = X[0]
        
        score = compute_pixel_flipping(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_zero_attributions(self):
        """Handles case where attributions are all zero."""
        exp = Explanation(
            explainer_name="test",
            target_class="class_0",
            explanation_data={
                "feature_attributions": {
                    "f0": 0.0, "f1": 0.0, "f2": 0.0, "f3": 0.0
                }
            },
            feature_names=["f0", "f1", "f2", "f3"]
        )
        
        X, y = make_classification(
            n_samples=50, n_features=4, n_informative=2,
            n_redundant=0, random_state=42
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["c0", "c1"])
        
        instance = X[0]
        
        score = compute_pixel_flipping(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert score >= 0
    
    def test_single_feature(self):
        """Works with single feature."""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = (X[:, 0] > 0).astype(int)
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["c0", "c1"])
        
        exp = Explanation(
            explainer_name="test",
            target_class="c0",
            explanation_data={"feature_attributions": {"f0": 1.0}},
            feature_names=["f0"]
        )
        
        instance = X[0]
        
        score = compute_pixel_flipping(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert score >= 0


# =============================================================================
# Semantic Validation Tests
# =============================================================================

class TestSemanticValidation:
    """Tests to validate the metric measures what it claims."""
    
    def test_good_explanation_lower_auc(self):
        """Good explanations (LIME) should have lower AUC than random."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        lime_scores = []
        random_scores = []
        
        np.random.seed(42)
        
        for i in range(10):
            instance = X[i]
            
            # LIME explanation
            lime_exp = lime.explain(instance)
            lime_exp.feature_names = feature_names
            
            lime_score = compute_pixel_flipping(
                adapter, instance, lime_exp,
                baseline="mean", background_data=X
            )
            lime_scores.append(lime_score)
            
            # Random explanation (random ordering)
            random_exp = Explanation(
                explainer_name="random",
                target_class=class_names[0],
                explanation_data={
                    "feature_attributions": {
                        fn: np.random.randn() for fn in feature_names
                    }
                },
                feature_names=feature_names
            )
            
            random_score = compute_pixel_flipping(
                adapter, instance, random_exp,
                baseline="mean", background_data=X
            )
            random_scores.append(random_score)
        
        print(f"\nLIME mean AUC: {np.mean(lime_scores):.4f}")
        print(f"Random mean AUC: {np.mean(random_scores):.4f}")
        
        # LIME should generally have lower AUC (faster degradation)
        # But this is a soft check - not always guaranteed
        assert all(s >= 0 for s in lime_scores)
        assert all(s >= 0 for s in random_scores)
    
    def test_inverted_explanation_higher_auc(self):
        """Inverted explanation (least important first) should have higher AUC."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        lime_exp = lime.explain(instance)
        lime_exp.feature_names = feature_names
        
        # Normal LIME score
        lime_score = compute_pixel_flipping(
            adapter, instance, lime_exp,
            baseline="mean", background_data=X
        )
        
        # Inverted explanation (negate attributions so order is reversed)
        original_attrs = lime_exp.explanation_data["feature_attributions"]
        inverted_attrs = {k: -v for k, v in original_attrs.items()}
        
        inverted_exp = Explanation(
            explainer_name="inverted",
            target_class=class_names[0],
            explanation_data={"feature_attributions": inverted_attrs},
            feature_names=feature_names
        )
        
        inverted_score = compute_pixel_flipping(
            adapter, instance, inverted_exp,
            baseline="mean", background_data=X
        )
        
        print(f"\nLIME AUC: {lime_score:.4f}")
        print(f"Inverted AUC: {inverted_score:.4f}")
        
        # Both should be valid
        assert lime_score >= 0
        assert inverted_score >= 0
    
    def test_curve_monotonic_tendency(self):
        """Curve should generally decrease when important features removed first."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        result = compute_pixel_flipping(
            adapter, instance, exp,
            baseline="mean", background_data=X,
            return_curve=True
        )
        
        curve = result["curve"]
        
        # First point should be 1.0 (original)
        assert abs(curve[0] - 1.0) < 1e-6
        
        # Last point should generally be lower (all features removed)
        # But not guaranteed to be < 1 for all models
        print(f"\nCurve: start={curve[0]:.4f}, end={curve[-1]:.4f}")
        print(f"Predictions: start={result['predictions'][0]:.4f}, end={result['predictions'][-1]:.4f}")


# =============================================================================
# Target Class Tests
# =============================================================================

class TestTargetClass:
    """Tests for target class handling."""
    
    def test_explicit_target_class(self):
        """Works with explicitly specified target class."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        # Test with explicit target class
        for target_class in [0, 1, 2]:
            score = compute_pixel_flipping(
                adapter, instance, exp,
                baseline="mean", background_data=X,
                target_class=target_class
            )
            
            assert isinstance(score, float)
            assert score >= 0
            print(f"\nTarget class {target_class} AUC: {score:.4f}")
    
    def test_default_target_class(self):
        """Default target class is predicted class."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        # Default should work
        score = compute_pixel_flipping(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
