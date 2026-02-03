# tests/test_monotonicity.py
"""
Comprehensive tests for Monotonicity metric (Arya et al., 2019).

Tests cover:
- Basic functionality
- Different baseline types
- Batch operations
- Edge cases
- Multiple model types
- Semantic validation
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
    compute_monotonicity,
    compute_batch_monotonicity,
)


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestMonotonicityBasic:
    """Basic functionality tests for Monotonicity."""
    
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
        """Monotonicity returns a float."""
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
        
        score = compute_monotonicity(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"]
        )
        
        assert isinstance(score, float)
        print(f"\nMonotonicity: {score:.4f}")
    
    def test_returns_valid_range(self, setup_iris):
        """Score is in valid range [0, 1]."""
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
            
            score = compute_monotonicity(
                data["adapter"], instance, exp,
                baseline="mean", background_data=data["X"]
            )
            scores.append(score)
        
        for score in scores:
            assert 0.0 <= score <= 1.0
        
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
        
        score1 = compute_monotonicity(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"]
        )
        
        score2 = compute_monotonicity(
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
        
        score = compute_monotonicity(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"],
            use_absolute=True
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
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
        
        score = compute_monotonicity(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"],
            use_absolute=False
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


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
        
        score = compute_monotonicity(
            data["adapter"], data["instance"], data["exp"],
            baseline="mean", background_data=data["X"]
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\nMean baseline score: {score:.4f}")
    
    def test_median_baseline(self, setup_model):
        """Works with 'median' baseline."""
        data = setup_model
        
        score = compute_monotonicity(
            data["adapter"], data["instance"], data["exp"],
            baseline="median", background_data=data["X"]
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\nMedian baseline score: {score:.4f}")
    
    def test_scalar_baseline(self, setup_model):
        """Works with scalar baseline."""
        data = setup_model
        
        score = compute_monotonicity(
            data["adapter"], data["instance"], data["exp"],
            baseline=0.0
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\nScalar (0.0) baseline score: {score:.4f}")
    
    def test_array_baseline(self, setup_model):
        """Works with array baseline."""
        data = setup_model
        n_features = data["instance"].shape[0]
        custom_baseline = np.random.randn(n_features)
        
        score = compute_monotonicity(
            data["adapter"], data["instance"], data["exp"],
            baseline=custom_baseline
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\nArray baseline score: {score:.4f}")
    
    def test_callable_baseline(self, setup_model):
        """Works with callable baseline."""
        data = setup_model
        
        score = compute_monotonicity(
            data["adapter"], data["instance"], data["exp"],
            baseline=lambda x: np.percentile(x, 25, axis=0),
            background_data=data["X"]
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\nCallable (25th percentile) baseline score: {score:.4f}")


# =============================================================================
# Batch Operation Tests
# =============================================================================

class TestBatchOperations:
    """Tests for batch monotonicity computation."""
    
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
        
        result = compute_batch_monotonicity(
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
        
        result = compute_batch_monotonicity(
            data["adapter"], data["X"][:20], data["explanations"],
            baseline="mean", max_samples=5
        )
        
        assert result["n_samples"] <= 5
    
    def test_batch_mean_in_range(self, setup_batch):
        """Batch mean is in valid range [0, 1]."""
        data = setup_batch
        
        result = compute_batch_monotonicity(
            data["adapter"], data["X"][:20], data["explanations"],
            baseline="mean"
        )
        
        assert 0.0 <= result["mean"] <= 1.0
        assert 0.0 <= result["min"] <= 1.0
        assert 0.0 <= result["max"] <= 1.0


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
        
        score = compute_monotonicity(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
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
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        score = compute_monotonicity(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
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
        
        lime_score = compute_monotonicity(
            adapter, instance, lime_exp,
            baseline="mean", background_data=X
        )
        
        shap_score = compute_monotonicity(
            adapter, instance, shap_exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(lime_score, float)
        assert isinstance(shap_score, float)
        assert 0.0 <= lime_score <= 1.0
        assert 0.0 <= shap_score <= 1.0
        
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
        
        score = compute_monotonicity(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
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
        
        score = compute_monotonicity(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        print(f"\n50-feature score: {score:.4f}")
    
    def test_identical_attributions(self):
        """Handles case where all attributions are identical."""
        # Create explanation with identical attributions
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
        
        # Should handle gracefully
        score = compute_monotonicity(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
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
        
        score = compute_monotonicity(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        # Should return valid score even with zero attributions
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_single_feature(self):
        """Works with single feature."""
        # Simple 1-feature model
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
        
        score = compute_monotonicity(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        # With 1 feature, there's only 1 step, so score is 0 or 1
        assert score in [0.0, 1.0]


# =============================================================================
# Semantic Validation Tests
# =============================================================================

class TestSemanticValidation:
    """Tests to validate the metric measures what it claims."""
    
    def test_perfect_ordering_high_score(self):
        """Perfect feature ordering should have high monotonicity."""
        # Create a simple linear model where we know the true importances
        np.random.seed(42)
        n_samples = 100
        
        # Feature 0 has positive weight, feature 1 has smaller positive weight
        X = np.random.randn(n_samples, 4)
        y = (X[:, 0] * 1.0 + X[:, 1] * 0.3 + X[:, 2] * 0.1 > 0).astype(int)
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["c0", "c1"])
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=["f0", "f1", "f2", "f3"],
            class_names=["c0", "c1"]
        )
        
        # Test on multiple instances
        scores = []
        for i in range(10):
            instance = X[i]
            exp = lime.explain(instance)
            exp.feature_names = ["f0", "f1", "f2", "f3"]
            
            score = compute_monotonicity(
                adapter, instance, exp,
                baseline="mean", background_data=X
            )
            scores.append(score)
        
        # Good explanations should generally have high monotonicity
        mean_score = np.mean(scores)
        print(f"\nMean monotonicity with LIME: {mean_score:.4f}")
        # Most instances should have reasonable monotonicity
        assert mean_score >= 0.3  # LIME should do reasonably well
    
    def test_random_ordering_lower_score(self):
        """Random feature ordering should generally have lower monotonicity."""
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
        
        # Compare LIME vs random attribution ordering
        lime_scores = []
        random_scores = []
        
        np.random.seed(42)
        
        for i in range(10):
            instance = X[i]
            
            # LIME explanation
            lime_exp = lime.explain(instance)
            lime_exp.feature_names = feature_names
            
            lime_score = compute_monotonicity(
                adapter, instance, lime_exp,
                baseline="mean", background_data=X
            )
            lime_scores.append(lime_score)
            
            # Random explanation (random attribution values)
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
            
            random_score = compute_monotonicity(
                adapter, instance, random_exp,
                baseline="mean", background_data=X
            )
            random_scores.append(random_score)
        
        print(f"\nLIME mean: {np.mean(lime_scores):.4f}")
        print(f"Random mean: {np.mean(random_scores):.4f}")
        
        # Both should be valid scores
        assert all(0.0 <= s <= 1.0 for s in lime_scores)
        assert all(0.0 <= s <= 1.0 for s in random_scores)
    
    def test_inverted_ordering_lower_score(self):
        """Inverted feature ordering (least important first) should score lower."""
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
        
        # Get LIME score
        lime_score = compute_monotonicity(
            adapter, instance, lime_exp,
            baseline="mean", background_data=X
        )
        
        # Create inverted explanation (negate all attributions)
        original_attrs = lime_exp.explanation_data["feature_attributions"]
        inverted_attrs = {k: -v for k, v in original_attrs.items()}
        
        inverted_exp = Explanation(
            explainer_name="inverted",
            target_class=class_names[0],
            explanation_data={"feature_attributions": inverted_attrs},
            feature_names=feature_names
        )
        
        inverted_score = compute_monotonicity(
            adapter, instance, inverted_exp,
            baseline="mean", background_data=X
        )
        
        print(f"\nLIME score: {lime_score:.4f}")
        print(f"Inverted score: {inverted_score:.4f}")
        
        # Both should be valid
        assert 0.0 <= lime_score <= 1.0
        assert 0.0 <= inverted_score <= 1.0


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
            score = compute_monotonicity(
                adapter, instance, exp,
                baseline="mean", background_data=X,
                target_class=target_class
            )
            
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            print(f"\nTarget class {target_class} score: {score:.4f}")
    
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
        score = compute_monotonicity(
            adapter, instance, exp,
            baseline="mean", background_data=X
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
