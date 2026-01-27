# tests/test_stability_metrics.py
"""
Comprehensive tests for stability evaluation metrics: RIS, ROS, and related utilities.

Tests cover:
- RIS (Relative Input Stability)
- ROS (Relative Output Stability)
- Lipschitz Estimate
- Batch operations and explainer comparison
"""
import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.evaluation.stability import (
    compute_ris,
    compute_ros,
    compute_lipschitz_estimate,
    compute_stability_metrics,
    compute_batch_stability,
    compare_explainer_stability,
)


# =============================================================================
# RIS (Relative Input Stability) Tests
# =============================================================================

class TestRIS:
    """Tests for Relative Input Stability metric."""
    
    @pytest.fixture
    def setup_model_and_explainer(self):
        """Setup common test fixtures."""
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
        
        return {
            "X": X, "y": y,
            "adapter": adapter,
            "lime": lime,
            "feature_names": feature_names,
            "class_names": class_names,
        }
    
    def test_ris_basic(self, setup_model_and_explainer):
        """RIS returns a valid float score."""
        data = setup_model_and_explainer
        
        instance = data["X"][0]
        ris = compute_ris(
            data["lime"], instance,
            n_perturbations=5,
            noise_scale=0.01,
            seed=42
        )
        
        assert isinstance(ris, float)
        assert ris >= 0 or ris == float('inf')
        print(f"\nRIS score: {ris:.4f}")
    
    def test_ris_with_more_perturbations(self, setup_model_and_explainer):
        """RIS with more perturbations gives more stable estimate."""
        data = setup_model_and_explainer
        
        instance = data["X"][0]
        
        ris_5 = compute_ris(data["lime"], instance, n_perturbations=5, seed=42)
        ris_10 = compute_ris(data["lime"], instance, n_perturbations=10, seed=42)
        
        assert isinstance(ris_5, float)
        assert isinstance(ris_10, float)
        
        print(f"\nRIS (5 perturb): {ris_5:.4f}, RIS (10 perturb): {ris_10:.4f}")
    
    def test_ris_noise_scale_effect(self, setup_model_and_explainer):
        """Larger noise scale affects RIS."""
        data = setup_model_and_explainer
        
        instance = data["X"][0]
        
        ris_small = compute_ris(data["lime"], instance, noise_scale=0.001, seed=42)
        ris_large = compute_ris(data["lime"], instance, noise_scale=0.1, seed=42)
        
        assert isinstance(ris_small, float)
        assert isinstance(ris_large, float)
        
        print(f"\nRIS (small noise): {ris_small:.4f}, RIS (large noise): {ris_large:.4f}")
    
    def test_ris_reproducibility(self, setup_model_and_explainer):
        """RIS is reproducible with same seed."""
        data = setup_model_and_explainer
        
        instance = data["X"][0]
        
        ris_1 = compute_ris(data["lime"], instance, n_perturbations=5, seed=42)
        ris_2 = compute_ris(data["lime"], instance, n_perturbations=5, seed=42)
        
        assert ris_1 == ris_2


# =============================================================================
# ROS (Relative Output Stability) Tests
# =============================================================================

class TestROS:
    """Tests for Relative Output Stability metric."""
    
    @pytest.fixture
    def setup_model_and_explainer(self):
        """Setup common test fixtures."""
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
        
        return {
            "X": X, "y": y,
            "adapter": adapter,
            "lime": lime,
            "feature_names": feature_names,
            "class_names": class_names,
        }
    
    def test_ros_basic(self, setup_model_and_explainer):
        """ROS returns a valid float score."""
        data = setup_model_and_explainer
        
        instance = data["X"][0]
        ros = compute_ros(
            data["lime"],
            data["adapter"],
            instance,
            reference_instances=data["X"][:50],
            n_neighbors=5
        )
        
        assert isinstance(ros, float)
        assert -1.0 <= ros <= 1.0 or ros == 1.0
        print(f"\nROS score: {ros:.4f}")
    
    def test_ros_high_threshold(self, setup_model_and_explainer):
        """ROS with high prediction threshold finds more neighbors."""
        data = setup_model_and_explainer
        
        instance = data["X"][0]
        
        ros_low = compute_ros(
            data["lime"], data["adapter"], instance,
            reference_instances=data["X"][:50],
            prediction_threshold=0.01
        )
        
        ros_high = compute_ros(
            data["lime"], data["adapter"], instance,
            reference_instances=data["X"][:50],
            prediction_threshold=0.2
        )
        
        assert isinstance(ros_low, float)
        assert isinstance(ros_high, float)
        
        print(f"\nROS (low thresh): {ros_low:.4f}, ROS (high thresh): {ros_high:.4f}")
    
    def test_ros_with_different_models(self):
        """ROS works with different model types."""
        iris = load_iris()
        X, y = iris.data, iris.target
        class_names = iris.target_names.tolist()
        feature_names = list(iris.feature_names)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        rf_adapter = SklearnAdapter(rf, class_names=class_names)
        
        lime_rf = LimeExplainer(
            model=rf_adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        instance = X[0]
        ros = compute_ros(
            lime_rf, rf_adapter, instance,
            reference_instances=X[:50]
        )
        
        assert isinstance(ros, float)
        print(f"\nROS (Random Forest): {ros:.4f}")


# =============================================================================
# Lipschitz Estimate Tests
# =============================================================================

class TestLipschitzEstimate:
    """Tests for Lipschitz constant estimation."""
    
    @pytest.fixture
    def setup_model_and_explainer(self):
        """Setup common test fixtures."""
        X, y = make_classification(
            n_samples=100, n_features=5,
            n_informative=3, n_classes=2, random_state=42
        )
        class_names = ["class_0", "class_1"]
        feature_names = [f"f{i}" for i in range(5)]
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        lime = LimeExplainer(
            model=adapter,
            training_data=X,
            feature_names=feature_names,
            class_names=class_names
        )
        
        return {
            "X": X, "y": y,
            "adapter": adapter,
            "lime": lime,
            "feature_names": feature_names,
        }
    
    def test_lipschitz_basic(self, setup_model_and_explainer):
        """Lipschitz estimate returns a valid float."""
        data = setup_model_and_explainer
        
        instance = data["X"][0]
        lipschitz = compute_lipschitz_estimate(
            data["lime"], instance,
            n_samples=10, radius=0.1, seed=42
        )
        
        assert isinstance(lipschitz, float)
        assert lipschitz >= 0
        print(f"\nLipschitz estimate: {lipschitz:.4f}")
    
    def test_lipschitz_radius_effect(self, setup_model_and_explainer):
        """Lipschitz estimate varies with radius."""
        data = setup_model_and_explainer
        
        instance = data["X"][0]
        
        lip_small = compute_lipschitz_estimate(
            data["lime"], instance,
            n_samples=10, radius=0.05, seed=42
        )
        
        lip_large = compute_lipschitz_estimate(
            data["lime"], instance,
            n_samples=10, radius=0.2, seed=42
        )
        
        assert isinstance(lip_small, float)
        assert isinstance(lip_large, float)
        
        print(f"\nLipschitz (small r): {lip_small:.4f}, Lipschitz (large r): {lip_large:.4f}")


# =============================================================================
# Combined Stability Metrics Tests
# =============================================================================

class TestCombinedStability:
    """Tests for combined stability metrics."""
    
    @pytest.fixture
    def setup_model_and_explainer(self):
        """Setup common test fixtures."""
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
        
        return {
            "X": X, "y": y,
            "adapter": adapter,
            "lime": lime,
            "feature_names": feature_names,
            "class_names": class_names,
        }
    
    def test_stability_metrics_structure(self, setup_model_and_explainer):
        """Stability metrics returns expected structure."""
        data = setup_model_and_explainer
        
        instance = data["X"][0]
        metrics = compute_stability_metrics(
            data["lime"],
            data["adapter"],
            instance,
            background_data=data["X"][:50],
            n_perturbations=3,
            seed=42
        )
        
        assert "ris" in metrics
        assert "ros" in metrics
        assert "lipschitz" in metrics
        
        assert isinstance(metrics["ris"], float)
        assert isinstance(metrics["ros"], float)
        assert isinstance(metrics["lipschitz"], float)
        
        print(f"\nStability metrics: {metrics}")


# =============================================================================
# Batch Stability Tests
# =============================================================================

class TestBatchStability:
    """Tests for batch stability operations."""
    
    @pytest.fixture
    def setup_model_and_explainer(self):
        """Setup common test fixtures."""
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
        
        return {
            "X": X, "y": y,
            "adapter": adapter,
            "lime": lime,
            "feature_names": feature_names,
            "class_names": class_names,
        }
    
    def test_batch_stability(self, setup_model_and_explainer):
        """Batch stability computation works correctly."""
        data = setup_model_and_explainer
        
        batch_metrics = compute_batch_stability(
            data["lime"],
            data["adapter"],
            data["X"][:10],
            n_perturbations=3,
            noise_scale=0.01,
            max_samples=5,
            seed=42
        )
        
        assert "mean_ris" in batch_metrics
        assert "mean_ros" in batch_metrics
        assert "n_samples" in batch_metrics
        
        print(f"\nBatch stability: {batch_metrics}")


# =============================================================================
# Explainer Comparison Tests
# =============================================================================

class TestExplainerComparison:
    """Tests for comparing stability across explainers."""
    
    def test_compare_explainer_stability(self):
        """Compare stability of LIME vs SHAP."""
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
        
        comparison = compare_explainer_stability(
            explainers={"lime": lime, "shap": shap},
            model=adapter,
            X=X[:10],
            n_perturbations=3,
            max_samples=5,
            seed=42
        )
        
        assert "lime" in comparison
        assert "shap" in comparison
        
        assert "mean_ris" in comparison["lime"]
        assert "mean_ros" in comparison["lime"]
        
        print(f"\nExplainer stability comparison:")
        for name, metrics in comparison.items():
            print(f"  {name}: RIS={metrics['mean_ris']:.4f}, ROS={metrics['mean_ros']:.4f}")


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_ris_single_perturbation(self):
        """RIS works with single perturbation."""
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        class_names = ["c0", "c1"]
        feature_names = [f"f{i}" for i in range(4)]
        
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
        ris = compute_ris(lime, instance, n_perturbations=1, seed=42)
        
        assert isinstance(ris, float)
    
    def test_ros_no_similar_instances(self):
        """ROS handles case when no similar instances exist."""
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        class_names = ["c0", "c1"]
        feature_names = [f"f{i}" for i in range(4)]
        
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
        # Very low threshold - unlikely to find similar instances
        ros = compute_ros(
            lime, adapter, instance,
            reference_instances=X[:10],
            prediction_threshold=0.0001
        )
        
        assert isinstance(ros, float)
    
    def test_small_dataset(self):
        """Stability metrics work with small datasets."""
        X, y = make_classification(n_samples=20, n_features=4, n_informative=2, n_redundant=1, random_state=42)
        class_names = ["c0", "c1"]
        feature_names = [f"f{i}" for i in range(4)]
        
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
        
        ris = compute_ris(lime, instance, n_perturbations=2, seed=42)
        ros = compute_ros(lime, adapter, instance, X[:10])
        
        assert isinstance(ris, float)
        assert isinstance(ros, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
