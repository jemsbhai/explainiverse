# tests/test_protodash.py
"""
Comprehensive tests for ProtoDash example-based explainer.

Tests cover:
- Basic prototype selection (find_prototypes)
- Instance explanation (explain)
- Different kernel functions (rbf, linear, cosine)
- Weight optimization
- Criticism selection (find_criticisms)
- Batch operations
- Edge cases and robustness
- Integration with models
- Registry integration
"""
import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from explainiverse.explainers.example_based.protodash import ProtoDashExplainer
from explainiverse.adapters.sklearn_adapter import SklearnAdapter


# =============================================================================
# Basic Prototype Selection Tests
# =============================================================================

class TestFindPrototypes:
    """Tests for find_prototypes method."""
    
    def test_find_prototypes_basic(self):
        """Basic prototype selection returns valid results."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5, kernel="rbf")
        result = explainer.find_prototypes(X)
        
        assert result.explainer_name == "ProtoDash"
        assert "prototype_indices" in result.explanation_data
        assert "weights" in result.explanation_data
        assert "prototypes" in result.explanation_data
        
        indices = result.explanation_data["prototype_indices"]
        weights = result.explanation_data["weights"]
        
        # With force_n_prototypes=True (default), should get exactly n_prototypes
        assert len(indices) == 5
        assert len(weights) == 5
        assert all(0 <= idx < 100 for idx in indices)
        assert all(w >= 0 for w in weights)  # Non-negative weights
        
        # Weights are normalized to sum to 1 for interpretability
        assert abs(sum(weights) - 1.0) < 1e-6
        
        print(f"\nPrototype indices: {indices}")
        print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    
    def test_find_prototypes_with_labels(self):
        """Prototype selection respects class labels."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5, kernel="rbf")
        
        # Find prototypes for class 0 only
        result = explainer.find_prototypes(X, y=y, target_class=0)
        
        indices = result.explanation_data["prototype_indices"]
        
        # All selected prototypes should be from class 0
        for idx in indices:
            assert y[idx] == 0, f"Prototype {idx} is from class {y[idx]}, expected class 0"
        
        print(f"\nClass 0 prototype indices: {indices}")
    
    def test_find_prototypes_returns_mmd(self):
        """MMD score is computed when requested."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5)
        result = explainer.find_prototypes(X, return_mmd=True)
        
        assert "mmd_score" in result.explanation_data
        mmd = result.explanation_data["mmd_score"]
        assert isinstance(mmd, float)
        assert mmd >= 0
        
        print(f"\nMMD score: {mmd:.6f}")
    
    def test_find_prototypes_with_feature_names(self):
        """Feature names are preserved in output."""
        X, _ = make_classification(n_samples=100, n_features=4, random_state=42)
        feature_names = ["feat_a", "feat_b", "feat_c", "feat_d"]
        
        explainer = ProtoDashExplainer(n_prototypes=3)
        result = explainer.find_prototypes(X, feature_names=feature_names)
        
        assert "feature_names" in result.explanation_data
        assert result.explanation_data["feature_names"] == feature_names
    
    def test_find_prototypes_fewer_than_requested(self):
        """Handles case where fewer prototypes available than requested."""
        X = np.random.randn(5, 3)  # Only 5 samples
        
        explainer = ProtoDashExplainer(n_prototypes=10)  # Request 10
        result = explainer.find_prototypes(X)
        
        # Should return at most 5 prototypes
        assert len(result.explanation_data["prototype_indices"]) <= 5


# =============================================================================
# Kernel Function Tests
# =============================================================================

class TestKernelFunctions:
    """Tests for different kernel functions."""
    
    def test_rbf_kernel(self):
        """RBF kernel produces valid results."""
        X, _ = make_classification(n_samples=50, n_features=4, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5, kernel="rbf")
        result = explainer.find_prototypes(X)
        
        assert result.explanation_data["kernel"] == "rbf"
        assert result.explanation_data["kernel_width"] is not None
        assert len(result.explanation_data["prototype_indices"]) == 5
        
        print(f"\nRBF kernel width: {result.explanation_data['kernel_width']:.4f}")
    
    def test_linear_kernel(self):
        """Linear kernel produces valid results."""
        X, _ = make_classification(n_samples=50, n_features=4, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5, kernel="linear")
        result = explainer.find_prototypes(X)
        
        assert result.explanation_data["kernel"] == "linear"
        assert result.explanation_data["kernel_width"] is None
        assert len(result.explanation_data["prototype_indices"]) == 5
    
    def test_cosine_kernel(self):
        """Cosine kernel produces valid results."""
        X, _ = make_classification(n_samples=50, n_features=4, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5, kernel="cosine")
        result = explainer.find_prototypes(X)
        
        assert result.explanation_data["kernel"] == "cosine"
        assert len(result.explanation_data["prototype_indices"]) == 5
    
    def test_custom_kernel_width(self):
        """Custom kernel width is respected."""
        X, _ = make_classification(n_samples=50, n_features=4, random_state=42)
        
        custom_width = 2.5
        explainer = ProtoDashExplainer(n_prototypes=5, kernel="rbf", kernel_width=custom_width)
        result = explainer.find_prototypes(X)
        
        assert result.explanation_data["kernel_width"] == custom_width
    
    def test_invalid_kernel_raises(self):
        """Invalid kernel name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            ProtoDashExplainer(n_prototypes=5, kernel="invalid_kernel")


# =============================================================================
# Instance Explanation Tests
# =============================================================================

class TestExplainInstance:
    """Tests for explain method (instance-level explanation)."""
    
    def test_explain_basic(self):
        """Basic instance explanation works."""
        X_ref, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        instance = X_ref[0]  # Explain first instance
        
        explainer = ProtoDashExplainer(n_prototypes=5)
        result = explainer.explain(instance, X_reference=X_ref[1:])  # Exclude the instance itself
        
        assert result.explainer_name == "ProtoDash"
        assert "prototype_indices" in result.explanation_data
        assert "weights" in result.explanation_data
        assert "similarity_scores" in result.explanation_data
        assert "instance" in result.explanation_data
        
        indices = result.explanation_data["prototype_indices"]
        weights = result.explanation_data["weights"]
        similarities = result.explanation_data["similarity_scores"]
        
        assert len(indices) == 5
        assert len(weights) == 5
        assert len(similarities) == 5
        
        print(f"\nPrototype indices: {indices}")
        print(f"Weights: {[f'{w:.4f}' for w in weights]}")
        print(f"Similarities: {[f'{s:.4f}' for s in similarities]}")
    
    def test_explain_with_model(self):
        """Explanation includes model predictions when model is provided."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["class_0", "class_1"])
        
        explainer = ProtoDashExplainer(model=adapter, n_prototypes=5)
        result = explainer.explain(X[0], X_reference=X[1:50])
        
        assert "instance_prediction" in result.explanation_data
        assert "prototype_predictions" in result.explanation_data
        
        proto_preds = result.explanation_data["prototype_predictions"]
        assert len(proto_preds) == 5
        
        print(f"\nInstance prediction shape: {np.array(result.explanation_data['instance_prediction']).shape}")
        print(f"Prototype predictions shape: {np.array(proto_preds).shape}")
    
    def test_explain_batch(self):
        """Batch explanation works correctly."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=3)
        results = explainer.explain_batch(X[:10], X_reference=X[10:])
        
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result.explainer_name == "ProtoDash"
            assert len(result.explanation_data["prototype_indices"]) == 3


# =============================================================================
# Criticism Selection Tests
# =============================================================================

class TestCriticisms:
    """Tests for find_criticisms method."""
    
    def test_find_criticisms_basic(self):
        """Basic criticism selection works."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5)
        proto_result = explainer.find_prototypes(X)
        proto_indices = proto_result.explanation_data["prototype_indices"]
        
        crit_result = explainer.find_criticisms(X, proto_indices, n_criticisms=5)
        
        assert crit_result.explainer_name == "ProtoDash_Criticisms"
        assert "criticism_indices" in crit_result.explanation_data
        assert "unusualness_scores" in crit_result.explanation_data
        
        crit_indices = crit_result.explanation_data["criticism_indices"]
        scores = crit_result.explanation_data["unusualness_scores"]
        
        assert len(crit_indices) == 5
        assert len(scores) == 5
        
        # Criticisms should not overlap with prototypes
        for idx in crit_indices:
            assert idx not in proto_indices
        
        print(f"\nCriticism indices: {crit_indices}")
        print(f"Unusualness scores: {[f'{s:.4f}' for s in scores]}")
    
    def test_criticisms_sorted_by_unusualness(self):
        """Criticisms are sorted by unusualness (most unusual first)."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5)
        proto_result = explainer.find_prototypes(X)
        proto_indices = proto_result.explanation_data["prototype_indices"]
        
        crit_result = explainer.find_criticisms(X, proto_indices, n_criticisms=10)
        scores = crit_result.explanation_data["unusualness_scores"]
        
        # Scores should be in descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# =============================================================================
# Summary Tests
# =============================================================================

class TestPrototypeSummary:
    """Tests for get_prototype_summary method."""
    
    def test_summary_with_criticisms(self):
        """Complete summary includes both prototypes and criticisms."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5)
        summary = explainer.get_prototype_summary(X, include_criticisms=True, n_criticisms=5)
        
        assert "prototypes" in summary
        assert "criticisms" in summary
        
        # With force_n_prototypes=True (default), should get exactly n_prototypes
        assert len(summary["prototypes"]["prototype_indices"]) == 5
        assert len(summary["criticisms"]["criticism_indices"]) == 5
    
    def test_summary_without_criticisms(self):
        """Summary can exclude criticisms."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5)
        summary = explainer.get_prototype_summary(X, include_criticisms=False)
        
        assert "prototypes" in summary
        assert "criticisms" not in summary


# =============================================================================
# Weight Optimization Tests
# =============================================================================

class TestWeightOptimization:
    """Tests for weight optimization."""
    
    def test_weights_sum_to_one(self):
        """Optimized weights sum to 1."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=10, optimize_weights=True)
        result = explainer.find_prototypes(X)
        
        weights = result.explanation_data["weights"]
        assert abs(sum(weights) - 1.0) < 1e-6
    
    def test_weights_non_negative(self):
        """All weights are non-negative."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=10, optimize_weights=True)
        result = explainer.find_prototypes(X)
        
        weights = result.explanation_data["weights"]
        assert all(w >= 0 for w in weights)
    
    def test_greedy_only_weights(self):
        """Greedy-only mode produces valid weights."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=5, optimize_weights=False)
        result = explainer.find_prototypes(X)
        
        weights = result.explanation_data["weights"]
        assert abs(sum(weights) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
    
    def test_original_paper_behavior(self):
        """Test force_n_prototypes=False for original ProtoDash behavior."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        # With force_n_prototypes=False, may select fewer prototypes if gain becomes negative
        explainer = ProtoDashExplainer(n_prototypes=10, force_n_prototypes=False)
        result = explainer.find_prototypes(X)
        
        indices = result.explanation_data["prototype_indices"]
        weights = result.explanation_data["weights"]
        
        # Should have at least 1 prototype, possibly fewer than requested
        assert len(indices) >= 1
        assert len(indices) <= 10
        assert len(weights) == len(indices)
        assert all(w >= 0 for w in weights)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_prototype(self):
        """Single prototype selection works."""
        X, _ = make_classification(n_samples=50, n_features=4, random_state=42)
        
        explainer = ProtoDashExplainer(n_prototypes=1)
        result = explainer.find_prototypes(X)
        
        assert len(result.explanation_data["prototype_indices"]) == 1
        assert result.explanation_data["weights"] == [1.0]
    
    def test_small_dataset(self):
        """Works with small datasets."""
        X = np.random.randn(10, 3)
        
        explainer = ProtoDashExplainer(n_prototypes=3)
        result = explainer.find_prototypes(X)
        
        assert len(result.explanation_data["prototype_indices"]) == 3
    
    def test_high_dimensional_data(self):
        """Works with high-dimensional data."""
        X = np.random.randn(100, 50)
        
        explainer = ProtoDashExplainer(n_prototypes=5)
        result = explainer.find_prototypes(X)
        
        assert len(result.explanation_data["prototype_indices"]) == 5
    
    def test_identical_samples(self):
        """Handles dataset with identical samples."""
        X = np.ones((20, 5))  # All identical
        X[:10] += 0.01  # Slight variation
        
        explainer = ProtoDashExplainer(n_prototypes=3)
        result = explainer.find_prototypes(X)
        
        # Should still produce valid output
        assert len(result.explanation_data["prototype_indices"]) <= 3
    
    def test_reproducibility(self):
        """Results are reproducible with same random state."""
        X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        
        explainer1 = ProtoDashExplainer(n_prototypes=5, random_state=123)
        result1 = explainer1.find_prototypes(X)
        
        explainer2 = ProtoDashExplainer(n_prototypes=5, random_state=123)
        result2 = explainer2.find_prototypes(X)
        
        assert result1.explanation_data["prototype_indices"] == result2.explanation_data["prototype_indices"]


# =============================================================================
# Real Dataset Tests
# =============================================================================

class TestRealDatasets:
    """Tests with real datasets."""
    
    def test_iris_dataset(self):
        """Works with Iris dataset."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        explainer = ProtoDashExplainer(n_prototypes=5)
        result = explainer.find_prototypes(X)
        
        assert len(result.explanation_data["prototype_indices"]) == 5
        
        # Check prototypes cover different regions
        proto_indices = result.explanation_data["prototype_indices"]
        prototypes = X[proto_indices]
        
        print(f"\nIris prototypes (indices): {proto_indices}")
        print(f"Prototype labels: {[y[i] for i in proto_indices]}")
    
    def test_iris_per_class(self):
        """Find class-specific prototypes for Iris."""
        iris = load_iris()
        X, y = iris.data, iris.target
        
        explainer = ProtoDashExplainer(n_prototypes=3)
        
        for class_idx in range(3):
            result = explainer.find_prototypes(X, y=y, target_class=class_idx)
            proto_indices = result.explanation_data["prototype_indices"]
            
            # All prototypes should be from the target class
            for idx in proto_indices:
                assert y[idx] == class_idx
            
            print(f"\nClass {iris.target_names[class_idx]} prototypes: {proto_indices}")


# =============================================================================
# Integration with Models
# =============================================================================

class TestModelIntegration:
    """Tests for integration with ML models."""
    
    def test_logistic_regression(self):
        """Works with Logistic Regression model."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["class_0", "class_1"])
        
        explainer = ProtoDashExplainer(model=adapter, n_prototypes=5)
        result = explainer.explain(X[0], X_reference=X[1:])
        
        assert "instance_prediction" in result.explanation_data
        assert "prototype_predictions" in result.explanation_data
    
    def test_random_forest(self):
        """Works with Random Forest model."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=["class_0", "class_1"])
        
        explainer = ProtoDashExplainer(model=adapter, n_prototypes=5)
        result = explainer.explain(X[0], X_reference=X[1:])
        
        assert "instance_prediction" in result.explanation_data


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestRegistryIntegration:
    """Tests for registry integration."""
    
    def test_protodash_in_registry(self):
        """ProtoDash is registered in default registry."""
        from explainiverse.core.registry import get_default_registry
        
        registry = get_default_registry()
        explainers = registry.list_explainers()
        
        assert "protodash" in explainers
    
    def test_create_from_registry(self):
        """Can create ProtoDash from registry."""
        from explainiverse.core.registry import get_default_registry
        
        registry = get_default_registry()
        explainer = registry.create("protodash", n_prototypes=5)
        
        assert isinstance(explainer, ProtoDashExplainer)
        assert explainer.n_prototypes == 5
    
    def test_registry_metadata(self):
        """ProtoDash metadata is correct."""
        from explainiverse.core.registry import get_default_registry
        
        registry = get_default_registry()
        meta = registry.get_meta("protodash")
        
        assert meta.scope == "local"
        assert "any" in meta.model_types
        assert "tabular" in meta.data_types
        assert meta.requires_training_data == True


# =============================================================================
# Cluster-Based Tests
# =============================================================================

class TestClusterData:
    """Tests with clustered data to verify prototype quality."""
    
    def test_prototypes_represent_clusters(self):
        """Prototypes should represent different clusters."""
        # Create data with clear clusters
        X, cluster_labels = make_blobs(
            n_samples=150, 
            n_features=2, 
            centers=3, 
            cluster_std=0.5,
            random_state=42
        )
        
        # Request 3 prototypes (one per cluster ideally)
        explainer = ProtoDashExplainer(n_prototypes=3, kernel="rbf")
        result = explainer.find_prototypes(X)
        
        proto_indices = result.explanation_data["prototype_indices"]
        proto_clusters = [cluster_labels[i] for i in proto_indices]
        
        # Ideally, prototypes should come from different clusters
        unique_clusters = set(proto_clusters)
        
        print(f"\nPrototype clusters: {proto_clusters}")
        print(f"Unique clusters represented: {len(unique_clusters)}")
        
        # At least 2 different clusters should be represented
        assert len(unique_clusters) >= 2
    
    def test_mmd_decreases_with_more_prototypes(self):
        """MMD should generally decrease with more prototypes."""
        X, _ = make_blobs(n_samples=100, n_features=3, centers=5, random_state=42)
        
        mmds = []
        for n_proto in [1, 3, 5, 10, 20]:
            explainer = ProtoDashExplainer(n_prototypes=n_proto, kernel="rbf", random_state=42)
            result = explainer.find_prototypes(X, return_mmd=True)
            mmds.append((n_proto, result.explanation_data["mmd_score"]))
        
        print("\nMMD vs n_prototypes:")
        for n, mmd in mmds:
            print(f"  n={n}: MMD={mmd:.6f}")
        
        # MMD should generally decrease (allow some tolerance)
        assert mmds[-1][1] <= mmds[0][1] * 1.1  # Last should be <= first (with tolerance)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
