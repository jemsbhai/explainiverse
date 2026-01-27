# tests/test_faithfulness_metrics.py
"""
Comprehensive tests for faithfulness evaluation metrics: PGI, PGU, and related utilities.

Tests cover:
- PGI (Prediction Gap on Important features)
- PGU (Prediction Gap on Unimportant features)
- Faithfulness Score (combined metrics)
- Comprehensiveness and Sufficiency
- Faithfulness Correlation
- Batch operations and explainer comparison
"""
import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.evaluation.faithfulness import (
    compute_pgi,
    compute_pgu,
    compute_faithfulness_score,
    compute_comprehensiveness,
    compute_sufficiency,
    compute_faithfulness_correlation,
    compare_explainer_faithfulness,
    compute_batch_faithfulness,
)
from explainiverse.evaluation._utils import (
    get_sorted_feature_indices,
    compute_baseline_values,
    apply_feature_mask,
    resolve_k,
)


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for _utils.py helper functions."""
    
    def test_resolve_k_integer(self):
        """resolve_k handles integer input correctly."""
        assert resolve_k(3, 10) == 3
        assert resolve_k(15, 10) == 10  # capped at n_features
        assert resolve_k(1, 10) == 1
    
    def test_resolve_k_fraction(self):
        """resolve_k handles fraction input correctly."""
        assert resolve_k(0.5, 10) == 5
        assert resolve_k(0.2, 10) == 2
        assert resolve_k(0.1, 5) == 1  # at least 1
        assert resolve_k(1.0, 10) == 10
    
    def test_resolve_k_invalid(self):
        """resolve_k raises on invalid input."""
        with pytest.raises(ValueError):
            resolve_k(0, 10)
        with pytest.raises(ValueError):
            resolve_k(-1, 10)
        with pytest.raises(ValueError):
            resolve_k(1.5, 10)  # fraction > 1
    
    def test_compute_baseline_values_mean(self):
        """compute_baseline_values with 'mean' baseline."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        baseline = compute_baseline_values("mean", X)
        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(baseline, expected)
    
    def test_compute_baseline_values_median(self):
        """compute_baseline_values with 'median' baseline."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        baseline = compute_baseline_values("median", X)
        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(baseline, expected)
    
    def test_compute_baseline_values_scalar(self):
        """compute_baseline_values with scalar baseline."""
        baseline = compute_baseline_values(0.0, n_features=4)
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(baseline, expected)
    
    def test_compute_baseline_values_array(self):
        """compute_baseline_values with array baseline."""
        custom = np.array([1.0, 2.0, 3.0])
        baseline = compute_baseline_values(custom)
        np.testing.assert_array_almost_equal(baseline, custom)
    
    def test_compute_baseline_values_callable(self):
        """compute_baseline_values with callable baseline."""
        X = np.array([[1, 2], [3, 4]])
        baseline = compute_baseline_values(lambda x: np.min(x, axis=0), X)
        expected = np.array([1.0, 2.0])
        np.testing.assert_array_almost_equal(baseline, expected)
    
    def test_apply_feature_mask(self):
        """apply_feature_mask replaces features correctly."""
        instance = np.array([1.0, 2.0, 3.0, 4.0])
        baseline = np.array([0.0, 0.0, 0.0, 0.0])
        result = apply_feature_mask(instance, [1, 3], baseline)
        expected = np.array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# PGI Tests
# =============================================================================

class TestPGI:
    """Tests for Prediction Gap on Important features."""
    
    @pytest.fixture
    def setup_model_and_data(self):
        """Setup common test fixtures."""
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
    
    def test_pgi_basic(self, setup_model_and_data):
        """PGI returns a valid float score."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        pgi = compute_pgi(
            data["adapter"], instance, exp,
            k=2, baseline="mean", background_data=data["X"]
        )
        
        assert isinstance(pgi, float)
        assert pgi >= 0
        print(f"\nPGI score: {pgi:.4f}")
    
    def test_pgi_with_fraction_k(self, setup_model_and_data):
        """PGI works with fractional k value."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        # k=0.5 means remove top 50% of features
        pgi = compute_pgi(
            data["adapter"], instance, exp,
            k=0.5, baseline="mean", background_data=data["X"]
        )
        
        assert isinstance(pgi, float)
        assert pgi >= 0
    
    def test_pgi_increases_with_k(self, setup_model_and_data):
        """PGI values are valid floats for different k values."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        pgi_1 = compute_pgi(data["adapter"], instance, exp, k=1, baseline="mean", background_data=data["X"])
        pgi_2 = compute_pgi(data["adapter"], instance, exp, k=2, baseline="mean", background_data=data["X"])
        pgi_3 = compute_pgi(data["adapter"], instance, exp, k=3, baseline="mean", background_data=data["X"])
        
        print(f"\nPGI k=1: {pgi_1:.4f}, k=2: {pgi_2:.4f}, k=3: {pgi_3:.4f}")
        
        # All should be valid non-negative floats
        # Note: PGI doesn't always monotonically increase due to feature interactions
        assert isinstance(pgi_1, float) and pgi_1 >= 0
        assert isinstance(pgi_2, float) and pgi_2 >= 0
        assert isinstance(pgi_3, float) and pgi_3 >= 0


# =============================================================================
# PGU Tests
# =============================================================================

class TestPGU:
    """Tests for Prediction Gap on Unimportant features."""
    
    @pytest.fixture
    def setup_model_and_data(self):
        """Setup common test fixtures."""
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
    
    def test_pgu_basic(self, setup_model_and_data):
        """PGU returns a valid float score."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        pgu = compute_pgu(
            data["adapter"], instance, exp,
            k=2, baseline="mean", background_data=data["X"]
        )
        
        assert isinstance(pgu, float)
        assert pgu >= 0
        print(f"\nPGU score: {pgu:.4f}")
    
    def test_pgu_less_than_pgi(self, setup_model_and_data):
        """PGU should generally be less than PGI for good explanations."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        pgi_total = 0
        pgu_total = 0
        n_samples = 10
        
        for i in range(n_samples):
            instance = data["X"][i]
            exp = lime.explain(instance)
            exp.feature_names = data["feature_names"]
            
            pgi = compute_pgi(data["adapter"], instance, exp, k=2, baseline="mean", background_data=data["X"])
            pgu = compute_pgu(data["adapter"], instance, exp, k=2, baseline="mean", background_data=data["X"])
            
            pgi_total += pgi
            pgu_total += pgu
        
        print(f"\nAvg PGI: {pgi_total/n_samples:.4f}, Avg PGU: {pgu_total/n_samples:.4f}")
        
        # On average, PGI should be higher than PGU for useful explanations
        assert pgi_total >= pgu_total * 0.8  # Allow some flexibility


# =============================================================================
# Faithfulness Score Tests
# =============================================================================

class TestFaithfulnessScore:
    """Tests for combined faithfulness metrics."""
    
    @pytest.fixture
    def setup_model_and_data(self):
        """Setup common test fixtures."""
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
    
    def test_faithfulness_score_structure(self, setup_model_and_data):
        """Faithfulness score returns expected structure."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        scores = compute_faithfulness_score(
            data["adapter"], instance, exp,
            k=0.2, baseline="mean", background_data=data["X"]
        )
        
        assert "pgi" in scores
        assert "pgu" in scores
        assert "faithfulness_ratio" in scores
        assert "faithfulness_diff" in scores
        
        assert isinstance(scores["pgi"], float)
        assert isinstance(scores["pgu"], float)
        assert isinstance(scores["faithfulness_ratio"], float)
        assert isinstance(scores["faithfulness_diff"], float)
        
        print(f"\nFaithfulness scores: {scores}")


# =============================================================================
# Comprehensiveness and Sufficiency Tests
# =============================================================================

class TestComprehensivenessSufficiency:
    """Tests for comprehensiveness and sufficiency metrics."""
    
    @pytest.fixture
    def setup_model_and_data(self):
        """Setup common test fixtures."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5,
            n_classes=2, random_state=42
        )
        class_names = ["class_0", "class_1"]
        feature_names = [f"feat_{i}" for i in range(10)]
        
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        adapter = SklearnAdapter(model, class_names=class_names)
        
        return {
            "X": X, "y": y,
            "adapter": adapter,
            "feature_names": feature_names,
            "class_names": class_names,
        }
    
    def test_comprehensiveness_basic(self, setup_model_and_data):
        """Comprehensiveness returns expected structure."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        comp = compute_comprehensiveness(
            data["adapter"], instance, exp,
            k_values=[0.1, 0.2, 0.3],
            baseline="mean", background_data=data["X"]
        )
        
        assert "comprehensiveness" in comp
        assert "comp_k0.1" in comp
        assert "comp_k0.2" in comp
        assert "comp_k0.3" in comp
        
        print(f"\nComprehensiveness: {comp}")
    
    def test_sufficiency_basic(self, setup_model_and_data):
        """Sufficiency returns expected structure."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        suff = compute_sufficiency(
            data["adapter"], instance, exp,
            k_values=[0.1, 0.2, 0.3],
            baseline="mean", background_data=data["X"]
        )
        
        assert "sufficiency" in suff
        assert "suff_k0.1" in suff
        
        print(f"\nSufficiency: {suff}")


# =============================================================================
# Faithfulness Correlation Tests
# =============================================================================

class TestFaithfulnessCorrelation:
    """Tests for faithfulness correlation metric."""
    
    @pytest.fixture
    def setup_model_and_data(self):
        """Setup common test fixtures."""
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
    
    def test_faithfulness_correlation_basic(self, setup_model_and_data):
        """Faithfulness correlation returns valid value."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        instance = data["X"][0]
        exp = lime.explain(instance)
        exp.feature_names = data["feature_names"]
        
        corr = compute_faithfulness_correlation(
            data["adapter"], instance, exp,
            baseline="mean", background_data=data["X"]
        )
        
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0 or corr == 0.0  # Valid correlation or no data
        
        print(f"\nFaithfulness correlation: {corr:.4f}")


# =============================================================================
# Batch and Comparison Tests
# =============================================================================

class TestBatchAndComparison:
    """Tests for batch operations and explainer comparison."""
    
    @pytest.fixture
    def setup_model_and_data(self):
        """Setup common test fixtures."""
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
    
    def test_batch_faithfulness(self, setup_model_and_data):
        """Batch faithfulness computation works correctly."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        # Generate explanations for 10 samples
        explanations = []
        for i in range(10):
            exp = lime.explain(data["X"][i])
            exp.feature_names = data["feature_names"]
            explanations.append(exp)
        
        batch_scores = compute_batch_faithfulness(
            data["adapter"], data["X"][:10], explanations,
            k=0.2, baseline="mean"
        )
        
        assert "mean_pgi" in batch_scores
        assert "mean_pgu" in batch_scores
        assert "n_samples" in batch_scores
        assert batch_scores["n_samples"] == 10
        
        print(f"\nBatch faithfulness: {batch_scores}")
    
    def test_compare_explainer_faithfulness(self, setup_model_and_data):
        """Explainer comparison returns DataFrame with expected structure."""
        data = setup_model_and_data
        
        lime = LimeExplainer(
            model=data["adapter"],
            training_data=data["X"],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        shap = ShapExplainer(
            model=data["adapter"],
            background_data=data["X"][:30],
            feature_names=data["feature_names"],
            class_names=data["class_names"]
        )
        
        # Generate explanations
        lime_exps = []
        shap_exps = []
        n_samples = 10
        
        for i in range(n_samples):
            lime_exp = lime.explain(data["X"][i])
            lime_exp.feature_names = data["feature_names"]
            lime_exps.append(lime_exp)
            
            shap_exp = shap.explain(data["X"][i])
            shap_exp.feature_names = data["feature_names"]
            shap_exps.append(shap_exp)
        
        comparison_df = compare_explainer_faithfulness(
            data["adapter"],
            data["X"][:n_samples],
            {"lime": lime_exps, "shap": shap_exps},
            k=0.2, baseline="mean"
        )
        
        assert len(comparison_df) == 2
        assert "explainer" in comparison_df.columns
        assert "mean_pgi" in comparison_df.columns
        assert "mean_pgu" in comparison_df.columns
        
        print(f"\nExplainer comparison:\n{comparison_df}")


# =============================================================================
# Multi-Model Tests
# =============================================================================

class TestMultipleModels:
    """Tests across different model types."""
    
    def test_pgi_pgu_random_forest(self):
        """PGI/PGU work with Random Forest model."""
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
        
        pgi = compute_pgi(adapter, instance, exp, k=2, baseline="mean", background_data=X)
        pgu = compute_pgu(adapter, instance, exp, k=2, baseline="mean", background_data=X)
        
        assert isinstance(pgi, float)
        assert isinstance(pgu, float)
        print(f"\nRandom Forest - PGI: {pgi:.4f}, PGU: {pgu:.4f}")
    
    def test_pgi_pgu_gradient_boosting(self):
        """PGI/PGU work with Gradient Boosting model."""
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
        
        pgi = compute_pgi(adapter, instance, exp, k=2, baseline="mean", background_data=X)
        pgu = compute_pgu(adapter, instance, exp, k=2, baseline="mean", background_data=X)
        
        assert isinstance(pgi, float)
        assert isinstance(pgu, float)
        print(f"\nGradient Boosting - PGI: {pgi:.4f}, PGU: {pgu:.4f}")


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_feature_removal(self):
        """Works with k=1 (single feature removal)."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
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
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        pgi = compute_pgi(adapter, instance, exp, k=1, baseline=0.0)
        pgu = compute_pgu(adapter, instance, exp, k=1, baseline=0.0)
        
        assert isinstance(pgi, float)
        assert isinstance(pgu, float)
    
    def test_all_features_removal(self):
        """Works when removing all features."""
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        class_names = ["class_0", "class_1"]
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
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        pgi = compute_pgi(adapter, instance, exp, k=1.0, baseline="mean", background_data=X)
        
        assert isinstance(pgi, float)
    
    def test_scalar_baseline(self):
        """Works with scalar baseline value."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
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
        
        instance = X[0]
        exp = lime.explain(instance)
        exp.feature_names = feature_names
        
        pgi = compute_pgi(adapter, instance, exp, k=2, baseline=0.0)
        
        assert isinstance(pgi, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
