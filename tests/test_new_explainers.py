# tests/test_new_explainers.py
"""
Tests for XAI methods: Anchors, Permutation Importance, Partial Dependence, 
ALE, SAGE, and Counterfactuals.

Note: TreeSHAP, Integrated Gradients, and SHAP Interaction tests are marked
as skip until those explainers are implemented.
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from explainiverse.adapters.sklearn_adapter import SklearnAdapter
from explainiverse.core.explanation import Explanation


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def iris_setup():
    """Common setup for iris dataset tests."""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(iris.feature_names),
        "class_names": iris.target_names.tolist(),
        "model": model,
        "adapter": adapter
    }


@pytest.fixture
def binary_setup():
    """Setup for binary classification tests."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    adapter = SklearnAdapter(model, class_names=["class_0", "class_1"])
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "class_names": ["class_0", "class_1"],
        "model": model,
        "adapter": adapter
    }


# =============================================================================
# Permutation Importance Tests
# =============================================================================

class TestPermutationImportance:
    """Tests for Permutation Feature Importance explainer."""
    
    def test_permutation_importance_basic(self, iris_setup):
        """Permutation importance produces valid global attributions."""
        from explainiverse.explainers.global_explainers.permutation_importance import (
            PermutationImportanceExplainer
        )
        
        explainer = PermutationImportanceExplainer(
            model=iris_setup["adapter"],
            X=iris_setup["X_test"],
            y=iris_setup["y_test"],
            feature_names=iris_setup["feature_names"]
        )
        
        explanation = explainer.explain()
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "PermutationImportance"
        assert "feature_attributions" in explanation.explanation_data
        
        attributions = explanation.explanation_data["feature_attributions"]
        assert len(attributions) == len(iris_setup["feature_names"])
    
    def test_permutation_importance_values_reasonable(self, iris_setup):
        """Permutation importance values should be non-negative for informative features."""
        from explainiverse.explainers.global_explainers.permutation_importance import (
            PermutationImportanceExplainer
        )
        
        explainer = PermutationImportanceExplainer(
            model=iris_setup["adapter"],
            X=iris_setup["X_test"],
            y=iris_setup["y_test"],
            feature_names=iris_setup["feature_names"]
        )
        
        explanation = explainer.explain()
        attributions = explanation.explanation_data["feature_attributions"]
        
        # Most values should be non-negative (importance = accuracy drop)
        values = list(attributions.values())
        assert sum(1 for v in values if v >= -0.01) >= len(values) // 2
    
    def test_permutation_importance_n_repeats(self, iris_setup):
        """Permutation importance respects n_repeats parameter."""
        from explainiverse.explainers.global_explainers.permutation_importance import (
            PermutationImportanceExplainer
        )
        
        explainer = PermutationImportanceExplainer(
            model=iris_setup["adapter"],
            X=iris_setup["X_test"],
            y=iris_setup["y_test"],
            feature_names=iris_setup["feature_names"],
            n_repeats=5
        )
        
        explanation = explainer.explain()
        
        # Should still produce valid output
        assert "feature_attributions" in explanation.explanation_data
        
        # Check that standard deviations are available
        assert "std" in explanation.explanation_data
    
    def test_permutation_importance_registered(self):
        """Permutation importance is registered in default registry."""
        from explainiverse.core.registry import default_registry
        
        assert "permutation_importance" in default_registry.list_explainers()
        meta = default_registry.get_meta("permutation_importance")
        assert meta.scope == "global"


# =============================================================================
# Partial Dependence Tests  
# =============================================================================

class TestPartialDependence:
    """Tests for Partial Dependence Plot explainer."""
    
    def test_pdp_single_feature(self, iris_setup):
        """PDP works for a single feature."""
        from explainiverse.explainers.global_explainers.partial_dependence import (
            PartialDependenceExplainer
        )
        
        explainer = PartialDependenceExplainer(
            model=iris_setup["adapter"],
            X=iris_setup["X_train"],
            feature_names=iris_setup["feature_names"]
        )
        
        explanation = explainer.explain(features=[0])  # First feature
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "PartialDependence"
        assert "pdp_values" in explanation.explanation_data
        assert "grid_values" in explanation.explanation_data
    
    def test_pdp_multiple_features(self, iris_setup):
        """PDP works for multiple features independently."""
        from explainiverse.explainers.global_explainers.partial_dependence import (
            PartialDependenceExplainer
        )
        
        explainer = PartialDependenceExplainer(
            model=iris_setup["adapter"],
            X=iris_setup["X_train"],
            feature_names=iris_setup["feature_names"]
        )
        
        explanation = explainer.explain(features=[0, 2])
        
        pdp_values = explanation.explanation_data["pdp_values"]
        assert len(pdp_values) == 2  # Two features
    
    def test_pdp_interaction(self, iris_setup):
        """PDP can compute 2-way interactions."""
        from explainiverse.explainers.global_explainers.partial_dependence import (
            PartialDependenceExplainer
        )
        
        explainer = PartialDependenceExplainer(
            model=iris_setup["adapter"],
            X=iris_setup["X_train"],
            feature_names=iris_setup["feature_names"]
        )
        
        explanation = explainer.explain(features=[(0, 1)])  # Interaction
        
        assert "pdp_values" in explanation.explanation_data
        # 2D interaction should have 2D grid
        assert explanation.explanation_data.get("interaction", False) or \
               len(explanation.explanation_data["pdp_values"]) > 0
    
    def test_pdp_registered(self):
        """PDP is registered in default registry."""
        from explainiverse.core.registry import default_registry
        
        assert "partial_dependence" in default_registry.list_explainers()
        meta = default_registry.get_meta("partial_dependence")
        assert meta.scope == "global"


# =============================================================================
# Accumulated Local Effects (ALE) Tests
# =============================================================================

class TestALE:
    """Tests for Accumulated Local Effects explainer."""
    
    def test_ale_basic(self, iris_setup):
        """ALE produces valid explanations."""
        from explainiverse.explainers.global_explainers.ale import ALEExplainer
        
        explainer = ALEExplainer(
            model=iris_setup["adapter"],
            X=iris_setup["X_train"],
            feature_names=iris_setup["feature_names"]
        )
        
        explanation = explainer.explain(feature=0)
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "ALE"
        assert "ale_values" in explanation.explanation_data
        assert "bin_edges" in explanation.explanation_data
    
    def test_ale_centered(self, binary_setup):
        """ALE values should be centered around zero."""
        from explainiverse.explainers.global_explainers.ale import ALEExplainer
        
        explainer = ALEExplainer(
            model=binary_setup["adapter"],
            X=binary_setup["X_train"],
            feature_names=binary_setup["feature_names"]
        )
        
        explanation = explainer.explain(feature=0)
        
        # ALE should produce values centered around 0
        ale_values = explanation.explanation_data["ale_values"]
        assert abs(np.mean(ale_values)) < 0.5  # Roughly centered
    
    def test_ale_all_features(self, iris_setup):
        """ALE can compute effects for all features."""
        from explainiverse.explainers.global_explainers.ale import ALEExplainer
        
        explainer = ALEExplainer(
            model=iris_setup["adapter"],
            X=iris_setup["X_train"],
            feature_names=iris_setup["feature_names"]
        )
        
        explanations = explainer.explain_all()
        
        assert len(explanations) == len(iris_setup["feature_names"])
        for exp in explanations:
            assert isinstance(exp, Explanation)
            assert "ale_values" in exp.explanation_data
    
    def test_ale_registered(self):
        """ALE is registered in default registry."""
        from explainiverse.core.registry import default_registry
        
        assert "ale" in default_registry.list_explainers()


# =============================================================================
# Anchors Tests
# =============================================================================

class TestAnchors:
    """Tests for Anchors explainer (rule-based explanations)."""
    
    def test_anchors_basic(self, iris_setup):
        """Anchors produces rule-based explanations."""
        from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer
        
        explainer = AnchorsExplainer(
            model=iris_setup["adapter"],
            training_data=iris_setup["X_train"],
            feature_names=iris_setup["feature_names"],
            class_names=iris_setup["class_names"],
            n_samples=500  # Reduce for faster testing
        )
        
        explanation = explainer.explain(iris_setup["X_test"][0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "Anchors"
        assert "rules" in explanation.explanation_data
        assert "precision" in explanation.explanation_data
        assert "coverage" in explanation.explanation_data
    
    def test_anchors_rules_are_valid(self, iris_setup):
        """Anchors rules should reference valid features."""
        from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer
        
        explainer = AnchorsExplainer(
            model=iris_setup["adapter"],
            training_data=iris_setup["X_train"],
            feature_names=iris_setup["feature_names"],
            class_names=iris_setup["class_names"],
            n_samples=500
        )
        
        explanation = explainer.explain(iris_setup["X_test"][0])
        rules = explanation.explanation_data["rules"]
        
        # Rules should be a list of strings
        assert isinstance(rules, list)
        if len(rules) > 0:
            assert all(isinstance(r, str) for r in rules)
    
    def test_anchors_precision_high(self, iris_setup):
        """Anchors should achieve high precision when found."""
        from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer
        
        explainer = AnchorsExplainer(
            model=iris_setup["adapter"],
            training_data=iris_setup["X_train"],
            feature_names=iris_setup["feature_names"],
            class_names=iris_setup["class_names"],
            threshold=0.90,
            n_samples=500
        )
        
        explanation = explainer.explain(iris_setup["X_test"][0])
        
        # If we found rules, precision should be at least the threshold (or close)
        precision = explanation.explanation_data["precision"]
        if len(explanation.explanation_data["rules"]) > 0:
            assert precision >= 0.80  # Allow some tolerance
    
    def test_anchors_registered(self):
        """Anchors is registered in default registry."""
        from explainiverse.core.registry import default_registry
        
        assert "anchors" in default_registry.list_explainers()
        meta = default_registry.get_meta("anchors")
        assert meta.scope == "local"


# =============================================================================
# Counterfactual Tests (DiCE-style)
# =============================================================================

class TestCounterfactuals:
    """Tests for Counterfactual explanations."""
    
    def test_counterfactual_basic(self, binary_setup):
        """Counterfactuals produce valid alternative instances."""
        from explainiverse.explainers.counterfactual.dice_wrapper import (
            CounterfactualExplainer
        )
        
        explainer = CounterfactualExplainer(
            model=binary_setup["adapter"],
            training_data=binary_setup["X_train"],
            feature_names=binary_setup["feature_names"],
            continuous_features=binary_setup["feature_names"]  # All continuous
        )
        
        instance = binary_setup["X_test"][0]
        explanation = explainer.explain(instance, num_counterfactuals=3)
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "Counterfactual"
        assert "counterfactuals" in explanation.explanation_data
        
        cfs = explanation.explanation_data["counterfactuals"]
        assert len(cfs) <= 3
    
    def test_counterfactual_changes_prediction(self, binary_setup):
        """Counterfactuals should change the prediction."""
        from explainiverse.explainers.counterfactual.dice_wrapper import (
            CounterfactualExplainer
        )
        
        explainer = CounterfactualExplainer(
            model=binary_setup["adapter"],
            training_data=binary_setup["X_train"],
            feature_names=binary_setup["feature_names"],
            continuous_features=binary_setup["feature_names"]
        )
        
        instance = binary_setup["X_test"][0]
        original_pred = np.argmax(binary_setup["adapter"].predict(instance.reshape(1, -1)))
        
        explanation = explainer.explain(instance, num_counterfactuals=1)
        cfs = explanation.explanation_data["counterfactuals"]
        
        if len(cfs) > 0:
            cf = np.array(cfs[0])
            cf_pred = np.argmax(binary_setup["adapter"].predict(cf.reshape(1, -1)))
            assert cf_pred != original_pred
    
    def test_counterfactual_tracks_changes(self, binary_setup):
        """Counterfactuals should track which features changed."""
        from explainiverse.explainers.counterfactual.dice_wrapper import (
            CounterfactualExplainer
        )
        
        explainer = CounterfactualExplainer(
            model=binary_setup["adapter"],
            training_data=binary_setup["X_train"],
            feature_names=binary_setup["feature_names"],
            continuous_features=binary_setup["feature_names"]
        )
        
        instance = binary_setup["X_test"][0]
        explanation = explainer.explain(instance, num_counterfactuals=1)
        
        if len(explanation.explanation_data["counterfactuals"]) > 0:
            # Check that changes are recorded
            assert "changes" in explanation.explanation_data
            changes = explanation.explanation_data["changes"]
            assert isinstance(changes, list)
    
    def test_counterfactual_registered(self):
        """Counterfactuals are registered in default registry."""
        from explainiverse.core.registry import default_registry
        
        assert "counterfactual" in default_registry.list_explainers()


# =============================================================================
# SAGE (Shapley Additive Global importancE) Tests
# =============================================================================

class TestSAGE:
    """Tests for SAGE (Shapley Additive Global importancE)."""
    
    def test_sage_basic(self, iris_setup):
        """SAGE produces global feature importance."""
        from explainiverse.explainers.global_explainers.sage import SAGEExplainer
        
        # Use small subset for faster testing
        X_small = iris_setup["X_train"][:50]
        y_small = iris_setup["y_train"][:50]
        
        explainer = SAGEExplainer(
            model=iris_setup["adapter"],
            X=X_small,
            y=y_small,
            feature_names=iris_setup["feature_names"],
            n_permutations=20  # Reduce for faster testing
        )
        
        explanation = explainer.explain()
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "SAGE"
        assert "feature_attributions" in explanation.explanation_data
        
        attributions = explanation.explanation_data["feature_attributions"]
        assert len(attributions) == len(iris_setup["feature_names"])
    
    def test_sage_informative_features_higher(self, binary_setup):
        """SAGE should assign higher importance to informative features."""
        from explainiverse.explainers.global_explainers.sage import SAGEExplainer
        
        # Use small subset for faster testing
        X_small = binary_setup["X_train"][:50]
        y_small = binary_setup["y_train"][:50]
        
        explainer = SAGEExplainer(
            model=binary_setup["adapter"],
            X=X_small,
            y=y_small,
            feature_names=binary_setup["feature_names"],
            n_permutations=20
        )
        
        explanation = explainer.explain()
        attributions = explanation.explanation_data["feature_attributions"]
        
        # At least one feature should have positive importance
        values = list(attributions.values())
        assert max(values) > 0
    
    def test_sage_registered(self):
        """SAGE is registered in default registry."""
        from explainiverse.core.registry import default_registry
        
        assert "sage" in default_registry.list_explainers()
        meta = default_registry.get_meta("sage")
        assert meta.scope == "global"


# =============================================================================
# TreeSHAP Tests
# =============================================================================

class TestTreeSHAP:
    """Tests for TreeSHAP (optimized SHAP for tree models)."""
    
    @pytest.fixture
    def rf_setup(self):
        """Setup with RandomForest for TreeSHAP tests."""
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": list(iris.feature_names),
            "class_names": iris.target_names.tolist(),
            "model": model
        }
    
    def test_treeshap_basic(self, rf_setup):
        """TreeSHAP works with RandomForest."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
        
        explainer = TreeShapExplainer(
            model=rf_setup["model"],
            feature_names=rf_setup["feature_names"],
            class_names=rf_setup["class_names"]
        )
        
        explanation = explainer.explain(rf_setup["X_test"][0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "TreeSHAP"
        assert "feature_attributions" in explanation.explanation_data
        assert "base_value" in explanation.explanation_data
        
        attributions = explanation.explanation_data["feature_attributions"]
        assert len(attributions) == len(rf_setup["feature_names"])
    
    def test_treeshap_with_xgboost(self):
        """TreeSHAP works with XGBoost."""
        from xgboost import XGBClassifier
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
        
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y[:len(X_train)])
        
        explainer = TreeShapExplainer(
            model=model,
            feature_names=list(iris.feature_names),
            class_names=iris.target_names.tolist()
        )
        
        explanation = explainer.explain(X_test[0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "TreeSHAP"
        assert "feature_attributions" in explanation.explanation_data
    
    def test_treeshap_with_gradient_boosting(self):
        """TreeSHAP works with GradientBoosting (binary classification)."""
        from sklearn.ensemble import GradientBoostingClassifier
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
        
        # GradientBoosting only supports binary in SHAP TreeExplainer
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5,
            n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = TreeShapExplainer(
            model=model,
            feature_names=feature_names,
            class_names=["class_0", "class_1"]
        )
        
        explanation = explainer.explain(X_test[0])
        
        assert isinstance(explanation, Explanation)
        assert "feature_attributions" in explanation.explanation_data
    
    def test_treeshap_rejects_non_tree_models(self):
        """TreeSHAP raises error for non-tree models."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
        
        iris = load_iris()
        X, y = iris.data, iris.target
        
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        
        with pytest.raises(ValueError, match="tree-based model"):
            TreeShapExplainer(
                model=model,
                feature_names=list(iris.feature_names),
                class_names=iris.target_names.tolist()
            )
    
    def test_treeshap_batch_explain(self, rf_setup):
        """TreeSHAP can efficiently explain batches."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
        
        explainer = TreeShapExplainer(
            model=rf_setup["model"],
            feature_names=rf_setup["feature_names"],
            class_names=rf_setup["class_names"]
        )
        
        # Explain 5 instances at once
        explanations = explainer.explain_batch(rf_setup["X_test"][:5])
        
        assert len(explanations) == 5
        for exp in explanations:
            assert isinstance(exp, Explanation)
            assert "feature_attributions" in exp.explanation_data
    
    def test_treeshap_interactions(self, rf_setup):
        """TreeSHAP can compute interaction values."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
        
        explainer = TreeShapExplainer(
            model=rf_setup["model"],
            feature_names=rf_setup["feature_names"],
            class_names=rf_setup["class_names"]
        )
        
        explanation = explainer.explain_interactions(rf_setup["X_test"][0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "TreeSHAP_Interactions"
        assert "interaction_matrix" in explanation.explanation_data
        assert "feature_attributions" in explanation.explanation_data  # Main effects
        
        # Interaction matrix should be n_features x n_features
        matrix = explanation.explanation_data["interaction_matrix"]
        n_features = len(rf_setup["feature_names"])
        assert len(matrix) == n_features
        assert len(matrix[0]) == n_features
    
    def test_treeshap_faster_than_kernelshap(self, rf_setup):
        """TreeSHAP should be faster than KernelSHAP for tree models."""
        import time
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
        from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
        from explainiverse.adapters.sklearn_adapter import SklearnAdapter
        
        # TreeSHAP timing
        tree_explainer = TreeShapExplainer(
            model=rf_setup["model"],
            feature_names=rf_setup["feature_names"],
            class_names=rf_setup["class_names"]
        )
        
        start = time.time()
        for i in range(10):
            tree_explainer.explain(rf_setup["X_test"][i])
        treeshap_time = time.time() - start
        
        # KernelSHAP timing (with adapter)
        adapter = SklearnAdapter(rf_setup["model"], class_names=rf_setup["class_names"])
        kernel_explainer = ShapExplainer(
            model=adapter,
            background_data=rf_setup["X_train"][:50],
            feature_names=rf_setup["feature_names"],
            class_names=rf_setup["class_names"]
        )
        
        start = time.time()
        for i in range(10):
            kernel_explainer.explain(rf_setup["X_test"][i])
        kernelshap_time = time.time() - start
        
        # TreeSHAP should be faster (at least 2x, usually 10x+)
        print(f"\nTreeSHAP: {treeshap_time:.3f}s, KernelSHAP: {kernelshap_time:.3f}s")
        assert treeshap_time < kernelshap_time
    
    def test_treeshap_registered(self):
        """TreeSHAP is registered in default registry."""
        from explainiverse.core.registry import default_registry
        
        assert "treeshap" in default_registry.list_explainers()
        meta = default_registry.get_meta("treeshap")
        assert meta.scope == "local"
        assert "tree" in meta.model_types
    
    def test_treeshap_via_registry(self, rf_setup):
        """TreeSHAP can be created via the registry."""
        from explainiverse.core.registry import default_registry
        
        explainer = default_registry.create(
            "treeshap",
            model=rf_setup["model"],
            feature_names=rf_setup["feature_names"],
            class_names=rf_setup["class_names"]
        )
        
        explanation = explainer.explain(rf_setup["X_test"][0])
        assert explanation.explainer_name == "TreeSHAP"
    
    def test_treeshap_accepts_adapter(self, rf_setup):
        """TreeSHAP can extract model from adapter."""
        from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
        from explainiverse.adapters.sklearn_adapter import SklearnAdapter
        
        adapter = SklearnAdapter(rf_setup["model"], class_names=rf_setup["class_names"])
        
        # Should work - extracts model from adapter
        explainer = TreeShapExplainer(
            model=adapter,
            feature_names=rf_setup["feature_names"],
            class_names=rf_setup["class_names"]
        )
        
        explanation = explainer.explain(rf_setup["X_test"][0])
        assert explanation.explainer_name == "TreeSHAP"


# =============================================================================
# Integrated Gradients Tests (SKIPPED - Not yet implemented)
# =============================================================================

class TestIntegratedGradients:
    """Tests for Integrated Gradients explainer."""
    
    @pytest.mark.skip(reason="Integrated Gradients not yet implemented - requires PyTorch adapter")
    def test_integrated_gradients_basic(self):
        """Integrated Gradients produces attributions."""
        pass
    
    @pytest.mark.skip(reason="Integrated Gradients not yet implemented - requires PyTorch adapter")
    def test_integrated_gradients_steps(self):
        """Integrated Gradients respects n_steps parameter."""
        pass
    
    @pytest.mark.skip(reason="Integrated Gradients not yet implemented - requires PyTorch adapter")
    def test_integrated_gradients_registered(self):
        """Integrated Gradients is registered (when torch available)."""
        pass


# =============================================================================
# Feature Interaction Tests (SKIPPED - Not yet implemented)
# =============================================================================

class TestFeatureInteraction:
    """Tests for SHAP-based feature interaction detection."""
    
    @pytest.mark.skip(reason="SHAP Interaction not yet implemented - planned for v0.2")
    def test_shap_interaction_values(self, iris_setup):
        """SHAP interaction values can be computed."""
        pass


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestRegistryIntegration:
    """Tests for registry integration with all explainers."""
    
    def test_all_local_explainers_registered(self):
        """All implemented local explainers are in the registry."""
        from explainiverse.core.registry import default_registry
        
        local_explainers = default_registry.filter(scope="local")
        
        expected = {"lime", "shap", "anchors", "counterfactual"}
        assert expected.issubset(set(local_explainers))
    
    def test_all_global_explainers_registered(self):
        """All implemented global explainers are in the registry."""
        from explainiverse.core.registry import default_registry
        
        global_explainers = default_registry.filter(scope="global")
        
        expected = {"permutation_importance", "partial_dependence", "ale", "sage"}
        assert expected.issubset(set(global_explainers))
    
    def test_registry_summary(self):
        """Registry summary includes all explainers."""
        from explainiverse.core.registry import default_registry
        
        summary = default_registry.summary()
        
        # Should mention key explainers
        assert "lime" in summary.lower()
        assert "shap" in summary.lower()
        assert "anchors" in summary.lower()
        assert "sage" in summary.lower()
    
    def test_registry_recommend(self, iris_setup):
        """Registry can recommend explainers for a use case."""
        from explainiverse.core.registry import default_registry
        
        recommendations = default_registry.recommend(
            model_type="any",
            data_type="tabular",
            scope_preference="local",
            max_results=3
        )
        
        assert len(recommendations) <= 3
        assert len(recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
