# tests/test_explainer_registry.py
"""
Test suite for the ExplainerRegistry - the plugin system that makes
adding new XAI methods trivial.
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from explainiverse.core.registry import ExplainerRegistry, ExplainerMeta
from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.adapters.sklearn_adapter import SklearnAdapter


class MockExplainer(BaseExplainer):
    """A minimal explainer for testing the registry."""
    
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.kwargs = kwargs
    
    def explain(self, instance, **kwargs):
        n_features = len(instance)
        return Explanation(
            explainer_name="MockExplainer",
            target_class="mock_class",
            explanation_data={
                "feature_attributions": {f"f{i}": 0.1 for i in range(n_features)}
            }
        )


# =============================================================================
# Registry Core Tests
# =============================================================================

def test_registry_register_and_retrieve():
    """Explainers can be registered and retrieved by name."""
    registry = ExplainerRegistry()
    
    registry.register(
        name="mock",
        explainer_class=MockExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["any"],
            data_types=["tabular"],
            description="A mock explainer for testing"
        )
    )
    
    assert "mock" in registry.list_explainers()
    retrieved = registry.get("mock")
    assert retrieved["class"] == MockExplainer


def test_registry_prevents_duplicate_registration():
    """Registry raises error on duplicate name registration."""
    registry = ExplainerRegistry()
    
    registry.register("mock", MockExplainer, ExplainerMeta(scope="local"))
    
    with pytest.raises(ValueError, match="already registered"):
        registry.register("mock", MockExplainer, ExplainerMeta(scope="local"))


def test_registry_allows_override_with_flag():
    """Registry allows override when explicitly requested."""
    registry = ExplainerRegistry()
    
    registry.register("mock", MockExplainer, ExplainerMeta(scope="local"))
    registry.register("mock", MockExplainer, ExplainerMeta(scope="global"), override=True)
    
    assert registry.get_meta("mock").scope == "global"


def test_registry_get_unknown_raises():
    """Registry raises KeyError for unknown explainer."""
    registry = ExplainerRegistry()
    
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_registry_unregister():
    """Explainers can be unregistered."""
    registry = ExplainerRegistry()
    registry.register("mock", MockExplainer, ExplainerMeta(scope="local"))
    
    registry.unregister("mock")
    
    assert "mock" not in registry.list_explainers()


def test_registry_unregister_unknown_raises():
    """Unregistering unknown explainer raises KeyError."""
    registry = ExplainerRegistry()
    
    with pytest.raises(KeyError):
        registry.unregister("nonexistent")


# =============================================================================
# Filtering & Discovery Tests
# =============================================================================

def test_registry_filter_by_scope():
    """Registry can filter explainers by scope (local/global)."""
    registry = ExplainerRegistry()
    
    registry.register("local1", MockExplainer, ExplainerMeta(scope="local"))
    registry.register("local2", MockExplainer, ExplainerMeta(scope="local"))
    registry.register("global1", MockExplainer, ExplainerMeta(scope="global"))
    
    local_explainers = registry.filter(scope="local")
    global_explainers = registry.filter(scope="global")
    
    assert set(local_explainers) == {"local1", "local2"}
    assert set(global_explainers) == {"global1"}


def test_registry_filter_by_model_type():
    """Registry can filter by supported model types."""
    registry = ExplainerRegistry()
    
    registry.register("tree_only", MockExplainer, 
                      ExplainerMeta(scope="local", model_types=["tree"]))
    registry.register("neural_only", MockExplainer,
                      ExplainerMeta(scope="local", model_types=["neural"]))
    registry.register("any_model", MockExplainer,
                      ExplainerMeta(scope="local", model_types=["any"]))
    
    tree_compatible = registry.filter(model_type="tree")
    
    assert "tree_only" in tree_compatible
    assert "any_model" in tree_compatible
    assert "neural_only" not in tree_compatible


def test_registry_filter_by_data_type():
    """Registry can filter by data type (tabular, image, text)."""
    registry = ExplainerRegistry()
    
    registry.register("tabular_exp", MockExplainer,
                      ExplainerMeta(scope="local", data_types=["tabular"]))
    registry.register("image_exp", MockExplainer,
                      ExplainerMeta(scope="local", data_types=["image"]))
    registry.register("multi_exp", MockExplainer,
                      ExplainerMeta(scope="local", data_types=["tabular", "image"]))
    
    tabular_compatible = registry.filter(data_type="tabular")
    
    assert "tabular_exp" in tabular_compatible
    assert "multi_exp" in tabular_compatible
    assert "image_exp" not in tabular_compatible


def test_registry_filter_by_task_type():
    """Registry can filter by task type (classification, regression)."""
    registry = ExplainerRegistry()
    
    registry.register("clf_only", MockExplainer,
                      ExplainerMeta(scope="local", task_types=["classification"]))
    registry.register("reg_only", MockExplainer,
                      ExplainerMeta(scope="local", task_types=["regression"]))
    registry.register("both", MockExplainer,
                      ExplainerMeta(scope="local", task_types=["classification", "regression"]))
    
    clf_compatible = registry.filter(task_type="classification")
    
    assert "clf_only" in clf_compatible
    assert "both" in clf_compatible
    assert "reg_only" not in clf_compatible


def test_registry_filter_combined():
    """Registry can filter with multiple criteria."""
    registry = ExplainerRegistry()
    
    registry.register("a", MockExplainer,
                      ExplainerMeta(scope="local", model_types=["tree"], data_types=["tabular"]))
    registry.register("b", MockExplainer,
                      ExplainerMeta(scope="global", model_types=["tree"], data_types=["tabular"]))
    registry.register("c", MockExplainer,
                      ExplainerMeta(scope="local", model_types=["neural"], data_types=["tabular"]))
    
    results = registry.filter(scope="local", model_type="tree", data_type="tabular")
    
    assert results == ["a"]


def test_registry_filter_no_matches():
    """Registry filter returns empty list when no matches."""
    registry = ExplainerRegistry()
    
    registry.register("a", MockExplainer, ExplainerMeta(scope="local"))
    
    results = registry.filter(scope="global")
    
    assert results == []


# =============================================================================
# Instantiation Tests
# =============================================================================

def test_registry_create_explainer_instance():
    """Registry can instantiate explainers with provided kwargs."""
    registry = ExplainerRegistry()
    registry.register("mock", MockExplainer, ExplainerMeta(scope="local"))
    
    iris = load_iris()
    model = LogisticRegression(max_iter=200).fit(iris.data, iris.target)
    adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())
    
    explainer = registry.create(
        "mock",
        model=adapter,
        custom_param="test_value"
    )
    
    assert isinstance(explainer, MockExplainer)
    assert explainer.kwargs.get("custom_param") == "test_value"


def test_registry_create_and_explain():
    """Full workflow: register, create, explain."""
    registry = ExplainerRegistry()
    registry.register("mock", MockExplainer, ExplainerMeta(scope="local"))
    
    iris = load_iris()
    model = LogisticRegression(max_iter=200).fit(iris.data, iris.target)
    adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())
    
    explainer = registry.create("mock", model=adapter)
    explanation = explainer.explain(iris.data[0])
    
    assert explanation.explainer_name == "MockExplainer"
    assert "feature_attributions" in explanation.explanation_data


def test_registry_create_unknown_raises():
    """Creating unknown explainer raises KeyError."""
    registry = ExplainerRegistry()
    
    with pytest.raises(KeyError):
        registry.create("nonexistent", model=None)


# =============================================================================
# Decorator Registration Tests
# =============================================================================

def test_decorator_registration():
    """Explainers can be registered via decorator."""
    registry = ExplainerRegistry()
    
    @registry.register_decorator(
        name="decorated_mock",
        meta=ExplainerMeta(scope="local", model_types=["any"])
    )
    class DecoratedExplainer(BaseExplainer):
        def __init__(self, model, **kwargs):
            super().__init__(model)
        
        def explain(self, instance, **kwargs):
            return Explanation("Decorated", "class", {"feature_attributions": {}})
    
    assert "decorated_mock" in registry.list_explainers()
    assert registry.get("decorated_mock")["class"] == DecoratedExplainer


def test_decorator_preserves_class():
    """Decorator returns the original class unchanged."""
    registry = ExplainerRegistry()
    
    @registry.register_decorator(name="test", meta=ExplainerMeta(scope="local"))
    class TestExplainer(BaseExplainer):
        CLASS_ATTR = "preserved"
        
        def explain(self, instance, **kwargs):
            return Explanation("Test", "class", {})
    
    assert TestExplainer.CLASS_ATTR == "preserved"


# =============================================================================
# Metadata Introspection Tests
# =============================================================================

def test_registry_get_metadata():
    """Registry provides metadata for explainers."""
    registry = ExplainerRegistry()
    
    meta = ExplainerMeta(
        scope="local",
        model_types=["tree", "linear"],
        data_types=["tabular"],
        task_types=["classification"],
        description="Test explainer",
        paper_reference="Doe et al., 2024",
        complexity="O(n)"
    )
    registry.register("test_exp", MockExplainer, meta)
    
    retrieved_meta = registry.get_meta("test_exp")
    
    assert retrieved_meta.scope == "local"
    assert "tree" in retrieved_meta.model_types
    assert retrieved_meta.paper_reference == "Doe et al., 2024"
    assert retrieved_meta.complexity == "O(n)"


def test_registry_get_meta_unknown_raises():
    """Getting metadata for unknown explainer raises KeyError."""
    registry = ExplainerRegistry()
    
    with pytest.raises(KeyError):
        registry.get_meta("nonexistent")


def test_registry_list_with_details():
    """Registry can list all explainers with their metadata."""
    registry = ExplainerRegistry()
    
    registry.register("exp1", MockExplainer, 
                      ExplainerMeta(scope="local", description="First"))
    registry.register("exp2", MockExplainer,
                      ExplainerMeta(scope="global", description="Second"))
    
    details = registry.list_explainers(with_meta=True)
    
    assert len(details) == 2
    assert details["exp1"]["meta"].description == "First"
    assert details["exp2"]["meta"].scope == "global"


def test_explainer_meta_defaults():
    """ExplainerMeta has sensible defaults."""
    meta = ExplainerMeta(scope="local")
    
    assert meta.model_types == ["any"]
    assert meta.data_types == ["tabular"]
    assert meta.task_types == ["classification", "regression"]
    assert meta.description == ""
    assert meta.paper_reference is None


# =============================================================================
# Global Registry Tests
# =============================================================================

def test_global_registry_has_builtin_explainers():
    """The default global registry includes LIME and SHAP."""
    from explainiverse.core.registry import default_registry
    
    explainers = default_registry.list_explainers()
    
    assert "lime" in explainers
    assert "shap" in explainers


def test_global_registry_lime_works():
    """LIME from global registry produces valid explanations."""
    from explainiverse.core.registry import default_registry
    
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200).fit(X, y)
    adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())
    
    explainer = default_registry.create(
        "lime",
        model=adapter,
        training_data=X,
        feature_names=iris.feature_names,
        class_names=iris.target_names.tolist()
    )
    
    explanation = explainer.explain(X[0])
    
    assert explanation.explainer_name == "LIME"
    assert len(explanation.explanation_data["feature_attributions"]) > 0


def test_global_registry_shap_works():
    """SHAP from global registry produces valid explanations."""
    from explainiverse.core.registry import default_registry
    
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=200).fit(X, y)
    adapter = SklearnAdapter(model, class_names=iris.target_names.tolist())
    
    explainer = default_registry.create(
        "shap",
        model=adapter,
        background_data=X[:30],
        feature_names=iris.feature_names,
        class_names=iris.target_names.tolist()
    )
    
    explanation = explainer.explain(X[0])
    
    assert explanation.explainer_name == "SHAP"
    assert len(explanation.explanation_data["feature_attributions"]) > 0


def test_global_registry_list_categories():
    """Global registry can list explainers by category."""
    from explainiverse.core.registry import default_registry
    
    local_explainers = default_registry.filter(scope="local")
    
    assert "lime" in local_explainers
    assert "shap" in local_explainers


# =============================================================================
# Utility Tests
# =============================================================================

def test_registry_summary():
    """Registry can produce a human-readable summary."""
    registry = ExplainerRegistry()
    
    registry.register("exp1", MockExplainer, 
                      ExplainerMeta(scope="local", description="First explainer"))
    registry.register("exp2", MockExplainer,
                      ExplainerMeta(scope="global", description="Second explainer"))
    
    summary = registry.summary()
    
    assert "exp1" in summary
    assert "exp2" in summary
    assert "local" in summary.lower()
    assert "global" in summary.lower()


def test_registry_recommend():
    """Registry can recommend explainers based on criteria."""
    registry = ExplainerRegistry()
    
    registry.register("fast_local", MockExplainer, 
                      ExplainerMeta(scope="local", model_types=["any"], complexity="O(n)"))
    registry.register("slow_global", MockExplainer,
                      ExplainerMeta(scope="global", model_types=["tree"], complexity="O(n^2)"))
    registry.register("neural_only", MockExplainer,
                      ExplainerMeta(scope="local", model_types=["neural"]))
    
    recommendations = registry.recommend(
        model_type="tree",
        data_type="tabular",
        scope_preference="local"
    )
    
    assert "fast_local" in recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
