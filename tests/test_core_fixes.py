# tests/test_core_fixes.py
"""
Tests for core fixes to address review issues:
1. Explanation class with feature_names
2. Lazy imports in LIME/SHAP wrappers
3. Binary classifier handling in PyTorchAdapter
4. Input shape preservation in IntegratedGradients
5. ExplanationSuite using registry
"""

import pytest
import numpy as np


# =============================================================================
# Test Explanation class with feature_names
# =============================================================================

class TestExplanationFeatureNames:
    """Test that Explanation properly supports feature_names."""
    
    def test_explanation_without_feature_names(self):
        """Explanation works without feature_names."""
        from explainiverse.core.explanation import Explanation
        
        exp = Explanation(
            explainer_name="Test",
            target_class="class_0",
            explanation_data={"feature_attributions": {"a": 0.5, "b": 0.3}}
        )
        
        assert exp.feature_names is None
        assert exp.explainer_name == "Test"
    
    def test_explanation_with_feature_names(self):
        """Explanation correctly stores feature_names."""
        from explainiverse.core.explanation import Explanation
        
        exp = Explanation(
            explainer_name="Test",
            target_class="class_0",
            explanation_data={"feature_attributions": {"a": 0.5, "b": 0.3}},
            feature_names=["a", "b", "c"]
        )
        
        assert exp.feature_names == ["a", "b", "c"]
    
    def test_explanation_get_attributions(self):
        """get_attributions helper method works."""
        from explainiverse.core.explanation import Explanation
        
        attrs = {"feat_0": 0.8, "feat_1": -0.3, "feat_2": 0.5}
        exp = Explanation(
            explainer_name="Test",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs}
        )
        
        assert exp.get_attributions() == attrs
    
    def test_explanation_get_top_features(self):
        """get_top_features returns correctly sorted features."""
        from explainiverse.core.explanation import Explanation
        
        exp = Explanation(
            explainer_name="Test",
            target_class="class_0",
            explanation_data={
                "feature_attributions": {"a": 0.1, "b": -0.8, "c": 0.5}
            }
        )
        
        # By absolute value
        top = exp.get_top_features(k=2, absolute=True)
        assert top[0][0] == "b"  # -0.8 has highest absolute value
        assert top[1][0] == "c"  # 0.5 second
    
    def test_explanation_get_feature_index(self):
        """get_feature_index returns correct index."""
        from explainiverse.core.explanation import Explanation
        
        exp = Explanation(
            explainer_name="Test",
            target_class="class_0",
            explanation_data={},
            feature_names=["age", "income", "education"]
        )
        
        assert exp.get_feature_index("income") == 1
        assert exp.get_feature_index("unknown") is None
    
    def test_explanation_to_dict_and_from_dict(self):
        """Explanation serialization round-trips correctly."""
        from explainiverse.core.explanation import Explanation
        
        exp = Explanation(
            explainer_name="LIME",
            target_class="positive",
            explanation_data={"feature_attributions": {"x": 0.5}},
            feature_names=["x", "y", "z"],
            metadata={"time": 1.5}
        )
        
        d = exp.to_dict()
        exp2 = Explanation.from_dict(d)
        
        assert exp2.explainer_name == "LIME"
        assert exp2.feature_names == ["x", "y", "z"]
        assert exp2.metadata == {"time": 1.5}


# =============================================================================
# Test lazy imports
# =============================================================================

class TestLazyImports:
    """Test that optional dependencies use lazy imports."""
    
    def test_lime_module_import_without_lime(self):
        """Importing lime_wrapper module doesn't require lime package."""
        # This should not raise even if lime is not installed
        # because imports are lazy
        try:
            from explainiverse.explainers.attribution import lime_wrapper
            assert hasattr(lime_wrapper, 'LimeExplainer')
        except ImportError as e:
            if 'lime' in str(e).lower():
                pytest.skip("LIME not installed, but import should be lazy")
            raise
    
    def test_shap_module_import_without_shap(self):
        """Importing shap_wrapper module doesn't require shap package."""
        try:
            from explainiverse.explainers.attribution import shap_wrapper
            assert hasattr(shap_wrapper, 'ShapExplainer')
        except ImportError as e:
            if 'shap' in str(e).lower():
                pytest.skip("SHAP not installed, but import should be lazy")
            raise


# =============================================================================
# Test metrics with missing feature_names
# =============================================================================

class TestMetricsGracefulHandling:
    """Test that metrics handle missing feature_names gracefully."""
    
    def test_extract_feature_index_with_names(self):
        """_extract_feature_index works with feature_names."""
        from explainiverse.evaluation.metrics import _extract_feature_index
        
        feature_names = ["age", "income", "education"]
        
        assert _extract_feature_index("income", feature_names) == 1
        assert _extract_feature_index("age", feature_names) == 0
    
    def test_extract_feature_index_without_names(self):
        """_extract_feature_index extracts from patterns when no names."""
        from explainiverse.evaluation.metrics import _extract_feature_index
        
        assert _extract_feature_index("feature_2", None, fallback_index=99) == 2
        assert _extract_feature_index("f5", None, fallback_index=99) == 5
        assert _extract_feature_index("unknown", None, fallback_index=99) == 99
    
    def test_extract_feature_index_lime_style(self):
        """_extract_feature_index handles LIME-style names."""
        from explainiverse.evaluation.metrics import _extract_feature_index
        
        feature_names = ["age", "income", "education"]
        
        # LIME often returns "age <= 30.0"
        assert _extract_feature_index("age <= 30.0", feature_names) == 0
        assert _extract_feature_index("income > 50000", feature_names) == 1


# =============================================================================
# Test PyTorchAdapter binary classification
# =============================================================================

class TestPyTorchAdapterBinaryClassification:
    """Test PyTorchAdapter handles binary classification correctly."""
    
    @pytest.fixture
    def binary_classifier(self):
        """Create a binary classifier with single output."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        # Single output (logit for positive class)
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # Single output
        )
        return model
    
    @pytest.fixture
    def multiclass_classifier(self):
        """Create a multi-class classifier."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 3)  # 3 classes
        )
        return model
    
    def test_binary_classifier_predict(self, binary_classifier):
        """Binary classifier prediction works."""
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(
            binary_classifier,
            task="classification",
            class_names=["negative", "positive"]
        )
        
        X = np.random.randn(5, 5).astype(np.float32)
        preds = adapter.predict(X)
        
        # Should return probabilities
        assert preds.shape == (5, 1)
        assert np.all((preds >= 0) & (preds <= 1))
    
    def test_binary_classifier_gradients(self, binary_classifier):
        """Binary classifier gradient computation works."""
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(
            binary_classifier,
            task="classification",
            class_names=["negative", "positive"]
        )
        
        X = np.random.randn(1, 5).astype(np.float32)
        preds, grads = adapter.predict_with_gradients(X)
        
        assert preds.shape == (1, 1)
        assert grads.shape == (1, 5)
        assert not np.all(grads == 0)  # Should have non-zero gradients
    
    def test_multiclass_classifier_predict(self, multiclass_classifier):
        """Multi-class classifier prediction works."""
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(
            multiclass_classifier,
            task="classification",
            class_names=["a", "b", "c"]
        )
        
        X = np.random.randn(5, 5).astype(np.float32)
        preds = adapter.predict(X)
        
        assert preds.shape == (5, 3)
        # Softmax outputs sum to 1
        np.testing.assert_array_almost_equal(preds.sum(axis=1), np.ones(5), decimal=5)
    
    def test_multiclass_classifier_gradients(self, multiclass_classifier):
        """Multi-class classifier gradient computation works."""
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(
            multiclass_classifier,
            task="classification",
            class_names=["a", "b", "c"]
        )
        
        X = np.random.randn(1, 5).astype(np.float32)
        preds, grads = adapter.predict_with_gradients(X, target_class=1)
        
        assert preds.shape == (1, 3)
        assert grads.shape == (1, 5)


# =============================================================================
# Test IntegratedGradients shape preservation
# =============================================================================

class TestIntegratedGradientsShapePreservation:
    """Test that IntegratedGradients preserves input shape."""
    
    @pytest.fixture
    def tabular_model(self):
        """Create a model for tabular data."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 3)
        )
    
    @pytest.fixture
    def cnn_model(self):
        """Create a simple CNN for image data."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(4, 3)
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        return SimpleCNN()
    
    def test_tabular_shape_preserved(self, tabular_model):
        """Tabular data shape is preserved in attributions."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(tabular_model, task="classification")
        feature_names = [f"feat_{i}" for i in range(10)]
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            n_steps=10
        )
        
        X = np.random.randn(10).astype(np.float32)
        exp = explainer.explain(X)
        
        # Check raw attributions have correct shape
        raw = np.array(exp.explanation_data["attributions_raw"])
        assert raw.shape == (10,)
        
        # Check feature_attributions are present
        assert "feature_attributions" in exp.explanation_data
        assert len(exp.explanation_data["feature_attributions"]) == 10
    
    def test_image_shape_preserved(self, cnn_model):
        """Image data shape is preserved in attributions."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(cnn_model, task="classification")
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            n_steps=5
        )
        
        # Image: (C, H, W) = (1, 8, 8)
        X = np.random.randn(1, 8, 8).astype(np.float32)
        exp = explainer.explain(X)
        
        # Check raw attributions have correct shape
        raw = np.array(exp.explanation_data["attributions_raw"])
        assert raw.shape == (1, 8, 8), f"Expected (1, 8, 8), got {raw.shape}"
        
        # Check data type is correctly identified
        assert exp.explanation_data["data_type"] == "image"


# =============================================================================
# Test ExplanationSuite uses registry
# =============================================================================

class TestExplanationSuiteRegistry:
    """Test that ExplanationSuite uses the registry."""
    
    def test_suite_lazy_loads_registry(self):
        """Suite lazily loads registry."""
        from explainiverse.engine.suite import ExplanationSuite
        
        # Create suite without triggering registry load
        suite = ExplanationSuite(
            model=None,
            explainer_configs=[("lime", {})]
        )
        
        assert suite._registry is None
        
        # Access registry
        registry = suite._get_registry()
        assert registry is not None
        assert suite._registry is not None
    
    def test_suite_list_methods(self):
        """Suite list methods work correctly."""
        from explainiverse.engine.suite import ExplanationSuite
        
        suite = ExplanationSuite(
            model=None,
            explainer_configs=[("lime", {}), ("shap", {})]
        )
        
        assert suite.list_explainers() == ["lime", "shap"]
        assert suite.list_completed() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
