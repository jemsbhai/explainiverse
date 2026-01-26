# tests/test_deeplift.py
"""
Tests for DeepLIFT and DeepSHAP explainers.

These tests require PyTorch to be installed. They will be skipped
if torch is not available.
"""

import pytest
import numpy as np

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed"
)


@pytest.fixture
def simple_classifier():
    """Create a simple PyTorch classifier for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 3)
    )
    
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    return model


@pytest.fixture
def simple_regressor():
    """Create a simple PyTorch regressor for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    return model


@pytest.fixture
def sample_data():
    """Create sample input data."""
    np.random.seed(42)
    X = np.random.randn(10, 4).astype(np.float32)
    return X


@pytest.fixture
def feature_names():
    return ["feature_0", "feature_1", "feature_2", "feature_3"]


@pytest.fixture
def class_names():
    return ["class_a", "class_b", "class_c"]


class TestDeepLIFTBasic:
    """Basic functionality tests for DeepLIFT."""
    
    def test_deeplift_creation(self, simple_classifier, feature_names, class_names):
        """DeepLIFT explainer can be created."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        assert explainer.feature_names == feature_names
        assert explainer.class_names == class_names
        assert explainer.multiply_by_inputs is True
    
    def test_deeplift_rejects_non_gradient_model(self, feature_names):
        """DeepLIFT raises error for models without gradient support."""
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        from explainiverse.adapters import SklearnAdapter
        from sklearn.linear_model import LogisticRegression
        
        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)
        
        with pytest.raises(TypeError, match="predict_with_gradients"):
            DeepLIFTExplainer(
                model=adapter,
                feature_names=feature_names
            )
    
    def test_deeplift_explain_classification(self, simple_classifier, sample_data, feature_names, class_names):
        """DeepLIFT produces valid explanations for classification."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "DeepLIFT"
        assert "feature_attributions" in explanation.explanation_data
        assert len(explanation.explanation_data["feature_attributions"]) == 4
    
    def test_deeplift_explain_regression(self, simple_regressor, sample_data, feature_names):
        """DeepLIFT produces valid explanations for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert "feature_attributions" in explanation.explanation_data
    
    def test_deeplift_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """DeepLIFT respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation_0 = explainer.explain(sample_data[0], target_class=0)
        explanation_1 = explainer.explain(sample_data[0], target_class=1)
        
        attr_0 = list(explanation_0.explanation_data["feature_attributions"].values())
        attr_1 = list(explanation_1.explanation_data["feature_attributions"].values())
        
        assert not np.allclose(attr_0, attr_1)
        assert explanation_0.target_class == "class_a"
        assert explanation_1.target_class == "class_b"


class TestDeepLIFTMethods:
    """Tests for different DeepLIFT methods."""
    
    def test_deeplift_rescale_method(self, simple_classifier, sample_data, feature_names, class_names):
        """Rescale method produces valid attributions."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0], method="rescale")
        
        assert explanation.explanation_data["method"] == "rescale"
        assert "attributions_raw" in explanation.explanation_data
    
    def test_deeplift_rescale_exact_method(self, simple_classifier, sample_data, feature_names, class_names):
        """Rescale exact method produces valid attributions."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0], method="rescale_exact")
        
        assert explanation.explanation_data["method"] == "rescale_exact"
    
    def test_deeplift_methods_similar(self, simple_classifier, sample_data, feature_names, class_names):
        """Different methods produce similar results."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        exp_rescale = explainer.explain(sample_data[0], method="rescale")
        exp_exact = explainer.explain(sample_data[0], method="rescale_exact")
        
        attr_rescale = exp_rescale.explanation_data["attributions_raw"]
        attr_exact = exp_exact.explanation_data["attributions_raw"]
        
        # Methods should correlate well
        corr = np.corrcoef(attr_rescale, attr_exact)[0, 1]
        assert corr > 0.8


class TestDeepLIFTBaselines:
    """Tests for different baseline options."""
    
    def test_deeplift_zero_baseline(self, simple_classifier, sample_data, feature_names, class_names):
        """Default zero baseline works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            baseline=None
        )
        
        explanation = explainer.explain(sample_data[0])
        
        baseline = explanation.explanation_data["baseline"]
        assert all(b == 0 for b in baseline)
    
    def test_deeplift_custom_baseline(self, simple_classifier, sample_data, feature_names, class_names):
        """Custom baseline works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        custom_baseline = np.ones(4, dtype=np.float32) * 0.5
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            baseline=custom_baseline
        )
        
        explanation = explainer.explain(sample_data[0])
        
        baseline = explanation.explanation_data["baseline"]
        assert np.allclose(baseline, custom_baseline)
    
    def test_deeplift_set_baseline_from_data(self, simple_classifier, sample_data, feature_names, class_names):
        """set_baseline from data works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explainer.set_baseline(sample_data, method="mean")
        
        explanation = explainer.explain(sample_data[0])
        
        expected_baseline = np.mean(sample_data, axis=0)
        actual_baseline = explanation.explanation_data["baseline"]
        assert np.allclose(actual_baseline, expected_baseline, atol=1e-5)


class TestDeepLIFTConvergence:
    """Tests for DeepLIFT summation-to-delta property."""
    
    def test_deeplift_convergence_delta(self, simple_classifier, sample_data, feature_names, class_names):
        """DeepLIFT attributions approximate F(x) - F(baseline)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(
            sample_data[0],
            return_convergence_delta=True
        )
        
        delta = explanation.explanation_data["convergence_delta"]
        pred_diff = abs(explanation.explanation_data["prediction_difference"])
        
        # Convergence should be reasonable
        assert delta < pred_diff + 0.1


class TestDeepLIFTMultipleBaselines:
    """Tests for multiple baselines averaging."""
    
    def test_deeplift_multiple_baselines(self, simple_classifier, sample_data, feature_names, class_names):
        """explain_with_multiple_baselines works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        baselines = sample_data[:5]  # Use first 5 samples as baselines
        
        explanation = explainer.explain_with_multiple_baselines(
            sample_data[5],
            baselines=baselines,
            target_class=0
        )
        
        assert explanation.explainer_name == "DeepLIFT_MultiBaseline"
        assert "attributions_std" in explanation.explanation_data
        assert explanation.explanation_data["n_baselines"] == 5


class TestDeepLIFTBatch:
    """Tests for batch processing."""
    
    def test_deeplift_batch_explain(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain processes multiple instances."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanations = explainer.explain_batch(sample_data[:5])
        
        assert len(explanations) == 5
        for exp in explanations:
            assert "feature_attributions" in exp.explanation_data


class TestDeepLIFTCompareIG:
    """Tests comparing DeepLIFT to Integrated Gradients."""
    
    def test_deeplift_ig_comparison(self, simple_classifier, sample_data, feature_names, class_names):
        """DeepLIFT compares favorably with Integrated Gradients."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        comparison = explainer.compare_with_integrated_gradients(
            sample_data[0],
            target_class=0,
            ig_steps=50
        )
        
        # DeepLIFT and IG should be highly correlated for ReLU networks
        assert comparison["correlation"] > 0.8
        assert "mse" in comparison
        assert "max_difference" in comparison


class TestDeepSHAPBasic:
    """Tests for DeepSHAP (DeepLIFT + multiple baselines)."""
    
    def test_deepshap_creation(self, simple_classifier, sample_data, feature_names, class_names):
        """DeepSHAP explainer can be created."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTShapExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = DeepLIFTShapExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            background_data=sample_data
        )
        
        assert explainer.feature_names == feature_names
        assert explainer._background_data is not None
    
    def test_deepshap_set_background(self, simple_classifier, sample_data, feature_names, class_names):
        """set_background works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTShapExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = DeepLIFTShapExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explainer.set_background(sample_data)
        
        assert explainer._background_data is not None
        assert len(explainer._background_data) == len(sample_data)
    
    def test_deepshap_requires_background(self, simple_classifier, sample_data, feature_names, class_names):
        """DeepSHAP raises error without background data."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTShapExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = DeepLIFTShapExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        with pytest.raises(ValueError, match="Background data not set"):
            explainer.explain(sample_data[0])
    
    def test_deepshap_explain(self, simple_classifier, sample_data, feature_names, class_names):
        """DeepSHAP produces valid explanations."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTShapExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = DeepLIFTShapExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            background_data=sample_data[:5]
        )
        
        explanation = explainer.explain(sample_data[5])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "DeepSHAP"
        assert "feature_attributions" in explanation.explanation_data
        assert "attributions_std" in explanation.explanation_data
        assert explanation.explanation_data["n_background_samples"] == 5


class TestDeepLIFTRegistry:
    """Tests for registry integration."""
    
    def test_deeplift_registered(self):
        """DeepLIFT is registered in default registry."""
        from explainiverse import default_registry
        
        explainers = default_registry.list_explainers()
        assert "deeplift" in explainers
    
    def test_deepshap_registered(self):
        """DeepSHAP is registered in default registry."""
        from explainiverse import default_registry
        
        explainers = default_registry.list_explainers()
        assert "deepshap" in explainers
    
    def test_deeplift_metadata(self):
        """DeepLIFT has correct metadata."""
        from explainiverse import default_registry
        
        meta = default_registry.get_meta("deeplift")
        
        assert meta.scope == "local"
        assert "neural" in meta.model_types
        assert "tabular" in meta.data_types
        assert "Shrikumar" in meta.paper_reference
    
    def test_deeplift_filter_neural(self):
        """DeepLIFT appears when filtering for neural network explainers."""
        from explainiverse import default_registry
        
        neural_explainers = default_registry.filter(model_type="neural")
        assert "deeplift" in neural_explainers
        assert "deepshap" in neural_explainers
    
    def test_deeplift_via_registry(self, simple_classifier, feature_names, class_names):
        """DeepLIFT can be created via registry."""
        from explainiverse import default_registry
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = default_registry.create(
            "deeplift",
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        assert explainer is not None
        assert explainer.feature_names == feature_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
