# tests/test_integrated_gradients.py
"""
Tests for Integrated Gradients explainer.

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
    
    # Initialize with deterministic weights for reproducibility
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


class TestIntegratedGradientsBasic:
    """Basic functionality tests for Integrated Gradients."""
    
    def test_ig_creation(self, simple_classifier, feature_names, class_names):
        """Integrated Gradients explainer can be created."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_steps=50
        )
        
        assert explainer.n_steps == 50
        assert explainer.feature_names == feature_names
        assert explainer.class_names == class_names
    
    def test_ig_rejects_non_gradient_model(self, feature_names):
        """IG raises error for models without gradient support."""
        from explainiverse.explainers import IntegratedGradientsExplainer
        from explainiverse.adapters import SklearnAdapter
        from sklearn.linear_model import LogisticRegression
        
        # SklearnAdapter doesn't have predict_with_gradients
        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)
        
        with pytest.raises(TypeError, match="predict_with_gradients"):
            IntegratedGradientsExplainer(
                model=adapter,
                feature_names=feature_names
            )
    
    def test_ig_explain_classification(self, simple_classifier, sample_data, feature_names, class_names):
        """IG produces valid explanations for classification."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_steps=20
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "IntegratedGradients"
        assert "feature_attributions" in explanation.explanation_data
        assert len(explanation.explanation_data["feature_attributions"]) == 4
    
    def test_ig_explain_regression(self, simple_regressor, sample_data, feature_names):
        """IG produces valid explanations for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            n_steps=20
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert "feature_attributions" in explanation.explanation_data
    
    def test_ig_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """IG respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_steps=20
        )
        
        explanation_0 = explainer.explain(sample_data[0], target_class=0)
        explanation_1 = explainer.explain(sample_data[0], target_class=1)
        
        # Different target classes should produce different attributions
        attr_0 = list(explanation_0.explanation_data["feature_attributions"].values())
        attr_1 = list(explanation_1.explanation_data["feature_attributions"].values())
        
        assert not np.allclose(attr_0, attr_1)
        assert explanation_0.target_class == "class_a"
        assert explanation_1.target_class == "class_b"


class TestIntegratedGradientsConvergence:
    """Tests for IG convergence (completeness axiom)."""
    
    def test_ig_convergence_delta(self, simple_classifier, sample_data, feature_names, class_names):
        """IG attributions should sum to approximately F(x) - F(baseline)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_steps=100  # More steps for better convergence
        )
        
        explanation = explainer.explain(
            sample_data[0],
            return_convergence_delta=True
        )
        
        # Convergence delta should be small
        delta = explanation.explanation_data["convergence_delta"]
        pred_diff = explanation.explanation_data["prediction_difference"]
        attr_sum = explanation.explanation_data["attribution_sum"]
        
        # The sum of attributions should be close to prediction difference
        # With 100 steps, error should be relatively small
        assert delta < 0.5 * abs(pred_diff) + 0.05  # Allow 50% relative error + small absolute


class TestIntegratedGradientsMethods:
    """Tests for different integration methods."""
    
    def test_ig_riemann_methods(self, simple_classifier, sample_data, feature_names, class_names):
        """Different Riemann methods should produce similar results."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        methods = ["riemann_left", "riemann_right", "riemann_middle", "riemann_trapezoid"]
        results = {}
        
        for method in methods:
            explainer = IntegratedGradientsExplainer(
                model=adapter,
                feature_names=feature_names,
                class_names=class_names,
                n_steps=50,
                method=method
            )
            
            explanation = explainer.explain(sample_data[0], target_class=0)
            results[method] = explanation.explanation_data["attributions_raw"]
        
        # All methods should produce reasonably similar results
        for method in methods[1:]:
            # Correlation should be high
            corr = np.corrcoef(results["riemann_middle"], results[method])[0, 1]
            assert corr > 0.9, f"Method {method} has low correlation with riemann_middle"


class TestIntegratedGradientsBaselines:
    """Tests for different baseline options."""
    
    def test_ig_zero_baseline(self, simple_classifier, sample_data, feature_names, class_names):
        """Default zero baseline works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            baseline=None  # Default: zeros
        )
        
        explanation = explainer.explain(sample_data[0])
        
        # Check that baseline is zeros
        baseline = explanation.explanation_data["baseline"]
        assert all(b == 0 for b in baseline)
    
    def test_ig_custom_baseline(self, simple_classifier, sample_data, feature_names, class_names):
        """Custom baseline works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        custom_baseline = np.ones(4, dtype=np.float32) * 0.5
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            baseline=custom_baseline
        )
        
        explanation = explainer.explain(sample_data[0])
        
        # Check that baseline is our custom one
        baseline = explanation.explanation_data["baseline"]
        assert np.allclose(baseline, custom_baseline)
    
    def test_ig_override_baseline(self, simple_classifier, sample_data, feature_names, class_names):
        """Baseline can be overridden per-explanation."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            baseline=None  # Default: zeros
        )
        
        override_baseline = np.ones(4, dtype=np.float32) * -0.5
        
        explanation = explainer.explain(sample_data[0], baseline=override_baseline)
        
        # Check that override baseline was used
        baseline = explanation.explanation_data["baseline"]
        assert np.allclose(baseline, override_baseline)


class TestIntegratedGradientsBatch:
    """Tests for batch processing."""
    
    def test_ig_batch_explain(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain processes multiple instances."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_steps=20
        )
        
        explanations = explainer.explain_batch(sample_data[:5])
        
        assert len(explanations) == 5
        for exp in explanations:
            assert "feature_attributions" in exp.explanation_data


class TestIntegratedGradientsSmoothGrad:
    """Tests for SmoothGrad-style noisy averaging."""
    
    def test_ig_smooth(self, simple_classifier, sample_data, feature_names, class_names):
        """Smooth IG with noisy baselines works."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_steps=20
        )
        
        explanation = explainer.compute_attributions_with_noise(
            sample_data[0],
            target_class=0,
            n_samples=5,
            noise_scale=0.1
        )
        
        assert explanation.explainer_name == "IntegratedGradients_Smooth"
        assert "feature_attributions" in explanation.explanation_data
        assert "attributions_std" in explanation.explanation_data


class TestIntegratedGradientsRegistry:
    """Tests for registry integration."""
    
    def test_ig_registered(self):
        """Integrated Gradients is registered in default registry."""
        from explainiverse import default_registry
        
        explainers = default_registry.list_explainers()
        assert "integrated_gradients" in explainers
    
    def test_ig_metadata(self):
        """Integrated Gradients has correct metadata."""
        from explainiverse import default_registry
        
        meta = default_registry.get_meta("integrated_gradients")
        
        assert meta.scope == "local"
        assert "neural" in meta.model_types
        assert "tabular" in meta.data_types
        assert "image" in meta.data_types
        assert "Sundararajan" in meta.paper_reference
    
    def test_ig_filter_neural(self):
        """IG appears when filtering for neural network explainers."""
        from explainiverse import default_registry
        
        neural_explainers = default_registry.filter(model_type="neural")
        assert "integrated_gradients" in neural_explainers
    
    def test_ig_via_registry(self, simple_classifier, feature_names, class_names):
        """IG can be created via registry."""
        from explainiverse import default_registry
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = default_registry.create(
            "integrated_gradients",
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        assert explainer is not None
        assert explainer.feature_names == feature_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
