# tests/test_saliency.py
"""
Tests for Saliency Maps explainer.

Saliency Maps compute feature attributions using the gradient of the output
with respect to the input. This is one of the simplest and fastest gradient-based
attribution methods.

These tests require PyTorch to be installed. They will be skipped
if torch is not available.

Reference:
    Simonyan et al., 2014 - "Deep Inside Convolutional Networks: Visualising
    Image Classification Models and Saliency Maps"
    https://arxiv.org/abs/1312.6034
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


# =============================================================================
# Fixtures
# =============================================================================

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
def linear_model():
    """Create a linear model (no nonlinearities) for testing gradient correctness."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Linear(4, 3)
    
    torch.manual_seed(42)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    
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


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestSaliencyBasic:
    """Basic functionality tests for Saliency Maps."""
    
    def test_saliency_creation(self, simple_classifier, feature_names, class_names):
        """Saliency explainer can be created with default parameters."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        assert explainer.feature_names == feature_names
        assert explainer.class_names == class_names
        assert explainer.absolute_value == True  # default
    
    def test_saliency_custom_parameters(self, simple_classifier, feature_names, class_names):
        """Saliency explainer accepts custom parameters."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            absolute_value=False
        )
        
        assert explainer.absolute_value == False
    
    def test_saliency_rejects_non_gradient_model(self, feature_names):
        """Saliency raises error for models without gradient support."""
        from explainiverse.explainers.gradient import SaliencyExplainer
        from explainiverse.adapters import SklearnAdapter
        from sklearn.linear_model import LogisticRegression
        
        # SklearnAdapter doesn't have predict_with_gradients
        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)
        
        with pytest.raises(TypeError, match="predict_with_gradients"):
            SaliencyExplainer(
                model=adapter,
                feature_names=feature_names
            )


# =============================================================================
# Classification Tests
# =============================================================================

class TestSaliencyClassification:
    """Tests for Saliency on classification models."""
    
    def test_saliency_explain_classification(self, simple_classifier, sample_data, feature_names, class_names):
        """Saliency produces valid explanations for classification."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "Saliency"
        assert "feature_attributions" in explanation.explanation_data
        assert len(explanation.explanation_data["feature_attributions"]) == 4
    
    def test_saliency_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """Saliency respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation_0 = explainer.explain(sample_data[0], target_class=0)
        explanation_1 = explainer.explain(sample_data[0], target_class=1)
        
        # Different target classes should produce different attributions
        attr_0 = list(explanation_0.explanation_data["feature_attributions"].values())
        attr_1 = list(explanation_1.explanation_data["feature_attributions"].values())
        
        assert not np.allclose(attr_0, attr_1)
        assert explanation_0.target_class == "class_a"
        assert explanation_1.target_class == "class_b"
    
    def test_saliency_auto_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """Saliency uses predicted class when target_class is None."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Get predicted class
        predictions = adapter.predict(sample_data[0].reshape(1, -1))
        predicted_class = int(np.argmax(predictions))
        
        explanation = explainer.explain(sample_data[0])
        
        # Should explain the predicted class
        assert explanation.target_class == class_names[predicted_class]
    
    def test_saliency_attribution_shape(self, simple_classifier, sample_data, feature_names, class_names):
        """Saliency produces attributions with correct shape."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions_raw = explanation.explanation_data["attributions_raw"]
        assert len(attributions_raw) == len(feature_names)
    
    def test_saliency_default_absolute_value(self, simple_classifier, sample_data, feature_names, class_names):
        """Default saliency uses absolute value (all non-negative)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            absolute_value=True
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(a >= 0 for a in attributions)


# =============================================================================
# Regression Tests
# =============================================================================

class TestSaliencyRegression:
    """Tests for Saliency on regression models."""
    
    def test_saliency_explain_regression(self, simple_regressor, sample_data, feature_names):
        """Saliency produces valid explanations for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert "feature_attributions" in explanation.explanation_data
    
    def test_saliency_regression_no_class_names(self, simple_regressor, sample_data, feature_names):
        """Saliency handles regression without class_names."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        # Target class should be "output" for regression
        assert explanation.target_class == "output"
    
    def test_saliency_regression_attributions_finite(self, simple_regressor, sample_data, feature_names):
        """Saliency attributions are finite for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)


# =============================================================================
# Saliency Variants Tests
# =============================================================================

class TestSaliencyVariants:
    """Tests for Saliency variants (signed, input×gradient)."""
    
    def test_saliency_signed(self, simple_classifier, sample_data, feature_names, class_names):
        """Signed saliency (raw gradient) can have negative values."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            absolute_value=False
        )
        
        # Test multiple instances to find one with negative gradients
        found_negative = False
        for i in range(len(sample_data)):
            explanation = explainer.explain(sample_data[i])
            attributions = list(explanation.explanation_data["feature_attributions"].values())
            if any(a < 0 for a in attributions):
                found_negative = True
                break
        
        # Should find at least one instance with negative gradients
        assert found_negative, "Expected at least one negative gradient in signed saliency"
    
    def test_input_times_gradient(self, simple_classifier, sample_data, feature_names, class_names):
        """Input × Gradient method works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_data[0], method="input_times_gradient")
        
        assert explanation.explainer_name == "InputTimesGradient"
        assert "feature_attributions" in explanation.explanation_data
    
    def test_saliency_vs_input_times_gradient_different(self, simple_classifier, sample_data, feature_names, class_names):
        """Saliency and Input×Gradient produce different results."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            absolute_value=False  # Keep signed for comparison
        )
        
        explanation_saliency = explainer.explain(sample_data[0], method="saliency")
        explanation_itg = explainer.explain(sample_data[0], method="input_times_gradient")
        
        attr_saliency = explanation_saliency.explanation_data["attributions_raw"]
        attr_itg = explanation_itg.explanation_data["attributions_raw"]
        
        # They should be different (unless all inputs are 1.0)
        assert not np.allclose(attr_saliency, attr_itg)
    
    def test_invalid_method(self, simple_classifier, sample_data, feature_names, class_names):
        """Saliency raises error for invalid method."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        with pytest.raises(ValueError, match="method"):
            explainer.explain(sample_data[0], method="invalid_method")


# =============================================================================
# Gradient Correctness Tests
# =============================================================================

class TestSaliencyGradientCorrectness:
    """Tests to verify gradient computation correctness."""
    
    def test_linear_model_gradient_equals_weights(self, linear_model, sample_data, feature_names, class_names):
        """For linear model, gradient equals the weight vector."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(linear_model, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            absolute_value=False
        )
        
        # For linear model f(x) = Wx + b, gradient = W[target_class]
        explanation = explainer.explain(sample_data[0], target_class=0)
        
        # Get the weight vector for class 0
        expected_gradient = linear_model.weight[0].detach().numpy()
        
        actual_gradient = np.array(explanation.explanation_data["attributions_raw"])
        
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-5)
    
    def test_gradient_independent_of_input_for_linear(self, linear_model, sample_data, feature_names, class_names):
        """For linear model, gradient is constant (independent of input)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(linear_model, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            absolute_value=False
        )
        
        # Gradient should be the same for any input
        explanation_1 = explainer.explain(sample_data[0], target_class=1)
        explanation_2 = explainer.explain(sample_data[5], target_class=1)
        
        attr_1 = np.array(explanation_1.explanation_data["attributions_raw"])
        attr_2 = np.array(explanation_2.explanation_data["attributions_raw"])
        
        np.testing.assert_allclose(attr_1, attr_2, rtol=1e-5)


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestSaliencyBatch:
    """Tests for batch processing."""
    
    def test_batch_explain(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain processes multiple instances."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanations = explainer.explain_batch(sample_data[:5])
        
        assert len(explanations) == 5
        for exp in explanations:
            assert "feature_attributions" in exp.explanation_data
    
    def test_batch_with_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanations = explainer.explain_batch(sample_data[:3], target_class=1)
        
        for exp in explanations:
            assert exp.target_class == "class_b"
    
    def test_batch_single_instance(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain handles single instance (1D input)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanations = explainer.explain_batch(sample_data[0])  # 1D input
        
        assert len(explanations) == 1
    
    def test_batch_with_method(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain supports different methods."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanations = explainer.explain_batch(sample_data[:3], method="input_times_gradient")
        
        for exp in explanations:
            assert exp.explainer_name == "InputTimesGradient"


# =============================================================================
# Performance Tests
# =============================================================================

class TestSaliencyPerformance:
    """Tests for performance characteristics."""
    
    def test_saliency_is_deterministic(self, simple_classifier, sample_data, feature_names, class_names):
        """Saliency produces identical results for same input."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        explanation_1 = explainer.explain(sample_data[0], target_class=0)
        explanation_2 = explainer.explain(sample_data[0], target_class=0)
        
        attr_1 = np.array(explanation_1.explanation_data["attributions_raw"])
        attr_2 = np.array(explanation_2.explanation_data["attributions_raw"])
        
        np.testing.assert_array_equal(attr_1, attr_2)
    
    def test_saliency_faster_than_smoothgrad(self, simple_classifier, sample_data, feature_names, class_names):
        """Saliency should be faster than SmoothGrad (single pass vs multiple)."""
        import time
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer, SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        saliency = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        smoothgrad = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        # Time saliency
        start = time.time()
        for _ in range(10):
            saliency.explain(sample_data[0])
        saliency_time = time.time() - start
        
        # Time smoothgrad
        start = time.time()
        for _ in range(10):
            smoothgrad.explain(sample_data[0])
        smoothgrad_time = time.time() - start
        
        # Saliency should be at least 5x faster
        assert saliency_time < smoothgrad_time / 5


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestSaliencyRegistry:
    """Tests for registry integration."""
    
    def test_saliency_registered(self):
        """Saliency is registered in default registry."""
        from explainiverse import default_registry
        
        explainers = default_registry.list_explainers()
        assert "saliency" in explainers
    
    def test_saliency_metadata(self):
        """Saliency has correct metadata."""
        from explainiverse import default_registry
        
        meta = default_registry.get_meta("saliency")
        
        assert meta.scope == "local"
        assert "neural" in meta.model_types
        assert "tabular" in meta.data_types
        assert "image" in meta.data_types
        assert "Simonyan" in meta.paper_reference
    
    def test_saliency_filter_neural(self):
        """Saliency appears when filtering for neural network explainers."""
        from explainiverse import default_registry
        
        neural_explainers = default_registry.filter(model_type="neural")
        assert "saliency" in neural_explainers
    
    def test_saliency_via_registry(self, simple_classifier, feature_names, class_names):
        """Saliency can be created via registry."""
        from explainiverse import default_registry
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = default_registry.create(
            "saliency",
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        assert explainer is not None
        assert explainer.feature_names == feature_names


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================

class TestSaliencyEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_feature(self, class_names):
        """Saliency handles single feature input."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        # Single feature model
        model = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=["single_feature"],
            class_names=class_names
        )
        
        instance = np.array([0.5], dtype=np.float32)
        explanation = explainer.explain(instance)
        
        assert len(explanation.explanation_data["feature_attributions"]) == 1
    
    def test_large_feature_space(self, class_names):
        """Saliency handles large feature space."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        n_features = 100
        model = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=[f"f_{i}" for i in range(n_features)],
            class_names=class_names
        )
        
        instance = np.random.randn(n_features).astype(np.float32)
        explanation = explainer.explain(instance)
        
        assert len(explanation.explanation_data["feature_attributions"]) == n_features
    
    def test_extreme_input_values(self, simple_classifier, feature_names, class_names):
        """Saliency handles extreme input values."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        # Large values
        large_instance = np.array([1000, -1000, 500, -500], dtype=np.float32)
        explanation = explainer.explain(large_instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_zero_input(self, simple_classifier, feature_names, class_names):
        """Saliency handles zero input."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        zero_instance = np.zeros(4, dtype=np.float32)
        explanation = explainer.explain(zero_instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_input_times_gradient_with_zero_input(self, simple_classifier, feature_names, class_names):
        """Input×Gradient gives zero attributions for zero input."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SaliencyExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SaliencyExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        zero_instance = np.zeros(4, dtype=np.float32)
        explanation = explainer.explain(zero_instance, method="input_times_gradient")
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        # Input × Gradient should be all zeros for zero input
        assert all(a == 0 for a in attributions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
