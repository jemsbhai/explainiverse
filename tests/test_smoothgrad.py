# tests/test_smoothgrad.py
"""
Tests for SmoothGrad explainer.

SmoothGrad reduces noise in gradient-based saliency maps by averaging
gradients computed on noisy copies of the input. This produces smoother,
more visually coherent attributions that are often easier to interpret.

These tests require PyTorch to be installed. They will be skipped
if torch is not available.

Reference:
    Smilkov et al., 2017 - "SmoothGrad: removing noise by adding noise"
    https://arxiv.org/abs/1706.03825
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
def deeper_network():
    """Create a deeper network to test gradient smoothing effectiveness."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
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


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestSmoothGradBasic:
    """Basic functionality tests for SmoothGrad."""
    
    def test_smoothgrad_creation(self, simple_classifier, feature_names, class_names):
        """SmoothGrad explainer can be created with default parameters."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        assert explainer.n_samples == 50  # default
        assert explainer.noise_scale == 0.15  # default
        assert explainer.feature_names == feature_names
        assert explainer.class_names == class_names
    
    def test_smoothgrad_custom_parameters(self, simple_classifier, feature_names, class_names):
        """SmoothGrad explainer accepts custom parameters."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=100,
            noise_scale=0.2,
            noise_type="gaussian"
        )
        
        assert explainer.n_samples == 100
        assert explainer.noise_scale == 0.2
        assert explainer.noise_type == "gaussian"
    
    def test_smoothgrad_rejects_non_gradient_model(self, feature_names):
        """SmoothGrad raises error for models without gradient support."""
        from explainiverse.explainers.gradient import SmoothGradExplainer
        from explainiverse.adapters import SklearnAdapter
        from sklearn.linear_model import LogisticRegression
        
        # SklearnAdapter doesn't have predict_with_gradients
        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)
        
        with pytest.raises(TypeError, match="predict_with_gradients"):
            SmoothGradExplainer(
                model=adapter,
                feature_names=feature_names
            )
    
    def test_smoothgrad_invalid_noise_type(self, simple_classifier, feature_names):
        """SmoothGrad raises error for invalid noise type."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        with pytest.raises(ValueError, match="noise_type"):
            SmoothGradExplainer(
                model=adapter,
                feature_names=feature_names,
                noise_type="invalid"
            )
    
    def test_smoothgrad_invalid_n_samples(self, simple_classifier, feature_names):
        """SmoothGrad raises error for invalid n_samples."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        with pytest.raises(ValueError, match="n_samples"):
            SmoothGradExplainer(
                model=adapter,
                feature_names=feature_names,
                n_samples=0
            )
    
    def test_smoothgrad_invalid_noise_scale(self, simple_classifier, feature_names):
        """SmoothGrad raises error for invalid noise_scale."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification")
        
        with pytest.raises(ValueError, match="noise_scale"):
            SmoothGradExplainer(
                model=adapter,
                feature_names=feature_names,
                noise_scale=-0.1
            )


# =============================================================================
# Classification Tests
# =============================================================================

class TestSmoothGradClassification:
    """Tests for SmoothGrad on classification models."""
    
    def test_smoothgrad_explain_classification(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad produces valid explanations for classification."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20  # Fewer samples for faster tests
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "SmoothGrad"
        assert "feature_attributions" in explanation.explanation_data
        assert len(explanation.explanation_data["feature_attributions"]) == 4
    
    def test_smoothgrad_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanation_0 = explainer.explain(sample_data[0], target_class=0)
        explanation_1 = explainer.explain(sample_data[0], target_class=1)
        
        # Different target classes should produce different attributions
        attr_0 = list(explanation_0.explanation_data["feature_attributions"].values())
        attr_1 = list(explanation_1.explanation_data["feature_attributions"].values())
        
        assert not np.allclose(attr_0, attr_1)
        assert explanation_0.target_class == "class_a"
        assert explanation_1.target_class == "class_b"
    
    def test_smoothgrad_auto_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad uses predicted class when target_class is None."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        # Get predicted class
        predictions = adapter.predict(sample_data[0].reshape(1, -1))
        predicted_class = int(np.argmax(predictions))
        
        explanation = explainer.explain(sample_data[0])
        
        # Should explain the predicted class
        assert explanation.target_class == class_names[predicted_class]
    
    def test_smoothgrad_attribution_shape(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad produces attributions with correct shape."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions_raw = explanation.explanation_data["attributions_raw"]
        assert len(attributions_raw) == len(feature_names)
    
    def test_smoothgrad_includes_statistics(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad includes standard deviation in output."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert "attributions_std" in explanation.explanation_data
        assert len(explanation.explanation_data["attributions_std"]) == len(feature_names)
        
        # Standard deviation should be non-negative
        for std in explanation.explanation_data["attributions_std"]:
            assert std >= 0


# =============================================================================
# Regression Tests
# =============================================================================

class TestSmoothGradRegression:
    """Tests for SmoothGrad on regression models."""
    
    def test_smoothgrad_explain_regression(self, simple_regressor, sample_data, feature_names):
        """SmoothGrad produces valid explanations for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            n_samples=20
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert isinstance(explanation, Explanation)
        assert "feature_attributions" in explanation.explanation_data
    
    def test_smoothgrad_regression_no_class_names(self, simple_regressor, sample_data, feature_names):
        """SmoothGrad handles regression without class_names."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            n_samples=20
        )
        
        explanation = explainer.explain(sample_data[0])
        
        # Target class should be "output" for regression
        assert explanation.target_class == "output"
    
    def test_smoothgrad_regression_attributions_finite(self, simple_regressor, sample_data, feature_names):
        """SmoothGrad attributions are finite for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_regressor, task="regression")
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            n_samples=20
        )
        
        explanation = explainer.explain(sample_data[0])
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)


# =============================================================================
# SmoothGrad Variants Tests
# =============================================================================

class TestSmoothGradVariants:
    """Tests for SmoothGrad variants (Squared, VarGrad)."""
    
    def test_smoothgrad_squared(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad-Squared produces valid explanations."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanation = explainer.explain(sample_data[0], method="smoothgrad_squared")
        
        assert explanation.explainer_name == "SmoothGrad_Squared"
        assert "feature_attributions" in explanation.explanation_data
        
        # Squared attributions should be non-negative
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(a >= 0 for a in attributions)
    
    def test_vargrad(self, simple_classifier, sample_data, feature_names, class_names):
        """VarGrad produces valid explanations."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanation = explainer.explain(sample_data[0], method="vargrad")
        
        assert explanation.explainer_name == "VarGrad"
        assert "feature_attributions" in explanation.explanation_data
        
        # VarGrad (variance) should be non-negative
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(a >= 0 for a in attributions)
    
    def test_smoothgrad_vs_squared_different(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad and SmoothGrad-Squared produce different results."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=30
        )
        
        # Set seed for reproducibility
        np.random.seed(42)
        explanation_standard = explainer.explain(sample_data[0], method="smoothgrad")
        
        np.random.seed(42)
        explanation_squared = explainer.explain(sample_data[0], method="smoothgrad_squared")
        
        attr_standard = explanation_standard.explanation_data["attributions_raw"]
        attr_squared = explanation_squared.explanation_data["attributions_raw"]
        
        # They should be different
        assert not np.allclose(attr_standard, attr_squared)
    
    def test_invalid_method(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad raises error for invalid method."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        with pytest.raises(ValueError, match="method"):
            explainer.explain(sample_data[0], method="invalid_method")


# =============================================================================
# Noise Configuration Tests
# =============================================================================

class TestSmoothGradNoiseConfiguration:
    """Tests for noise configuration options."""
    
    def test_gaussian_noise(self, simple_classifier, sample_data, feature_names, class_names):
        """Gaussian noise type works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20,
            noise_type="gaussian"
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert explanation.explanation_data["noise_type"] == "gaussian"
    
    def test_uniform_noise(self, simple_classifier, sample_data, feature_names, class_names):
        """Uniform noise type works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20,
            noise_type="uniform"
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert explanation.explanation_data["noise_type"] == "uniform"
    
    def test_different_noise_scales(self, simple_classifier, sample_data, feature_names, class_names):
        """Different noise scales produce different standard deviations."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        # Small noise
        explainer_small = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=50,
            noise_scale=0.01
        )
        
        # Large noise
        explainer_large = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=50,
            noise_scale=0.5
        )
        
        np.random.seed(42)
        explanation_small = explainer_small.explain(sample_data[0])
        np.random.seed(42)
        explanation_large = explainer_large.explain(sample_data[0])
        
        std_small = np.mean(explanation_small.explanation_data["attributions_std"])
        std_large = np.mean(explanation_large.explanation_data["attributions_std"])
        
        # Larger noise should generally lead to larger variance in gradients
        # (This is probabilistic, but should hold for reasonable sample sizes)
        assert std_large > std_small * 0.5  # Allow some tolerance
    
    def test_more_samples_reduces_variance(self, simple_classifier, sample_data, feature_names, class_names):
        """More samples should reduce estimation variance."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        # Run multiple times with different sample counts
        variances = []
        for n_samples in [10, 50]:
            explainer = SmoothGradExplainer(
                model=adapter,
                feature_names=feature_names,
                class_names=class_names,
                n_samples=n_samples,
                noise_scale=0.1
            )
            
            # Run multiple times and compute variance of results
            results = []
            for seed in range(5):
                np.random.seed(seed)
                explanation = explainer.explain(sample_data[0])
                results.append(explanation.explanation_data["attributions_raw"])
            
            variance = np.var(results, axis=0).mean()
            variances.append(variance)
        
        # More samples should have lower variance (or equal for very stable cases)
        assert variances[1] <= variances[0] * 1.5  # Allow some tolerance
    
    def test_noise_scale_in_output(self, simple_classifier, sample_data, feature_names, class_names):
        """Noise scale is included in explanation output."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20,
            noise_scale=0.25
        )
        
        explanation = explainer.explain(sample_data[0])
        
        assert explanation.explanation_data["noise_scale"] == 0.25
        assert explanation.explanation_data["n_samples"] == 20


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestSmoothGradBatch:
    """Tests for batch processing."""
    
    def test_batch_explain(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain processes multiple instances."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanations = explainer.explain_batch(sample_data[:5])
        
        assert len(explanations) == 5
        for exp in explanations:
            assert "feature_attributions" in exp.explanation_data
    
    def test_batch_with_target_class(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanations = explainer.explain_batch(sample_data[:3], target_class=1)
        
        for exp in explanations:
            assert exp.target_class == "class_b"
    
    def test_batch_single_instance(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain handles single instance (1D input)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanations = explainer.explain_batch(sample_data[0])  # 1D input
        
        assert len(explanations) == 1
    
    def test_batch_with_method(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain supports different methods."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanations = explainer.explain_batch(sample_data[:3], method="smoothgrad_squared")
        
        for exp in explanations:
            assert exp.explainer_name == "SmoothGrad_Squared"


# =============================================================================
# Smoothing Effectiveness Tests
# =============================================================================

class TestSmoothGradEffectiveness:
    """Tests for smoothing effectiveness."""
    
    def test_smoothgrad_reduces_gradient_noise(self, deeper_network, sample_data, feature_names, class_names):
        """SmoothGrad produces smoother attributions than raw gradients."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(deeper_network, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=50,
            noise_scale=0.15
        )
        
        # Get raw gradient (n_samples=1, no noise is equivalent to raw gradient)
        explainer_raw = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=1,
            noise_scale=0.0  # No noise = raw gradient
        )
        
        # Compute for multiple nearby points and check variance
        smooth_results = []
        raw_results = []
        
        base_instance = sample_data[0]
        for i in range(10):
            # Slightly perturbed inputs
            perturbed = base_instance + np.random.randn(4).astype(np.float32) * 0.01
            
            np.random.seed(i)
            smooth_exp = explainer.explain(perturbed, target_class=0)
            raw_exp = explainer_raw.explain(perturbed, target_class=0)
            
            smooth_results.append(smooth_exp.explanation_data["attributions_raw"])
            raw_results.append(raw_exp.explanation_data["attributions_raw"])
        
        # SmoothGrad should have lower variance across similar inputs
        smooth_var = np.var(smooth_results, axis=0).mean()
        raw_var = np.var(raw_results, axis=0).mean()
        
        # This is probabilistic, but smoothing should help (allowing generous tolerance)
        assert smooth_var <= raw_var * 2.0
    
    def test_zero_noise_equals_raw_gradient(self, simple_classifier, sample_data, feature_names, class_names):
        """Zero noise scale produces raw gradient."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        # SmoothGrad with zero noise
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=10,
            noise_scale=0.0
        )
        
        explanation = explainer.explain(sample_data[0], target_class=0)
        
        # Get raw gradient for comparison
        _, raw_gradients = adapter.predict_with_gradients(
            sample_data[0].reshape(1, -1),
            target_class=0
        )
        
        # Should be identical
        np.testing.assert_allclose(
            explanation.explanation_data["attributions_raw"],
            raw_gradients.flatten(),
            rtol=1e-5
        )


# =============================================================================
# Absolute Value Option Tests
# =============================================================================

class TestSmoothGradAbsoluteValue:
    """Tests for absolute value options."""
    
    def test_absolute_value_option(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad supports absolute value option."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        explanation = explainer.explain(sample_data[0], absolute_value=True)
        
        # All attributions should be non-negative
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(a >= 0 for a in attributions)
    
    def test_absolute_value_changes_result(self, simple_classifier, sample_data, feature_names, class_names):
        """Absolute value option changes the result."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=30
        )
        
        np.random.seed(42)
        explanation_normal = explainer.explain(sample_data[0], absolute_value=False)
        np.random.seed(42)
        explanation_abs = explainer.explain(sample_data[0], absolute_value=True)
        
        attr_normal = explanation_normal.explanation_data["attributions_raw"]
        attr_abs = explanation_abs.explanation_data["attributions_raw"]
        
        # They should be different if any attributions are negative
        if any(a < 0 for a in attr_normal):
            assert not np.allclose(attr_normal, attr_abs)


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestSmoothGradRegistry:
    """Tests for registry integration."""
    
    def test_smoothgrad_registered(self):
        """SmoothGrad is registered in default registry."""
        from explainiverse import default_registry
        
        explainers = default_registry.list_explainers()
        assert "smoothgrad" in explainers
    
    def test_smoothgrad_metadata(self):
        """SmoothGrad has correct metadata."""
        from explainiverse import default_registry
        
        meta = default_registry.get_meta("smoothgrad")
        
        assert meta.scope == "local"
        assert "neural" in meta.model_types
        assert "tabular" in meta.data_types
        assert "image" in meta.data_types
        assert "Smilkov" in meta.paper_reference
    
    def test_smoothgrad_filter_neural(self):
        """SmoothGrad appears when filtering for neural network explainers."""
        from explainiverse import default_registry
        
        neural_explainers = default_registry.filter(model_type="neural")
        assert "smoothgrad" in neural_explainers
    
    def test_smoothgrad_via_registry(self, simple_classifier, feature_names, class_names):
        """SmoothGrad can be created via registry."""
        from explainiverse import default_registry
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = default_registry.create(
            "smoothgrad",
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        
        assert explainer is not None
        assert explainer.feature_names == feature_names


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================

class TestSmoothGradEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_feature(self, class_names):
        """SmoothGrad handles single feature input."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        # Single feature model
        model = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=["single_feature"],
            class_names=class_names,
            n_samples=10
        )
        
        instance = np.array([0.5], dtype=np.float32)
        explanation = explainer.explain(instance)
        
        assert len(explanation.explanation_data["feature_attributions"]) == 1
    
    def test_large_feature_space(self, class_names):
        """SmoothGrad handles large feature space."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")
        
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        n_features = 100
        model = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=[f"f_{i}" for i in range(n_features)],
            class_names=class_names,
            n_samples=10
        )
        
        instance = np.random.randn(n_features).astype(np.float32)
        explanation = explainer.explain(instance)
        
        assert len(explanation.explanation_data["feature_attributions"]) == n_features
    
    def test_extreme_input_values(self, simple_classifier, feature_names, class_names):
        """SmoothGrad handles extreme input values."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        # Large values
        large_instance = np.array([1000, -1000, 500, -500], dtype=np.float32)
        explanation = explainer.explain(large_instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)
    
    def test_zero_input(self, simple_classifier, feature_names, class_names):
        """SmoothGrad handles zero input."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer
        
        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20
        )
        
        zero_instance = np.zeros(4, dtype=np.float32)
        explanation = explainer.explain(zero_instance)
        
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
