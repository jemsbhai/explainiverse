# tests/test_smoothgrad.py
"""
Tests for SmoothGrad explainer.

This implementation averages raw gradients computed on noisy copies of the
input. These tests do not assert perceptual smoothness, interpretability,
uncertainty, or explanation quality.

These tests require PyTorch to be installed. They will be skipped
if torch is not available.

Reference:
    Smilkov et al., 2017 - "SmoothGrad: removing noise by adding noise"
    https://arxiv.org/abs/1706.03825
"""

import numpy as np
import pytest

# Check if torch is available
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_classifier():
    """Create a simple PyTorch classifier for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")

    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))

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

    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))

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
            model=adapter, feature_names=feature_names, class_names=class_names
        )

        assert explainer.n_samples == 50  # default
        assert explainer.noise_scale == 0.15  # default
        assert explainer.random_state is None
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
            noise_type="gaussian",
            random_state=17,
        )

        assert explainer.n_samples == 100
        assert explainer.noise_scale == 0.2
        assert explainer.noise_type == "gaussian"
        assert explainer.random_state == 17

    def test_smoothgrad_rejects_non_gradient_model(self, feature_names):
        """SmoothGrad raises error for models without gradient support."""
        from sklearn.linear_model import LogisticRegression

        from explainiverse.adapters import SklearnAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        # SklearnAdapter doesn't have predict_with_gradients
        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)

        with pytest.raises(TypeError, match="predict_with_gradients"):
            SmoothGradExplainer(model=adapter, feature_names=feature_names)

    def test_smoothgrad_invalid_noise_type(self, simple_classifier, feature_names):
        """SmoothGrad raises error for invalid noise type."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")

        with pytest.raises(ValueError, match="noise_type"):
            SmoothGradExplainer(model=adapter, feature_names=feature_names, noise_type="invalid")

    def test_smoothgrad_invalid_n_samples(self, simple_classifier, feature_names):
        """SmoothGrad raises error for invalid n_samples."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")

        with pytest.raises(ValueError, match="n_samples"):
            SmoothGradExplainer(model=adapter, feature_names=feature_names, n_samples=0)

        with pytest.raises(TypeError, match="n_samples"):
            SmoothGradExplainer(model=adapter, feature_names=feature_names, n_samples=2.5)

    def test_smoothgrad_invalid_noise_scale(self, simple_classifier, feature_names):
        """SmoothGrad raises error for invalid noise_scale."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")

        with pytest.raises(ValueError, match="noise_scale"):
            SmoothGradExplainer(model=adapter, feature_names=feature_names, noise_scale=-0.1)

        with pytest.raises(ValueError, match="noise_scale"):
            SmoothGradExplainer(model=adapter, feature_names=feature_names, noise_scale=np.nan)

    @pytest.mark.parametrize("random_state", [True, 1.5, "7"])
    def test_smoothgrad_invalid_random_state_type(
        self, simple_classifier, feature_names, random_state
    ):
        """SmoothGrad rejects ambiguous non-integer random-state values."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")

        with pytest.raises(TypeError, match="random_state"):
            SmoothGradExplainer(
                model=adapter, feature_names=feature_names, random_state=random_state
            )

    def test_smoothgrad_invalid_negative_random_state(self, simple_classifier, feature_names):
        """SmoothGrad rejects negative seeds at construction time."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")

        with pytest.raises(ValueError, match="random_state"):
            SmoothGradExplainer(model=adapter, feature_names=feature_names, random_state=-1)


# =============================================================================
# Classification Tests
# =============================================================================


class TestSmoothGradClassification:
    """Tests for SmoothGrad on classification models."""

    def test_smoothgrad_explain_classification(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """SmoothGrad produces valid explanations for classification."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.core.explanation import Explanation
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20,  # Fewer samples for faster tests
        )

        explanation = explainer.explain(sample_data[0])

        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "SmoothGrad"
        assert "feature_attributions" in explanation.explanation_data
        assert len(explanation.explanation_data["feature_attributions"]) == 4

    def test_smoothgrad_target_class(self, feature_names, class_names):
        """Each selected linear output returns its exact gradient vector."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        model = nn.Linear(4, 3, bias=False)
        weights = np.array(
            [
                [1.0, -2.0, 0.5, 3.0],
                [-0.25, 4.0, 2.0, -1.0],
                [0.75, 0.5, -3.0, 2.0],
            ],
            dtype=np.float32,
        )
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(weights))
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=7,
            noise_scale=0.8,
            random_state=19,
        )
        instance = np.array([0.3, -0.8, 1.2, 0.5], dtype=np.float32)

        for target_class, expected in enumerate(weights[:2]):
            explanation = explainer.explain(instance, target_class=target_class)
            np.testing.assert_allclose(
                explanation.explanation_data["attributions_raw"], expected, atol=1e-7
            )
            assert explanation.target_class == class_names[target_class]

    def test_smoothgrad_auto_target_class(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """SmoothGrad uses predicted class when target_class is None."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
        )

        # Get predicted class
        predictions = adapter.predict(sample_data[0].reshape(1, -1))
        predicted_class = int(np.argmax(predictions))

        explanation = explainer.explain(sample_data[0])

        # Should explain the predicted class
        assert explanation.target_class == class_names[predicted_class]

    def test_smoothgrad_attribution_shape(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """SmoothGrad produces attributions with correct shape."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
        )

        explanation = explainer.explain(sample_data[0])

        attributions_raw = explanation.explanation_data["attributions_raw"]
        assert len(attributions_raw) == len(feature_names)

    def test_smoothgrad_includes_statistics(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """SmoothGrad includes standard deviation in output."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
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
        from explainiverse.core.explanation import Explanation
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_regressor, task="regression")

        explainer = SmoothGradExplainer(model=adapter, feature_names=feature_names, n_samples=20)

        explanation = explainer.explain(sample_data[0])

        assert isinstance(explanation, Explanation)
        assert "feature_attributions" in explanation.explanation_data

    def test_smoothgrad_regression_no_class_names(
        self, simple_regressor, sample_data, feature_names
    ):
        """SmoothGrad handles regression without class_names."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_regressor, task="regression")

        explainer = SmoothGradExplainer(model=adapter, feature_names=feature_names, n_samples=20)

        explanation = explainer.explain(sample_data[0])

        # Target class should be "output" for regression
        assert explanation.target_class == "output"

    def test_smoothgrad_regression_attributions_finite(
        self, simple_regressor, sample_data, feature_names
    ):
        """SmoothGrad attributions are finite for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_regressor, task="regression")

        explainer = SmoothGradExplainer(model=adapter, feature_names=feature_names, n_samples=20)

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
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
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
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
        )

        explanation = explainer.explain(sample_data[0], method="vargrad")

        assert explanation.explainer_name == "VarGrad"
        assert "feature_attributions" in explanation.explanation_data

        # VarGrad (variance) should be non-negative
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(a >= 0 for a in attributions)

    def test_linear_gradient_variant_identities(self, feature_names, class_names):
        """Linear gradients give exact mean, mean-square, and variance oracles."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        model = nn.Linear(4, 3, bias=False)
        weights = np.array(
            [
                [1.0, -2.0, 0.5, 3.0],
                [-0.25, 4.0, 2.0, -1.0],
                [0.75, 0.5, -3.0, 2.0],
            ],
            dtype=np.float32,
        )
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(weights))
        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=7,
            noise_scale=0.8,
            random_state=42,
        )
        instance = np.array([0.3, -0.8, 1.2, 0.5], dtype=np.float32)

        standard = explainer.explain(instance, target_class=1, method="smoothgrad")
        squared = explainer.explain(instance, target_class=1, method="smoothgrad_squared")
        variance = explainer.explain(instance, target_class=1, method="vargrad")

        np.testing.assert_allclose(standard.explanation_data["attributions_raw"], weights[1])
        np.testing.assert_allclose(squared.explanation_data["attributions_raw"], weights[1] ** 2)
        np.testing.assert_allclose(variance.explanation_data["attributions_raw"], 0.0, atol=1e-12)

    def test_invalid_method(self, simple_classifier, sample_data, feature_names, class_names):
        """SmoothGrad raises error for invalid method."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
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
            noise_type="gaussian",
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
            noise_type="uniform",
        )

        explanation = explainer.explain(sample_data[0])

        assert explanation.explanation_data["noise_type"] == "uniform"

    @pytest.mark.parametrize("noise_type", ["gaussian", "uniform"])
    def test_noise_draws_match_local_generator_contract(
        self, simple_classifier, feature_names, noise_type
    ):
        """Configured draws exactly match a fresh NumPy Generator with the seed."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            noise_scale=0.25,
            noise_type=noise_type,
            random_state=31,
        )
        reference_rng = np.random.default_rng(31)
        if noise_type == "gaussian":
            expected = reference_rng.normal(0, 0.25, (2, 4)).astype(np.float32)
        else:
            expected = reference_rng.uniform(-0.25, 0.25, (2, 4)).astype(np.float32)

        np.testing.assert_array_equal(explainer._generate_noise((2, 4)), expected)

    def test_noise_scale_in_output(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """Noise scale is included in explanation output."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=20,
            noise_scale=0.25,
        )

        explanation = explainer.explain(sample_data[0])

        assert explanation.explanation_data["noise_scale"] == 0.25
        assert explanation.explanation_data["n_samples"] == 20

    def test_fixed_random_state_repeats_public_call(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """A fixed seed restarts the same local perturbation sequence per call."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=11,
            noise_scale=0.4,
            random_state=23,
        )

        first = explainer.explain(sample_data[0], target_class=1)
        second = explainer.explain(sample_data[0], target_class=1)

        np.testing.assert_array_equal(
            first.explanation_data["attributions_raw"],
            second.explanation_data["attributions_raw"],
        )
        np.testing.assert_array_equal(
            first.explanation_data["attributions_std"],
            second.explanation_data["attributions_std"],
        )
        assert first.explanation_data["random_state"] == 23

    @pytest.mark.parametrize("random_state", [None, 23])
    def test_local_generator_does_not_advance_global_numpy_state(
        self, simple_classifier, sample_data, feature_names, class_names, random_state
    ):
        """Seeded and entropy-backed local generators isolate legacy global RNG state."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=5,
            noise_scale=0.4,
            random_state=random_state,
        )

        np.random.seed(913)
        expected = np.random.random(6)
        np.random.seed(913)
        explainer.explain(sample_data[0], target_class=1)
        actual = np.random.random(6)

        np.testing.assert_array_equal(actual, expected)


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
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
        )

        explanations = explainer.explain_batch(sample_data[:5])

        assert len(explanations) == 5
        for exp in explanations:
            assert "feature_attributions" in exp.explanation_data

    def test_batch_with_target_class(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """Batch explain respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
        )

        explanations = explainer.explain_batch(sample_data[:3], target_class=1)

        for exp in explanations:
            assert exp.target_class == "class_b"

    def test_batch_single_instance(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """Batch explain handles single instance (1D input)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
        )

        explanations = explainer.explain_batch(sample_data[0])  # 1D input

        assert len(explanations) == 1

    def test_batch_with_method(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain supports different methods."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
        )

        explanations = explainer.explain_batch(sample_data[:3], method="smoothgrad_squared")

        for exp in explanations:
            assert exp.explainer_name == "SmoothGrad_Squared"


# =============================================================================
# Exact aggregation-contract tests
# =============================================================================


class TestSmoothGradAggregationIdentities:
    """Tests for exact aggregation identities, not perceptual smoothness."""

    def test_zero_noise_equals_raw_gradient(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
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
            noise_scale=0.0,
        )

        explanation = explainer.explain(sample_data[0], target_class=0)

        # Get raw gradient for comparison
        _, raw_gradients = adapter.predict_with_gradients(
            sample_data[0].reshape(1, -1), target_class=0
        )

        # Should be identical
        np.testing.assert_allclose(
            explanation.explanation_data["attributions_raw"], raw_gradients.flatten(), rtol=1e-5
        )

    def test_baseline_comparison_reuses_one_noisy_gradient_sample_set(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """VarGrad equals E[g^2] - E[g]^2 under common random numbers."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        explainer = SmoothGradExplainer(
            adapter,
            feature_names,
            class_names=class_names,
            n_samples=17,
            noise_scale=0.4,
            random_state=19,
        )

        comparison = explainer.compute_with_baseline_comparison(sample_data[0], target_class=1)
        mean = np.asarray(comparison["smoothgrad"])
        mean_square = np.asarray(comparison["smoothgrad_squared"])
        variance = np.asarray(comparison["vargrad"])

        np.testing.assert_allclose(variance, mean_square - mean**2, rtol=1e-6, atol=1e-8)
        assert comparison["common_random_numbers"] is True

    def test_baseline_comparison_reports_undefined_single_feature_correlation(self):
        """An undefined Pearson correlation is represented explicitly, not as NaN."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        model = nn.Linear(1, 2, bias=False)
        adapter = PyTorchAdapter(model, task="classification")
        explainer = SmoothGradExplainer(
            adapter,
            ["x"],
            n_samples=3,
            noise_scale=0.2,
            random_state=5,
        )

        comparison = explainer.compute_with_baseline_comparison(
            np.array([0.5], dtype=np.float32), target_class=0
        )

        assert comparison["correlation"] is None
        assert comparison["correlation_defined"] is False
        assert comparison["correlation_reason"] == "requires_at_least_two_features"


# =============================================================================
# Absolute Value Option Tests
# =============================================================================


class TestSmoothGradAbsoluteValue:
    """Tests for absolute value options."""

    def test_absolute_value_option(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """SmoothGrad supports absolute value option."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
        )

        explanation = explainer.explain(sample_data[0], absolute_value=True)

        # All attributions should be non-negative
        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(a >= 0 for a in attributions)

    def test_absolute_value_changes_result(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """Absolute value option changes the result."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            n_samples=30,
            random_state=42,
        )

        explanation_normal = explainer.explain(sample_data[0], absolute_value=False)
        explanation_abs = explainer.explain(sample_data[0], absolute_value=True)

        attr_normal = explanation_normal.explanation_data["attributions_raw"]
        attr_abs = explanation_abs.explanation_data["attributions_raw"]

        np.testing.assert_allclose(attr_abs, np.abs(attr_normal), atol=0.0, rtol=0.0)


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
        assert "image" not in meta.data_types
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
            "smoothgrad", model=adapter, feature_names=feature_names, class_names=class_names
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
        model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 3))

        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter, feature_names=["single_feature"], class_names=class_names, n_samples=10
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
        model = nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, 3))

        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)

        explainer = SmoothGradExplainer(
            model=adapter,
            feature_names=[f"f_{i}" for i in range(n_features)],
            class_names=class_names,
            n_samples=10,
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
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
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
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=20
        )

        zero_instance = np.zeros(4, dtype=np.float32)
        explanation = explainer.explain(zero_instance)

        attributions = list(explanation.explanation_data["feature_attributions"].values())
        assert all(np.isfinite(a) for a in attributions)

    def test_rejects_spatial_instance_instead_of_flattening(
        self, simple_classifier, feature_names, class_names
    ):
        """Spatial structure is rejected because the verified scope is flat/tabular."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_samples=2
        )

        with pytest.raises(ValueError, match="Image/spatial tensors are not supported"):
            explainer.explain(np.ones((2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="Image/spatial tensors are not supported"):
            explainer.explain_batch(np.ones((2, 1, 2, 2), dtype=np.float32))

    def test_rejects_incomplete_feature_identity(
        self, simple_classifier, feature_names, class_names
    ):
        """Every attribution must have one declared feature identity."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import SmoothGradExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        explainer = SmoothGradExplainer(
            model=adapter, feature_names=feature_names[:-1], class_names=class_names, n_samples=2
        )

        with pytest.raises(ValueError, match="feature count must match"):
            explainer.explain(np.ones(4, dtype=np.float32))

    def test_rejects_wrong_gradient_feature_count(self, feature_names):
        """The adapter cannot silently return a partial gradient vector."""
        from explainiverse.explainers.gradient import SmoothGradExplainer

        class WrongGradientAdapter:
            task = "regression"

            @staticmethod
            def predict_with_gradients(data, target_class=None):
                return np.zeros((1, 1)), np.zeros((1, 3))

        explainer = SmoothGradExplainer(
            model=WrongGradientAdapter(), feature_names=feature_names, n_samples=2
        )

        with pytest.raises(ValueError, match="wrong feature count"):
            explainer.explain(np.ones(4, dtype=np.float32))

    def test_rejects_gradient_with_transposed_batch_and_feature_axes(self):
        """Equal element counts cannot conceal a violated gradient axis contract."""
        from explainiverse.explainers.gradient import SmoothGradExplainer

        class TransposedGradientAdapter:
            task = "regression"

            @staticmethod
            def predict_with_gradients(data, target_class=None):
                return np.zeros((1, 1)), np.arange(4, dtype=float).reshape(4, 1)

        explainer = SmoothGradExplainer(
            model=TransposedGradientAdapter(),
            feature_names=["a", "b", "c", "d"],
            n_samples=2,
        )

        with pytest.raises(ValueError, match=r"expected \(1, 4\), got \(4, 1\)"):
            explainer.explain(np.ones(4, dtype=np.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
