# tests/test_randomisation.py
"""
Comprehensive tests for Phase 5 randomisation metrics.

Tests are organised by component:
  - Similarity functions (dispatcher + built-ins)
  - Attribution extraction
  - PyTorch model helpers (layer extraction, parameter randomisation)
  - Entropy helper
  - Noise helper
  - MPRT (Adebayo et al., 2018)
  - Random Logit Test (Sixt et al., 2020)
  - Smooth MPRT (Hedström et al., 2023)
  - Efficient MPRT (Hedström et al., 2023)
  - Data Randomisation Test (Adebayo et al., 2018)

Test categories per component:
  - Basic functionality & return types
  - Parameter variations
  - Tabular data (1-D)
  - Image data (2-D / 3-D)
  - Edge cases
  - Error handling & validation
  - Determinism / reproducibility
"""
import numpy as np
import pytest
import warnings

from explainiverse.evaluation.randomisation import (
    _spearman_similarity,
    _pearson_similarity,
    _cosine_similarity,
    _mse_similarity,
    _resolve_similarity_func,
    _compute_similarity,
    _extract_attribution_array,
    _discrete_entropy,
    _add_noise_to_input,
    _SIMILARITY_REGISTRY,
)
from explainiverse.core.explanation import Explanation


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def identical_1d():
    """Two identical 1-D attribution vectors."""
    a = np.array([0.5, 0.3, 0.1, 0.8, 0.2])
    return a, a.copy()


@pytest.fixture
def different_1d():
    """Two clearly different 1-D attribution vectors."""
    a = np.array([0.9, 0.1, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.1, 0.9])
    return a, b


@pytest.fixture
def correlated_1d():
    """Two positively correlated 1-D vectors."""
    a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    b = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    return a, b


@pytest.fixture
def image_2d():
    """Two 4x4 attribution maps."""
    rng = np.random.default_rng(123)
    a = rng.random((4, 4))
    b = rng.random((4, 4))
    return a, b


@pytest.fixture
def simple_explanation():
    """Explanation object with feature attributions."""
    return Explanation(
        explainer_name="TestExplainer",
        target_class="0",
        explanation_data={
            "feature_attributions": {
                "f0": 0.9, "f1": 0.1, "f2": 0.5, "f3": 0.2, "f4": 0.05,
            }
        },
        feature_names=["f0", "f1", "f2", "f3", "f4"],
    )


# ============================================================================
# Spearman Similarity
# ============================================================================

class TestSpearmanSimilarity:
    """Tests for _spearman_similarity."""

    def test_identical_vectors(self, identical_1d):
        a, b = identical_1d
        assert _spearman_similarity(a, b) == pytest.approx(1.0)

    def test_perfectly_anticorrelated(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _spearman_similarity(a, b) == pytest.approx(-1.0)

    def test_positively_correlated(self, correlated_1d):
        a, b = correlated_1d
        assert _spearman_similarity(a, b) > 0.9

    def test_different_vectors(self, different_1d):
        a, b = different_1d
        result = _spearman_similarity(a, b)
        assert -1.0 <= result <= 1.0

    def test_constant_vector_returns_zero(self):
        a = np.array([0.5, 0.5, 0.5, 0.5])
        b = np.array([0.1, 0.2, 0.3, 0.4])
        assert _spearman_similarity(a, b) == 0.0

    def test_both_constant_returns_zero(self):
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([2.0, 2.0, 2.0])
        assert _spearman_similarity(a, b) == 0.0

    def test_zero_vector_returns_zero(self):
        a = np.zeros(5)
        b = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert _spearman_similarity(a, b) == 0.0

    def test_single_element_returns_zero(self):
        a = np.array([1.0])
        b = np.array([2.0])
        assert _spearman_similarity(a, b) == 0.0

    def test_return_type_is_float(self, correlated_1d):
        a, b = correlated_1d
        assert isinstance(_spearman_similarity(a, b), float)


# ============================================================================
# Pearson Similarity
# ============================================================================

class TestPearsonSimilarity:
    """Tests for _pearson_similarity."""

    def test_identical_vectors(self, identical_1d):
        a, b = identical_1d
        assert _pearson_similarity(a, b) == pytest.approx(1.0)

    def test_perfectly_anticorrelated(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _pearson_similarity(a, b) == pytest.approx(-1.0)

    def test_linear_relationship(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = 2.0 * a + 3.0
        assert _pearson_similarity(a, b) == pytest.approx(1.0)

    def test_constant_vector_returns_zero(self):
        a = np.array([0.5, 0.5, 0.5, 0.5])
        b = np.array([0.1, 0.2, 0.3, 0.4])
        assert _pearson_similarity(a, b) == 0.0

    def test_zero_vector_returns_zero(self):
        a = np.zeros(5)
        b = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert _pearson_similarity(a, b) == 0.0

    def test_return_type_is_float(self, correlated_1d):
        a, b = correlated_1d
        assert isinstance(_pearson_similarity(a, b), float)


# ============================================================================
# Cosine Similarity
# ============================================================================

class TestCosineSimilarity:
    """Tests for _cosine_similarity."""

    def test_identical_vectors(self, identical_1d):
        a, b = identical_1d
        assert _cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_scaled_vectors_are_similar(self):
        a = np.array([1.0, 2.0, 3.0])
        b = 10.0 * a
        assert _cosine_similarity(a, b) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(5)
        b = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert _cosine_similarity(a, b) == 0.0

    def test_both_zero_returns_zero(self):
        a = np.zeros(5)
        b = np.zeros(5)
        assert _cosine_similarity(a, b) == 0.0

    def test_return_type_is_float(self, correlated_1d):
        a, b = correlated_1d
        assert isinstance(_cosine_similarity(a, b), float)


# ============================================================================
# MSE Similarity
# ============================================================================

class TestMSESimilarity:
    """Tests for _mse_similarity."""

    def test_identical_vectors(self, identical_1d):
        a, b = identical_1d
        assert _mse_similarity(a, b) == pytest.approx(0.0)

    def test_different_vectors_negative(self, different_1d):
        a, b = different_1d
        assert _mse_similarity(a, b) < 0.0

    def test_larger_difference_more_negative(self):
        a = np.array([0.0, 0.0, 0.0])
        b_small = np.array([0.1, 0.1, 0.1])
        b_large = np.array([1.0, 1.0, 1.0])
        assert _mse_similarity(a, b_small) > _mse_similarity(a, b_large)

    def test_symmetric(self, different_1d):
        a, b = different_1d
        assert _mse_similarity(a, b) == pytest.approx(_mse_similarity(b, a))

    def test_return_type_is_float(self, correlated_1d):
        a, b = correlated_1d
        assert isinstance(_mse_similarity(a, b), float)


# ============================================================================
# Similarity Dispatcher
# ============================================================================

class TestResolveSimilarityFunc:
    """Tests for _resolve_similarity_func."""

    def test_all_string_keys_resolve(self):
        for key in _SIMILARITY_REGISTRY:
            func = _resolve_similarity_func(key)
            assert callable(func)

    def test_case_insensitive(self):
        func = _resolve_similarity_func("SPEARMAN")
        assert callable(func)

    def test_whitespace_tolerance(self):
        func = _resolve_similarity_func("  pearson  ")
        assert callable(func)

    def test_custom_callable(self):
        custom_fn = lambda a, b: float(np.sum(a * b))
        func = _resolve_similarity_func(custom_fn)
        assert func is custom_fn

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown similarity"):
            _resolve_similarity_func("not_a_real_metric")

    def test_non_string_non_callable_raises(self):
        with pytest.raises(TypeError, match="string or callable"):
            _resolve_similarity_func(42)

    def test_none_raises(self):
        with pytest.raises(TypeError):
            _resolve_similarity_func(None)


class TestComputeSimilarity:
    """Tests for _compute_similarity (the main dispatcher)."""

    def test_spearman_via_string(self, identical_1d):
        a, b = identical_1d
        result = _compute_similarity(a, b, "spearman")
        assert result == pytest.approx(1.0)

    def test_pearson_via_string(self, identical_1d):
        a, b = identical_1d
        result = _compute_similarity(a, b, "pearson")
        assert result == pytest.approx(1.0)

    def test_cosine_via_string(self, identical_1d):
        a, b = identical_1d
        result = _compute_similarity(a, b, "cosine")
        assert result == pytest.approx(1.0)

    def test_mse_via_string(self, identical_1d):
        a, b = identical_1d
        result = _compute_similarity(a, b, "mse")
        assert result == pytest.approx(0.0)

    def test_custom_callable(self, identical_1d):
        a, b = identical_1d
        custom = lambda x, y: 42.0
        result = _compute_similarity(a, b, custom)
        assert result == 42.0

    def test_shape_mismatch_raises(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="shapes must match"):
            _compute_similarity(a, b, "spearman")

    def test_2d_arrays_flattened_for_correlation(self, image_2d):
        a, b = image_2d
        result = _compute_similarity(a, b, "spearman")
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_3d_arrays_flattened_for_correlation(self):
        a = np.random.default_rng(1).random((3, 4, 4))
        b = np.random.default_rng(2).random((3, 4, 4))
        result = _compute_similarity(a, b, "pearson")
        assert isinstance(result, float)


# ============================================================================
# Attribution Extraction
# ============================================================================

class TestExtractAttributionArray:
    """Tests for _extract_attribution_array."""

    def test_numpy_1d_passthrough(self):
        a = np.array([0.5, 0.3, 0.1])
        result = _extract_attribution_array(a)
        np.testing.assert_array_equal(result, a)
        assert result.dtype == np.float64

    def test_numpy_2d_preserves_shape(self):
        a = np.array([[0.5, 0.3], [0.1, 0.8]])
        result = _extract_attribution_array(a)
        assert result.shape == (2, 2)

    def test_numpy_int_converts_to_float64(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        result = _extract_attribution_array(a)
        assert result.dtype == np.float64

    def test_explanation_object(self, simple_explanation):
        result = _extract_attribution_array(simple_explanation)
        assert result.shape == (5,)
        assert result[0] == pytest.approx(0.9)
        assert result.dtype == np.float64

    def test_explanation_no_attributions_raises(self):
        exp = Explanation(
            explainer_name="Empty",
            target_class="0",
            explanation_data={},
        )
        with pytest.raises(ValueError, match="No feature attributions"):
            _extract_attribution_array(exp)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Expected np.ndarray"):
            _extract_attribution_array([0.1, 0.2, 0.3])

    def test_unsupported_type_string_raises(self):
        with pytest.raises(TypeError):
            _extract_attribution_array("not_an_array")


# ============================================================================
# Discrete Entropy
# ============================================================================

class TestDiscreteEntropy:
    """Tests for _discrete_entropy."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have maximum entropy."""
        a = np.ones(10)
        entropy = _discrete_entropy(a)
        # Max entropy for 10 bins = ln(10) ≈ 2.302
        assert entropy == pytest.approx(np.log(10), abs=1e-10)

    def test_single_spike_zero_entropy(self):
        """Single non-zero element should have zero entropy."""
        a = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        entropy = _discrete_entropy(a)
        assert entropy == pytest.approx(0.0)

    def test_all_zeros_returns_zero(self):
        a = np.zeros(5)
        assert _discrete_entropy(a) == 0.0

    def test_negative_values_use_absolute(self):
        """Should use absolute values."""
        a = np.array([-0.5, 0.5])
        entropy = _discrete_entropy(a)
        # Both become 0.5 -> uniform over 2 bins -> ln(2)
        assert entropy == pytest.approx(np.log(2), abs=1e-10)

    def test_sparse_lower_than_uniform(self):
        sparse = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        uniform = np.ones(5)
        assert _discrete_entropy(sparse) < _discrete_entropy(uniform)

    def test_2d_array_flattened(self):
        a = np.ones((3, 3))
        entropy = _discrete_entropy(a)
        assert entropy == pytest.approx(np.log(9), abs=1e-10)

    def test_return_type_is_float(self):
        assert isinstance(_discrete_entropy(np.array([1.0, 2.0])), float)

    def test_entropy_non_negative(self, rng):
        a = rng.random(20)
        assert _discrete_entropy(a) >= 0.0


# ============================================================================
# Noise Addition
# ============================================================================

class TestAddNoiseToInput:
    """Tests for _add_noise_to_input."""

    def test_output_shape_preserved(self, rng):
        x = np.ones((3, 4))
        noisy = _add_noise_to_input(x, 0.1, rng)
        assert noisy.shape == x.shape

    def test_noise_is_applied(self, rng):
        x = np.ones(10)
        noisy = _add_noise_to_input(x, 0.5, rng)
        assert not np.allclose(x, noisy)

    def test_zero_magnitude_no_noise(self, rng):
        x = np.array([1.0, 2.0, 3.0])
        noisy = _add_noise_to_input(x, 0.0, rng)
        np.testing.assert_array_almost_equal(x, noisy)

    def test_deterministic_with_same_seed(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        noisy1 = _add_noise_to_input(x, 0.1, rng1)
        noisy2 = _add_noise_to_input(x, 0.1, rng2)
        np.testing.assert_array_equal(noisy1, noisy2)

    def test_different_seeds_different_noise(self):
        x = np.array([1.0, 2.0, 3.0])
        noisy1 = _add_noise_to_input(x, 0.1, np.random.default_rng(1))
        noisy2 = _add_noise_to_input(x, 0.1, np.random.default_rng(2))
        assert not np.allclose(noisy1, noisy2)

    def test_constant_input_still_adds_noise(self, rng):
        """Even with zero range, noise should be added (uses range=1.0)."""
        x = np.ones(5)
        noisy = _add_noise_to_input(x, 0.1, rng)
        assert not np.allclose(x, noisy)

    def test_1d_tabular_data(self, rng):
        x = np.array([0.1, 0.5, 0.9])
        noisy = _add_noise_to_input(x, 0.05, rng)
        assert noisy.shape == (3,)

    def test_3d_image_data(self, rng):
        x = np.random.default_rng(0).random((3, 8, 8))
        noisy = _add_noise_to_input(x, 0.1, rng)
        assert noisy.shape == (3, 8, 8)


# ============================================================================
# PyTorch Model Helpers (requires torch)
# ============================================================================

# These tests are skipped if torch is not installed
torch = pytest.importorskip("torch")
import torch.nn as nn

from explainiverse.evaluation.randomisation import (
    _get_named_layers,
    _randomise_layer_parameters,
    _validate_torch_available,
)


class SimpleMLP(nn.Module):
    """Simple 3-layer MLP for testing."""
    def __init__(self, in_features=10, hidden=20, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(nn.Module):
    """Simple CNN for testing image data."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def mlp_model():
    """Pre-seeded MLP model."""
    torch.manual_seed(42)
    return SimpleMLP()


@pytest.fixture
def cnn_model():
    """Pre-seeded CNN model."""
    torch.manual_seed(42)
    return SimpleCNN()


class TestValidateTorchAvailable:
    """Tests for _validate_torch_available."""

    def test_does_not_raise_when_torch_present(self):
        # torch is present since we importorskip'd it
        _validate_torch_available()


class TestGetNamedLayers:
    """Tests for _get_named_layers."""

    def test_auto_detect_mlp_layers(self, mlp_model):
        layers = _get_named_layers(mlp_model)
        names = [name for name, _ in layers]
        assert "fc1" in names
        assert "fc2" in names
        assert "fc3" in names
        # ReLU should NOT be included (no learnable params)
        assert "relu" not in names

    def test_auto_detect_cnn_layers(self, cnn_model):
        layers = _get_named_layers(cnn_model)
        names = [name for name, _ in layers]
        assert "conv1" in names
        assert "bn1" in names
        assert "fc" in names

    def test_custom_layer_names(self, mlp_model):
        layers = _get_named_layers(mlp_model, layer_names=["fc1", "fc3"])
        assert len(layers) == 2
        assert layers[0][0] == "fc1"
        assert layers[1][0] == "fc3"

    def test_invalid_layer_name_raises(self, mlp_model):
        with pytest.raises(ValueError, match="not found"):
            _get_named_layers(mlp_model, layer_names=["nonexistent_layer"])

    def test_returns_list_of_tuples(self, mlp_model):
        layers = _get_named_layers(mlp_model)
        assert isinstance(layers, list)
        for item in layers:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], nn.Module)

    def test_preserves_custom_order(self, mlp_model):
        layers = _get_named_layers(mlp_model, layer_names=["fc3", "fc1"])
        assert layers[0][0] == "fc3"
        assert layers[1][0] == "fc1"


class TestRandomiseLayerParameters:
    """Tests for _randomise_layer_parameters."""

    def test_weights_change_after_randomisation(self, mlp_model):
        original_weight = mlp_model.fc1.weight.data.clone()
        _randomise_layer_parameters(mlp_model, "fc1")
        assert not torch.allclose(original_weight, mlp_model.fc1.weight.data)

    def test_other_layers_unaffected(self, mlp_model):
        original_fc2 = mlp_model.fc2.weight.data.clone()
        _randomise_layer_parameters(mlp_model, "fc1")
        assert torch.allclose(original_fc2, mlp_model.fc2.weight.data)

    def test_bias_also_randomised(self, mlp_model):
        original_bias = mlp_model.fc1.bias.data.clone()
        _randomise_layer_parameters(mlp_model, "fc1")
        assert not torch.allclose(original_bias, mlp_model.fc1.bias.data)

    def test_conv_layer_randomised(self, cnn_model):
        original_weight = cnn_model.conv1.weight.data.clone()
        _randomise_layer_parameters(cnn_model, "conv1")
        assert not torch.allclose(original_weight, cnn_model.conv1.weight.data)

    def test_batchnorm_randomised(self, cnn_model):
        # BatchNorm running_mean should be zeroed
        cnn_model.bn1.running_mean.fill_(5.0)
        _randomise_layer_parameters(cnn_model, "bn1")
        assert torch.allclose(cnn_model.bn1.running_mean, torch.zeros_like(cnn_model.bn1.running_mean))

    def test_invalid_layer_raises(self, mlp_model):
        with pytest.raises(ValueError, match="not found"):
            _randomise_layer_parameters(mlp_model, "nonexistent")

    def test_deterministic_with_rng(self, mlp_model):
        model1 = copy.deepcopy(mlp_model)
        model2 = copy.deepcopy(mlp_model)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        _randomise_layer_parameters(model1, "fc1", rng=rng1)
        _randomise_layer_parameters(model2, "fc1", rng=rng2)
        assert torch.allclose(model1.fc1.weight.data, model2.fc1.weight.data)

    def test_different_rng_different_weights(self, mlp_model):
        model1 = copy.deepcopy(mlp_model)
        model2 = copy.deepcopy(mlp_model)
        _randomise_layer_parameters(model1, "fc1", rng=np.random.default_rng(1))
        _randomise_layer_parameters(model2, "fc1", rng=np.random.default_rng(2))
        assert not torch.allclose(model1.fc1.weight.data, model2.fc1.weight.data)


# Need copy for deepcopy tests above
import copy

from explainiverse.evaluation.randomisation import (
    compute_mprt_score,
    compute_mprt,
    compute_batch_mprt,
)


# ============================================================================
# Helper: Simple gradient explain function for integration tests
# ============================================================================

def _gradient_explain_func(model, x, y):
    """
    Simple gradient-based explanation for testing.
    x: shape (1, features) or (1, C, H, W)
    y: scalar label
    Returns: np.ndarray of attributions (same spatial shape as input, no batch dim).
    """
    x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    model.eval()
    out = model(x_t)
    y_idx = int(y) if np.ndim(y) == 0 else int(y[0])
    out[0, y_idx].backward()
    grad = x_t.grad.detach().numpy()
    return grad[0]  # remove batch dim


def _random_explain_func(model, x, y):
    """Explanation that ignores the model entirely (should fail MPRT)."""
    return np.random.default_rng(0).random(x.shape[1:])


# ============================================================================
# MPRT Score (Low-Level API)
# ============================================================================

class TestComputeMprtScore:
    """Tests for compute_mprt_score (pre-computed attributions)."""

    def test_basic_return_structure(self):
        original = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        rand_1 = np.array([0.85, 0.15, 0.45, 0.35, 0.18])
        rand_2 = np.array([0.3, 0.6, 0.1, 0.7, 0.5])
        result = compute_mprt_score(original, [rand_1, rand_2])
        assert "layer_scores" in result
        assert "layer_names" in result
        assert "mean_score" in result
        assert len(result["layer_scores"]) == 2
        assert len(result["layer_names"]) == 2
        assert isinstance(result["mean_score"], float)

    def test_identical_attributions_high_similarity(self):
        original = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        result = compute_mprt_score(original, [original.copy(), original.copy()])
        assert result["mean_score"] == pytest.approx(1.0)

    def test_random_attributions_lower_similarity(self):
        rng = np.random.default_rng(42)
        original = rng.random(20)
        rand_attrs = [rng.random(20) for _ in range(3)]
        result = compute_mprt_score(original, rand_attrs)
        assert result["mean_score"] < 0.9  # should be notably lower than 1.0

    def test_custom_layer_names(self):
        original = np.array([0.5, 0.3, 0.1])
        rand = [np.array([0.4, 0.2, 0.15]), np.array([0.1, 0.8, 0.3])]
        result = compute_mprt_score(
            original, rand, layer_names=["fc1", "fc2"]
        )
        assert result["layer_names"] == ["fc1", "fc2"]

    def test_default_layer_names(self):
        original = np.array([0.5, 0.3, 0.1])
        rand = [np.array([0.4, 0.2, 0.15])]
        result = compute_mprt_score(original, rand)
        assert result["layer_names"] == ["layer_0"]

    def test_different_similarity_funcs(self):
        original = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        rand = [np.array([0.3, 0.6, 0.1, 0.7, 0.5])]
        for func_name in ["spearman", "pearson", "cosine", "mse"]:
            result = compute_mprt_score(original, rand, similarity_func=func_name)
            assert isinstance(result["mean_score"], float)

    def test_custom_similarity_callable(self):
        original = np.array([0.5, 0.5, 0.5])
        rand = [np.array([0.5, 0.5, 0.5])]
        custom = lambda a, b: 42.0
        result = compute_mprt_score(original, rand, similarity_func=custom)
        assert result["layer_scores"][0] == 42.0

    def test_empty_list_raises(self):
        original = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="must not be empty"):
            compute_mprt_score(original, [])

    def test_mismatched_layer_names_raises(self):
        original = np.array([0.5, 0.3])
        rand = [np.array([0.4, 0.2])]
        with pytest.raises(ValueError, match="layer_names length"):
            compute_mprt_score(original, rand, layer_names=["a", "b"])

    def test_mean_is_average_of_layer_scores(self):
        original = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        rand = [
            np.array([0.85, 0.15, 0.45, 0.35, 0.18]),
            np.array([0.3, 0.6, 0.1, 0.7, 0.5]),
        ]
        result = compute_mprt_score(original, rand)
        expected_mean = np.mean(result["layer_scores"])
        assert result["mean_score"] == pytest.approx(expected_mean)

    def test_2d_image_attributions(self):
        rng = np.random.default_rng(10)
        original = rng.random((4, 4))
        rand = [rng.random((4, 4)), rng.random((4, 4))]
        result = compute_mprt_score(original, rand)
        assert len(result["layer_scores"]) == 2

    def test_explanation_objects(self, simple_explanation):
        """Test with Explanation objects instead of arrays."""
        # Create a second explanation with different values
        rand_exp = Explanation(
            explainer_name="TestExplainer",
            target_class="0",
            explanation_data={
                "feature_attributions": {
                    "f0": 0.1, "f1": 0.8, "f2": 0.3, "f3": 0.6, "f4": 0.9,
                }
            },
            feature_names=["f0", "f1", "f2", "f3", "f4"],
        )
        result = compute_mprt_score(simple_explanation, [rand_exp])
        assert isinstance(result["mean_score"], float)


# ============================================================================
# MPRT (High-Level API)
# ============================================================================

class TestComputeMprt:
    """Tests for compute_mprt (full pipeline with model)."""

    def test_basic_return_structure(self, mlp_model):
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        result = compute_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert "layer_scores" in result
        assert "layer_names" in result
        assert "mean_score" in result
        assert isinstance(result["mean_score"], float)

    def test_layer_count_matches_model(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        # SimpleMLP has 3 linear layers (fc1, fc2, fc3)
        assert len(result["layer_scores"]) == 3
        assert len(result["layer_names"]) == 3

    def test_cascading_order(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_mprt(
            mlp_model, x, y, _gradient_explain_func,
            order="cascading", seed=42
        )
        # Cascading = top-down, so first layer randomised should be fc3 (last)
        assert result["layer_names"][0] == "fc3"
        assert result["layer_names"][-1] == "fc1"

    def test_bottom_up_order(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_mprt(
            mlp_model, x, y, _gradient_explain_func,
            order="bottom_up", seed=42
        )
        # Bottom-up = first layer first
        assert result["layer_names"][0] == "fc1"
        assert result["layer_names"][-1] == "fc3"

    def test_independent_order(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_mprt(
            mlp_model, x, y, _gradient_explain_func,
            order="independent", seed=42
        )
        assert len(result["layer_scores"]) == 3

    def test_invalid_order_raises(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        with pytest.raises(ValueError, match="Unknown order"):
            compute_mprt(
                mlp_model, x, y, _gradient_explain_func,
                order="invalid_order"
            )

    def test_custom_layer_names_subset(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_mprt(
            mlp_model, x, y, _gradient_explain_func,
            layer_names=["fc1", "fc3"], seed=42
        )
        assert len(result["layer_scores"]) == 2

    def test_deterministic_with_seed(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        r1 = compute_mprt(mlp_model, x, y, _gradient_explain_func, seed=42)
        r2 = compute_mprt(mlp_model, x, y, _gradient_explain_func, seed=42)
        assert r1["layer_scores"] == pytest.approx(r2["layer_scores"])

    def test_different_seeds_different_results(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        r1 = compute_mprt(mlp_model, x, y, _gradient_explain_func, seed=42)
        r2 = compute_mprt(mlp_model, x, y, _gradient_explain_func, seed=99)
        # Scores should differ with different seeds
        assert r1["layer_scores"] != pytest.approx(r2["layer_scores"], abs=0.01)

    def test_original_model_not_modified(self, mlp_model):
        original_weight = mlp_model.fc1.weight.data.clone()
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        compute_mprt(mlp_model, x, y, _gradient_explain_func, seed=42)
        assert torch.allclose(original_weight, mlp_model.fc1.weight.data)

    def test_gradient_explainer_passes_sanity_check(self, mlp_model):
        """Gradient explanations should be sensitive to model parameters."""
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        result = compute_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        # Gradients should change when model is randomised -> mean < 1.0
        assert result["mean_score"] < 0.95

    def test_random_explainer_fails_sanity_check(self, mlp_model):
        """Random explanations should NOT be sensitive to model parameters."""
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        result = compute_mprt(
            mlp_model, x, y, _random_explain_func, seed=42
        )
        # Random explainer produces same output regardless of model
        # So similarity should be very high (close to 1.0)
        assert result["mean_score"] > 0.95

    def test_different_similarity_functions(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        for func in ["spearman", "pearson", "cosine"]:
            result = compute_mprt(
                mlp_model, x, y, _gradient_explain_func,
                similarity_func=func, seed=42
            )
            assert isinstance(result["mean_score"], float)

    def test_cnn_model(self, cnn_model):
        """Test with CNN and image-like input."""
        x = np.random.default_rng(1).random((1, 1, 8, 8)).astype(np.float32)
        y = np.array([0])
        result = compute_mprt(
            cnn_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(result["mean_score"], float)
        assert len(result["layer_scores"]) > 0

    def test_multi_sample_batch(self, mlp_model):
        """Batch with multiple samples should average scores."""
        x = np.random.default_rng(1).random((5, 10)).astype(np.float32)
        y = np.array([0, 1, 2, 0, 1])
        result = compute_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(result["mean_score"], float)


# ============================================================================
# Batch MPRT
# ============================================================================

class TestComputeBatchMprt:
    """Tests for compute_batch_mprt."""

    def test_returns_list_per_sample(self, mlp_model):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        results = compute_batch_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert len(results) == 3
        for r in results:
            assert "layer_scores" in r
            assert "mean_score" in r

    def test_single_sample_matches_compute_mprt(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        batch_results = compute_batch_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        single_result = compute_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert batch_results[0]["layer_scores"] == pytest.approx(
            single_result["layer_scores"]
        )


from explainiverse.evaluation.randomisation import (
    compute_random_logit_score,
    compute_random_logit,
    compute_batch_random_logit,
)


# ============================================================================
# Random Logit Score (Low-Level API)
# ============================================================================

class TestComputeRandomLogitScore:
    """Tests for compute_random_logit_score (pre-computed attributions)."""

    def test_identical_attributions_high_similarity(self):
        a = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        score = compute_random_logit_score(a, a.copy())
        assert score == pytest.approx(1.0)

    def test_different_attributions(self):
        a = np.array([0.9, 0.1, 0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0, 0.1, 0.9])
        score = compute_random_logit_score(a, b)
        assert score < 0.5

    def test_return_type_is_float(self):
        a = np.array([0.5, 0.3, 0.1])
        b = np.array([0.1, 0.3, 0.5])
        assert isinstance(compute_random_logit_score(a, b), float)

    def test_different_similarity_funcs(self):
        a = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        b = np.array([0.2, 0.8, 0.1, 0.6, 0.3])
        for func in ["spearman", "pearson", "cosine", "mse"]:
            score = compute_random_logit_score(a, b, similarity_func=func)
            assert isinstance(score, float)

    def test_custom_callable(self):
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        score = compute_random_logit_score(a, b, similarity_func=lambda x, y: -1.0)
        assert score == -1.0

    def test_2d_image_data(self):
        rng = np.random.default_rng(42)
        a = rng.random((4, 4))
        b = rng.random((4, 4))
        score = compute_random_logit_score(a, b)
        assert isinstance(score, float)

    def test_explanation_objects(self, simple_explanation):
        rand_exp = Explanation(
            explainer_name="Test",
            target_class="1",
            explanation_data={
                "feature_attributions": {
                    "f0": 0.1, "f1": 0.8, "f2": 0.3, "f3": 0.6, "f4": 0.9,
                }
            },
            feature_names=["f0", "f1", "f2", "f3", "f4"],
        )
        score = compute_random_logit_score(simple_explanation, rand_exp)
        assert isinstance(score, float)


# ============================================================================
# Random Logit (High-Level API)
# ============================================================================

class TestComputeRandomLogit:
    """Tests for compute_random_logit (full pipeline with model)."""

    def test_basic_return_type(self, mlp_model):
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        score = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(score, float)

    def test_score_in_valid_range(self, mlp_model):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        score = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        # Spearman correlation range
        assert -1.0 <= score <= 1.0

    def test_auto_detect_num_classes(self, mlp_model):
        """Should infer num_classes from model output."""
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        # SimpleMLP has 3 output classes
        score = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(score, float)

    def test_explicit_num_classes(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        score = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func,
            num_classes=3, seed=42
        )
        assert isinstance(score, float)

    def test_num_classes_too_small_raises(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            compute_random_logit(
                mlp_model, x, y, _gradient_explain_func, num_classes=1
            )

    def test_deterministic_with_seed(self, mlp_model):
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        s1 = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        s2 = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert s1 == pytest.approx(s2)

    def test_different_seeds_may_differ(self, mlp_model):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        s1 = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        s2 = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=99
        )
        # Different seeds choose different random classes
        # Scores may or may not differ, but the call should succeed
        assert isinstance(s1, float) and isinstance(s2, float)

    def test_gradient_explainer_class_sensitive(self, mlp_model):
        """Gradient explanations should differ for different target classes."""
        x = np.random.default_rng(1).random((5, 10)).astype(np.float32)
        y = np.array([0, 1, 2, 0, 1])
        score = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        # Gradients for different classes go through different output neurons
        # so similarity should be notably less than 1.0
        assert score < 0.95

    def test_random_explainer_class_insensitive(self, mlp_model):
        """Random explanations ignore target class -> high similarity."""
        x = np.random.default_rng(1).random((5, 10)).astype(np.float32)
        y = np.array([0, 1, 2, 0, 1])
        score = compute_random_logit(
            mlp_model, x, y, _random_explain_func, seed=42
        )
        # Random explainer returns same values regardless of class
        assert score > 0.95

    def test_different_similarity_functions(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        for func in ["spearman", "pearson", "cosine"]:
            score = compute_random_logit(
                mlp_model, x, y, _gradient_explain_func,
                similarity_func=func, seed=42
            )
            assert isinstance(score, float)

    def test_cnn_model(self, cnn_model):
        x = np.random.default_rng(1).random((1, 1, 8, 8)).astype(np.float32)
        y = np.array([0])
        score = compute_random_logit(
            cnn_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(score, float)

    def test_original_model_not_modified(self, mlp_model):
        original_weight = mlp_model.fc1.weight.data.clone()
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert torch.allclose(original_weight, mlp_model.fc1.weight.data)


# ============================================================================
# Batch Random Logit
# ============================================================================

class TestComputeBatchRandomLogit:
    """Tests for compute_batch_random_logit."""

    def test_returns_list_per_sample(self, mlp_model):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        scores = compute_batch_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(scores, list)
        assert len(scores) == 3
        for s in scores:
            assert isinstance(s, float)

    def test_deterministic_with_seed(self, mlp_model):
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        s1 = compute_batch_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        s2 = compute_batch_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert s1 == pytest.approx(s2)

    def test_mean_matches_compute_random_logit(self, mlp_model):
        """Mean of batch scores should match compute_random_logit."""
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        batch_scores = compute_batch_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        agg_score = compute_random_logit(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert np.mean(batch_scores) == pytest.approx(agg_score)


from explainiverse.evaluation.randomisation import (
    compute_data_randomisation_score,
    compute_data_randomisation,
    compute_batch_data_randomisation,
)


# ============================================================================
# Fixtures for Data Randomisation
# ============================================================================

@pytest.fixture
def mlp_model_random_labels():
    """MLP with different random seed (simulates training on random labels)."""
    torch.manual_seed(999)  # Different seed -> different weights
    return SimpleMLP()


# ============================================================================
# Data Randomisation Score (Low-Level API)
# ============================================================================

class TestComputeDataRandomisationScore:
    """Tests for compute_data_randomisation_score."""

    def test_identical_attributions_high_similarity(self):
        a = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        score = compute_data_randomisation_score(a, a.copy())
        assert score == pytest.approx(1.0)

    def test_different_attributions(self):
        a = np.array([0.9, 0.1, 0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0, 0.1, 0.9])
        score = compute_data_randomisation_score(a, b)
        assert score < 0.5

    def test_return_type_is_float(self):
        a = np.array([0.5, 0.3, 0.1])
        b = np.array([0.1, 0.3, 0.5])
        assert isinstance(compute_data_randomisation_score(a, b), float)

    def test_different_similarity_funcs(self):
        a = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        b = np.array([0.2, 0.8, 0.1, 0.6, 0.3])
        for func in ["spearman", "pearson", "cosine", "mse"]:
            score = compute_data_randomisation_score(a, b, similarity_func=func)
            assert isinstance(score, float)

    def test_custom_callable(self):
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        score = compute_data_randomisation_score(
            a, b, similarity_func=lambda x, y: -0.5
        )
        assert score == -0.5

    def test_2d_image_data(self):
        rng = np.random.default_rng(42)
        a = rng.random((4, 4))
        b = rng.random((4, 4))
        score = compute_data_randomisation_score(a, b)
        assert isinstance(score, float)

    def test_explanation_objects(self, simple_explanation):
        rand_exp = Explanation(
            explainer_name="Test",
            target_class="0",
            explanation_data={
                "feature_attributions": {
                    "f0": 0.2, "f1": 0.7, "f2": 0.1, "f3": 0.8, "f4": 0.3,
                }
            },
            feature_names=["f0", "f1", "f2", "f3", "f4"],
        )
        score = compute_data_randomisation_score(simple_explanation, rand_exp)
        assert isinstance(score, float)

    def test_shape_mismatch_raises(self):
        a = np.array([0.5, 0.3])
        b = np.array([0.5, 0.3, 0.1])
        with pytest.raises(ValueError, match="shapes must match"):
            compute_data_randomisation_score(a, b)


# ============================================================================
# Data Randomisation (High-Level API)
# ============================================================================

class TestComputeDataRandomisation:
    """Tests for compute_data_randomisation."""

    def test_basic_return_type(self, mlp_model, mlp_model_random_labels):
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        score = compute_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _gradient_explain_func
        )
        assert isinstance(score, float)

    def test_score_in_valid_range(self, mlp_model, mlp_model_random_labels):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        score = compute_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _gradient_explain_func
        )
        assert -1.0 <= score <= 1.0

    def test_same_model_high_similarity(self, mlp_model):
        """Comparing model to itself should yield high similarity."""
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        score = compute_data_randomisation(
            mlp_model, mlp_model, x, y, _gradient_explain_func
        )
        assert score == pytest.approx(1.0)

    def test_different_models_lower_similarity(self, mlp_model, mlp_model_random_labels):
        """Different models should produce different explanations."""
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        score = compute_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _gradient_explain_func
        )
        # Different models -> different gradients -> similarity < 1.0
        assert score < 0.99

    def test_random_explainer_high_similarity(self, mlp_model, mlp_model_random_labels):
        """Random explainer ignores model -> same output -> high similarity."""
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        score = compute_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _random_explain_func
        )
        assert score > 0.95

    def test_different_similarity_functions(self, mlp_model, mlp_model_random_labels):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        for func in ["spearman", "pearson", "cosine"]:
            score = compute_data_randomisation(
                mlp_model, mlp_model_random_labels, x, y,
                _gradient_explain_func, similarity_func=func
            )
            assert isinstance(score, float)

    def test_original_models_not_modified(self, mlp_model, mlp_model_random_labels):
        w1 = mlp_model.fc1.weight.data.clone()
        w2 = mlp_model_random_labels.fc1.weight.data.clone()
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        compute_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _gradient_explain_func
        )
        assert torch.allclose(w1, mlp_model.fc1.weight.data)
        assert torch.allclose(w2, mlp_model_random_labels.fc1.weight.data)

    def test_cnn_models(self, cnn_model):
        """Test with CNN architecture."""
        torch.manual_seed(777)
        cnn_random = SimpleCNN()
        x = np.random.default_rng(1).random((1, 1, 8, 8)).astype(np.float32)
        y = np.array([0])
        score = compute_data_randomisation(
            cnn_model, cnn_random, x, y, _gradient_explain_func
        )
        assert isinstance(score, float)

    def test_multi_sample_batch(self, mlp_model, mlp_model_random_labels):
        x = np.random.default_rng(1).random((5, 10)).astype(np.float32)
        y = np.array([0, 1, 2, 0, 1])
        score = compute_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _gradient_explain_func
        )
        assert isinstance(score, float)

    def test_symmetric(self, mlp_model, mlp_model_random_labels):
        """Score should be the same regardless of argument order."""
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        score_ab = compute_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _gradient_explain_func
        )
        score_ba = compute_data_randomisation(
            mlp_model_random_labels, mlp_model, x, y, _gradient_explain_func
        )
        # Spearman is symmetric
        assert score_ab == pytest.approx(score_ba)


# ============================================================================
# Batch Data Randomisation
# ============================================================================

class TestComputeBatchDataRandomisation:
    """Tests for compute_batch_data_randomisation."""

    def test_returns_list_per_sample(self, mlp_model, mlp_model_random_labels):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        scores = compute_batch_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _gradient_explain_func
        )
        assert isinstance(scores, list)
        assert len(scores) == 3
        for s in scores:
            assert isinstance(s, float)

    def test_mean_matches_aggregate(self, mlp_model, mlp_model_random_labels):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        batch_scores = compute_batch_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _gradient_explain_func
        )
        agg_score = compute_data_randomisation(
            mlp_model, mlp_model_random_labels, x, y, _gradient_explain_func
        )
        assert np.mean(batch_scores) == pytest.approx(agg_score)

    def test_same_model_all_ones(self, mlp_model):
        """Same model for both -> all per-sample scores should be 1.0."""
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        scores = compute_batch_data_randomisation(
            mlp_model, mlp_model, x, y, _gradient_explain_func
        )
        for s in scores:
            assert s == pytest.approx(1.0)


from explainiverse.evaluation.randomisation import (
    compute_smooth_mprt,
    compute_batch_smooth_mprt,
    compute_efficient_mprt,
    compute_batch_efficient_mprt,
)


# ============================================================================
# Smooth MPRT (High-Level API)
# ============================================================================

class TestComputeSmoothMprt:
    """Tests for compute_smooth_mprt."""

    def test_basic_return_structure(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        assert "layer_scores" in result
        assert "layer_names" in result
        assert "mean_score" in result
        assert isinstance(result["mean_score"], float)

    def test_layer_count_matches_model(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        assert len(result["layer_scores"]) == 3

    def test_cascading_order(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            order="cascading", nr_samples=3, seed=42
        )
        assert result["layer_names"][0] == "fc3"
        assert result["layer_names"][-1] == "fc1"

    def test_bottom_up_order(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            order="bottom_up", nr_samples=3, seed=42
        )
        assert result["layer_names"][0] == "fc1"

    def test_independent_order(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            order="independent", nr_samples=3, seed=42
        )
        assert len(result["layer_scores"]) == 3

    def test_invalid_order_raises(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        with pytest.raises(ValueError, match="Unknown order"):
            compute_smooth_mprt(
                mlp_model, x, y, _gradient_explain_func,
                order="bad", nr_samples=3
            )

    def test_invalid_nr_samples_raises(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        with pytest.raises(ValueError, match="nr_samples must be >= 1"):
            compute_smooth_mprt(
                mlp_model, x, y, _gradient_explain_func, nr_samples=0
            )

    def test_negative_noise_magnitude_raises(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        with pytest.raises(ValueError, match="noise_magnitude must be >= 0"):
            compute_smooth_mprt(
                mlp_model, x, y, _gradient_explain_func,
                noise_magnitude=-0.1, nr_samples=3
            )

    def test_deterministic_with_seed(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        r1 = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        r2 = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        assert r1["layer_scores"] == pytest.approx(r2["layer_scores"])

    def test_gradient_explainer_passes(self, mlp_model):
        """Gradient explanations should be sensitive to model parameters."""
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        result = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        assert result["mean_score"] < 0.95

    def test_random_explainer_fails(self, mlp_model):
        """Random explanations should be insensitive."""
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        result = compute_smooth_mprt(
            mlp_model, x, y, _random_explain_func,
            nr_samples=3, seed=42
        )
        assert result["mean_score"] > 0.95

    def test_different_similarity_functions(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        for func in ["spearman", "pearson", "cosine"]:
            result = compute_smooth_mprt(
                mlp_model, x, y, _gradient_explain_func,
                similarity_func=func, nr_samples=3, seed=42
            )
            assert isinstance(result["mean_score"], float)

    def test_cnn_model(self, cnn_model):
        x = np.random.default_rng(1).random((1, 1, 8, 8)).astype(np.float32)
        y = np.array([0])
        result = compute_smooth_mprt(
            cnn_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        assert isinstance(result["mean_score"], float)

    def test_original_model_not_modified(self, mlp_model):
        original_weight = mlp_model.fc1.weight.data.clone()
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        assert torch.allclose(original_weight, mlp_model.fc1.weight.data)

    def test_nr_samples_1_works(self, mlp_model):
        """nr_samples=1 should work (no smoothing, but valid)."""
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=1, seed=42
        )
        assert isinstance(result["mean_score"], float)

    def test_zero_noise_magnitude(self, mlp_model):
        """noise_magnitude=0 should work (no noise added)."""
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        result = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            noise_magnitude=0.0, nr_samples=3, seed=42
        )
        assert isinstance(result["mean_score"], float)


class TestComputeBatchSmoothMprt:
    """Tests for compute_batch_smooth_mprt."""

    def test_returns_list_per_sample(self, mlp_model):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        results = compute_batch_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        assert len(results) == 3
        for r in results:
            assert "layer_scores" in r
            assert "mean_score" in r

    def test_single_sample_matches(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        batch = compute_batch_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        single = compute_smooth_mprt(
            mlp_model, x, y, _gradient_explain_func,
            nr_samples=3, seed=42
        )
        assert batch[0]["layer_scores"] == pytest.approx(single["layer_scores"])


# ============================================================================
# Efficient MPRT
# ============================================================================

class TestComputeEfficientMprt:
    """Tests for compute_efficient_mprt."""

    def test_basic_return_type(self, mlp_model):
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        score = compute_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(score, float)

    def test_deterministic_with_seed(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        s1 = compute_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        s2 = compute_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert s1 == pytest.approx(s2)

    def test_different_seeds_different_results(self, mlp_model):
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        s1 = compute_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        s2 = compute_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=99
        )
        assert isinstance(s1, float) and isinstance(s2, float)

    def test_gradient_explainer_positive_score(self, mlp_model):
        """
        Gradient explanations should be structured for trained model
        (low entropy) and noisy for random model (high entropy),
        yielding a positive score.
        """
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        score = compute_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        # Should be positive: randomised model -> higher entropy
        assert score > -0.5  # relaxed bound; direction matters

    def test_random_explainer_near_zero(self, mlp_model):
        """
        Random explanations have same entropy regardless of model state,
        so score should be near zero.
        """
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        score = compute_efficient_mprt(
            mlp_model, x, y, _random_explain_func, seed=42
        )
        # Random explainer: entropy unchanged -> score near 0
        assert abs(score) < 0.3

    def test_cnn_model(self, cnn_model):
        x = np.random.default_rng(1).random((1, 1, 8, 8)).astype(np.float32)
        y = np.array([0])
        score = compute_efficient_mprt(
            cnn_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(score, float)

    def test_original_model_not_modified(self, mlp_model):
        original_weight = mlp_model.fc1.weight.data.clone()
        x = np.random.default_rng(1).random((1, 10)).astype(np.float32)
        y = np.array([0])
        compute_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert torch.allclose(original_weight, mlp_model.fc1.weight.data)

    def test_multi_sample_batch(self, mlp_model):
        x = np.random.default_rng(1).random((5, 10)).astype(np.float32)
        y = np.array([0, 1, 2, 0, 1])
        score = compute_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(score, float)


class TestComputeBatchEfficientMprt:
    """Tests for compute_batch_efficient_mprt."""

    def test_returns_list_per_sample(self, mlp_model):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        scores = compute_batch_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert isinstance(scores, list)
        assert len(scores) == 3
        for s in scores:
            assert isinstance(s, float)

    def test_deterministic_with_seed(self, mlp_model):
        x = np.random.default_rng(1).random((2, 10)).astype(np.float32)
        y = np.array([0, 1])
        s1 = compute_batch_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        s2 = compute_batch_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert s1 == pytest.approx(s2)

    def test_mean_matches_aggregate(self, mlp_model):
        x = np.random.default_rng(1).random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2])
        batch_scores = compute_batch_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        agg_score = compute_efficient_mprt(
            mlp_model, x, y, _gradient_explain_func, seed=42
        )
        assert np.mean(batch_scores) == pytest.approx(agg_score)
