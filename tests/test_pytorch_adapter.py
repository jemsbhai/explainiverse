# tests/test_pytorch_adapter.py
"""
Tests for PyTorchAdapter.

These tests require PyTorch to be installed. They will be skipped
if torch is not available.
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


@pytest.fixture
def simple_classifier():
    """Create a simple PyTorch classifier for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")

    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))

    # Initialize with some weights
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

    return model


@pytest.fixture
def sample_data():
    """Create sample input data."""
    np.random.seed(42)
    X = np.random.randn(10, 4).astype(np.float32)
    return X


class TestPyTorchAdapterBasic:
    """Basic functionality tests for PyTorchAdapter."""

    def test_adapter_creation(self, simple_classifier):
        """Adapter can be created with a PyTorch model."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(
            model=simple_classifier,
            task="classification",
            class_names=["setosa", "versicolor", "virginica"],
        )

        assert adapter.model is not None
        assert adapter.task == "classification"
        assert adapter.class_names == ["setosa", "versicolor", "virginica"]

    def test_adapter_rejects_non_pytorch_models(self):
        """Adapter raises error for non-PyTorch models."""
        from sklearn.linear_model import LogisticRegression

        from explainiverse.adapters import PyTorchAdapter

        sklearn_model = LogisticRegression()

        with pytest.raises(TypeError, match="nn.Module"):
            PyTorchAdapter(sklearn_model)

    def test_adapter_predict_classification(self, simple_classifier, sample_data):
        """Adapter produces valid classification predictions."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(
            model=simple_classifier, task="classification", class_names=["a", "b", "c"]
        )

        predictions = adapter.predict(sample_data)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (10, 3)
        # Softmax outputs should sum to 1
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)
        # All probabilities should be between 0 and 1
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_adapter_predict_regression(self, simple_regressor, sample_data):
        """Adapter produces valid regression predictions."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_regressor, task="regression")

        predictions = adapter.predict(sample_data)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (10, 1)

    def test_adapter_predict_single_instance(self, simple_classifier):
        """Adapter handles single instance input."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(
            model=simple_classifier, task="classification", class_names=["a", "b", "c"]
        )

        single_instance = np.random.randn(4).astype(np.float32)
        predictions = adapter.predict(single_instance)

        assert predictions.shape == (1, 3)

    def test_adapter_eval_mode(self, simple_classifier):
        """Adapter sets model to eval mode by default."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier)

        assert not adapter.model.training


class TestPyTorchAdapterGradients:
    """Tests for gradient-based functionality."""

    def test_predict_with_gradients(self, simple_classifier, sample_data):
        """Adapter can compute gradients w.r.t. inputs."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(
            model=simple_classifier, task="classification", class_names=["a", "b", "c"]
        )

        predictions, gradients = adapter.predict_with_gradients(sample_data[:1])

        assert isinstance(predictions, np.ndarray)
        assert isinstance(gradients, np.ndarray)
        assert predictions.shape == (1, 3)
        assert gradients.shape == (1, 4)  # Same shape as input
        # Gradients should not be all zeros (model is initialized)
        assert not np.allclose(gradients, 0)

    def test_each_target_gradient_matches_direct_autograd(self, simple_classifier, sample_data):
        """Every explicit target differentiates that exact raw-logit column."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(
            model=simple_classifier, task="classification", class_names=["a", "b", "c"]
        )

        direct_input = torch.tensor(sample_data[:1], requires_grad=True)
        direct_logits = simple_classifier(direct_input)

        for target_class in [0, 1]:
            scores, actual_gradient = adapter.predict_with_gradients(
                sample_data[:1], target_class=target_class
            )
            expected_gradient = torch.autograd.grad(
                direct_logits[0, target_class], direct_input, retain_graph=True
            )[0]

            np.testing.assert_allclose(scores, direct_logits.detach().numpy(), rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(
                actual_gradient,
                expected_gradient.detach().numpy(),
                rtol=1e-6,
                atol=1e-7,
            )


class TestPyTorchAdapterLayers:
    """Tests for layer access functionality."""

    def test_list_layers(self, simple_classifier):
        """Adapter can list model layers."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier)

        layers = adapter.list_layers()

        assert isinstance(layers, list)
        assert len(layers) > 0
        # Should include numbered layers from Sequential
        assert "0" in layers  # First Linear

    def test_get_layer_output(self, simple_classifier, sample_data):
        """Adapter can extract layer activations."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier)

        layers = adapter.list_layers()
        first_layer = layers[0]  # First Linear: 4 -> 16

        activations = adapter.get_layer_output(sample_data[:1], first_layer)

        assert isinstance(activations, np.ndarray)
        assert activations.shape[0] == 1  # One sample

    def test_get_layer_output_invalid_layer(self, simple_classifier, sample_data):
        """Adapter raises error for invalid layer name."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier)

        with pytest.raises(ValueError, match="not found"):
            adapter.get_layer_output(sample_data[:1], "nonexistent_layer")

    def test_get_layer_gradients(self, simple_classifier, sample_data):
        """Adapter can compute gradients for intermediate layers."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(
            model=simple_classifier, task="classification", class_names=["a", "b", "c"]
        )

        layers = adapter.list_layers()
        first_layer = layers[0]

        activations, gradients = adapter.get_layer_gradients(
            sample_data[:1], first_layer, target_class=0
        )

        assert isinstance(activations, np.ndarray)
        assert isinstance(gradients, np.ndarray)
        # Shapes should match
        assert activations.shape == gradients.shape

    def test_layer_gradients_support_inplace_relu_and_preserve_model_state(self):
        """Layer gradients do not leak state or install a backward-hook view."""
        from explainiverse.adapters import PyTorchAdapter

        class InplaceCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
                self.relu = nn.ReLU(inplace=True)
                self.batch_norm = nn.BatchNorm2d(2)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Linear(2, 2)

            def forward(self, x):
                x = self.batch_norm(self.relu(self.conv(x)))
                return self.head(self.pool(x).flatten(1))

        model = InplaceCNN()
        adapter = PyTorchAdapter(model, task="classification")
        # Exercise a mixed training-state tree and pre-existing gradients.
        model.train()
        model.conv.training = False
        parameters = list(model.parameters())
        parameters[0].grad = torch.full_like(parameters[0], 0.25)
        parameters[1].grad = None
        parameters[2].grad = torch.full_like(parameters[2], -0.5)
        parameters[3].grad = torch.full_like(parameters[3], 0.75)
        gradient_objects = [parameter.grad for parameter in parameters]
        gradient_values = [
            None if gradient is None else gradient.detach().clone() for gradient in gradient_objects
        ]
        modules = list(model.modules())
        training_states = [module.training for module in modules]
        buffers = list(model.buffers())
        buffer_values = [buffer.detach().clone() for buffer in buffers]

        activations, gradients = adapter.get_layer_gradients(
            np.ones((1, 1, 4, 4), dtype=np.float32),
            "conv",
            target_class=0,
        )

        assert activations.shape == gradients.shape == (1, 2, 4, 4)
        assert [module.training for module in modules] == training_states
        for buffer, expected in zip(buffers, buffer_values):
            torch.testing.assert_close(buffer, expected)
        for parameter, original, expected in zip(parameters, gradient_objects, gradient_values):
            assert parameter.grad is original
            if expected is not None:
                torch.testing.assert_close(parameter.grad, expected)


class TestPyTorchAdapterDevice:
    """Tests for device management."""

    def test_auto_device_detection(self, simple_classifier):
        """Adapter auto-detects device from model."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier)

        # Should detect CPU (model is on CPU by default)
        assert adapter.device.type == "cpu"

    def test_explicit_device_setting(self, simple_classifier):
        """Adapter respects explicit device setting."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier, device="cpu")

        assert adapter.device.type == "cpu"

    def test_device_change(self, simple_classifier):
        """Adapter can move model to different device."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier)

        # Should be chainable
        result = adapter.to("cpu")

        assert result is adapter
        assert adapter.device.type == "cpu"


class TestPyTorchAdapterOutputActivation:
    """Tests for output activation options."""

    def test_softmax_activation(self, simple_classifier, sample_data):
        """Softmax activation produces valid probabilities."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(
            model=simple_classifier, task="classification", output_activation="softmax"
        )

        predictions = adapter.predict(sample_data)

        # Should sum to 1
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)

    def test_sigmoid_activation(self, simple_classifier, sample_data):
        """Sigmoid activation produces values in [0, 1]."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier, output_activation="sigmoid")

        predictions = adapter.predict(sample_data)

        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_no_activation(self, simple_classifier, sample_data):
        """No activation returns raw logits."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier, output_activation="none")

        predictions = adapter.predict(sample_data)

        # Raw logits can be any value (not constrained to [0,1] or sum to 1)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (10, 3)


class TestPyTorchAdapterModes:
    """Tests for train/eval mode switching."""

    def test_train_mode(self, simple_classifier):
        """Adapter can switch to training mode."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier)

        result = adapter.train_mode()

        assert result is adapter
        assert adapter.model.training

    def test_eval_mode(self, simple_classifier):
        """Adapter can switch to evaluation mode."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier)
        adapter.train_mode()  # First set to train

        result = adapter.eval_mode()

        assert result is adapter
        assert not adapter.model.training


class TestPyTorchAdapterBatching:
    """Tests for batch processing."""

    def test_large_batch_processing(self, simple_classifier):
        """Adapter handles large inputs via batching."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(model=simple_classifier, task="classification", batch_size=16)

        # Create large input
        large_data = np.random.randn(100, 4).astype(np.float32)

        predictions = adapter.predict(large_data)

        assert predictions.shape == (100, 3)
        # All rows should sum to 1 (softmax)
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)


class TestPyTorchAdapterVerifiedOutputContract:
    """Analytical tests for the public prediction/gradient score contract."""

    @staticmethod
    def _binary_logit_model():
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(2.0)
            model.bias.fill_(-0.5)
        return model

    @staticmethod
    def _multiclass_logit_model():
        model = nn.Linear(2, 3)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -2.0],
                        [0.5, 3.0],
                        [-1.0, 0.25],
                    ]
                )
            )
            model.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        return model

    def test_one_logit_predict_expands_to_two_probability_columns(self):
        """A single binary logit is exposed as [P(class 0), P(class 1)]."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(
            self._binary_logit_model(),
            task="classification",
            class_names=["negative", "positive"],
        )
        x = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)

        predictions = adapter.predict(x)
        logits = 2.0 * x[:, 0] - 0.5
        positive = 1.0 / (1.0 + np.exp(-logits))

        assert predictions.shape == (3, 2)
        np.testing.assert_allclose(predictions[:, 0], 1.0 - positive, rtol=1e-6)
        np.testing.assert_allclose(predictions[:, 1], positive, rtol=1e-6)
        np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-7)

    def test_one_logit_target_gradients_match_probability_columns(self):
        """Binary class gradients are opposite derivatives of complementary probabilities."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(self._binary_logit_model(), task="classification")
        x = np.array([[0.5]], dtype=np.float32)
        expected_predictions = adapter.predict(x)
        positive = expected_predictions[0, 1]
        expected_positive_gradient = 2.0 * positive * (1.0 - positive)

        scores_0, gradient_0 = adapter.predict_with_gradients(x, target_class=np.int64(0))
        scores_1, gradient_1 = adapter.predict_with_gradients(x, target_class=np.int64(1))

        np.testing.assert_allclose(scores_0, expected_predictions, rtol=1e-6)
        np.testing.assert_allclose(scores_1, expected_predictions, rtol=1e-6)
        np.testing.assert_allclose(gradient_0, [[-expected_positive_gradient]], rtol=1e-6)
        np.testing.assert_allclose(gradient_1, [[expected_positive_gradient]], rtol=1e-6)
        assert adapter.last_gradient_output_space == "prediction"

    @pytest.mark.parametrize("target_class", [np.int64(0), np.int64(1)])
    def test_one_logit_gradient_completeness_uses_returned_score_space(self, target_class):
        """Integrated adapter gradients sum to the returned class-score difference."""
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(self._binary_logit_model(), task="classification")
        path = np.linspace(0.0, 1.0, 1001, dtype=np.float32).reshape(-1, 1)

        path_scores, path_gradients = adapter.predict_with_gradients(
            path, target_class=target_class
        )
        path_widths = np.diff(path[:, 0])
        integrated_gradient = np.sum(
            0.5 * (path_gradients[:-1, 0] + path_gradients[1:, 0]) * path_widths
        )
        returned_score_difference = (
            path_scores[-1, int(target_class)] - path_scores[0, int(target_class)]
        )

        np.testing.assert_allclose(
            integrated_gradient, returned_score_difference, rtol=1e-5, atol=1e-6
        )

    def test_multiclass_model_space_returns_and_differentiates_logits(self):
        """The backward-compatible model space returns raw logits and their gradient."""
        from explainiverse.adapters import PyTorchAdapter

        model = self._multiclass_logit_model()
        adapter = PyTorchAdapter(model, task="classification", gradient_output="model")
        x = np.array([[0.25, -0.5]], dtype=np.float32)

        scores, gradients = adapter.predict_with_gradients(x, target_class=np.int64(1))
        expected_scores = model(torch.from_numpy(x)).detach().numpy()

        np.testing.assert_allclose(scores, expected_scores, rtol=1e-6)
        np.testing.assert_allclose(gradients, [[0.5, 3.0]], rtol=1e-6)
        assert adapter.last_gradient_output_space == "model"

    def test_multiclass_prediction_space_returns_and_differentiates_probabilities(self):
        """Prediction-space gradients are the derivative of returned softmax values."""
        from explainiverse.adapters import PyTorchAdapter

        model = self._multiclass_logit_model()
        adapter = PyTorchAdapter(model, task="classification", gradient_output="prediction")
        x = np.array([[0.25, -0.5]], dtype=np.float32)

        scores, gradients = adapter.predict_with_gradients(x, target_class=1)
        probabilities = adapter.predict(x)
        weights = model.weight.detach().numpy()
        expected_gradient = probabilities[0, 1] * (weights[1] - probabilities[0] @ weights)

        np.testing.assert_allclose(scores, probabilities, rtol=1e-6)
        np.testing.assert_allclose(gradients[0], expected_gradient, rtol=1e-6)
        assert adapter.last_gradient_output_space == "prediction"

    def test_explicit_none_does_not_reactivate_probability_model(self):
        """Models that already return probabilities can opt out of double activation."""
        from explainiverse.adapters import PyTorchAdapter

        logits = self._multiclass_logit_model()
        model = nn.Sequential(logits, nn.Softmax(dim=-1))
        adapter = PyTorchAdapter(
            model,
            task="classification",
            output_activation="none",
            gradient_output="prediction",
        )
        x = np.array([[0.25, -0.5]], dtype=np.float32)

        expected = model(torch.from_numpy(x)).detach().numpy()
        predictions = adapter.predict(x)
        gradient_scores, _ = adapter.predict_with_gradients(x, target_class=2)

        np.testing.assert_allclose(predictions, expected, rtol=1e-6)
        np.testing.assert_allclose(gradient_scores, expected, rtol=1e-6)

    def test_one_probability_output_is_not_double_activated(self):
        """An explicit scalar P(class 1) is only expanded, never re-sigmoided."""
        from explainiverse.adapters import PyTorchAdapter

        logits = self._binary_logit_model()
        model = nn.Sequential(logits, nn.Sigmoid())
        adapter = PyTorchAdapter(model, task="classification", output_activation="none")
        x = np.array([[0.25]], dtype=np.float32)

        positive = model(torch.from_numpy(x)).detach().numpy()[0, 0]
        expected = np.array([[1.0 - positive, positive]])
        predictions = adapter.predict(x)
        gradient_scores, _ = adapter.predict_with_gradients(x, target_class=1)

        np.testing.assert_allclose(predictions, expected, rtol=1e-6)
        np.testing.assert_allclose(gradient_scores, expected, rtol=1e-6)

    def test_multioutput_regression_requires_an_explicit_output(self):
        """Multi-output regression never silently differentiates the output sum."""
        from explainiverse.adapters import PyTorchAdapter

        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0], [-3.0, 4.0]]))
        adapter = PyTorchAdapter(model, task="regression")
        x = np.array([[0.5, -0.25]], dtype=np.float32)

        with pytest.raises(ValueError, match="multi-output regression"):
            adapter.predict_with_gradients(x)

        scores, gradients = adapter.predict_with_gradients(x, target_class=np.int64(1))
        expected_scores = model(torch.from_numpy(x)).detach().numpy()

        np.testing.assert_allclose(scores, expected_scores, rtol=1e-6)
        np.testing.assert_allclose(gradients, [[-3.0, 4.0]], rtol=1e-6)


def test_auto_dtype_aligns_floating_inputs_to_a_double_model():
    from explainiverse.adapters import PyTorchAdapter

    model = nn.Linear(2, 1).double()
    adapter = PyTorchAdapter(model, task="regression")

    predictions = adapter.predict(np.array([[1.0, 2.0]], dtype=np.float32))
    assert predictions.dtype == np.float64


def test_explicit_input_dtype_configuration_is_applied():
    from explainiverse.adapters import PyTorchAdapter

    model = nn.Linear(2, 1).double()
    adapter = PyTorchAdapter(model, task="regression", input_dtype="float64")
    predictions = adapter.predict(np.array([[1.0, 2.0]], dtype=np.float32))
    assert predictions.dtype == np.float64


def test_integer_inputs_are_preserved_for_embedding_models():
    from explainiverse.adapters import PyTorchAdapter

    class EmbeddingClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 3)
            self.output = nn.Linear(3, 2)

        def forward(self, values):
            return self.output(self.embedding(values).mean(dim=1))

    adapter = PyTorchAdapter(
        EmbeddingClassifier(), task="classification", class_names=["no", "yes"]
    )
    predictions = adapter.predict(np.array([[1, 2], [3, 4]], dtype=np.int64))
    assert predictions.shape == (2, 2)


def test_numpy_class_names_are_detached_and_output_width_is_checked():
    from explainiverse.adapters import PyTorchAdapter

    names = np.array(["a", "b"], dtype=object)
    adapter = PyTorchAdapter(nn.Linear(2, 3), class_names=names)
    names[0] = "changed"
    assert adapter.class_names == ["a", "b"]
    with pytest.raises(ValueError, match="class_names has 2 entries"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


def test_whitespace_only_class_names_are_rejected():
    from explainiverse.adapters import PyTorchAdapter

    with pytest.raises(ValueError, match="non-empty"):
        PyTorchAdapter(nn.Linear(2, 2), class_names=["a", "   "])


def test_model_output_must_preserve_the_input_batch_dimension():
    from explainiverse.adapters import PyTorchAdapter

    class DropsRows(nn.Module):
        def forward(self, values):
            return values[:1, :2]

    adapter = PyTorchAdapter(DropsRows(), class_names=["a", "b"])
    with pytest.raises(ValueError, match="rows for a batch of 2"):
        adapter.predict(np.ones((2, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "batch_size, error", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_batch_size_is_strictly_validated(batch_size, error):
    from explainiverse.adapters import PyTorchAdapter

    with pytest.raises(error, match="positive integer"):
        PyTorchAdapter(nn.Linear(2, 1), task="regression", batch_size=batch_size)


def test_model_output_rank_and_finiteness_are_strictly_validated():
    from explainiverse.adapters import PyTorchAdapter

    class RankThree(nn.Module):
        def forward(self, values):
            return values[:, :1, None]

    class NonFinite(nn.Module):
        def forward(self, values):
            return values[:, :1] * float("nan")

    with pytest.raises(ValueError, match="one- or two-dimensional|shape"):
        PyTorchAdapter(RankThree()).predict(np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        PyTorchAdapter(NonFinite()).predict(np.ones((1, 2), dtype=np.float32))


def test_classification_outputs_must_be_floating_scores():
    from explainiverse.adapters import PyTorchAdapter

    class IntegerScores(nn.Module):
        def forward(self, values):
            return values[:, :2].to(torch.int64)

    with pytest.raises(TypeError, match="floating-point scores"):
        PyTorchAdapter(IntegerScores(), class_names=["a", "b"]).predict(
            np.ones((1, 2), dtype=np.float32)
        )


def test_integer_inputs_fail_clearly_for_input_gradients():
    from explainiverse.adapters import PyTorchAdapter

    class EmbeddingClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 2)

        def forward(self, values):
            return self.embedding(values).mean(dim=1)

    adapter = PyTorchAdapter(EmbeddingClassifier(), class_names=["a", "b"])
    with pytest.raises(TypeError, match="floating-point input"):
        adapter.predict_with_gradients(np.array([[1, 2]], dtype=np.int64), target_class=0)


def test_per_sample_targets_must_be_one_dimensional():
    from explainiverse.adapters import PyTorchAdapter

    adapter = PyTorchAdapter(nn.Linear(2, 2), class_names=["a", "b"])
    with pytest.raises(ValueError, match="one-dimensional"):
        adapter.predict_with_gradients(
            np.ones((2, 2), dtype=np.float32),
            target_class=np.array([[0], [1]], dtype=np.int64),
        )


def test_complex_torch_targets_are_rejected_as_non_indices():
    from explainiverse.adapters import PyTorchAdapter

    adapter = PyTorchAdapter(nn.Linear(2, 2), class_names=["a", "b"])
    with pytest.raises(TypeError, match="integer indices"):
        adapter.predict_with_gradients(
            np.ones((1, 2), dtype=np.float32),
            target_class=torch.tensor([0 + 0j]),
        )


def test_gradient_paths_do_not_mutate_caller_tensor_autograd_state():
    from explainiverse.adapters import PyTorchAdapter

    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    adapter = PyTorchAdapter(model, class_names=["a", "b"], input_dtype="preserve")
    caller_input = torch.ones((1, 2), dtype=torch.float32)

    adapter.predict_with_gradients(caller_input, target_class=0)
    assert caller_input.requires_grad is False
    assert caller_input.grad is None

    adapter.get_layer_gradients(caller_input, "0", target_class=0)
    assert caller_input.requires_grad is False
    assert caller_input.grad is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
