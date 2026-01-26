# tests/test_pytorch_adapter.py
"""
Tests for PyTorchAdapter.

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
    
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
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
            class_names=["setosa", "versicolor", "virginica"]
        )
        
        assert adapter.model is not None
        assert adapter.task == "classification"
        assert adapter.class_names == ["setosa", "versicolor", "virginica"]
    
    def test_adapter_rejects_non_pytorch_models(self):
        """Adapter raises error for non-PyTorch models."""
        from explainiverse.adapters import PyTorchAdapter
        from sklearn.linear_model import LogisticRegression
        
        sklearn_model = LogisticRegression()
        
        with pytest.raises(TypeError, match="nn.Module"):
            PyTorchAdapter(sklearn_model)
    
    def test_adapter_predict_classification(self, simple_classifier, sample_data):
        """Adapter produces valid classification predictions."""
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(
            model=simple_classifier,
            task="classification",
            class_names=["a", "b", "c"]
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
        
        adapter = PyTorchAdapter(
            model=simple_regressor,
            task="regression"
        )
        
        predictions = adapter.predict(sample_data)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (10, 1)
    
    def test_adapter_predict_single_instance(self, simple_classifier):
        """Adapter handles single instance input."""
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(
            model=simple_classifier,
            task="classification",
            class_names=["a", "b", "c"]
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
            model=simple_classifier,
            task="classification",
            class_names=["a", "b", "c"]
        )
        
        predictions, gradients = adapter.predict_with_gradients(sample_data[:1])
        
        assert isinstance(predictions, np.ndarray)
        assert isinstance(gradients, np.ndarray)
        assert predictions.shape == (1, 3)
        assert gradients.shape == (1, 4)  # Same shape as input
        # Gradients should not be all zeros (model is initialized)
        assert not np.allclose(gradients, 0)
    
    def test_predict_with_gradients_target_class(self, simple_classifier, sample_data):
        """Gradient computation respects target class."""
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(
            model=simple_classifier,
            task="classification",
            class_names=["a", "b", "c"]
        )
        
        _, gradients_class_0 = adapter.predict_with_gradients(sample_data[:1], target_class=0)
        _, gradients_class_1 = adapter.predict_with_gradients(sample_data[:1], target_class=1)
        
        # Gradients for different classes should differ
        assert not np.allclose(gradients_class_0, gradients_class_1)


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
            model=simple_classifier,
            task="classification",
            class_names=["a", "b", "c"]
        )
        
        layers = adapter.list_layers()
        first_layer = layers[0]
        
        activations, gradients = adapter.get_layer_gradients(
            sample_data[:1],
            first_layer,
            target_class=0
        )
        
        assert isinstance(activations, np.ndarray)
        assert isinstance(gradients, np.ndarray)
        # Shapes should match
        assert activations.shape == gradients.shape


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
            model=simple_classifier,
            task="classification",
            output_activation="softmax"
        )
        
        predictions = adapter.predict(sample_data)
        
        # Should sum to 1
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)
    
    def test_sigmoid_activation(self, simple_classifier, sample_data):
        """Sigmoid activation produces values in [0, 1]."""
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(
            model=simple_classifier,
            output_activation="sigmoid"
        )
        
        predictions = adapter.predict(sample_data)
        
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
    
    def test_no_activation(self, simple_classifier, sample_data):
        """No activation returns raw logits."""
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(
            model=simple_classifier,
            output_activation="none"
        )
        
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
        
        adapter = PyTorchAdapter(
            model=simple_classifier,
            task="classification",
            batch_size=16
        )
        
        # Create large input
        large_data = np.random.randn(100, 4).astype(np.float32)
        
        predictions = adapter.predict(large_data)
        
        assert predictions.shape == (100, 3)
        # All rows should sum to 1 (softmax)
        assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
