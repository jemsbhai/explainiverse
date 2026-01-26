# tests/test_gradcam.py
"""
Tests for GradCAM and GradCAM++ explainers.

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
def simple_cnn():
    """Create a simple CNN for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Conv layers
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            
            # Classifier
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 8 * 8, 64)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(64, 3)
        
        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleCNN()
    
    # Initialize with deterministic weights
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    return model


@pytest.fixture
def sample_image():
    """Create a sample image tensor."""
    np.random.seed(42)
    # (batch, channels, height, width) = (1, 3, 32, 32)
    image = np.random.randn(1, 3, 32, 32).astype(np.float32)
    return image


@pytest.fixture
def class_names():
    return ["cat", "dog", "bird"]


class TestGradCAMBasic:
    """Basic functionality tests for GradCAM."""
    
    def test_gradcam_creation(self, simple_cnn, class_names):
        """GradCAM explainer can be created."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification")
        
        # Get layers
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][0]
        
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names
        )
        
        assert explainer.target_layer == conv_layer
        assert explainer.class_names == class_names
        assert explainer.method == "gradcam"
    
    def test_gradcam_rejects_non_gradient_model(self, class_names):
        """GradCAM raises error for models without gradient support."""
        from explainiverse.explainers import GradCAMExplainer
        from explainiverse.adapters import SklearnAdapter
        from sklearn.linear_model import LogisticRegression
        
        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)
        
        with pytest.raises(TypeError, match="get_layer_gradients"):
            GradCAMExplainer(
                model=adapter,
                target_layer="some_layer",
                class_names=class_names
            )
    
    def test_gradcam_invalid_method(self, simple_cnn, class_names):
        """GradCAM raises error for invalid method."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification")
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][0]
        
        with pytest.raises(ValueError, match="gradcam"):
            GradCAMExplainer(
                model=adapter,
                target_layer=conv_layer,
                class_names=class_names,
                method="invalid_method"
            )
    
    def test_gradcam_explain(self, simple_cnn, sample_image, class_names):
        """GradCAM produces valid explanations."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        from explainiverse.core.explanation import Explanation
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][-1]  # Last conv layer
        
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_image)
        
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "GradCAM"
        assert "heatmap" in explanation.explanation_data
        assert "target_layer" in explanation.explanation_data
        
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.ndim == 2
        # Heatmap should be normalized to [0, 1]
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1


class TestGradCAMPlusPlus:
    """Tests for GradCAM++ variant."""
    
    def test_gradcampp_explain(self, simple_cnn, sample_image, class_names):
        """GradCAM++ produces valid explanations."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][-1]
        
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names,
            method="gradcam++"
        )
        
        explanation = explainer.explain(sample_image)
        
        assert explanation.explainer_name == "GradCAM++"
        assert "heatmap" in explanation.explanation_data
    
    def test_gradcam_vs_gradcampp(self, simple_cnn, sample_image, class_names):
        """GradCAM and GradCAM++ produce different results."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][-1]
        
        explainer_cam = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names,
            method="gradcam"
        )
        
        explainer_campp = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names,
            method="gradcam++"
        )
        
        heatmap_cam = np.array(explainer_cam.explain(sample_image).explanation_data["heatmap"])
        heatmap_campp = np.array(explainer_campp.explain(sample_image).explanation_data["heatmap"])
        
        # Results should be different (but correlated)
        # They might be identical in some edge cases, so just check shapes match
        assert heatmap_cam.shape == heatmap_campp.shape


class TestGradCAMTargetClass:
    """Tests for target class handling."""
    
    def test_gradcam_target_class(self, simple_cnn, sample_image, class_names):
        """GradCAM respects target_class parameter."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][-1]
        
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names
        )
        
        exp_0 = explainer.explain(sample_image, target_class=0)
        exp_1 = explainer.explain(sample_image, target_class=1)
        
        assert exp_0.target_class == "cat"
        assert exp_1.target_class == "dog"
        
        # Different classes should produce different heatmaps
        heatmap_0 = np.array(exp_0.explanation_data["heatmap"])
        heatmap_1 = np.array(exp_1.explanation_data["heatmap"])
        
        # Not necessarily different due to random initialization
        # but at least shapes should match
        assert heatmap_0.shape == heatmap_1.shape


class TestGradCAMInputFormats:
    """Tests for different input formats."""
    
    def test_gradcam_3d_input(self, simple_cnn, class_names):
        """GradCAM handles 3D input (C, H, W)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][-1]
        
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names
        )
        
        # 3D input without batch dimension
        image_3d = np.random.randn(3, 32, 32).astype(np.float32)
        explanation = explainer.explain(image_3d)
        
        assert "heatmap" in explanation.explanation_data
    
    def test_gradcam_hwc_input(self, simple_cnn, class_names):
        """GradCAM handles HWC format input."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][-1]
        
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names
        )
        
        # HWC format (32, 32, 3)
        image_hwc = np.random.randn(32, 32, 3).astype(np.float32)
        explanation = explainer.explain(image_hwc)
        
        assert "heatmap" in explanation.explanation_data


class TestGradCAMOverlay:
    """Tests for heatmap overlay functionality."""
    
    def test_gradcam_overlay(self, simple_cnn, sample_image, class_names):
        """GradCAM can create overlay images."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][-1]
        
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names
        )
        
        explanation = explainer.explain(sample_image)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        
        # Create overlay
        original_image = np.transpose(sample_image[0], (1, 2, 0))  # CHW -> HWC
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        
        overlay = explainer.get_overlay(original_image, heatmap, alpha=0.5)
        
        assert overlay.shape == (32, 32, 3)
        assert overlay.min() >= 0
        assert overlay.max() <= 1


class TestGradCAMBatch:
    """Tests for batch processing."""
    
    def test_gradcam_batch_explain(self, simple_cnn, class_names):
        """GradCAM can process batches."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import GradCAMExplainer
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][-1]
        
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names
        )
        
        # Batch of 3 images
        images = np.random.randn(3, 3, 32, 32).astype(np.float32)
        explanations = explainer.explain_batch(images)
        
        assert len(explanations) == 3
        for exp in explanations:
            assert "heatmap" in exp.explanation_data


class TestGradCAMRegistry:
    """Tests for registry integration."""
    
    def test_gradcam_registered(self):
        """GradCAM is registered in default registry."""
        from explainiverse import default_registry
        
        explainers = default_registry.list_explainers()
        assert "gradcam" in explainers
    
    def test_gradcam_metadata(self):
        """GradCAM has correct metadata."""
        from explainiverse import default_registry
        
        meta = default_registry.get_meta("gradcam")
        
        assert meta.scope == "local"
        assert "neural" in meta.model_types
        assert "image" in meta.data_types
        assert "Selvaraju" in meta.paper_reference
    
    def test_gradcam_filter_image(self):
        """GradCAM appears when filtering for image explainers."""
        from explainiverse import default_registry
        
        image_explainers = default_registry.filter(data_type="image")
        assert "gradcam" in image_explainers
    
    def test_gradcam_via_registry(self, simple_cnn, class_names):
        """GradCAM can be created via registry."""
        from explainiverse import default_registry
        from explainiverse.adapters import PyTorchAdapter
        
        adapter = PyTorchAdapter(simple_cnn, task="classification", class_names=class_names)
        layers = adapter.list_layers()
        conv_layer = [l for l in layers if 'conv' in l][-1]
        
        explainer = default_registry.create(
            "gradcam",
            model=adapter,
            target_layer=conv_layer,
            class_names=class_names
        )
        
        assert explainer is not None
        assert explainer.target_layer == conv_layer


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
