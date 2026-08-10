# tests/test_cam_variants.py
"""
Tests for CAM variant explainers: HiResCAM, XGradCAM, LayerCAM, EigenCAM, ScoreCAM.

These tests exercise the implemented APIs; formula-level comparisons live in
``test_cam_accuracy.py`` and ``tests/reference/test_ref_cam.py``.

References:
    HiResCAM: Draelos & Carin, 2020 — "Use HiResCAM instead of Grad-CAM for
        faithful explanations of neural networks"
    XGradCAM: Fu et al., 2020 — "Axiom-based Grad-CAM: Towards Accurate
        Visualization and Explanation of CNNs"
    LayerCAM: Jiang et al., 2021 — "LayerCAM: Exploring Hierarchical
        Class Activation Maps for Localization" (IEEE TIP)
    EigenCAM: Muhammad & Yeasin, 2020 — "Eigen-CAM: Class Activation Map
        using Principal Components"
    ScoreCAM: Wang et al., 2020 — "Score-CAM: Score-Weighted Visual
        Explanations for Convolutional Neural Networks"
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


# ──────────────────────────────────────────────
# Shared Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def simple_cnn():
    """Create a simple CNN for testing CAM methods."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)

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
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


@pytest.fixture
def adapter(simple_cnn):
    """PyTorchAdapter wrapping the simple CNN."""
    from explainiverse.adapters import PyTorchAdapter

    return PyTorchAdapter(simple_cnn, task="classification", class_names=["cat", "dog", "bird"])


@pytest.fixture
def last_conv_layer(adapter):
    """Name of the last convolutional layer."""
    layers = adapter.list_layers()
    conv_layers = [layer for layer in layers if "conv" in layer]
    return conv_layers[-1]


@pytest.fixture
def sample_image():
    """Single sample image (1, 3, 32, 32)."""
    np.random.seed(42)
    return np.random.randn(1, 3, 32, 32).astype(np.float32)


@pytest.fixture
def class_names():
    return ["cat", "dog", "bird"]


# ──────────────────────────────────────────────
# Helper to validate a CAM explanation
# ──────────────────────────────────────────────


def _assert_valid_cam_explanation(explanation, expected_name):
    """Validate common properties of a CAM explanation."""
    from explainiverse.core.explanation import Explanation

    assert isinstance(explanation, Explanation)
    assert explanation.explainer_name == expected_name
    assert "heatmap" in explanation.explanation_data
    assert "target_layer" in explanation.explanation_data
    assert "method" in explanation.explanation_data

    heatmap = np.array(explanation.explanation_data["heatmap"])
    assert heatmap.ndim == 2, f"Heatmap should be 2D, got {heatmap.ndim}D"
    assert heatmap.min() >= 0, "Heatmap min should be >= 0"
    assert heatmap.max() <= 1.0 + 1e-6, "Heatmap max should be <= 1"


# ══════════════════════════════════════════════
# HiResCAM Tests
# ══════════════════════════════════════════════


class TestHiResCAMBasic:
    """Basic functionality tests for HiResCAM."""

    def test_creation(self, adapter, last_conv_layer, class_names):
        """HiResCAM explainer can be created."""
        from explainiverse.explainers.gradient import HiResCAMExplainer

        explainer = HiResCAMExplainer(
            model=adapter,
            target_layer=last_conv_layer,
            class_names=class_names,
        )
        assert explainer.target_layer == last_conv_layer
        assert explainer.class_names == class_names

    def test_explain(self, adapter, last_conv_layer, sample_image, class_names):
        """HiResCAM produces valid explanation."""
        from explainiverse.explainers.gradient import HiResCAMExplainer

        explainer = HiResCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        _assert_valid_cam_explanation(explanation, "HiResCAM")

    def test_target_class(self, adapter, last_conv_layer, sample_image, class_names):
        """HiResCAM respects target_class parameter."""
        from explainiverse.explainers.gradient import HiResCAMExplainer

        explainer = HiResCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        exp0 = explainer.explain(sample_image, target_class=0)
        exp1 = explainer.explain(sample_image, target_class=1)

        assert exp0.target_class == "cat"
        assert exp1.target_class == "dog"

    def test_heatmap_resized_to_input(self, adapter, last_conv_layer, sample_image, class_names):
        """HiResCAM resizes heatmap to input spatial dimensions."""
        from explainiverse.explainers.gradient import HiResCAMExplainer

        explainer = HiResCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=True)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.shape == (32, 32)

    def test_3d_input(self, adapter, last_conv_layer, class_names):
        """HiResCAM handles 3D (C, H, W) input."""
        from explainiverse.explainers.gradient import HiResCAMExplainer

        explainer = HiResCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        image_3d = np.random.randn(3, 32, 32).astype(np.float32)
        explanation = explainer.explain(image_3d)
        _assert_valid_cam_explanation(explanation, "HiResCAM")

    def test_elementwise_product(self, adapter, last_conv_layer, sample_image, class_names):
        """HiResCAM applies the element-wise formula without a blanket guarantee."""
        from explainiverse.explainers.gradient import HiResCAMExplainer

        explainer = HiResCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        # Simply verify it runs and the method is recorded
        explanation = explainer.explain(sample_image)
        assert explanation.explanation_data["method"] == "hirescam"
        assert explanation.metadata["faithfulness_guarantee_asserted"] is False

    def test_batch_explain(self, adapter, last_conv_layer, class_names):
        """HiResCAM can process batches."""
        from explainiverse.explainers.gradient import HiResCAMExplainer

        explainer = HiResCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        images = np.random.randn(3, 3, 32, 32).astype(np.float32)
        explanations = explainer.explain_batch(images)
        assert len(explanations) == 3
        for exp in explanations:
            _assert_valid_cam_explanation(exp, "HiResCAM")

    def test_rejects_sklearn(self, class_names):
        """HiResCAM rejects non-gradient models."""
        from sklearn.linear_model import LogisticRegression

        from explainiverse.adapters import SklearnAdapter
        from explainiverse.explainers.gradient import HiResCAMExplainer

        sk = LogisticRegression()
        sk.fit(np.random.randn(20, 4), np.random.randint(0, 3, 20))
        with pytest.raises(TypeError, match="get_layer_gradients"):
            HiResCAMExplainer(model=SklearnAdapter(sk), target_layer="x", class_names=class_names)


# ══════════════════════════════════════════════
# XGradCAM Tests
# ══════════════════════════════════════════════


class TestXGradCAMBasic:
    """Basic functionality tests for XGradCAM."""

    def test_creation(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import XGradCAMExplainer

        explainer = XGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        assert explainer.target_layer == last_conv_layer

    def test_explain(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import XGradCAMExplainer

        explainer = XGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        _assert_valid_cam_explanation(explanation, "XGradCAM")

    def test_target_class(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import XGradCAMExplainer

        explainer = XGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        exp0 = explainer.explain(sample_image, target_class=0)
        exp1 = explainer.explain(sample_image, target_class=2)
        assert exp0.target_class == "cat"
        assert exp1.target_class == "bird"

    def test_axiom_based_weights(self, adapter, last_conv_layer, sample_image, class_names):
        """XGradCAM uses activation-normalized gradient weights."""
        from explainiverse.explainers.gradient import XGradCAMExplainer

        explainer = XGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        assert explanation.explanation_data["method"] == "xgradcam"

    def test_heatmap_shape(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import XGradCAMExplainer

        explainer = XGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=True)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.shape == (32, 32)

    def test_batch_explain(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import XGradCAMExplainer

        explainer = XGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        images = np.random.randn(3, 3, 32, 32).astype(np.float32)
        explanations = explainer.explain_batch(images)
        assert len(explanations) == 3

    def test_3d_input(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import XGradCAMExplainer

        explainer = XGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        image_3d = np.random.randn(3, 32, 32).astype(np.float32)
        explanation = explainer.explain(image_3d)
        _assert_valid_cam_explanation(explanation, "XGradCAM")


# ══════════════════════════════════════════════
# LayerCAM Tests
# ══════════════════════════════════════════════


class TestLayerCAMBasic:
    """Basic functionality tests for LayerCAM."""

    def test_creation(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import LayerCAMExplainer

        explainer = LayerCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        assert explainer.target_layer == last_conv_layer

    def test_explain(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import LayerCAMExplainer

        explainer = LayerCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        _assert_valid_cam_explanation(explanation, "LayerCAM")

    def test_target_class(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import LayerCAMExplainer

        explainer = LayerCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        exp0 = explainer.explain(sample_image, target_class=0)
        exp1 = explainer.explain(sample_image, target_class=1)
        assert exp0.target_class == "cat"
        assert exp1.target_class == "dog"

    def test_spatial_weighting(self, adapter, last_conv_layer, sample_image, class_names):
        """LayerCAM uses spatial-weighted ReLU(grads) * activations."""
        from explainiverse.explainers.gradient import LayerCAMExplainer

        explainer = LayerCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        assert explanation.explanation_data["method"] == "layercam"

    def test_works_on_earlier_layer(self, adapter, sample_image, class_names):
        """LayerCAM is designed to work at any layer, not just the last conv."""
        from explainiverse.explainers.gradient import LayerCAMExplainer

        layers = adapter.list_layers()
        first_conv = [layer for layer in layers if "conv" in layer][0]

        explainer = LayerCAMExplainer(
            model=adapter, target_layer=first_conv, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        _assert_valid_cam_explanation(explanation, "LayerCAM")

    def test_batch_explain(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import LayerCAMExplainer

        explainer = LayerCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        images = np.random.randn(3, 3, 32, 32).astype(np.float32)
        explanations = explainer.explain_batch(images)
        assert len(explanations) == 3

    def test_heatmap_shape(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import LayerCAMExplainer

        explainer = LayerCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=True)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.shape == (32, 32)


# ══════════════════════════════════════════════
# EigenCAM Tests
# ══════════════════════════════════════════════


class TestEigenCAMBasic:
    """Basic functionality tests for EigenCAM."""

    def test_creation(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import EigenCAMExplainer

        explainer = EigenCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        assert explainer.target_layer == last_conv_layer

    def test_explain(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import EigenCAMExplainer

        explainer = EigenCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        _assert_valid_cam_explanation(explanation, "EigenCAM")

    def test_gradient_free(self, adapter, last_conv_layer, sample_image, class_names):
        """EigenCAM does not require gradients (uses SVD on activations only)."""
        from explainiverse.explainers.gradient import EigenCAMExplainer

        explainer = EigenCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        assert explanation.explanation_data["method"] == "eigencam"

    def test_class_agnostic(self, adapter, last_conv_layer, sample_image, class_names):
        """EigenCAM is class-agnostic — same heatmap regardless of target_class."""
        from explainiverse.explainers.gradient import EigenCAMExplainer

        explainer = EigenCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        assert explanation.target_class == "class_agnostic"
        assert explanation.metadata["class_agnostic"] is True
        with pytest.raises(ValueError, match="class-agnostic"):
            explainer.explain(sample_image, target_class=0)

    def test_heatmap_shape(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import EigenCAMExplainer

        explainer = EigenCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=True)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.shape == (32, 32)

    def test_batch_explain(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import EigenCAMExplainer

        explainer = EigenCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        images = np.random.randn(3, 3, 32, 32).astype(np.float32)
        explanations = explainer.explain_batch(images)
        assert len(explanations) == 3

    def test_3d_input(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import EigenCAMExplainer

        explainer = EigenCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        image_3d = np.random.randn(3, 32, 32).astype(np.float32)
        explanation = explainer.explain(image_3d)
        _assert_valid_cam_explanation(explanation, "EigenCAM")


# ══════════════════════════════════════════════
# ScoreCAM Tests
# ══════════════════════════════════════════════


class TestScoreCAMBasic:
    """Basic functionality tests for ScoreCAM."""

    def test_creation(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import ScoreCAMExplainer

        explainer = ScoreCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        assert explainer.target_layer == last_conv_layer

    def test_explain(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import ScoreCAMExplainer

        explainer = ScoreCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        _assert_valid_cam_explanation(explanation, "ScoreCAM")

    def test_gradient_free(self, adapter, last_conv_layer, sample_image, class_names):
        """ScoreCAM does not use gradients — uses forward-pass scoring."""
        from explainiverse.explainers.gradient import ScoreCAMExplainer

        explainer = ScoreCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        assert explanation.explanation_data["method"] == "scorecam"

    def test_target_class(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import ScoreCAMExplainer

        explainer = ScoreCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        exp0 = explainer.explain(sample_image, target_class=0)
        exp1 = explainer.explain(sample_image, target_class=1)
        assert exp0.target_class == "cat"
        assert exp1.target_class == "dog"

    def test_heatmap_shape(self, adapter, last_conv_layer, sample_image, class_names):
        from explainiverse.explainers.gradient import ScoreCAMExplainer

        explainer = ScoreCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=True)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.shape == (32, 32)

    def test_batch_size_parameter(self, adapter, last_conv_layer, class_names):
        """ScoreCAM accepts batch_size for forward-pass efficiency."""
        from explainiverse.explainers.gradient import ScoreCAMExplainer

        explainer = ScoreCAMExplainer(
            model=adapter,
            target_layer=last_conv_layer,
            class_names=class_names,
            batch_size=8,
        )
        assert explainer.batch_size == 8

    def test_batch_explain(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import ScoreCAMExplainer

        explainer = ScoreCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        images = np.random.randn(2, 3, 32, 32).astype(np.float32)
        explanations = explainer.explain_batch(images)
        assert len(explanations) == 2

    def test_3d_input(self, adapter, last_conv_layer, class_names):
        from explainiverse.explainers.gradient import ScoreCAMExplainer

        explainer = ScoreCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        image_3d = np.random.randn(3, 32, 32).astype(np.float32)
        explanation = explainer.explain(image_3d)
        _assert_valid_cam_explanation(explanation, "ScoreCAM")


# ══════════════════════════════════════════════
# Cross-Method Comparison Tests
# ══════════════════════════════════════════════


class TestCAMCrossComparison:
    """Tests comparing different CAM methods against each other."""

    def _get_heatmap(self, explainer_cls, adapter, layer, image, class_names):
        explainer = explainer_cls(model=adapter, target_layer=layer, class_names=class_names)
        explanation = explainer.explain(image)
        return np.array(explanation.explanation_data["heatmap"])

    def test_all_methods_produce_same_shape(
        self, adapter, last_conv_layer, sample_image, class_names
    ):
        """All CAM variants produce heatmaps of the same spatial dimensions."""
        from explainiverse.explainers.gradient import (
            EigenCAMExplainer,
            HiResCAMExplainer,
            LayerCAMExplainer,
            ScoreCAMExplainer,
            XGradCAMExplainer,
        )

        methods = [
            HiResCAMExplainer,
            XGradCAMExplainer,
            LayerCAMExplainer,
            EigenCAMExplainer,
            ScoreCAMExplainer,
        ]
        shapes = []
        for cls in methods:
            h = self._get_heatmap(cls, adapter, last_conv_layer, sample_image, class_names)
            shapes.append(h.shape)

        assert len(set(shapes)) == 1, f"Shapes differ across methods: {shapes}"

    def test_gradient_and_gradient_free_heatmaps_are_nonnegative(
        self, adapter, last_conv_layer, sample_image, class_names
    ):
        """Two configured CAM variants return non-negative normalized maps."""
        from explainiverse.explainers.gradient import EigenCAMExplainer, HiResCAMExplainer

        h_hires = self._get_heatmap(
            HiResCAMExplainer, adapter, last_conv_layer, sample_image, class_names
        )
        h_eigen = self._get_heatmap(
            EigenCAMExplainer, adapter, last_conv_layer, sample_image, class_names
        )
        assert h_hires.min() >= 0
        assert h_eigen.min() >= 0

    def test_selected_methods_deterministic(
        self, adapter, last_conv_layer, sample_image, class_names
    ):
        """The four selected variants repeat a fixed-input heatmap."""
        from explainiverse.explainers.gradient import (
            EigenCAMExplainer,
            HiResCAMExplainer,
            LayerCAMExplainer,
            XGradCAMExplainer,
        )

        for cls in [HiResCAMExplainer, XGradCAMExplainer, LayerCAMExplainer, EigenCAMExplainer]:
            h1 = self._get_heatmap(cls, adapter, last_conv_layer, sample_image, class_names)
            h2 = self._get_heatmap(cls, adapter, last_conv_layer, sample_image, class_names)
            np.testing.assert_array_almost_equal(
                h1, h2, decimal=5, err_msg=f"{cls.__name__} is not deterministic"
            )


# ══════════════════════════════════════════════
# Registry Integration Tests
# ══════════════════════════════════════════════


class TestCAMVariantsRegistry:
    """Test that all CAM variants are registered in the ExplainerRegistry."""

    @pytest.mark.parametrize(
        "name,expected_ref",
        [
            ("hirescam", "Draelos"),
            ("xgradcam", "Fu"),
            ("layercam", "Jiang"),
            ("eigencam", "Muhammad"),
            ("scorecam", "Wang"),
        ],
    )
    def test_registered(self, name, expected_ref):
        from explainiverse import default_registry

        explainers = default_registry.list_explainers()
        assert name in explainers, f"{name} not found in registry"

        meta = default_registry.get_meta(name)
        assert meta.scope == "local"
        assert "image" in meta.data_types
        assert "neural" in meta.model_types
        assert expected_ref in meta.paper_reference

    @pytest.mark.parametrize(
        "name",
        [
            "hirescam",
            "xgradcam",
            "layercam",
            "eigencam",
            "scorecam",
        ],
    )
    def test_filter_image(self, name):
        from explainiverse import default_registry

        image_explainers = default_registry.filter(data_type="image")
        assert name in image_explainers

    @pytest.mark.parametrize(
        "name",
        [
            "hirescam",
            "xgradcam",
            "layercam",
            "eigencam",
            "scorecam",
        ],
    )
    def test_create_via_registry(self, adapter, last_conv_layer, class_names, name):
        from explainiverse import default_registry

        explainer = default_registry.create(
            name,
            model=adapter,
            target_layer=last_conv_layer,
            class_names=class_names,
        )
        assert explainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
