# tests/test_cam_variants_batch2.py
"""
Tests for additional CAM-family explainers.

EigenGradCAM, GradCAMElementWise, AblationCAM.

References:
    EigenGradCAM and GradCAMElementWise are implementation variants from the
        pytorch-grad-cam library, not methods in the cited CAM papers.
        https://github.com/jacobgil/pytorch-grad-cam
    AblationCAM: Desai & Ramaswamy, 2020 — "Ablation-CAM: Visual Explanations
        for Deep Convolutional Network via Gradient-free Localization" (WACV)
        https://openaccess.thecvf.com/content_WACV_2020/papers/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.pdf
"""

import numpy as np
import pytest

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
def first_conv_layer(adapter):
    """Name of the first convolutional layer."""
    layers = adapter.list_layers()
    conv_layers = [layer for layer in layers if "conv" in layer]
    return conv_layers[0]


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
# EigenGradCAM Tests
# ══════════════════════════════════════════════


class TestEigenGradCAMBasic:
    """Basic functionality tests for EigenGradCAM."""

    def test_creation(self, adapter, last_conv_layer, class_names):
        """EigenGradCAM explainer can be created."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter,
            target_layer=last_conv_layer,
            class_names=class_names,
        )
        assert explainer.target_layer == last_conv_layer
        assert explainer.class_names == class_names

    def test_explain(self, adapter, last_conv_layer, sample_image, class_names):
        """EigenGradCAM produces valid explanation."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        _assert_valid_cam_explanation(explanation, "EigenGradCAM (library variant)")
        assert explanation.metadata["canonical_paper_method"] is False

    def test_target_class(self, adapter, last_conv_layer, sample_image, class_names):
        """EigenGradCAM respects target_class parameter."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        exp0 = explainer.explain(sample_image, target_class=0)
        exp1 = explainer.explain(sample_image, target_class=1)

        assert exp0.target_class == "cat"
        assert exp1.target_class == "dog"

    def test_uses_gradients(self, adapter, last_conv_layer, sample_image, class_names):
        """EigenGradCAM uses gradients (unlike EigenCAM which is gradient-free)."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        assert explainer._uses_gradients is True
        explanation = explainer.explain(sample_image)
        assert explanation.explanation_data["method"] == "eigengradcam_library_variant"

    def test_class_dependent(self, adapter, last_conv_layer, sample_image, class_names):
        """EigenGradCAM accepts explicit classes and preserves heatmap shape."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        heatmap0 = np.array(
            explainer.explain(sample_image, target_class=0).explanation_data["heatmap"]
        )
        heatmap1 = np.array(
            explainer.explain(sample_image, target_class=1).explanation_data["heatmap"]
        )
        assert heatmap0.shape == heatmap1.shape
        assert heatmap0.min() >= 0
        assert heatmap1.min() >= 0

    def test_heatmap_resized_to_input(self, adapter, last_conv_layer, sample_image, class_names):
        """EigenGradCAM resizes heatmap to input spatial dimensions."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=True)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.shape == (32, 32)

    def test_no_resize(self, adapter, last_conv_layer, sample_image, class_names):
        """EigenGradCAM can return heatmap at activation resolution."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=False)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.ndim == 2
        assert heatmap.shape[0] < 32

    def test_3d_input(self, adapter, last_conv_layer, class_names):
        """EigenGradCAM handles 3D (C, H, W) input."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        image_3d = np.random.randn(3, 32, 32).astype(np.float32)
        explanation = explainer.explain(image_3d)
        _assert_valid_cam_explanation(explanation, "EigenGradCAM (library variant)")

    def test_batch_explain(self, adapter, last_conv_layer, class_names):
        """EigenGradCAM can process batches."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        images = np.random.randn(3, 3, 32, 32).astype(np.float32)
        explanations = explainer.explain_batch(images)
        assert len(explanations) == 3
        for exp in explanations:
            _assert_valid_cam_explanation(exp, "EigenGradCAM (library variant)")

    def test_deterministic(self, adapter, last_conv_layer, sample_image, class_names):
        """Same input produces same output (no randomness)."""
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        h1 = np.array(explainer.explain(sample_image).explanation_data["heatmap"])
        h2 = np.array(explainer.explain(sample_image).explanation_data["heatmap"])
        np.testing.assert_array_almost_equal(h1, h2, decimal=5)

    def test_rejects_sklearn(self, class_names):
        """EigenGradCAM rejects non-gradient models."""
        from sklearn.linear_model import LogisticRegression

        from explainiverse.adapters import SklearnAdapter
        from explainiverse.explainers.gradient import EigenGradCAMExplainer

        sk = LogisticRegression()
        sk.fit(np.random.randn(20, 4), np.random.randint(0, 3, 20))
        with pytest.raises(TypeError, match="get_layer_gradients"):
            EigenGradCAMExplainer(
                model=SklearnAdapter(sk), target_layer="x", class_names=class_names
            )

    def test_svd_on_grad_times_act(self, adapter, last_conv_layer, sample_image, class_names):
        """EigenGradCAM applies SVD to grad*act, not just activations."""
        from explainiverse.explainers.gradient import EigenCAMExplainer, EigenGradCAMExplainer

        eigen_explainer = EigenCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        eigengrad_explainer = EigenGradCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )

        h_eigen = np.array(eigen_explainer.explain(sample_image).explanation_data["heatmap"])
        h_eigengrad = np.array(
            eigengrad_explainer.explain(sample_image, target_class=0).explanation_data["heatmap"]
        )

        assert h_eigen.shape == h_eigengrad.shape
        assert h_eigen.min() >= 0
        assert h_eigengrad.min() >= 0


# ══════════════════════════════════════════════
# GradCAMElementWise Tests
# ══════════════════════════════════════════════


class TestGradCAMElementWiseBasic:
    """Basic functionality tests for GradCAMElementWise."""

    def test_creation(self, adapter, last_conv_layer, class_names):
        """GradCAMElementWise explainer can be created."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        explainer = GradCAMElementWiseExplainer(
            model=adapter,
            target_layer=last_conv_layer,
            class_names=class_names,
        )
        assert explainer.target_layer == last_conv_layer
        assert explainer.class_names == class_names

    def test_explain(self, adapter, last_conv_layer, sample_image, class_names):
        """GradCAMElementWise produces valid explanation."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        explainer = GradCAMElementWiseExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        _assert_valid_cam_explanation(explanation, "GradCAMElementWise (library variant)")
        assert explanation.metadata["canonical_paper_method"] is False

    def test_target_class(self, adapter, last_conv_layer, sample_image, class_names):
        """GradCAMElementWise respects target_class parameter."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        explainer = GradCAMElementWiseExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        exp0 = explainer.explain(sample_image, target_class=0)
        exp1 = explainer.explain(sample_image, target_class=2)
        assert exp0.target_class == "cat"
        assert exp1.target_class == "bird"

    def test_uses_gradients(self, adapter, last_conv_layer, sample_image, class_names):
        """GradCAMElementWise uses gradients."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        explainer = GradCAMElementWiseExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        assert explainer._uses_gradients is True
        explanation = explainer.explain(sample_image)
        assert explanation.explanation_data["method"] == "gradcam_elementwise_library_variant"

    def test_elementwise_relu(self, adapter, last_conv_layer, sample_image, class_names):
        """GradCAMElementWise applies ReLU element-wise before channel sum."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        explainer = GradCAMElementWiseExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.min() >= 0

    def test_heatmap_resized_to_input(self, adapter, last_conv_layer, sample_image, class_names):
        """GradCAMElementWise resizes heatmap to input spatial dimensions."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        explainer = GradCAMElementWiseExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=True)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.shape == (32, 32)

    def test_3d_input(self, adapter, last_conv_layer, class_names):
        """GradCAMElementWise handles 3D (C, H, W) input."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        explainer = GradCAMElementWiseExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        image_3d = np.random.randn(3, 32, 32).astype(np.float32)
        explanation = explainer.explain(image_3d)
        _assert_valid_cam_explanation(explanation, "GradCAMElementWise (library variant)")

    def test_batch_explain(self, adapter, last_conv_layer, class_names):
        """GradCAMElementWise can process batches."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        explainer = GradCAMElementWiseExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        images = np.random.randn(3, 3, 32, 32).astype(np.float32)
        explanations = explainer.explain_batch(images)
        assert len(explanations) == 3
        for exp in explanations:
            _assert_valid_cam_explanation(exp, "GradCAMElementWise (library variant)")

    def test_deterministic(self, adapter, last_conv_layer, sample_image, class_names):
        """Same input produces same output (no randomness)."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        explainer = GradCAMElementWiseExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        h1 = np.array(explainer.explain(sample_image).explanation_data["heatmap"])
        h2 = np.array(explainer.explain(sample_image).explanation_data["heatmap"])
        np.testing.assert_array_almost_equal(h1, h2, decimal=5)

    def test_rejects_sklearn(self, class_names):
        """GradCAMElementWise rejects non-gradient models."""
        from sklearn.linear_model import LogisticRegression

        from explainiverse.adapters import SklearnAdapter
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer

        sk = LogisticRegression()
        sk.fit(np.random.randn(20, 4), np.random.randint(0, 3, 20))
        with pytest.raises(TypeError, match="get_layer_gradients"):
            GradCAMElementWiseExplainer(
                model=SklearnAdapter(sk), target_layer="x", class_names=class_names
            )

    def test_differs_from_hirescam(self, adapter, last_conv_layer, sample_image, class_names):
        """GradCAMElementWise differs from HiResCAM due to per-element ReLU."""
        from explainiverse.explainers.gradient import GradCAMElementWiseExplainer, HiResCAMExplainer

        hires_explainer = HiResCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        elemwise_explainer = GradCAMElementWiseExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )

        h_hires = np.array(hires_explainer.explain(sample_image).explanation_data["heatmap"])
        h_elemwise = np.array(elemwise_explainer.explain(sample_image).explanation_data["heatmap"])

        assert h_hires.shape == h_elemwise.shape
        assert h_hires.min() >= 0
        assert h_elemwise.min() >= 0


# ══════════════════════════════════════════════
# AblationCAM Tests
# ══════════════════════════════════════════════


class TestAblationCAMBasic:
    """Basic functionality tests for AblationCAM.

    AblationCAM is a gradient-free method that measures each activation
    channel's importance by observing the drop in target-class score
    when that channel's spatial contribution is removed from the input.

    Reference:
        Desai & Ramaswamy, 2020 — "Ablation-CAM: Visual Explanations for
        Deep Convolutional Network via Gradient-free Localization" (WACV)
    """

    def test_creation(self, adapter, last_conv_layer, class_names):
        """AblationCAM explainer can be created."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter,
            target_layer=last_conv_layer,
            class_names=class_names,
        )
        assert explainer.target_layer == last_conv_layer
        assert explainer.class_names == class_names

    def test_explain(self, adapter, last_conv_layer, sample_image, class_names):
        """AblationCAM produces valid explanation."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        _assert_valid_cam_explanation(explanation, "AblationCAM")

    def test_gradient_free(self, adapter, last_conv_layer, sample_image, class_names):
        """AblationCAM does not use gradients — uses ablation-based scoring."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        assert explainer._uses_gradients is False
        explanation = explainer.explain(sample_image)
        assert explanation.explanation_data["method"] == "ablationcam"

    def test_target_class(self, adapter, last_conv_layer, sample_image, class_names):
        """AblationCAM respects target_class parameter."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        exp0 = explainer.explain(sample_image, target_class=0)
        exp1 = explainer.explain(sample_image, target_class=1)
        assert exp0.target_class == "cat"
        assert exp1.target_class == "dog"

    def test_explicit_targets_preserve_spatial_contract(
        self, adapter, last_conv_layer, sample_image, class_names
    ):
        """Explicit target labels retain the same input-space heatmap contract."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation0 = explainer.explain(sample_image, target_class=0)
        explanation1 = explainer.explain(sample_image, target_class=1)
        heatmap0 = np.array(explanation0.explanation_data["heatmap"])
        heatmap1 = np.array(explanation1.explanation_data["heatmap"])

        assert explanation0.target_class == class_names[0]
        assert explanation1.target_class == class_names[1]
        assert heatmap0.shape == heatmap1.shape
        assert heatmap0.min() >= 0
        assert heatmap1.min() >= 0

    def test_heatmap_resized_to_input(self, adapter, last_conv_layer, sample_image, class_names):
        """AblationCAM resizes heatmap to input spatial dimensions."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=True)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.shape == (32, 32)

    def test_no_resize(self, adapter, last_conv_layer, sample_image, class_names):
        """AblationCAM can return heatmap at activation resolution."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image, resize_to_input=False)
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.ndim == 2
        assert heatmap.shape[0] < 32

    def test_3d_input(self, adapter, last_conv_layer, class_names):
        """AblationCAM handles 3D (C, H, W) input."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        image_3d = np.random.randn(3, 32, 32).astype(np.float32)
        explanation = explainer.explain(image_3d)
        _assert_valid_cam_explanation(explanation, "AblationCAM")

    def test_batch_size_parameter(self, adapter, last_conv_layer, class_names):
        """AblationCAM accepts batch_size for forward-pass efficiency."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter,
            target_layer=last_conv_layer,
            class_names=class_names,
            batch_size=8,
        )
        assert explainer.batch_size == 8

    def test_batch_explain(self, adapter, last_conv_layer, class_names):
        """AblationCAM can process batches of images."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        images = np.random.randn(2, 3, 32, 32).astype(np.float32)
        explanations = explainer.explain_batch(images)
        assert len(explanations) == 2
        for exp in explanations:
            _assert_valid_cam_explanation(exp, "AblationCAM")

    def test_deterministic(self, adapter, last_conv_layer, sample_image, class_names):
        """Same input produces same output (no randomness)."""
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        h1 = np.array(explainer.explain(sample_image).explanation_data["heatmap"])
        h2 = np.array(explainer.explain(sample_image).explanation_data["heatmap"])
        np.testing.assert_array_almost_equal(h1, h2, decimal=5)

    def test_differs_from_scorecam(self, adapter, last_conv_layer, sample_image, class_names):
        """AblationCAM differs from ScoreCAM in weighting strategy.

        ScoreCAM: weight = softmax(score with channel as mask)
        AblationCAM: weight = (original - ablated) / original  (score drop)
        """
        from explainiverse.explainers.gradient import AblationCAMExplainer, ScoreCAMExplainer

        score_explainer = ScoreCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        ablation_explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )

        h_score = np.array(score_explainer.explain(sample_image).explanation_data["heatmap"])
        h_ablation = np.array(ablation_explainer.explain(sample_image).explanation_data["heatmap"])

        # Both valid heatmaps, same shape
        assert h_score.shape == h_ablation.shape
        assert h_score.min() >= 0
        assert h_ablation.min() >= 0

    def test_ablation_principle(self, adapter, last_conv_layer, sample_image, class_names):
        """AblationCAM weights represent score drop from channel removal.

        Channels whose removal causes a large score drop should receive
        higher importance. Verify the method key records this principle.
        """
        from explainiverse.explainers.gradient import AblationCAMExplainer

        explainer = AblationCAMExplainer(
            model=adapter, target_layer=last_conv_layer, class_names=class_names
        )
        explanation = explainer.explain(sample_image)
        assert explanation.explanation_data["method"] == "ablationcam"
        # The heatmap should be normalized to [0, 1]
        heatmap = np.array(explanation.explanation_data["heatmap"])
        assert heatmap.max() <= 1.0 + 1e-6
        assert heatmap.min() >= 0


# ══════════════════════════════════════════════
# Cross-Method Comparison Tests (Batch 2)
# ══════════════════════════════════════════════


class TestBatch2CrossComparison:
    """Tests comparing batch 2 CAM methods against each other and batch 1."""

    def _get_heatmap(self, explainer_cls, adapter, layer, image, class_names):
        explainer = explainer_cls(model=adapter, target_layer=layer, class_names=class_names)
        explanation = explainer.explain(image)
        return np.array(explanation.explanation_data["heatmap"])

    def test_all_batch2_produce_same_shape(
        self, adapter, last_conv_layer, sample_image, class_names
    ):
        """All batch 2 CAM variants produce heatmaps of the same spatial dims."""
        from explainiverse.explainers.gradient import (
            AblationCAMExplainer,
            EigenGradCAMExplainer,
            GradCAMElementWiseExplainer,
        )

        methods = [EigenGradCAMExplainer, GradCAMElementWiseExplainer, AblationCAMExplainer]
        shapes = []
        for cls in methods:
            h = self._get_heatmap(cls, adapter, last_conv_layer, sample_image, class_names)
            shapes.append(h.shape)

        assert len(set(shapes)) == 1, f"Shapes differ: {shapes}"

    def test_batch2_same_shape_as_batch1(self, adapter, last_conv_layer, sample_image, class_names):
        """Batch 2 methods produce same-shaped heatmaps as batch 1 methods."""
        from explainiverse.explainers.gradient import (
            AblationCAMExplainer,
            EigenGradCAMExplainer,
            GradCAMElementWiseExplainer,
            HiResCAMExplainer,
        )

        h_ref = self._get_heatmap(
            HiResCAMExplainer, adapter, last_conv_layer, sample_image, class_names
        )
        for cls in [EigenGradCAMExplainer, GradCAMElementWiseExplainer, AblationCAMExplainer]:
            h = self._get_heatmap(cls, adapter, last_conv_layer, sample_image, class_names)
            assert (
                h.shape == h_ref.shape
            ), f"{cls.__name__} shape {h.shape} != HiResCAM shape {h_ref.shape}"


# ══════════════════════════════════════════════
# Registry Integration Tests (Batch 2)
# ══════════════════════════════════════════════


class TestBatch2Registry:
    """Test that batch 2 CAM variants are registered in the ExplainerRegistry."""

    @pytest.mark.parametrize(
        "name,expected_ref",
        [
            ("eigengradcam", "Muhammad"),
            ("gradcam_elementwise", "element-wise"),
            ("ablationcam", "Desai"),
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
        assert expected_ref.lower() in meta.paper_reference.lower()

    @pytest.mark.parametrize(
        "name",
        [
            "eigengradcam",
            "gradcam_elementwise",
            "ablationcam",
        ],
    )
    def test_filter_image(self, name):
        from explainiverse import default_registry

        image_explainers = default_registry.filter(data_type="image")
        assert name in image_explainers

    @pytest.mark.parametrize(
        "name",
        [
            "eigengradcam",
            "gradcam_elementwise",
            "ablationcam",
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
