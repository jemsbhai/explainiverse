# tests/test_bugfixes_v093.py
"""
Regression tests for v0.9.3 bug fixes.

Bug 1: LRP device mismatch — tensors created on CPU regardless of model device
Bug 2: LRP double reshape — Unflatten + Conv2d models get input reshaped twice
Bug 3: LRP MaxPool2d unpooling — max_unpool2d receives wrong output_size
Bug 4: GradCAM input shape validation — rejects flat input for Unflatten models
Bug 5: scikit-learn dependency conflict — pyproject.toml change, no unit test needed

These tests require PyTorch.
"""

import pytest
import numpy as np

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
def class_names():
    return ["class_a", "class_b", "class_c"]


@pytest.fixture
def cnn_with_unflatten():
    """
    CNN model where first layer is Unflatten — expects flat (1, 784) input
    and reshapes internally to (1, 1, 28, 28) for Conv2d.
    This is the architecture pattern that triggered Bugs 2 and 4.
    """
    model = nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),       # (batch, 784) -> (batch, 1, 28, 28)
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),                     # 28x28 -> 14x14
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),                        # (batch, 16, 14, 14) -> (batch, 3136)
        nn.Linear(16 * 14 * 14, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.eval()
    return model


@pytest.fixture
def simple_cnn_sequential():
    """Standard CNN (no Unflatten) — expects 4D input (batch, 1, 8, 8)."""
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),                     # 8x8 -> 4x4
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.eval()
    return model


@pytest.fixture
def simple_mlp():
    """Simple MLP for device placement tests."""
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    model.eval()
    return model


# =============================================================================
# Bug 1: LRP Device Mismatch
# =============================================================================

class TestBug1DeviceMismatch:
    """
    Bug 1: _prepare_input_tensor created tensors on CPU regardless of
    where the model parameters lived, causing RuntimeError on CUDA models.
    Fix: Added _get_model_device() and pass device= to torch.tensor().
    """

    def test_get_model_device_cpu(self, simple_mlp, class_names):
        """_get_model_device returns CPU for a CPU model."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(simple_mlp, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"f{i}" for i in range(4)],
            class_names=class_names
        )
        device = explainer._get_model_device()
        assert device == torch.device("cpu")

    def test_prepare_input_tensor_device_cpu(self, simple_mlp, class_names):
        """Prepared tensor lives on the same device as model (CPU)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(simple_mlp, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"f{i}" for i in range(4)],
            class_names=class_names
        )
        instance = np.random.randn(4).astype(np.float32)
        tensor = explainer._prepare_input_tensor(instance)
        assert tensor.device == torch.device("cpu")

    @pytest.mark.skipif(
        not TORCH_AVAILABLE or not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_get_model_device_cuda(self, simple_mlp, class_names):
        """_get_model_device returns CUDA for a CUDA model."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        cuda_model = simple_mlp.to("cuda")
        adapter = PyTorchAdapter(cuda_model, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"f{i}" for i in range(4)],
            class_names=class_names
        )
        device = explainer._get_model_device()
        assert device.type == "cuda"

    @pytest.mark.skipif(
        not TORCH_AVAILABLE or not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_prepare_input_tensor_device_cuda(self, simple_mlp, class_names):
        """Prepared tensor lives on the same device as model (CUDA)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        cuda_model = simple_mlp.to("cuda")
        adapter = PyTorchAdapter(cuda_model, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"f{i}" for i in range(4)],
            class_names=class_names
        )
        instance = np.random.randn(4).astype(np.float32)
        tensor = explainer._prepare_input_tensor(instance)
        assert tensor.device.type == "cuda"

    @pytest.mark.skipif(
        not TORCH_AVAILABLE or not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_lrp_explain_cuda_model(self, simple_mlp, class_names):
        """Full LRP explain() works on CUDA model without device mismatch."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        cuda_model = simple_mlp.to("cuda")
        adapter = PyTorchAdapter(cuda_model, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"f{i}" for i in range(4)],
            class_names=class_names
        )
        instance = np.random.randn(4).astype(np.float32)
        explanation = explainer.explain(instance)
        attrs = list(explanation.explanation_data["feature_attributions"].values())
        assert len(attrs) == 4
        assert all(np.isfinite(a) for a in attrs)

    @pytest.mark.skipif(
        not TORCH_AVAILABLE or not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_lrp_cnn_explain_cuda(self, simple_cnn_sequential, class_names):
        """Full LRP explain() works on CUDA CNN model."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        cuda_model = simple_cnn_sequential.to("cuda")
        adapter = PyTorchAdapter(cuda_model, task="classification", class_names=class_names)
        feature_names = [f"pixel_{i}" for i in range(64)]
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        instance = np.random.randn(1, 8, 8).astype(np.float32)
        explanation = explainer.explain(instance)
        attrs = list(explanation.explanation_data["feature_attributions"].values())
        assert len(attrs) == 64
        assert all(np.isfinite(a) for a in attrs)


# =============================================================================
# Bug 2: LRP Double Reshape (Unflatten + Conv2d)
# =============================================================================

class TestBug2DoubleReshape:
    """
    Bug 2: When a model has Unflatten before Conv2d, _prepare_input_tensor
    detected Conv2d as first weighted layer and pre-reshaped the input to 4D.
    Then Unflatten tried to reshape already-4D input, causing dimension errors.
    Fix: Added _has_unflatten_before_conv() to detect this and skip pre-reshape.
    """

    def test_has_unflatten_before_conv_true(self, cnn_with_unflatten, class_names):
        """Correctly detects Unflatten before Conv2d."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(cnn_with_unflatten, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"pixel_{i}" for i in range(784)],
            class_names=class_names
        )
        assert explainer._has_unflatten_before_conv() is True

    def test_has_unflatten_before_conv_false(self, simple_cnn_sequential, class_names):
        """Returns False for standard CNN without Unflatten."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(simple_cnn_sequential, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"pixel_{i}" for i in range(64)],
            class_names=class_names
        )
        assert explainer._has_unflatten_before_conv() is False

    def test_has_unflatten_before_conv_false_mlp(self, simple_mlp, class_names):
        """Returns False for MLP (no Conv2d at all)."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(simple_mlp, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"f{i}" for i in range(4)],
            class_names=class_names
        )
        assert explainer._has_unflatten_before_conv() is False

    def test_prepare_input_flat_for_unflatten_model(self, cnn_with_unflatten, class_names):
        """Flat input stays 2D (batch, features) when model has Unflatten."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(cnn_with_unflatten, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"pixel_{i}" for i in range(784)],
            class_names=class_names
        )
        instance = np.random.randn(784).astype(np.float32)
        tensor = explainer._prepare_input_tensor(instance)
        # Should be 2D (1, 784), NOT 4D (1, 1, 28, 28)
        assert tensor.shape == (1, 784)

    def test_prepare_input_4d_for_standard_cnn(self, simple_cnn_sequential, class_names):
        """Flat input gets reshaped to 4D for standard CNN without Unflatten."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(simple_cnn_sequential, task="classification", class_names=class_names)
        explainer = LRPExplainer(
            model=adapter,
            feature_names=[f"pixel_{i}" for i in range(64)],
            class_names=class_names
        )
        instance = np.random.randn(64).astype(np.float32)
        tensor = explainer._prepare_input_tensor(instance)
        # Should be 4D (1, 1, 8, 8) for CNN
        assert tensor.dim() == 4
        assert tensor.shape == (1, 1, 8, 8)

    def test_lrp_explain_unflatten_cnn(self, cnn_with_unflatten, class_names):
        """
        Full LRP explain() works on Unflatten+Conv2d model without double reshape.
        This was the primary failure mode of Bug 2.
        """
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(cnn_with_unflatten, task="classification", class_names=class_names)
        feature_names = [f"pixel_{i}" for i in range(784)]
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        instance = np.random.randn(784).astype(np.float32)
        explanation = explainer.explain(instance)
        attrs = list(explanation.explanation_data["feature_attributions"].values())
        assert len(attrs) == 784
        assert all(np.isfinite(a) for a in attrs)

    def test_lrp_batch_unflatten_cnn(self, cnn_with_unflatten, class_names):
        """Batch processing works on Unflatten+Conv2d model."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(cnn_with_unflatten, task="classification", class_names=class_names)
        feature_names = [f"pixel_{i}" for i in range(784)]
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        batch = np.random.randn(3, 784).astype(np.float32)
        explanations = explainer.explain_batch(batch)
        assert len(explanations) == 3
        for exp in explanations:
            attrs = list(exp.explanation_data["feature_attributions"].values())
            assert len(attrs) == 784
            assert all(np.isfinite(a) for a in attrs)


# =============================================================================
# Bug 3: LRP MaxPool2d Unpooling
# =============================================================================

class TestBug3MaxPoolUnpooling:
    """
    Bug 3: F.max_unpool2d could receive mismatched shapes when relevance
    tensor was not 4D or didn't match the pooled indices shape.
    Fix: Added 4D assertion, relevance shape matching before unpooling.
    """

    def test_maxpool2d_propagation_standard_cnn(self, simple_cnn_sequential, class_names):
        """LRP with MaxPool2d produces finite attributions on standard CNN."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(simple_cnn_sequential, task="classification", class_names=class_names)
        feature_names = [f"pixel_{i}" for i in range(64)]
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        instance = np.random.randn(1, 8, 8).astype(np.float32)
        explanation = explainer.explain(instance)
        attrs = list(explanation.explanation_data["feature_attributions"].values())
        assert len(attrs) == 64
        assert all(np.isfinite(a) for a in attrs)

    def test_maxpool2d_propagation_unflatten_cnn(self, cnn_with_unflatten, class_names):
        """LRP with MaxPool2d works on Unflatten+Conv2d model."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(cnn_with_unflatten, task="classification", class_names=class_names)
        feature_names = [f"pixel_{i}" for i in range(784)]
        explainer = LRPExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names
        )
        instance = np.random.randn(784).astype(np.float32)
        explanation = explainer.explain(instance)
        attrs = list(explanation.explanation_data["feature_attributions"].values())
        assert len(attrs) == 784
        assert all(np.isfinite(a) for a in attrs)

    def test_maxpool2d_all_lrp_rules(self, simple_cnn_sequential, class_names):
        """All LRP rules produce finite results through MaxPool2d."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import LRPExplainer

        adapter = PyTorchAdapter(simple_cnn_sequential, task="classification", class_names=class_names)
        feature_names = [f"pixel_{i}" for i in range(64)]
        instance = np.random.randn(1, 8, 8).astype(np.float32)

        for rule in ["epsilon", "gamma", "z_plus"]:
            explainer = LRPExplainer(
                model=adapter,
                feature_names=feature_names,
                class_names=class_names,
                rule=rule
            )
            explanation = explainer.explain(instance)
            attrs = list(explanation.explanation_data["feature_attributions"].values())
            assert all(np.isfinite(a) for a in attrs), (
                f"Rule '{rule}' produced non-finite attributions through MaxPool2d"
            )


# =============================================================================
# Bug 4: GradCAM Input Shape Validation
# =============================================================================

class TestBug4GradCAMFlatInput:
    """
    Bug 4: GradCAM.explain() rejected flat (1D/2D) input with ValueError,
    but models with Unflatten layers expect flat input.
    Fix: Added _model_has_unflatten() check and flat input handling path.
    """

    @pytest.fixture
    def cnn_with_unflatten_named(self):
        """
        CNN with Unflatten and named conv layers (needed for GradCAM target_layer).
        GradCAM requires a non-Sequential model with named attributes for layer access.
        """
        class UnflattenCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.unflatten = nn.Unflatten(1, (1, 28, 28))
                self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(2)
                self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
                self.relu2 = nn.ReLU()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(16 * 14 * 14, 64)
                self.relu3 = nn.ReLU()
                self.fc2 = nn.Linear(64, 3)

            def forward(self, x):
                x = self.unflatten(x)
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.relu2(self.conv2(x))
                x = self.flatten(x)
                x = self.relu3(self.fc1(x))
                x = self.fc2(x)
                return x

        model = UnflattenCNN()
        torch.manual_seed(42)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.eval()
        return model

    def test_model_has_unflatten_true(self, cnn_with_unflatten_named, class_names):
        """_model_has_unflatten detects Unflatten in model."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import GradCAMExplainer

        adapter = PyTorchAdapter(
            cnn_with_unflatten_named, task="classification", class_names=class_names
        )
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer="conv2",
            class_names=class_names
        )
        assert explainer._model_has_unflatten() is True

    def test_model_has_unflatten_false(self, class_names):
        """_model_has_unflatten returns False for standard CNN."""
        class StandardCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(16 * 8 * 8, 3)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.flatten(x)
                x = self.fc(x)
                return x

        model = StandardCNN()
        model.eval()

        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import GradCAMExplainer

        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer="conv1",
            class_names=class_names
        )
        assert explainer._model_has_unflatten() is False

    def test_gradcam_flat_input_accepted(self, cnn_with_unflatten_named, class_names):
        """
        GradCAM.explain() accepts flat (1D) input when model has Unflatten.
        This was the primary failure mode of Bug 4 — ValueError was raised.
        """
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import GradCAMExplainer

        adapter = PyTorchAdapter(
            cnn_with_unflatten_named, task="classification", class_names=class_names
        )
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer="conv2",
            class_names=class_names
        )
        # Flat input — previously would raise ValueError
        flat_input = np.random.randn(784).astype(np.float32)
        explanation = explainer.explain(flat_input)
        assert "heatmap" in explanation.explanation_data

    def test_gradcam_flat_input_2d_accepted(self, cnn_with_unflatten_named, class_names):
        """GradCAM.explain() accepts 2D (batch, features) input for Unflatten model."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import GradCAMExplainer

        adapter = PyTorchAdapter(
            cnn_with_unflatten_named, task="classification", class_names=class_names
        )
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer="conv2",
            class_names=class_names
        )
        batched_flat_input = np.random.randn(1, 784).astype(np.float32)
        explanation = explainer.explain(batched_flat_input)
        assert "heatmap" in explanation.explanation_data

    def test_gradcam_standard_cnn_still_rejects_flat(self, class_names):
        """Standard CNN (no Unflatten) still rejects flat input with clear error."""
        class StandardCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(16 * 8 * 8, 3)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.flatten(x)
                x = self.fc(x)
                return x

        model = StandardCNN()
        model.eval()

        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import GradCAMExplainer

        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer="conv1",
            class_names=class_names
        )
        flat_input = np.random.randn(192).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            explainer.explain(flat_input)

    def test_gradcam_standard_cnn_4d_still_works(self, class_names):
        """Standard CNN still works with proper 4D input (no regression)."""
        class StandardCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(16, 3)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.flatten(x)
                x = self.fc(x)
                return x

        model = StandardCNN()
        torch.manual_seed(42)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.eval()

        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import GradCAMExplainer

        adapter = PyTorchAdapter(model, task="classification", class_names=class_names)
        explainer = GradCAMExplainer(
            model=adapter,
            target_layer="conv1",
            class_names=class_names
        )
        image = np.random.randn(1, 3, 8, 8).astype(np.float32)
        explanation = explainer.explain(image)
        assert "heatmap" in explanation.explanation_data


# =============================================================================
# Bug 5: scikit-learn Dependency (pyproject.toml only — integration test)
# =============================================================================

class TestBug5ScikitLearnDependency:
    """
    Bug 5: pyproject.toml allowed scikit-learn >=1.2 which is incompatible
    with umap-learn (requires >=1.6). Fixed by raising lower bound.
    This is a packaging test — verify we can import sklearn and it's >=1.6.
    """

    def test_sklearn_version_meets_minimum(self):
        """Installed scikit-learn meets the >=1.6 minimum."""
        import sklearn
        from packaging.version import Version
        assert Version(sklearn.__version__) >= Version("1.6"), (
            f"scikit-learn {sklearn.__version__} < 1.6 — pyproject.toml fix not applied"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
