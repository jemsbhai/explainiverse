# tests/test_integrated_gradients.py
"""
Tests for Integrated Gradients explainer.

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


class TestIntegratedGradientsBasic:
    """Basic functionality tests for Integrated Gradients."""

    def test_ig_creation(self, simple_classifier, feature_names, class_names):
        """Integrated Gradients explainer can be created."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")

        explainer = IntegratedGradientsExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_steps=50
        )

        assert explainer.n_steps == 50
        assert explainer.feature_names == feature_names
        assert explainer.class_names == class_names

    def test_ig_rejects_non_gradient_model(self, feature_names):
        """IG raises error for models without gradient support."""
        from sklearn.linear_model import LogisticRegression

        from explainiverse.adapters import SklearnAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        # SklearnAdapter doesn't have predict_with_gradients
        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)

        with pytest.raises(TypeError, match="predict_with_gradients"):
            IntegratedGradientsExplainer(model=adapter, feature_names=feature_names)

    def test_ig_explain_classification(
        self, simple_classifier, sample_data, feature_names, class_names
    ):
        """IG produces valid explanations for classification."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.core.explanation import Explanation
        from explainiverse.explainers import IntegratedGradientsExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = IntegratedGradientsExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_steps=20
        )

        explanation = explainer.explain(sample_data[0])

        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "IntegratedGradients"
        assert "feature_attributions" in explanation.explanation_data
        assert len(explanation.explanation_data["feature_attributions"]) == 4

    def test_ig_explain_regression(self, simple_regressor, sample_data, feature_names):
        """IG produces valid explanations for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.core.explanation import Explanation
        from explainiverse.explainers import IntegratedGradientsExplainer

        adapter = PyTorchAdapter(simple_regressor, task="regression")

        explainer = IntegratedGradientsExplainer(
            model=adapter, feature_names=feature_names, n_steps=20
        )

        explanation = explainer.explain(sample_data[0])

        assert isinstance(explanation, Explanation)
        assert "feature_attributions" in explanation.explanation_data

    def test_ig_target_class(self, feature_names, class_names):
        """Each linear output equals (input - baseline) times its weights."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

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

        explainer = IntegratedGradientsExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_steps=7
        )
        instance = np.array([0.3, -0.8, 1.2, 0.5], dtype=np.float32)

        for target_class, weights_for_output in enumerate(weights[:2]):
            explanation = explainer.explain(instance, target_class=target_class)
            expected = instance * weights_for_output
            np.testing.assert_allclose(
                explanation.explanation_data["attributions_raw"], expected, atol=1e-7
            )
            assert explanation.target_class == class_names[target_class]


class TestIntegratedGradientsBaselines:
    """Tests for different baseline options."""

    def test_ig_zero_baseline(self, simple_classifier, sample_data, feature_names, class_names):
        """Default zero baseline works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            baseline=None,  # Default: zeros
        )

        explanation = explainer.explain(sample_data[0])

        # Check that baseline is zeros
        baseline = explanation.explanation_data["baseline"]
        assert all(b == 0 for b in baseline)

    def test_ig_custom_baseline(self, simple_classifier, sample_data, feature_names, class_names):
        """Custom baseline works correctly."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        custom_baseline = np.ones(4, dtype=np.float32) * 0.5

        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            baseline=custom_baseline,
        )

        explanation = explainer.explain(sample_data[0])

        # Check that baseline is our custom one
        baseline = explanation.explanation_data["baseline"]
        assert np.allclose(baseline, custom_baseline)

    def test_ig_override_baseline(self, simple_classifier, sample_data, feature_names, class_names):
        """Baseline can be overridden per-explanation."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = IntegratedGradientsExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            baseline=None,  # Default: zeros
        )

        override_baseline = np.ones(4, dtype=np.float32) * -0.5

        explanation = explainer.explain(sample_data[0], baseline=override_baseline)

        # Check that override baseline was used
        baseline = explanation.explanation_data["baseline"]
        assert np.allclose(baseline, override_baseline)


class TestIntegratedGradientsBatch:
    """Tests for batch processing."""

    def test_ig_batch_explain(self, simple_classifier, sample_data, feature_names, class_names):
        """Batch explain processes multiple instances."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = IntegratedGradientsExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_steps=20
        )

        explanations = explainer.explain_batch(sample_data[:5])

        assert len(explanations) == 5
        for exp in explanations:
            assert "feature_attributions" in exp.explanation_data


class TestIntegratedGradientsSmoothGrad:
    """Tests for SmoothGrad-style noisy averaging."""

    def test_ig_smooth(self, simple_classifier, sample_data, feature_names, class_names):
        """Smooth IG with noisy baselines works."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = IntegratedGradientsExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names, n_steps=20
        )

        explanation = explainer.compute_attributions_with_noise(
            sample_data[0], target_class=0, n_samples=5, noise_scale=0.1
        )

        assert explanation.explainer_name == "IntegratedGradients_Smooth"
        assert "feature_attributions" in explanation.explanation_data
        assert "attributions_std" in explanation.explanation_data


class TestIntegratedGradientsContracts:
    """Strict shape, numeric-domain, and model-state regression contracts."""

    def test_input_shape_is_inferred_then_enforced(self, simple_classifier, feature_names):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        explainer = IntegratedGradientsExplainer(
            PyTorchAdapter(simple_classifier, task="classification"),
            feature_names=feature_names,
            n_steps=2,
        )
        explainer.explain(np.ones(4, dtype=np.float32), target_class=0)

        assert explainer.input_shape == (4,)
        with pytest.raises(ValueError, match="input_shape exactly"):
            explainer.explain(np.ones((2, 2), dtype=np.float32), target_class=0)

    @pytest.mark.parametrize(
        "input_shape",
        [(), (0,), (-1, 2), (True, 2), (2.5,), [4]],
    )
    def test_invalid_explicit_input_shape_is_rejected(self, input_shape):
        from explainiverse.explainers import IntegratedGradientsExplainer

        class NeverCalledAdapter:
            def predict_with_gradients(self, inputs, target_class=None):
                raise AssertionError("model must not be called")

        with pytest.raises(ValueError, match="input_shape"):
            IntegratedGradientsExplainer(NeverCalledAdapter(), input_shape=input_shape)

    def test_explicit_input_shape_is_enforced_before_model_call(self):
        from explainiverse.explainers import IntegratedGradientsExplainer

        class NeverCalledAdapter:
            task = "regression"

            def predict_with_gradients(self, inputs, target_class=None):
                raise AssertionError("model must not be called")

        explainer = IntegratedGradientsExplainer(
            NeverCalledAdapter(), n_steps=2, input_shape=(2, 2)
        )
        with pytest.raises(ValueError, match=r"expected \(2, 2\), got \(4,\)"):
            explainer.explain(np.ones(4, dtype=np.float32))

    @pytest.mark.parametrize(
        "instance",
        [
            np.array([np.nan], dtype=np.float32),
            np.array([np.inf], dtype=np.float32),
            np.array([1.0 + 2.0j]),
            np.array([], dtype=np.float32),
        ],
    )
    def test_nonfinite_complex_and_empty_instances_are_rejected(self, instance):
        from explainiverse.explainers import IntegratedGradientsExplainer

        class NeverCalledAdapter:
            task = "regression"

            def predict_with_gradients(self, inputs, target_class=None):
                raise AssertionError("model must not be called")

        explainer = IntegratedGradientsExplainer(NeverCalledAdapter(), n_steps=2)
        with pytest.raises(ValueError, match="non-empty|finite real"):
            explainer.explain(instance)

    @pytest.mark.parametrize(
        "baseline",
        [
            np.ones((2, 2), dtype=np.float32),
            np.array([0.0, 0.0, np.nan, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0j, 0.0]),
        ],
    )
    def test_baseline_must_have_exact_shape_and_finite_real_values(
        self, simple_classifier, feature_names, baseline
    ):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        explainer = IntegratedGradientsExplainer(
            PyTorchAdapter(simple_classifier, task="classification"),
            feature_names=feature_names,
            n_steps=2,
        )
        with pytest.raises(ValueError, match="shape must match|finite real"):
            explainer.explain(np.ones(4, dtype=np.float32), target_class=0, baseline=baseline)

    @pytest.mark.parametrize("failure", ["shape", "nonfinite"])
    def test_adapter_gradient_must_match_input_exactly_and_be_finite(self, failure):
        from explainiverse.explainers import IntegratedGradientsExplainer

        class BadGradientAdapter:
            task = "regression"

            def predict_with_gradients(self, inputs, target_class=None):
                if failure == "shape":
                    gradient = np.zeros((1, 1), dtype=np.float32)
                else:
                    gradient = np.array([[np.nan, 0.0]], dtype=np.float32)
                return np.zeros((1, 1), dtype=np.float32), gradient

        explainer = IntegratedGradientsExplainer(
            BadGradientAdapter(), feature_names=["a", "b"], n_steps=2
        )
        expected = ValueError if failure == "shape" else FloatingPointError
        with pytest.raises(expected, match="gradient shape|finite real"):
            explainer.explain(np.ones(2, dtype=np.float32))

    def test_eval_mode_is_temporary_and_buffers_and_gradients_are_restored(self):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        class StatefulNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_norm = nn.BatchNorm1d(2)
                self.dropout = nn.Dropout(p=0.95)
                self.output = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.output.weight.copy_(torch.eye(2))

            def forward(self, inputs):
                return self.output(self.dropout(self.batch_norm(inputs)))

        network = StatefulNet()
        adapter = PyTorchAdapter(network, task="classification")
        network.train()
        network.dropout.eval()
        network.output.weight.grad = torch.full_like(network.output.weight, 7.0)
        flags_before = [module.training for module in network.modules()]
        mean_before = network.batch_norm.running_mean.detach().clone()
        gradient_object = network.output.weight.grad

        explainer = IntegratedGradientsExplainer(adapter, feature_names=["a", "b"], n_steps=3)
        first = explainer.explain(np.array([1.0, -1.0], dtype=np.float32), target_class=0)
        second = explainer.explain(np.array([1.0, -1.0], dtype=np.float32), target_class=0)

        np.testing.assert_allclose(
            first.explanation_data["attributions_raw"],
            second.explanation_data["attributions_raw"],
        )
        assert [module.training for module in network.modules()] == flags_before
        torch.testing.assert_close(network.batch_norm.running_mean, mean_before)
        assert network.output.weight.grad is gradient_object
        torch.testing.assert_close(
            network.output.weight.grad,
            torch.full_like(network.output.weight, 7.0),
        )

    @pytest.mark.parametrize("operation", ["random_baseline", "noisy_baselines"])
    def test_random_operations_use_reproducible_local_rng_without_global_leak(
        self, simple_classifier, feature_names, operation
    ):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers import IntegratedGradientsExplainer

        explainer = IntegratedGradientsExplainer(
            PyTorchAdapter(simple_classifier, task="classification"),
            feature_names=feature_names,
            n_steps=2,
            baseline="random" if operation == "random_baseline" else None,
            random_state=23,
        )
        instance = np.array([-2.0, 0.5, 1.0, 4.0], dtype=np.float32)

        np.random.seed(913)
        expected_global = np.random.RandomState(913).random_sample()
        if operation == "random_baseline":
            first = explainer.explain(instance, target_class=0)
            second = explainer.explain(instance, target_class=0)
            np.testing.assert_allclose(
                first.explanation_data["baseline"],
                second.explanation_data["baseline"],
            )
        else:
            first = explainer.compute_attributions_with_noise(instance, target_class=0, n_samples=3)
            second = explainer.compute_attributions_with_noise(
                instance, target_class=0, n_samples=3
            )
            np.testing.assert_allclose(
                first.explanation_data["attributions_raw"],
                second.explanation_data["attributions_raw"],
            )

        assert np.random.random() == expected_global
        assert first.explanation_data["random_state"] == 23

    @pytest.mark.parametrize(
        ("random_state", "expected"),
        [(True, TypeError), (1.5, TypeError), (-1, ValueError)],
    )
    def test_invalid_random_state_is_rejected(self, random_state, expected):
        from explainiverse.explainers import IntegratedGradientsExplainer

        class GradientAdapter:
            def predict_with_gradients(self, X, target_class=None):
                raise AssertionError("model must not be called")

        with pytest.raises(expected, match="random_state"):
            IntegratedGradientsExplainer(
                model=GradientAdapter(),
                random_state=random_state,
            )


class TestIntegratedGradientsRegistry:
    """Tests for registry integration."""

    def test_ig_registered(self):
        """Integrated Gradients is registered in default registry."""
        from explainiverse import default_registry

        explainers = default_registry.list_explainers()
        assert "integrated_gradients" in explainers

    def test_ig_metadata(self):
        """Integrated Gradients has correct metadata."""
        from explainiverse import default_registry

        meta = default_registry.get_meta("integrated_gradients")

        assert meta.scope == "local"
        assert "neural" in meta.model_types
        assert "tabular" in meta.data_types
        assert "image" in meta.data_types
        assert "Sundararajan" in meta.paper_reference

    def test_ig_filter_neural(self):
        """IG appears when filtering for neural network explainers."""
        from explainiverse import default_registry

        neural_explainers = default_registry.filter(model_type="neural")
        assert "integrated_gradients" in neural_explainers

    def test_ig_via_registry(self, simple_classifier, feature_names, class_names):
        """IG can be created via registry."""
        from explainiverse import default_registry
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)

        explainer = default_registry.create(
            "integrated_gradients",
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
        )

        assert explainer is not None
        assert explainer.feature_names == feature_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
