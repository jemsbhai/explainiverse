# tests/test_deeplift.py
"""
Tests for DeepLIFT and DeepSHAP explainers.

These tests require PyTorch to be installed. They will be skipped
if torch is not available.

Reference:
    Shrikumar et al., 2017 — "Learning Important Features Through
    Propagating Activation Differences." ICML 2017.
"""

import subprocess
import sys
from types import MethodType

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
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def simple_classifier():
    """Create a simple PyTorch classifier for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")

    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))

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
    return np.random.randn(10, 4).astype(np.float32)


@pytest.fixture
def feature_names():
    return ["feature_0", "feature_1", "feature_2", "feature_3"]


@pytest.fixture
def class_names():
    return ["class_a", "class_b", "class_c"]


@pytest.fixture
def deeplift_explainer(simple_classifier, feature_names, class_names):
    """Pre-built DeepLIFT explainer for convenience."""
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import DeepLIFTExplainer

    adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
    return DeepLIFTExplainer(model=adapter, feature_names=feature_names, class_names=class_names)


@pytest.fixture
def deepshap_explainer(simple_classifier, sample_data, feature_names, class_names):
    """Pre-built DeepSHAP explainer for convenience."""
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import DeepLIFTShapExplainer

    adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
    return DeepLIFTShapExplainer(
        model=adapter,
        feature_names=feature_names,
        class_names=class_names,
        background_data=sample_data[:5],
    )


# ──────────────────────────────────────────────
# DeepLIFT Basic Tests
# ──────────────────────────────────────────────


class TestDeepLIFTBasic:
    """Basic functionality tests for DeepLIFT."""

    def test_creation(self, simple_classifier, feature_names, class_names):
        """DeepLIFT explainer can be created."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")
        explainer = DeepLIFTExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names
        )

        assert explainer.feature_names == feature_names
        assert explainer.class_names == class_names
        assert explainer.multiply_by_inputs is True

    def test_rejects_non_gradient_model(self, feature_names):
        """DeepLIFT raises error for models without gradient support."""
        from sklearn.linear_model import LogisticRegression

        from explainiverse.adapters import SklearnAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)

        with pytest.raises(TypeError, match="PyTorchAdapter"):
            DeepLIFTExplainer(model=adapter, feature_names=feature_names)

    def test_explain_classification(self, deeplift_explainer, sample_data):
        """DeepLIFT produces valid explanations for classification."""
        from explainiverse.core.explanation import Explanation

        explanation = deeplift_explainer.explain(sample_data[0])

        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "DeepLIFT"
        assert "feature_attributions" in explanation.explanation_data
        assert len(explanation.explanation_data["feature_attributions"]) == 4

    def test_explain_regression(self, simple_regressor, sample_data, feature_names):
        """DeepLIFT produces valid explanations for regression."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.core.explanation import Explanation
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        adapter = PyTorchAdapter(simple_regressor, task="regression")
        explainer = DeepLIFTExplainer(model=adapter, feature_names=feature_names)

        explanation = explainer.explain(sample_data[0])
        assert isinstance(explanation, Explanation)
        assert "feature_attributions" in explanation.explanation_data

    def test_target_class(self, deeplift_explainer, sample_data):
        """DeepLIFT respects target_class parameter."""
        exp_0 = deeplift_explainer.explain(sample_data[0], target_class=0)
        exp_1 = deeplift_explainer.explain(sample_data[0], target_class=1)

        attr_0 = list(exp_0.explanation_data["feature_attributions"].values())
        attr_1 = list(exp_1.explanation_data["feature_attributions"].values())

        assert not np.allclose(attr_0, attr_1)
        assert exp_0.target_class == "class_a"
        assert exp_1.target_class == "class_b"


# ──────────────────────────────────────────────
# Critical: Key Matching & feature_names Tests
# ──────────────────────────────────────────────


class TestDeepLIFTKeyMatching:
    """Critical tests: attribution keys match feature names,
    and feature_names is stored on Explanation objects."""

    def test_attribution_keys_match_feature_names(
        self, deeplift_explainer, sample_data, feature_names
    ):
        """Attribution dict keys must be original feature names."""
        for i in range(min(5, len(sample_data))):
            explanation = deeplift_explainer.explain(sample_data[i])
            keys = set(explanation.explanation_data["feature_attributions"].keys())
            assert keys == set(
                feature_names
            ), f"Instance {i}: keys {keys} != feature names {set(feature_names)}"

    def test_attribution_count(self, deeplift_explainer, sample_data, feature_names):
        """Number of attributions equals number of features."""
        explanation = deeplift_explainer.explain(sample_data[0])
        assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)

    def test_attribution_values_are_float(self, deeplift_explainer, sample_data):
        """All attribution values are floats."""
        explanation = deeplift_explainer.explain(sample_data[0])
        for k, v in explanation.explanation_data["feature_attributions"].items():
            assert isinstance(v, float), f"'{k}' has type {type(v)}"

    def test_feature_names_stored_on_explanation(
        self, deeplift_explainer, sample_data, feature_names
    ):
        """Explanation must have feature_names attribute for evaluation metrics."""
        explanation = deeplift_explainer.explain(sample_data[0])
        assert hasattr(
            explanation, "feature_names"
        ), "Explanation missing feature_names — evaluation metrics will fail"
        assert explanation.feature_names == feature_names

    def test_attributions_raw_length(self, deeplift_explainer, sample_data, feature_names):
        """attributions_raw has exactly n_features entries."""
        explanation = deeplift_explainer.explain(sample_data[0])
        raw = explanation.explanation_data["attributions_raw"]
        assert len(raw) == len(
            feature_names
        ), f"attributions_raw has {len(raw)} entries, expected {len(feature_names)}"

    def test_attributions_raw_matches_dict(self, deeplift_explainer, sample_data, feature_names):
        """attributions_raw values must match feature_attributions values."""
        explanation = deeplift_explainer.explain(sample_data[0])
        raw = explanation.explanation_data["attributions_raw"]
        attributions = explanation.explanation_data["feature_attributions"]

        for i, fname in enumerate(feature_names):
            assert (
                abs(raw[i] - attributions[fname]) < 1e-10
            ), f"raw[{i}]={raw[i]} != attributions['{fname}']={attributions[fname]}"

    def test_values_are_finite(self, deeplift_explainer, sample_data):
        """All attribution values must be finite."""
        for i in range(min(5, len(sample_data))):
            explanation = deeplift_explainer.explain(sample_data[i])
            for fname, val in explanation.explanation_data["feature_attributions"].items():
                assert np.isfinite(val), f"Non-finite value for '{fname}': {val}"

    def test_deterministic(self, deeplift_explainer, sample_data):
        """Same input produces same output."""
        v1 = list(
            deeplift_explainer.explain(sample_data[0])
            .explanation_data["feature_attributions"]
            .values()
        )
        v2 = list(
            deeplift_explainer.explain(sample_data[0])
            .explanation_data["feature_attributions"]
            .values()
        )
        np.testing.assert_array_almost_equal(v1, v2, decimal=8)


# ──────────────────────────────────────────────
# DeepLIFT Multi-Baseline Key Matching
# ──────────────────────────────────────────────


class TestDeepLIFTMultiBaselineKeyMatching:
    """Key matching and feature_names for multi-baseline explanations."""

    def test_multi_baseline_keys_match(self, deeplift_explainer, sample_data, feature_names):
        """Multi-baseline attributions keyed by original feature names."""
        explanation = deeplift_explainer.explain_with_multiple_baselines(
            sample_data[5], baselines=sample_data[:5], target_class=0
        )
        keys = set(explanation.explanation_data["feature_attributions"].keys())
        assert keys == set(feature_names)

    def test_multi_baseline_feature_names_stored(
        self, deeplift_explainer, sample_data, feature_names
    ):
        """Multi-baseline Explanation has feature_names attribute."""
        explanation = deeplift_explainer.explain_with_multiple_baselines(
            sample_data[5], baselines=sample_data[:5], target_class=0
        )
        assert hasattr(
            explanation, "feature_names"
        ), "Multi-baseline Explanation missing feature_names"
        assert explanation.feature_names == feature_names

    def test_multi_baseline_raw_length(self, deeplift_explainer, sample_data, feature_names):
        """Multi-baseline attributions_raw has n_features entries."""
        explanation = deeplift_explainer.explain_with_multiple_baselines(
            sample_data[5], baselines=sample_data[:5], target_class=0
        )
        raw = explanation.explanation_data["attributions_raw"]
        assert len(raw) == len(feature_names)


# ──────────────────────────────────────────────
# DeepLIFT Batch Key Matching
# ──────────────────────────────────────────────


class TestDeepLIFTBatchKeyMatching:
    """Key matching and feature_names for batch explanations."""

    def test_batch_keys_match(self, deeplift_explainer, sample_data, feature_names):
        """Each batch explanation has correct attribution keys."""
        explanations = deeplift_explainer.explain_batch(sample_data[:5])
        assert len(explanations) == 5
        for i, exp in enumerate(explanations):
            keys = set(exp.explanation_data["feature_attributions"].keys())
            assert keys == set(feature_names), f"Batch instance {i}: keys {keys} != feature names"

    def test_batch_feature_names_stored(self, deeplift_explainer, sample_data, feature_names):
        """Each batch explanation has feature_names attribute."""
        explanations = deeplift_explainer.explain_batch(sample_data[:3])
        for i, exp in enumerate(explanations):
            assert hasattr(exp, "feature_names"), f"Batch instance {i}: missing feature_names"
            assert exp.feature_names == feature_names

    def test_batch_raw_length(self, deeplift_explainer, sample_data, feature_names):
        """Each batch explanation has correct attributions_raw length."""
        explanations = deeplift_explainer.explain_batch(sample_data[:3])
        for i, exp in enumerate(explanations):
            raw = exp.explanation_data["attributions_raw"]
            assert len(raw) == len(
                feature_names
            ), f"Batch {i}: raw length {len(raw)} != {len(feature_names)}"


# ──────────────────────────────────────────────
# DeepSHAP Key Matching & feature_names Tests
# ──────────────────────────────────────────────


class TestDeepSHAPKeyMatching:
    """Critical tests for DeepSHAP attribution correctness."""

    def test_attribution_keys_match_feature_names(
        self, deepshap_explainer, sample_data, feature_names
    ):
        """DeepSHAP attribution keys must be original feature names."""
        explanation = deepshap_explainer.explain(sample_data[5])
        keys = set(explanation.explanation_data["feature_attributions"].keys())
        assert keys == set(feature_names), f"Keys {keys} != feature names {set(feature_names)}"

    def test_attribution_count(self, deepshap_explainer, sample_data, feature_names):
        """DeepSHAP returns correct number of attributions."""
        explanation = deepshap_explainer.explain(sample_data[5])
        assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)

    def test_feature_names_stored_on_explanation(
        self, deepshap_explainer, sample_data, feature_names
    ):
        """DeepSHAP Explanation must have feature_names attribute."""
        explanation = deepshap_explainer.explain(sample_data[5])
        assert hasattr(explanation, "feature_names"), "DeepSHAP Explanation missing feature_names"
        assert explanation.feature_names == feature_names

    def test_raw_length(self, deepshap_explainer, sample_data, feature_names):
        """DeepSHAP attributions_raw has n_features entries."""
        explanation = deepshap_explainer.explain(sample_data[5])
        raw = explanation.explanation_data["attributions_raw"]
        assert len(raw) == len(feature_names)

    def test_raw_matches_dict(self, deepshap_explainer, sample_data, feature_names):
        """DeepSHAP raw values match dict values."""
        explanation = deepshap_explainer.explain(sample_data[5])
        raw = explanation.explanation_data["attributions_raw"]
        attributions = explanation.explanation_data["feature_attributions"]

        for i, fname in enumerate(feature_names):
            assert abs(raw[i] - attributions[fname]) < 1e-10

    def test_values_are_finite(self, deepshap_explainer, sample_data):
        """DeepSHAP values must be finite."""
        explanation = deepshap_explainer.explain(sample_data[5])
        for fname, val in explanation.explanation_data["feature_attributions"].items():
            assert np.isfinite(val), f"Non-finite DeepSHAP value for '{fname}': {val}"

    def test_target_class_matches_prediction(self, deepshap_explainer, sample_data, class_names):
        """DeepSHAP target_class must match model prediction."""
        explanation = deepshap_explainer.explain(sample_data[5])
        preds = deepshap_explainer.model.predict(sample_data[5:6])
        predicted_label = class_names[np.argmax(preds[0])]
        assert explanation.target_class == predicted_label


# ──────────────────────────────────────────────
# DeepLIFT Methods Tests
# ──────────────────────────────────────────────


class TestDeepLIFTMethods:
    """Tests for different DeepLIFT methods."""

    def test_rescale_method(self, deeplift_explainer, sample_data):
        """Rescale method produces valid attributions."""
        explanation = deeplift_explainer.explain(sample_data[0], method="rescale")
        assert explanation.explanation_data["method"] == "rescale"
        assert "attributions_raw" in explanation.explanation_data

    def test_rescale_exact_method_is_rejected(self, deeplift_explainer, sample_data):
        """The former low-step IG approximation must not be called DeepLIFT."""
        with pytest.raises(ValueError, match="not DeepLIFT"):
            deeplift_explainer.explain(sample_data[0], method="rescale_exact")

    def test_backend_and_score_space_are_recorded(self, deeplift_explainer, sample_data):
        """Every result states which verified backend and score were used."""
        explanation = deeplift_explainer.explain(sample_data[0], target_class=1)
        assert explanation.explanation_data["backend"] == "captum.DeepLift"
        assert explanation.explanation_data["output_space"] == "model"
        assert explanation.explanation_data["target_index"] == 1


# ──────────────────────────────────────────────
# DeepLIFT Baselines Tests
# ──────────────────────────────────────────────


class TestDeepLIFTBaselines:
    """Tests for different baseline options."""

    def test_zero_baseline(self, deeplift_explainer, sample_data):
        """Default zero baseline works."""
        explanation = deeplift_explainer.explain(sample_data[0])
        baseline = explanation.explanation_data["baseline"]
        assert all(b == 0 for b in baseline)

    def test_custom_baseline(self, simple_classifier, sample_data, feature_names, class_names):
        """Custom baseline works."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        custom_baseline = np.ones(4, dtype=np.float32) * 0.5
        explainer = DeepLIFTExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            baseline=custom_baseline,
        )

        explanation = explainer.explain(sample_data[0])
        assert np.allclose(explanation.explanation_data["baseline"], custom_baseline)

    def test_set_baseline_from_data(self, deeplift_explainer, sample_data):
        """set_baseline from data works."""
        deeplift_explainer.set_baseline(sample_data, method="mean")
        explanation = deeplift_explainer.explain(sample_data[0])
        expected = np.mean(sample_data, axis=0)
        assert np.allclose(explanation.explanation_data["baseline"], expected, atol=1e-5)


# ──────────────────────────────────────────────
# DeepLIFT Convergence Tests
# ──────────────────────────────────────────────


class TestDeepLIFTConvergence:
    """Tests for summation-to-delta property."""

    def test_convergence_delta(self, deeplift_explainer, sample_data):
        """DeepLIFT attributions approximate F(x) - F(baseline)."""
        explanation = deeplift_explainer.explain(sample_data[0], return_convergence_delta=True)
        delta = explanation.explanation_data["convergence_delta"]
        assert delta < 1e-5
        assert explanation.explanation_data["captum_convergence_delta"] < 1e-5


# ──────────────────────────────────────────────
# DeepLIFT Multiple Baselines Tests
# ──────────────────────────────────────────────


class TestDeepLIFTMultipleBaselines:
    """Tests for multiple baselines averaging."""

    def test_multiple_baselines(self, deeplift_explainer, sample_data):
        """explain_with_multiple_baselines works."""
        explanation = deeplift_explainer.explain_with_multiple_baselines(
            sample_data[5], baselines=sample_data[:5], target_class=0
        )
        assert explanation.explainer_name == "DeepLIFT_MultiBaseline"
        assert "attributions_std" in explanation.explanation_data
        assert explanation.explanation_data["n_baselines"] == 5


# ──────────────────────────────────────────────
# DeepLIFT Compare IG Tests
# ──────────────────────────────────────────────


class TestDeepLIFTCompareIG:
    """Tests comparing DeepLIFT to Integrated Gradients."""

    def test_ig_comparison(self, deeplift_explainer, sample_data):
        """DeepLIFT correlates with Integrated Gradients for ReLU nets."""
        comparison = deeplift_explainer.compare_with_integrated_gradients(
            sample_data[0], target_class=0, ig_steps=50
        )
        assert comparison["correlation"] > 0.8

    def test_constant_vectors_report_undefined_pearson_correlation(self):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        model = nn.Sequential(nn.Linear(2, 1, bias=False))
        with torch.no_grad():
            model[0].weight.fill_(1.0)
        explainer = DeepLIFTExplainer(PyTorchAdapter(model, task="regression"), ["a", "b"])

        comparison = explainer.compare_with_integrated_gradients(
            np.ones(2, dtype=np.float32),
            baseline=np.zeros(2, dtype=np.float32),
            ig_steps=10,
        )

        assert comparison["correlation"] is None
        assert comparison["correlation_defined"] is False
        assert "constant" in comparison["correlation_undefined_reason"]


# ──────────────────────────────────────────────
# DeepSHAP Basic Tests
# ──────────────────────────────────────────────


class TestDeepSHAPBasic:
    """Basic DeepSHAP tests."""

    def test_creation(self, simple_classifier, sample_data, feature_names, class_names):
        """DeepSHAP can be created."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTShapExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")
        explainer = DeepLIFTShapExplainer(
            model=adapter,
            feature_names=feature_names,
            class_names=class_names,
            background_data=sample_data,
        )
        assert explainer._background_data is not None

    def test_requires_background(self, simple_classifier, sample_data, feature_names, class_names):
        """DeepSHAP raises error without background data."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTShapExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")
        explainer = DeepLIFTShapExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names
        )
        with pytest.raises(ValueError, match="Background data not set"):
            explainer.explain(sample_data[0])

    def test_per_call_baseline_is_rejected(self, deepshap_explainer, sample_data):
        """DeepSHAP's reference is its configured background distribution."""
        with pytest.raises(ValueError, match="background distribution"):
            deepshap_explainer.explain(sample_data[5], baseline=np.zeros(4, dtype=np.float32))

    def test_legacy_positional_method_slots_remain_compatible(
        self, deepshap_explainer, sample_data
    ):
        """The parent-compatible signature preserves prior positional calls."""
        explanation = deepshap_explainer.explain(sample_data[5], None, "rescale", True)

        assert explanation.explanation_data["method"] == "rescale"
        assert "convergence_delta" in explanation.explanation_data

    def test_explain(self, deepshap_explainer, sample_data):
        """DeepSHAP produces valid explanations."""
        from explainiverse.core.explanation import Explanation

        explanation = deepshap_explainer.explain(sample_data[5])
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "DeepSHAP"
        assert "feature_attributions" in explanation.explanation_data
        assert "attributions_std" in explanation.explanation_data
        assert explanation.explanation_data["n_background_samples"] == 5


# ---------------------------------------------------------------------------
# Accuracy and support-contract tests
# ---------------------------------------------------------------------------


class TestDeepLIFTVerifiedAccuracy:
    """Tests with analytical or canonical-reference oracles."""

    def test_import_does_not_require_zeropad1d_export(self):
        """Torch 2.0 can import DeepLIFT without the later ZeroPad1d export."""
        check = """
import torch.nn as nn

if hasattr(nn, "ZeroPad1d"):
    delattr(nn, "ZeroPad1d")

import explainiverse.explainers.gradient.deeplift as deeplift

assert nn.ConstantPad1d in deeplift._SUPPORTED_LINEAR_TYPES
"""
        completed = subprocess.run(
            [sys.executable, "-c", check],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr

    def test_available_constant_padding_types_are_supported_without_version_skips(self):
        """The Torch floor and newer ZeroPad alias share one verified base."""
        import explainiverse.explainers.gradient.deeplift as module

        padding_modules = [nn.ConstantPad1d(1, 0.0)]
        if hasattr(nn, "ZeroPad1d"):
            padding_modules.append(nn.ZeroPad1d(1))
        for padding in padding_modules:
            model = nn.Sequential(padding, nn.Linear(3, 1))
            assert isinstance(model[0], module._SUPPORTED_LINEAR_TYPES)
            module._validate_supported_model(model)

    def test_saturated_sigmoid_matches_output_delta(self):
        """Rescale handles saturation where midpoint gradient is badly wrong."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Sigmoid())
        with torch.no_grad():
            model[0].weight.fill_(10.0)
        adapter = PyTorchAdapter(model, task="regression", output_activation="none")
        explainer = DeepLIFTExplainer(adapter, ["x"])

        explanation = explainer.explain(
            np.array([1.0], dtype=np.float32),
            baseline=np.array([0.0], dtype=np.float32),
            return_convergence_delta=True,
        )
        value = explanation.explanation_data["attributions_raw"][0]
        expected = torch.sigmoid(torch.tensor(10.0)).item() - 0.5

        assert value == pytest.approx(expected, abs=1e-7)
        assert explanation.explanation_data["convergence_delta"] < 1e-7
        # The removed midpoint-gradient approximation returned about 0.066.
        assert value > 0.49

    def test_matches_direct_captum_on_relu_network(self):
        """Wrapper output agrees with the canonical backend on raw logits."""
        from captum.attr import DeepLift

        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1.0, -2.0], [0.5, 1.5], [-1.0, 0.25]]))
            model[0].bias.copy_(torch.tensor([-0.2, 0.1, 0.3]))
            model[2].weight.copy_(torch.tensor([[0.5, -1.0, 2.0], [-0.25, 1.5, -0.5]]))
            model[2].bias.copy_(torch.tensor([0.1, -0.4]))

        x = torch.tensor([[1.2, -0.7]])
        baseline = torch.tensor([[-0.1, 0.2]])
        reference = (
            DeepLift(model).attribute(x, baselines=baseline, target=1).detach().numpy().reshape(-1)
        )

        adapter = PyTorchAdapter(model, task="classification")
        explainer = DeepLIFTExplainer(adapter, ["a", "b"])
        actual = np.asarray(
            explainer.explain(
                x.numpy().reshape(-1),
                baseline=baseline.numpy().reshape(-1),
                target_class=1,
            ).explanation_data["attributions_raw"]
        )
        np.testing.assert_allclose(actual, reference, atol=1e-7, rtol=1e-6)

    def test_implicit_target_is_fixed_at_the_input(self):
        """The selected class must not change between input and baseline."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        model = nn.Sequential(nn.Linear(1, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[-1.0], [1.0]]))
        adapter = PyTorchAdapter(model, task="classification")
        explainer = DeepLIFTExplainer(adapter, ["x"])

        explanation = explainer.explain(
            np.array([1.0], dtype=np.float32),
            baseline=np.array([-1.0], dtype=np.float32),
            return_convergence_delta=True,
        )

        assert explanation.explanation_data["target_index"] == 1
        assert explanation.explanation_data["attributions_raw"] == pytest.approx([2.0])
        assert explanation.explanation_data["prediction_difference"] == pytest.approx(2.0)
        assert explanation.explanation_data["convergence_delta"] < 1e-7

    def test_one_logit_binary_classes_are_complementary(self):
        """Both binary classes are valid targets in prediction-score space."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        model = nn.Sequential(nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model[0].weight.fill_(4.0)
        adapter = PyTorchAdapter(
            model,
            task="classification",
            class_names=["negative", "positive"],
        )
        explainer = DeepLIFTExplainer(adapter, ["x"], class_names=["negative", "positive"])

        negative = explainer.explain(
            np.array([1.0], dtype=np.float32),
            target_class=0,
            return_convergence_delta=True,
        )
        positive = explainer.explain(
            np.array([1.0], dtype=np.float32),
            target_class=1,
            return_convergence_delta=True,
        )
        neg_value = negative.explanation_data["attributions_raw"][0]
        pos_value = positive.explanation_data["attributions_raw"][0]
        expected = torch.sigmoid(torch.tensor(4.0)).item() - 0.5

        assert pos_value == pytest.approx(expected, abs=1e-7)
        assert neg_value == pytest.approx(-expected, abs=1e-7)
        assert negative.explanation_data["output_space"] == "prediction"
        assert positive.explanation_data["output_space"] == "prediction"
        assert negative.explanation_data["convergence_delta"] < 1e-7
        assert positive.explanation_data["convergence_delta"] < 1e-7

    def test_raw_multiplier_has_verified_rescale_meaning(self):
        """multiply_by_inputs=False returns contribution divided by input delta."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Sigmoid())
        with torch.no_grad():
            model[0].weight.fill_(10.0)
        adapter = PyTorchAdapter(model, task="regression", output_activation="none")
        explainer = DeepLIFTExplainer(adapter, ["x"], multiply_by_inputs=False)

        explanation = explainer.explain(
            np.array([2.0], dtype=np.float32),
            baseline=np.array([0.0], dtype=np.float32),
        )
        output_delta = torch.sigmoid(torch.tensor(20.0)).item() - 0.5
        multiplier = explanation.explanation_data["attributions_raw"][0]

        assert multiplier == pytest.approx(output_delta / 2.0, abs=1e-7)
        with pytest.raises(ValueError, match="Convergence delta"):
            explainer.explain(
                np.array([2.0], dtype=np.float32),
                return_convergence_delta=True,
            )

    def test_deepshap_matches_direct_captum_multiple_baselines(self):
        """DeepSHAP is the Captum DeepLiftShap baseline expectation."""
        from captum.attr import DeepLiftShap

        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTShapExplainer

        model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[2.0, -1.0]]))
            model[0].bias.fill_(-0.25)
        x = torch.tensor([[1.0, 0.5]])
        backgrounds = torch.tensor([[0.0, 0.0], [0.25, -0.5], [-1.0, 1.0]])
        reference = (
            DeepLiftShap(model)
            .attribute(x, baselines=backgrounds, target=0)
            .detach()
            .numpy()
            .reshape(-1)
        )

        adapter = PyTorchAdapter(model, task="regression", output_activation="none")
        explainer = DeepLIFTShapExplainer(
            adapter,
            ["a", "b"],
            background_data=backgrounds.numpy(),
        )
        explanation = explainer.explain(x.numpy().reshape(-1), return_convergence_delta=True)
        actual = np.asarray(explanation.explanation_data["attributions_raw"])

        np.testing.assert_allclose(actual, reference, atol=1e-7, rtol=1e-6)
        assert explanation.explanation_data["backend"] == "captum.DeepLiftShap"
        assert explanation.explanation_data["n_background_samples"] == 3
        assert explanation.explanation_data["convergence_delta"] < 1e-6

    def test_functional_nonlinearity_is_rejected(self):
        """Functional ReLU cannot silently degrade to gradient-times-input."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        class FunctionalReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                return torch.relu(self.linear(x))

        adapter = PyTorchAdapter(FunctionalReLU(), task="regression")
        with pytest.raises(NotImplementedError, match="exact nn.Sequential.*functional"):
            DeepLIFTExplainer(adapter, ["x"])

    def test_unsupported_nonlinear_module_is_rejected(self):
        """Unsupported nonlinear modules fail instead of returning gradients."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        adapter = PyTorchAdapter(
            nn.Sequential(nn.Linear(1, 2), nn.GELU(), nn.Linear(2, 1)),
            task="regression",
        )
        with pytest.raises(NotImplementedError, match="GELU"):
            DeepLIFTExplainer(adapter, ["x"])

    @pytest.mark.parametrize("method", ["deeplift", "deepshap"])
    def test_reused_nonlinear_module_is_rejected_before_model_work(self, method):
        """Captum hook ambiguity cannot return a DeepLIFT/DeepSHAP attribution."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer, DeepLIFTShapExplainer

        shared_relu = nn.ReLU()
        model = nn.Sequential(
            nn.Linear(2, 2),
            shared_relu,
            nn.Linear(2, 2),
            shared_relu,
            nn.Linear(2, 1),
        )
        adapter = PyTorchAdapter(model, task="regression")

        with pytest.raises(NotImplementedError, match="cannot safely reuse.*nonlinear module"):
            if method == "deeplift":
                DeepLIFTExplainer(adapter, ["x0", "x1"])
            else:
                DeepLIFTShapExplainer(
                    adapter,
                    ["x0", "x1"],
                    background_data=np.zeros((2, 2), dtype=np.float32),
                )

    @pytest.mark.parametrize("method", ["deeplift", "deepshap"])
    def test_dynamic_untraceable_graph_is_rejected_before_model_work(self, method):
        """Tensor-dependent control flow fails closed for both Captum surfaces."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer, DeepLIFTShapExplainer

        class TensorBranch(nn.Module):
            def __init__(self):
                super().__init__()
                self.positive = nn.Linear(2, 1)
                self.negative = nn.Linear(2, 1)
                self.tensor_forward_calls = 0

            def forward(self, values):
                if isinstance(values, torch.Tensor):
                    self.tensor_forward_calls += 1
                if values.sum() > 0:
                    return self.positive(values)
                return self.negative(values)

        model = TensorBranch()
        adapter = PyTorchAdapter(model, task="regression")

        with pytest.raises(
            NotImplementedError,
            match="exact nn.Sequential.*dynamic",
        ):
            if method == "deeplift":
                DeepLIFTExplainer(adapter, ["x0", "x1"])
            else:
                DeepLIFTShapExplainer(
                    adapter,
                    ["x0", "x1"],
                    background_data=np.zeros((2, 2), dtype=np.float32),
                )

        assert model.tensor_forward_calls == 0

    def test_training_state_is_temporarily_isolated_and_restored(self):
        """Construction and attribution preserve mixed modes, buffers, and grads."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        network = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Dropout(p=0.95),
            nn.Linear(2, 1, bias=False),
        )
        batch_norm = network[0]
        dropout = network[1]
        output = network[2]
        with torch.no_grad():
            output.weight.fill_(1.0)
        adapter = PyTorchAdapter(network, task="regression")
        network.train()
        dropout.eval()
        output.weight.grad = torch.full_like(output.weight, 5.0)
        flags_before = [module.training for module in network.modules()]
        mean_before = batch_norm.running_mean.detach().clone()
        gradient_object = output.weight.grad

        explainer = DeepLIFTExplainer(adapter, ["a", "b"])
        assert [module.training for module in network.modules()] == flags_before

        instance = np.array([1.0, -1.0], dtype=np.float32)
        first = explainer.explain(instance, target_class=0)
        second = explainer.explain(instance, target_class=0)

        np.testing.assert_allclose(
            first.explanation_data["attributions_raw"],
            second.explanation_data["attributions_raw"],
        )
        assert [module.training for module in network.modules()] == flags_before
        torch.testing.assert_close(batch_norm.running_mean, mean_before)
        assert output.weight.grad is gradient_object
        torch.testing.assert_close(
            output.weight.grad,
            torch.full_like(output.weight, 5.0),
        )

    def test_missing_captum_has_clear_error(self, monkeypatch):
        """There is no fake fallback when the canonical backend is absent."""
        import explainiverse.explainers.gradient.deeplift as module
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(nn.Sequential(nn.Linear(1, 1)), task="regression")
        monkeypatch.setattr(module, "CAPTUM_AVAILABLE", False)
        with pytest.raises(ImportError, match="pip install captum"):
            module.DeepLIFTExplainer(adapter, ["x"])

    def test_flat_contract_rejects_matrix_instance_and_baseline_with_same_value_count(self):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        adapter = PyTorchAdapter(nn.Sequential(nn.Linear(4, 1)), task="regression")
        explainer = DeepLIFTExplainer(adapter, ["a", "b", "c", "d"])
        flat = np.ones(4, dtype=np.float32)

        with pytest.raises(ValueError, match="one-dimensional flat feature vector"):
            explainer.explain(np.ones((2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="one-dimensional flat feature vector"):
            explainer.explain(flat, baseline=np.zeros((2, 2), dtype=np.float32))

    def test_random_baseline_uses_reproducible_local_rng(self):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        adapter = PyTorchAdapter(nn.Sequential(nn.Linear(2, 1)), task="regression")
        first = DeepLIFTExplainer(adapter, ["a", "b"], baseline="random", random_state=23)
        second = DeepLIFTExplainer(adapter, ["a", "b"], baseline="random", random_state=23)
        instance = np.array([-2.0, 4.0], dtype=np.float32)

        np.random.seed(1234)
        expected_next = np.random.RandomState(1234).random_sample()
        first_baseline = first._get_baseline(instance)
        actual_next = np.random.random()
        second_baseline = second._get_baseline(instance)

        assert actual_next == expected_next
        np.testing.assert_allclose(first_baseline, second_baseline)
        np.testing.assert_allclose(first_baseline, first._get_baseline(instance))

    def test_deepshap_subsampling_uses_reproducible_local_rng(self):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTShapExplainer

        adapter = PyTorchAdapter(nn.Sequential(nn.Linear(2, 1)), task="regression")
        background = np.arange(40, dtype=np.float32).reshape(20, 2)

        np.random.seed(4321)
        expected_next = np.random.RandomState(4321).random_sample()
        first = DeepLIFTShapExplainer(
            adapter,
            ["a", "b"],
            background_data=background,
            n_background_samples=5,
            random_state=29,
        )
        actual_next = np.random.random()
        second = DeepLIFTShapExplainer(
            adapter,
            ["a", "b"],
            background_data=background,
            n_background_samples=5,
            random_state=29,
        )

        assert actual_next == expected_next
        np.testing.assert_array_equal(first._background_data, second._background_data)

    @pytest.mark.parametrize(
        ("random_state", "expected"),
        [(True, TypeError), (1.5, TypeError), (-1, ValueError)],
    )
    def test_random_state_validation(self, random_state, expected):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        adapter = PyTorchAdapter(nn.Sequential(nn.Linear(1, 1)), task="regression")
        with pytest.raises(expected, match="random_state"):
            DeepLIFTExplainer(adapter, ["x"], random_state=random_state)


# ──────────────────────────────────────────────
# Registry Tests
# ──────────────────────────────────────────────


class TestDeepLIFTRegistry:
    """Registry integration tests."""

    def test_deeplift_registered(self):
        from explainiverse import default_registry

        assert "deeplift" in default_registry.list_explainers()

    def test_deepshap_registered(self):
        from explainiverse import default_registry

        assert "deepshap" in default_registry.list_explainers()

    def test_deeplift_metadata(self):
        from explainiverse import default_registry

        meta = default_registry.get_meta("deeplift")
        assert meta.scope == "local"
        assert "neural" in meta.model_types
        assert "Shrikumar" in meta.paper_reference

    def test_deeplift_filter_neural(self):
        from explainiverse import default_registry

        neural = default_registry.filter(model_type="neural")
        assert "deeplift" in neural
        assert "deepshap" in neural

    def test_deeplift_via_registry(self, simple_classifier, feature_names, class_names):
        from explainiverse import default_registry
        from explainiverse.adapters import PyTorchAdapter

        adapter = PyTorchAdapter(simple_classifier, task="classification", class_names=class_names)
        explainer = default_registry.create(
            "deeplift", model=adapter, feature_names=feature_names, class_names=class_names
        )
        assert explainer is not None


def _integrity_deep_explainer(kind, model, *, task="regression", adapter_kwargs=None):
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import DeepLIFTExplainer, DeepLIFTShapExplainer

    adapter = PyTorchAdapter(model, task=task, **(adapter_kwargs or {}))
    if kind == "deeplift":
        return DeepLIFTExplainer(adapter, ["x0", "x1"])
    return DeepLIFTShapExplainer(
        adapter,
        ["x0", "x1"],
        background_data=np.zeros((2, 2), dtype=np.float32),
    )


@pytest.mark.parametrize("kind", ["deeplift", "deepshap"])
def test_canonical_forward_identity_rejects_wrong_feature_swap_and_clean_control(kind):
    relu = nn.ReLU()
    output = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        output.weight.copy_(torch.tensor([[1.0, 0.0]]))
    model = nn.Sequential(relu, output)

    def swapped_forward(self, values):
        return torch.relu(values.flip(1))

    relu.forward = MethodType(swapped_forward, relu)
    with pytest.raises(RuntimeError, match="instance-shadowed forward"):
        _integrity_deep_explainer(kind, model)

    clean_output = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        clean_output.weight.copy_(torch.tensor([[1.0, 0.0]]))
    explainer = _integrity_deep_explainer(kind, nn.Sequential(nn.ReLU(), clean_output))
    result = explainer.explain(np.array([2.0, 3.0], dtype=np.float32), target_class=0)
    np.testing.assert_allclose(result.explanation_data["attributions_raw"], [2.0, 0.0])


@pytest.mark.parametrize("kind", ["deeplift", "deepshap"])
@pytest.mark.parametrize("target", ["root_forward", "root_hook", "relu_call_impl"])
def test_root_and_call_pipeline_integrity_fail_closed(kind, target):
    model = nn.Sequential(nn.ReLU(), nn.Linear(2, 1, bias=False))
    if target == "root_forward":
        model.forward = MethodType(lambda self, values: values.flip(1), model)
    elif target == "root_hook":
        model.register_forward_hook(lambda _module, _inputs, output: output.flip(1))
    else:
        model[0]._call_impl = MethodType(lambda self, values: values * values, model[0])

    with pytest.raises(RuntimeError, match="instance-shadowed|pre-existing"):
        _integrity_deep_explainer(kind, model)


@pytest.mark.parametrize("kind", ["deeplift", "deepshap"])
def test_batchnorm_requires_tracked_statistics_and_tracked_control_is_conservative(kind):
    untracked = nn.Sequential(
        nn.BatchNorm1d(2, track_running_stats=False),
        nn.Linear(2, 1, bias=False),
    )
    with pytest.raises(NotImplementedError, match="tracked running statistics"):
        _integrity_deep_explainer(kind, untracked)

    tracked_output = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        tracked_output.weight.copy_(torch.tensor([[1.0, -1.0]]))
    explainer = _integrity_deep_explainer(kind, nn.Sequential(nn.BatchNorm1d(2), tracked_output))
    result = explainer.explain(
        np.array([2.0, 1.0], dtype=np.float32),
        target_class=0,
        return_convergence_delta=True,
    )
    assert result.explanation_data["convergence_delta"] < 1e-6


@pytest.mark.parametrize("kind", ["deeplift", "deepshap"])
def test_softmax_is_rejected_before_nonconservative_attribution(kind):
    model = nn.Sequential(nn.Softmax(dim=1), nn.Linear(2, 1, bias=False))
    with pytest.raises(NotImplementedError, match="Softmax.*completeness"):
        _integrity_deep_explainer(kind, model)


@pytest.mark.parametrize(
    "registration",
    [
        "register_module_forward_pre_hook",
        "register_module_forward_hook",
        "register_module_full_backward_pre_hook",
        "register_module_full_backward_hook",
    ],
)
def test_global_module_execution_hooks_are_rejected(registration):
    import torch.nn.modules.module as module_runtime

    register = getattr(module_runtime, registration)
    handle = register(lambda *_args: None)
    try:
        with pytest.raises(RuntimeError, match="process-global"):
            _integrity_deep_explainer("deeplift", nn.Sequential(nn.ReLU(), nn.Linear(2, 1)))
    finally:
        handle.remove()


@pytest.mark.parametrize("kind", ["deeplift", "deepshap"])
@pytest.mark.parametrize("mutation", ["replace_child", "forward", "hook", "call_impl"])
def test_post_construction_graph_mutations_are_revalidated_before_model_work(kind, mutation):
    model = nn.Sequential(nn.ReLU(), nn.Linear(2, 1, bias=False))
    explainer = _integrity_deep_explainer(kind, model)
    if mutation == "replace_child":
        model[0] = nn.Sigmoid()
        match = "model graph changed"
    elif mutation == "forward":
        model[0].forward = MethodType(lambda self, values: values.flip(1), model[0])
        match = "instance-shadowed forward"
    elif mutation == "hook":
        model[0].register_forward_hook(lambda _module, _inputs, output: output.flip(1))
        match = "pre-existing"
    else:
        model[0]._call_impl = MethodType(lambda self, values: values * values, model[0])
        match = "instance-shadowed _call_impl"

    with pytest.raises(RuntimeError, match=match):
        explainer.explain(np.array([2.0, 3.0], dtype=np.float32), target_class=0)


def test_class_level_forward_monkeypatch_is_not_blessed_by_runtime_validation(monkeypatch):
    model = nn.Sequential(nn.ReLU(), nn.Linear(2, 1, bias=False))
    explainer = _integrity_deep_explainer("deeplift", model)
    monkeypatch.setattr(nn.ReLU, "forward", lambda self, values: values.flip(1))

    with pytest.raises(RuntimeError, match="canonical forward"):
        explainer.explain(np.array([2.0, 3.0], dtype=np.float32), target_class=0)


def test_multiclass_softmax_prediction_score_is_rejected_but_raw_scores_work():
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import DeepLIFTExplainer

    prediction_adapter = PyTorchAdapter(
        nn.Sequential(nn.Linear(2, 2, bias=False)),
        task="classification",
        gradient_output="prediction",
    )
    with pytest.raises(NotImplementedError, match="softmax prediction-score"):
        DeepLIFTExplainer(prediction_adapter, ["x0", "x1"])

    raw_adapter = PyTorchAdapter(
        nn.Sequential(nn.Linear(2, 2, bias=False)),
        task="classification",
        gradient_output="model",
    )
    result = DeepLIFTExplainer(raw_adapter, ["x0", "x1"]).explain(
        np.array([1.0, 2.0], dtype=np.float32), target_class=0
    )
    assert result.explanation_data["output_space"] == "model"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
