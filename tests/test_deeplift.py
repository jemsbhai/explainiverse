# tests/test_deeplift.py
"""
Tests for DeepLIFT and DeepSHAP explainers.

These tests require PyTorch to be installed. They will be skipped
if torch is not available.

Reference:
    Shrikumar et al., 2017 — "Learning Important Features Through
    Propagating Activation Differences." ICML 2017.
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


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

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

    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

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

    adapter = PyTorchAdapter(simple_classifier, task="classification",
                              class_names=class_names)
    return DeepLIFTExplainer(
        model=adapter,
        feature_names=feature_names,
        class_names=class_names
    )


@pytest.fixture
def deepshap_explainer(simple_classifier, sample_data, feature_names, class_names):
    """Pre-built DeepSHAP explainer for convenience."""
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import DeepLIFTShapExplainer

    adapter = PyTorchAdapter(simple_classifier, task="classification",
                              class_names=class_names)
    return DeepLIFTShapExplainer(
        model=adapter,
        feature_names=feature_names,
        class_names=class_names,
        background_data=sample_data[:5]
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
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        from explainiverse.adapters import SklearnAdapter
        from sklearn.linear_model import LogisticRegression

        sklearn_model = LogisticRegression()
        sklearn_model.fit(np.random.randn(100, 4), np.random.randint(0, 3, 100))
        adapter = SklearnAdapter(sklearn_model)

        with pytest.raises(TypeError, match="predict_with_gradients"):
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
        from explainiverse.explainers.gradient import DeepLIFTExplainer
        from explainiverse.core.explanation import Explanation

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

    def test_attribution_keys_match_feature_names(self, deeplift_explainer,
                                                    sample_data, feature_names):
        """Attribution dict keys must be original feature names."""
        for i in range(min(5, len(sample_data))):
            explanation = deeplift_explainer.explain(sample_data[i])
            keys = set(explanation.explanation_data["feature_attributions"].keys())
            assert keys == set(feature_names), \
                f"Instance {i}: keys {keys} != feature names {set(feature_names)}"

    def test_attribution_count(self, deeplift_explainer, sample_data, feature_names):
        """Number of attributions equals number of features."""
        explanation = deeplift_explainer.explain(sample_data[0])
        assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)

    def test_attribution_values_are_float(self, deeplift_explainer, sample_data):
        """All attribution values are floats."""
        explanation = deeplift_explainer.explain(sample_data[0])
        for k, v in explanation.explanation_data["feature_attributions"].items():
            assert isinstance(v, float), f"'{k}' has type {type(v)}"

    def test_feature_names_stored_on_explanation(self, deeplift_explainer,
                                                   sample_data, feature_names):
        """Explanation must have feature_names attribute for evaluation metrics."""
        explanation = deeplift_explainer.explain(sample_data[0])
        assert hasattr(explanation, "feature_names"), \
            "Explanation missing feature_names — evaluation metrics will fail"
        assert explanation.feature_names == feature_names

    def test_attributions_raw_length(self, deeplift_explainer, sample_data,
                                      feature_names):
        """attributions_raw has exactly n_features entries."""
        explanation = deeplift_explainer.explain(sample_data[0])
        raw = explanation.explanation_data["attributions_raw"]
        assert len(raw) == len(feature_names), \
            f"attributions_raw has {len(raw)} entries, expected {len(feature_names)}"

    def test_attributions_raw_matches_dict(self, deeplift_explainer, sample_data,
                                            feature_names):
        """attributions_raw values must match feature_attributions values."""
        explanation = deeplift_explainer.explain(sample_data[0])
        raw = explanation.explanation_data["attributions_raw"]
        attributions = explanation.explanation_data["feature_attributions"]

        for i, fname in enumerate(feature_names):
            assert abs(raw[i] - attributions[fname]) < 1e-10, \
                f"raw[{i}]={raw[i]} != attributions['{fname}']={attributions[fname]}"

    def test_values_are_finite(self, deeplift_explainer, sample_data):
        """All attribution values must be finite."""
        for i in range(min(5, len(sample_data))):
            explanation = deeplift_explainer.explain(sample_data[i])
            for fname, val in explanation.explanation_data["feature_attributions"].items():
                assert np.isfinite(val), f"Non-finite value for '{fname}': {val}"

    def test_deterministic(self, deeplift_explainer, sample_data):
        """Same input produces same output."""
        v1 = list(deeplift_explainer.explain(sample_data[0])
                   .explanation_data["feature_attributions"].values())
        v2 = list(deeplift_explainer.explain(sample_data[0])
                   .explanation_data["feature_attributions"].values())
        np.testing.assert_array_almost_equal(v1, v2, decimal=8)


# ──────────────────────────────────────────────
# DeepLIFT Multi-Baseline Key Matching
# ──────────────────────────────────────────────

class TestDeepLIFTMultiBaselineKeyMatching:
    """Key matching and feature_names for multi-baseline explanations."""

    def test_multi_baseline_keys_match(self, deeplift_explainer, sample_data,
                                        feature_names):
        """Multi-baseline attributions keyed by original feature names."""
        explanation = deeplift_explainer.explain_with_multiple_baselines(
            sample_data[5], baselines=sample_data[:5], target_class=0
        )
        keys = set(explanation.explanation_data["feature_attributions"].keys())
        assert keys == set(feature_names)

    def test_multi_baseline_feature_names_stored(self, deeplift_explainer,
                                                   sample_data, feature_names):
        """Multi-baseline Explanation has feature_names attribute."""
        explanation = deeplift_explainer.explain_with_multiple_baselines(
            sample_data[5], baselines=sample_data[:5], target_class=0
        )
        assert hasattr(explanation, "feature_names"), \
            "Multi-baseline Explanation missing feature_names"
        assert explanation.feature_names == feature_names

    def test_multi_baseline_raw_length(self, deeplift_explainer, sample_data,
                                        feature_names):
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
            assert keys == set(feature_names), \
                f"Batch instance {i}: keys {keys} != feature names"

    def test_batch_feature_names_stored(self, deeplift_explainer, sample_data,
                                         feature_names):
        """Each batch explanation has feature_names attribute."""
        explanations = deeplift_explainer.explain_batch(sample_data[:3])
        for i, exp in enumerate(explanations):
            assert hasattr(exp, "feature_names"), \
                f"Batch instance {i}: missing feature_names"
            assert exp.feature_names == feature_names

    def test_batch_raw_length(self, deeplift_explainer, sample_data, feature_names):
        """Each batch explanation has correct attributions_raw length."""
        explanations = deeplift_explainer.explain_batch(sample_data[:3])
        for i, exp in enumerate(explanations):
            raw = exp.explanation_data["attributions_raw"]
            assert len(raw) == len(feature_names), \
                f"Batch {i}: raw length {len(raw)} != {len(feature_names)}"


# ──────────────────────────────────────────────
# DeepSHAP Key Matching & feature_names Tests
# ──────────────────────────────────────────────

class TestDeepSHAPKeyMatching:
    """Critical tests for DeepSHAP attribution correctness."""

    def test_attribution_keys_match_feature_names(self, deepshap_explainer,
                                                    sample_data, feature_names):
        """DeepSHAP attribution keys must be original feature names."""
        explanation = deepshap_explainer.explain(sample_data[5])
        keys = set(explanation.explanation_data["feature_attributions"].keys())
        assert keys == set(feature_names), \
            f"Keys {keys} != feature names {set(feature_names)}"

    def test_attribution_count(self, deepshap_explainer, sample_data, feature_names):
        """DeepSHAP returns correct number of attributions."""
        explanation = deepshap_explainer.explain(sample_data[5])
        assert len(explanation.explanation_data["feature_attributions"]) == len(feature_names)

    def test_feature_names_stored_on_explanation(self, deepshap_explainer,
                                                   sample_data, feature_names):
        """DeepSHAP Explanation must have feature_names attribute."""
        explanation = deepshap_explainer.explain(sample_data[5])
        assert hasattr(explanation, "feature_names"), \
            "DeepSHAP Explanation missing feature_names"
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

    def test_target_class_matches_prediction(self, deepshap_explainer,
                                               sample_data, class_names):
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

    def test_rescale_exact_method(self, deeplift_explainer, sample_data):
        """Rescale exact method produces valid attributions."""
        explanation = deeplift_explainer.explain(sample_data[0], method="rescale_exact")
        assert explanation.explanation_data["method"] == "rescale_exact"

    def test_methods_correlate(self, deeplift_explainer, sample_data):
        """Different methods produce correlated results."""
        exp_rescale = deeplift_explainer.explain(sample_data[0], method="rescale")
        exp_exact = deeplift_explainer.explain(sample_data[0], method="rescale_exact")

        attr_rescale = exp_rescale.explanation_data["attributions_raw"]
        attr_exact = exp_exact.explanation_data["attributions_raw"]

        corr = np.corrcoef(attr_rescale, attr_exact)[0, 1]
        assert corr > 0.8


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

    def test_custom_baseline(self, simple_classifier, sample_data, feature_names,
                              class_names):
        """Custom baseline works."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification",
                                  class_names=class_names)
        custom_baseline = np.ones(4, dtype=np.float32) * 0.5
        explainer = DeepLIFTExplainer(
            model=adapter, feature_names=feature_names,
            class_names=class_names, baseline=custom_baseline
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
        explanation = deeplift_explainer.explain(
            sample_data[0], return_convergence_delta=True
        )
        delta = explanation.explanation_data["convergence_delta"]
        pred_diff = abs(explanation.explanation_data["prediction_difference"])
        assert delta < pred_diff + 0.1


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
            model=adapter, feature_names=feature_names,
            class_names=class_names, background_data=sample_data
        )
        assert explainer._background_data is not None

    def test_requires_background(self, simple_classifier, sample_data,
                                   feature_names, class_names):
        """DeepSHAP raises error without background data."""
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTShapExplainer

        adapter = PyTorchAdapter(simple_classifier, task="classification")
        explainer = DeepLIFTShapExplainer(
            model=adapter, feature_names=feature_names, class_names=class_names
        )
        with pytest.raises(ValueError, match="Background data not set"):
            explainer.explain(sample_data[0])

    def test_explain(self, deepshap_explainer, sample_data):
        """DeepSHAP produces valid explanations."""
        from explainiverse.core.explanation import Explanation

        explanation = deepshap_explainer.explain(sample_data[5])
        assert isinstance(explanation, Explanation)
        assert explanation.explainer_name == "DeepSHAP"
        assert "feature_attributions" in explanation.explanation_data
        assert "attributions_std" in explanation.explanation_data
        assert explanation.explanation_data["n_background_samples"] == 5


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

        adapter = PyTorchAdapter(simple_classifier, task="classification",
                                  class_names=class_names)
        explainer = default_registry.create(
            "deeplift", model=adapter,
            feature_names=feature_names, class_names=class_names
        )
        assert explainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
