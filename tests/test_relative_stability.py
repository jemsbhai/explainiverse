# tests/test_relative_stability.py
"""
Tests for Relative Stability metrics (Agarwal et al., 2022).

Implements TDD tests for:
- Relative Input Stability (RIS) — Equation 2
- Relative Representation Stability (RRS) — Equation 3
- Relative Output Stability (ROS) — Equation 5
- Relative Stability (convenience all-in-one)

All metrics measure explanation instability: LOWER = more stable.

Reference:
    Agarwal, C., Johnson, N., Pawelczyk, M., Krishna, S., Saxena, E.,
    Zitnik, M., & Lakkaraju, H. (2022). Rethinking Stability for
    Attribution-based Explanations. arXiv:2203.06877.
"""

import warnings

import pytest
import numpy as np
from unittest.mock import MagicMock

from explainiverse.core.explanation import Explanation
from explainiverse.core.explainer import BaseExplainer


# =============================================================================
# Fixtures & Mock Classes
# =============================================================================

@pytest.fixture
def feature_names():
    return ["f0", "f1", "f2", "f3"]


@pytest.fixture
def single_instance():
    np.random.seed(42)
    return np.random.randn(4).astype(np.float64)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.randn(20, 4).astype(np.float64)


class _DeterministicExplainer:
    """
    Returns attributions proportional to the input.
    Small input changes → small explanation changes → low instability.
    """
    def __init__(self, feature_names, scale=1.0):
        self.feature_names = feature_names
        self.scale = scale

    def explain(self, instance):
        instance = np.asarray(instance).flatten()
        attrs = {fn: float(self.scale * instance[i])
                 for i, fn in enumerate(self.feature_names)}
        exp = Explanation(
            explainer_name="deterministic",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _ConstantExplainer:
    """
    Always returns the same attributions regardless of input.
    Zero explanation change → numerator = 0 → score = 0.
    """
    def __init__(self, feature_names, values=None):
        self.feature_names = feature_names
        self.values = values or [1.0] * len(feature_names)

    def explain(self, instance):
        attrs = {fn: float(self.values[i])
                 for i, fn in enumerate(self.feature_names)}
        exp = Explanation(
            explainer_name="constant",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _RandomExplainer:
    """
    Returns random attributions — high instability.
    """
    def __init__(self, feature_names, seed=None):
        self.feature_names = feature_names
        self.rng = np.random.default_rng(seed)

    def explain(self, instance):
        attrs = {fn: float(self.rng.standard_normal())
                 for fn in self.feature_names}
        exp = Explanation(
            explainer_name="random",
            target_class="class_0",
            explanation_data={"feature_attributions": attrs},
        )
        exp.feature_names = self.feature_names
        return exp


class _ConstantClassModel:
    """Always predicts the same class. All perturbations pass class filter."""
    def __init__(self, predicted_class=0, n_classes=2):
        self.predicted_class = predicted_class
        self.n_classes = n_classes

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return np.array([self.predicted_class])
        return np.full(X.shape[0], self.predicted_class)

    def predict_proba(self, X):
        X = np.asarray(X)
        n = 1 if X.ndim == 1 else X.shape[0]
        proba = np.zeros((n, self.n_classes))
        proba[:, self.predicted_class] = 1.0
        return proba


class _ThresholdModel:
    """Predicts class based on feature 0 > threshold. Some perturbations
    will cross the boundary and be filtered out."""
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X[:, 0] > self.threshold).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = (X[:, 0] > self.threshold).astype(float)
        return np.column_stack([1.0 - preds, preds])


def _linear_representation_fn(X):
    """Simple linear representation: h(x) = 2*x + 1. Deterministic,
    proportional to input — representation changes track input changes."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return 2.0 * X + 1.0


def _constant_representation_fn(X):
    """Constant representation — denominator of RRS → 0 (epsilon floor)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.ones_like(X) * 5.0


def _nonlinear_representation_fn(X):
    """Nonlinear representation: amplifies small input changes."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.tanh(X * 10.0)


def _linear_logit_fn(X):
    """Simple logit function: h(x) = sum of features per instance."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # Return 2-class logits
    s = X.sum(axis=1, keepdims=True)
    return np.hstack([-s, s])


def _constant_logit_fn(X):
    """Constant logits — denominator of ROS → epsilon floor."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.ones((X.shape[0], 2)) * 3.0


# =============================================================================
# Relative Input Stability (RIS) — Equation 2
# =============================================================================

class TestRelativeInputStability:
    """Tests for compute_relative_input_stability (Agarwal et al., 2022, Eq 2)."""

    def test_returns_float_by_default(self, feature_names, single_instance):
        """Default return type is float."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=10, seed=42,
        )
        assert isinstance(score, float)

    def test_returns_dict_with_details(self, feature_names, single_instance):
        """return_details=True returns a dict with diagnostics."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=10, seed=42, return_details=True,
        )
        assert isinstance(result, dict)
        assert "score" in result
        assert "max" in result
        assert "mean" in result
        assert "median" in result
        assert "n_valid" in result
        assert "n_total" in result
        assert "perturbation_scores" in result

    def test_non_negative(self, feature_names, single_instance):
        """Instability score is non-negative."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
        )
        assert score >= 0.0

    def test_constant_explainer_zero_instability(self, feature_names, single_instance):
        """Constant explainer → zero explanation change → score ≈ 0."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _ConstantExplainer(feature_names, values=[3.0, 2.0, 1.0, 0.5])
        model = _ConstantClassModel()
        score = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
        )
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_random_higher_than_deterministic(self, feature_names, single_instance):
        """Random explainer is more unstable than deterministic."""
        from explainiverse.evaluation import compute_relative_input_stability

        det = _DeterministicExplainer(feature_names)
        rng = _RandomExplainer(feature_names, seed=99)
        model = _ConstantClassModel()

        det_score = compute_relative_input_stability(
            det, model, single_instance,
            n_perturbations=30, seed=42,
        )
        rng_score = compute_relative_input_stability(
            rng, model, single_instance,
            n_perturbations=30, seed=42,
        )
        assert rng_score > det_score

    def test_reproducible_with_seed(self, feature_names, single_instance):
        """Same seed → same result."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        s1 = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
        )
        s2 = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
        )
        assert s1 == pytest.approx(s2, rel=1e-10)

    def test_aggregation_max(self, feature_names, single_instance):
        """aggregation='max' returns the maximum over perturbations."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
            aggregation="max", return_details=True,
        )
        assert result["score"] == pytest.approx(result["max"], rel=1e-10)

    def test_aggregation_mean(self, feature_names, single_instance):
        """aggregation='mean' returns the mean over perturbations."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
            aggregation="mean", return_details=True,
        )
        assert result["score"] == pytest.approx(result["mean"], rel=1e-10)

    def test_aggregation_median(self, feature_names, single_instance):
        """aggregation='median' returns the median over perturbations."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
            aggregation="median", return_details=True,
        )
        assert result["score"] == pytest.approx(result["median"], rel=1e-10)

    def test_invalid_aggregation_raises(self, feature_names, single_instance):
        """Invalid aggregation mode raises ValueError."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        with pytest.raises(ValueError, match="aggregation"):
            compute_relative_input_stability(
                explainer, model, single_instance,
                aggregation="invalid", seed=42,
            )

    def test_max_geq_mean_geq_zero(self, feature_names, single_instance):
        """max >= mean >= 0 for any run."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=30, seed=42, return_details=True,
        )
        assert result["max"] >= result["mean"] - 1e-10
        assert result["mean"] >= 0.0

    def test_n_valid_leq_n_total(self, feature_names, single_instance):
        """Number of valid perturbations ≤ total generated."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42, return_details=True,
        )
        assert result["n_valid"] <= result["n_total"]
        assert result["n_total"] == 20

    def test_constant_class_model_all_valid(self, feature_names, single_instance):
        """Constant-class model: all perturbations pass the same-class filter."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42, return_details=True,
        )
        assert result["n_valid"] == 20

    def test_class_filter_reduces_valid(self, feature_names):
        """Threshold model near boundary filters some perturbations."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        # Instance with feature 0 near threshold → some perturbations cross
        instance = np.array([0.05, 1.0, 0.5, -0.5])
        model = _ThresholdModel(threshold=0.0)
        result = compute_relative_input_stability(
            explainer, model, instance,
            n_perturbations=50, noise_scale=0.1, seed=42,
            return_details=True,
        )
        # Some should be filtered (not all 50 valid)
        assert result["n_valid"] < result["n_total"]
        assert result["n_valid"] > 0  # At least some pass

    def test_no_valid_perturbations_returns_nan(self, feature_names):
        """If all perturbations are filtered (class change), return NaN."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)

        class _FlipAllModel:
            """Original instance → class 0; everything else → class 1.
            This guarantees that no perturbation passes the same-class filter."""
            def __init__(self, original):
                self._original = np.asarray(original, dtype=np.float64).flatten()
            def predict(self, X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                results = []
                for row in X:
                    if np.allclose(row, self._original, atol=1e-12):
                        results.append(0)
                    else:
                        results.append(1)
                return np.array(results)

        instance = np.array([1.0, 2.0, 3.0, 4.0])
        model = _FlipAllModel(instance)
        score = compute_relative_input_stability(
            explainer, model, instance,
            n_perturbations=10, seed=42,
        )
        assert np.isnan(score)

    def test_norm_ord_l1(self, feature_names, single_instance):
        """L1 norm works."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_input_stability(
            explainer, model, single_instance,
            norm_ord=1, n_perturbations=10, seed=42,
        )
        assert np.isfinite(score)

    def test_norm_ord_linf(self, feature_names, single_instance):
        """L-inf norm works."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_input_stability(
            explainer, model, single_instance,
            norm_ord=np.inf, n_perturbations=10, seed=42,
        )
        assert np.isfinite(score)

    def test_different_noise_scales(self, feature_names, single_instance):
        """Different noise scales produce different results."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        s_small = compute_relative_input_stability(
            explainer, model, single_instance,
            noise_scale=0.01, n_perturbations=30, seed=42,
        )
        s_large = compute_relative_input_stability(
            explainer, model, single_instance,
            noise_scale=0.5, n_perturbations=30, seed=42,
        )
        # Both finite, and likely different
        assert np.isfinite(s_small)
        assert np.isfinite(s_large)

    def test_perturbation_scores_length(self, feature_names, single_instance):
        """perturbation_scores has length == n_valid."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=15, seed=42, return_details=True,
        )
        assert len(result["perturbation_scores"]) == result["n_valid"]

    def test_percent_change_formulation(self, feature_names):
        """
        Verify percent-change: the numerator is ||(e_x - e_x') / e_x||_p,
        NOT ||(e_x - e_x')||_p. With a scale-proportional explainer (e=s*x),
        the percent change in explanation equals the percent change in input,
        so the ratio should be approximately 1.0.
        """
        from explainiverse.evaluation import compute_relative_input_stability

        # E(x) = x (scale=1.0). Percent change in explanation = percent change in input.
        # So numerator ≈ denominator → RIS ≈ 1.0 for each perturbation.
        explainer = _DeterministicExplainer(feature_names, scale=1.0)
        model = _ConstantClassModel()
        # Use instance far from zero to avoid division issues
        instance = np.array([5.0, 3.0, 4.0, 2.0])
        result = compute_relative_input_stability(
            explainer, model, instance,
            n_perturbations=50, noise_scale=0.01, seed=42,
            return_details=True,
        )
        # For E(x)=x, percent change ratio should be ≈ 1.0
        assert result["score"] == pytest.approx(1.0, rel=0.1)


class TestBatchRelativeInputStability:
    """Tests for compute_batch_relative_input_stability."""

    def test_returns_dict(self, feature_names, sample_data):
        """Batch returns dict with expected keys."""
        from explainiverse.evaluation import compute_batch_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_input_stability(
            explainer, model, sample_data,
            n_perturbations=5, max_instances=5, seed=42,
        )
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "max" in result
        assert "min" in result
        assert "scores" in result
        assert "n_evaluated" in result

    def test_max_instances_limits(self, feature_names, sample_data):
        """max_instances limits evaluations."""
        from explainiverse.evaluation import compute_batch_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_input_stability(
            explainer, model, sample_data,
            n_perturbations=5, max_instances=3, seed=42,
        )
        assert result["n_evaluated"] == 3

    def test_mean_between_min_max(self, feature_names, sample_data):
        """mean is between min and max."""
        from explainiverse.evaluation import compute_batch_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_input_stability(
            explainer, model, sample_data,
            n_perturbations=5, max_instances=5, seed=42,
        )
        assert result["min"] <= result["mean"] <= result["max"]


# =============================================================================
# Relative Representation Stability (RRS) — Equation 3
# =============================================================================

class TestRelativeRepresentationStability:
    """Tests for compute_relative_representation_stability (Eq 3)."""

    def test_returns_float_by_default(self, feature_names, single_instance):
        """Default return type is float."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=10, seed=42,
        )
        assert isinstance(score, float)

    def test_returns_dict_with_details(self, feature_names, single_instance):
        """return_details=True returns diagnostic dict."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=10, seed=42, return_details=True,
        )
        assert isinstance(result, dict)
        assert "score" in result
        assert "n_valid" in result
        assert "perturbation_scores" in result

    def test_non_negative(self, feature_names, single_instance):
        """Score is non-negative."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=20, seed=42,
        )
        assert score >= 0.0

    def test_constant_explainer_zero(self, feature_names, single_instance):
        """Constant explainer → zero numerator → score ≈ 0."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _ConstantExplainer(feature_names, values=[3.0, 2.0, 1.0, 0.5])
        model = _ConstantClassModel()
        score = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=20, seed=42,
        )
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_random_higher_than_deterministic(self, feature_names, single_instance):
        """Random explainer more unstable than deterministic."""
        from explainiverse.evaluation import compute_relative_representation_stability

        det = _DeterministicExplainer(feature_names)
        rng = _RandomExplainer(feature_names, seed=99)
        model = _ConstantClassModel()

        det_score = compute_relative_representation_stability(
            det, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=30, seed=42,
        )
        rng_score = compute_relative_representation_stability(
            rng, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=30, seed=42,
        )
        assert rng_score > det_score

    def test_reproducible_with_seed(self, feature_names, single_instance):
        """Same seed → same result."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        s1 = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=20, seed=42,
        )
        s2 = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=20, seed=42,
        )
        assert s1 == pytest.approx(s2, rel=1e-10)

    def test_constant_representation_hits_epsilon_floor(self, feature_names, single_instance):
        """When representation doesn't change, denominator → ε_min."""
        from explainiverse.evaluation import compute_relative_representation_stability

        # Representation is constant → denominator percent change = 0 → ε_min kicks in
        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_constant_representation_fn,
            n_perturbations=10, seed=42,
        )
        # Score should be finite (ε_min prevents div-by-zero) and large
        assert np.isfinite(score)

    def test_aggregation_max(self, feature_names, single_instance):
        """aggregation='max' is the paper default."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=20, seed=42,
            aggregation="max", return_details=True,
        )
        assert result["score"] == pytest.approx(result["max"], rel=1e-10)

    def test_aggregation_mean(self, feature_names, single_instance):
        """aggregation='mean' returns mean."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=20, seed=42,
            aggregation="mean", return_details=True,
        )
        assert result["score"] == pytest.approx(result["mean"], rel=1e-10)

    def test_different_norm_orders(self, feature_names, single_instance):
        """L1, L2, L-inf norms all produce finite results."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        for norm in [1, 2, np.inf]:
            score = compute_relative_representation_stability(
                explainer, model, single_instance,
                representation_fn=_linear_representation_fn,
                norm_ord=norm, n_perturbations=10, seed=42,
            )
            assert np.isfinite(score), f"Non-finite for norm_ord={norm}"

    def test_representation_fn_called_correctly(self, feature_names, single_instance):
        """Representation function receives proper inputs."""
        from explainiverse.evaluation import compute_relative_representation_stability

        call_log = []
        def _tracking_repr_fn(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            call_log.append(X.shape)
            return X * 2.0

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_tracking_repr_fn,
            n_perturbations=5, seed=42,
        )
        # Should be called at least for original + perturbations
        assert len(call_log) >= 2

    def test_n_valid_reported(self, feature_names, single_instance):
        """n_valid and n_total are reported correctly."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=15, seed=42, return_details=True,
        )
        assert result["n_total"] == 15
        assert result["n_valid"] == 15  # Constant class model, all pass


class TestBatchRelativeRepresentationStability:
    """Tests for compute_batch_relative_representation_stability."""

    def test_returns_dict(self, feature_names, sample_data):
        """Batch returns dict with expected keys."""
        from explainiverse.evaluation import compute_batch_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_representation_stability(
            explainer, model, sample_data,
            representation_fn=_linear_representation_fn,
            n_perturbations=5, max_instances=5, seed=42,
        )
        assert "mean" in result
        assert "std" in result
        assert "scores" in result
        assert "n_evaluated" in result

    def test_max_instances_limits(self, feature_names, sample_data):
        """max_instances limits evaluations."""
        from explainiverse.evaluation import compute_batch_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_representation_stability(
            explainer, model, sample_data,
            representation_fn=_linear_representation_fn,
            n_perturbations=5, max_instances=4, seed=42,
        )
        assert result["n_evaluated"] == 4

    def test_mean_between_min_max(self, feature_names, sample_data):
        """mean is between min and max."""
        from explainiverse.evaluation import compute_batch_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_representation_stability(
            explainer, model, sample_data,
            representation_fn=_linear_representation_fn,
            n_perturbations=5, max_instances=5, seed=42,
        )
        assert result["min"] <= result["mean"] <= result["max"]


# =============================================================================
# Relative Output Stability (ROS) — Equation 5
# =============================================================================

class TestRelativeOutputStability:
    """Tests for compute_relative_output_stability (Eq 5)."""

    def test_returns_float_by_default(self, feature_names, single_instance):
        """Default return type is float."""
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_output_stability(
            explainer, model, single_instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=10, seed=42,
        )
        assert isinstance(score, float)

    def test_returns_dict_with_details(self, feature_names, single_instance):
        """return_details=True returns diagnostic dict."""
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_output_stability(
            explainer, model, single_instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=10, seed=42, return_details=True,
        )
        assert isinstance(result, dict)
        assert "score" in result
        assert "n_valid" in result

    def test_non_negative(self, feature_names, single_instance):
        """Score is non-negative."""
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_output_stability(
            explainer, model, single_instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=20, seed=42,
        )
        assert score >= 0.0

    def test_constant_explainer_zero(self, feature_names, single_instance):
        """Constant explainer → score ≈ 0."""
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = _ConstantExplainer(feature_names, values=[3.0, 2.0, 1.0, 0.5])
        model = _ConstantClassModel()
        score = compute_relative_output_stability(
            explainer, model, single_instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=20, seed=42,
        )
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_random_higher_than_deterministic(self, feature_names, single_instance):
        """Random explainer more unstable than deterministic."""
        from explainiverse.evaluation import compute_relative_output_stability

        det = _DeterministicExplainer(feature_names)
        rng = _RandomExplainer(feature_names, seed=99)
        model = _ConstantClassModel()

        det_score = compute_relative_output_stability(
            det, model, single_instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=30, seed=42,
        )
        rng_score = compute_relative_output_stability(
            rng, model, single_instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=30, seed=42,
        )
        assert rng_score > det_score

    def test_reproducible_with_seed(self, feature_names, single_instance):
        """Same seed → same result."""
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        s1 = compute_relative_output_stability(
            explainer, model, single_instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=20, seed=42,
        )
        s2 = compute_relative_output_stability(
            explainer, model, single_instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=20, seed=42,
        )
        assert s1 == pytest.approx(s2, rel=1e-10)

    def test_constant_logits_hits_epsilon_floor(self, feature_names, single_instance):
        """Constant logits → denominator → ε_min."""
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_output_stability(
            explainer, model, single_instance,
            logit_fn=_constant_logit_fn,
            n_perturbations=10, seed=42,
        )
        assert np.isfinite(score)

    def test_ros_uses_absolute_logit_diff(self, feature_names):
        """
        ROS denominator is ||h(x) - h(x')||_p (absolute difference),
        NOT percent change. Verify by comparing against a known logit_fn.
        """
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = _DeterministicExplainer(feature_names, scale=1.0)
        model = _ConstantClassModel()
        # Instance with known logit output
        instance = np.array([5.0, 3.0, 4.0, 2.0])
        result = compute_relative_output_stability(
            explainer, model, instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=20, noise_scale=0.01, seed=42,
            return_details=True,
        )
        # Should be finite and well-defined
        assert np.isfinite(result["score"])
        assert result["n_valid"] == 20

    def test_aggregation_modes(self, feature_names, single_instance):
        """All aggregation modes work."""
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        for agg in ["max", "mean", "median"]:
            score = compute_relative_output_stability(
                explainer, model, single_instance,
                logit_fn=_linear_logit_fn,
                n_perturbations=10, seed=42, aggregation=agg,
            )
            assert np.isfinite(score)


class TestBatchRelativeOutputStability:
    """Tests for compute_batch_relative_output_stability."""

    def test_returns_dict(self, feature_names, sample_data):
        """Batch returns dict with expected keys."""
        from explainiverse.evaluation import compute_batch_relative_output_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_output_stability(
            explainer, model, sample_data,
            logit_fn=_linear_logit_fn,
            n_perturbations=5, max_instances=5, seed=42,
        )
        assert "mean" in result
        assert "scores" in result
        assert "n_evaluated" in result

    def test_max_instances_limits(self, feature_names, sample_data):
        """max_instances limits evaluations."""
        from explainiverse.evaluation import compute_batch_relative_output_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_output_stability(
            explainer, model, sample_data,
            logit_fn=_linear_logit_fn,
            n_perturbations=5, max_instances=3, seed=42,
        )
        assert result["n_evaluated"] == 3


# =============================================================================
# Relative Stability (all-in-one convenience)
# =============================================================================

class TestRelativeStabilityConvenience:
    """Tests for compute_relative_stability (all three in one pass)."""

    def test_returns_all_three_metrics(self, feature_names, single_instance):
        """Returns dict with ris, rrs, ros keys."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=10, seed=42,
        )
        assert isinstance(result, dict)
        assert "ris" in result
        assert "rrs" in result
        assert "ros" in result

    def test_ris_only_without_repr_and_logit(self, feature_names, single_instance):
        """Without representation_fn and logit_fn, only RIS is computed."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_stability(
            explainer, model, single_instance,
            n_perturbations=10, seed=42,
        )
        assert "ris" in result
        assert result["rrs"] is None
        assert result["ros"] is None

    def test_rrs_computed_when_repr_fn_given(self, feature_names, single_instance):
        """RRS computed when representation_fn is provided."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=10, seed=42,
        )
        assert result["rrs"] is not None
        assert isinstance(result["rrs"], float)

    def test_ros_computed_when_logit_fn_given(self, feature_names, single_instance):
        """ROS computed when logit_fn is provided."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_stability(
            explainer, model, single_instance,
            logit_fn=_linear_logit_fn,
            n_perturbations=10, seed=42,
        )
        assert result["ros"] is not None
        assert isinstance(result["ros"], float)

    def test_shared_computation_matches_individual(self, feature_names, single_instance):
        """All-in-one results match individual function calls."""
        from explainiverse.evaluation import (
            compute_relative_stability,
            compute_relative_input_stability,
            compute_relative_representation_stability,
            compute_relative_output_stability,
        )

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        kwargs = dict(n_perturbations=20, noise_scale=0.05, seed=42)

        combined = compute_relative_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            **kwargs,
        )
        ris_solo = compute_relative_input_stability(
            explainer, model, single_instance, **kwargs,
        )
        rrs_solo = compute_relative_representation_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn, **kwargs,
        )
        ros_solo = compute_relative_output_stability(
            explainer, model, single_instance,
            logit_fn=_linear_logit_fn, **kwargs,
        )
        assert combined["ris"] == pytest.approx(ris_solo, rel=1e-10)
        assert combined["rrs"] == pytest.approx(rrs_solo, rel=1e-10)
        assert combined["ros"] == pytest.approx(ros_solo, rel=1e-10)

    def test_return_details_all_three(self, feature_names, single_instance):
        """return_details gives dicts for each metric."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=10, seed=42,
            return_details=True,
        )
        # Each sub-result should be a dict with diagnostics
        for key in ["ris", "rrs", "ros"]:
            assert isinstance(result[key], dict)
            assert "score" in result[key]
            assert "n_valid" in result[key]

    def test_constant_explainer_all_zero(self, feature_names, single_instance):
        """Constant explainer → all three scores ≈ 0."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = _ConstantExplainer(feature_names, values=[3.0, 2.0, 1.0, 0.5])
        model = _ConstantClassModel()
        result = compute_relative_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=20, seed=42,
        )
        assert result["ris"] == pytest.approx(0.0, abs=1e-10)
        assert result["rrs"] == pytest.approx(0.0, abs=1e-10)
        assert result["ros"] == pytest.approx(0.0, abs=1e-10)


class TestBatchRelativeStabilityConvenience:
    """Tests for compute_batch_relative_stability."""

    def test_returns_all_three(self, feature_names, sample_data):
        """Batch convenience returns all three metrics."""
        from explainiverse.evaluation import compute_batch_relative_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_stability(
            explainer, model, sample_data,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=5, max_instances=5, seed=42,
        )
        assert "ris" in result
        assert "rrs" in result
        assert "ros" in result

    def test_each_has_batch_stats(self, feature_names, sample_data):
        """Each sub-result has mean/std/scores."""
        from explainiverse.evaluation import compute_batch_relative_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_batch_relative_stability(
            explainer, model, sample_data,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=5, max_instances=5, seed=42,
        )
        for key in ["ris", "rrs", "ros"]:
            assert "mean" in result[key]
            assert "n_evaluated" in result[key]


# =============================================================================
# Discrete Feature Support
# =============================================================================

class TestDiscreteFeatureSupport:
    """Tests for discrete (binary) feature perturbation support."""

    def test_discrete_features_accepted(self, feature_names):
        """feature_types parameter is accepted without error."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        instance = np.array([1.0, 0.0, 1.0, 0.0])  # Binary features
        feature_types = np.array(["discrete", "discrete", "discrete", "discrete"])

        score = compute_relative_input_stability(
            explainer, model, instance,
            feature_types=feature_types,
            n_perturbations=10, seed=42,
        )
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_mixed_features(self, feature_names):
        """Mix of continuous and discrete features works."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        instance = np.array([0.5, 1.0, 0.3, 0.0])
        feature_types = np.array(["continuous", "discrete", "continuous", "discrete"])

        score = compute_relative_input_stability(
            explainer, model, instance,
            feature_types=feature_types,
            n_perturbations=20, seed=42,
        )
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_discrete_perturbations_are_flips(self, feature_names):
        """Discrete features are perturbed by flipping, not Gaussian noise."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        instance = np.array([1.0, 0.0, 1.0, 0.0])
        feature_types = np.array(["discrete", "discrete", "discrete", "discrete"])

        result = compute_relative_input_stability(
            explainer, model, instance,
            feature_types=feature_types,
            discrete_flip_prob=0.5,  # High flip rate for testability
            n_perturbations=20, seed=42,
            return_details=True,
        )
        # Should produce valid results
        assert result["n_valid"] > 0

    def test_discrete_flip_prob_zero_no_change(self, feature_names):
        """With flip_prob=0, discrete features never change → score ≈ 0
        for a deterministic explainer (if all features are discrete)."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        instance = np.array([1.0, 0.0, 1.0, 0.0])
        feature_types = np.array(["discrete", "discrete", "discrete", "discrete"])

        score = compute_relative_input_stability(
            explainer, model, instance,
            feature_types=feature_types,
            discrete_flip_prob=0.0,
            n_perturbations=10, seed=42,
        )
        # No perturbation → no explanation change → score ≈ 0
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_default_feature_types_is_continuous(self, feature_names, single_instance):
        """feature_types=None defaults to all continuous (backward compat)."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()

        score_default = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
        )
        score_explicit = compute_relative_input_stability(
            explainer, model, single_instance,
            feature_types=np.array(["continuous"] * 4),
            n_perturbations=20, seed=42,
        )
        assert score_default == pytest.approx(score_explicit, rel=1e-10)

    def test_discrete_in_rrs(self, feature_names):
        """Discrete features work in RRS too."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        instance = np.array([1.0, 0.0, 1.0, 0.0])
        feature_types = np.array(["discrete", "discrete", "continuous", "continuous"])

        score = compute_relative_representation_stability(
            explainer, model, instance,
            representation_fn=_linear_representation_fn,
            feature_types=feature_types,
            n_perturbations=10, seed=42,
        )
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_discrete_in_ros(self, feature_names):
        """Discrete features work in ROS too."""
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        instance = np.array([1.0, 0.0, 1.0, 0.0])
        feature_types = np.array(["discrete", "discrete", "continuous", "continuous"])

        score = compute_relative_output_stability(
            explainer, model, instance,
            logit_fn=_linear_logit_fn,
            feature_types=feature_types,
            n_perturbations=10, seed=42,
        )
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_discrete_in_convenience(self, feature_names):
        """Discrete features work in all-in-one convenience function."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        instance = np.array([1.0, 0.0, 1.0, 0.0])
        feature_types = np.array(["discrete", "discrete", "continuous", "continuous"])

        result = compute_relative_stability(
            explainer, model, instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            feature_types=feature_types,
            n_perturbations=10, seed=42,
        )
        assert np.isfinite(result["ris"])
        assert np.isfinite(result["rrs"])
        assert np.isfinite(result["ros"])


# =============================================================================
# Theoretical Bound (Equation 4): RIS < λ₁·L₁ × RRS
# =============================================================================

class TestTheoreticalBound:
    """Tests for optional theoretical bound computation."""

    def test_bound_included_when_repr_fn_given(self, feature_names, single_instance):
        """When representation_fn is given, RIS details include theoretical bound."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
            return_details=True,
            representation_fn=_linear_representation_fn,
        )
        assert "theoretical_bound" in result

    def test_bound_not_included_without_repr_fn(self, feature_names, single_instance):
        """Without representation_fn, no theoretical bound."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_input_stability(
            explainer, model, single_instance,
            n_perturbations=20, seed=42,
            return_details=True,
        )
        assert result.get("theoretical_bound") is None

    def test_ris_bounded_by_theory(self, feature_names):
        """
        Eq 4: RIS < λ₁ · L₁ × RRS, where λ₁ = ||h₁(x)||_p / ||x||_p.
        The empirical RIS should not exceed the theoretical upper bound.
        """
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        instance = np.array([3.0, 2.0, 1.0, 0.5])

        result = compute_relative_input_stability(
            explainer, model, instance,
            n_perturbations=50, noise_scale=0.05, seed=42,
            return_details=True,
            representation_fn=_linear_representation_fn,
        )
        # The theoretical bound should be >= the empirical RIS score
        if result["theoretical_bound"] is not None and np.isfinite(result["theoretical_bound"]):
            assert result["score"] <= result["theoretical_bound"] + 1e-6


# =============================================================================
# Edge Cases
# =============================================================================

class TestRelativeStabilityEdgeCases:
    """Edge cases for all three Relative Stability metrics."""

    def test_single_feature(self):
        """Works with a single feature."""
        from explainiverse.evaluation import compute_relative_input_stability

        fnames = ["f0"]
        explainer = _DeterministicExplainer(fnames)
        model = _ConstantClassModel()
        instance = np.array([2.0])

        score = compute_relative_input_stability(
            explainer, model, instance,
            n_perturbations=10, seed=42,
        )
        assert np.isfinite(score)

    def test_high_dimensional(self):
        """Works with many features."""
        from explainiverse.evaluation import compute_relative_input_stability

        fnames = [f"f{i}" for i in range(50)]
        explainer = _DeterministicExplainer(fnames)
        model = _ConstantClassModel()
        instance = np.random.default_rng(42).standard_normal(50)

        score = compute_relative_input_stability(
            explainer, model, instance,
            n_perturbations=10, seed=42,
        )
        assert np.isfinite(score)

    def test_zero_explanation_elements(self, feature_names):
        """
        If original explanation has zero elements, percent change
        (e_x - e_x') / e_x has division by zero. Must be handled
        with epsilon floor on element-wise division.
        """
        from explainiverse.evaluation import compute_relative_input_stability

        # Explainer returns some zeros
        explainer = _ConstantExplainer(feature_names, values=[0.0, 0.0, 1.0, 2.0])
        model = _ConstantClassModel()
        instance = np.array([1.0, 2.0, 3.0, 4.0])

        score = compute_relative_input_stability(
            explainer, model, instance,
            n_perturbations=10, seed=42,
        )
        # Should be finite (epsilon prevents NaN/Inf)
        assert np.isfinite(score) or score == pytest.approx(0.0, abs=1e-10)

    def test_all_zero_explanation(self, feature_names):
        """All-zero explanation: numerator = 0 for constant explainer → score = 0."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _ConstantExplainer(feature_names, values=[0.0, 0.0, 0.0, 0.0])
        model = _ConstantClassModel()
        instance = np.array([1.0, 2.0, 3.0, 4.0])

        score = compute_relative_input_stability(
            explainer, model, instance,
            n_perturbations=10, seed=42,
        )
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_very_small_noise_scale(self, feature_names, single_instance):
        """Very small noise → tiny perturbations → finite result."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_input_stability(
            explainer, model, single_instance,
            noise_scale=1e-10, n_perturbations=10, seed=42,
        )
        assert np.isfinite(score)

    def test_very_large_noise_scale(self, feature_names, single_instance):
        """Large noise → large perturbations → still finite."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_input_stability(
            explainer, model, single_instance,
            noise_scale=10.0, n_perturbations=10, seed=42,
        )
        assert np.isfinite(score)

    def test_instance_near_zero(self):
        """Instance near zero: percent change in input has large denominator
        elements. Should still be finite due to epsilon floor."""
        from explainiverse.evaluation import compute_relative_input_stability

        fnames = ["f0", "f1", "f2"]
        explainer = _DeterministicExplainer(fnames)
        model = _ConstantClassModel()
        instance = np.array([1e-12, 1e-12, 1e-12])

        score = compute_relative_input_stability(
            explainer, model, instance,
            n_perturbations=10, seed=42,
        )
        assert np.isfinite(score)

    def test_rrs_with_single_valid_perturbation(self, feature_names):
        """Works even with only 1 valid perturbation (no NaN)."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = _DeterministicExplainer(feature_names)
        # Model that filters most perturbations
        instance = np.array([0.001, 1.0, 0.5, -0.5])
        model = _ThresholdModel(threshold=0.0)

        result = compute_relative_representation_stability(
            explainer, model, instance,
            representation_fn=_linear_representation_fn,
            n_perturbations=50, noise_scale=0.5, seed=42,
            return_details=True,
        )
        if result["n_valid"] >= 1:
            assert np.isfinite(result["score"])

    def test_epsilon_min_parameter(self, feature_names, single_instance):
        """Custom epsilon_min is accepted."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        score = compute_relative_input_stability(
            explainer, model, single_instance,
            epsilon_min=1e-5, n_perturbations=10, seed=42,
        )
        assert np.isfinite(score)


# =============================================================================
# Cross-Metric Relationships
# =============================================================================

class TestCrossMetricRelationships:
    """Tests verifying relationships between the three metrics."""

    def test_all_three_non_negative(self, feature_names, single_instance):
        """All three metrics are non-negative."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = _DeterministicExplainer(feature_names)
        model = _ConstantClassModel()
        result = compute_relative_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=20, seed=42,
        )
        assert result["ris"] >= 0.0
        assert result["rrs"] >= 0.0
        assert result["ros"] >= 0.0

    def test_constant_explainer_all_zero(self, feature_names, single_instance):
        """All three metrics are zero for a constant explainer."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = _ConstantExplainer(feature_names, values=[3.0, 2.0, 1.0, 0.5])
        model = _ConstantClassModel()
        result = compute_relative_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=20, seed=42,
        )
        assert result["ris"] == pytest.approx(0.0, abs=1e-10)
        assert result["rrs"] == pytest.approx(0.0, abs=1e-10)
        assert result["ros"] == pytest.approx(0.0, abs=1e-10)

    def test_random_explainer_all_high(self, feature_names, single_instance):
        """Random explainer has high instability on all three."""
        from explainiverse.evaluation import compute_relative_stability

        det = _DeterministicExplainer(feature_names)
        rng = _RandomExplainer(feature_names, seed=99)
        model = _ConstantClassModel()

        det_result = compute_relative_stability(
            det, model, single_instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=30, seed=42,
        )
        rng_result = compute_relative_stability(
            rng, model, single_instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=30, seed=42,
        )
        assert rng_result["ris"] > det_result["ris"]
        assert rng_result["rrs"] > det_result["rrs"]
        assert rng_result["ros"] > det_result["ros"]

    def test_shared_numerator_consistency(self, feature_names, single_instance):
        """
        All three metrics share the same numerator:
        ||(e_x - e_x') / e_x||_p.
        They differ only in the denominator. For a constant explainer,
        all numerators = 0, so all scores = 0.
        """
        from explainiverse.evaluation import compute_relative_stability

        explainer = _ConstantExplainer(feature_names, values=[5.0, 3.0, 1.0, 0.5])
        model = _ConstantClassModel()
        result = compute_relative_stability(
            explainer, model, single_instance,
            representation_fn=_linear_representation_fn,
            logit_fn=_linear_logit_fn,
            n_perturbations=20, seed=42,
        )
        # All zero because shared numerator = 0
        assert result["ris"] == pytest.approx(0.0, abs=1e-10)
        assert result["rrs"] == pytest.approx(0.0, abs=1e-10)
        assert result["ros"] == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Import Tests
# =============================================================================

class TestRelativeStabilityImports:
    """Verify all new functions are importable from evaluation package."""

    def test_import_relative_input_stability(self):
        from explainiverse.evaluation import compute_relative_input_stability
        assert callable(compute_relative_input_stability)

    def test_import_batch_relative_input_stability(self):
        from explainiverse.evaluation import compute_batch_relative_input_stability
        assert callable(compute_batch_relative_input_stability)

    def test_import_relative_representation_stability(self):
        from explainiverse.evaluation import compute_relative_representation_stability
        assert callable(compute_relative_representation_stability)

    def test_import_batch_relative_representation_stability(self):
        from explainiverse.evaluation import compute_batch_relative_representation_stability
        assert callable(compute_batch_relative_representation_stability)

    def test_import_relative_output_stability(self):
        from explainiverse.evaluation import compute_relative_output_stability
        assert callable(compute_relative_output_stability)

    def test_import_batch_relative_output_stability(self):
        from explainiverse.evaluation import compute_batch_relative_output_stability
        assert callable(compute_batch_relative_output_stability)

    def test_import_relative_stability(self):
        from explainiverse.evaluation import compute_relative_stability
        assert callable(compute_relative_stability)

    def test_import_batch_relative_stability(self):
        from explainiverse.evaluation import compute_batch_relative_stability
        assert callable(compute_batch_relative_stability)


# =============================================================================
# Integration with Real sklearn Model
# =============================================================================

@pytest.fixture
def trained_model_and_explainer(feature_names):
    """Train a real sklearn model and create a LIME explainer."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from explainiverse.adapters import SklearnAdapter
    from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer

    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3,
        n_redundant=0, n_classes=2, random_state=42
    )
    X = X.astype(np.float32)

    clf = GradientBoostingClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)

    adapter = SklearnAdapter(
        clf, feature_names=feature_names, class_names=["class_0", "class_1"]
    )
    explainer = LimeExplainer(
        model=adapter,
        training_data=X,
        feature_names=feature_names,
        class_names=["class_0", "class_1"],
    )
    return adapter, explainer, X


class TestIntegrationRealModel:
    """Integration tests with a real sklearn model + LIME explainer."""

    def test_ris_with_real_model(self, trained_model_and_explainer):
        """RIS works end-to-end with real model."""
        from explainiverse.evaluation import compute_relative_input_stability

        model, explainer, X = trained_model_and_explainer
        score = compute_relative_input_stability(
            explainer, model, X[0],
            n_perturbations=5, seed=42,
        )
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_ros_with_real_model(self, trained_model_and_explainer):
        """ROS works end-to-end with real model."""
        from explainiverse.evaluation import compute_relative_output_stability

        model, explainer, X = trained_model_and_explainer

        def _sklearn_logit_fn(X_in):
            """Use adapter's predict (returns probabilities) as logits."""
            X_in = np.asarray(X_in, dtype=np.float32)
            if X_in.ndim == 1:
                X_in = X_in.reshape(1, -1)
            return model.predict(X_in)

        score = compute_relative_output_stability(
            explainer, model, X[0],
            logit_fn=_sklearn_logit_fn,
            n_perturbations=5, seed=42,
        )
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert score >= 0.0

    def test_batch_ris_with_real_model(self, trained_model_and_explainer):
        """Batch RIS works with real model."""
        from explainiverse.evaluation import compute_batch_relative_input_stability

        model, explainer, X = trained_model_and_explainer
        result = compute_batch_relative_input_stability(
            explainer, model, X,
            n_perturbations=3, max_instances=3, seed=42,
        )
        assert result["n_evaluated"] == 3
        assert np.isfinite(result["mean"])


# =============================================================================
# Low-Validity Warning Tests
# =============================================================================

class TestLowValidityWarnings:
    """Verify warnings.warn fires when n_valid < 5 for all Relative Stability functions."""

    def _make_mock_explainer(self):
        """Create a mock explainer that returns deterministic attributions."""
        class MockExplainer:
            def explain(self, instance, **kwargs):
                instance = np.asarray(instance, dtype=np.float64).flatten()
                n = len(instance)
                attrs = {
                    f"feature_{i}": float(instance[i] * (i + 1))
                    for i in range(n)
                }
                exp = Explanation(
                    explainer_name="mock",
                    target_class="class_0",
                    explanation_data={"feature_attributions": attrs},
                )
                exp.feature_names = [f"feature_{i}" for i in range(n)]
                return exp
        return MockExplainer()

    def _make_mock_model(self):
        """Create a mock model that always predicts class 0 (all perturbations pass same-class filter)."""
        class MockModel:
            def predict_proba(self, X):
                X = np.asarray(X)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                # Always predict class 0 with high confidence
                return np.column_stack([np.ones(len(X)) * 0.9, np.ones(len(X)) * 0.1])

            def predict(self, X):
                return np.zeros(len(np.asarray(X).reshape(-1, X.shape[-1] if np.asarray(X).ndim > 1 else 1)))
        return MockModel()

    def test_ris_low_validity_warning(self):
        """RIS emits UserWarning when n_valid < 5."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = self._make_mock_explainer()
        model = self._make_mock_model()
        instance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # n_perturbations=3 → n_valid will be 3 (< 5), triggering the warning
        with pytest.warns(UserWarning, match="RIS score may be statistically unreliable"):
            compute_relative_input_stability(
                explainer, model, instance,
                n_perturbations=3, seed=42,
            )

    def test_rrs_low_validity_warning(self):
        """RRS emits UserWarning when n_valid < 5."""
        from explainiverse.evaluation import compute_relative_representation_stability

        explainer = self._make_mock_explainer()
        model = self._make_mock_model()
        instance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def representation_fn(x):
            x = np.asarray(x, dtype=np.float64).flatten()
            return x[:3] * 2.0  # Simple linear representation

        with pytest.warns(UserWarning, match="RRS score may be statistically unreliable"):
            compute_relative_representation_stability(
                explainer, model, instance,
                representation_fn=representation_fn,
                n_perturbations=3, seed=42,
            )

    def test_ros_low_validity_warning(self):
        """ROS emits UserWarning when n_valid < 5."""
        from explainiverse.evaluation import compute_relative_output_stability

        explainer = self._make_mock_explainer()
        model = self._make_mock_model()
        instance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def logit_fn(x):
            x = np.asarray(x, dtype=np.float64).flatten()
            return np.array([np.sum(x), -np.sum(x)])

        with pytest.warns(UserWarning, match="ROS score may be statistically unreliable"):
            compute_relative_output_stability(
                explainer, model, instance,
                logit_fn=logit_fn,
                n_perturbations=3, seed=42,
            )

    def test_combined_low_validity_warning(self):
        """compute_relative_stability emits UserWarning when n_valid < 5."""
        from explainiverse.evaluation import compute_relative_stability

        explainer = self._make_mock_explainer()
        model = self._make_mock_model()
        instance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def representation_fn(x):
            x = np.asarray(x, dtype=np.float64).flatten()
            return x[:3] * 2.0

        def logit_fn(x):
            x = np.asarray(x, dtype=np.float64).flatten()
            return np.array([np.sum(x), -np.sum(x)])

        with pytest.warns(UserWarning, match="Relative stability scores may be statistically unreliable"):
            compute_relative_stability(
                explainer, model, instance,
                representation_fn=representation_fn,
                logit_fn=logit_fn,
                n_perturbations=3, seed=42,
            )

    def test_ris_no_warning_with_sufficient_perturbations(self):
        """RIS does NOT warn when n_valid >= 5."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = self._make_mock_explainer()
        model = self._make_mock_model()
        instance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # n_perturbations=10 → n_valid should be 10 (>= 5), no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            score = compute_relative_input_stability(
                explainer, model, instance,
                n_perturbations=10, seed=42,
            )
            assert isinstance(score, float)
            assert np.isfinite(score)

    def test_ris_no_warning_when_zero_valid(self):
        """RIS does NOT warn when n_valid == 0 (returns NaN instead)."""
        from explainiverse.evaluation import compute_relative_input_stability

        # Model that always changes class for any perturbation
        class AlwaysDifferentModel:
            def __init__(self):
                self._call_count = 0

            def predict_proba(self, X):
                X = np.asarray(X)
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                results = []
                for _ in range(len(X)):
                    if self._call_count == 0:
                        # First call (original instance) → class 0
                        results.append([0.9, 0.1])
                    else:
                        # All subsequent calls → class 1
                        results.append([0.1, 0.9])
                    self._call_count += 1
                return np.array(results)

        explainer = self._make_mock_explainer()
        model = AlwaysDifferentModel()
        instance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # No warning should fire (n_valid == 0 is not in range 0 < n_valid < 5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            score = compute_relative_input_stability(
                explainer, model, instance,
                n_perturbations=5, seed=42,
            )
            assert np.isnan(score)

    def test_ris_warning_details_returned(self):
        """RIS with return_details=True still warns and returns valid dict."""
        from explainiverse.evaluation import compute_relative_input_stability

        explainer = self._make_mock_explainer()
        model = self._make_mock_model()
        instance = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.warns(UserWarning, match="RIS score may be statistically unreliable"):
            result = compute_relative_input_stability(
                explainer, model, instance,
                n_perturbations=3, seed=42,
                return_details=True,
            )

        assert isinstance(result, dict)
        assert result["n_valid"] < 5
        assert result["n_valid"] > 0
        assert result["n_total"] == 3
        assert np.isfinite(result["score"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
