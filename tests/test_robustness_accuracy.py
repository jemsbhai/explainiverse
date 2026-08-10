"""Formula- and contract-level checks for robustness/stability metrics."""

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation import robustness
from explainiverse.evaluation.robustness import (
    _element_wise_percent_change,
    _extract_attribution_vector,
    _generate_mixed_perturbations,
    _get_predicted_class,
    compute_batch_relative_input_stability,
    compute_consistency,
    compute_max_sensitivity,
    compute_relative_input_stability,
    compute_relative_output_stability,
    compute_relative_representation_stability,
)
from explainiverse.evaluation.stability import compute_lipschitz_estimate


class _LinearExplainer:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.feature_names = ["f0", "f1"]
        self.targets = []

    def explain(self, instance, target_class=None):
        values = self.scale * np.asarray(instance, dtype=float)
        self.targets.append(target_class)
        return Explanation(
            explainer_name="linear",
            target_class="fixed" if target_class is None else target_class,
            explanation_data={"feature_attributions": dict(zip(self.feature_names, values))},
            feature_names=self.feature_names,
        )


class _ConstantClassModel:
    def predict_proba(self, X):
        X = np.asarray(X)
        return np.tile([0.8, 0.2], (len(X), 1))


class _ProbabilityPredictAdapter:
    """Mimic Explainiverse adapters: predict returns a probability matrix."""

    task = "classification"

    def predict(self, X):
        X = np.asarray(X)
        positive = (X[:, 0] > 0).astype(float)
        return np.column_stack([1.0 - positive, positive])


def test_attribution_vector_requires_exact_declared_feature_mapping():
    explanation = Explanation(
        explainer_name="invalid",
        target_class=0,
        explanation_data={"feature_attributions": {"f0": 1.0, "extra": 2.0}},
        feature_names=["f0"],
    )
    with pytest.raises(ValueError, match="map exactly"):
        _extract_attribution_vector(explanation)


def test_indexed_attributions_without_names_are_ordered_without_mutation():
    explanation = Explanation(
        explainer_name="indexed",
        target_class=0,
        explanation_data={"feature_attributions": {"feature_1": 2.0, "feature_0": 1.0}},
    )

    assert _extract_attribution_vector(explanation) == pytest.approx([1.0, 2.0])
    assert explanation.feature_names is None


def test_unidentified_attribution_order_is_rejected():
    explanation = Explanation(
        explainer_name="unidentified",
        target_class=0,
        explanation_data={"feature_attributions": {"first": 1.0, "second": 2.0}},
    )
    with pytest.raises(ValueError, match="feature_names"):
        _extract_attribution_vector(explanation)


def _patch_perturbations(monkeypatch, values):
    fixed = np.asarray(values, dtype=float)

    def generate(instance, n_perturbations, noise_scale, rng, **kwargs):
        assert n_perturbations == len(fixed)
        return fixed.copy()

    monkeypatch.setattr(robustness, "_generate_mixed_perturbations", generate)


def test_predict_only_adapter_uses_argmax_not_first_probability_cast():
    model = _ProbabilityPredictAdapter()
    assert _get_predicted_class(model, np.array([-1.0, 0.0])) == 0
    assert _get_predicted_class(model, np.array([1.0, 0.0])) == 1


def test_known_regression_model_is_rejected_by_class_conditioned_metrics():
    class RegressionAdapter:
        task = "regression"

        def predict(self, X):
            return np.asarray(X)[:, :1]

    with pytest.raises(ValueError, match="categorical model predictions"):
        _get_predicted_class(RegressionAdapter(), np.array([1.0, 0.0]))


def test_ambiguous_one_column_binary_probability_is_rejected():
    class AmbiguousModel:
        def predict_proba(self, X):
            return np.full((len(X), 1), 0.8)

    with pytest.raises(ValueError, match="one-column probabilistic output is ambiguous"):
        _get_predicted_class(AmbiguousModel(), np.array([1.0, 0.0]))


def test_ris_same_class_filter_works_for_adapter_probability_output(monkeypatch):
    _patch_perturbations(monkeypatch, [[0.5, 1.0], [-1.0, 1.0]])
    details = compute_relative_input_stability(
        _LinearExplainer(),
        _ProbabilityPredictAdapter(),
        np.array([1.0, 1.0]),
        n_perturbations=2,
        return_details=True,
    )
    assert details["n_valid"] == 1


def test_batch_reports_instances_with_undefined_same_class_estimate(monkeypatch):
    _patch_perturbations(monkeypatch, [[-1.0, 1.0]])
    result = compute_batch_relative_input_stability(
        _LinearExplainer(),
        _ProbabilityPredictAdapter(),
        np.array([[1.0, 1.0]]),
        n_perturbations=1,
    )
    assert result["n_attempted"] == 1
    assert result["n_evaluated"] == 0
    assert result["n_undefined"] == 1


def test_relative_objectives_match_official_quantus(monkeypatch):
    quantus = pytest.importorskip("quantus")
    x = np.array([2.0, 4.0])
    x_prime = np.array([2.2, 3.0])
    _patch_perturbations(monkeypatch, [x_prime])
    explainer = _LinearExplainer(scale=2.0)
    model = _ConstantClassModel()

    e_x = (2.0 * x)[None, :]
    e_prime = (2.0 * x_prime)[None, :]

    ris_reference = quantus.RelativeInputStability(
        nr_samples=1, disable_warnings=True
    ).relative_input_stability_objective(x[None, :], x_prime[None, :], e_x, e_prime)[0]
    ris = compute_relative_input_stability(explainer, model, x, n_perturbations=1, epsilon_min=1e-6)
    assert ris == pytest.approx(ris_reference)

    def representation(value):
        return np.asarray(value, dtype=float) * np.array([3.0, 0.5])

    l_x = representation(x)[None, :]
    l_prime = representation(x_prime)[None, :]
    rrs_reference = quantus.RelativeRepresentationStability(
        nr_samples=1, disable_warnings=True
    ).relative_representation_stability_objective(l_x, l_prime, e_x, e_prime)[0]
    rrs = compute_relative_representation_stability(
        explainer,
        model,
        x,
        representation_fn=representation,
        n_perturbations=1,
        epsilon_min=1e-6,
    )
    assert rrs == pytest.approx(rrs_reference)

    def logits(value):
        value = np.asarray(value, dtype=float)
        return np.array([value[0] - 1.0, 2.0 * value[1]])

    h_x = logits(x)[None, :]
    h_prime = logits(x_prime)[None, :]
    ros_reference = quantus.RelativeOutputStability(
        nr_samples=1, disable_warnings=True
    ).relative_output_stability_objective(h_x, h_prime, e_x, e_prime)[0]
    ros = compute_relative_output_stability(
        explainer,
        model,
        x,
        logit_fn=logits,
        n_perturbations=1,
        epsilon_min=1e-6,
    )
    assert ros == pytest.approx(ros_reference)


def test_percent_change_regularises_exact_zeros_only_like_reference():
    result = _element_wise_percent_change(
        np.array([0.0, 1e-12]),
        np.array([1.0, 0.0]),
        epsilon_min=1e-6,
    )
    assert result == pytest.approx(np.array([-1e6, 1.0]))


def test_discrete_sampler_matches_paper_bernoulli_replacement():
    instance = np.array([1.0, 0.0, 1.0])
    feature_types = np.array(["discrete", "discrete", "discrete"])
    expected_rng = np.random.default_rng(7)
    expected = expected_rng.binomial(1, 0.2, size=(5, 3))
    actual = _generate_mixed_perturbations(
        instance,
        5,
        0.05,
        np.random.default_rng(7),
        feature_types=feature_types,
        discrete_flip_prob=0.2,
    )
    assert np.array_equal(actual, expected)


def test_discrete_sampler_rejects_nonbinary_feature_domain():
    with pytest.raises(ValueError, match="must be binary 0/1"):
        _generate_mixed_perturbations(
            np.array([2.0]),
            2,
            0.05,
            np.random.default_rng(1),
            feature_types=np.array(["discrete"]),
        )


def test_details_disclose_discrete_perturbation_contract(monkeypatch):
    _patch_perturbations(monkeypatch, [[0.0, 1.0]])
    result = compute_relative_input_stability(
        _LinearExplainer(),
        _ConstantClassModel(),
        np.array([1.0, 1.0]),
        n_perturbations=1,
        feature_types=np.array(["discrete", "discrete"]),
        return_details=True,
    )
    assert result["discrete_perturbation_contract"] == "paper_text_bernoulli_replacement"


def test_max_sensitivity_holds_target_fixed_and_preserves_global_rng():
    explainer = _LinearExplainer()
    np.random.seed(2024)
    before = np.random.get_state()
    compute_max_sensitivity(
        explainer,
        np.array([1.0, 2.0]),
        n_samples=5,
        seed=9,
        target_class=1,
    )
    after = np.random.get_state()
    assert explainer.targets == [1] * 6
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_explicit_target_fails_if_explainer_ignores_it():
    class IgnoringTargetExplainer(_LinearExplainer):
        def explain(self, instance, target_class=None):
            explanation = super().explain(instance, target_class=target_class)
            explanation.target_class = 0
            return explanation

    with pytest.raises(ValueError, match="did not honor the requested target_class"):
        compute_max_sensitivity(
            IgnoringTargetExplainer(),
            np.array([1.0, 2.0]),
            n_samples=1,
            seed=9,
            target_class=1,
        )


def test_noncanonical_normalization_does_not_switch_off_for_tiny_nonzero_norm():
    explainer = _LinearExplainer(scale=1e-14)
    instance = np.array([1.0, 2.0])
    absolute = compute_max_sensitivity(
        explainer,
        instance,
        n_samples=5,
        seed=3,
        normalize=False,
    )
    relative = compute_max_sensitivity(
        explainer,
        instance,
        n_samples=5,
        seed=3,
        normalize=True,
    )
    expected = absolute / np.linalg.norm(1e-14 * instance)
    assert relative == pytest.approx(expected)


def test_relative_stability_holds_the_explained_target_fixed(monkeypatch):
    _patch_perturbations(monkeypatch, [[1.1, 2.0], [0.9, 2.0]])
    explainer = _LinearExplainer()
    compute_relative_input_stability(
        explainer,
        _ConstantClassModel(),
        np.array([1.0, 2.0]),
        n_perturbations=2,
        target_class=1,
    )
    assert explainer.targets == [1, 1, 1]


def test_target_drift_from_none_is_not_silently_accepted(monkeypatch):
    class DriftingTargetExplainer(_LinearExplainer):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def explain(self, instance, target_class=None):
            explanation = super().explain(instance, target_class=target_class)
            explanation.target_class = None if self.calls == 0 else 1
            self.calls += 1
            return explanation

    _patch_perturbations(monkeypatch, [[1.1, 2.0]])
    with pytest.raises(ValueError, match="changed target_class"):
        compute_relative_input_stability(
            DriftingTargetExplainer(),
            _ConstantClassModel(),
            np.array([1.0, 2.0]),
            n_perturbations=1,
        )


def test_local_lipschitz_uses_fixed_anchor_and_is_exact_for_linear_map():
    score = compute_lipschitz_estimate(
        _LinearExplainer(scale=3.0),
        np.array([1.0, 2.0]),
        n_samples=20,
        radius=0.2,
        seed=4,
    )
    assert score == pytest.approx(3.0)


class _GroupedExplainer:
    feature_names = ["f0", "f1", "label"]

    def explain(self, instance):
        group = int(instance[0])
        values = [10.0, 0.0, 0.0] if group == 0 else [0.0, 10.0, 0.0]
        return Explanation(
            "grouped",
            "fixed",
            {"feature_attributions": dict(zip(self.feature_names, values))},
            feature_names=self.feature_names,
        )


class _EncodedLabelModel:
    def predict(self, X):
        return np.asarray(X)[:, 2]


def test_global_consistency_is_query_weighted_not_pair_pooled():
    # Group A (size 2): labels 0,1 -> local scores 0,0.
    # Group B (size 4): labels 0,0,0,1 -> local scores 2/3,2/3,2/3,0.
    # Global consistency is the mean over six query-local scores = 1/3.
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    score = compute_consistency(_GroupedExplainer(), _EncodedLabelModel(), X, top_k=1)
    assert score == pytest.approx(1.0 / 3.0)
    assert score != pytest.approx(3.0 / 7.0)  # old pair-pooled result


def test_consistency_rejects_noninteger_discretisation_size():
    X = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(TypeError, match="top_k must be an integer"):
        compute_consistency(_GroupedExplainer(), _EncodedLabelModel(), X, top_k=1.5)


def test_batch_and_single_metric_errors_are_not_silently_dropped(monkeypatch):
    class FailingExplainer(_LinearExplainer):
        def explain(self, instance, target_class=None):
            if np.asarray(instance)[0] < 0:
                raise RuntimeError("intentional explainer failure")
            return super().explain(instance, target_class=target_class)

    _patch_perturbations(monkeypatch, [[-1.0, 1.0]])
    with pytest.raises(RuntimeError, match="intentional explainer failure"):
        compute_relative_input_stability(
            FailingExplainer(),
            _ConstantClassModel(),
            np.array([1.0, 1.0]),
            n_perturbations=1,
        )
    with pytest.raises(RuntimeError, match="intentional explainer failure"):
        compute_batch_relative_input_stability(
            FailingExplainer(),
            _ConstantClassModel(),
            np.array([[1.0, 1.0]]),
            n_perturbations=1,
        )


def test_empirical_rhs_is_not_exposed_as_theoretical_guarantee(monkeypatch):
    _patch_perturbations(monkeypatch, [[1.1, 1.9]])
    result = compute_relative_input_stability(
        _LinearExplainer(),
        _ConstantClassModel(),
        np.array([1.0, 2.0]),
        n_perturbations=1,
        representation_fn=lambda value: np.asarray(value) * 2.0,
        return_details=True,
    )
    assert result["empirical_bound_estimate"] is not None
    assert result["theoretical_bound"] is None
