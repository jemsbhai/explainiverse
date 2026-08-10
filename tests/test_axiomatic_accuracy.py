"""Reference-contract tests for axiomatic attribution diagnostics.

The formulas exercised here follow:

* Sundararajan, Taly & Yan (2017), completeness and symmetry-preserving
  preconditions: https://proceedings.mlr.press/v70/sundararajan17a.html
* Kindermans et al. (2017), compensated constant input shifts:
  https://arxiv.org/abs/1711.00867
* Nguyen & Martinez (2020), Non-Sensitivity ``|A0 symmetric_difference X0|``:
  https://arxiv.org/abs/2007.07584
"""

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.axiomatic import (
    _detect_non_sensitive_features,
    _safe_model_output,
    compute_batch_completeness,
    compute_batch_input_invariance,
    compute_batch_non_sensitivity,
    compute_batch_symmetry,
    compute_completeness,
    compute_completeness_score,
    compute_input_invariance,
    compute_input_invariance_pytorch,
    compute_non_sensitivity,
    compute_symmetry,
)


def _scalar_model(x):
    return float(np.dot(np.array([2.0, -1.0, 0.0]), np.asarray(x)))


class _TargetAwareExplainer:
    def explain(self, instance, target=None, baseline=None):
        if target != "selected-output":
            raise AssertionError("target was not forwarded")
        x = np.asarray(instance, dtype=float)
        reference = np.asarray(baseline, dtype=float)
        attrs = np.array([2.0, -1.0, 0.0]) * (x - reference)
        names = ["a", "b", "c"]
        return Explanation(
            explainer_name="target-aware",
            target_class=target,
            explanation_data={"feature_attributions": dict(zip(names, attrs, strict=True))},
            feature_names=names,
        )


class TestCompletenessReferenceContract:
    def test_exact_nonzero_baseline_identity(self):
        x = np.array([3.0, 4.0, 5.0])
        baseline = np.array([1.0, -2.0, 7.0])
        attrs = np.array([2.0, -1.0, 0.0]) * (x - baseline)
        assert compute_completeness(attrs, _scalar_model, x, baseline) == pytest.approx(0.0)

    def test_wrong_baseline_is_detected_as_residual(self):
        x = np.array([3.0, 4.0, 5.0])
        true_baseline = np.array([1.0, -2.0, 7.0])
        attrs = np.array([2.0, -1.0, 0.0]) * (x - true_baseline)
        residual = compute_completeness(attrs, _scalar_model, x, baseline=0.0)
        assert residual > 0.0

    def test_multioutput_is_never_silently_indexed(self):
        with pytest.raises(ValueError, match="Select the target/output explicitly"):
            compute_completeness(np.zeros(3), lambda x: np.array([1.0, 2.0]), np.ones(3))

    def test_output_selector_receives_raw_multioutput(self):
        x = np.array([3.0, 4.0, 5.0])
        baseline = np.array([1.0, -2.0, 7.0])
        attrs = np.array([2.0, -1.0, 0.0]) * (x - baseline)

        def vector_model(value):
            score = _scalar_model(value)
            return np.array([99.0, score])

        score = compute_completeness(
            attrs,
            vector_model,
            x,
            baseline,
            output_func=lambda output: output[1],
        )
        assert score == pytest.approx(0.0)

    def test_high_level_forwards_target_and_baseline(self):
        x = np.array([3.0, 4.0, 5.0])
        baseline = np.array([1.0, -2.0, 7.0])
        score = compute_completeness_score(
            _TargetAwareExplainer(),
            _scalar_model,
            x,
            baseline,
            explain_kwargs={"target": "selected-output", "baseline": baseline},
        )
        assert score == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "attributions,instance",
        [
            (np.array([0.0, np.nan]), np.ones(2)),
            (np.zeros(2), np.array([1.0, np.inf])),
            (np.zeros((1, 2)), np.ones(2)),
        ],
    )
    def test_rejects_nonfinite_or_ambiguous_vectors(self, attributions, instance):
        with pytest.raises(ValueError):
            compute_completeness(attributions, lambda x: 0.0, instance)

    def test_rejects_nonfinite_scalar_output(self):
        with pytest.raises(ValueError, match="must be finite"):
            compute_completeness(np.zeros(3), lambda x: np.nan, np.ones(3))

    def test_explanation_missing_feature_is_not_filled_with_zero(self):
        class MissingFeatureExplainer:
            def explain(self, instance):
                return Explanation(
                    "bad",
                    "target",
                    {"feature_attributions": {"a": 1.0, "b": 2.0}},
                    feature_names=["a", "b", "c"],
                )

        with pytest.raises(ValueError, match="match feature_names exactly"):
            compute_completeness_score(MissingFeatureExplainer(), lambda x: 0.0, np.ones(3))

    def test_stochastic_explainer_is_rejected_by_default(self):
        class StochasticExplainer:
            def __init__(self):
                self.rng = np.random.RandomState(11)

            def explain(self, instance):
                names = ["a", "b", "c"]
                values = self.rng.normal(size=3)
                return Explanation(
                    "stochastic",
                    "target",
                    {"feature_attributions": dict(zip(names, values, strict=True))},
                    feature_names=names,
                )

        with pytest.raises(RuntimeError, match="stochastic"):
            compute_completeness_score(StochasticExplainer(), lambda x: 0.0, np.ones(3))

    def test_batch_failure_is_not_silently_omitted(self):
        with pytest.raises(ValueError, match="finite"):
            compute_batch_completeness(
                attributions_list=[np.zeros(3), np.array([0.0, np.nan, 0.0])],
                model_fn=lambda x: 0.0,
                X=np.ones((2, 3)),
            )

    def test_safe_output_rejects_class_vector(self):
        with pytest.raises(ValueError, match="exactly one scalar"):
            _safe_model_output(lambda x: np.array([0.2, 0.8]), np.ones(3))


class TestNonSensitivityReferenceContract:
    def test_primary_formula_counts_both_mismatch_directions(self):
        # A0={0,2}; X0={1,2}; symmetric difference={0,1}.
        attrs = np.array([0.0, 1.0, 0.0, 2.0])
        x0 = np.array([False, True, True, False])
        score = compute_non_sensitivity(attrs, lambda x: 0.0, np.ones(4), x0)
        assert score == 2.0

    def test_normalized_formula_divides_cardinality_by_feature_count(self):
        attrs = np.array([0.0, 1.0, 0.0, 2.0])
        x0 = np.array([False, True, True, False])
        score = compute_non_sensitivity(attrs, lambda x: 0.0, np.ones(4), x0, normalize=True)
        assert score == 0.5

    def test_false_zero_on_sensitive_feature_is_a_violation(self):
        attrs = np.array([0.0, 1.0])
        score = compute_non_sensitivity(
            attrs,
            lambda x: float(np.sum(x)),
            np.ones(2),
            np.array([False, False]),
        )
        assert score == 1.0

    def test_attribution_zero_tolerance_is_explicit(self):
        attrs = np.array([1e-8, 1.0])
        reference = np.array([True, False])
        exact = compute_non_sensitivity(attrs, lambda x: 0.0, np.ones(2), reference)
        tolerant = compute_non_sensitivity(
            attrs,
            lambda x: 0.0,
            np.ones(2),
            reference,
            attribution_tolerance=1e-7,
        )
        assert exact == 1.0
        assert tolerant == 0.0

    def test_reference_mask_is_required(self):
        with pytest.raises(ValueError, match="must be supplied"):
            compute_non_sensitivity(np.zeros(2), lambda x: 0.0, np.ones(2))

    def test_reference_mask_must_be_boolean(self):
        with pytest.raises(TypeError, match="boolean mask"):
            compute_non_sensitivity(np.zeros(2), lambda x: 0.0, np.ones(2), np.array([0, 1]))

    def test_local_proxy_does_not_prove_global_independence(self):
        # This function is flat throughout the sampled neighbourhood but does
        # depend on x0 outside it. The helper may label x0 locally inactive,
        # illustrating why its mask is not accepted implicitly as X0.
        def plateau_model(x):
            return float(abs(x[0]) >= 100.0)

        proxy = _detect_non_sensitive_features(
            plateau_model,
            np.zeros(2),
            n_perturbations=20,
            perturbation_scale=0.01,
            seed=7,
        )
        assert proxy[0]
        assert plateau_model(np.array([101.0, 0.0])) != plateau_model(np.zeros(2))

    def test_batch_failure_is_not_silently_omitted(self):
        with pytest.raises(ValueError, match="finite"):
            compute_batch_non_sensitivity(
                attributions_list=[np.zeros(2), np.array([np.nan, 0.0])],
                model_fn=lambda x: 0.0,
                X=np.ones((2, 2)),
                non_sensitive_features=np.ones(2, dtype=bool),
            )


class TestInputShiftContracts:
    def test_uncompensated_api_warns_that_it_is_only_a_proxy(self):
        with pytest.warns(RuntimeWarning, match="not a test"):
            score = compute_input_invariance(lambda x: np.asarray(x), np.ones(3), shift=1.0)
        assert score == pytest.approx(1.0)

    def test_uncompensated_api_rejects_stochastic_explanations(self):
        rng = np.random.RandomState(19)
        with pytest.warns(RuntimeWarning):
            with pytest.raises(RuntimeError, match="stochastic"):
                compute_input_invariance(lambda x: rng.normal(size=len(x)), np.ones(3), shift=1.0)

    def test_batch_uses_one_constant_shift_for_every_sample(self):
        with pytest.warns(RuntimeWarning, match="uncompensated"):
            result = compute_batch_input_invariance(
                lambda x: np.asarray(x), np.ones((4, 5)), seed=91
            )
        np.testing.assert_allclose(result["scores"], result["scores"][0])

    def test_local_randomness_does_not_mutate_numpy_global_rng(self):
        np.random.seed(1234)
        expected = np.random.random(4)
        np.random.seed(1234)
        with pytest.warns(RuntimeWarning):
            compute_input_invariance(lambda x: np.asarray(x), np.ones(3), seed=8)
        actual = np.random.random(4)
        np.testing.assert_array_equal(actual, expected)

    def test_zero_shift_is_vacuous_and_rejected(self):
        with pytest.warns(RuntimeWarning):
            with pytest.raises(ValueError, match="non-zero"):
                compute_input_invariance(lambda x: np.asarray(x), np.ones(3), shift=0.0)

    def test_pytorch_compensation_preserves_gradient_explanation(self):
        torch = pytest.importorskip("torch")
        nn = torch.nn
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)).eval()

        def gradient(explained_model, value):
            tensor = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
            tensor.requires_grad_(True)
            output = explained_model(tensor).sum()
            return torch.autograd.grad(output, tensor)[0].detach().numpy()[0]

        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        score = compute_input_invariance_pytorch(
            model, gradient, np.array([0.2, -0.7, 1.3]), shift=0.4
        )
        assert score == pytest.approx(0.0, abs=1e-6)
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter, before[name])

    def test_pytorch_compensation_exposes_gradient_times_input_shift(self):
        torch = pytest.importorskip("torch")
        nn = torch.nn
        model = nn.Sequential(nn.Linear(3, 1, bias=False)).eval()
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1.0, -2.0, 0.5]]))

        def gradient_times_input(explained_model, value):
            tensor = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
            tensor.requires_grad_(True)
            output = explained_model(tensor).sum()
            gradient = torch.autograd.grad(output, tensor)[0]
            return (gradient * tensor).detach().numpy()[0]

        score = compute_input_invariance_pytorch(
            model, gradient_times_input, np.array([1.0, 2.0, 3.0]), shift=0.5
        )
        assert score > 0.0

    def test_pytorch_rejects_training_mode(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Sequential(torch.nn.Linear(2, 1)).train()
        with pytest.raises(ValueError, match="evaluation mode"):
            compute_input_invariance_pytorch(
                model, lambda model, x: np.zeros(2), np.ones(2), shift=1.0
            )

    def test_pytorch_rejects_unproven_arbitrary_model_structure(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(2, 1).eval()
        with pytest.raises(TypeError, match="nn.Sequential"):
            compute_input_invariance_pytorch(
                model, lambda model, x: np.zeros(2), np.ones(2), shift=1.0
            )

    def test_pytorch_rejects_wrong_attribution_shape(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Sequential(torch.nn.Linear(2, 1)).eval()
        with pytest.raises(ValueError, match="expected length"):
            compute_input_invariance_pytorch(
                model, lambda model, x: np.zeros(1), np.ones(2), shift=1.0
            )


class TestSymmetryPreconditions:
    def test_equal_input_and_baseline_values_enable_conditional_check(self):
        score = compute_symmetry(
            np.array([0.4, 0.4, 1.0]),
            [(0, 1)],
            instance=np.array([2.0, 2.0, 3.0]),
            baseline=np.array([0.0, 0.0, 0.0]),
        )
        assert score == 0.0

    def test_unequal_input_values_violate_axiom_precondition(self):
        with pytest.raises(ValueError, match="instance values"):
            compute_symmetry(
                np.array([0.4, 0.4]),
                [(0, 1)],
                instance=np.array([1.0, 2.0]),
                baseline=np.zeros(2),
            )

    def test_unequal_baseline_values_violate_axiom_precondition(self):
        with pytest.raises(ValueError, match="baseline values"):
            compute_symmetry(
                np.array([0.4, 0.4]),
                [(0, 1)],
                instance=np.array([2.0, 2.0]),
                baseline=np.array([0.0, 1.0]),
            )

    def test_pair_disparity_does_not_require_unavailable_model_claim(self):
        # Omitting input/baseline intentionally requests only the documented
        # caller-certified disparity diagnostic.
        assert compute_symmetry(np.array([0.2, 0.7]), [(0, 1)]) == pytest.approx(0.5)

    def test_duplicate_or_self_pairs_are_vacuous_and_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            compute_symmetry(np.ones(3), [(0, 1), (1, 0)])
        with pytest.raises(ValueError, match="distinct"):
            compute_symmetry(np.ones(3), [(1, 1)])

    def test_batch_precomputed_attributions_can_validate_preconditions(self):
        result = compute_batch_symmetry(
            [(0, 1)],
            attributions_list=[np.array([0.5, 0.5]), np.array([0.2, 0.2])],
            X=np.array([[1.0, 1.0], [3.0, 3.0]]),
            baseline=np.zeros(2),
        )
        assert result["n_evaluated"] == 2
        assert result["mean"] == 0.0

    def test_batch_precondition_failure_is_not_silently_omitted(self):
        with pytest.raises(ValueError, match="instance values"):
            compute_batch_symmetry(
                [(0, 1)],
                attributions_list=[np.zeros(2), np.zeros(2)],
                X=np.array([[1.0, 1.0], [1.0, 2.0]]),
                baseline=np.zeros(2),
            )
