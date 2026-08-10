"""Analytical and state-contract tests for ProtoDash."""

import numpy as np
import pytest

from explainiverse.explainers.example_based.protodash import ProtoDashExplainer


def test_first_selection_uses_canonical_maximum_gradient_mu():
    # Algorithm 2 initializes w=0 and selects the largest gradient component,
    # so gains are exactly mu. The kernel diagonals affect the optimized weight
    # after support selection, not the first selected candidate.
    reference = np.array([[10.0, 0.0], [1.0, 1.0]])
    target = np.array([1.0, 1.0])
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="linear")

    result = explainer.explain(target, reference)
    data = result.explanation_data

    assert data["prototype_indices"] == [0]
    assert data["objective_weights"] == pytest.approx([0.1])
    assert data["weights"] == pytest.approx([1.0])


def test_orthogonal_linear_problem_has_analytical_weights():
    reference = np.eye(3)
    target = np.array([2.0, 1.0, 0.0])
    explainer = ProtoDashExplainer(n_prototypes=2, kernel="linear")

    result = explainer.explain(target, reference)
    data = result.explanation_data

    assert data["prototype_indices"] == [0, 1]
    assert data["objective_weights"] == pytest.approx([2.0, 1.0], abs=1e-7)
    assert data["weights"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0], abs=1e-7)
    assert data["objective_weight_semantics"] == "unnormalized_protodash_weights"


def test_auto_rbf_width_is_per_call_and_does_not_leak_state():
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="rbf", random_state=7)
    narrow = explainer.find_prototypes(np.array([[0.0], [1.0]]))
    wide = explainer.find_prototypes(np.array([[0.0], [100.0]]))

    assert narrow.explanation_data["kernel_width"] == pytest.approx(1.0)
    assert wide.explanation_data["kernel_width"] == pytest.approx(100.0)
    assert explainer.kernel_width is None


def test_class_conditional_mode_represents_the_requested_class_only():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["low", "low", "high", "high"])
    conditional = ProtoDashExplainer(
        n_prototypes=2, kernel="rbf", kernel_width=2.0
    ).find_prototypes(X, y=y, target_class="high", return_mmd=True)
    direct = ProtoDashExplainer(n_prototypes=2, kernel="rbf", kernel_width=2.0).find_prototypes(
        X[y == "high"], return_mmd=True
    )

    conditional_data = conditional.explanation_data
    direct_data = direct.explanation_data
    assert conditional_data["target_distribution"] == "requested_class"
    assert conditional_data["target_distribution_size"] == 2
    assert conditional_data["candidate_count"] == 2
    assert conditional_data["prototype_indices"] == [
        [2, 3][index] for index in direct_data["prototype_indices"]
    ]
    assert conditional_data["objective_weights"] == pytest.approx(direct_data["objective_weights"])
    assert conditional_data["mmd_score"] == pytest.approx(direct_data["mmd_score"])


def test_prediction_augmented_similarity_is_reported_in_same_space():
    class OneColumnModel:
        def predict(self, X):
            X = np.asarray(X)
            return 2.0 * X[:, :1]

    reference = np.array([[1.0], [2.0], [3.0]])
    target = np.array([2.5])
    explainer = ProtoDashExplainer(model=OneColumnModel(), n_prototypes=2, kernel="linear")

    result = explainer.explain(target, reference, use_predictions=True, return_similarity=True)
    data = result.explanation_data
    selected = np.asarray(data["prototype_indices"])
    target_augmented = np.array([2.5, 5.0])
    reference_augmented = np.column_stack((reference[:, 0], 2.0 * reference[:, 0]))
    expected = target_augmented @ reference_augmented[selected].T

    assert data["kernel_input_space"] == "input_plus_model_prediction"
    assert data["similarity_space"] == "input_plus_model_prediction"
    assert data["similarity_scores"] == pytest.approx(expected.tolist())


def test_criticism_helper_does_not_claim_mmd_critic():
    X = np.array([[0.0], [1.0], [2.0], [10.0]])
    explainer = ProtoDashExplainer(n_prototypes=1, kernel="rbf", kernel_width=1.0)
    result = explainer.find_criticisms(X, [1], n_criticisms=2)

    assert result.explanation_data["algorithm"] == "kernel_witness_ranking"
    assert result.explanation_data["is_mmd_critic_implementation"] is False


def test_target_class_requires_aligned_labels():
    explainer = ProtoDashExplainer(n_prototypes=1)
    X = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="y is required"):
        explainer.find_prototypes(X, target_class=1)
    with pytest.raises(ValueError, match="aligned"):
        explainer.find_prototypes(X, y=np.array([0]))
