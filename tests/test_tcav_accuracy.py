"""Accuracy and contract tests for TCAV.

These tests exercise the mathematical contracts that the broad API tests do
not cover: CAVs live in the complete bottleneck activation space, conceptual
sensitivity differentiates one fixed target score, and inference compares
repeated concept-vs-random runs with repeated random-vs-random runs.
"""

from math import comb

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
stats = pytest.importorskip("scipy.stats")
pytest.importorskip("sklearn")

from explainiverse.adapters import PyTorchAdapter  # noqa: E402
from explainiverse.explainers.gradient.tcav import (  # noqa: E402
    ConceptActivationVector,
    TCAVExplainer,
)


class SpatialIdentityAdapter:
    """Small deterministic adapter exposing inputs as bottleneck tensors."""

    task = "classification"
    last_gradient_output_space = "model"

    def list_layers(self):
        return ["bottleneck"]

    def get_layer_output(self, inputs, layer_name):
        assert layer_name == "bottleneck"
        return np.asarray(inputs, dtype=float)

    def get_layer_gradients(self, inputs, layer_name, target_class=None):
        assert layer_name == "bottleneck"
        gradients = np.asarray(inputs, dtype=float)
        return gradients.copy(), gradients

    def predict(self, inputs):
        n_samples = len(np.asarray(inputs))
        return np.tile(np.array([[0.25, 0.75]]), (n_samples, 1))


class SwitchingPredictionAdapter(SpatialIdentityAdapter):
    def __init__(self):
        self.requested_targets = []

    def predict(self, inputs):
        assert len(inputs) == 3
        return np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]])

    def get_layer_gradients(self, inputs, layer_name, target_class=None):
        self.requested_targets.append(target_class)
        values = np.asarray(inputs, dtype=float)
        return values, values if target_class == 1 else -values


def _manual_cav(vector, layer_name="bottleneck"):
    return ConceptActivationVector(
        concept_name="manual",
        layer_name=layer_name,
        vector=np.asarray(vector, dtype=float),
        classifier=None,
        accuracy=1.0,
    )


def test_spatial_bottleneck_is_flattened_without_pooling():
    adapter = SpatialIdentityAdapter()
    explainer = TCAVExplainer(adapter, "bottleneck", random_seed=7)
    concept = np.array(
        [
            [[[3.0, 0.0], [0.0, 0.0]]],
            [[[4.0, 0.0], [0.0, 0.0]]],
            [[[5.0, 0.0], [0.0, 0.0]]],
            [[[6.0, 0.0], [0.0, 0.0]]],
            [[[7.0, 0.0], [0.0, 0.0]]],
        ]
    )
    negative = -concept

    activations = explainer._get_activations(concept)
    cav = explainer.learn_concept(
        "first_spatial_coordinate",
        concept,
        negative,
        test_size=0.0,
        min_accuracy=0.0,
    )

    assert activations.shape == (5, 4)
    assert cav.vector.shape == (4,)
    assert cav.vector[0] > 0.99
    np.testing.assert_allclose(cav.vector[1:], 0.0, atol=1e-12)


def test_spatial_directional_derivative_uses_same_flattened_space():
    adapter = SpatialIdentityAdapter()
    explainer = TCAVExplainer(adapter, "bottleneck")
    cav = _manual_cav([1.0, 2.0, 3.0, 4.0])
    inputs = np.array(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
        ]
    )

    actual = explainer.compute_directional_derivative(inputs, cav, target_class=1)
    expected = inputs.reshape(2, -1) @ cav.vector

    np.testing.assert_allclose(actual, expected)


class OneLogitNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Linear(2, 2, bias=False)
        self.output = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.bottleneck.weight.copy_(torch.eye(2))
            self.output.weight.copy_(torch.tensor([[2.0, -3.0]]))

    def forward(self, inputs):
        return self.output(self.bottleneck(inputs))


class TwoLogitNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Linear(2, 2, bias=False)
        self.output = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.bottleneck.weight.copy_(torch.eye(2))
            self.output.weight.copy_(torch.tensor([[2.0, -3.0], [-1.0, 4.0]]))

    def forward(self, inputs):
        return self.output(self.bottleneck(inputs))


def test_one_logit_tcav_uses_explicit_probability_score_and_complements():
    network = OneLogitNetwork()
    adapter = PyTorchAdapter(
        network,
        task="classification",
        output_activation="auto",
        gradient_output="model",
    )
    explainer = TCAVExplainer(adapter, "bottleneck")
    cav = _manual_cav([1.0, 0.0])
    inputs = np.array([[0.0, 0.0], [0.5, -0.25]], dtype=np.float32)

    class_one = explainer.compute_directional_derivative(inputs, cav, target_class=np.int64(1))
    class_zero = explainer.compute_directional_derivative(inputs, cav, target_class=0)
    logits = 2.0 * inputs[:, 0] - 3.0 * inputs[:, 1]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    expected = 2.0 * probabilities * (1.0 - probabilities)

    np.testing.assert_allclose(class_one, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(class_zero, -expected, rtol=1e-6, atol=1e-7)
    assert adapter.last_gradient_output_space == "prediction"
    assert explainer.last_tcav_variant == "prediction_space_directional_derivative_variant"
    assert explainer.last_tcav_is_canonical is False


def test_canonical_tcav_requires_and_records_effective_class_logits():
    adapter = PyTorchAdapter(TwoLogitNetwork(), task="classification")
    explainer = TCAVExplainer(
        adapter,
        "bottleneck",
        require_logit_scores=True,
    )
    cav = _manual_cav([1.0, 0.0])

    derivatives = explainer.compute_directional_derivative(
        np.array([[0.0, 0.0], [0.5, -0.25]], dtype=np.float32),
        cav,
        target_class=0,
    )

    np.testing.assert_allclose(derivatives, [2.0, 2.0], rtol=0.0, atol=0.0)
    assert explainer.last_target_score_space == "model"
    assert explainer.last_tcav_variant == "canonical_class_logit_tcav"
    assert explainer.last_tcav_is_canonical is True


def test_canonical_tcav_rejects_effective_prediction_space_for_one_logit_model():
    adapter = PyTorchAdapter(OneLogitNetwork(), task="classification")
    explainer = TCAVExplainer(
        adapter,
        "bottleneck",
        require_logit_scores=True,
    )

    with pytest.raises(ValueError, match="effective class-logit gradients"):
        explainer.compute_directional_derivative(
            np.array([[0.0, 0.0]], dtype=np.float32),
            _manual_cav([1.0, 0.0]),
            target_class=1,
        )


def test_one_logit_explanation_reports_effective_score_contract():
    network = OneLogitNetwork()
    adapter = PyTorchAdapter(network, task="classification")
    explainer = TCAVExplainer(
        adapter,
        "bottleneck",
        class_names=["negative", "positive"],
    )
    concept = np.array([[2.0, 0.0], [3.0, 0.1], [4.0, -0.1], [5.0, 0.0]])
    negative = -concept
    explainer.learn_concept("positive_x", concept, negative, test_size=0.0, min_accuracy=0.0)

    explanation = explainer.explain(concept, target_class=1)

    assert explanation.target_class == "positive"
    assert explanation.explanation_data["target_score_fixed_for_batch"] is True
    assert (
        explanation.explanation_data["tcav_scores"]["positive_x"]["target_score_space"]
        == "prediction"
    )
    assert (
        explanation.explanation_data["test_input_class_membership"]
        == "caller_supplied_not_validated"
    )
    assert explanation.explanation_data["aggregate_scope"] == ("global_target_class_input_set")
    assert explanation.explanation_data["returns_per_instance_scores"] is False
    assert explanation.explanation_data["tcav_variant"] == (
        "prediction_space_directional_derivative_variant"
    )
    assert explanation.explanation_data["canonical_class_logit_tcav"] is False
    assert explanation.metadata["explanation_scope"] == "global"


def test_auto_target_is_resolved_once_and_fixed_for_the_batch():
    adapter = SwitchingPredictionAdapter()
    explainer = TCAVExplainer(adapter, "bottleneck")
    concept = np.array([[2.0, 0.0], [3.0, 0.1], [4.0, -0.1]])
    explainer.learn_concept("x", concept, -concept, test_size=0.0, min_accuracy=0.0)

    explanation = explainer.explain(concept, target_class=None)

    assert explanation.explanation_data["target_class"] == 1
    assert adapter.requested_targets == [1]


def test_layer_hooks_are_removed_even_when_cav_dimension_is_invalid():
    network = OneLogitNetwork()
    adapter = PyTorchAdapter(network, task="classification")
    explainer = TCAVExplainer(adapter, "bottleneck")
    invalid_cav = _manual_cav([1.0, 0.0, 0.0])
    before = (
        len(network.bottleneck._forward_hooks),
        len(network.bottleneck._backward_hooks),
    )

    with pytest.raises(ValueError, match="CAV dimension"):
        explainer.compute_directional_derivative(
            np.array([[0.0, 0.0]], dtype=np.float32),
            invalid_cav,
            target_class=1,
        )

    after = (
        len(network.bottleneck._forward_hooks),
        len(network.bottleneck._backward_hooks),
    )
    assert after == before


def test_cav_training_does_not_mutate_numpy_global_rng():
    adapter = SpatialIdentityAdapter()
    explainer = TCAVExplainer(adapter, "bottleneck", random_seed=11)
    positive = np.arange(24, dtype=float).reshape(6, 4)
    negative = -np.arange(40, dtype=float).reshape(10, 4)

    np.random.seed(1234)
    expected_next = np.random.RandomState(1234).random_sample()
    explainer._train_cav(positive, negative, test_size=0.0)
    actual_next = np.random.random()

    assert actual_next == expected_next


def test_cav_accuracy_metadata_distinguishes_holdout_from_training_fallback():
    adapter = SpatialIdentityAdapter()
    explainer = TCAVExplainer(adapter, "bottleneck", random_seed=11)
    concept = np.column_stack((np.linspace(2.0, 5.0, 6), np.zeros(6)))
    negative = np.column_stack((np.linspace(-5.0, -2.0, 6), np.zeros(6)))

    held_out = explainer.learn_concept(
        "held_out",
        concept,
        negative,
        test_size=0.25,
        min_accuracy=0.0,
    )
    tiny = explainer.learn_concept(
        "tiny",
        concept[:2],
        negative[:2],
        test_size=0.2,
        min_accuracy=0.0,
    )

    assert held_out.metadata["accuracy_evaluation"] == "held_out"
    assert held_out.metadata["accuracy_effective_test_size"] == pytest.approx(0.25)
    assert tiny.metadata["accuracy_evaluation"] == "training"
    assert tiny.metadata["accuracy_effective_test_size"] == 0.0


def test_tcav_temporary_eval_restores_mixed_modes_buffers_and_gradients():
    class StatefulTCAVNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_norm = nn.BatchNorm1d(2)
            self.dropout = nn.Dropout(p=0.95)
            self.bottleneck = nn.Linear(2, 2, bias=False)
            self.output = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.bottleneck.weight.copy_(torch.eye(2))
                self.output.weight.copy_(torch.eye(2))

        def forward(self, inputs):
            hidden = self.dropout(self.batch_norm(inputs))
            return self.output(self.bottleneck(hidden))

    network = StatefulTCAVNet()
    adapter = PyTorchAdapter(network, task="classification")
    network.train()
    network.dropout.eval()
    network.output.weight.grad = torch.full_like(network.output.weight, 3.0)
    flags_before = [module.training for module in network.modules()]
    mean_before = network.batch_norm.running_mean.detach().clone()
    gradient_object = network.output.weight.grad
    inputs = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    explainer = TCAVExplainer(adapter, "bottleneck")

    activations_first = explainer._get_activations(inputs)
    activations_second = explainer._get_activations(inputs)
    gradients_first = explainer._get_gradients_wrt_layer(inputs, target_class=0)
    gradients_second = explainer._get_gradients_wrt_layer(inputs, target_class=0)

    np.testing.assert_allclose(activations_first, activations_second)
    np.testing.assert_allclose(gradients_first, gradients_second)
    assert [module.training for module in network.modules()] == flags_before
    torch.testing.assert_close(network.batch_norm.running_mean, mean_before)
    assert network.output.weight.grad is gradient_object
    torch.testing.assert_close(
        network.output.weight.grad,
        torch.full_like(network.output.weight, 3.0),
    )


def test_invalid_cav_vector_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        _manual_cav([0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        _manual_cav([1.0, np.nan])


def test_tcav_rejects_unverified_or_inconsistent_public_inputs():
    adapter = SpatialIdentityAdapter()
    with pytest.raises(ValueError, match="cav_classifier"):
        TCAVExplainer(adapter, "bottleneck", cav_classifier="not-linear")
    with pytest.raises(ValueError, match="unique"):
        TCAVExplainer(adapter, "bottleneck", class_names=["same", "same"])

    explainer = TCAVExplainer(adapter, "bottleneck")
    concept = np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    with pytest.raises(ValueError, match="negative_examples is required"):
        explainer.learn_concept("x", concept)

    explainer.learn_concept("x", concept, -concept, test_size=0.0, min_accuracy=0.0)
    with pytest.raises(ValueError, match="Unknown concepts"):
        explainer.explain(concept, target_class=1, concept_names=["missing"])

    mismatched_names = TCAVExplainer(
        adapter,
        "bottleneck",
        class_names=["zero", "one", "extra"],
    )
    mismatched_names.learn_concept("x", concept, -concept, test_size=0.0, min_accuracy=0.0)
    with pytest.raises(ValueError, match="class_names length"):
        mismatched_names.explain(concept, target_class=1)


def test_repeated_tcav_inference_compares_two_score_distributions():
    adapter = SpatialIdentityAdapter()
    explainer = TCAVExplainer(adapter, "bottleneck", random_seed=19)
    concept = np.array([[4.0, 0.0], [5.0, 0.1], [4.5, -0.1], [5.5, 0.0], [6.0, 0.2]])
    initial_negative = np.array([[-2.0, 0.0], [-3.0, 0.1], [-2.5, -0.1], [-4.0, 0.0], [-3.5, 0.2]])
    explainer.learn_concept(
        "positive_x",
        concept,
        initial_negative,
        test_size=0.0,
        min_accuracy=0.0,
    )
    random_sets = [
        np.array([[-4.0, 0.0], [-3.5, 0.2], [-4.5, -0.2], [-3.0, 0.1]]),
        np.array([[0.0, -4.0], [0.2, -3.5], [-0.2, -4.5], [0.1, -3.0]]),
        np.array([[2.5, 0.0], [3.0, 0.2], [2.0, -0.2], [3.5, 0.1]]),
        np.array([[0.0, 4.0], [0.2, 3.5], [-0.2, 4.5], [0.1, 3.0]]),
    ]
    test_gradients = np.array([[1.0, 0.0], [1.0, 0.2], [1.0, -0.2], [-1.0, 0.0], [0.0, 1.0]])

    result = explainer.statistical_significance_test(
        test_inputs=test_gradients,
        target_class=1,
        concept_name="positive_x",
        n_random=4,
        random_example_sets=random_sets,
    )

    assert len(result["concept_scores"]) == 4
    assert len(result["random_scores"]) == comb(4, 2)
    assert len(result["concept_cav_accuracies"]) == 4
    assert len(result["random_cav_accuracies"]) == comb(4, 2)
    assert result["concept_cav_accuracy_evaluations"] == ["held_out"] * 4
    assert result["random_cav_accuracy_evaluations"] == ["held_out"] * comb(4, 2)
    assert result["cav_accuracy_requested_test_size"] == pytest.approx(0.2)
    expected = stats.ttest_ind(
        result["concept_scores"],
        result["random_scores"],
        equal_var=False,
    )
    assert result["t_statistic"] == pytest.approx(float(expected.statistic))
    assert result["p_value"] == pytest.approx(float(expected.pvalue))
    assert result["test_method"] == "two-sided Welch t-test"
    assert result["supports_confirmatory_significance_claim"] is False
    assert result["multiple_comparisons_corrected"] is False
    assert result["random_set_source"] == "explicit_sets"
    assert result["n_concept_runs"] == 4
    assert result["n_random_baseline_runs"] == 6


def test_statistical_test_partitions_pool_without_reusing_test_inputs():
    adapter = SpatialIdentityAdapter()
    explainer = TCAVExplainer(adapter, "bottleneck", random_seed=5)
    concept = np.column_stack((np.linspace(3.0, 5.0, 12), np.zeros(12)))
    negative = np.column_stack((np.linspace(-5.0, -1.0, 12), np.zeros(12)))
    explainer.learn_concept(
        "x",
        concept,
        negative,
        test_size=0.0,
        min_accuracy=0.0,
    )

    result = explainer.statistical_significance_test(
        test_inputs=np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]),
        target_class=1,
        concept_name="x",
        n_random=3,
        negative_examples=negative,
    )

    assert result["random_set_source"] == "disjoint_partition"
    assert result["random_set_size"] == 4
    repeated = explainer.statistical_significance_test(
        test_inputs=np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]),
        target_class=1,
        concept_name="x",
        n_random=3,
        negative_examples=negative,
    )
    assert repeated["concept_scores"] == result["concept_scores"]
    assert repeated["random_scores"] == result["random_scores"]
    assert repeated["p_value"] == result["p_value"]
    with pytest.raises(ValueError, match="negative_examples"):
        explainer.statistical_significance_test(
            test_inputs=np.array([[1.0, 0.0], [-1.0, 0.0]]),
            target_class=1,
            concept_name="x",
            n_random=3,
        )


def test_statistical_test_validates_number_of_runs_and_alpha():
    adapter = SpatialIdentityAdapter()
    explainer = TCAVExplainer(adapter, "bottleneck")
    concept = np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    negative = -concept
    explainer.learn_concept("x", concept, negative, test_size=0.0, min_accuracy=0.0)

    with pytest.raises(ValueError, match="at least 3"):
        explainer.statistical_significance_test(
            concept,
            1,
            "x",
            n_random=2,
            negative_examples=np.tile(negative, (2, 1)),
        )
    with pytest.raises(ValueError, match="alpha"):
        explainer.statistical_significance_test(
            concept,
            1,
            "x",
            n_random=3,
            negative_examples=np.tile(negative, (3, 1)),
            alpha=1.5,
        )


def test_welch_diagnostic_has_defined_constant_sample_limits():
    equal = TCAVExplainer._welch_test(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]))
    separated = TCAVExplainer._welch_test(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))

    assert equal == (0.0, 1.0)
    assert separated == (float("inf"), 0.0)
