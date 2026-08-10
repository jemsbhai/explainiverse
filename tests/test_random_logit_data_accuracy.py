"""Accuracy contracts for Random Logit and Data Randomisation metrics."""

import copy

import numpy as np
import pytest
from scipy import stats

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from explainiverse.evaluation.randomisation import (  # noqa: E402
    compute_batch_data_randomisation,
    compute_batch_random_logit,
    compute_data_randomisation,
    compute_data_randomisation_score,
    compute_random_logit,
    compute_random_logit_score,
)


class ThreeLogitModel(nn.Module):
    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            values = torch.tensor(
                [
                    [0.1, 0.4, -0.3, 0.2],
                    [-0.4, 0.2, 0.5, -0.1],
                    [0.7, 0.6, 0.3, 0.8],
                ]
            )
            self.linear.weight.copy_(values + offset)

    def forward(self, x):
        return self.linear(x)


class OneLogitModel(nn.Module):
    def __init__(self, offset: float = 0.0, squeeze: bool = False):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        self.squeeze = squeeze
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.2 + offset, -0.4, 0.7, 0.1]]))

    def forward(self, x):
        output = self.linear(x)
        return output[:, 0] if self.squeeze else output


class SpatialOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.linear(x).unsqueeze(-1)


_TARGET_PATTERNS = np.array(
    [
        [0.0, 1.0, 3.0, 2.0],
        [3.0, 0.0, 2.0, 1.0],
        [1.0, 3.0, 0.0, 2.0],
    ]
)


def _target_pattern_explainer(call_log):
    def explain(model, x, target):
        del model, x
        call_log.append(int(target))
        return _TARGET_PATTERNS[int(target)].copy()

    return explain


def _weight_explainer(model, x, target):
    del x
    return model.linear.weight[int(target)].detach().cpu().numpy().copy()


def _assert_numpy_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


class TestRandomLogitAccuracy:
    def test_uses_supplied_reference_targets_not_model_predictions(self):
        model = ThreeLogitModel()
        x = np.ones((1, 4), dtype=np.float32)
        assert int(model(torch.as_tensor(x)).argmax(dim=1)[0]) == 2
        calls = []

        compute_random_logit(
            model,
            x,
            np.array([0]),
            _target_pattern_explainer(calls),
            seed=4,
        )

        assert calls[0] == 0
        assert calls[1] in {1, 2}

    def test_seeded_off_targets_match_local_uniform_sampling(self):
        model = ThreeLogitModel()
        x = np.ones((5, 4), dtype=np.float32)
        targets = np.array([0, 1, 2, 0, 2])
        calls = []

        compute_batch_random_logit(model, x, targets, _target_pattern_explainer(calls), seed=17)

        rng = np.random.default_rng(17)
        expected_off_targets = []
        for target in targets:
            sampled = int(rng.integers(0, 2))
            expected_off_targets.append(sampled + int(sampled >= target))
        assert calls[::2] == targets.tolist()
        assert calls[1::2] == expected_off_targets
        assert all(off != target for off, target in zip(calls[1::2], targets))

    def test_seed_does_not_mutate_numpy_or_torch_global_rng(self):
        model = ThreeLogitModel()
        x = np.ones((2, 4), dtype=np.float32)
        y = np.array([0, 1])
        np.random.seed(123)
        numpy_state = copy.deepcopy(np.random.get_state())
        torch.manual_seed(456)
        torch_state = torch.get_rng_state().clone()

        compute_random_logit(model, x, y, _target_pattern_explainer([]), seed=99)

        _assert_numpy_rng_state_equal(numpy_state, np.random.get_state())
        assert torch.equal(torch_state, torch.get_rng_state())

    @pytest.mark.parametrize("squeeze", [False, True])
    def test_one_logit_binary_model_is_explicitly_unsupported(self, squeeze):
        with pytest.raises(ValueError, match="num_classes must be >= 2"):
            compute_random_logit(
                OneLogitModel(squeeze=squeeze),
                np.ones((1, 4), dtype=np.float32),
                np.array([0]),
                _target_pattern_explainer([]),
                seed=0,
            )

    def test_num_classes_cannot_override_actual_output_width(self):
        with pytest.raises(ValueError, match="does not match.*width 3"):
            compute_random_logit(
                ThreeLogitModel(),
                np.ones((1, 4), dtype=np.float32),
                np.array([0]),
                _target_pattern_explainer([]),
                num_classes=4,
            )

    @pytest.mark.parametrize("target", [-1, 3])
    def test_target_must_be_in_model_output_range(self, target):
        with pytest.raises(ValueError, match="values must be in"):
            compute_random_logit(
                ThreeLogitModel(),
                np.ones((1, 4), dtype=np.float32),
                np.array([target]),
                _target_pattern_explainer([]),
            )

    def test_float_targets_are_not_silently_truncated(self):
        with pytest.raises(TypeError, match="integer output indices"):
            compute_random_logit(
                ThreeLogitModel(),
                np.ones((1, 4), dtype=np.float32),
                np.array([1.0]),
                _target_pattern_explainer([]),
            )

    def test_explicit_class_count_requires_an_integer(self):
        with pytest.raises(TypeError, match="must be an integer"):
            compute_random_logit(
                ThreeLogitModel(),
                np.ones((1, 4), dtype=np.float32),
                np.array([1]),
                _target_pattern_explainer([]),
                num_classes=3.0,
            )

    @pytest.mark.parametrize(
        ("seed", "error_type"),
        [(3.0, TypeError), (True, TypeError), (-1, ValueError)],
    )
    def test_seed_contract_is_explicit(self, seed, error_type):
        with pytest.raises(error_type, match="non-negative integer"):
            compute_random_logit(
                ThreeLogitModel(),
                np.ones((1, 4), dtype=np.float32),
                np.array([1]),
                _target_pattern_explainer([]),
                seed=seed,
            )

    def test_validation_uses_model_device_dtype(self):
        model = ThreeLogitModel().double()
        score = compute_random_logit(
            model,
            np.ones((1, 4), dtype=np.float32),
            np.array([0]),
            _target_pattern_explainer([]),
            seed=1,
        )
        assert np.isfinite(score)

    def test_batch_validation_rejects_misaligned_targets(self):
        with pytest.raises(ValueError, match="same batch size"):
            compute_random_logit(
                ThreeLogitModel(),
                np.ones((2, 4), dtype=np.float32),
                np.array([0]),
                _target_pattern_explainer([]),
            )

    @pytest.mark.parametrize(
        ("x", "y", "error_type", "message"),
        [
            ([[1.0, 1.0, 1.0, 1.0]], np.array([0]), TypeError, "NumPy arrays"),
            (np.ones(4), np.array([0]), ValueError, "batch dimension"),
            (np.empty((0, 4)), np.empty(0, dtype=int), ValueError, "must not be empty"),
            (np.ones((1, 4)), np.array([[0]]), ValueError, "one-dimensional"),
            (
                np.array([[np.nan, 1.0, 1.0, 1.0]]),
                np.array([0]),
                ValueError,
                "only finite",
            ),
        ],
    )
    def test_batch_contract_rejects_ambiguous_inputs(self, x, y, error_type, message):
        with pytest.raises(error_type, match=message):
            compute_random_logit(ThreeLogitModel(), x, y, _target_pattern_explainer([]))

    def test_model_must_be_torch_module(self):
        with pytest.raises(TypeError, match="model must be a torch.nn.Module"):
            compute_random_logit(
                object(),
                np.ones((1, 4), dtype=np.float32),
                np.array([0]),
                _target_pattern_explainer([]),
            )

    def test_spatial_model_output_is_not_guessed_as_class_axis(self):
        with pytest.raises(ValueError, match="must return shape"):
            compute_random_logit(
                SpatialOutputModel(),
                np.ones((1, 4), dtype=np.float32),
                np.array([0]),
                _target_pattern_explainer([]),
            )

    def test_non_callable_explainer_is_rejected(self):
        with pytest.raises(TypeError, match="explain_func must be callable"):
            compute_random_logit(
                ThreeLogitModel(),
                np.ones((1, 4), dtype=np.float32),
                np.array([0]),
                None,
            )

    def test_similarity_must_be_finite_scalar(self):
        a = np.arange(4.0)
        b = a[::-1]
        with pytest.raises(ValueError, match="non-finite"):
            compute_random_logit_score(a, b, lambda left, right: np.nan)
        with pytest.raises(ValueError, match="one scalar"):
            compute_random_logit_score(a, b, lambda left, right: np.array([0.1, 0.2]))


class TestDataRandomisationAccuracy:
    def test_default_is_raw_signed_spearman_comparison(self):
        trained = np.array([-2.0, 0.5, 3.0, 1.0])
        random_labels = np.array([3.0, 0.0, -1.0, 2.0])
        expected = stats.spearmanr(trained, random_labels).statistic
        assert compute_data_randomisation_score(trained, random_labels) == pytest.approx(expected)

    def test_same_integer_target_is_used_for_both_models(self):
        calls = []

        def explain(model, x, target):
            del x
            calls.append(int(target))
            return model.linear.weight[int(target)].detach().numpy().copy()

        compute_batch_data_randomisation(
            ThreeLogitModel(0.0),
            ThreeLogitModel(0.15),
            np.ones((2, 4), dtype=np.float32),
            np.array([0, 2]),
            explain,
        )
        assert calls == [0, 0, 2, 2]

    def test_requires_same_architecture_and_parameter_shapes(self):
        different_architecture = nn.Sequential(nn.Linear(4, 3, bias=False))
        with pytest.raises(ValueError, match="same module architecture"):
            compute_data_randomisation(
                ThreeLogitModel(),
                different_architecture,
                np.ones((1, 4), dtype=np.float32),
                np.array([0]),
                _weight_explainer,
            )

    @pytest.mark.parametrize("squeeze", [False, True])
    def test_one_output_model_has_explicit_output_index_zero_contract(self, squeeze):
        score = compute_data_randomisation(
            OneLogitModel(0.0, squeeze=squeeze),
            OneLogitModel(0.2, squeeze=squeeze),
            np.ones((1, 4), dtype=np.float32),
            np.array([0]),
            _weight_explainer,
        )
        assert np.isfinite(score)

        with pytest.raises(ValueError, match=r"values must be in \[0, 0\]"):
            compute_data_randomisation(
                OneLogitModel(0.0, squeeze=squeeze),
                OneLogitModel(0.2, squeeze=squeeze),
                np.ones((1, 4), dtype=np.float32),
                np.array([1]),
                _weight_explainer,
            )

    def test_aggregate_is_arithmetic_mean_of_per_sample_scores(self):
        trained = ThreeLogitModel(0.0)
        random_labels = ThreeLogitModel(0.15)
        x = np.ones((3, 4), dtype=np.float32)
        y = np.array([0, 1, 2])
        scores = compute_batch_data_randomisation(trained, random_labels, x, y, _weight_explainer)
        aggregate = compute_data_randomisation(trained, random_labels, x, y, _weight_explainer)
        assert aggregate == pytest.approx(np.mean(scores))

    def test_models_and_global_rng_states_are_not_modified(self):
        trained = ThreeLogitModel(0.0)
        random_labels = ThreeLogitModel(0.15)
        trained.train()
        random_labels.train()
        trained_state = copy.deepcopy(trained.state_dict())
        random_state = copy.deepcopy(random_labels.state_dict())
        np.random.seed(812)
        numpy_state = copy.deepcopy(np.random.get_state())
        torch.manual_seed(913)
        torch_state = torch.get_rng_state().clone()

        compute_data_randomisation(
            trained,
            random_labels,
            np.ones((1, 4), dtype=np.float32),
            np.array([1]),
            _weight_explainer,
        )

        assert trained.training and random_labels.training
        for name, value in trained_state.items():
            assert torch.equal(value, trained.state_dict()[name])
        for name, value in random_state.items():
            assert torch.equal(value, random_labels.state_dict()[name])
        _assert_numpy_rng_state_equal(numpy_state, np.random.get_state())
        assert torch.equal(torch_state, torch.get_rng_state())

    def test_float_target_and_non_callable_explainer_are_rejected(self):
        args = (
            ThreeLogitModel(0.0),
            ThreeLogitModel(0.1),
            np.ones((1, 4), dtype=np.float32),
        )
        with pytest.raises(TypeError, match="integer output indices"):
            compute_data_randomisation(*args, np.array([1.0]), _weight_explainer)
        with pytest.raises(TypeError, match="explain_func must be callable"):
            compute_data_randomisation(*args, np.array([1]), None)

    def test_model_and_batch_contracts_are_validated(self):
        x = np.ones((1, 4), dtype=np.float32)
        y = np.array([0])
        with pytest.raises(TypeError, match="model_random_labels must be a torch.nn.Module"):
            compute_data_randomisation(ThreeLogitModel(), object(), x, y, _weight_explainer)
        with pytest.raises(ValueError, match="same batch size"):
            compute_data_randomisation(
                ThreeLogitModel(),
                ThreeLogitModel(0.1),
                np.ones((2, 4), dtype=np.float32),
                y,
                _weight_explainer,
            )

    def test_similarity_must_be_finite_scalar(self):
        a = np.arange(4.0)
        b = a[::-1]
        with pytest.raises(ValueError, match="non-finite"):
            compute_data_randomisation_score(a, b, lambda left, right: np.inf)
        with pytest.raises(ValueError, match="one scalar"):
            compute_data_randomisation_score(a, b, lambda left, right: np.array([0.1]))
