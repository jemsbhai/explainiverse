"""Accuracy and contract tests for Layer-wise Relevance Propagation."""

from __future__ import annotations

import copy
from types import MethodType

import numpy as np
import pytest

torch = pytest.importorskip("torch")
captum = pytest.importorskip("captum")
from captum.attr import LRP as CaptumLRP  # noqa: E402
from captum.attr._utils.lrp_rules import (  # noqa: E402
    Alpha1_Beta0_Rule,
    EpsilonRule,
    GammaRule,
    PropagationRule,
)
from torch import nn  # noqa: E402

from explainiverse.adapters import PyTorchAdapter  # noqa: E402
from explainiverse.explainers.gradient import LRPExplainer  # noqa: E402


class _ReferenceReshapeRule(PropagationRule):
    """Independent relevance-preserving reshape for direct Captum checks."""

    def _manipulate_weights(self, module, inputs, outputs):
        return None

    def _create_backward_hook_input(self, inputs):
        def hook(grad):
            return self.relevance_output[grad.device].reshape_as(inputs)

        return hook


def _adapter(model: nn.Module, *, task: str = "classification", **kwargs):
    return PyTorchAdapter(model, task=task, device="cpu", **kwargs)


def _attrs(explanation):
    return np.asarray(explanation.explanation_data["attributions_raw"])


@pytest.mark.parametrize(
    ("rule", "rule_factory"),
    [
        ("epsilon", lambda: EpsilonRule(epsilon=1e-6)),
        ("gamma", lambda: GammaRule(gamma=0.3)),
    ],
)
def test_matches_captum_backend_for_supported_rules(rule, rule_factory):
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[1.0, -0.5, 0.2], [0.3, 0.8, -0.4], [-0.2, 0.1, 0.7], [0.6, 0.2, 0.1]])
        )
        model[0].bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        model[2].weight.copy_(torch.tensor([[0.5, -0.3, 0.8, 0.2], [-0.4, 0.9, 0.1, 0.7]]))
        model[2].bias.copy_(torch.tensor([0.2, -0.1]))

    instance = np.array([0.7, -0.4, 1.2], dtype=np.float32)
    reference_model = copy.deepcopy(model)
    for module in reference_model.modules():
        if isinstance(module, nn.Linear):
            module.rule = rule_factory()
    expected = (
        CaptumLRP(reference_model)
        .attribute(torch.tensor(instance).unsqueeze(0), target=1)
        .detach()
        .numpy()
        .reshape(-1)
    )

    kwargs = {"epsilon": 1e-6} if rule == "epsilon" else {"gamma": 0.3}
    explainer = LRPExplainer(_adapter(model), ["a", "b", "c"], rule=rule, **kwargs)
    actual = _attrs(explainer.explain(instance, target_class=1))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_alpha_beta_uses_negative_activations_and_conserves_bias_free_score():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0]]))

    explainer = LRPExplainer(
        _adapter(model, task="regression"),
        ["negative_contribution", "positive_contribution"],
        rule="alpha_beta",
        alpha=2.0,
        beta=1.0,
        epsilon=1e-12,
    )
    explanation = explainer.explain(
        np.array([-1.0, -1.0], dtype=np.float32), return_convergence_delta=True
    )

    np.testing.assert_allclose(_attrs(explanation), [-1.0, 2.0], atol=1e-6)
    assert explanation.explanation_data["target_output"] == pytest.approx(1.0)
    assert explanation.explanation_data["attribution_sum"] == pytest.approx(1.0)
    assert explanation.explanation_data["convergence_delta"] == pytest.approx(0.0, abs=1e-6)


def test_alpha1_beta0_matches_captum_on_nonnegative_relu_chain():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -0.5], [0.2, 0.4]]))
        model[2].weight.copy_(torch.tensor([[1.5, 0.8]]))
    instance = np.array([1.0, 1.0], dtype=np.float32)

    reference = copy.deepcopy(model)
    for module in reference.modules():
        if isinstance(module, nn.Linear):
            module.rule = Alpha1_Beta0_Rule(set_bias_to_zero=True)
    expected = (
        CaptumLRP(reference)
        .attribute(torch.tensor(instance).unsqueeze(0), target=0)
        .detach()
        .numpy()
        .reshape(-1)
    )
    actual = _attrs(
        LRPExplainer(
            _adapter(model, task="regression"),
            ["x0", "x1"],
            rule="alpha_beta",
            alpha=1,
            beta=0,
            epsilon=1e-9,
        ).explain(instance)
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_one_logit_epsilon_class_zero_matches_explicit_negative_margin_model():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
    explainer = LRPExplainer(
        _adapter(model, class_names=["no", "yes"]),
        ["x0", "x1"],
        class_names=["no", "yes"],
        rule="epsilon",
        epsilon=1e-12,
    )
    instance = np.array([2.0, 1.0], dtype=np.float32)

    explicit_negative_model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        explicit_negative_model.weight.copy_(torch.tensor([[-2.0, 1.0]]))
    explicit_negative = LRPExplainer(
        _adapter(explicit_negative_model, class_names=["unused", "negative_margin"]),
        ["x0", "x1"],
        class_names=["unused", "negative_margin"],
        rule="epsilon",
        epsilon=1e-12,
    ).explain(instance, target_class=1, return_convergence_delta=True)

    class_1 = explainer.explain(instance, target_class=1, return_convergence_delta=True)
    class_0 = explainer.explain(instance, target_class=0, return_convergence_delta=True)

    np.testing.assert_allclose(_attrs(class_0), _attrs(explicit_negative), atol=1e-6)
    np.testing.assert_allclose(_attrs(class_0), -_attrs(class_1), atol=1e-6)
    assert class_1.explanation_data["target_output"] == pytest.approx(3.0)
    assert class_0.explanation_data["target_output"] == pytest.approx(-3.0)
    assert class_0.explanation_data["score_space"] == "binary_logit_margin"
    assert class_0.target_class == "no"
    assert class_1.target_class == "yes"


@pytest.mark.parametrize(
    "rule_kwargs",
    [
        {"rule": "gamma", "gamma": 0.3},
        {"rule": "z_plus"},
        {"rule": "alpha_beta", "alpha": 2.0, "beta": 1.0, "epsilon": 1e-12},
    ],
)
def test_one_logit_class_zero_rejects_sign_asymmetric_rules(rule_kwargs):
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
    explainer = LRPExplainer(
        _adapter(model, class_names=["no", "yes"]),
        ["x0", "x1"],
        class_names=["no", "yes"],
        **rule_kwargs,
    )

    with pytest.raises(ValueError, match="sign-asymmetric"):
        explainer.explain(np.array([2.0, 1.0], dtype=np.float32), target_class=0)


def test_one_logit_class_zero_rejects_asymmetric_composite():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
    explainer = LRPExplainer(
        _adapter(model, class_names=["no", "yes"]),
        ["x0", "x1"],
        class_names=["no", "yes"],
        rule="composite",
    ).set_composite_rule({0: "gamma"})

    with pytest.raises(ValueError, match="sign-asymmetric"):
        explainer.explain(np.array([2.0, 1.0], dtype=np.float32), target_class=0)


def test_default_multiclass_target_does_not_depend_on_class_names():
    model = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]))
    explainer = LRPExplainer(_adapter(model), ["x0", "x1"], epsilon=1e-12)

    explanation = explainer.explain(np.array([1.0, 2.0], dtype=np.float32))

    assert explanation.target_class == "class_1"
    assert explanation.explanation_data["target_class_index"] == 1
    assert explanation.explanation_data["target_output"] == pytest.approx(4.0)


def test_multioutput_regression_requires_explicit_output_index():
    model = nn.Linear(2, 2, bias=False)
    explainer = LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])

    with pytest.raises(ValueError, match="output index"):
        explainer.explain(np.array([1.0, 2.0], dtype=np.float32))


def test_multioutput_regression_explains_only_requested_raw_output():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
    explainer = LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"], epsilon=0)

    explanation = explainer.explain(np.array([1.0, 2.0], dtype=np.float32), target_class=1)

    np.testing.assert_allclose(_attrs(explanation), [0.0, 4.0], atol=1e-6)
    assert explanation.target_class == "output_1"
    assert explanation.explanation_data["target_class_index"] == 1


def test_rejects_nonsequential_graph_instead_of_linearizing_hook_order():
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(2, 2, bias=False)
            self.right = nn.Linear(2, 2, bias=False)
            self.out = nn.Linear(2, 1, bias=False)

        def forward(self, x):
            return self.out(self.left(x) + self.right(x))

    with pytest.raises(TypeError, match="nn.Sequential"):
        LRPExplainer(_adapter(Residual(), task="regression"), ["x0", "x1"])


def test_rejects_unsupported_activation_instead_of_passthrough():
    model = nn.Sequential(nn.Linear(2, 2), nn.GELU(), nn.Linear(2, 1))

    with pytest.raises(TypeError, match="GELU"):
        LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])


def test_alpha_beta_rejects_convolution_instead_of_epsilon_fallback():
    model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Flatten(), nn.Linear(4, 1))

    with pytest.raises(NotImplementedError, match="alpha_beta.*Linear"):
        LRPExplainer(
            _adapter(model, task="regression"),
            [f"pixel_{i}" for i in range(4)],
            rule="alpha_beta",
            alpha=2.0,
            beta=1.0,
        )


def test_feature_name_mismatch_is_rejected():
    explainer = LRPExplainer(_adapter(nn.Linear(2, 1), task="regression"), ["only_one_name"])

    with pytest.raises(ValueError, match="feature_names"):
        explainer.explain(np.array([1.0, 2.0], dtype=np.float32))


def test_spatial_input_requires_an_explicit_channel_dimension():
    model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Flatten(), nn.Linear(4, 1))
    explainer = LRPExplainer(_adapter(model, task="regression"), [f"p{i}" for i in range(4)])

    with pytest.raises(ValueError, match="channels, height, width"):
        explainer.explain(np.ones((2, 2), dtype=np.float32))


def test_model_state_and_training_flags_are_restored():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Dropout(0.5), nn.Linear(3, 1))
    adapter = _adapter(model, task="regression")
    model.train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    flags_before = {name: module.training for name, module in model.named_modules()}

    LRPExplainer(adapter, ["x0", "x1"], rule="gamma", gamma=0.7).explain(
        np.array([1.0, 2.0], dtype=np.float32)
    )

    flags_after = {name: module.training for name, module in model.named_modules()}
    assert flags_after == flags_before
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert not any(hasattr(module, "rule") for module in model.modules())
    assert not any(module._forward_hooks for module in model.modules())
    assert not any(module._forward_pre_hooks for module in model.modules())
    assert not any(module._backward_hooks for module in model.modules())


def test_model_state_and_hooks_are_restored_when_propagation_fails():
    model = nn.Sequential(nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.zero_()
    adapter = _adapter(model, task="regression")
    model.train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    flags_before = {name: module.training for name, module in model.named_modules()}

    explainer = LRPExplainer(adapter, ["x0", "x1"], epsilon=0)
    with pytest.raises(FloatingPointError, match="non-finite relevance"):
        explainer.explain(np.array([1.0, 2.0], dtype=np.float32))

    assert {name: module.training for name, module in model.named_modules()} == flags_before
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert not any(hasattr(module, "rule") for module in model.modules())
    assert not any(module._forward_hooks for module in model.modules())
    assert not any(module._forward_pre_hooks for module in model.modules())
    assert not any(module._backward_hooks for module in model.modules())


def test_invalid_target_is_rejected_before_indexing():
    explainer = LRPExplainer(_adapter(nn.Linear(2, 3)), ["x0", "x1"])

    with pytest.raises(ValueError, match="target_class"):
        explainer.explain(np.array([1.0, 2.0], dtype=np.float32), target_class=3)


def test_zero_denominator_is_finite_and_residual_is_reported():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -1.0]]))
    explainer = LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"], epsilon=1e-6)

    explanation = explainer.explain(
        np.array([1.0, 1.0], dtype=np.float32), return_convergence_delta=True
    )

    assert np.isfinite(_attrs(explanation)).all()
    assert np.isfinite(explanation.explanation_data["convergence_delta"])


def test_z_plus_matches_captum_alpha1_beta0_with_bias_excluded():
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [0.2, 0.3, 0.4]]))
        model[0].bias.copy_(torch.tensor([0.7, -0.4]))
        model[2].weight.copy_(torch.tensor([[1.5, 0.8]]))
        model[2].bias.copy_(torch.tensor([0.6]))
    instance = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    reference = copy.deepcopy(model)
    for module in reference.modules():
        if isinstance(module, nn.Linear):
            module.rule = Alpha1_Beta0_Rule(set_bias_to_zero=True)
    expected = (
        CaptumLRP(reference)
        .attribute(torch.tensor(instance).unsqueeze(0), target=0)
        .detach()
        .numpy()
        .reshape(-1)
    )
    actual = _attrs(
        LRPExplainer(_adapter(model, task="regression"), ["a", "b", "c"], rule="z_plus").explain(
            instance
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_composite_matches_direct_captum_per_layer_rules():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    torch.manual_seed(7)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -0.7, 0.9)
            nn.init.uniform_(module.bias, -0.2, 0.2)
    instance = np.array([0.8, 1.4], dtype=np.float32)

    reference = copy.deepcopy(model)
    reference[0].rule = GammaRule(gamma=0.4)
    reference[2].rule = EpsilonRule(epsilon=1e-5)
    expected = (
        CaptumLRP(reference)
        .attribute(torch.tensor(instance).unsqueeze(0), target=0)
        .detach()
        .numpy()
        .reshape(-1)
    )

    explainer = LRPExplainer(
        _adapter(model, task="regression"),
        ["x0", "x1"],
        rule="composite",
        gamma=0.4,
        epsilon=1e-5,
    ).set_composite_rule({0: "gamma", 2: "epsilon"})
    actual = _attrs(explainer.explain(instance))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    metadata = explainer.explain(instance).explanation_data
    assert metadata["effective_layer_rules"] == {
        0: "gamma",
        1: "relevance_passthrough",
        2: "epsilon",
    }


def test_conv2d_epsilon_has_analytical_z_rule_decomposition():
    model = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=1, bias=False),
        nn.Flatten(),
        nn.Linear(4, 1, bias=False),
    )
    with torch.no_grad():
        model[0].weight.fill_(2.0)
        model[2].weight.fill_(1.0)
    instance = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    explanation = LRPExplainer(
        _adapter(model, task="regression"),
        [f"p{i}" for i in range(4)],
        epsilon=0,
    ).explain(instance, return_convergence_delta=True)

    np.testing.assert_allclose(_attrs(explanation), [2.0, 4.0, 6.0, 8.0], atol=1e-6)
    assert explanation.explanation_data["target_output"] == pytest.approx(20.0)
    assert explanation.explanation_data["convergence_delta"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("pool", "expected"),
    [
        (nn.MaxPool2d(2), [0.0, 0.0, 0.0, 4.0]),
        (nn.AvgPool2d(2), [0.25, 0.5, 0.75, 1.0]),
        (nn.AdaptiveAvgPool2d((1, 1)), [0.25, 0.5, 0.75, 1.0]),
    ],
)
def test_pooling_layers_have_analytical_decompositions(pool, expected):
    model = nn.Sequential(pool, nn.Flatten(), nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[2].weight.fill_(1.0)
    instance = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    explanation = LRPExplainer(
        _adapter(model, task="regression"),
        [f"p{i}" for i in range(4)],
        epsilon=0,
    ).explain(instance, return_convergence_delta=True)

    np.testing.assert_allclose(_attrs(explanation), expected, atol=1e-6)
    assert explanation.explanation_data["convergence_delta"] == pytest.approx(0.0, abs=1e-6)


def test_batchnorm1d_affine_scale_is_decomposed_analytically():
    batchnorm = nn.BatchNorm1d(2)
    model = nn.Sequential(batchnorm, nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        batchnorm.running_mean.zero_()
        batchnorm.running_var.fill_(1.0)
        batchnorm.weight.copy_(torch.tensor([2.0, 3.0]))
        batchnorm.bias.zero_()
        model[1].weight.fill_(1.0)
    model.eval()
    instance = np.array([1.5, 2.0], dtype=np.float32)
    scale = np.array([2.0, 3.0]) / np.sqrt(1.0 + batchnorm.eps)

    explanation = LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"], epsilon=0).explain(
        instance, return_convergence_delta=True
    )

    np.testing.assert_allclose(_attrs(explanation), instance * scale, rtol=1e-6, atol=1e-6)
    assert explanation.explanation_data["convergence_delta"] == pytest.approx(0.0, abs=1e-5)


def test_batchnorm2d_affine_scale_is_decomposed_analytically():
    batchnorm = nn.BatchNorm2d(1)
    model = nn.Sequential(batchnorm, nn.Flatten(), nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        batchnorm.running_mean.zero_()
        batchnorm.running_var.fill_(1.0)
        batchnorm.weight.fill_(2.0)
        batchnorm.bias.zero_()
        model[2].weight.fill_(1.0)
    model.eval()
    instance = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    scale = 2.0 / np.sqrt(1.0 + batchnorm.eps)

    explanation = LRPExplainer(
        _adapter(model, task="regression"),
        [f"p{i}" for i in range(4)],
        epsilon=0,
    ).explain(instance, return_convergence_delta=True)

    np.testing.assert_allclose(
        _attrs(explanation), instance.reshape(-1) * scale, rtol=1e-6, atol=1e-6
    )
    assert explanation.explanation_data["convergence_delta"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    ("activation", "instance"),
    [
        (nn.ReLU(), np.array([1.0, 2.0], dtype=np.float32)),
        (nn.LeakyReLU(0.2), np.array([-1.0, 2.0], dtype=np.float32)),
        (nn.ELU(), np.array([-1.0, 2.0], dtype=np.float32)),
        (nn.Tanh(), np.array([-1.0, 2.0], dtype=np.float32)),
        (nn.Sigmoid(), np.array([-1.0, 2.0], dtype=np.float32)),
        (nn.Dropout(0.7), np.array([-1.0, 2.0], dtype=np.float32)),
    ],
)
def test_supported_pointwise_layers_preserve_neuron_relevance(activation, instance):
    model = nn.Sequential(nn.Linear(2, 2, bias=False), activation, nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
        model[2].weight.fill_(1.0)
    model.eval()
    expected = activation(torch.tensor(instance)).detach().numpy()

    explanation = LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"], epsilon=0).explain(
        instance, return_convergence_delta=True
    )

    np.testing.assert_allclose(_attrs(explanation), expected, rtol=1e-6, atol=1e-6)
    assert explanation.explanation_data["convergence_delta"] == pytest.approx(0.0, abs=1e-5)


def test_reshape_layers_preserve_relevance():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Unflatten(1, (1, 2, 2)),
        nn.Flatten(),
        nn.Linear(4, 1, bias=False),
    )
    with torch.no_grad():
        model[3].weight.fill_(1.0)
    model.eval()
    instance = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    explanation = LRPExplainer(
        _adapter(model, task="regression"),
        [f"p{i}" for i in range(4)],
        epsilon=0,
    ).explain(instance, return_convergence_delta=True)

    np.testing.assert_allclose(_attrs(explanation), instance.reshape(-1), atol=1e-6)
    assert explanation.explanation_data["convergence_delta"] == pytest.approx(0.0, abs=1e-6)


def test_dropout2d_is_rejected_instead_of_treated_as_identity():
    model = nn.Sequential(nn.Dropout2d(0.9), nn.Flatten(), nn.Linear(4, 1, bias=False))

    with pytest.raises(TypeError, match="Dropout2d"):
        LRPExplainer(_adapter(model, task="regression"), [f"p{i}" for i in range(4)])


def test_alpha_beta_bias_and_degenerate_sign_residual_are_reported():
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.fill_(3.0)
    explanation = LRPExplainer(
        _adapter(model, task="regression"),
        ["x0", "x1"],
        rule="alpha_beta",
        alpha=2,
        beta=1,
        epsilon=0,
    ).explain(np.array([1.0, 1.0], dtype=np.float32), return_convergence_delta=True)

    np.testing.assert_allclose(_attrs(explanation), [4.0, 8.0], atol=1e-6)
    assert explanation.explanation_data["target_output"] == pytest.approx(6.0)
    assert explanation.explanation_data["signed_convergence_delta"] == pytest.approx(-6.0)
    assert explanation.explanation_data["bias_treatment"] == (
        "excluded_from_local_denominators; " "output_relevance_including_bias_is_propagated"
    )


def test_one_output_probability_model_rejects_class_zero_constant_complement():
    model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    explainer = LRPExplainer(
        _adapter(model, output_activation="none", class_names=["no", "yes"]),
        ["x0", "x1"],
        class_names=["no", "yes"],
    )

    with pytest.raises(ValueError, match="Class-0 LRP is undefined"):
        explainer.explain(np.array([-5.0, -5.0], dtype=np.float32), target_class=0)


@pytest.mark.parametrize(
    ("rule", "factory"),
    [
        ("gamma", lambda: GammaRule(gamma=0.35)),
        ("z_plus", lambda: Alpha1_Beta0_Rule(set_bias_to_zero=True)),
    ],
)
def test_convolutional_gamma_and_z_plus_match_direct_captum(rule, factory):
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8, 1),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[[[1.0]]], [[[0.4]]]]))
        model[0].bias.copy_(torch.tensor([0.2, -0.1]))
        model[3].weight.copy_(torch.tensor([[1.0, -0.3, 0.2, 0.7, -0.4, 0.6, 0.8, 0.1]]))
        model[3].bias.fill_(0.25)
    instance = np.array([[[0.5, 1.0], [2.0, 3.0]]], dtype=np.float32)

    reference = copy.deepcopy(model)
    reference[0].rule = factory()
    reference[2].rule = _ReferenceReshapeRule()
    reference[3].rule = factory()
    expected = (
        CaptumLRP(reference)
        .attribute(torch.tensor(instance).unsqueeze(0), target=0)
        .detach()
        .numpy()
        .reshape(-1)
    )
    kwargs = {"gamma": 0.35} if rule == "gamma" else {}
    actual = _attrs(
        LRPExplainer(
            _adapter(model, task="regression"),
            [f"p{i}" for i in range(4)],
            rule=rule,
            **kwargs,
        ).explain(instance)
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_single_output_regression_reports_output_zero_index():
    explanation = LRPExplainer(
        _adapter(nn.Linear(2, 1, bias=False), task="regression"), ["x0", "x1"]
    ).explain(np.array([1.0, 2.0], dtype=np.float32))

    assert explanation.explanation_data["target_class_index"] == 0


def test_sequential_subclass_with_overridden_graph_is_rejected():
    class BranchedSequential(nn.Sequential):
        def forward(self, x):
            return self[0](x) + self[1](x)

    model = BranchedSequential(nn.Linear(2, 1), nn.Linear(2, 1))

    with pytest.raises(TypeError, match="nn.Sequential"):
        LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])


def test_reused_layer_is_rejected_before_model_work():
    shared = nn.Linear(2, 2, bias=False)
    model = nn.Sequential(shared, nn.ReLU(), shared, nn.Linear(2, 1, bias=False))
    with pytest.raises(TypeError, match="Layer 2 is reused; shared modules are unsupported"):
        LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])


def test_batchnorm_without_running_statistics_is_rejected():
    model = nn.Sequential(nn.BatchNorm1d(2, track_running_stats=False), nn.Linear(2, 1))

    with pytest.raises(TypeError, match="running statistics"):
        LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])


def test_maxpool_returning_indices_is_rejected():
    model = nn.Sequential(nn.MaxPool2d(2, return_indices=True), nn.Flatten(), nn.Linear(1, 1))

    with pytest.raises(TypeError, match="return_indices"):
        LRPExplainer(_adapter(model, task="regression"), [f"p{i}" for i in range(4)])


def _integrity_lrp_model():
    output = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        output.weight.copy_(torch.tensor([[1.0, 0.0]]))
    return nn.Sequential(nn.ReLU(), output)


def test_lrp_canonical_forward_rejects_wrong_feature_swap_and_clean_control():
    model = _integrity_lrp_model()
    model[0].forward = MethodType(lambda self, values: torch.relu(values.flip(1)), model[0])
    with pytest.raises(RuntimeError, match="instance-shadowed forward"):
        LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])

    clean = LRPExplainer(
        _adapter(_integrity_lrp_model(), task="regression"),
        ["x0", "x1"],
        epsilon=0.0,
    ).explain(np.array([2.0, 3.0], dtype=np.float32), target_class=0)
    np.testing.assert_allclose(clean.explanation_data["attributions_raw"], [2.0, 0.0])


@pytest.mark.parametrize("target", ["root_forward", "root_hook", "relu_call_impl"])
def test_lrp_root_and_call_pipeline_integrity_fail_closed(target):
    model = _integrity_lrp_model()
    if target == "root_forward":
        model.forward = MethodType(lambda self, values: values.flip(1), model)
    elif target == "root_hook":
        model.register_forward_hook(lambda _module, _inputs, output: output.flip(1))
    else:
        model[0]._call_impl = MethodType(lambda self, values: values * values, model[0])

    with pytest.raises(RuntimeError, match="instance-shadowed|pre-existing"):
        LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])


@pytest.mark.parametrize(
    "registry_name",
    [
        "_state_dict_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    ],
)
def test_lrp_rejects_state_io_hooks_used_by_captum_restoration(registry_name):
    model = _integrity_lrp_model()
    object.__getattribute__(model, "__dict__")[registry_name][12345] = lambda *_args: None

    with pytest.raises(RuntimeError, match=registry_name):
        LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])


@pytest.mark.parametrize("mutation", ["replace_child", "forward", "hook", "call_impl"])
def test_lrp_post_construction_mutations_are_revalidated_before_model_work(mutation):
    model = _integrity_lrp_model()
    explainer = LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])
    if mutation == "replace_child":
        model[0] = nn.Sigmoid()
        match = "model graph changed"
    elif mutation == "forward":
        model[0].forward = MethodType(lambda self, values: values.flip(1), model[0])
        match = "instance-shadowed forward"
    elif mutation == "hook":
        model[0].register_full_backward_hook(lambda _module, grad_input, _grad_output: grad_input)
        match = "pre-existing"
    else:
        model[0]._call_impl = MethodType(lambda self, values: values * values, model[0])
        match = "instance-shadowed _call_impl"

    with pytest.raises(RuntimeError, match=match):
        explainer.explain(np.array([2.0, 3.0], dtype=np.float32), target_class=0)


def test_lrp_class_forward_monkeypatch_is_not_blessed(monkeypatch):
    model = _integrity_lrp_model()
    explainer = LRPExplainer(_adapter(model, task="regression"), ["x0", "x1"])
    monkeypatch.setattr(nn.ReLU, "forward", lambda self, values: values.flip(1))

    with pytest.raises(RuntimeError, match="canonical forward"):
        explainer.explain(np.array([2.0, 3.0], dtype=np.float32), target_class=0)
