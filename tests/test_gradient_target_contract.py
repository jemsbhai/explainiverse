"""Accuracy oracles for gradient-explainer target and score-space semantics."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class SwitchingLogits(nn.Module):
    """Two linear scores whose predicted class switches along [0, 1]."""

    def forward(self, inputs):
        return torch.cat((inputs, 1.0 - inputs), dim=1)


class OneLogitCNN(nn.Module):
    """Minimal CNN with one binary logit and an inspectable conv layer."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)

    def forward(self, inputs):
        features = self.conv(inputs)
        return features.mean(dim=(1, 2, 3)).unsqueeze(1)


def _switching_adapter():
    from explainiverse.adapters import PyTorchAdapter

    return PyTorchAdapter(
        SwitchingLogits(),
        task="classification",
        gradient_output="model",
    )


def _one_logit_adapter():
    from explainiverse.adapters import PyTorchAdapter

    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(-0.5)
    return PyTorchAdapter(model, task="classification")


def test_ig_resolves_default_target_once_without_class_names():
    """IG integrates class 0, rather than switching classes along its path."""
    from explainiverse.explainers.gradient import IntegratedGradientsExplainer

    explainer = IntegratedGradientsExplainer(
        _switching_adapter(),
        feature_names=["x"],
        n_steps=100,
        method="riemann_middle",
    )

    explanation = explainer.explain(
        np.array([1.0], dtype=np.float32),
        baseline=np.array([0.0], dtype=np.float32),
        return_convergence_delta=True,
    )

    assert explanation.target_class == "class_0"
    assert explanation.explanation_data["feature_attributions"]["x"] == pytest.approx(1.0)
    assert explanation.explanation_data["prediction_difference"] == pytest.approx(1.0)
    assert explanation.explanation_data["convergence_delta"] < 1e-7
    assert explanation.metadata["score_space"] == "model"


def test_smoothgrad_holds_original_target_fixed_across_noise():
    """Noisy copies may cross a decision boundary without changing the target."""
    from explainiverse.explainers.gradient import SmoothGradExplainer

    np.random.seed(7)
    explainer = SmoothGradExplainer(
        _switching_adapter(),
        feature_names=["x"],
        n_samples=50,
        noise_scale=2.0,
    )

    explanation = explainer.explain(np.array([1.0], dtype=np.float32))

    assert explanation.target_class == "class_0"
    assert explanation.explanation_data["feature_attributions"]["x"] == pytest.approx(1.0)
    assert explanation.metadata["score_space"] == "model"


def test_saliency_resolves_default_target_without_class_names():
    """Display labels must not control whether the predicted target is fixed."""
    from explainiverse.explainers.gradient import SaliencyExplainer

    explainer = SaliencyExplainer(
        _switching_adapter(),
        feature_names=["x"],
        absolute_value=False,
    )

    explanation = explainer.explain(np.array([1.0], dtype=np.float32))

    assert explanation.target_class == "class_0"
    assert explanation.explanation_data["feature_attributions"]["x"] == pytest.approx(1.0)
    assert explanation.metadata["score_space"] == "model"


@pytest.mark.parametrize("target_class", [True, 1.5, "1"])
@pytest.mark.parametrize("explainer_name", ["ig", "saliency", "smoothgrad"])
def test_flat_gradient_explainers_reject_non_integral_targets(explainer_name, target_class):
    """Explainers must not silently coerce a target to another output index."""
    from explainiverse.explainers.gradient import (
        IntegratedGradientsExplainer,
        SaliencyExplainer,
        SmoothGradExplainer,
    )

    adapter = _switching_adapter()
    if explainer_name == "ig":
        explainer = IntegratedGradientsExplainer(adapter, ["x"], n_steps=2)
    elif explainer_name == "saliency":
        explainer = SaliencyExplainer(adapter, ["x"])
    else:
        explainer = SmoothGradExplainer(adapter, ["x"], n_samples=2)

    with pytest.raises(TypeError, match="target_class must be an integer"):
        explainer.explain(np.array([1.0], dtype=np.float32), target_class=target_class)


@pytest.mark.parametrize("target_class", [-1, 2])
@pytest.mark.parametrize("explainer_name", ["ig", "saliency", "smoothgrad"])
def test_flat_gradient_explainers_reject_out_of_range_targets(explainer_name, target_class):
    """Negative and unavailable output indices fail rather than relabeling output."""
    from explainiverse.explainers.gradient import (
        IntegratedGradientsExplainer,
        SaliencyExplainer,
        SmoothGradExplainer,
    )

    adapter = _switching_adapter()
    if explainer_name == "ig":
        explainer = IntegratedGradientsExplainer(adapter, ["x"], n_steps=2)
    elif explainer_name == "saliency":
        explainer = SaliencyExplainer(adapter, ["x"])
    else:
        explainer = SmoothGradExplainer(adapter, ["x"], n_samples=2)

    with pytest.raises(ValueError, match="target_class"):
        explainer.explain(np.array([1.0], dtype=np.float32), target_class=target_class)


@pytest.mark.parametrize("n_steps", [True, 0, -1, 2.5])
def test_ig_rejects_invalid_integration_step_counts(n_steps):
    """Invalid quadrature sizes fail at construction, before path arithmetic."""
    from explainiverse.explainers.gradient import IntegratedGradientsExplainer

    expected = TypeError if isinstance(n_steps, (bool, float)) else ValueError
    with pytest.raises(expected, match="n_steps"):
        IntegratedGradientsExplainer(_switching_adapter(), ["x"], n_steps=n_steps)


@pytest.mark.parametrize("n_samples", [True, 0, -1, 2.5])
def test_noisy_ig_rejects_invalid_sample_counts(n_samples):
    """Noisy-baseline IG never averages an empty or ill-defined sample set."""
    from explainiverse.explainers.gradient import IntegratedGradientsExplainer

    explainer = IntegratedGradientsExplainer(_switching_adapter(), ["x"], n_steps=2)
    expected = TypeError if isinstance(n_samples, (bool, float)) else ValueError
    with pytest.raises(expected, match="n_samples"):
        explainer.compute_attributions_with_noise(
            np.array([1.0], dtype=np.float32), n_samples=n_samples
        )


def test_saliency_rejects_spatial_and_mismatched_flat_inputs():
    """Saliency preserves its verified flat-feature identity contract."""
    from explainiverse.explainers.gradient import SaliencyExplainer

    explainer = SaliencyExplainer(_switching_adapter(), ["x"])
    with pytest.raises(ValueError, match="Spatial image tensors are not supported"):
        explainer.explain(np.ones((1, 1), dtype=np.float32))

    mismatched = SaliencyExplainer(_switching_adapter(), ["x", "extra"])
    with pytest.raises(ValueError, match="feature count must match"):
        mismatched.explain(np.array([1.0], dtype=np.float32))


@pytest.mark.parametrize("method", ["standard", "noisy_baseline"])
def test_ig_rejects_mismatched_flat_feature_identity(method):
    """Both IG entry points reject incomplete tabular feature identity."""
    from explainiverse.explainers.gradient import IntegratedGradientsExplainer

    explainer = IntegratedGradientsExplainer(
        _switching_adapter(),
        feature_names=["x", "extra"],
        n_steps=2,
    )
    instance = np.array([1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="feature count must match"):
        if method == "standard":
            explainer.explain(instance)
        else:
            explainer.compute_attributions_with_noise(instance, n_samples=2)


@pytest.mark.parametrize("target_class", [0, 1])
def test_one_logit_ig_completeness_for_both_binary_classes(target_class):
    """One-logit IG explains each complementary probability column exactly."""
    from explainiverse.explainers.gradient import IntegratedGradientsExplainer

    adapter = _one_logit_adapter()
    explainer = IntegratedGradientsExplainer(
        adapter,
        feature_names=["x"],
        class_names=["negative", "positive"],
        n_steps=1000,
        method="riemann_trapezoid",
    )
    baseline = np.array([0.0], dtype=np.float32)
    instance = np.array([1.0], dtype=np.float32)

    explanation = explainer.explain(
        instance,
        target_class=target_class,
        baseline=baseline,
        return_convergence_delta=True,
    )
    positive_input = 1.0 / (1.0 + np.exp(-1.5))
    positive_baseline = 1.0 / (1.0 + np.exp(0.5))
    positive_difference = positive_input - positive_baseline
    expected_difference = -positive_difference if target_class == 0 else positive_difference

    attribution = explanation.explanation_data["feature_attributions"]["x"]
    assert attribution == pytest.approx(expected_difference, abs=1e-6)
    assert explanation.explanation_data["prediction_difference"] == pytest.approx(
        expected_difference
    )
    assert explanation.explanation_data["convergence_delta"] < 1e-6
    assert explanation.metadata["score_space"] == "prediction"


def test_one_logit_saliency_and_smoothgrad_classes_are_opposites():
    """Binary class 0 and class 1 differentiate complementary probabilities."""
    from explainiverse.explainers.gradient import SaliencyExplainer, SmoothGradExplainer

    adapter = _one_logit_adapter()
    instance = np.array([0.5], dtype=np.float32)
    saliency = SaliencyExplainer(adapter, feature_names=["x"], absolute_value=False)

    saliency_0 = saliency.explain(instance, target_class=0)
    saliency_1 = saliency.explain(instance, target_class=1)
    gradient_0 = saliency_0.explanation_data["feature_attributions"]["x"]
    gradient_1 = saliency_1.explanation_data["feature_attributions"]["x"]

    positive_probability = 1.0 / (1.0 + np.exp(-0.5))
    expected_positive_gradient = 2.0 * positive_probability * (1.0 - positive_probability)

    assert gradient_0 == pytest.approx(-expected_positive_gradient)
    assert gradient_1 == pytest.approx(expected_positive_gradient)
    assert saliency_0.metadata["score_space"] == "prediction"
    assert saliency_1.metadata["score_space"] == "prediction"

    smoothgrad = SmoothGradExplainer(
        adapter,
        feature_names=["x"],
        n_samples=3,
        noise_scale=0.0,
    )
    smooth_0 = smoothgrad.explain(instance, target_class=0)
    smooth_1 = smoothgrad.explain(instance, target_class=1)
    smooth_gradient_0 = smooth_0.explanation_data["feature_attributions"]["x"]
    smooth_gradient_1 = smooth_1.explanation_data["feature_attributions"]["x"]

    assert smooth_gradient_0 == pytest.approx(-expected_positive_gradient)
    assert smooth_gradient_1 == pytest.approx(expected_positive_gradient)
    assert smooth_0.metadata["score_space"] == "prediction"
    assert smooth_1.metadata["score_space"] == "prediction"


@pytest.mark.parametrize("explainer_name", ["saliency", "smoothgrad"])
def test_flat_gradient_explainers_use_temporary_eval_without_state_leaks(explainer_name):
    """Dropout, BatchNorm, mixed modes, and existing parameter grads survive."""
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import SaliencyExplainer, SmoothGradExplainer

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
    network.output.weight.grad = torch.full_like(network.output.weight, 5.0)
    flags_before = [module.training for module in network.modules()]
    mean_before = network.batch_norm.running_mean.detach().clone()
    gradient_object = network.output.weight.grad

    if explainer_name == "saliency":
        explainer = SaliencyExplainer(adapter, ["a", "b"], absolute_value=False)
    else:
        explainer = SmoothGradExplainer(
            adapter,
            ["a", "b"],
            n_samples=3,
            noise_scale=0.0,
            random_state=17,
        )

    instance = np.array([1.0, -1.0], dtype=np.float32)
    first = explainer.explain(instance, target_class=0)
    second = explainer.explain(instance, target_class=0)

    np.testing.assert_allclose(
        first.explanation_data["attributions_raw"],
        second.explanation_data["attributions_raw"],
    )
    assert [module.training for module in network.modules()] == flags_before
    torch.testing.assert_close(network.batch_norm.running_mean, mean_before)
    assert network.output.weight.grad is gradient_object
    torch.testing.assert_close(
        network.output.weight.grad,
        torch.full_like(network.output.weight, 5.0),
    )


def test_gradcam_default_target_uses_expanded_binary_probabilities():
    """A positive one-logit CNN defaults to class 1, not column-0 argmax."""
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import GradCAMExplainer

    adapter = PyTorchAdapter(OneLogitCNN(), task="classification")
    explainer = GradCAMExplainer(
        adapter,
        target_layer="conv",
        class_names=["negative", "positive"],
        input_layout="chw",
    )

    explanation = explainer.explain(np.ones((1, 4, 4), dtype=np.float32))

    assert explanation.target_class == "positive"
    assert explanation.metadata["score_space"] == "prediction"
    assert np.asarray(explanation.explanation_data["heatmap"]).shape == (4, 4)
