"""Accuracy, counterexample, and state-isolation gates for CAM explainers."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class ToyCAMNet(nn.Module):
    """Two-channel spatial model with analytically tractable target scores."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        self.head = nn.Linear(8, 2, bias=True)
        with torch.no_grad():
            self.conv.weight[:, 0, 0, 0] = torch.tensor([1.0, -1.0])
            self.conv.bias[:] = torch.tensor([0.0, 5.0])
            self.head.weight.zero_()
            self.head.weight[0] = torch.tensor([2.0, -1.0, 0.0, 1.0, -2.0, 0.0, 1.0, 0.0])
            self.head.bias[:] = torch.tensor([10.0, -3.0])

    def forward(self, inputs):
        activations = self.conv(inputs)
        return self.head(activations.reshape(inputs.shape[0], -1))


class OneLogitCAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)

    def forward(self, inputs):
        return self.conv(inputs).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)


class StatefulCAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(2)
        self.dropout = nn.Dropout2d(0.5)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs):
        activations = self.conv(inputs)
        activations = self.batch_norm(activations)
        activations = self.dropout(activations)
        return self.head(activations.reshape(inputs.shape[0], -1))


class ConstantCAMNet(nn.Module):
    """Produces a spatially uniform, strictly positive target CAM."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.fill_(2.0)

    def forward(self, inputs):
        pooled = self.conv(inputs).mean(dim=(1, 2, 3))
        return torch.stack((pooled, -pooled), dim=1)


@pytest.fixture
def image():
    return np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)


@pytest.fixture
def adapter():
    from explainiverse.adapters import PyTorchAdapter

    return PyTorchAdapter(ToyCAMNet(), task="classification")


def _raw_toy_scores(model, inputs):
    with torch.no_grad():
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        return model(tensor).cpu().numpy()


def test_scorecam_matches_paper_mask_logit_softmax_formula(adapter, image):
    from explainiverse.explainers.gradient import ScoreCAMExplainer
    from explainiverse.explainers.gradient.gradcam import _normalize_cam

    explainer = ScoreCAMExplainer(adapter, "conv", batch_size=1)
    prepared = image[None, ...]
    activations = adapter.get_layer_output(prepared, "conv")
    masks = np.stack([_normalize_cam(channel) for channel in activations[0]])
    masked = masks[:, None, :, :] * image[None, :, :, :]
    logits = _raw_toy_scores(adapter.model, masked)[:, 0]
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    expected = np.sum(weights[:, None, None] * activations[0], axis=0)

    actual = explainer._compute_cam(activations, None, prepared, 0)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    explanation = explainer.explain(image, target_class=0, resize_to_input=False)
    assert explanation.metadata["canonical_paper_method"] is False
    assert explanation.metadata["score_space"] == "raw_model_output"
    assert explanation.metadata["declared_raw_model_output_space"] == "logit"
    assert explanation.metadata["scorecam_variant"] == (
        "paper_algorithm_1_raw_output_channel_softmax"
    )
    assert explanation.metadata["paper_algorithm_1_score_space_match"] is True
    assert explanation.metadata["official_probability_weighting_match"] is False
    assert explanation.metadata["baseline_raw_output_omitted_by_softmax_shift_invariance"]
    assert explanation.metadata["baseline_logit_omitted_by_softmax_shift_invariance"]


def test_scorecam_does_not_label_probability_outputs_as_logits(image):
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import ScoreCAMExplainer

    probability_model = nn.Sequential(ToyCAMNet(), nn.Softmax(dim=1))
    adapter = PyTorchAdapter(
        probability_model,
        task="classification",
        output_activation="none",
        gradient_output="model",
    )

    explanation = ScoreCAMExplainer(adapter, "0.conv", batch_size=1).explain(image, target_class=0)

    assert explanation.metadata["score_space"] == "raw_model_output"
    assert explanation.metadata["declared_raw_model_output_space"] == "unspecified"
    assert explanation.metadata["paper_score_space_match"] is False
    assert explanation.metadata["baseline_logit_omitted_by_softmax_shift_invariance"] is False


def test_ablationcam_does_not_claim_probability_outputs_match_paper(image):
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import AblationCAMExplainer

    probability_model = nn.Sequential(ToyCAMNet(), nn.Softmax(dim=1))
    adapter = PyTorchAdapter(
        probability_model,
        task="classification",
        output_activation="none",
        gradient_output="model",
    )

    explanation = AblationCAMExplainer(adapter, "0.conv", batch_size=1).explain(
        image, target_class=0
    )

    assert explanation.metadata["score_space"] == "raw_model_output"
    assert explanation.metadata["declared_raw_model_output_space"] == "unspecified"
    assert explanation.metadata["paper_score_space_match"] is False
    assert explanation.metadata["canonical_paper_method"] is False


def test_gradient_cam_variant_labels_prediction_score_space_as_noncanonical(image):
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import HiResCAMExplainer

    adapter = PyTorchAdapter(ToyCAMNet(), task="classification", gradient_output="prediction")
    explanation = HiResCAMExplainer(adapter, "conv").explain(image, target_class=0)

    assert explanation.metadata["score_space"] == "prediction"
    assert explanation.metadata["declared_raw_model_output_space"] == "logit"
    assert explanation.metadata["paper_score_space_match"] is False
    assert explanation.metadata["canonical_paper_method"] is False


@pytest.mark.parametrize("method", ["gradcam", "scorecam"])
def test_constant_positive_cam_is_disclosed_even_when_heatmap_is_zero(image, method):
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import GradCAMExplainer, ScoreCAMExplainer

    adapter = PyTorchAdapter(ConstantCAMNet(), task="classification")
    explainer = (
        GradCAMExplainer(adapter, "conv")
        if method == "gradcam"
        else ScoreCAMExplainer(adapter, "conv", batch_size=1)
    )

    explanation = explainer.explain(image, target_class=0, resize_to_input=False)

    assert np.all(np.asarray(explanation.explanation_data["heatmap"]) == 0.0)
    assert explanation.metadata["normalization_degenerate"] is True
    assert explanation.metadata["constant_map_value"] > 0.0
    assert explanation.metadata["normalization_input_min"] == pytest.approx(
        explanation.metadata["normalization_input_max"]
    )


def test_scorecam_rejects_one_logit_two_class_target_mismatch(image):
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import ScoreCAMExplainer

    adapter = PyTorchAdapter(OneLogitCAMNet(), task="classification")
    explainer = ScoreCAMExplainer(adapter, "conv")

    with pytest.raises(ValueError, match="one-logit"):
        explainer.explain(image, target_class=1)


def test_ablationcam_rejects_one_logit_two_class_target_mismatch(image):
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import AblationCAMExplainer

    adapter = PyTorchAdapter(OneLogitCAMNet(), task="classification")
    explainer = AblationCAMExplainer(adapter, "conv")

    with pytest.raises(ValueError, match="one-logit"):
        explainer.explain(image, target_class=1)


def test_ablationcam_matches_true_target_layer_channel_zeroing(adapter, image):
    from explainiverse.explainers.gradient import AblationCAMExplainer

    explainer = AblationCAMExplainer(adapter, "conv", batch_size=1)
    prepared = image[None, ...]
    activations = adapter.get_layer_output(prepared, "conv")

    with torch.no_grad():
        feature_tensor = torch.as_tensor(activations, dtype=torch.float32)
        original_logits = adapter.model.head(feature_tensor.reshape(1, -1))
        original_score = original_logits[0, 0].item()
        ablated_scores = []
        for channel in range(feature_tensor.shape[1]):
            ablated = feature_tensor.clone()
            ablated[:, channel] = 0
            logits = adapter.model.head(ablated.reshape(1, -1))
            ablated_scores.append(logits[0, 0].item())
    weights = (original_score - np.asarray(ablated_scores)) / original_score
    assert weights[0] > 0 and weights[1] < 0  # counterexample to per-weight ReLU
    expected = np.sum(weights[:, None, None] * activations[0], axis=0)

    actual = explainer._compute_cam(activations, None, prepared, 0)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    explanation = explainer.explain(image, target_class=0, resize_to_input=False)
    assert explanation.metadata["intervention"] == "zero_target_layer_channel"
    assert explanation.metadata["paper_score_space_match"] is True


def test_gradcam_restores_training_flags_buffers_gradients_and_hooks(image):
    from explainiverse.adapters import PyTorchAdapter
    from explainiverse.explainers.gradient import GradCAMExplainer

    model = StatefulCAMNet()
    adapter = PyTorchAdapter(model, task="classification")
    model.train()
    model.dropout.eval()  # preserve a deliberately mixed module state
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 1))

    modules = list(model.modules())
    training_before = [module.training for module in modules]
    gradient_objects_before = [parameter.grad for parameter in model.parameters()]
    gradients_before = [parameter.grad.clone() for parameter in model.parameters()]
    running_mean_before = model.batch_norm.running_mean.clone()
    running_var_before = model.batch_norm.running_var.clone()
    forward_hooks_before = len(model.conv._forward_hooks)
    backward_hooks_before = len(model.conv._backward_hooks)

    GradCAMExplainer(adapter, "conv").explain(image, target_class=0)

    assert [module.training for module in modules] == training_before
    for actual, original, expected in zip(
        model.parameters(), gradient_objects_before, gradients_before
    ):
        assert actual.grad is original
        torch.testing.assert_close(actual.grad, expected)
    torch.testing.assert_close(model.batch_norm.running_mean, running_mean_before)
    torch.testing.assert_close(model.batch_norm.running_var, running_var_before)
    assert len(model.conv._forward_hooks) == forward_hooks_before
    assert len(model.conv._backward_hooks) == backward_hooks_before


@pytest.mark.parametrize("explainer_name", ["score", "ablation"])
def test_forward_only_cam_methods_do_not_leak_hooks(adapter, image, explainer_name):
    from explainiverse.explainers.gradient import AblationCAMExplainer, ScoreCAMExplainer

    explainer_class = ScoreCAMExplainer if explainer_name == "score" else AblationCAMExplainer
    explainer = explainer_class(adapter, "conv", batch_size=1)
    before = len(adapter.model.conv._forward_hooks)

    explainer.explain(image, target_class=0)

    assert len(adapter.model.conv._forward_hooks) == before


def test_per_image_batch_targets_are_not_replaced_by_global_target(adapter, image):
    from explainiverse.explainers.gradient import GradCAMExplainer

    images = np.stack((image, image * 0.5), axis=0)
    explanations = GradCAMExplainer(adapter, "conv").explain_batch(
        images, target_class=np.array([0, 1], dtype=np.int64)
    )

    assert [item.explanation_data["target_index"] for item in explanations] == [0, 1]


def test_eigencam_is_explicitly_class_agnostic(adapter, image):
    from explainiverse.explainers.gradient import EigenCAMExplainer

    explainer = EigenCAMExplainer(adapter, "conv")
    explanation = explainer.explain(image)

    assert explanation.target_class == "class_agnostic"
    assert explanation.explanation_data["target_index"] is None
    with pytest.raises(ValueError, match="class-agnostic"):
        explainer.explain(image, target_class=0)


def test_xgradcam_does_not_hide_undefined_zero_sum_channel(adapter):
    from explainiverse.explainers.gradient import XGradCAMExplainer

    activations = np.array([[[[1.0, -1.0], [2.0, -2.0]]]])
    gradients = np.ones_like(activations)
    explainer = XGradCAMExplainer(adapter, "conv")

    with pytest.raises(ValueError, match="spatial activation sum is zero"):
        explainer._compute_cam(activations, gradients, None, 0)


@pytest.mark.parametrize(
    "bad_image",
    [
        np.full((1, 2, 2), np.nan, dtype=np.float32),
        np.empty((1, 0, 2), dtype=np.float32),
        np.ones((2, 1, 2, 2), dtype=np.float32),
    ],
)
def test_single_image_api_rejects_nonfinite_empty_or_batched_inputs(adapter, bad_image):
    from explainiverse.explainers.gradient import GradCAMExplainer

    with pytest.raises(ValueError):
        GradCAMExplainer(adapter, "conv").explain(bad_image, target_class=0)


def test_cam_rejects_boolean_target_and_nonspatial_target_layer(adapter, image):
    from explainiverse.explainers.gradient import GradCAMExplainer

    with pytest.raises(TypeError, match="integer"):
        GradCAMExplainer(adapter, "conv").explain(image, target_class=True)
    with pytest.raises(ValueError, match="spatial tensor"):
        GradCAMExplainer(adapter, "head").explain(image, target_class=0)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_perturbation_cam_batch_size_is_strict(adapter, batch_size):
    from explainiverse.explainers.gradient import ScoreCAMExplainer

    with pytest.raises((TypeError, ValueError), match="positive integer"):
        ScoreCAMExplainer(adapter, "conv", batch_size=batch_size)


def test_gradcamplusplus_claim_is_quarantined(adapter):
    from explainiverse.explainers.gradient import GradCAMExplainer

    with pytest.raises(NotImplementedError, match="second- and third-order"):
        GradCAMExplainer(adapter, "conv", method="gradcam++")


def test_library_variants_disclose_nonpaper_identity(adapter, image):
    from explainiverse.explainers.gradient import EigenGradCAMExplainer, GradCAMElementWiseExplainer

    for explainer_class in (EigenGradCAMExplainer, GradCAMElementWiseExplainer):
        explanation = explainer_class(adapter, "conv").explain(image, target_class=0)
        assert explanation.metadata["canonical_paper_method"] is False
        assert explanation.metadata["variant_origin"] == "pytorch-grad-cam library"
        assert "library variant" in explanation.explainer_name
