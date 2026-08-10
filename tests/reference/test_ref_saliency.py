"""
Reference validation: Saliency (Explainiverse) vs Saliency (captum).

Saliency maps are the absolute value of the gradient of the output w.r.t. input.
This is a deterministic method — given identical model, input, and target class,
the values MUST match exactly (within floating-point tolerance).

Reference: Simonyan et al., "Deep Inside Convolutional Networks: Visualising
Image Classification Models and Saliency Maps", ICLR 2014.

Canonical implementation: captum.attr.Saliency
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Make helpers importable from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from helpers import assert_numerical_match  # noqa: E402

# Skip all tests if captum is not available
captum = pytest.importorskip("captum")
from captum.attr import Saliency as CaptumSaliency  # noqa: E402


def _extract_attribution_vector(explanation, feature_names):
    """Extract attribution values as a numpy array aligned to feature_names."""
    attrs = explanation.get_attributions()
    assert attrs is not None, "get_attributions() returned None"
    return np.array([attrs.get(fname, 0.0) for fname in feature_names])


class TestSaliencyVsCaptum:
    """Explainiverse Saliency must produce identical values to captum Saliency."""

    def test_multiclass_single_instance(
        self, torch_mlp_multiclass, iris_test_instances, adapted_mlp_multiclass, iris_data
    ):
        """Single Iris instance: Explainiverse vs captum, multiclass."""
        from explainiverse.explainers.gradient.saliency import SaliencyExplainer

        instance = iris_test_instances["instances"][0:1]
        label = int(iris_test_instances["labels"][0])

        # --- captum reference ---
        ref_saliency = CaptumSaliency(torch_mlp_multiclass)
        x_t = torch.FloatTensor(instance).requires_grad_(True)
        ref_attr = ref_saliency.attribute(x_t, target=label)
        ref_values = ref_attr.detach().numpy().flatten()

        # --- explainiverse ---
        explainer = SaliencyExplainer(
            adapted_mlp_multiclass,
            feature_names=iris_data["feature_names"],
        )
        explanation = explainer.explain(instance.flatten(), target_class=label)
        ev_values = _extract_attribution_vector(explanation, iris_data["feature_names"])

        assert_numerical_match(ev_values, ref_values, "Saliency multiclass single")

    def test_multiclass_all_instances(
        self, torch_mlp_multiclass, iris_test_instances, adapted_mlp_multiclass, iris_data
    ):
        """All Iris test instances: Explainiverse vs captum, multiclass."""
        from explainiverse.explainers.gradient.saliency import SaliencyExplainer

        ref_saliency = CaptumSaliency(torch_mlp_multiclass)
        explainer = SaliencyExplainer(
            adapted_mlp_multiclass,
            feature_names=iris_data["feature_names"],
        )

        for i in range(len(iris_test_instances["instances"])):
            instance = iris_test_instances["instances"][i : i + 1]
            label = int(iris_test_instances["labels"][i])

            # captum
            x_t = torch.FloatTensor(instance).requires_grad_(True)
            ref_attr = ref_saliency.attribute(x_t, target=label)
            ref_values = ref_attr.detach().numpy().flatten()

            # explainiverse
            explanation = explainer.explain(instance.flatten(), target_class=label)
            ev_values = _extract_attribution_vector(explanation, iris_data["feature_names"])

            assert_numerical_match(ev_values, ref_values, f"Saliency multiclass instance {i}")

    def test_binary_single_instance(
        self, torch_mlp_binary, bc_test_instances, adapted_mlp_binary, breast_cancer_data
    ):
        """Single Breast Cancer instance: Explainiverse vs captum, binary."""
        from explainiverse.explainers.gradient.saliency import SaliencyExplainer

        instance = bc_test_instances["instances"][0:1]
        label = int(bc_test_instances["labels"][0])

        # captum
        ref_saliency = CaptumSaliency(torch_mlp_binary)
        x_t = torch.FloatTensor(instance).requires_grad_(True)
        ref_attr = ref_saliency.attribute(x_t, target=label)
        ref_values = ref_attr.detach().numpy().flatten()

        # explainiverse
        explainer = SaliencyExplainer(
            adapted_mlp_binary,
            feature_names=breast_cancer_data["feature_names"],
        )
        explanation = explainer.explain(instance.flatten(), target_class=label)
        ev_values = _extract_attribution_vector(explanation, breast_cancer_data["feature_names"])

        assert_numerical_match(ev_values, ref_values, "Saliency binary single")

    def test_binary_all_instances(
        self, torch_mlp_binary, bc_test_instances, adapted_mlp_binary, breast_cancer_data
    ):
        """All Breast Cancer test instances: Explainiverse vs captum, binary."""
        from explainiverse.explainers.gradient.saliency import SaliencyExplainer

        ref_saliency = CaptumSaliency(torch_mlp_binary)
        explainer = SaliencyExplainer(
            adapted_mlp_binary,
            feature_names=breast_cancer_data["feature_names"],
        )

        for i in range(len(bc_test_instances["instances"])):
            instance = bc_test_instances["instances"][i : i + 1]
            label = int(bc_test_instances["labels"][i])

            # captum
            x_t = torch.FloatTensor(instance).requires_grad_(True)
            ref_attr = ref_saliency.attribute(x_t, target=label)
            ref_values = ref_attr.detach().numpy().flatten()

            # explainiverse
            explanation = explainer.explain(instance.flatten(), target_class=label)
            ev_values = _extract_attribution_vector(
                explanation, breast_cancer_data["feature_names"]
            )

            assert_numerical_match(ev_values, ref_values, f"Saliency binary instance {i}")

    def test_attributions_are_absolute_gradients(self, torch_mlp_multiclass, iris_test_instances):
        """
        Sanity check: verify captum Saliency is |gradient|.

        This confirms our reference itself is correct — captum Saliency
        should equal the absolute value of manually computed gradients.
        """
        ref_saliency = CaptumSaliency(torch_mlp_multiclass)

        for i in range(len(iris_test_instances["instances"])):
            instance = iris_test_instances["instances"][i : i + 1]
            label = int(iris_test_instances["labels"][i])

            # Manual gradient
            x_t = torch.FloatTensor(instance).requires_grad_(True)
            out = torch_mlp_multiclass(x_t)
            out[0, label].backward()
            manual_grad = np.abs(x_t.grad.detach().numpy().flatten())

            # captum
            x_t2 = torch.FloatTensor(instance).requires_grad_(True)
            ref_attr = ref_saliency.attribute(x_t2, target=label)
            captum_values = ref_attr.detach().numpy().flatten()

            assert_numerical_match(
                captum_values,
                manual_grad,
                f"captum Saliency vs manual |grad| instance {i}",
            )

    def test_feature_names_present(self, adapted_mlp_multiclass, iris_data, iris_test_instances):
        """Verify explanation.feature_names is set correctly."""
        from explainiverse.explainers.gradient.saliency import SaliencyExplainer

        explainer = SaliencyExplainer(
            adapted_mlp_multiclass,
            feature_names=iris_data["feature_names"],
        )
        instance = iris_test_instances["instances"][0]
        label = int(iris_test_instances["labels"][0])
        explanation = explainer.explain(instance, target_class=label)

        assert explanation.feature_names is not None, "feature_names is None"
        assert (
            explanation.feature_names == iris_data["feature_names"]
        ), f"feature_names mismatch: {explanation.feature_names} vs {iris_data['feature_names']}"

    def test_attribution_keys_match_feature_names(
        self, adapted_mlp_multiclass, iris_data, iris_test_instances
    ):
        """Every attribution key must be an exact feature name."""
        from explainiverse.explainers.gradient.saliency import SaliencyExplainer

        explainer = SaliencyExplainer(
            adapted_mlp_multiclass,
            feature_names=iris_data["feature_names"],
        )
        instance = iris_test_instances["instances"][0]
        label = int(iris_test_instances["labels"][0])
        explanation = explainer.explain(instance, target_class=label)

        attrs = explanation.get_attributions()
        assert attrs is not None, "get_attributions() returned None"

        attr_keys = set(attrs.keys())
        expected_keys = set(iris_data["feature_names"])

        assert attr_keys == expected_keys, (
            f"Key mismatch — extra: {attr_keys - expected_keys}, "
            f"missing: {expected_keys - attr_keys}"
        )
