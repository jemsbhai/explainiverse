"""Reference validation for Explainiverse DeepLIFT.

The analytical saturation and completeness oracles live in
``tests/test_deeplift.py``.  This file independently verifies the framework's
adapter / target plumbing against Captum on a trained multiclass ReLU model.
"""

import numpy as np


def _values(explanation, feature_names):
    attributions = explanation.get_attributions()
    assert attributions is not None
    return np.asarray([attributions[name] for name in feature_names])


class TestDeepLIFTReferenceAgreement:
    def test_matches_captum_on_trained_multiclass_model(
        self,
        adapted_mlp_multiclass,
        iris_test_instances,
        iris_data,
        captum_deeplift_iris,
    ):
        from explainiverse.adapters import PyTorchAdapter
        from explainiverse.explainers.gradient import DeepLIFTExplainer

        # The trained fixture's custom root is intentionally outside the
        # verified graph contract. Its complete computation is the owned exact
        # Sequential child, which preserves the same weights and outputs.
        exact_adapter = PyTorchAdapter(
            adapted_mlp_multiclass.model.net,
            task="classification",
            feature_names=iris_data["feature_names"],
            class_names=iris_data["class_names"],
        )
        explainer = DeepLIFTExplainer(
            exact_adapter,
            feature_names=iris_data["feature_names"],
            class_names=iris_data["class_names"],
        )

        for index, instance in enumerate(iris_test_instances["instances"]):
            target = int(iris_test_instances["labels"][index])
            explanation = explainer.explain(instance, target_class=target)
            actual = _values(explanation, iris_data["feature_names"])
            expected = captum_deeplift_iris[index]

            np.testing.assert_allclose(
                actual,
                expected,
                atol=1e-6,
                rtol=1e-5,
                err_msg=f"DeepLIFT reference mismatch for Iris instance {index}",
            )
            assert explanation.explanation_data["output_space"] == "model"
            assert explanation.explanation_data["target_index"] == target
