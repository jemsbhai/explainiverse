"""Reference validation for TCAV's CAV orientation and activation space.

The official TensorFlow TCAV implementation flattens every full bottleneck
tensor, labels the requested concept first (class 0), and negates the binary
linear classifier coefficient to recover the direction toward that concept:
https://github.com/tensorflow/tcav/blob/master/tcav/cav.py
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression

from explainiverse.explainers.gradient.tcav import TCAVExplainer


class IdentityLayerAdapter:
    task = "classification"
    last_gradient_output_space = "model"

    def list_layers(self):
        return ["bottleneck"]

    def get_layer_output(self, inputs, layer_name):
        return np.asarray(inputs, dtype=float)

    def get_layer_gradients(self, inputs, layer_name, target_class=None):
        values = np.asarray(inputs, dtype=float)
        return values, values

    def predict(self, inputs):
        return np.tile([[0.5, 0.5]], (len(inputs), 1))


def test_cav_matches_official_binary_orientation_and_flattening():
    concept = np.array(
        [
            [[[3.0, 1.0], [0.0, -1.0]]],
            [[[4.0, 0.5], [0.2, -0.8]]],
            [[[5.0, 1.2], [-0.2, -1.1]]],
            [[[3.5, 0.8], [0.1, -0.9]]],
            [[[4.5, 1.1], [-0.1, -1.2]]],
            [[[5.5, 0.7], [0.3, -0.7]]],
        ]
    )
    random_concept = -concept

    # Direct transcription of the relevant official CAV binary convention:
    # flatten full bottlenecks, concept label 0, random label 1, then negate
    # coef_[0] because sklearn's binary coefficient points toward label 1.
    reference_x = np.vstack(
        (concept.reshape(len(concept), -1), random_concept.reshape(len(concept), -1))
    )
    reference_y = np.array([0] * len(concept) + [1] * len(random_concept))
    reference_classifier = LogisticRegression(max_iter=1000, solver="lbfgs")
    reference_classifier.fit(reference_x, reference_y)
    reference_cav = -reference_classifier.coef_[0]
    reference_cav /= np.linalg.norm(reference_cav)

    explainer = TCAVExplainer(
        IdentityLayerAdapter(),
        "bottleneck",
        cav_classifier="logistic",
        random_seed=42,
    )
    actual = explainer.learn_concept(
        "concept",
        concept,
        random_concept,
        test_size=0.0,
        min_accuracy=0.0,
    )

    np.testing.assert_allclose(actual.vector, reference_cav, rtol=1e-8, atol=1e-10)
    assert actual.metadata["activation_space"] == "flattened_full_bottleneck"
    assert actual.metadata["cav_direction"] == "toward_concept_label"
