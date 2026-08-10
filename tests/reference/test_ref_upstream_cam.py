"""Direct parity checks for the two quarantined pytorch-grad-cam variants."""

import numpy as np
import pytest

from explainiverse.explainers.gradient.cam_variants import (
    EigenGradCAMExplainer,
    GradCAMElementWiseExplainer,
)

pytorch_grad_cam = pytest.importorskip("pytorch_grad_cam")

ACTIVATIONS = np.array(
    [
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[4.0, 3.0], [2.0, 1.0]],
        ]
    ]
)
GRADIENTS = np.array(
    [
        [
            [[2.0, -1.0], [0.0, 1.0]],
            [[-2.0, 1.0], [3.0, 0.0]],
        ]
    ]
)


def test_gradcam_elementwise_matches_upstream_get_cam_image():
    upstream = pytorch_grad_cam.GradCAMElementWise.get_cam_image(
        None,
        None,
        None,
        None,
        ACTIVATIONS.copy(),
        GRADIENTS.copy(),
        False,
    )[0]
    actual = object.__new__(GradCAMElementWiseExplainer)._compute_cam(
        ACTIVATIONS, GRADIENTS, None, 0
    )
    np.testing.assert_allclose(actual, upstream, rtol=0.0, atol=0.0)


def test_eigengradcam_matches_upstream_projection_up_to_svd_sign():
    upstream = pytorch_grad_cam.EigenGradCAM.get_cam_image(
        None,
        None,
        None,
        None,
        ACTIVATIONS.copy(),
        GRADIENTS.copy(),
        False,
    )[0].astype(np.float64)
    actual = object.__new__(EigenGradCAMExplainer)._compute_cam(ACTIVATIONS, GRADIENTS, None, 0)
    if float(np.vdot(upstream, actual)) < 0.0:
        upstream = -upstream
    np.testing.assert_allclose(actual, upstream, rtol=1e-6, atol=1e-6)
