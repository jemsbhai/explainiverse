"""Equation-level CAM reference tests.

The expected arrays below are direct NumPy transcriptions of the paper
equations and the two explicitly library-defined variants:

* Grad-CAM, Selvaraju et al. equations 1-2:
  https://arxiv.org/abs/1610.02391
* HiResCAM, Draelos & Carin:
  https://arxiv.org/abs/2011.08891
* XGrad-CAM, Fu et al. equations 7-8:
  https://www.bmva-archive.org.uk/bmvc/2020/assets/papers/0631.pdf
* LayerCAM, Jiang et al.:
  http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf
* Eigen-CAM, Muhammad & Yeasin equations 2-3:
  https://arxiv.org/abs/2008.00299
* pytorch-grad-cam library variants:
  https://github.com/jacobgil/pytorch-grad-cam
"""

import numpy as np

from explainiverse.explainers.gradient.cam_variants import (
    EigenCAMExplainer,
    EigenGradCAMExplainer,
    GradCAMElementWiseExplainer,
    HiResCAMExplainer,
    LayerCAMExplainer,
    XGradCAMExplainer,
)
from explainiverse.explainers.gradient.gradcam import GradCAMExplainer

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


def _uninitialized(explainer_class):
    return object.__new__(explainer_class)


def test_gradcam_matches_paper_channel_average_equations():
    weights = GRADIENTS.mean(axis=(2, 3), keepdims=True)
    expected = np.maximum((weights * ACTIVATIONS).sum(axis=1)[0], 0.0)

    actual = GradCAMExplainer._compute_gradcam(ACTIVATIONS, GRADIENTS)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_hirescam_matches_elementwise_product_equation():
    expected = (GRADIENTS * ACTIVATIONS).sum(axis=1)[0]

    actual = _uninitialized(HiResCAMExplainer)._compute_cam(ACTIVATIONS, GRADIENTS, None, 0)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_xgradcam_matches_activation_normalized_equations():
    denominators = ACTIVATIONS.sum(axis=(2, 3))
    weights = (GRADIENTS * ACTIVATIONS).sum(axis=(2, 3)) / denominators
    expected = (weights[:, :, None, None] * ACTIVATIONS).sum(axis=1)[0]

    actual = _uninitialized(XGradCAMExplainer)._compute_cam(ACTIVATIONS, GRADIENTS, None, 0)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_layercam_matches_positive_spatial_gradient_equation():
    expected = (np.maximum(GRADIENTS, 0.0) * ACTIVATIONS).sum(axis=1)[0]

    actual = _uninitialized(LayerCAMExplainer)._compute_cam(ACTIVATIONS, GRADIENTS, None, 0)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_eigencam_matches_raw_activation_projection():
    channels, height, width = ACTIVATIONS.shape[1:]
    matrix = ACTIVATIONS[0].reshape(channels, height * width).T
    _, _, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    expected = matrix @ right_vectors[0]
    if expected[np.argmax(np.abs(expected))] < 0:
        expected = -expected
    expected = expected.reshape(height, width)

    actual = _uninitialized(EigenCAMExplainer)._compute_cam(ACTIVATIONS, None, None, None)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_library_variants_match_upstream_operations():
    expected_elementwise = np.maximum(GRADIENTS * ACTIVATIONS, 0.0).sum(axis=1)[0]
    actual_elementwise = _uninitialized(GradCAMElementWiseExplainer)._compute_cam(
        ACTIVATIONS, GRADIENTS, None, 0
    )
    np.testing.assert_allclose(actual_elementwise, expected_elementwise, rtol=0.0, atol=0.0)

    weighted = GRADIENTS * ACTIVATIONS
    channels, height, width = weighted.shape[1:]
    matrix = weighted[0].reshape(channels, height * width).T
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    expected_eigen = matrix @ right_vectors[0]
    if expected_eigen[np.argmax(np.abs(expected_eigen))] < 0:
        expected_eigen = -expected_eigen
    expected_eigen = expected_eigen.reshape(height, width)

    actual_eigen = _uninitialized(EigenGradCAMExplainer)._compute_cam(
        ACTIVATIONS, GRADIENTS, None, 0
    )
    np.testing.assert_allclose(actual_eigen, expected_eigen, rtol=1e-14, atol=1e-14)
