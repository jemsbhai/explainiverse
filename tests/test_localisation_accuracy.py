"""Reference and contract tests for localisation metrics.

Primary formula sources:

* Zhang et al., Pointing Game, section 4.1:
  https://arxiv.org/pdf/1608.00507
* Kohlbrenner et al., Attribution Localisation, equation 5:
  https://arxiv.org/pdf/1910.09840
* Theiner et al., Top-K Intersection, equation 1:
  https://arxiv.org/pdf/2104.14995
* Arras et al., Relevance Mass/Rank Accuracy, equations 1--4 and the
  single-channel positive-map contract in section 3.4:
  https://arxiv.org/pdf/2003.07258
* Arias-Duart et al., Focus, equation 1:
  https://arxiv.org/pdf/2109.15035
* Wang et al., Energy-Based Pointing Game, section 4.3:
  https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.pdf
"""

import warnings

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.localisation import (
    LocalisationMask,
    compute_attribution_iou,
    compute_attribution_localisation,
    compute_auc,
    compute_batch_relevance_mass_accuracy,
    compute_energy_based_pointing_game,
    compute_focus,
    compute_pointing_game,
    compute_relevance_mass_accuracy,
    compute_relevance_rank_accuracy,
    compute_top_k_intersection,
)

quantus = pytest.importorskip("quantus")


def test_pointing_game_matches_official_quantus_for_tied_maxima():
    attributions = np.array([0.9, 0.9, 0.1, 0.0])
    mask = np.array([0, 1, 0, 0])

    reference = quantus.PointingGame(
        abs=False, normalise=False, disable_warnings=True
    ).evaluate_batch(
        a_batch=attributions[None, :],
        s_batch=mask[None, :],
    )[
        0
    ]

    assert compute_pointing_game(attributions, mask) == float(reference)
    assert reference


def test_pointing_game_default_ranks_raw_values_not_magnitude():
    attributions = np.array([-10.0, 0.5, 0.1])
    mask = np.array([0, 1, 0])
    assert compute_pointing_game(attributions, mask) == 1.0
    assert compute_pointing_game(attributions, mask, use_abs=True) == 0.0


def test_attribution_localisation_matches_equation_and_official_quantus():
    attributions = np.array([0.6, -0.5, 0.4, 0.0])
    mask = np.array([1, 1, 0, 0])
    expected = 0.6 / (0.6 + 0.4)

    metric = quantus.AttributionLocalisation(
        positive_attributions=True,
        abs=False,
        normalise=False,
        disable_warnings=True,
    )
    # Quantus 0.6 computes an unused size ratio from x_batch and emits a
    # warning for this direct evaluate_batch call; its returned core ratio is
    # nevertheless the official reference used here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = metric.evaluate_batch(
            x_batch=np.zeros((1, 4)),
            a_batch=attributions[None, :],
            s_batch=mask[None, :],
        )[0]

    assert compute_attribution_localisation(attributions, mask) == pytest.approx(expected)
    assert reference == pytest.approx(expected)


def test_top_k_intersection_matches_equation_and_official_quantus():
    attributions = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    mask = np.array([1, 0, 1, 0, 0])
    expected = 2.0 / 3.0
    reference = quantus.TopKIntersection(
        k=3, abs=False, normalise=False, disable_warnings=True
    ).evaluate_batch(
        a_batch=attributions[None, :],
        s_batch=mask[None, :],
    )[
        0
    ]

    assert compute_top_k_intersection(attributions, mask, k=3) == pytest.approx(expected)
    assert reference == pytest.approx(expected)


def test_top_k_rejects_implicit_k_and_ambiguous_boundary_tie():
    with pytest.raises(ValueError, match="explicit k"):
        compute_top_k_intersection(np.array([0.9, 0.8]), np.array([1, 0]))
    with pytest.raises(ValueError, match="tie straddles"):
        compute_top_k_intersection(
            np.array([1.0, 0.5, 0.5, 0.0]),
            np.array([1, 1, 0, 0]),
            k=2,
        )


def test_relevance_mass_accuracy_matches_arras_and_official_quantus():
    relevance = np.array([0.7, 0.2, 0.1, 0.0])
    mask = np.array([1, 0, 1, 0])
    expected = 0.8
    reference = quantus.RelevanceMassAccuracy(
        abs=False, normalise=False, disable_warnings=True
    ).evaluate_batch(
        a_batch=relevance[None, :],
        s_batch=mask[None, :],
    )[
        0
    ]

    assert compute_relevance_mass_accuracy(relevance, mask) == pytest.approx(expected)
    assert reference == pytest.approx(expected)


def test_rma_scale_normalisation_cannot_shift_signed_values():
    relevance = np.array([0.5, 0.5, 0.5, 0.5])
    mask = np.array([1, 0, 0, 0])
    assert compute_relevance_mass_accuracy(relevance, mask, normalise=True) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="non-negative, pre-pooled"):
        compute_relevance_mass_accuracy(np.array([-1.0, 0.5, 1.0, 0.0]), mask, normalise=True)


def test_relevance_rank_accuracy_matches_arras_and_official_quantus():
    relevance = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    mask = np.array([1, 0, 1, 0, 0])
    expected = 0.5
    reference = quantus.RelevanceRankAccuracy(
        abs=False, normalise=False, disable_warnings=True
    ).evaluate_batch(
        a_batch=relevance[None, :],
        s_batch=mask[None, :],
    )[
        0
    ]

    assert compute_relevance_rank_accuracy(relevance, mask) == pytest.approx(expected)
    assert reference == pytest.approx(expected)


def test_auc_matches_sklearn_and_official_quantus_with_ties():
    scores = np.array([0.9, 0.5, 0.5, 0.1, 0.0])
    mask = np.array([1, 1, 0, 0, 0])
    expected = roc_auc_score(mask, scores)
    reference = quantus.AUC(abs=False, normalise=False, disable_warnings=True).evaluate_batch(
        a_batch=scores[None, :],
        s_batch=mask[None, :],
    )[0]

    assert compute_auc(scores, mask) == pytest.approx(expected)
    assert reference == pytest.approx(expected)


@pytest.mark.parametrize("mask", [np.zeros(3), np.ones(3)])
def test_auc_rejects_one_class_ground_truth(mask):
    with pytest.raises(ValueError, match="undefined|no relevant elements"):
        compute_auc(np.array([0.9, 0.5, 0.1]), mask)


def test_energy_based_pointing_game_matches_scorecam_equation():
    saliency = np.array([[0.6, 0.2], [0.1, 0.1]])
    box = np.array([[1, 1], [0, 0]])
    assert compute_energy_based_pointing_game(saliency, box) == pytest.approx(0.8)


def test_energy_based_pointing_game_rejects_signed_or_zero_energy():
    mask = np.array([1, 0])
    with pytest.raises(ValueError, match="non-negative saliency"):
        compute_energy_based_pointing_game(np.array([-0.5, 1.0]), mask)
    with pytest.raises(ValueError, match="total saliency energy is zero"):
        compute_energy_based_pointing_game(np.zeros(2), mask)


def test_focus_matches_primary_formula_and_official_quantus():
    attributions = np.arange(16, dtype=float).reshape(4, 4)
    # Target images occupy top-left and bottom-left quadrants.
    mask = np.zeros((4, 4))
    mask[:, :2] = 1.0
    expected = np.sum(attributions[:, :2]) / np.sum(attributions)
    reference = quantus.Focus(abs=False, normalise=False, disable_warnings=True).evaluate_batch(
        a_batch=attributions[None, None, :, :],
        c_batch=np.array([[True, False, True, False]]),
    )[0]

    assert compute_focus(attributions, mask) == pytest.approx(expected)
    assert reference == pytest.approx(expected)


def test_focus_quarantines_non_mosaic_mass_ratio():
    with pytest.raises(ValueError, match="2-D, 2-by-2"):
        compute_focus(np.array([0.8, 0.2]), np.array([1, 0]))
    attributions = np.ones((4, 4))
    arbitrary_mask = np.zeros((4, 4))
    arbitrary_mask[0, 0] = 1.0
    with pytest.raises(ValueError, match="complete mosaic quadrants"):
        compute_focus(attributions, arbitrary_mask)


def test_multi_channel_maps_must_be_prepooled():
    # Silent signed channel summation could cancel a genuine attribution map.
    channels = np.stack([np.ones((2, 2)), -np.ones((2, 2))])
    mask = np.array([[1, 0], [0, 0]])
    with pytest.raises(ValueError, match="pre-pooled"):
        compute_attribution_localisation(channels, mask)


def test_equal_size_but_different_spatial_shapes_do_not_align():
    with pytest.raises(ValueError, match="shape .* does not match"):
        compute_auc(np.arange(6.0).reshape(2, 3), np.ones((3, 2)))


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (compute_pointing_game, {}),
        (compute_attribution_localisation, {}),
        (compute_top_k_intersection, {"k": 1}),
        (compute_relevance_mass_accuracy, {}),
        (compute_relevance_rank_accuracy, {}),
        (compute_auc, {}),
        (compute_energy_based_pointing_game, {}),
        (compute_attribution_iou, {"threshold": 0.5}),
    ],
)
def test_all_metrics_reject_nonfinite_attributions(function, kwargs):
    with pytest.raises(ValueError, match="finite"):
        function(np.array([np.nan, 1.0]), np.array([1, 0]), **kwargs)


def test_complex_values_are_not_silently_projected_to_real_numbers():
    with pytest.raises(TypeError, match="not complex"):
        compute_pointing_game(np.array([1.0 + 2.0j, 0.0]), np.array([1, 0]))
    with pytest.raises(TypeError, match="not complex"):
        compute_pointing_game(np.array([1.0, 0.0]), np.array([1 + 2j, 0]))
    with pytest.raises(TypeError, match="numeric or boolean"):
        LocalisationMask(np.array([1 + 2j, 0]))


def test_iou_parameters_are_finite_and_binarisation_is_strict():
    attributions = np.array([0.5, 0.5, 0.1])
    mask = np.array([1, 0, 0])
    # Strict > excludes both values tied exactly at the cutoff.
    assert compute_attribution_iou(attributions, mask, threshold=0.5) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="finite"):
        compute_attribution_iou(attributions, mask, threshold=np.inf)


def test_batch_error_identifies_misaligned_item():
    with pytest.raises(ValueError, match="batch item 1"):
        compute_batch_relevance_mass_accuracy(
            [np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0])],
            [np.array([1, 0]), np.array([1, 0])],
        )


def test_mask_factories_fail_fast_on_invalid_geometry_and_indices():
    with pytest.raises(ValueError, match="y coordinates"):
        LocalisationMask.from_bounding_box(4, 4, -1, 2, 0, 2)
    with pytest.raises(ValueError, match="x coordinates"):
        LocalisationMask.from_bounding_box(4, 4, 0, 2, 3, 3)
    with pytest.raises(TypeError, match="only integers"):
        LocalisationMask.from_feature_indices(4, [1.5])


def test_explanation_missing_named_attribution_is_not_filled_with_zero():
    explanation = Explanation(
        explainer_name="test",
        target_class="target",
        explanation_data={"feature_attributions": {"a": 1.0}},
        feature_names=["a", "b"],
    )
    with pytest.raises(ValueError, match="missing attributions"):
        compute_pointing_game(explanation, np.array([1, 0]))
