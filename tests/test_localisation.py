# tests/test_localisation.py
"""Contract and analytical tests for localisation metrics.

Tests are organised by component:
  - LocalisationMask dataclass
  - Pointing Game (Zhang et al., 2018)
  - Additional exported localisation diagnostics

Test categories per metric:
  - Basic functionality & return types
  - Parameter variations
  - Image data (2-D / 3-D)
  - Tabular data (1-D)
  - Edge cases (all-zero mask, all-one mask, single element, etc.)
  - Batch operations
  - Error handling & validation
  - Formula-controlled mask/attribution relationships
  - Explanation object input
  - Determinism / reproducibility
"""
import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation.localisation import (
    LocalisationMask,
    compute_attribution_iou,
    compute_attribution_localisation,
    compute_auc,
    compute_batch_attribution_iou,
    compute_batch_attribution_localisation,
    compute_batch_auc,
    compute_batch_energy_based_pointing_game,
    compute_batch_focus,
    compute_batch_pointing_game,
    compute_batch_relevance_mass_accuracy,
    compute_batch_relevance_rank_accuracy,
    compute_batch_top_k_intersection,
    compute_energy_based_pointing_game,
    compute_focus,
    compute_pointing_game,
    compute_relevance_mass_accuracy,
    compute_relevance_rank_accuracy,
    compute_top_k_intersection,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_tabular_mask():
    """1-D binary mask: features 0, 2 are relevant."""
    return np.array([1, 0, 1, 0, 0], dtype=np.float64)


@pytest.fixture
def simple_image_mask():
    """4x4 binary mask with a 2x2 relevant region in the top-left."""
    mask = np.zeros((4, 4), dtype=np.float64)
    mask[0:2, 0:2] = 1.0
    return mask


@pytest.fixture
def simple_explanation():
    """Explanation object with feature attributions."""
    return Explanation(
        explainer_name="TestExplainer",
        target_class="0",
        explanation_data={
            "feature_attributions": {
                "f0": 0.9,
                "f1": 0.1,
                "f2": 0.5,
                "f3": 0.2,
                "f4": 0.05,
            }
        },
        feature_names=["f0", "f1", "f2", "f3", "f4"],
    )


# ============================================================================
# LocalisationMask — Construction & Validation
# ============================================================================


class TestLocalisationMaskConstruction:
    """Tests for LocalisationMask dataclass creation and validation."""

    def test_basic_creation_tabular(self):
        m = LocalisationMask(mask=np.array([1, 0, 1]), mask_type="feature_set")
        assert m.n_relevant == 2
        assert m.n_total == 3
        assert m.is_tabular
        assert not m.is_image

    def test_basic_creation_image(self):
        m = LocalisationMask(mask=np.ones((8, 8)), mask_type="segmentation")
        assert m.n_relevant == 64
        assert m.is_image
        assert not m.is_tabular

    def test_default_mask_type_is_segmentation(self):
        m = LocalisationMask(mask=np.array([1, 0]))
        assert m.mask_type == "segmentation"

    def test_bounding_box_mask_type(self):
        m = LocalisationMask(mask=np.array([0, 1]), mask_type="bounding_box")
        assert m.mask_type == "bounding_box"

    def test_metadata_default_empty(self):
        m = LocalisationMask(mask=np.array([1]))
        assert m.metadata == {}

    def test_metadata_custom(self):
        m = LocalisationMask(
            mask=np.array([1, 0]),
            metadata={"label": "cat", "source": "COCO"},
        )
        assert m.metadata["label"] == "cat"

    def test_mask_cast_to_float64(self):
        m = LocalisationMask(mask=np.array([1, 0], dtype=np.int32))
        assert m.mask.dtype == np.float64

    def test_shape_property(self):
        m = LocalisationMask(mask=np.zeros((3, 5)))
        assert m.shape == (3, 5)

    def test_rejects_non_binary_mask(self):
        with pytest.raises(ValueError, match="binary"):
            LocalisationMask(mask=np.array([0, 0.5, 1]))

    def test_rejects_negative_values(self):
        with pytest.raises(ValueError, match="binary"):
            LocalisationMask(mask=np.array([0, -1, 1]))

    def test_rejects_values_above_one(self):
        with pytest.raises(ValueError, match="binary"):
            LocalisationMask(mask=np.array([0, 2, 1]))

    def test_rejects_non_ndarray(self):
        with pytest.raises(TypeError, match="numpy ndarray"):
            LocalisationMask(mask=[1, 0, 1])

    def test_rejects_empty_mask(self):
        with pytest.raises(ValueError, match="empty"):
            LocalisationMask(mask=np.array([]))

    def test_rejects_invalid_mask_type(self):
        with pytest.raises(ValueError, match="mask_type"):
            LocalisationMask(mask=np.array([1]), mask_type="invalid")

    def test_all_zeros_mask_accepted(self):
        m = LocalisationMask(mask=np.zeros(5))
        assert m.n_relevant == 0

    def test_all_ones_mask_accepted(self):
        m = LocalisationMask(mask=np.ones(5))
        assert m.n_relevant == 5

    def test_3d_mask_rejected_as_ambiguous(self):
        with pytest.raises(ValueError, match="one-dimensional.*two-dimensional"):
            LocalisationMask(mask=np.ones((3, 4, 4)))


class TestLocalisationMaskFactoryMethods:
    """Tests for convenience factory class methods."""

    def test_from_bounding_box_basic(self):
        m = LocalisationMask.from_bounding_box(
            height=10,
            width=10,
            y_min=2,
            y_max=5,
            x_min=3,
            x_max=7,
        )
        assert m.mask_type == "bounding_box"
        assert m.mask.shape == (10, 10)
        assert m.n_relevant == 3 * 4  # (5-2) * (7-3) = 12

    def test_from_bounding_box_full_image(self):
        m = LocalisationMask.from_bounding_box(
            height=4,
            width=4,
            y_min=0,
            y_max=4,
            x_min=0,
            x_max=4,
        )
        assert m.n_relevant == 16

    def test_from_bounding_box_with_metadata(self):
        m = LocalisationMask.from_bounding_box(
            height=10,
            width=10,
            y_min=0,
            y_max=2,
            x_min=0,
            x_max=2,
            label="dog",
        )
        assert m.metadata["label"] == "dog"

    def test_from_feature_indices_basic(self):
        m = LocalisationMask.from_feature_indices(
            n_features=5,
            relevant_indices=[0, 2, 4],
        )
        assert m.mask_type == "feature_set"
        assert m.n_relevant == 3
        np.testing.assert_array_equal(m.mask, [1, 0, 1, 0, 1])

    def test_from_feature_indices_empty(self):
        m = LocalisationMask.from_feature_indices(
            n_features=3,
            relevant_indices=[],
        )
        assert m.n_relevant == 0

    def test_from_feature_indices_invalid_index(self):
        with pytest.raises(ValueError, match="out of range"):
            LocalisationMask.from_feature_indices(
                n_features=3,
                relevant_indices=[5],
            )

    def test_from_feature_indices_negative_index(self):
        with pytest.raises(ValueError, match="out of range"):
            LocalisationMask.from_feature_indices(
                n_features=3,
                relevant_indices=[-1],
            )

    def test_from_feature_indices_with_metadata(self):
        m = LocalisationMask.from_feature_indices(
            n_features=5,
            relevant_indices=[1],
            source="expert",
        )
        assert m.metadata["source"] == "expert"


# ============================================================================
# Pointing Game — Basic Functionality
# ============================================================================


class TestPointingGameBasic:
    """Basic functionality tests for Pointing Game."""

    def test_max_inside_mask_returns_1(self, simple_tabular_mask):
        # Feature 0 has highest abs attribution, and it's in the mask
        a = np.array([0.9, 0.1, 0.3, 0.2, 0.05])
        assert compute_pointing_game(a, simple_tabular_mask) == 1.0

    def test_max_outside_mask_returns_0(self, simple_tabular_mask):
        # Feature 3 has highest abs attribution, and it's NOT in the mask
        a = np.array([0.1, 0.2, 0.3, 0.9, 0.05])
        assert compute_pointing_game(a, simple_tabular_mask) == 0.0

    def test_returns_float(self, simple_tabular_mask):
        a = np.array([0.5, 0.1, 0.3, 0.2, 0.05])
        result = compute_pointing_game(a, simple_tabular_mask)
        assert isinstance(result, float)

    def test_result_is_binary(self, simple_tabular_mask):
        a = np.array([0.5, 0.1, 0.3, 0.2, 0.05])
        result = compute_pointing_game(a, simple_tabular_mask)
        assert result in (0.0, 1.0)

    def test_negative_attributions_with_abs(self, simple_tabular_mask):
        # Feature 0 has largest |attribution| = |-1.0| = 1.0, and it's in mask
        a = np.array([-1.0, 0.1, 0.3, 0.2, 0.05])
        assert compute_pointing_game(a, simple_tabular_mask, use_abs=True) == 1.0

    def test_negative_attributions_without_abs(self, simple_tabular_mask):
        # Without abs, max is at feature 3 (0.9), which is NOT in mask
        a = np.array([-1.0, 0.1, 0.3, 0.9, 0.05])
        assert compute_pointing_game(a, simple_tabular_mask, use_abs=False) == 0.0


class TestPointingGameImageData:
    """Pointing Game with 2-D and 3-D image attribution maps."""

    def test_2d_attribution_2d_mask_hit(self, simple_image_mask):
        a = np.zeros((4, 4))
        a[0, 1] = 1.0  # inside the 2x2 top-left mask
        assert compute_pointing_game(a, simple_image_mask) == 1.0

    def test_2d_attribution_2d_mask_miss(self, simple_image_mask):
        a = np.zeros((4, 4))
        a[3, 3] = 1.0  # outside the mask
        assert compute_pointing_game(a, simple_image_mask) == 0.0

    def test_3d_attribution_2d_mask_rejected(self, simple_image_mask):
        # (C, H, W) attribution, (H, W) mask — channel sum
        a = np.zeros((3, 4, 4))
        a[0, 0, 0] = 0.5
        a[1, 0, 0] = 0.3
        a[2, 0, 0] = 0.2  # total at (0,0) = 1.0, inside mask
        with pytest.raises(ValueError, match="pre-pooled"):
            compute_pointing_game(a, simple_image_mask)

    def test_3d_attribution_2d_mask_miss_rejected(self, simple_image_mask):
        a = np.zeros((3, 4, 4))
        a[:, 3, 3] = 1.0  # outside mask
        with pytest.raises(ValueError, match="pre-pooled"):
            compute_pointing_game(a, simple_image_mask)


class TestPointingGameTolerance:
    """Pointing Game with spatial tolerance."""

    def test_tolerance_0_exact(self):
        mask = np.zeros((8, 8))
        mask[4, 4] = 1.0
        a = np.zeros((8, 8))
        a[4, 5] = 1.0  # max is adjacent, not inside
        assert compute_pointing_game(a, mask, tolerance=0) == 0.0

    def test_tolerance_1_hits_adjacent(self):
        mask = np.zeros((8, 8))
        mask[4, 4] = 1.0
        a = np.zeros((8, 8))
        a[4, 5] = 1.0  # 1 pixel away
        assert compute_pointing_game(a, mask, tolerance=1) == 1.0

    def test_tolerance_at_corner(self):
        mask = np.zeros((8, 8))
        mask[0, 0] = 1.0
        a = np.zeros((8, 8))
        a[1, 1] = 1.0  # diagonal distance
        assert compute_pointing_game(a, mask, tolerance=1) == 1.0

    def test_tolerance_too_far(self):
        mask = np.zeros((8, 8))
        mask[0, 0] = 1.0
        a = np.zeros((8, 8))
        a[4, 4] = 1.0  # 4 pixels away
        assert compute_pointing_game(a, mask, tolerance=1) == 0.0

    def test_tolerance_rejected_for_tabular(self, simple_tabular_mask):
        """Tolerance only applies to spatial (≥2D) masks; 1-D uses flat path."""
        a = np.array([0.1, 0.9, 0.3, 0.2, 0.05])  # max at idx 1, not in mask
        # tolerance is ignored for 1-D
        with pytest.raises(ValueError, match="only for 2-D"):
            compute_pointing_game(a, simple_tabular_mask, tolerance=1)


class TestPointingGameWithLocalisationMask:
    """Pointing Game with LocalisationMask dataclass input."""

    def test_with_feature_set_mask(self):
        m = LocalisationMask.from_feature_indices(
            n_features=5,
            relevant_indices=[0, 2],
        )
        a = np.array([0.9, 0.1, 0.3, 0.2, 0.05])
        assert compute_pointing_game(a, m) == 1.0

    def test_with_bounding_box_mask(self):
        m = LocalisationMask.from_bounding_box(
            height=4,
            width=4,
            y_min=0,
            y_max=2,
            x_min=0,
            x_max=2,
        )
        a = np.zeros((4, 4))
        a[0, 0] = 1.0
        assert compute_pointing_game(a, m) == 1.0

    def test_with_segmentation_mask(self):
        mask_arr = np.zeros((4, 4))
        mask_arr[2, 2] = 1.0
        m = LocalisationMask(mask=mask_arr, mask_type="segmentation")
        a = np.zeros((4, 4))
        a[2, 2] = 1.0
        assert compute_pointing_game(a, m) == 1.0


class TestPointingGameWithExplanation:
    """Pointing Game with Explanation object input."""

    def test_explanation_input_hit(self, simple_explanation, simple_tabular_mask):
        # f0=0.9 is max, and feature 0 is in mask
        assert compute_pointing_game(simple_explanation, simple_tabular_mask) == 1.0

    def test_explanation_input_miss(self, simple_explanation):
        mask = np.array([0, 0, 0, 1, 1])  # features 3,4 relevant
        # f0=0.9 is max, but feature 0 is NOT in this mask
        assert compute_pointing_game(simple_explanation, mask) == 0.0


class TestPointingGameEdgeCases:
    """Edge cases for Pointing Game."""

    def test_single_element_hit(self):
        a = np.array([1.0])
        s = np.array([1.0])
        assert compute_pointing_game(a, s) == 1.0

    def test_single_element_miss(self):
        a = np.array([1.0])
        s = np.array([0.0])
        with pytest.raises(ValueError, match="no relevant elements"):
            compute_pointing_game(a, s)

    def test_all_zero_attributions(self):
        # argmax of all-zero is index 0
        a = np.zeros(5)
        s = np.array([1, 0, 0, 0, 0], dtype=np.float64)
        with pytest.raises(ValueError, match="uniform"):
            compute_pointing_game(a, s)

    def test_all_equal_attributions_hit(self):
        a = np.ones(5) * 0.5
        s = np.array([1, 0, 0, 0, 0], dtype=np.float64)
        # argmax returns first occurrence (idx 0), which is in mask
        with pytest.raises(ValueError, match="uniform"):
            compute_pointing_game(a, s)

    def test_multiple_maxima_any_hit_wins(self):
        # Two equal maxima; np.argmax returns the first
        a = np.array([0.1, 0.9, 0.9, 0.1])
        s = np.array([0, 1, 0, 0], dtype=np.float64)  # idx 1 in mask
        assert compute_pointing_game(a, s) == 1.0

    def test_all_mask_ones(self):
        a = np.array([0.1, 0.9, 0.3])
        s = np.ones(3)
        assert compute_pointing_game(a, s) == 1.0

    def test_large_array(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(10000)
        s = np.zeros(10000)
        max_idx = np.argmax(np.abs(a))
        s[max_idx] = 1.0  # put the mask exactly at the argmax
        assert compute_pointing_game(a, s, use_abs=True) == 1.0


class TestPointingGameErrorHandling:
    """Error handling for Pointing Game."""

    def test_empty_attributions(self):
        with pytest.raises(ValueError, match="empty"):
            compute_pointing_game(np.array([]), np.array([1]))

    def test_empty_mask(self):
        with pytest.raises(ValueError, match="empty"):
            compute_pointing_game(np.array([1.0]), np.array([]))

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="does not match"):
            compute_pointing_game(np.array([1, 2, 3]), np.array([1, 0]))

    def test_invalid_attribution_type(self):
        with pytest.raises(TypeError):
            compute_pointing_game([1, 2, 3], np.array([1, 0, 1]))

    def test_invalid_mask_type(self):
        with pytest.raises(TypeError):
            compute_pointing_game(np.array([1, 2, 3]), [1, 0, 1])

    def test_non_binary_mask(self):
        with pytest.raises(ValueError, match="binary"):
            compute_pointing_game(np.array([1.0]), np.array([0.5]))


class TestPointingGameDeterminism:
    """Reproducibility tests."""

    def test_deterministic_results(self, simple_tabular_mask):
        a = np.array([0.9, 0.1, 0.3, 0.2, 0.05])
        results = [compute_pointing_game(a, simple_tabular_mask) for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestPointingGameFormula:
    """Direct consequences of the Pointing Game definition."""

    def test_perfect_explanation_scores_1(self):
        """If max attribution is always inside mask, score is 1."""
        s = np.array([1, 0, 0, 0, 0], dtype=np.float64)
        a = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert compute_pointing_game(a, s) == 1.0

    def test_worst_explanation_scores_0(self):
        """If max attribution is always outside mask, score is 0."""
        s = np.array([1, 0, 0, 0, 0], dtype=np.float64)
        a = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        assert compute_pointing_game(a, s) == 0.0


# ============================================================================
# Batch Pointing Game
# ============================================================================


class TestBatchPointingGame:
    """Tests for compute_batch_pointing_game."""

    def test_basic_batch(self):
        a_batch = [
            np.array([0.9, 0.1, 0.3]),
            np.array([0.1, 0.9, 0.3]),
        ]
        s_batch = [
            np.array([1, 0, 0], dtype=np.float64),
            np.array([0, 1, 0], dtype=np.float64),
        ]
        results = compute_batch_pointing_game(a_batch, s_batch)
        assert results == [1.0, 1.0]

    def test_batch_mixed_results(self):
        a_batch = [
            np.array([0.9, 0.1]),  # max at 0 — hit
            np.array([0.1, 0.9]),  # max at 1 — miss
        ]
        s_batch = [
            np.array([1, 0], dtype=np.float64),
            np.array([1, 0], dtype=np.float64),
        ]
        results = compute_batch_pointing_game(a_batch, s_batch)
        assert results == [1.0, 0.0]

    def test_batch_with_localisation_masks(self):
        m1 = LocalisationMask.from_feature_indices(3, [0])
        m2 = LocalisationMask.from_feature_indices(3, [2])
        a_batch = [np.array([0.9, 0.1, 0.3]), np.array([0.1, 0.3, 0.9])]
        results = compute_batch_pointing_game(a_batch, [m1, m2])
        assert results == [1.0, 1.0]

    def test_batch_size_mismatch(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            compute_batch_pointing_game(
                [np.array([1.0])],
                [np.array([1.0]), np.array([0.0])],
            )

    def test_batch_empty(self):
        results = compute_batch_pointing_game([], [])
        assert results == []

    def test_batch_single_element(self):
        results = compute_batch_pointing_game(
            [np.array([1.0, 0.5])],
            [np.array([1.0, 0.0])],
        )
        assert results == [1.0]

    def test_batch_with_image_data(self):
        s = np.zeros((4, 4))
        s[0, 0] = 1.0
        a1 = np.zeros((4, 4))
        a1[0, 0] = 1.0  # hit
        a2 = np.zeros((4, 4))
        a2[3, 3] = 1.0  # miss
        results = compute_batch_pointing_game([a1, a2], [s, s])
        assert results == [1.0, 0.0]

    def test_batch_passes_kwargs(self):
        s = np.zeros((8, 8))
        s[4, 4] = 1.0
        a = np.zeros((8, 8))
        a[4, 5] = 1.0  # 1 pixel off
        # tolerance=0 → miss, tolerance=1 → hit
        assert compute_batch_pointing_game([a], [s], tolerance=0) == [0.0]
        assert compute_batch_pointing_game([a], [s], tolerance=1) == [1.0]


# ============================================================================
# Attribution Localisation (Kohlbrenner et al., 2020)
# ============================================================================


class TestAttributionLocalisationBasic:
    """Core functionality of Attribution Localisation."""

    def test_perfect_localisation(self):
        """All positive attributions inside the mask → 1.0."""
        a = np.array([0.5, 0.3, 0.0, 0.0, -0.2])
        s = np.array([1, 1, 0, 0, 0])
        assert compute_attribution_localisation(a, s) == pytest.approx(1.0)

    def test_no_localisation(self):
        """All positive attributions outside the mask → 0.0."""
        a = np.array([0.0, 0.0, 0.5, 0.3, 0.0])
        s = np.array([1, 1, 0, 0, 0])
        assert compute_attribution_localisation(a, s) == pytest.approx(0.0)

    def test_partial_localisation(self):
        """Some positive attributions inside, some outside."""
        a = np.array([0.6, 0.0, 0.4, 0.0])
        s = np.array([1, 0, 0, 0])
        # positive inside = 0.6, total positive = 0.6 + 0.4 = 1.0
        assert compute_attribution_localisation(a, s) == pytest.approx(0.6)

    def test_negative_attributions_ignored(self):
        """Negative attributions are clipped to 0 by default."""
        a = np.array([0.5, -0.8, 0.0, 0.0])
        s = np.array([1, 1, 0, 0])
        # positive: [0.5, 0.0, 0.0, 0.0]; inside mask = 0.5; total = 0.5
        assert compute_attribution_localisation(a, s) == pytest.approx(1.0)

    def test_use_abs(self):
        """use_abs considers absolute values."""
        a = np.array([0.5, -0.3, 0.0, 0.2])
        s = np.array([1, 1, 0, 0])
        # abs: [0.5, 0.3, 0.0, 0.2]; inside = 0.8; total = 1.0
        assert compute_attribution_localisation(a, s, use_abs=True) == pytest.approx(0.8)

    def test_all_zero_attributions(self):
        """The positive-relevance ratio is undefined with zero mass."""
        a = np.zeros(5)
        s = np.array([1, 1, 0, 0, 0])
        with pytest.raises(ValueError, match="total positive relevance is zero"):
            compute_attribution_localisation(a, s)

    def test_all_negative_attributions(self):
        """All-negative inputs also have no positive mass."""
        a = np.array([-0.5, -0.3, -0.1])
        s = np.array([1, 0, 0])
        with pytest.raises(ValueError, match="total positive relevance is zero"):
            compute_attribution_localisation(a, s)

    def test_return_type(self):
        a = np.array([1.0, 0.0])
        s = np.array([1, 0])
        result = compute_attribution_localisation(a, s)
        assert isinstance(result, float)


class TestAttributionLocalisationImage:
    """2-D and 3-D attribution maps."""

    def test_2d_perfect(self):
        a = np.zeros((4, 4))
        a[0:2, 0:2] = 1.0
        s = np.zeros((4, 4))
        s[0:2, 0:2] = 1.0
        assert compute_attribution_localisation(a, s) == pytest.approx(1.0)

    def test_2d_partial(self):
        a = np.ones((4, 4))  # uniform positive
        s = np.zeros((4, 4))
        s[0:2, 0:2] = 1.0  # 4 of 16 pixels
        assert compute_attribution_localisation(a, s) == pytest.approx(4.0 / 16.0)

    def test_3d_with_channels_rejected(self):
        a = np.zeros((3, 4, 4))
        a[:, 0:2, 0:2] = 1.0  # all channels, top-left
        s = np.zeros((4, 4))
        s[0:2, 0:2] = 1.0
        with pytest.raises(ValueError, match="pre-pooled"):
            compute_attribution_localisation(a, s)


class TestAttributionLocalisationWithDataclass:
    """Test with LocalisationMask and Explanation inputs."""

    def test_localisation_mask_input(self):
        a = np.array([0.6, 0.0, 0.4, 0.0])
        mask = LocalisationMask(mask=np.array([1, 0, 0, 0]), mask_type="feature_set")
        assert compute_attribution_localisation(a, mask) == pytest.approx(0.6)

    def test_explanation_input(self):
        from explainiverse.core import Explanation

        exp = Explanation(
            explainer_name="test",
            target_class="pos",
            explanation_data={"feature_attributions": {"f0": 0.8, "f1": 0.2, "f2": 0.0}},
            feature_names=["f0", "f1", "f2"],
        )
        s = np.array([1, 0, 0])
        # f0=0.8 inside mask, f1=0.2 outside, f2=0.0 outside
        # positive inside = 0.8, total positive = 1.0
        assert compute_attribution_localisation(exp, s) == pytest.approx(0.8)

    def test_named_explanation_rejects_extra_feature_identity(self):
        exp = Explanation(
            explainer_name="test",
            target_class="pos",
            explanation_data={"feature_attributions": {"f0": 0.8, "f1": 0.2}},
            feature_names=["f0"],
        )
        with pytest.raises(ValueError, match="unexpected attributions"):
            compute_attribution_localisation(exp, np.array([1]))

    def test_unnamed_explanation_requires_complete_indexed_keys(self):
        ambiguous = Explanation(
            explainer_name="test",
            target_class="pos",
            explanation_data={"feature_attributions": {"age": 0.8, "income": 0.2}},
        )
        with pytest.raises(ValueError, match="without feature_names"):
            compute_attribution_localisation(ambiguous, np.array([1, 0]))

        incomplete = Explanation(
            explainer_name="test",
            target_class="pos",
            explanation_data={"feature_attributions": {"f0": 0.8, "f2": 0.2}},
        )
        with pytest.raises(ValueError, match="cover every zero-based index"):
            compute_attribution_localisation(incomplete, np.array([1, 0]))

    def test_unnamed_indexed_explanation_is_ordered_by_feature_index(self):
        exp = Explanation(
            explainer_name="test",
            target_class="pos",
            explanation_data={"feature_attributions": {"f1": 0.2, "f0": 0.8}},
        )

        assert compute_attribution_localisation(exp, np.array([1, 0])) == pytest.approx(0.8)


class TestAttributionLocalisationSemantic:
    """Semantic validation tests."""

    def test_concentrated_beats_spread(self):
        """Explanation concentrated in GT region scores higher."""
        s = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        a_good = np.array([0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        a_bad = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        assert compute_attribution_localisation(a_good, s) > compute_attribution_localisation(
            a_bad, s
        )


class TestAttributionLocalisationErrorHandling:
    """Error and edge cases."""

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            compute_attribution_localisation(np.ones(5), np.ones(3))

    def test_empty_input(self):
        with pytest.raises(ValueError):
            compute_attribution_localisation(np.array([]), np.array([]))


class TestBatchAttributionLocalisation:
    """Batch processing tests."""

    def test_basic_batch(self):
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.0, 1.0])
        s = np.array([1, 0])
        results = compute_batch_attribution_localisation([a1, a2], [s, s])
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)

    def test_batch_size_mismatch(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            compute_batch_attribution_localisation([np.ones(3)], [np.ones(3), np.ones(3)])

    def test_batch_use_abs(self):
        a = np.array([-0.5, 0.5])
        s = np.array([1, 0])
        results = compute_batch_attribution_localisation([a], [s], use_abs=True)
        assert results[0] == pytest.approx(0.5)


# ============================================================================
# Top-K Intersection (Theiner et al., 2021)
# ============================================================================


class TestTopKIntersectionBasic:
    """Core functionality of Top-K Intersection."""

    def test_perfect_top_k(self):
        """Top-k elements are all inside the mask."""
        a = np.array([0.9, 0.8, 0.1, 0.0])
        s = np.array([1, 1, 0, 0])
        assert compute_top_k_intersection(a, s, k=2) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Top-k elements are all outside the mask."""
        a = np.array([0.0, 0.0, 0.9, 0.8])
        s = np.array([1, 1, 0, 0])
        assert compute_top_k_intersection(a, s, k=2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """One of top-2 is inside the mask."""
        a = np.array([0.9, 0.1, 0.8, 0.0])
        s = np.array([1, 1, 0, 0])
        # top-2 by abs: indices 0 (0.9) and 2 (0.8); s[0]=1, s[2]=0 → 1/2
        assert compute_top_k_intersection(a, s, k=2) == pytest.approx(0.5)

    def test_k_must_be_explicit(self):
        a = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
        s = np.array([1, 1, 1, 0, 0])
        with pytest.raises(ValueError, match="explicit k"):
            compute_top_k_intersection(a, s)

    def test_use_abs_false(self):
        """When use_abs=False, rank by raw value (negatives rank low)."""
        a = np.array([-0.9, 0.5, 0.3])
        s = np.array([1, 0, 0])
        # by raw value, top-1 is index 1 (0.5); s[1]=0 → 0.0
        assert compute_top_k_intersection(a, s, k=1, use_abs=False) == pytest.approx(0.0)
        # by abs, top-1 is index 0 (|-0.9|=0.9); s[0]=1 → 1.0
        assert compute_top_k_intersection(a, s, k=1, use_abs=True) == pytest.approx(1.0)

    def test_return_type(self):
        a = np.array([1.0, 0.0])
        s = np.array([1, 0])
        assert isinstance(compute_top_k_intersection(a, s, k=1), float)


class TestTopKIntersectionEdgeCases:
    """Edge cases and parameter validation."""

    def test_k_equals_n(self):
        """k = total number of elements."""
        a = np.array([0.9, 0.1])
        s = np.array([1, 0])
        # top-2 = all elements; 1 inside mask / 2 = 0.5
        assert compute_top_k_intersection(a, s, k=2) == pytest.approx(0.5)

    def test_k_equals_1(self):
        """k=1 is equivalent to pointing game (but returns float)."""
        a = np.array([0.1, 0.9, 0.0])
        s = np.array([0, 1, 0])
        assert compute_top_k_intersection(a, s, k=1) == pytest.approx(1.0)

    def test_all_zero_mask_raises(self):
        a = np.array([0.5, 0.3])
        s = np.zeros(2)
        with pytest.raises(ValueError, match="no relevant elements"):
            compute_top_k_intersection(a, s, k=1)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            compute_top_k_intersection(np.ones(3), np.ones(3), k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            compute_top_k_intersection(np.ones(3), np.ones(3), k=-1)

    def test_k_exceeds_n_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            compute_top_k_intersection(np.ones(3), np.ones(3), k=5)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            compute_top_k_intersection(np.ones(5), np.ones(3), k=1)


class TestTopKIntersectionImage:
    """2-D attribution maps."""

    def test_2d_top_k(self):
        a = np.zeros((4, 4))
        a[0, 0] = 0.9
        a[0, 1] = 0.8
        s = np.zeros((4, 4))
        s[0, 0] = 1
        s[0, 1] = 1
        assert compute_top_k_intersection(a, s, k=2) == pytest.approx(1.0)


class TestTopKIntersectionSemantic:
    """Semantic validation."""

    def test_good_explanation_scores_higher(self):
        s = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        a_good = np.array([0.9, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        a_bad = np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
        assert compute_top_k_intersection(a_good, s, k=2) > compute_top_k_intersection(
            a_bad, s, k=2
        )


class TestBatchTopKIntersection:
    """Batch processing."""

    def test_basic_batch(self):
        a1 = np.array([0.9, 0.1])
        a2 = np.array([0.1, 0.9])
        s = np.array([1, 0])
        results = compute_batch_top_k_intersection([a1, a2], [s, s], k=1)
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)

    def test_batch_size_mismatch(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            compute_batch_top_k_intersection([np.ones(3)], [np.ones(3), np.ones(3)])


# ============================================================================
# Relevance Mass Accuracy (Arras et al., 2022)
# ============================================================================


class TestRelevanceMassAccuracyBasic:
    """Core functionality of Relevance Mass Accuracy."""

    def test_perfect_mass(self):
        """All positive mass inside mask."""
        a = np.array([0.8, 0.5, 0.0, 0.0])
        s = np.array([1, 1, 0, 0])
        assert compute_relevance_mass_accuracy(a, s) == pytest.approx(1.0)

    def test_no_mass_in_mask(self):
        """No positive mass inside mask."""
        a = np.array([0.0, 0.0, 0.8, 0.5])
        s = np.array([1, 1, 0, 0])
        # After normalisation [0,1]: [0.0, 0.0, 0.375, 1.0] → 0 inside
        # Wait - normalise to [0,1]: min=0, max=0.8, so [0, 0, 1.0, 0.625]
        # inside mask = 0, total = 1.625 → 0.0
        assert compute_relevance_mass_accuracy(a, s) == pytest.approx(0.0)

    def test_scale_normalisation_preserves_ratio(self):
        """Max scaling is valid because it preserves the mass ratio."""
        a = np.array([2.0, 1.0])
        s = np.array([1, 0])
        with_norm = compute_relevance_mass_accuracy(a, s, normalise=True)
        without_norm = compute_relevance_mass_accuracy(a, s, normalise=False)
        assert with_norm == pytest.approx(2.0 / 3.0)
        assert with_norm == pytest.approx(without_norm)

    def test_signed_heatmap_requires_explicit_pooling(self):
        a = np.array([-0.5, 0.1, 0.3])
        s = np.array([1, 1, 0])
        with pytest.raises(ValueError, match="non-negative, pre-pooled"):
            compute_relevance_mass_accuracy(a, s)

    def test_all_zero_raises(self):
        a = np.zeros(4)
        s = np.array([1, 1, 0, 0])
        with pytest.raises(ValueError, match="total relevance is zero"):
            compute_relevance_mass_accuracy(a, s)

    def test_constant_positive_attributions_return_mask_fraction(self):
        a = np.array([0.5, 0.5, 0.5])
        s = np.array([1, 0, 0])
        assert compute_relevance_mass_accuracy(a, s, normalise=True) == pytest.approx(1.0 / 3.0)

    def test_use_abs(self):
        """use_abs takes absolute before normalisation."""
        a = np.array([-0.9, 0.1, 0.0])
        s = np.array([1, 0, 0])
        # abs: [0.9, 0.1, 0.0]; normalise: [1.0, 0.111, 0.0]
        # inside = 1.0, total = 1.111 → ~0.9
        result = compute_relevance_mass_accuracy(a, s, use_abs=True, normalise=True)
        assert result > 0.8

    def test_return_type(self):
        a = np.array([1.0, 0.0])
        s = np.array([1, 0])
        assert isinstance(compute_relevance_mass_accuracy(a, s), float)


class TestRelevanceMassAccuracyVsAttributionLocalisation:
    """Verify relationship to Attribution Localisation."""

    def test_equivalent_without_normalisation(self):
        """For a non-negative map, RMA equals Attribution Localisation."""
        a = np.array([0.6, 0.2, 0.4, 0.1])
        s = np.array([1, 0, 0, 1])
        al = compute_attribution_localisation(a, s)
        rma = compute_relevance_mass_accuracy(a, s, normalise=False)
        assert al == pytest.approx(rma)

    def test_scale_normalisation_does_not_change_rma(self):
        a = np.array([1.0, 0.5, 0.25, 0.0])
        s = np.array([1, 1, 0, 0])
        raw = compute_relevance_mass_accuracy(a, s, normalise=False)
        scaled = compute_relevance_mass_accuracy(a, s, normalise=True)
        assert scaled == pytest.approx(raw)


class TestRelevanceMassAccuracyImage:
    """2-D and 3-D inputs."""

    def test_2d_input(self):
        a = np.zeros((4, 4))
        a[0:2, 0:2] = 1.0
        s = np.zeros((4, 4))
        s[0:2, 0:2] = 1.0
        assert compute_relevance_mass_accuracy(a, s) == pytest.approx(1.0)


class TestBatchRelevanceMassAccuracy:
    """Batch processing."""

    def test_basic_batch(self):
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.0, 1.0])
        s = np.array([1, 0])
        results = compute_batch_relevance_mass_accuracy([a1, a2], [s, s])
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)

    def test_batch_size_mismatch(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            compute_batch_relevance_mass_accuracy([np.ones(3)], [np.ones(3), np.ones(3)])

    def test_batch_normalise_kwarg(self):
        a = np.array([0.5, 0.5, 0.5])
        s = np.array([1, 0, 0])
        results = compute_batch_relevance_mass_accuracy([a], [s], normalise=True)
        assert results[0] == pytest.approx(1.0 / 3.0)


# ============================================================================
# Relevance Rank Accuracy (Arras et al., 2022)
# ============================================================================


class TestRelevanceRankAccuracyBasic:
    """Core functionality of Relevance Rank Accuracy."""

    def test_perfect_rank(self):
        """Top-|s| elements are all in the mask."""
        a = np.array([0.9, 0.8, 0.1, 0.0])
        s = np.array([1, 1, 0, 0])
        assert compute_relevance_rank_accuracy(a, s) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Top-|s| elements all outside mask."""
        a = np.array([0.0, 0.0, 0.9, 0.8])
        s = np.array([1, 1, 0, 0])
        assert compute_relevance_rank_accuracy(a, s) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Half of top-|s| inside mask."""
        a = np.array([0.9, 0.1, 0.8, 0.0])
        s = np.array([1, 1, 0, 0])  # |s|=2
        # top-2: indices 0 (0.9) and 2 (0.8); s[0]=1, s[2]=0 → 1/2
        assert compute_relevance_rank_accuracy(a, s) == pytest.approx(0.5)

    def test_use_abs_false(self):
        a = np.array([-0.9, 0.5, 0.3])
        s = np.array([1, 0, 0])  # |s|=1
        with pytest.raises(ValueError, match="non-negative, pre-pooled"):
            compute_relevance_rank_accuracy(a, s, use_abs=False)
        assert compute_relevance_rank_accuracy(a, s, use_abs=True) == pytest.approx(1.0)

    def test_all_zero_mask_raises(self):
        a = np.array([0.5, 0.3])
        s = np.zeros(2)
        with pytest.raises(ValueError, match="no relevant elements"):
            compute_relevance_rank_accuracy(a, s)

    def test_return_type(self):
        a = np.array([1.0, 0.0])
        s = np.array([1, 0])
        assert isinstance(compute_relevance_rank_accuracy(a, s), float)


class TestRelevanceRankAccuracyVsTopK:
    """Verify relationship to Top-K Intersection."""

    def test_equivalent_when_k_equals_mask_size(self):
        """RRA == TKI when k=|s| (and |s| is the normaliser for both)."""
        a = np.array([0.9, 0.1, 0.8, 0.3, 0.0])
        s = np.array([1, 1, 0, 0, 0])  # |s|=2
        rra = compute_relevance_rank_accuracy(a, s)
        tki = compute_top_k_intersection(a, s, k=2)
        assert rra == pytest.approx(tki)


class TestRelevanceRankAccuracyImage:
    """2-D and 3-D inputs."""

    def test_2d_input(self):
        a = np.zeros((4, 4))
        a[0, 0] = 0.9
        a[1, 1] = 0.8
        s = np.zeros((4, 4))
        s[0, 0] = 1
        s[1, 1] = 1
        assert compute_relevance_rank_accuracy(a, s) == pytest.approx(1.0)


class TestRelevanceRankAccuracySemantic:
    """Semantic validation."""

    def test_targeted_beats_uniform(self):
        s = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        a_good = np.array([0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        a_bad = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        assert compute_relevance_rank_accuracy(a_good, s) == pytest.approx(1.0)
        with pytest.raises(ValueError, match="tie straddles"):
            compute_relevance_rank_accuracy(a_bad, s)


class TestBatchRelevanceRankAccuracy:
    """Batch processing."""

    def test_basic_batch(self):
        a1 = np.array([0.9, 0.1])
        a2 = np.array([0.1, 0.9])
        s = np.array([1, 0])
        results = compute_batch_relevance_rank_accuracy([a1, a2], [s, s])
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)

    def test_batch_size_mismatch(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            compute_batch_relevance_rank_accuracy([np.ones(3)], [np.ones(3), np.ones(3)])


# ============================================================================
# AUC (Fawcett, 2006)
# ============================================================================


class TestAUCBasic:
    """Core functionality of ROC-AUC localisation metric."""

    def test_perfect_separation(self):
        """Attributions perfectly separate relevant from irrelevant."""
        a = np.array([0.9, 0.8, 0.1, 0.0])
        s = np.array([1, 1, 0, 0])
        assert compute_auc(a, s) == pytest.approx(1.0)

    def test_inverse_separation(self):
        """Attributions anti-correlated with mask → AUC near 0."""
        a = np.array([0.0, 0.1, 0.8, 0.9])
        s = np.array([1, 1, 0, 0])
        assert compute_auc(a, s) == pytest.approx(0.0)

    def test_random_scores(self):
        """Uniform attributions → AUC ~0.5."""
        a = np.array([0.5, 0.5, 0.5, 0.5])
        s = np.array([1, 1, 0, 0])
        assert compute_auc(a, s) == pytest.approx(0.5)

    def test_partial_separation(self):
        """Partial overlap → AUC between 0.5 and 1.0."""
        a = np.array([0.9, 0.3, 0.7, 0.1])
        s = np.array([1, 1, 0, 0])
        result = compute_auc(a, s)
        assert 0.0 < result < 1.0

    def test_use_abs(self):
        """Negative attributions become positive with use_abs=True."""
        a = np.array([-0.9, -0.8, 0.1, 0.0])
        s = np.array([1, 1, 0, 0])
        # abs: [0.9, 0.8, 0.1, 0.0] → perfect separation
        assert compute_auc(a, s, use_abs=True) == pytest.approx(1.0)

    def test_use_abs_false(self):
        """Without abs, negative attributions rank low."""
        a = np.array([-0.9, -0.8, 0.1, 0.0])
        s = np.array([1, 1, 0, 0])
        # Raw: [-0.9, -0.8, 0.1, 0.0]; positives rank lowest → AUC = 0
        assert compute_auc(a, s, use_abs=False) == pytest.approx(0.0)

    def test_return_type(self):
        a = np.array([1.0, 0.0])
        s = np.array([1, 0])
        assert isinstance(compute_auc(a, s), float)


class TestAUCEdgeCases:
    """Edge cases for AUC."""

    def test_all_zero_mask_raises(self):
        a = np.array([0.5, 0.3])
        s = np.zeros(2)
        with pytest.raises(ValueError, match="no relevant elements"):
            compute_auc(a, s)

    def test_all_one_mask_raises(self):
        a = np.array([0.5, 0.3])
        s = np.ones(2)
        with pytest.raises(ValueError, match="no negative elements"):
            compute_auc(a, s)

    def test_two_elements_perfect(self):
        """Smallest non-degenerate case."""
        a = np.array([1.0, 0.0])
        s = np.array([1, 0])
        assert compute_auc(a, s) == pytest.approx(1.0)

    def test_ties_handled(self):
        """Tied scores should not crash and should give 0.5-like result."""
        a = np.array([0.5, 0.5, 0.5, 0.5])
        s = np.array([1, 0, 1, 0])
        assert compute_auc(a, s) == pytest.approx(0.5)


class TestAUCImage:
    """2-D attribution maps."""

    def test_2d_perfect(self):
        a = np.zeros((4, 4))
        a[0:2, :] = 1.0
        s = np.zeros((4, 4))
        s[0:2, :] = 1.0
        assert compute_auc(a, s) == pytest.approx(1.0)


class TestAUCSemantic:
    """Semantic validation."""

    def test_good_beats_bad(self):
        s = np.array([1, 1, 0, 0, 0, 0])
        a_good = np.array([0.9, 0.8, 0.1, 0.0, 0.0, 0.0])
        a_bad = np.array([0.0, 0.0, 0.1, 0.2, 0.8, 0.9])
        assert compute_auc(a_good, s) > compute_auc(a_bad, s)

    def test_against_sklearn_formula(self):
        """Hand-verified AUC for a small example."""
        # Scores: [0.9, 0.4, 0.6, 0.1], Labels: [1, 0, 1, 0]
        # Sorted desc by score: (0.9,1), (0.6,1), (0.4,0), (0.1,0)
        # TPR/FPR: (0,0)→(0.5,0)→(1.0,0)→(1.0,0.5)→(1.0,1.0)
        # AUC = 1.0
        a = np.array([0.9, 0.4, 0.6, 0.1])
        s = np.array([1, 0, 1, 0])
        assert compute_auc(a, s) == pytest.approx(1.0)


class TestBatchAUC:
    """Batch processing."""

    def test_basic_batch(self):
        a1 = np.array([0.9, 0.1])
        a2 = np.array([0.1, 0.9])
        s = np.array([1, 0])
        results = compute_batch_auc([a1, a2], [s, s])
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)

    def test_batch_size_mismatch(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            compute_batch_auc([np.ones(3)], [np.ones(3), np.ones(3)])


# ============================================================================
# Energy-Based Pointing Game (Wang et al., 2020)
# ============================================================================


class TestEBPGBasic:
    """Core functionality of Energy-Based Pointing Game."""

    def test_all_energy_inside(self):
        """All attribution energy inside mask → 1.0."""
        a = np.array([0.5, 0.3, 0.0, 0.0])
        s = np.array([1, 1, 0, 0])
        assert compute_energy_based_pointing_game(a, s) == pytest.approx(1.0)

    def test_no_energy_inside(self):
        """All attribution energy outside mask → 0.0."""
        a = np.array([0.0, 0.0, 0.5, 0.3])
        s = np.array([1, 1, 0, 0])
        assert compute_energy_based_pointing_game(a, s) == pytest.approx(0.0)

    def test_partial_energy(self):
        """Half the energy inside."""
        a = np.array([0.5, 0.0, 0.5, 0.0])
        s = np.array([1, 0, 0, 0])
        # inside = 0.5, total = 1.0
        assert compute_energy_based_pointing_game(a, s) == pytest.approx(0.5)

    def test_negative_attributions_rejected(self):
        a = np.array([0.8, -0.2, 0.0, 0.0])
        s = np.array([1, 1, 0, 0])
        with pytest.raises(ValueError, match="non-negative saliency"):
            compute_energy_based_pointing_game(a, s)

    def test_signed_map_cannot_produce_out_of_range_score(self):
        a = np.array([-0.5, 0.0, 0.8, 0.0])
        s = np.array([1, 0, 0, 0])
        with pytest.raises(ValueError, match="non-negative saliency"):
            compute_energy_based_pointing_game(a, s)

    def test_use_abs(self):
        """use_abs makes all attributions positive."""
        a = np.array([-0.5, 0.0, 0.3, 0.0])
        s = np.array([1, 0, 0, 0])
        # abs: [0.5, 0.0, 0.3, 0.0]; inside = 0.5, total = 0.8
        assert compute_energy_based_pointing_game(a, s, use_abs=True) == pytest.approx(0.5 / 0.8)

    def test_all_zero_raises(self):
        a = np.zeros(4)
        s = np.array([1, 1, 0, 0])
        with pytest.raises(ValueError, match="total saliency energy is zero"):
            compute_energy_based_pointing_game(a, s)

    def test_return_type(self):
        a = np.array([1.0, 0.0])
        s = np.array([1, 0])
        assert isinstance(compute_energy_based_pointing_game(a, s), float)


class TestEBPGVsAttributionLocalisation:
    """Verify difference from Attribution Localisation."""

    def test_equivalent_for_nonnegative(self):
        """For non-negative attributions, EBPG == AL."""
        a = np.array([0.6, 0.2, 0.4, 0.1])
        s = np.array([1, 1, 0, 0])
        al = compute_attribution_localisation(a, s)
        ebpg = compute_energy_based_pointing_game(a, s)
        assert al == pytest.approx(ebpg)

    def test_signed_input_requires_explicit_preprocessing(self):
        a = np.array([0.5, -0.3, 0.4, 0.0])
        s = np.array([1, 1, 0, 0])
        with pytest.raises(ValueError, match="non-negative saliency"):
            compute_energy_based_pointing_game(a, s)


class TestEBPGImage:
    """2-D inputs."""

    def test_2d_input(self):
        a = np.zeros((4, 4))
        a[0:2, 0:2] = 1.0
        s = np.zeros((4, 4))
        s[0:2, 0:2] = 1.0
        assert compute_energy_based_pointing_game(a, s) == pytest.approx(1.0)


class TestBatchEBPG:
    """Batch processing."""

    def test_basic_batch(self):
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.0, 1.0])
        s = np.array([1, 0])
        results = compute_batch_energy_based_pointing_game([a1, a2], [s, s])
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)

    def test_batch_size_mismatch(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            compute_batch_energy_based_pointing_game([np.ones(3)], [np.ones(3), np.ones(3)])


# ============================================================================
# Focus (Arias-Duart et al., 2022)
# ============================================================================


class TestFocusBasic:
    """Core functionality of Focus metric."""

    def test_perfect_focus(self):
        a = np.zeros((4, 4))
        a[:2, :] = 1.0
        s = np.zeros((4, 4))
        s[:2, :] = 1.0
        assert compute_focus(a, s) == pytest.approx(1.0)

    def test_no_focus(self):
        a = np.zeros((4, 4))
        a[2:, :] = 1.0
        s = np.zeros((4, 4))
        s[:2, :] = 1.0
        assert compute_focus(a, s) == pytest.approx(0.0)

    def test_partial_focus(self):
        a = np.ones((4, 4))
        s = np.zeros((4, 4))
        s[:2, :] = 1.0
        assert compute_focus(a, s) == pytest.approx(0.5)

    def test_negative_attributions_clipped(self):
        a = np.full((4, 4), -1.0)
        a[:2, :] = 0.5
        s = np.zeros((4, 4))
        s[:2, :] = 1.0
        assert compute_focus(a, s) == pytest.approx(1.0)

    def test_all_zero_raises(self):
        a = np.zeros((4, 4))
        s = np.zeros((4, 4))
        s[:2, :] = 1.0
        with pytest.raises(ValueError, match="total positive relevance is zero"):
            compute_focus(a, s)

    def test_return_type(self):
        a = np.ones((2, 2))
        s = np.array([[1, 1], [0, 0]])
        assert isinstance(compute_focus(a, s), float)


class TestFocusEquivalenceToAL:
    """The ratio matches AL once the canonical mosaic is established."""

    def test_equivalent_to_al_default(self):
        a = np.arange(16, dtype=float).reshape(4, 4) - 4.0
        s = np.zeros((4, 4))
        s[:, :2] = 1.0
        al = compute_attribution_localisation(a, s, use_abs=False)
        focus = compute_focus(a, s)
        assert al == pytest.approx(focus)


class TestFocusMosaicScenario:
    """Simulated mosaic evaluation scenario."""

    def test_2x2_mosaic_correct_tile(self):
        a = np.zeros((8, 8))
        a[0:4, 0:4] = 0.9
        a[4:8, 4:8] = 0.9
        a[0:4, 4:8] = 0.05

        s = np.zeros((8, 8))
        s[0:4, 0:4] = 1.0
        s[4:8, 4:8] = 1.0

        result = compute_focus(a, s)
        assert result > 0.8

    def test_2x2_mosaic_wrong_tile(self):
        a = np.zeros((8, 8))
        a[0:4, 4:8] = 0.9
        a[4:8, 0:4] = 0.9

        s = np.zeros((8, 8))
        s[0:4, 0:4] = 1.0
        s[4:8, 4:8] = 1.0

        result = compute_focus(a, s)
        assert result == pytest.approx(0.0)

    def test_arbitrary_region_rejected(self):
        a = np.ones((8, 8))
        s = np.zeros((8, 8))
        s[1:3, 1:3] = 1.0
        with pytest.raises(ValueError, match="complete mosaic quadrants"):
            compute_focus(a, s)

    def test_one_target_quadrant_rejected(self):
        a = np.ones((8, 8))
        s = np.zeros((8, 8))
        s[:4, :4] = 1.0
        with pytest.raises(ValueError, match="exactly two"):
            compute_focus(a, s)


class TestBatchFocus:
    """Batch processing."""

    def test_basic_batch(self):
        a1 = np.array([[1.0, 1.0], [0.0, 0.0]])
        a2 = np.array([[0.0, 0.0], [1.0, 1.0]])
        s = np.array([[1, 1], [0, 0]])
        results = compute_batch_focus([a1, a2], [s, s])
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)

    def test_batch_size_mismatch(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            compute_batch_focus([np.ones(3)], [np.ones(3), np.ones(3)])


# ============================================================================
# Attribution IoU
# ============================================================================


class TestAttributionIoUBasic:
    """Core functionality of Attribution IoU."""

    def test_perfect_iou(self):
        """Binarised attribution exactly matches mask → IoU=1.0."""
        a = np.array([0.9, 0.8, 0.0, 0.0])
        s = np.array([1, 1, 0, 0])
        # threshold=0.5: binarised = [1,1,0,0] = s
        assert compute_attribution_iou(a, s, threshold=0.5) == pytest.approx(1.0)

    def test_no_overlap(self):
        """No intersection between binarised attribution and mask."""
        a = np.array([0.0, 0.0, 0.9, 0.8])
        s = np.array([1, 1, 0, 0])
        assert compute_attribution_iou(a, s, threshold=0.5) == pytest.approx(0.0)

    def test_partial_iou(self):
        """Partial overlap."""
        a = np.array([0.9, 0.1, 0.8, 0.0])
        s = np.array([1, 1, 0, 0])
        # threshold=0.5: binarised = [1,0,1,0]
        # intersection with s = {idx 0} = 1
        # union = {0,1,2} = 3
        assert compute_attribution_iou(a, s, threshold=0.5) == pytest.approx(1.0 / 3.0)

    def test_threshold_effect(self):
        """Different thresholds give different results."""
        a = np.array([0.9, 0.6, 0.3, 0.0])
        s = np.array([1, 1, 0, 0])
        # threshold=0.5: binarised = [1,1,0,0] → IoU=1.0
        # threshold=0.2: binarised = [1,1,1,0] → IoU=2/3
        iou_high = compute_attribution_iou(a, s, threshold=0.5)
        iou_low = compute_attribution_iou(a, s, threshold=0.2)
        assert iou_high > iou_low

    def test_use_abs(self):
        """use_abs=True applies threshold to |a|."""
        a = np.array([-0.9, 0.0, 0.1, 0.0])
        s = np.array([1, 0, 0, 0])
        # abs: [0.9, 0.0, 0.1, 0.0]; threshold=0.5: [1,0,0,0]
        assert compute_attribution_iou(a, s, threshold=0.5, use_abs=True) == pytest.approx(1.0)

    def test_use_abs_false(self):
        """use_abs=False: negative values stay negative, below threshold."""
        a = np.array([-0.9, 0.0, 0.8, 0.0])
        s = np.array([1, 0, 0, 0])
        # raw: [-0.9, 0.0, 0.8, 0.0]; threshold=0.5: [0,0,1,0]
        # intersection = 0, union = {0,2} = 2 → 0.0
        assert compute_attribution_iou(a, s, threshold=0.5, use_abs=False) == pytest.approx(0.0)

    def test_return_type(self):
        a = np.array([1.0, 0.0])
        s = np.array([1, 0])
        assert isinstance(compute_attribution_iou(a, s, threshold=0.5), float)


class TestAttributionIoUPercentile:
    """Percentile-based thresholding."""

    def test_percentile_basic(self):
        """percentile=50 uses the median as threshold."""
        a = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        s = np.array([0, 0, 0, 1, 1, 1])
        # np.percentile([0,.2,.4,.6,.8,1], 50) = 0.5 (median)
        # elements > 0.5: [0.6, 0.8, 1.0] → binarised = [0,0,0,1,1,1]
        # intersection = {3,4,5} = 3, union = {3,4,5} = 3 → IoU = 1.0
        result = compute_attribution_iou(a, s, percentile=50)
        assert result == pytest.approx(1.0)

    def test_percentile_produces_partial_overlap(self):
        """Lower percentile creates wider binarisation → partial IoU."""
        a = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        s = np.array([0, 0, 0, 1, 1, 1])
        # np.percentile(..., 25) = 0.25; elements > 0.25: [0.4,0.6,0.8,1.0]
        # binarised = [0,0,1,1,1,1]; intersection={3,4,5}=3, union={2,3,4,5}=4
        result = compute_attribution_iou(a, s, percentile=25)
        assert result == pytest.approx(3.0 / 4.0)

    def test_percentile_0(self):
        """percentile=0 → threshold = min value."""
        a = np.array([0.5, 0.5, 0.5])
        s = np.array([1, 0, 0])
        # threshold = 0.5; elements > 0.5: none → IoU = 0.0
        # (since > is strict)
        result = compute_attribution_iou(a, s, percentile=0)
        assert result == pytest.approx(0.0)

    def test_percentile_invalid_raises(self):
        with pytest.raises(ValueError, match="percentile"):
            compute_attribution_iou(np.ones(3), np.ones(3), percentile=101)
        with pytest.raises(ValueError, match="percentile"):
            compute_attribution_iou(np.ones(3), np.ones(3), percentile=-1)


class TestAttributionIoUParameterValidation:
    """Parameter validation."""

    def test_neither_threshold_nor_percentile_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            compute_attribution_iou(np.ones(3), np.ones(3))

    def test_both_threshold_and_percentile_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            compute_attribution_iou(np.ones(3), np.ones(3), threshold=0.5, percentile=50)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="does not match"):
            compute_attribution_iou(np.ones(5), np.ones(3), threshold=0.5)


class TestAttributionIoUEdgeCases:
    """Edge cases."""

    def test_empty_ground_truth_region_raises(self):
        a = np.array([0.0, 0.0, 0.0])
        s = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="no relevant elements"):
            compute_attribution_iou(a, s, threshold=0.5)

    def test_high_threshold_no_selection(self):
        """Threshold so high nothing is selected."""
        a = np.array([0.1, 0.2, 0.3])
        s = np.array([1, 1, 0])
        # threshold=0.5: nothing above → binarised = [0,0,0]
        # intersection = 0, union = {0,1} = 2
        assert compute_attribution_iou(a, s, threshold=0.5) == pytest.approx(0.0)


class TestAttributionIoUImage:
    """2-D inputs."""

    def test_2d_perfect(self):
        a = np.zeros((4, 4))
        a[0:2, 0:2] = 1.0
        s = np.zeros((4, 4))
        s[0:2, 0:2] = 1.0
        assert compute_attribution_iou(a, s, threshold=0.5) == pytest.approx(1.0)


class TestAttributionIoUSemantic:
    """Semantic validation."""

    def test_targeted_beats_spread(self):
        s = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        a_good = np.array([0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        a_bad = np.array([0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8])
        iou_good = compute_attribution_iou(a_good, s, threshold=0.5)
        iou_bad = compute_attribution_iou(a_bad, s, threshold=0.5)
        assert iou_good > iou_bad


class TestBatchAttributionIoU:
    """Batch processing."""

    def test_basic_batch(self):
        a1 = np.array([0.9, 0.0])
        a2 = np.array([0.0, 0.9])
        s = np.array([1, 0])
        results = compute_batch_attribution_iou([a1, a2], [s, s], threshold=0.5)
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)

    def test_batch_size_mismatch(self):
        with pytest.raises(ValueError, match="Batch sizes"):
            compute_batch_attribution_iou([np.ones(3)], [np.ones(3), np.ones(3)], threshold=0.5)

    def test_batch_percentile_kwarg(self):
        a = np.array([0.0, 0.5, 1.0])
        s = np.array([0, 0, 1])
        results = compute_batch_attribution_iou([a], [s], percentile=50)
        assert results[0] > 0.0
