# src/explainiverse/evaluation/localisation.py
"""
Localisation evaluation metrics for explanations (Phase 3).

Localisation metrics test whether explainable evidence is centred around a
region of interest (RoI) defined by a ground-truth segmentation mask,
bounding box, or feature set.  They require an external ground truth
indicating which features/pixels *should* be important.

Supports both **image** data (2-D/3-D attribution maps with 2-D masks)
and **tabular** data (1-D attribution vectors with 1-D binary masks).

Implemented metrics
-------------------
1. Pointing Game (Zhang et al., 2018)
2. Attribution Localisation (Kohlbrenner et al., 2020)
3. Top-K Intersection (Theiner et al., 2021)
4. Relevance Mass Accuracy (Arras et al., 2022)
5. Relevance Rank Accuracy (Arras et al., 2022)
6. AUC (Fawcett, 2006)
7. Focus (Arias-Duart et al., 2022)
8. Energy-Based Pointing Game (Wang et al., 2020)
9. Attribution IoU

References
----------
Zhang, J., Bargal, S. A., Lin, Z., Brandt, J., Shen, X., & Sclaroff, S.
    (2018).  Top-Down Neural Attention by Excitation Backprop.  International
    Journal of Computer Vision, 126(10), 1084-1102.

Kohlbrenner, M., Bauer, A., Nakajima, S., Binder, A., Samek, W., &
    Lapuschkin, S. (2020).  Towards Best Practice in Explaining Neural
    Network Decisions with LRP.  IEEE IJCNN, 1-7.

Theiner, J., Müller-Budack, E., & Ewerth, R. (2021).  Interpretable
    Semantic Photo Geolocalization.  arXiv:2104.14995.

Arras, L., Osman, A., & Samek, W. (2022).  CLEVR-XAI: A benchmark
    dataset for the ground truth evaluation of neural network explanations.
    Information Fusion, 81, 14-40.

Fawcett, T. (2006).  An Introduction to ROC Analysis.  Pattern Recognition
    Letters, 27(8), 861-874.

Arias-Duart, A., Parés, F., Garcia-Gasulla, D., & Giménez-Ábalos, V.
    (2022).  Focus! Rating XAI Methods and Finding Biases.  IEEE FUZZ-IEEE.
    arXiv:2109.15035.

Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., Mardziel, P.,
    & Hu, X. (2020).  Score-CAM: Score-Weighted Visual Explanations for
    Convolutional Neural Networks.  CVPR Workshops.  arXiv:1910.01279.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from explainiverse.core.explanation import Explanation


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LocalisationMask:
    """Ground-truth mask for localisation evaluation.

    Wraps a binary NumPy array with metadata describing the kind of ground
    truth (segmentation mask, bounding box, or tabular feature set).  The
    class is intentionally lightweight and extensible.

    Parameters
    ----------
    mask : np.ndarray
        Binary array where 1 indicates a ground-truth relevant element
        and 0 indicates irrelevant.  Accepted shapes:

        * *Tabular*: ``(n_features,)``  — 1-D binary vector.
        * *Image*:   ``(H, W)``         — 2-D spatial mask.
        * *Image*:   ``(C, H, W)``      — channel-first (mask is
          broadcast across channels; only spatial dimensions are used
          for evaluation, so all channels should be identical or the
          caller should pre-reduce).

    mask_type : str
        One of ``"segmentation"``, ``"bounding_box"``, or
        ``"feature_set"``.  Used for documentation / dispatching; does
        not currently change metric computation.
    metadata : dict, optional
        Arbitrary additional information (label name, source dataset,
        IoU threshold used to generate the mask, etc.).

    Raises
    ------
    ValueError
        If *mask* contains values other than 0 and 1, or is empty.
    TypeError
        If *mask* is not a NumPy array.

    Examples
    --------
    >>> import numpy as np
    >>> mask = LocalisationMask(
    ...     mask=np.array([1, 0, 1, 0, 0]),
    ...     mask_type="feature_set",
    ... )
    >>> mask.n_relevant
    2
    """

    mask: np.ndarray
    mask_type: str = "segmentation"
    metadata: Dict[str, Any] = field(default_factory=dict)

    _VALID_TYPES = {"segmentation", "bounding_box", "feature_set"}

    def __post_init__(self) -> None:
        # --- type checks ---
        if not isinstance(self.mask, np.ndarray):
            raise TypeError(
                f"mask must be a numpy ndarray, got {type(self.mask).__name__}."
            )
        if self.mask.size == 0:
            raise ValueError("mask must not be empty.")
        if self.mask_type not in self._VALID_TYPES:
            raise ValueError(
                f"mask_type must be one of {self._VALID_TYPES}, "
                f"got '{self.mask_type}'."
            )

        # Ensure binary (allow float arrays that are 0.0/1.0)
        unique = np.unique(self.mask)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(
                "mask must be binary (contain only 0 and 1). "
                f"Found unique values: {unique}."
            )

        # Store as float64 for consistent arithmetic
        self.mask = self.mask.astype(np.float64)

    # --- convenience properties ---

    @property
    def n_relevant(self) -> int:
        """Number of ground-truth relevant elements (mask == 1)."""
        return int(np.sum(self.mask))

    @property
    def n_total(self) -> int:
        """Total number of elements in the mask."""
        return int(self.mask.size)

    @property
    def shape(self) -> tuple:
        """Shape of the underlying mask array."""
        return self.mask.shape

    @property
    def is_tabular(self) -> bool:
        """True if the mask is 1-D (tabular feature set)."""
        return self.mask.ndim == 1

    @property
    def is_image(self) -> bool:
        """True if the mask is 2-D or 3-D (spatial)."""
        return self.mask.ndim >= 2

    @classmethod
    def from_bounding_box(
        cls,
        height: int,
        width: int,
        y_min: int,
        y_max: int,
        x_min: int,
        x_max: int,
        **metadata: Any,
    ) -> "LocalisationMask":
        """Create a mask from bounding box coordinates.

        Parameters
        ----------
        height, width : int
            Spatial dimensions of the full image.
        y_min, y_max, x_min, x_max : int
            Bounding box coordinates (inclusive on min, exclusive on max).

        Returns
        -------
        LocalisationMask
        """
        mask = np.zeros((height, width), dtype=np.float64)
        mask[y_min:y_max, x_min:x_max] = 1.0
        return cls(mask=mask, mask_type="bounding_box", metadata=metadata)

    @classmethod
    def from_feature_indices(
        cls,
        n_features: int,
        relevant_indices: Sequence[int],
        **metadata: Any,
    ) -> "LocalisationMask":
        """Create a 1-D mask from a list of relevant feature indices.

        Parameters
        ----------
        n_features : int
            Total number of features.
        relevant_indices : sequence of int
            Indices of relevant features.

        Returns
        -------
        LocalisationMask
        """
        mask = np.zeros(n_features, dtype=np.float64)
        for idx in relevant_indices:
            if not 0 <= idx < n_features:
                raise ValueError(
                    f"Index {idx} out of range for n_features={n_features}."
                )
            mask[idx] = 1.0
        return cls(mask=mask, mask_type="feature_set", metadata=metadata)


# =============================================================================
# Internal Helpers
# =============================================================================

def _extract_attributions(
    attributions: Union[np.ndarray, Explanation],
) -> np.ndarray:
    """Extract a flat float64 attribution array from various input types.

    Accepts a raw ``np.ndarray`` (returned as-is after dtype cast) or
    an ``Explanation`` object (extracts ``feature_attributions``).

    Returns
    -------
    np.ndarray
        Attribution values as float64.  Shape is preserved.
    """
    if isinstance(attributions, Explanation):
        attr_dict = attributions.explanation_data.get("feature_attributions", {})
        if not attr_dict:
            raise ValueError("No feature attributions found in Explanation.")
        feature_names = getattr(attributions, "feature_names", None)
        if feature_names:
            values = [attr_dict.get(fn, 0.0) for fn in feature_names]
        else:
            values = list(attr_dict.values())
        return np.array(values, dtype=np.float64)

    if isinstance(attributions, np.ndarray):
        return attributions.astype(np.float64)

    raise TypeError(
        f"Expected np.ndarray or Explanation, got {type(attributions).__name__}"
    )


def _extract_mask(
    mask: Union[np.ndarray, LocalisationMask],
) -> np.ndarray:
    """Extract a binary float64 mask array.

    Accepts a raw ``np.ndarray`` or a ``LocalisationMask``.
    """
    if isinstance(mask, LocalisationMask):
        return mask.mask  # already float64 and validated
    if isinstance(mask, np.ndarray):
        arr = mask.astype(np.float64)
        unique = np.unique(arr)
        if not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError(
                "mask array must be binary (0 and 1 only). "
                f"Found unique values: {unique}."
            )
        return arr
    raise TypeError(
        f"Expected np.ndarray or LocalisationMask, got {type(mask).__name__}"
    )


def _flatten_spatial(a: np.ndarray, s: np.ndarray) -> tuple:
    """Flatten attribution and mask arrays to 1-D for metric computation.

    For image data the attribution map may be (C, H, W) or (H, W) and
    the mask is typically (H, W).  This helper reduces both to aligned
    1-D vectors over spatial elements.

    If the attribution has more dimensions than the mask (e.g. channel
    dim), the absolute values are summed across the extra leading
    dimensions so the result is aligned with the mask.

    Returns
    -------
    a_flat : np.ndarray, shape (N,)
    s_flat : np.ndarray, shape (N,)
    """
    if a.ndim == s.ndim:
        return a.ravel(), s.ravel()

    # Attribution has extra leading dims (e.g. channels)
    if a.ndim > s.ndim:
        # Sum over leading (channel) dimensions
        extra = a.ndim - s.ndim
        for _ in range(extra):
            a = np.sum(a, axis=0)
        return a.ravel(), s.ravel()

    raise ValueError(
        f"Attribution array ndim ({a.ndim}) must be >= mask ndim ({s.ndim})."
    )


def _validate_attribution_mask(
    a: np.ndarray,
    s: np.ndarray,
    metric_name: str,
) -> tuple:
    """Common validation for attribution + mask pairs.

    Returns flattened (a_flat, s_flat) ready for metric computation.
    """
    if a.size == 0:
        raise ValueError(f"{metric_name}: attribution array must not be empty.")
    if s.size == 0:
        raise ValueError(f"{metric_name}: mask must not be empty.")

    a_flat, s_flat = _flatten_spatial(a, s)

    if a_flat.shape[0] != s_flat.shape[0]:
        raise ValueError(
            f"{metric_name}: flattened attribution length ({a_flat.shape[0]}) "
            f"does not match flattened mask length ({s_flat.shape[0]})."
        )

    return a_flat, s_flat


# =============================================================================
# Metric 1: Pointing Game  (Zhang et al., 2018)
# =============================================================================

def compute_pointing_game(
    attributions: Union[np.ndarray, Explanation],
    mask: Union[np.ndarray, LocalisationMask],
    *,
    use_abs: bool = True,
    tolerance: int = 0,
) -> float:
    """Pointing Game (Zhang et al., 2018).

    Checks whether the single highest-attributed element falls inside the
    ground-truth mask.

    .. math::

        \\text{PG}(a, s) = \\mathbb{1}[\\arg\\max_i |a_i| \\in s]

    When *tolerance* > 0 (image data only), a square neighbourhood of
    side ``2 * tolerance + 1`` centred on the argmax is checked instead
    of the single element.

    Parameters
    ----------
    attributions : np.ndarray or Explanation
        Attribution map.  Shape ``(n_features,)`` for tabular or
        ``(H, W)`` / ``(C, H, W)`` for image data.
    mask : np.ndarray or LocalisationMask
        Binary ground-truth mask aligned with *attributions*.
    use_abs : bool, default True
        If True, rank features by absolute attribution value.
    tolerance : int, default 0
        For image masks, expand the argmax to a square patch of this
        radius.  0 means exact single-element check.

    Returns
    -------
    float
        1.0 if the highest-attributed element is inside the mask,
        0.0 otherwise.

    Raises
    ------
    ValueError
        If shapes are incompatible or arrays are empty.

    Reference
    ---------
    Zhang, J., Bargal, S. A., Lin, Z., Brandt, J., Shen, X., &
    Sclaroff, S. (2018).  Top-Down Neural Attention by Excitation
    Backprop.  IJCV, 126(10), 1084-1102.
    """
    a = _extract_attributions(attributions)
    s = _extract_mask(mask)

    # --- tolerance > 0 requires 2-D spatial handling ---
    if tolerance > 0 and s.ndim >= 2:
        # Reduce attribution to spatial dims matching mask
        a_spatial = a.copy()
        if a_spatial.ndim > s.ndim:
            extra = a_spatial.ndim - s.ndim
            for _ in range(extra):
                a_spatial = np.sum(a_spatial, axis=0)

        if a_spatial.shape != s.shape:
            raise ValueError(
                f"Pointing Game: spatial attribution shape {a_spatial.shape} "
                f"does not match mask shape {s.shape}."
            )

        vals = np.abs(a_spatial) if use_abs else a_spatial
        max_idx = np.unravel_index(np.argmax(vals), vals.shape)
        # Check tolerance region
        h, w = s.shape[-2], s.shape[-1]
        y, x = max_idx[-2], max_idx[-1]
        y_lo = max(0, y - tolerance)
        y_hi = min(h, y + tolerance + 1)
        x_lo = max(0, x - tolerance)
        x_hi = min(w, x + tolerance + 1)
        patch = s[..., y_lo:y_hi, x_lo:x_hi]
        return 1.0 if np.any(patch > 0) else 0.0

    # --- standard (flat) path ---
    a_flat, s_flat = _validate_attribution_mask(a, s, "Pointing Game")
    vals = np.abs(a_flat) if use_abs else a_flat
    max_idx = int(np.argmax(vals))
    return 1.0 if s_flat[max_idx] > 0 else 0.0


def compute_batch_pointing_game(
    attributions_batch: List[Union[np.ndarray, Explanation]],
    masks_batch: List[Union[np.ndarray, LocalisationMask]],
    *,
    use_abs: bool = True,
    tolerance: int = 0,
) -> List[float]:
    """Batch Pointing Game.

    Parameters
    ----------
    attributions_batch : list
        List of attribution maps.
    masks_batch : list
        List of corresponding ground-truth masks.
    use_abs : bool, default True
        Rank features by absolute attribution value.
    tolerance : int, default 0
        Spatial tolerance radius (image data only).

    Returns
    -------
    list of float
        One score per instance.
    """
    if len(attributions_batch) != len(masks_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_batch)} "
            f"attributions and {len(masks_batch)} masks."
        )
    return [
        compute_pointing_game(a, s, use_abs=use_abs, tolerance=tolerance)
        for a, s in zip(attributions_batch, masks_batch)
    ]


# =============================================================================
# Metric 2: Attribution Localisation  (Kohlbrenner et al., 2020)
# =============================================================================

def compute_attribution_localisation(
    attributions: Union[np.ndarray, Explanation],
    mask: Union[np.ndarray, LocalisationMask],
    *,
    use_abs: bool = False,
) -> float:
    """Attribution Localisation (Kohlbrenner et al., 2020).

    Measures the ratio of positive attributions inside the ground-truth
    mask to the total positive attributions:

    .. math::

        \\text{AL}(a, s) = \\frac{\\sum_i a_i^+ \\cdot s_i}
                                  {\\sum_i a_i^+}

    where :math:`a_i^+ = \\max(a_i, 0)` and :math:`s_i \\in \\{0, 1\\}`.

    If *use_abs* is True, absolute attribution values are used instead
    of clipping to positive (this is an optional variant found in some
    implementations).

    Parameters
    ----------
    attributions : np.ndarray or Explanation
        Attribution map.
    mask : np.ndarray or LocalisationMask
        Binary ground-truth mask.
    use_abs : bool, default False
        If True, use ``|a_i|`` instead of ``max(a_i, 0)``.

    Returns
    -------
    float
        Score in [0, 1].  Higher = better localisation.
        Returns 0.0 if there are no positive attributions.

    Reference
    ---------
    Kohlbrenner, M., Bauer, A., Nakajima, S., Binder, A., Samek, W., &
    Lapuschkin, S. (2020).  Towards Best Practice in Explaining Neural
    Network Decisions with LRP.  IEEE IJCNN, 1-7.
    """
    a = _extract_attributions(attributions)
    s = _extract_mask(mask)
    a_flat, s_flat = _validate_attribution_mask(
        a, s, "Attribution Localisation"
    )

    if use_abs:
        a_pos = np.abs(a_flat)
    else:
        a_pos = np.maximum(a_flat, 0.0)

    total_positive = np.sum(a_pos)
    if total_positive == 0.0:
        return 0.0

    return float(np.sum(a_pos * s_flat) / total_positive)


def compute_batch_attribution_localisation(
    attributions_batch: List[Union[np.ndarray, Explanation]],
    masks_batch: List[Union[np.ndarray, LocalisationMask]],
    *,
    use_abs: bool = False,
) -> List[float]:
    """Batch Attribution Localisation (Kohlbrenner et al., 2020)."""
    if len(attributions_batch) != len(masks_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_batch)} "
            f"and {len(masks_batch)}."
        )
    return [
        compute_attribution_localisation(a, s, use_abs=use_abs)
        for a, s in zip(attributions_batch, masks_batch)
    ]


# =============================================================================
# Metric 3: Top-K Intersection  (Theiner et al., 2021)
# =============================================================================

def compute_top_k_intersection(
    attributions: Union[np.ndarray, Explanation],
    mask: Union[np.ndarray, LocalisationMask],
    *,
    k: Optional[int] = None,
    use_abs: bool = True,
) -> float:
    """Top-K Intersection (Theiner et al., 2021).

    Computes the intersection between the top-k attributed elements and
    the ground-truth mask, normalised by k:

    .. math::

        \\text{TKI}(a, s, k) = \\frac{|\\text{top}_k(a) \\cap s|}{k}

    If *k* is None it defaults to the number of relevant elements in
    the mask (``|s|``).

    Parameters
    ----------
    attributions : np.ndarray or Explanation
        Attribution map.
    mask : np.ndarray or LocalisationMask
        Binary ground-truth mask.
    k : int or None, default None
        Number of top elements.  If None, uses ``sum(mask)``.
    use_abs : bool, default True
        Rank by absolute attribution value.

    Returns
    -------
    float
        Score in [0, 1].  Higher = better localisation.

    Raises
    ------
    ValueError
        If k < 1 or k > number of elements.

    Reference
    ---------
    Theiner, J., Müller-Budack, E., & Ewerth, R. (2021).  Interpretable
    Semantic Photo Geolocalization.  arXiv:2104.14995.
    """
    a = _extract_attributions(attributions)
    s = _extract_mask(mask)
    a_flat, s_flat = _validate_attribution_mask(
        a, s, "Top-K Intersection"
    )

    n = a_flat.shape[0]
    n_relevant = int(np.sum(s_flat))

    if k is None:
        k = n_relevant
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}.")
    if k > n:
        raise ValueError(f"k={k} exceeds number of elements n={n}.")

    if n_relevant == 0:
        warnings.warn(
            "Top-K Intersection: mask has no relevant elements (all zeros). "
            "Returning 0.0.",
            stacklevel=2,
        )
        return 0.0

    vals = np.abs(a_flat) if use_abs else a_flat
    top_k_indices = np.argsort(vals)[-k:]  # indices of top-k
    # Count how many of the top-k fall inside the mask
    hits = np.sum(s_flat[top_k_indices])
    return float(hits / k)


def compute_batch_top_k_intersection(
    attributions_batch: List[Union[np.ndarray, Explanation]],
    masks_batch: List[Union[np.ndarray, LocalisationMask]],
    *,
    k: Optional[int] = None,
    use_abs: bool = True,
) -> List[float]:
    """Batch Top-K Intersection (Theiner et al., 2021)."""
    if len(attributions_batch) != len(masks_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_batch)} "
            f"and {len(masks_batch)}."
        )
    return [
        compute_top_k_intersection(a, s, k=k, use_abs=use_abs)
        for a, s in zip(attributions_batch, masks_batch)
    ]


# =============================================================================
# Metric 4: Relevance Mass Accuracy  (Arras et al., 2022)
# =============================================================================

def compute_relevance_mass_accuracy(
    attributions: Union[np.ndarray, Explanation],
    mask: Union[np.ndarray, LocalisationMask],
    *,
    use_abs: bool = False,
    normalise: bool = True,
) -> float:
    """Relevance Mass Accuracy (Arras et al., 2022).

    Computes the ratio of positive attributions inside the ground-truth
    mask to the sum of all positive attributions:

    .. math::

        \\text{RMA}(a, s) = \\frac{\\sum_i a_i^+ \\cdot s_i}
                                   {\\sum_i a_i^+}

    This is mathematically equivalent to Attribution Localisation in its
    default configuration, but originates from a different paper and is
    commonly parameterised with a *normalise* option that normalises the
    attribution map to [0, 1] before computation.

    Parameters
    ----------
    attributions : np.ndarray or Explanation
        Attribution map.
    mask : np.ndarray or LocalisationMask
        Binary ground-truth mask.
    use_abs : bool, default False
        If True, use absolute attribution values.
    normalise : bool, default True
        If True, normalise attribution values to [0, 1] before
        computing the ratio.

    Returns
    -------
    float
        Score in [0, 1].  Higher = more positive attribution mass
        falls inside the ground-truth region.

    Reference
    ---------
    Arras, L., Osman, A., & Samek, W. (2022).  CLEVR-XAI: A benchmark
    dataset for the ground truth evaluation of neural network
    explanations.  Information Fusion, 81, 14-40.
    """
    a = _extract_attributions(attributions)
    s = _extract_mask(mask)
    a_flat, s_flat = _validate_attribution_mask(
        a, s, "Relevance Mass Accuracy"
    )

    if use_abs:
        a_flat = np.abs(a_flat)

    if normalise:
        a_min = np.min(a_flat)
        a_max = np.max(a_flat)
        denom = a_max - a_min
        if denom > 0:
            a_flat = (a_flat - a_min) / denom
        else:
            # All values identical — normalise to zeros
            a_flat = np.zeros_like(a_flat)

    a_pos = np.maximum(a_flat, 0.0)
    total_positive = np.sum(a_pos)
    if total_positive == 0.0:
        return 0.0

    return float(np.sum(a_pos * s_flat) / total_positive)


def compute_batch_relevance_mass_accuracy(
    attributions_batch: List[Union[np.ndarray, Explanation]],
    masks_batch: List[Union[np.ndarray, LocalisationMask]],
    *,
    use_abs: bool = False,
    normalise: bool = True,
) -> List[float]:
    """Batch Relevance Mass Accuracy (Arras et al., 2022)."""
    if len(attributions_batch) != len(masks_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_batch)} "
            f"and {len(masks_batch)}."
        )
    return [
        compute_relevance_mass_accuracy(
            a, s, use_abs=use_abs, normalise=normalise,
        )
        for a, s in zip(attributions_batch, masks_batch)
    ]


# =============================================================================
# Metric 5: Relevance Rank Accuracy  (Arras et al., 2022)
# =============================================================================

def compute_relevance_rank_accuracy(
    attributions: Union[np.ndarray, Explanation],
    mask: Union[np.ndarray, LocalisationMask],
    *,
    use_abs: bool = True,
) -> float:
    """Relevance Rank Accuracy (Arras et al., 2022).

    Measures the fraction of the top-k highest-attributed elements that
    fall inside the ground-truth mask, where k equals the number of
    relevant elements in the mask:

    .. math::

        \\text{RRA}(a, s) = \\frac{|\\text{top}_{|s|}(a) \\cap s|}{|s|}

    This is equivalent to Top-K Intersection with ``k = |s|`` but
    normalised by the mask size rather than k (which are the same
    when ``k = |s|``).

    Parameters
    ----------
    attributions : np.ndarray or Explanation
        Attribution map.
    mask : np.ndarray or LocalisationMask
        Binary ground-truth mask.
    use_abs : bool, default True
        Rank by absolute attribution value.

    Returns
    -------
    float
        Score in [0, 1].  Higher = more top-ranked elements are
        inside the ground-truth region.

    Reference
    ---------
    Arras, L., Osman, A., & Samek, W. (2022).  CLEVR-XAI: A benchmark
    dataset for the ground truth evaluation of neural network
    explanations.  Information Fusion, 81, 14-40.
    """
    a = _extract_attributions(attributions)
    s = _extract_mask(mask)
    a_flat, s_flat = _validate_attribution_mask(
        a, s, "Relevance Rank Accuracy"
    )

    n_relevant = int(np.sum(s_flat))
    if n_relevant == 0:
        warnings.warn(
            "Relevance Rank Accuracy: mask has no relevant elements. "
            "Returning 0.0.",
            stacklevel=2,
        )
        return 0.0

    vals = np.abs(a_flat) if use_abs else a_flat
    top_k_indices = np.argsort(vals)[-n_relevant:]
    hits = np.sum(s_flat[top_k_indices])
    return float(hits / n_relevant)


def compute_batch_relevance_rank_accuracy(
    attributions_batch: List[Union[np.ndarray, Explanation]],
    masks_batch: List[Union[np.ndarray, LocalisationMask]],
    *,
    use_abs: bool = True,
) -> List[float]:
    """Batch Relevance Rank Accuracy (Arras et al., 2022)."""
    if len(attributions_batch) != len(masks_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_batch)} "
            f"and {len(masks_batch)}."
        )
    return [
        compute_relevance_rank_accuracy(a, s, use_abs=use_abs)
        for a, s in zip(attributions_batch, masks_batch)
    ]


# =============================================================================
# Metric 6: AUC  (Fawcett, 2006)
# =============================================================================

def compute_auc(
    attributions: Union[np.ndarray, Explanation],
    mask: Union[np.ndarray, LocalisationMask],
    *,
    use_abs: bool = True,
) -> float:
    """Area Under the ROC Curve for localisation (Fawcett, 2006).

    Treats the (optionally absolute) attribution values as continuous
    scores and the binary ground-truth mask as labels, then computes
    the ROC-AUC.  A score of 1.0 means the attribution perfectly
    separates relevant from irrelevant elements; 0.5 is random.

    The implementation uses the efficient rank-based (Mann–Whitney U)
    formulation so that no external dependency (e.g. scikit-learn) is
    required.

    Parameters
    ----------
    attributions : np.ndarray or Explanation
        Attribution map.
    mask : np.ndarray or LocalisationMask
        Binary ground-truth mask.
    use_abs : bool, default True
        If True, use absolute attribution values as scores.

    Returns
    -------
    float
        ROC-AUC in [0, 1].
        Returns 0.5 if the mask is all-zero or all-one (degenerate).

    Reference
    ---------
    Fawcett, T. (2006).  An Introduction to ROC Analysis.  Pattern
    Recognition Letters, 27(8), 861–874.
    """
    a = _extract_attributions(attributions)
    s = _extract_mask(mask)
    a_flat, s_flat = _validate_attribution_mask(a, s, "AUC")

    n_pos = int(np.sum(s_flat))
    n_neg = len(s_flat) - n_pos

    if n_pos == 0 or n_neg == 0:
        warnings.warn(
            "AUC: mask is degenerate (all-zero or all-one). "
            "Returning 0.5.",
            stacklevel=2,
        )
        return 0.5

    scores = np.abs(a_flat) if use_abs else a_flat

    # Mann–Whitney U statistic via ranking
    # rank from 1..N (average ranks for ties)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    # Average ranks for ties
    unique_vals = np.unique(scores)
    if len(unique_vals) < len(scores):
        for v in unique_vals:
            tie_mask = scores == v
            if np.sum(tie_mask) > 1:
                ranks[tie_mask] = np.mean(ranks[tie_mask])

    rank_sum_pos = np.sum(ranks[s_flat == 1.0])
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_batch_auc(
    attributions_batch: List[Union[np.ndarray, Explanation]],
    masks_batch: List[Union[np.ndarray, LocalisationMask]],
    *,
    use_abs: bool = True,
) -> List[float]:
    """Batch AUC (Fawcett, 2006)."""
    if len(attributions_batch) != len(masks_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_batch)} "
            f"and {len(masks_batch)}."
        )
    return [
        compute_auc(a, s, use_abs=use_abs)
        for a, s in zip(attributions_batch, masks_batch)
    ]


# =============================================================================
# Metric 7: Energy-Based Pointing Game  (Wang et al., 2020)
# =============================================================================

def compute_energy_based_pointing_game(
    attributions: Union[np.ndarray, Explanation],
    mask: Union[np.ndarray, LocalisationMask],
    *,
    use_abs: bool = False,
) -> float:
    """Energy-Based Pointing Game (Wang et al., 2020 — Score-CAM).

    Measures the proportion of total attribution energy that falls
    inside the ground-truth mask:

    .. math::

        \\text{EBPG}(a, s) = \\frac{\\sum_i a_i \\cdot s_i}
                                     {\\sum_i a_i}

    Unlike Attribution Localisation, this metric uses *all* attribution
    values (not just positive).  The original paper (Score-CAM) assumes
    non-negative attributions; if attributions contain negative values
    the result may fall outside [0, 1].

    Parameters
    ----------
    attributions : np.ndarray or Explanation
        Attribution map (typically non-negative, e.g. from CAM methods).
    mask : np.ndarray or LocalisationMask
        Binary ground-truth mask.
    use_abs : bool, default False
        If True, use absolute attribution values.

    Returns
    -------
    float
        Proportion of attribution energy inside the mask.
        Returns 0.0 if total attribution sum is zero.

    Reference
    ---------
    Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S.,
    Mardziel, P., & Hu, X. (2020).  Score-CAM: Score-Weighted Visual
    Explanations for Convolutional Neural Networks.  CVPR Workshops.
    arXiv:1910.01279.
    """
    a = _extract_attributions(attributions)
    s = _extract_mask(mask)
    a_flat, s_flat = _validate_attribution_mask(
        a, s, "Energy-Based Pointing Game"
    )

    if use_abs:
        a_flat = np.abs(a_flat)

    total = np.sum(a_flat)
    if total == 0.0:
        return 0.0

    return float(np.sum(a_flat * s_flat) / total)


def compute_batch_energy_based_pointing_game(
    attributions_batch: List[Union[np.ndarray, Explanation]],
    masks_batch: List[Union[np.ndarray, LocalisationMask]],
    *,
    use_abs: bool = False,
) -> List[float]:
    """Batch Energy-Based Pointing Game (Wang et al., 2020)."""
    if len(attributions_batch) != len(masks_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_batch)} "
            f"and {len(masks_batch)}."
        )
    return [
        compute_energy_based_pointing_game(a, s, use_abs=use_abs)
        for a, s in zip(attributions_batch, masks_batch)
    ]


# =============================================================================
# Metric 8: Focus  (Arias-Duart et al., 2022)
# =============================================================================

def compute_focus(
    attributions: Union[np.ndarray, Explanation],
    mask: Union[np.ndarray, LocalisationMask],
) -> float:
    """Focus (Arias-Duart et al., 2022).

    Designed for mosaic-based evaluation: given a tiled image containing
    multiple sub-images, *Focus* measures the proportion of positive
    attribution mass that falls inside the tile corresponding to the
    predicted class.

    .. math::

        \\text{Focus}(a, s) = \\frac{\\sum_i a_i^+ \\cdot s_i}
                                     {\\sum_i a_i^+}

    The formula is identical to Attribution Localisation; the
    conceptual difference is that the mask *s* denotes the spatial
    region of a specific mosaic tile rather than a per-object
    segmentation.

    Parameters
    ----------
    attributions : np.ndarray or Explanation
        Attribution map over the full mosaic / image.
    mask : np.ndarray or LocalisationMask
        Binary mask indicating the tile region of interest.

    Returns
    -------
    float
        Score in [0, 1].  Higher = more positive attribution focused
        in the correct tile.
        Returns 0.0 if no positive attributions exist.

    Reference
    ---------
    Arias-Duart, A., Parés, F., Garcia-Gasulla, D., & Giménez-Ábalos,
    V. (2022).  Focus! Rating XAI Methods and Finding Biases.  IEEE
    FUZZ-IEEE.  arXiv:2109.15035.
    """
    a = _extract_attributions(attributions)
    s = _extract_mask(mask)
    a_flat, s_flat = _validate_attribution_mask(a, s, "Focus")

    a_pos = np.maximum(a_flat, 0.0)
    total_positive = np.sum(a_pos)
    if total_positive == 0.0:
        return 0.0

    return float(np.sum(a_pos * s_flat) / total_positive)


def compute_batch_focus(
    attributions_batch: List[Union[np.ndarray, Explanation]],
    masks_batch: List[Union[np.ndarray, LocalisationMask]],
) -> List[float]:
    """Batch Focus (Arias-Duart et al., 2022)."""
    if len(attributions_batch) != len(masks_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_batch)} "
            f"and {len(masks_batch)}."
        )
    return [
        compute_focus(a, s)
        for a, s in zip(attributions_batch, masks_batch)
    ]


# =============================================================================
# Metric 9: Attribution IoU
# =============================================================================

def compute_attribution_iou(
    attributions: Union[np.ndarray, Explanation],
    mask: Union[np.ndarray, LocalisationMask],
    *,
    threshold: Optional[float] = None,
    use_abs: bool = True,
    percentile: Optional[float] = None,
) -> float:
    """Attribution Intersection-over-Union.

    Binarises the attribution map with a threshold and computes the
    standard IoU against the ground-truth mask:

    .. math::

        \\text{IoU}(a, s, \\tau) =
            \\frac{|\\{a_i > \\tau\\} \\cap s|}
                  {|\\{a_i > \\tau\\} \\cup s|}

    Exactly one of *threshold* or *percentile* must be provided.

    Parameters
    ----------
    attributions : np.ndarray or Explanation
        Attribution map.
    mask : np.ndarray or LocalisationMask
        Binary ground-truth mask.
    threshold : float or None
        Absolute threshold to binarise attributions.
    use_abs : bool, default True
        If True, apply threshold to ``|a_i|``.
    percentile : float or None
        Percentile (0–100) of attribution values to use as threshold.
        E.g. ``percentile=75`` uses the 75th percentile value.

    Returns
    -------
    float
        IoU in [0, 1].  Returns 0.0 if both the binarised attribution
        and the mask are all zeros (empty union).

    Raises
    ------
    ValueError
        If neither or both of *threshold* and *percentile* are given.
    """
    # --- parameter validation ---
    if (threshold is None) == (percentile is None):
        raise ValueError(
            "Exactly one of 'threshold' or 'percentile' must be provided."
        )
    if percentile is not None:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(
                f"percentile must be in [0, 100], got {percentile}."
            )

    a = _extract_attributions(attributions)
    s = _extract_mask(mask)
    a_flat, s_flat = _validate_attribution_mask(
        a, s, "Attribution IoU"
    )

    vals = np.abs(a_flat) if use_abs else a_flat

    if percentile is not None:
        threshold = float(np.percentile(vals, percentile))

    a_bin = (vals > threshold).astype(np.float64)

    intersection = np.sum(a_bin * s_flat)
    union = np.sum(np.clip(a_bin + s_flat, 0.0, 1.0))

    if union == 0.0:
        return 0.0

    return float(intersection / union)


def compute_batch_attribution_iou(
    attributions_batch: List[Union[np.ndarray, Explanation]],
    masks_batch: List[Union[np.ndarray, LocalisationMask]],
    *,
    threshold: Optional[float] = None,
    use_abs: bool = True,
    percentile: Optional[float] = None,
) -> List[float]:
    """Batch Attribution IoU."""
    if len(attributions_batch) != len(masks_batch):
        raise ValueError(
            f"Batch sizes must match: got {len(attributions_batch)} "
            f"and {len(masks_batch)}."
        )
    return [
        compute_attribution_iou(
            a, s, threshold=threshold, use_abs=use_abs,
            percentile=percentile,
        )
        for a, s in zip(attributions_batch, masks_batch)
    ]
