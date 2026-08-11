"""Ground-truth localisation metrics for attribution maps.

The functions in this module operate on one scalar attribution value per
evaluated feature or spatial location.  Image attributions therefore must be
pooled to a single ``(H, W)`` map *before* they are passed here.  In
particular, this module does not silently choose a channel-pooling rule: the
CLEVR-XAI authors explicitly evaluate several incompatible pooling rules and
report that there is no consensus rule.

Undefined cases are rejected instead of being assigned plausible-looking
scores.  Examples include an empty ground-truth region, zero positive
relevance for a mass ratio, a one-class ROC problem, or a tie that straddles a
top-k boundary.

Implemented paper metrics
-------------------------
* Pointing Game (Zhang et al., 2016/2018).
* Attribution Localisation (Kohlbrenner et al., 2020), unweighted ``mu``.
* Top-K Intersection (Theiner et al., 2021), without concept-size weighting.
* Relevance Mass Accuracy and Relevance Rank Accuracy (Arras et al., 2022).
* ROC AUC (Fawcett, 2006), applied pixel/feature-wise.
* Energy-Based Pointing Game (Wang et al., 2020).
* Focus (Arias-Duart et al., 2022), for canonical 2-by-2 mosaics.

``compute_attribution_iou`` is a clearly labelled, library-defined thresholded
overlap diagnostic.  It is not attributed to a particular XAI paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import _percentile_mask

AttributionInput = Union[np.ndarray, Explanation]
MaskInput = Union[np.ndarray, "LocalisationMask"]


_INDEXED_FEATURE_KEY = re.compile(r"^(feature_|feat_|f|x)(0|[1-9][0-9]*)$", re.IGNORECASE)


def _ordered_explanation_attribution_values(
    raw: Mapping,
    feature_names: Optional[List[str]],
) -> List[object]:
    """Return values only when the mapping's feature identity is complete."""
    if feature_names is not None:
        if isinstance(feature_names, (str, bytes)):
            raise ValueError("Explanation feature_names must be a sequence of strings.")
        try:
            names = list(feature_names)
        except TypeError as exc:
            raise ValueError("Explanation feature_names must be a sequence of strings.") from exc
        if not names:
            raise ValueError("Explanation feature_names must not be empty.")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Explanation feature_names must contain non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError("Explanation feature_names must be unique.")

        name_set = set(names)
        missing = [name for name in names if name not in raw]
        unexpected = [key for key in raw if key not in name_set]
        if missing or unexpected:
            problems = []
            if missing:
                problems.append(f"missing attributions for feature_names: {missing}")
            if unexpected:
                problems.append(
                    f"unexpected attributions not present in feature_names: {unexpected}"
                )
            raise ValueError("Explanation feature identity mismatch: " + "; ".join(problems) + ".")
        return [raw[name] for name in names]

    indexed = {}
    key_scheme = None
    for key, value in raw.items():
        match = _INDEXED_FEATURE_KEY.fullmatch(key) if isinstance(key, str) else None
        if match is None:
            raise ValueError(
                "Explanation without feature_names must use one consistent, "
                "zero-based indexed key scheme such as 'feature_0' or 'f0'."
            )
        scheme = match.group(1).lower()
        if key_scheme is None:
            key_scheme = scheme
        elif scheme != key_scheme:
            raise ValueError(
                "Explanation without feature_names must use one consistent, "
                "zero-based indexed key scheme such as 'feature_0' or 'f0'."
            )
        index = int(match.group(2))
        if index in indexed:
            raise ValueError("Explanation indexed feature keys must identify unique indices.")
        indexed[index] = value

    expected_indices = set(range(len(raw)))
    if set(indexed) != expected_indices:
        raise ValueError(
            "Explanation indexed feature keys must cover every zero-based index "
            f"from 0 through {len(raw) - 1}."
        )
    return [indexed[index] for index in range(len(raw))]


@dataclass(frozen=True)
class LocalisationMask:
    """A validated one-dimensional feature mask or two-dimensional image mask.

    ``mask`` must be a non-empty binary NumPy array with shape
    ``(n_features,)`` or ``(height, width)``.  Multi-channel masks are rejected:
    the metrics evaluate spatial locations, not colour channels.
    """

    mask: np.ndarray
    mask_type: str = "segmentation"
    metadata: Dict[str, Any] = field(default_factory=dict)

    _VALID_TYPES = {"segmentation", "bounding_box", "feature_set"}

    def __post_init__(self) -> None:
        if not isinstance(self.mask, np.ndarray):
            raise TypeError(f"mask must be a numpy ndarray, got {type(self.mask).__name__}.")
        if self.mask.ndim not in (1, 2):
            raise ValueError(
                "mask must be one-dimensional (features) or two-dimensional "
                f"(spatial), got shape {self.mask.shape}."
            )
        if self.mask.size == 0:
            raise ValueError("mask must not be empty.")
        if self.mask_type not in self._VALID_TYPES:
            raise ValueError(
                f"mask_type must be one of {self._VALID_TYPES}, " f"got {self.mask_type!r}."
            )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        if not (
            np.issubdtype(self.mask.dtype, np.integer)
            or np.issubdtype(self.mask.dtype, np.floating)
            or np.issubdtype(self.mask.dtype, np.bool_)
        ):
            raise TypeError("mask must have a numeric or boolean dtype.")

        # Own an immutable copy: retaining a caller-owned writable array lets
        # post-construction mutation bypass every invariant established here.
        mask: np.ndarray = self.mask.astype(np.float64, copy=True)
        if not np.all(np.isfinite(mask)):
            raise ValueError("mask must contain only finite values.")
        if not np.all((mask == 0.0) | (mask == 1.0)):
            raise ValueError("mask must be binary (contain only 0 and 1).")
        # An ndarray that owns its storage can have WRITEABLE re-enabled by a
        # caller. Re-backing it with immutable bytes makes the read-only
        # contract irreversible through NumPy's public flag API.
        mask = np.frombuffer(mask.tobytes(), dtype=np.float64).reshape(mask.shape)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_relevant(self) -> int:
        """Number of elements belonging to the ground-truth region."""
        return int(np.sum(self.mask))

    @property
    def n_total(self) -> int:
        """Total number of evaluated elements."""
        return int(self.mask.size)

    @property
    def shape(self) -> tuple:
        """Shape of the mask."""
        return self.mask.shape

    @property
    def is_tabular(self) -> bool:
        """Whether this is a one-dimensional feature mask."""
        return self.mask.ndim == 1

    @property
    def is_image(self) -> bool:
        """Whether this is a two-dimensional spatial mask."""
        return self.mask.ndim == 2

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
        """Create a mask from half-open bounds ``[min, max)``."""
        dimensions = {"height": height, "width": width}
        for name, value in dimensions.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        coordinates = {
            "y_min": y_min,
            "y_max": y_max,
            "x_min": x_min,
            "x_max": x_max,
        }
        for name, value in coordinates.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if not (0 <= y_min < y_max <= height):
            raise ValueError(
                "bounding-box y coordinates must satisfy "
                f"0 <= y_min < y_max <= height; got {(y_min, y_max)}."
            )
        if not (0 <= x_min < x_max <= width):
            raise ValueError(
                "bounding-box x coordinates must satisfy "
                f"0 <= x_min < x_max <= width; got {(x_min, x_max)}."
            )

        mask: np.ndarray = np.zeros((int(height), int(width)), dtype=np.float64)
        mask[int(y_min) : int(y_max), int(x_min) : int(x_max)] = 1.0
        return cls(mask=mask, mask_type="bounding_box", metadata=metadata)

    @classmethod
    def from_feature_indices(
        cls,
        n_features: int,
        relevant_indices: Sequence[int],
        **metadata: Any,
    ) -> "LocalisationMask":
        """Create a feature mask from zero-based feature indices."""
        if isinstance(n_features, (bool, np.bool_)) or not isinstance(n_features, Integral):
            raise TypeError("n_features must be an integer.")
        if n_features <= 0:
            raise ValueError("n_features must be positive.")

        mask = np.zeros(int(n_features), dtype=np.float64)
        for index in relevant_indices:
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
                raise TypeError("relevant_indices must contain only integers.")
            if not 0 <= index < n_features:
                raise ValueError(f"Index {index} out of range for n_features={n_features}.")
            mask[int(index)] = 1.0
        return cls(mask=mask, mask_type="feature_set", metadata=metadata)


def _extract_attributions(attributions: AttributionInput) -> np.ndarray:
    """Extract and validate finite float64 attribution values."""
    if isinstance(attributions, Explanation):
        raw = attributions.explanation_data.get("feature_attributions")
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(
                "Explanation must contain a non-empty 'feature_attributions' " "mapping."
            )
        feature_names = attributions.feature_names
        values = _ordered_explanation_attribution_values(raw, feature_names)
        candidate = np.asarray(values)
        if not (
            np.issubdtype(candidate.dtype, np.integer)
            or np.issubdtype(candidate.dtype, np.floating)
        ):
            raise ValueError("Explanation feature attributions must be real numeric scalars.")
        try:
            array = candidate.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Explanation feature attributions must be numeric scalars.") from exc
    elif isinstance(attributions, np.ndarray):
        if np.issubdtype(attributions.dtype, np.complexfloating):
            raise TypeError("attributions must be real-valued, not complex.")
        if not (
            np.issubdtype(attributions.dtype, np.integer)
            or np.issubdtype(attributions.dtype, np.floating)
        ):
            raise TypeError("attributions must have a real numeric dtype.")
        try:
            array = attributions.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("attributions must have a numeric dtype.") from exc
    else:
        raise TypeError(
            "Expected np.ndarray or Explanation, got " f"{type(attributions).__name__}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("attributions must contain only finite values.")
    return array


def _extract_mask(mask: MaskInput) -> np.ndarray:
    """Extract and validate a finite binary float64 mask."""
    if isinstance(mask, LocalisationMask):
        # Revalidate even immutable containers. This also protects metrics
        # from unusually deserialised instances that bypassed __post_init__.
        return _extract_mask(mask.mask)
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected np.ndarray or LocalisationMask, got {type(mask).__name__}.")
    if np.issubdtype(mask.dtype, np.complexfloating):
        raise TypeError("mask must be real-valued, not complex.")
    if not (
        np.issubdtype(mask.dtype, np.integer)
        or np.issubdtype(mask.dtype, np.floating)
        or np.issubdtype(mask.dtype, np.bool_)
    ):
        raise TypeError("mask must have a numeric or boolean dtype.")
    try:
        array: np.ndarray = mask.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("mask must have a numeric or boolean dtype.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError("mask must contain only finite values.")
    if not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError("mask array must be binary (0 and 1 only).")
    return array


def _validate_attribution_mask(
    attributions: np.ndarray,
    mask: np.ndarray,
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the shared scalar-map contract and return flat views."""
    if attributions.ndim not in (1, 2):
        raise ValueError(
            f"{metric_name}: attributions must be a 1-D feature vector or "
            "a pre-pooled 2-D spatial map; multi-channel attribution arrays "
            f"are ambiguous (got shape {attributions.shape})."
        )
    if mask.ndim not in (1, 2):
        raise ValueError(f"{metric_name}: mask must be 1-D or 2-D, got shape {mask.shape}.")
    if attributions.size == 0:
        raise ValueError(f"{metric_name}: attribution array must not be empty.")
    if mask.size == 0:
        raise ValueError(f"{metric_name}: mask must not be empty.")
    if attributions.shape != mask.shape:
        raise ValueError(
            f"{metric_name}: attribution shape {attributions.shape} does not "
            f"match mask shape {mask.shape}."
        )
    if not np.any(mask):
        raise ValueError(f"{metric_name}: ground-truth mask has no relevant elements.")
    return attributions.reshape(-1), mask.reshape(-1)


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def _rank_values(values: np.ndarray, use_abs: bool) -> np.ndarray:
    return np.abs(values) if _validate_bool(use_abs, "use_abs") else values


def _stable_nonnegative_mass_ratio(values: np.ndarray, mask: np.ndarray) -> float:
    """Return mass inside ``mask`` with one final binary64 rounding."""
    with localcontext() as context:
        context.prec = 2500 + len(str(values.size))
        decimal_values = [Decimal.from_float(float(value)) for value in values.reshape(-1)]
        decimal_mask = [Decimal.from_float(float(value)) for value in mask.reshape(-1)]
        denominator = sum(decimal_values, start=Decimal(0))
        numerator = sum(
            (value * mask_value for value, mask_value in zip(decimal_values, decimal_mask)),
            start=Decimal(0),
        )
        if denominator == 0:
            raise ValueError("total relevance is zero; the ratio is undefined.")
        exact = numerator / denominator
        result = float(exact)
    if not np.isfinite(result) or (result == 0.0 and exact != 0):
        raise FloatingPointError("localisation mass ratio is not representable")
    return result


def _top_k_hit_fraction(
    values: np.ndarray,
    mask: np.ndarray,
    k: int,
    metric_name: str,
) -> float:
    """Return the exact top-k hit fraction, rejecting ambiguous cut-off ties."""
    order = np.argsort(-values, kind="stable")
    if k < values.size and values[order[k - 1]] == values[order[k]]:
        raise ValueError(
            f"{metric_name}: a tie straddles the top-{k} boundary, so the "
            "selected set is not uniquely defined."
        )
    return float(np.sum(mask[order[:k]]) / k)


def _batch_apply(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
    metric_name: str,
    function: Callable[[AttributionInput, MaskInput], float],
) -> List[float]:
    """Apply a scalar metric to aligned batches with index-aware failures."""
    try:
        n_attributions = len(attributions_batch)
        n_masks = len(masks_batch)
    except TypeError as exc:
        raise TypeError("Batch inputs must be sized sequences.") from exc
    if n_attributions != n_masks:
        raise ValueError(
            f"Batch sizes must match: got {n_attributions} attributions and " f"{n_masks} masks."
        )

    scores: List[float] = []
    for index, (attributions, mask) in enumerate(zip(attributions_batch, masks_batch)):
        try:
            scores.append(function(attributions, mask))
        except (TypeError, ValueError) as exc:
            message = f"{metric_name} batch item {index}: {exc}"
            raise type(exc)(message) from exc
    return scores


def compute_pointing_game(
    attributions: AttributionInput,
    mask: MaskInput,
    *,
    use_abs: bool = False,
    tolerance: int = 0,
) -> float:
    """Compute the Pointing Game hit for one attribution map.

    The maximum attribution point is a hit when it lies in the ground-truth
    region.  Following the official Quantus implementation, if several points
    share the maximum, a hit is recorded when *any* maximum is in the region.
    A spatial ``tolerance`` checks a square margin around every maximum; the
    original paper used a 15-pixel ground-truth margin in its experiments.

    ``use_abs=True`` is explicit preprocessing, not part of the metric itself.
    """
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Integral):
        raise TypeError("tolerance must be a non-negative integer.")
    if tolerance < 0:
        raise ValueError("tolerance must be a non-negative integer.")

    array = _extract_attributions(attributions)
    segmentation = _extract_mask(mask)
    flat, mask_flat = _validate_attribution_mask(array, segmentation, "Pointing Game")
    values = _rank_values(flat, use_abs)
    if values.size > 1 and np.all(values == values[0]):
        raise ValueError(
            "Pointing Game: a uniform attribution map has no distinguished " "maximum point."
        )

    maximum = np.max(values)
    maxima = values == maximum
    if tolerance == 0:
        return float(np.any(mask_flat[maxima]))
    if array.ndim != 2:
        raise ValueError("Pointing Game: tolerance is defined only for 2-D spatial maps.")

    height, width = array.shape
    for y, x in np.argwhere(maxima.reshape(array.shape)):
        y_min = max(0, int(y) - int(tolerance))
        y_max = min(height, int(y) + int(tolerance) + 1)
        x_min = max(0, int(x) - int(tolerance))
        x_max = min(width, int(x) + int(tolerance) + 1)
        if np.any(segmentation[y_min:y_max, x_min:x_max]):
            return 1.0
    return 0.0


def compute_batch_pointing_game(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
    *,
    use_abs: bool = False,
    tolerance: int = 0,
) -> List[float]:
    """Compute Pointing Game hits for aligned instances."""
    return _batch_apply(
        attributions_batch,
        masks_batch,
        "Pointing Game",
        lambda a, s: compute_pointing_game(a, s, use_abs=use_abs, tolerance=tolerance),
    )


def compute_attribution_localisation(
    attributions: AttributionInput,
    mask: MaskInput,
    *,
    use_abs: bool = False,
) -> float:
    """Compute unweighted Attribution Localisation ``R_in / R_total``.

    Kohlbrenner et al. define both terms using positive relevance.  Therefore
    the canonical default clips negative relevance to zero.  ``use_abs=True``
    is a non-canonical but explicit absolute-relevance variant.
    """
    array = _extract_attributions(attributions)
    segmentation = _extract_mask(mask)
    flat, mask_flat = _validate_attribution_mask(array, segmentation, "Attribution Localisation")
    relevance = np.abs(flat) if _validate_bool(use_abs, "use_abs") else np.clip(flat, 0.0, None)
    if not np.any(relevance):
        raise ValueError(
            "Attribution Localisation: total positive relevance is zero; " "the ratio is undefined."
        )
    return _stable_nonnegative_mass_ratio(relevance, mask_flat)


def compute_batch_attribution_localisation(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
    *,
    use_abs: bool = False,
) -> List[float]:
    """Compute Attribution Localisation for aligned instances."""
    return _batch_apply(
        attributions_batch,
        masks_batch,
        "Attribution Localisation",
        lambda a, s: compute_attribution_localisation(a, s, use_abs=use_abs),
    )


def compute_top_k_intersection(
    attributions: AttributionInput,
    mask: MaskInput,
    *,
    k: Optional[int] = None,
    use_abs: bool = False,
) -> float:
    """Compute Theiner et al.'s top-k intersection ``|top_k ∩ S| / k``.

    ``k`` is required because the paper treats it as an independent parameter
    (and uses ``k=1000``); setting it to the mask size would silently turn this
    into Relevance Rank Accuracy.
    """
    if k is None:
        raise ValueError(
            "Top-K Intersection requires an explicit k; use Relevance Rank "
            "Accuracy when k should equal the mask size."
        )
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}.")

    array = _extract_attributions(attributions)
    segmentation = _extract_mask(mask)
    flat, mask_flat = _validate_attribution_mask(array, segmentation, "Top-K Intersection")
    if k > flat.size:
        raise ValueError(f"k={k} exceeds number of elements n={flat.size}.")
    return _top_k_hit_fraction(_rank_values(flat, use_abs), mask_flat, int(k), "Top-K Intersection")


def compute_batch_top_k_intersection(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
    *,
    k: Optional[int] = None,
    use_abs: bool = False,
) -> List[float]:
    """Compute Top-K Intersection for aligned instances."""
    return _batch_apply(
        attributions_batch,
        masks_batch,
        "Top-K Intersection",
        lambda a, s: compute_top_k_intersection(a, s, k=k, use_abs=use_abs),
    )


def compute_relevance_mass_accuracy(
    attributions: AttributionInput,
    mask: MaskInput,
    *,
    use_abs: bool = False,
    normalise: bool = False,
) -> float:
    """Compute CLEVR-XAI Relevance Mass Accuracy.

    The paper's input is an already pooled, positive, single-channel heatmap.
    Signed inputs are therefore rejected unless ``use_abs=True`` is explicitly
    requested.  Optional ``normalise=True`` divides by the positive maximum;
    unlike min-max shifting, this scale-only operation preserves the canonical
    mass ratio.
    """
    array = _extract_attributions(attributions)
    segmentation = _extract_mask(mask)
    flat, mask_flat = _validate_attribution_mask(array, segmentation, "Relevance Mass Accuracy")
    absolute = _validate_bool(use_abs, "use_abs")
    do_normalise = _validate_bool(normalise, "normalise")
    relevance = np.abs(flat) if absolute else flat
    if not absolute and np.any(relevance < 0.0):
        raise ValueError(
            "Relevance Mass Accuracy requires a non-negative, pre-pooled "
            "heatmap; pass use_abs=True only when absolute relevance is the "
            "intended pooling convention."
        )
    if do_normalise:
        maximum = float(np.max(relevance))
        if maximum > 0.0:
            relevance = relevance / maximum

    if not np.any(relevance):
        raise ValueError(
            "Relevance Mass Accuracy: total relevance is zero; the ratio is " "undefined."
        )
    return _stable_nonnegative_mass_ratio(relevance, mask_flat)


def compute_batch_relevance_mass_accuracy(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
    *,
    use_abs: bool = False,
    normalise: bool = False,
) -> List[float]:
    """Compute Relevance Mass Accuracy for aligned instances."""
    return _batch_apply(
        attributions_batch,
        masks_batch,
        "Relevance Mass Accuracy",
        lambda a, s: compute_relevance_mass_accuracy(a, s, use_abs=use_abs, normalise=normalise),
    )


def compute_relevance_rank_accuracy(
    attributions: AttributionInput,
    mask: MaskInput,
    *,
    use_abs: bool = False,
) -> float:
    """Compute CLEVR-XAI Relevance Rank Accuracy.

    The top ``|S|`` pixels of the positive, already pooled heatmap are selected
    and their overlap with ``S`` is divided by ``|S|``.
    """
    array = _extract_attributions(attributions)
    segmentation = _extract_mask(mask)
    flat, mask_flat = _validate_attribution_mask(array, segmentation, "Relevance Rank Accuracy")
    absolute = _validate_bool(use_abs, "use_abs")
    relevance = np.abs(flat) if absolute else flat
    if not absolute and np.any(relevance < 0.0):
        raise ValueError(
            "Relevance Rank Accuracy requires a non-negative, pre-pooled "
            "heatmap; pass use_abs=True only when absolute relevance is the "
            "intended pooling convention."
        )
    k = int(np.sum(mask_flat))
    return _top_k_hit_fraction(relevance, mask_flat, k, "Relevance Rank Accuracy")


def compute_batch_relevance_rank_accuracy(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
    *,
    use_abs: bool = False,
) -> List[float]:
    """Compute Relevance Rank Accuracy for aligned instances."""
    return _batch_apply(
        attributions_batch,
        masks_batch,
        "Relevance Rank Accuracy",
        lambda a, s: compute_relevance_rank_accuracy(a, s, use_abs=use_abs),
    )


def compute_auc(
    attributions: AttributionInput,
    mask: MaskInput,
    *,
    use_abs: bool = False,
) -> float:
    """Compute pixel/feature-wise ROC AUC using Mann-Whitney ranks.

    Tied scores receive average ranks.  A mask containing only one class is
    rejected because the ROC curve and its area are undefined.
    """
    array = _extract_attributions(attributions)
    segmentation = _extract_mask(mask)
    flat, mask_flat = _validate_attribution_mask(array, segmentation, "AUC")
    n_positive = int(np.sum(mask_flat))
    n_negative = int(mask_flat.size - n_positive)
    if n_negative == 0:
        raise ValueError(
            "AUC: ground-truth mask contains no negative elements; ROC AUC is " "undefined."
        )

    scores = _rank_values(flat, use_abs)
    order = np.argsort(scores, kind="stable")
    ranks: np.ndarray = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop

    rank_sum_positive = float(np.sum(ranks[mask_flat == 1.0]))
    mann_whitney_u = rank_sum_positive - n_positive * (n_positive + 1) / 2.0
    return float(mann_whitney_u / (n_positive * n_negative))


def compute_batch_auc(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
    *,
    use_abs: bool = False,
) -> List[float]:
    """Compute ROC AUC for aligned instances."""
    return _batch_apply(
        attributions_batch,
        masks_batch,
        "AUC",
        lambda a, s: compute_auc(a, s, use_abs=use_abs),
    )


def compute_energy_based_pointing_game(
    attributions: AttributionInput,
    mask: MaskInput,
    *,
    use_abs: bool = False,
) -> float:
    """Compute Score-CAM's Energy-Based Pointing Game proportion.

    The source metric divides non-negative saliency energy inside the target
    box by total saliency energy.  Signed inputs are rejected unless the caller
    explicitly selects the absolute-relevance preprocessing variant.
    """
    array = _extract_attributions(attributions)
    segmentation = _extract_mask(mask)
    flat, mask_flat = _validate_attribution_mask(array, segmentation, "Energy-Based Pointing Game")
    absolute = _validate_bool(use_abs, "use_abs")
    energy = np.abs(flat) if absolute else flat
    if not absolute and np.any(energy < 0.0):
        raise ValueError(
            "Energy-Based Pointing Game requires a non-negative saliency map; "
            "pass use_abs=True only when absolute saliency is intended."
        )
    if not np.any(energy):
        raise ValueError(
            "Energy-Based Pointing Game: total saliency energy is zero; the "
            "proportion is undefined."
        )
    return _stable_nonnegative_mass_ratio(energy, mask_flat)


def compute_batch_energy_based_pointing_game(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
    *,
    use_abs: bool = False,
) -> List[float]:
    """Compute Energy-Based Pointing Game for aligned instances."""
    return _batch_apply(
        attributions_batch,
        masks_batch,
        "Energy-Based Pointing Game",
        lambda a, s: compute_energy_based_pointing_game(a, s, use_abs=use_abs),
    )


def _validate_focus_mosaic(
    attributions: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the 2-by-2 mosaic experiment defined by the Focus paper."""
    flat, mask_flat = _validate_attribution_mask(attributions, mask, "Focus")
    if attributions.ndim != 2:
        raise ValueError("Focus is defined for a 2-D, 2-by-2 image mosaic.")
    height, width = attributions.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            "Focus requires even spatial dimensions so the image divides into "
            "four equal quadrants."
        )

    half_height, half_width = height // 2, width // 2
    quadrants = (
        mask[:half_height, :half_width],
        mask[:half_height, half_width:],
        mask[half_height:, :half_width],
        mask[half_height:, half_width:],
    )
    selected: List[bool] = []
    for quadrant in quadrants:
        if not np.all(quadrant == quadrant.flat[0]):
            raise ValueError(
                "Focus mask must select complete mosaic quadrants, not an "
                "arbitrary spatial region."
            )
        selected.append(bool(quadrant.flat[0]))
    if sum(selected) != 2:
        raise ValueError(
            "Focus requires exactly two of the four mosaic quadrants to belong "
            "to the target class."
        )
    return flat, mask_flat


def compute_focus(attributions: AttributionInput, mask: MaskInput) -> float:
    """Compute Focus for a canonical 2-by-2 mosaic.

    ``mask`` must identify the two complete quadrants whose images belong to
    the target class.  Focus is the positive relevance in those quadrants
    divided by total positive relevance.
    """
    array = _extract_attributions(attributions)
    segmentation = _extract_mask(mask)
    flat, mask_flat = _validate_focus_mosaic(array, segmentation)
    positive = np.clip(flat, 0.0, None)
    if not np.any(positive):
        raise ValueError("Focus: total positive relevance is zero; the score is undefined.")
    return _stable_nonnegative_mass_ratio(positive, mask_flat)


def compute_batch_focus(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
) -> List[float]:
    """Compute Focus for aligned canonical mosaics."""
    return _batch_apply(attributions_batch, masks_batch, "Focus", compute_focus)


def compute_attribution_iou(
    attributions: AttributionInput,
    mask: MaskInput,
    *,
    threshold: Optional[float] = None,
    use_abs: bool = True,
    percentile: Optional[float] = None,
) -> float:
    """Compute a library-defined thresholded attribution IoU diagnostic.

    Exactly one threshold mode is required.  Selection uses the documented
    strict comparison ``value > threshold``; consequently, percentile ties may
    select fewer elements than a nominal percentile would suggest.
    """
    if (threshold is None) == (percentile is None):
        raise ValueError("Exactly one of 'threshold' or 'percentile' must be provided.")
    if threshold is not None:
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
            raise TypeError("threshold must be a finite real number.")
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite.")
    if percentile is not None:
        if isinstance(percentile, (bool, np.bool_)) or not isinstance(percentile, Real):
            raise TypeError("percentile must be a finite real number.")
        if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be finite and in [0, 100], got {percentile}.")

    array = _extract_attributions(attributions)
    segmentation = _extract_mask(mask)
    flat, mask_flat = _validate_attribution_mask(array, segmentation, "Attribution IoU")
    values = _rank_values(flat, use_abs)
    if percentile is not None:
        selected = _percentile_mask(values, float(percentile), comparison="above")
    else:
        assert threshold is not None
        cutoff = float(threshold)
        selected = values > cutoff
    target: np.ndarray = mask_flat.astype(bool)
    union = int(np.count_nonzero(selected | target))
    if union == 0:
        raise ValueError("Attribution IoU: union is empty; IoU is undefined.")
    intersection = int(np.count_nonzero(selected & target))
    return float(intersection / union)


def compute_batch_attribution_iou(
    attributions_batch: Sequence[AttributionInput],
    masks_batch: Sequence[MaskInput],
    *,
    threshold: Optional[float] = None,
    use_abs: bool = True,
    percentile: Optional[float] = None,
) -> List[float]:
    """Compute thresholded Attribution IoU for aligned instances."""
    return _batch_apply(
        attributions_batch,
        masks_batch,
        "Attribution IoU",
        lambda a, s: compute_attribution_iou(
            a,
            s,
            threshold=threshold,
            use_abs=use_abs,
            percentile=percentile,
        ),
    )
