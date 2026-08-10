"""
First-order Accumulated Local Effects (ALE).

This implementation follows the continuous one-dimensional estimator in the
authors' reference ALEPlot implementation: local finite differences are
averaged within empirical quantile bins, accumulated at bin edges, and centered
with bin-count-weighted trapezoid values.

Reference:
    Apley, D.W. & Zhu, J. (2020). Visualizing the Effects of Predictor
    Variables in Black Box Supervised Learning Models. JRSS B, 82(4),
    1059-1086.
"""

from numbers import Integral
from typing import List, Optional, Tuple, Union

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence
from explainiverse.explainers.global_explainers.partial_dependence import (
    _normalize_predictions,
    _resolve_output_index,
    _resolve_task,
    _select_prediction_output,
)

Feature = Union[int, str]


class ALEExplainer(BaseExplainer):
    """Compute continuous first-order ALE curves at quantile-bin edges."""

    def __init__(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        n_bins: int = 20,
        task: Optional[str] = None,
    ):
        """
        Initialize ALE.

        Args:
            model: Adapter with ``predict`` and a declared ``task`` attribute.
            X: Non-empty numeric reference data.
            feature_names: Unique feature names in column order.
            n_bins: Positive maximum number of empirical quantile bins.
            task: Explicit task override, which must agree with ``model.task``.

        The implementation supports continuous/numeric first-order ALE. It does
        not impose an arbitrary order on nominal categorical features.
        """
        super().__init__(model)
        self.X: np.ndarray = np.asarray(X)
        validated_names = validate_name_sequence(feature_names, name="feature_names")
        assert validated_names is not None
        self.feature_names: List[str] = validated_names
        self.task: str = _resolve_task(model, task)

        if self.X.ndim != 2 or self.X.shape[0] == 0 or self.X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names length must equal the number of columns in X")
        if not isinstance(n_bins, Integral) or isinstance(n_bins, bool):
            raise TypeError("n_bins must be an integer")
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        self.n_bins: int = int(n_bins)

    def _get_feature_idx(self, feature: Feature) -> int:
        if isinstance(feature, str):
            try:
                return self.feature_names.index(feature)
            except ValueError as exc:
                raise ValueError(f"Unknown feature name: {feature!r}") from exc
        if not isinstance(feature, Integral) or isinstance(feature, bool):
            raise TypeError("feature must be an integer index or feature name")
        feature_idx = int(feature)
        if feature_idx < 0 or feature_idx >= self.X.shape[1]:
            raise ValueError(
                f"feature index must be in [0, {self.X.shape[1] - 1}]; " f"got {feature_idx}"
            )
        return feature_idx

    def _numeric_feature_values(self, feature_idx: int) -> np.ndarray:
        try:
            values = as_real_array(
                self.X[:, feature_idx],
                name=f"Feature {self.feature_names[feature_idx]!r}",
                dtype=float,
            )
        except ValueError as exc:
            if "complex values" in str(exc):
                raise
            raise TypeError(
                "First-order ALE currently requires an ordered numeric feature; "
                f"{self.feature_names[feature_idx]!r} is non-numeric"
            ) from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Feature {self.feature_names[feature_idx]!r} contains non-finite values"
            )
        return values

    def _compute_quantile_bins(self, values: np.ndarray) -> np.ndarray:
        """Return ALEPlot-compatible inverse-ECDF quantile edges."""
        percentiles = np.linspace(0.0, 100.0, self.n_bins + 1)
        return np.unique(np.percentile(values, percentiles, method="inverted_cdf"))

    def _selected_predictions(self, X: np.ndarray, output_index: int) -> np.ndarray:
        predictions = _normalize_predictions(self.model.predict(X), X.shape[0])
        return _select_prediction_output(predictions, self.task, output_index)

    def _compute_ale_details(
        self,
        feature_idx: int,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return edge grid, centered ALE, local effects, counts, and centers."""
        values = self._numeric_feature_values(feature_idx)
        bin_edges = self._compute_quantile_bins(values)
        reference = _normalize_predictions(self.model.predict(self.X[:1]), 1)
        output_index = _resolve_output_index(reference, self.task, target_class)

        if len(bin_edges) == 1:
            # No finite difference can be defined for a constant feature.
            return (
                bin_edges.copy(),
                np.array([0.0]),
                np.array([0.0]),
                np.array([len(values)], dtype=int),
                bin_edges.copy(),
            )

        # Right-closed intervals reproduce R's cut(..., include.lowest=TRUE),
        # which is used by the authors' ALEPlot reference implementation.
        # Requested empirical quantiles can collapse in tied/discrete numeric
        # data. Defensively merge an empty interval with its successor rather
        # than inventing a zero local effect.
        while True:
            n_intervals = len(bin_edges) - 1
            bin_indices: np.ndarray = np.searchsorted(bin_edges[1:], values, side="left").astype(
                int
            )
            bin_indices = np.clip(bin_indices, 0, n_intervals - 1)
            bin_counts = np.bincount(bin_indices, minlength=n_intervals)
            empty_bins = np.flatnonzero(bin_counts == 0)
            if len(empty_bins) == 0:
                break
            # Min and max guarantee the boundary bins are occupied; an empty
            # bin is therefore internal, and deleting its upper edge merges it
            # into the following occupied interval.
            bin_edges = np.delete(bin_edges, int(empty_bins[0]) + 1)

        # Use floating copies for numerical model evaluation so assigning a
        # floating boundary never truncates through the reference array dtype.
        if np.issubdtype(self.X.dtype, np.integer):
            X_lower: np.ndarray = self.X.astype(float, copy=True)
            X_upper: np.ndarray = self.X.astype(float, copy=True)
        else:
            X_lower = self.X.copy()
            X_upper = self.X.copy()
        X_lower[:, feature_idx] = bin_edges[bin_indices]
        X_upper[:, feature_idx] = bin_edges[bin_indices + 1]

        prediction_differences = self._selected_predictions(
            X_upper, output_index
        ) - self._selected_predictions(X_lower, output_index)
        local_effects = (
            np.bincount(bin_indices, weights=prediction_differences, minlength=n_intervals)
            / bin_counts
        )

        # Effects and coordinates are both at bin edges. The prior
        # implementation paired these upper-edge cumulative values with bin
        # centers, shifting every plotted point by half a bin.
        accumulated_at_edges = np.concatenate(([0.0], np.cumsum(local_effects)))

        # Canonical ALEPlot centering: approximate the accumulated effect for
        # each observation by the trapezoid midpoint of its bin, then take the
        # empirical (bin-count-weighted) mean.
        effects_at_bin_centers = (accumulated_at_edges[:-1] + accumulated_at_edges[1:]) / 2.0
        centering_constant = float(np.average(effects_at_bin_centers, weights=bin_counts))
        centered_at_edges = accumulated_at_edges - centering_constant
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        return (
            bin_edges,
            centered_at_edges,
            local_effects,
            bin_counts,
            bin_centers,
        )

    def _compute_ale_1d(
        self,
        feature_idx: int,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward-compatible private wrapper returning aligned edge values."""
        grid_values, ale_values, _, _, _ = self._compute_ale_details(feature_idx, target_class)
        return grid_values, ale_values, grid_values.copy()

    def explain(  # type: ignore[override]
        self,
        feature: Feature,
        target_class: Optional[int] = None,
        **kwargs,
    ) -> Explanation:
        """Compute continuous first-order ALE for one feature."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        feature_idx = self._get_feature_idx(feature)
        reference = _normalize_predictions(self.model.predict(self.X[:1]), 1)
        output_index = _resolve_output_index(reference, self.task, target_class)
        (
            grid_values,
            ale_values,
            local_effects,
            bin_counts,
            bin_centers,
        ) = self._compute_ale_details(feature_idx, output_index)

        if len(ale_values) == 1:
            ale_at_bin_centers = np.array([0.0])
        else:
            ale_at_bin_centers = (ale_values[:-1] + ale_values[1:]) / 2.0

        feature_name = self.feature_names[feature_idx]
        target_name = (
            f"class_{output_index}" if self.task == "classification" else f"output_{output_index}"
        )
        output_space = (
            "classification_score" if self.task == "classification" else "regression_response"
        )
        return Explanation(
            explainer_name="ALE",
            target_class=target_name,
            feature_names=self.feature_names,
            explanation_data={
                "ale_values": ale_values.tolist(),
                "grid_values": grid_values.tolist(),
                "bin_edges": grid_values.tolist(),
                "bin_centers": bin_centers.tolist(),
                "ale_values_at_bin_centers": ale_at_bin_centers.tolist(),
                "local_effects": local_effects.tolist(),
                "bin_counts": bin_counts.tolist(),
                "feature": feature_name,
                "curve_range": float(np.ptp(ale_values)),
                "curve_range_semantics": "max_minus_min_of_centered_ale_curve",
                "task": self.task,
                "output_index": output_index,
                "output_space": output_space,
            },
            metadata={
                "task": self.task,
                "output_index": output_index,
                "output_space": output_space,
                "prediction_output": "model.predict",
                "curve_coordinates": "grid_values (quantile-bin edges)",
                "centering": "empirical_bin_count_weighted_trapezoid",
                "quantile_method": "inverse_empirical_cdf (R type=1)",
                "n_requested_bins": self.n_bins,
                "n_effective_bins": max(0, len(grid_values) - 1),
            },
        )

    def explain_all(self, target_class: Optional[int] = None) -> List[Explanation]:
        """Compute first-order ALE independently for every feature."""
        return [
            self.explain(feature_idx, target_class=target_class)
            for feature_idx in range(len(self.feature_names))
        ]
