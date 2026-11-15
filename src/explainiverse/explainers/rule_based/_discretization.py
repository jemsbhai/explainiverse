"""Empirical quantile cuts for the tabular Anchors explainers."""

from math import isfinite

import numpy as np

from explainiverse.evaluation._utils import _exact_percentile


def quantile_cuts(values: np.ndarray, discretizer: str) -> np.ndarray:
    percentiles = [25, 50, 75] if discretizer == "quartile" else list(range(10, 100, 10))
    if isfinite(float(values.max()) - float(values.min())):
        return np.unique(np.percentile(values, percentiles))

    # Subtracting neighboring order statistics can overflow even though
    # their interpolated cut lies between the two finite endpoints.
    return np.unique([float(_exact_percentile(values, p)) for p in percentiles])
