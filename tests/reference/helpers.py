"""
Helper utilities for reference validation tests.

These are placed in a separate file so test modules can import them
without needing 'tests' to be an installable package.

Usage in test files:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from helpers import assert_numerical_match, assert_rank_correlation
"""

import numpy as np

TOLERANCE_ATOL = 1e-5
TOLERANCE_RTOL = 1e-4


def assert_numerical_match(
    explainiverse_value,
    reference_value,
    metric_name: str,
    atol: float = TOLERANCE_ATOL,
    rtol: float = TOLERANCE_RTOL,
):
    """
    Assert two values are numerically close, with a clear error message.

    Works with scalars, 1D arrays, and 2D arrays.
    """
    ev = np.asarray(explainiverse_value, dtype=np.float64)
    rv = np.asarray(reference_value, dtype=np.float64)

    assert ev.shape == rv.shape, (
        f"{metric_name}: shape mismatch — " f"explainiverse={ev.shape}, reference={rv.shape}"
    )

    if not np.allclose(ev, rv, atol=atol, rtol=rtol):
        max_diff = np.max(np.abs(ev - rv))
        mean_diff = np.mean(np.abs(ev - rv))
        raise AssertionError(
            f"{metric_name}: numerical mismatch — "
            f"max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
            f"atol={atol}, rtol={rtol}\n"
            f"  explainiverse: {ev}\n"
            f"  reference:     {rv}"
        )


def assert_rank_correlation(
    explainiverse_attrs: np.ndarray,
    reference_attrs: np.ndarray,
    metric_name: str,
    min_correlation: float = 0.95,
):
    """
    Enforce a configured Spearman threshold on this fixed reference fixture.

    This is a regression criterion for the selected data, seeds, and settings;
    it is not evidence of universal method equivalence.
    """
    from scipy.stats import spearmanr

    for i in range(len(explainiverse_attrs)):
        corr, pval = spearmanr(explainiverse_attrs[i], reference_attrs[i])
        assert corr >= min_correlation, (
            f"{metric_name} instance {i}: rank correlation too low — "
            f"rho={corr:.4f} (min={min_correlation})\n"
            f"  explainiverse: {explainiverse_attrs[i]}\n"
            f"  reference:     {reference_attrs[i]}"
        )
