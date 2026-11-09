"""Agreement metrics reject complex values before converting real inputs."""

import warnings
from decimal import Decimal

import numpy as np
import pytest

from explainiverse import Explanation
from explainiverse.evaluation.agreement import (
    compute_batch_feature_agreement,
    compute_batch_rank_agreement,
    compute_feature_agreement,
    compute_rank_agreement,
)


def _explanation(values):
    return Explanation(
        "fixed",
        0,
        {"feature_attributions": dict(zip(["f0", "f1"], values))},
        feature_names=["f0", "f1"],
    )


@pytest.mark.parametrize("metric", [compute_feature_agreement, compute_rank_agreement])
@pytest.mark.parametrize("argument", [0, 1], ids=["left", "right"])
@pytest.mark.parametrize(
    "container,value",
    [
        pytest.param("native", np.complex64(1 + 100j), id="complex64"),
        pytest.param("native", np.complex128(1 + 0j), id="zero-imaginary"),
        pytest.param("native", np.complex128(complex(1, np.inf)), id="infinite-imaginary"),
        pytest.param("native", np.complex128(complex(1, np.nan)), id="nan-imaginary"),
        pytest.param("object", np.complex128(1 + 100j), id="object-numpy-complex"),
        pytest.param("object", np.complex128(1 + 0j), id="object-zero-imaginary"),
        pytest.param("object", complex(1, 100), id="object-python-complex"),
        pytest.param("explanation", np.complex128(1 + 100j), id="explanation-numpy-complex"),
        pytest.param("explanation", complex(1, 100), id="explanation-python-complex"),
        pytest.param("explanation", np.complex128(1 + 0j), id="explanation-zero-imaginary"),
    ],
)
def test_complex_attributions_fail_before_any_lossy_cast(metric, argument, container, value):
    reference = np.array([1.0, 2.0])
    if container == "native":
        values = np.array([value, type(value)(2.0)])
    elif container == "object":
        values = np.array([value, 2.0], dtype=object)
    else:
        values = _explanation([value, 2.0])
        # Exercise validation before the two-Explanation identity alignment.
        reference = _explanation([1.0, 2.0])
    arguments = [reference, reference]
    arguments[argument] = values

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="complex"):
            metric(*arguments, k=1)


@pytest.mark.parametrize("metric", [compute_feature_agreement, compute_rank_agreement])
@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1], dtype=np.int64),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([Decimal(0), Decimal(1)], dtype=object),
        np.array([False, True], dtype=object),
    ],
    ids=["integer", "float32", "real-object", "boolean-object"],
)
def test_existing_real_conversions_keep_analytical_agreement(metric, values):
    before = values.copy()
    assert metric(values, np.array([1.0, 0.0]), k=1) == 0.0
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("metric", [compute_feature_agreement, compute_rank_agreement])
def test_real_explanations_keep_feature_identity_alignment(metric):
    left = _explanation([1.0, 2.0])
    right = Explanation(
        "fixed",
        0,
        {"feature_attributions": {"f1": Decimal(2), "f0": Decimal(1)}},
        feature_names=["f1", "f0"],
    )
    assert metric(left, right, k=2) == 1.0


@pytest.mark.parametrize("metric", [compute_batch_feature_agreement, compute_batch_rank_agreement])
@pytest.mark.parametrize("argument", [0, 1], ids=["left", "right"])
def test_batch_rejects_a_later_complex_pair(metric, argument):
    arguments = [
        [np.array([1.0, 2.0]), np.array([1.0, 2.0])],
        [np.array([1.0, 2.0]), np.array([1.0, 2.0])],
    ]
    arguments[argument][1] = _explanation([np.complex128(1 + 100j), 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="complex"):
            metric(*arguments, k=1)
