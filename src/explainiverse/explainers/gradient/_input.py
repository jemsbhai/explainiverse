"""Numeric input preparation shared by gradient explainers."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np

from explainiverse.explainers._validation import as_real_array


def as_floating_array(values, *, name: str) -> np.ndarray:
    """Return a finite real array without narrowing an existing float dtype.

    Gradient methods require differentiable floating inputs. Integer inputs are
    therefore promoted to float64 and subsequently aligned to the wrapped
    model by :class:`~explainiverse.adapters.PyTorchAdapter`; floating inputs
    retain their caller-provided precision.
    """

    array = as_real_array(values, name=name)
    if array.dtype.kind == "b" or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real values")
    if np.issubdtype(array.dtype, np.floating):
        return np.array(array, copy=True)
    return array.astype(np.float64, copy=True)


def _reduction_axes(axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    raw_axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    axes = tuple(int(value) % ndim for value in raw_axes)
    if len(set(axes)) != len(axes):
        raise ValueError("reduction axes must be unique")
    return tuple(sorted(axes))


def _stable_reduction(values: np.ndarray, axis, *, divisor: int = 1) -> np.ndarray:
    """Reduce binary64 slices with compensated and exact fallback arithmetic.

    ``math.fsum`` retains residuals that an order-dependent NumPy reduction can
    erase.  Its accumulator can still overflow before a later cancellation,
    however, so those rare slices are recomputed from the exact decimal value
    of each binary64 input.  Division is performed before conversion back to a
    float, allowing a representable mean even when its unscaled sum is not.
    """

    array = np.asarray(values, dtype=np.float64)
    axes = _reduction_axes(axis, array.ndim)
    if not axes:
        return array / divisor
    remaining = tuple(index for index in range(array.ndim) if index not in axes)
    transposed = np.transpose(array, remaining + axes)
    outer_shape = tuple(array.shape[index] for index in remaining)
    reduction_size = int(np.prod([array.shape[index] for index in axes], dtype=np.int64))
    if reduction_size == 0:
        raise ValueError("values must not be empty along the reduction axes")
    rows = transposed.reshape(-1, reduction_size)

    def reduce_row(row: np.ndarray) -> float:
        def exact_reduction() -> float:
            with localcontext() as context:
                context.prec = 1500 + len(str(reduction_size))
                total = sum(
                    (Decimal.from_float(float(value)) for value in row),
                    start=Decimal(0),
                )
                exact = total / Decimal(divisor)
                reduced = float(exact)
            if not np.isfinite(reduced) or (reduced == 0.0 and exact != 0):
                raise FloatingPointError("reduction result is not representable")
            return reduced

        # A rounded binary64 sum followed by division can double-round a mean.
        # Keep the exact sum and division together so there is only one public
        # binary64 rounding whenever a divisor is present.
        if divisor != 1:
            return exact_reduction()
        try:
            result = math.fsum(float(value) for value in row) / divisor
        except OverflowError:
            return exact_reduction()
        if not np.isfinite(result) or (result == 0.0 and np.any(row != 0.0)):
            return exact_reduction()
        return result

    sums = np.fromiter(
        (reduce_row(row) for row in rows),
        dtype=np.float64,
        count=rows.shape[0],
    )
    return sums.reshape(outer_shape)


def _stable_std_reduction(values: np.ndarray, axis) -> np.ndarray:
    """Compute a correctly centered population standard deviation.

    Centering on a rounded mean overstates the spread of adjacent floats.  The
    mean, squared deviations, division, and square root therefore remain in
    exact Decimal arithmetic until the one binary64 result rounding.
    """

    array = np.asarray(values, dtype=np.float64)
    axes = _reduction_axes(axis, array.ndim)
    if not axes:
        return np.zeros_like(array)
    remaining = tuple(index for index in range(array.ndim) if index not in axes)
    order = remaining + axes
    outer_shape = tuple(array.shape[index] for index in remaining)
    reduction_size = int(np.prod([array.shape[index] for index in axes], dtype=np.int64))
    if reduction_size == 0:
        raise ValueError("values must not be empty along the reduction axes")
    rows = np.transpose(array, order).reshape(-1, reduction_size)

    def reduce_row(row: np.ndarray) -> float:
        with localcontext() as context:
            context.prec = 3000 + len(str(reduction_size))
            decimal_values = [Decimal.from_float(float(value)) for value in row]
            mean = sum(decimal_values, start=Decimal(0)) / Decimal(reduction_size)
            variance = sum(
                ((value - mean) * (value - mean) for value in decimal_values),
                start=Decimal(0),
            ) / Decimal(reduction_size)
            exact = variance.sqrt()
            result = float(exact)
        if not np.isfinite(result) or (result == 0.0 and exact != 0):
            raise FloatingPointError("standard deviation is not representable")
        return result

    result = np.fromiter(
        (reduce_row(row) for row in rows),
        dtype=np.float64,
        count=rows.shape[0],
    )
    return result.reshape(outer_shape)


def _stable_product_sum_reduction(
    left: np.ndarray,
    right: np.ndarray,
    axis,
    *,
    divisor: int = 1,
) -> np.ndarray:
    """Return order-independent dot products, including extreme cancellation.

    Finite pairwise products use ``math.fsum``.  If a product or compensated
    accumulator is not representable, exact decimal multiplication and
    summation avoid losing a representable residual hidden between products
    whose individual magnitudes exceed binary64 range.
    """

    left_values, right_values = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    )
    axes = _reduction_axes(axis, left_values.ndim)
    if not axes:
        return left_values * right_values
    remaining = tuple(index for index in range(left_values.ndim) if index not in axes)
    order = remaining + axes
    outer_shape = tuple(left_values.shape[index] for index in remaining)
    reduction_size = int(np.prod([left_values.shape[index] for index in axes], dtype=np.int64))
    if reduction_size == 0:
        raise ValueError("values must not be empty along the reduction axes")
    left_rows = np.transpose(left_values, order).reshape(-1, reduction_size)
    right_rows = np.transpose(right_values, order).reshape(-1, reduction_size)

    def exact_row(left_row: np.ndarray, right_row: np.ndarray) -> float:
        # Products of exact binary64 decimal expansions can span about 1,300
        # decimal orders of magnitude.  This precision retains every possible
        # binary64 product digit plus accumulation carry digits.
        with localcontext() as context:
            context.prec = 2500 + len(str(reduction_size))
            total = sum(
                (
                    Decimal.from_float(float(left_value)) * Decimal.from_float(float(right_value))
                    for left_value, right_value in zip(left_row, right_row)
                ),
                start=Decimal(0),
            )
            exact = total / Decimal(divisor)
            result = float(exact)
        if not np.isfinite(result) or (result == 0.0 and exact != 0):
            raise FloatingPointError("product-sum result is not representable")
        return result

    def reduce_row(left_row: np.ndarray, right_row: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            products = left_row * right_row
        underflowed_product = np.any((products == 0.0) & (left_row != 0.0) & (right_row != 0.0))
        if divisor == 1 and np.all(np.isfinite(products)) and not underflowed_product:
            try:
                return math.fsum(float(value) for value in products)
            except OverflowError:
                pass
        return exact_row(left_row, right_row)

    result = np.fromiter(
        (reduce_row(left_row, right_row) for left_row, right_row in zip(left_rows, right_rows)),
        dtype=np.float64,
        count=left_rows.shape[0],
    )
    return result.reshape(outer_shape)


def _stable_multi_product_sum_reduction(*factors: np.ndarray, axis) -> np.ndarray:
    """Exactly multiply each factor tuple and sum over one or more axes."""

    arrays = np.broadcast_arrays(*(np.asarray(factor, dtype=np.float64) for factor in factors))
    if not arrays:
        raise ValueError("at least one product factor is required")
    axes = _reduction_axes(axis, arrays[0].ndim)
    if not axes:
        return scale_safe_product(*arrays)
    remaining = tuple(index for index in range(arrays[0].ndim) if index not in axes)
    order = remaining + axes
    outer_shape = tuple(arrays[0].shape[index] for index in remaining)
    reduction_size = int(np.prod([arrays[0].shape[index] for index in axes], dtype=np.int64))
    if reduction_size == 0:
        raise ValueError("values must not be empty along the reduction axes")
    rows_by_factor = [np.transpose(array, order).reshape(-1, reduction_size) for array in arrays]
    result = np.empty(rows_by_factor[0].shape[0], dtype=np.float64)
    with localcontext() as context:
        context.prec = 3500 + len(str(reduction_size * len(arrays)))
        for row_index in range(result.size):
            total = Decimal(0)
            for reduction_index in range(reduction_size):
                product = Decimal(1)
                for rows in rows_by_factor:
                    product *= Decimal.from_float(float(rows[row_index, reduction_index]))
                total += product
            value = float(total)
            if not np.isfinite(value) or (value == 0.0 and total != 0):
                raise FloatingPointError("multi-product sum is not representable")
            result[row_index] = value
    return result.reshape(outer_shape)


def _restore_reduced_axes(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values)
    for axis in axes:
        result = np.expand_dims(result, axis=axis)
    return result


def scale_safe_mean_std(values: np.ndarray, *, axis=0) -> tuple[np.ndarray, np.ndarray]:
    """Return an axis-0 mean and population standard deviation without sum overflow.

    Compensated/exact means preserve order-independent residuals across the
    full binary64 exponent range. Standard deviations retain exact binary64
    inputs through centering, squaring, averaging, and square root so adjacent
    values and extreme scales receive one correctly checked result rounding.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.size == 0:
        raise ValueError("values must contain at least one sample")
    axes = _reduction_axes(axis, array.ndim)
    count = int(np.prod([array.shape[index] for index in axes], dtype=np.int64))
    if count == 0:
        raise ValueError("values must contain at least one sample")
    mean = _stable_reduction(array, axes, divisor=count)
    return mean, _stable_std_reduction(array, axes)


def scale_safe_mean(
    values: np.ndarray,
    *,
    axis=0,
    keepdims: bool = False,
) -> np.ndarray:
    """Compute a correctly rounded, order-independent binary64 mean."""

    array = np.asarray(values, dtype=np.float64)
    axes = _reduction_axes(axis, array.ndim)
    count = int(np.prod([array.shape[index] for index in axes], dtype=np.int64))
    if count == 0:
        raise ValueError("values must not be empty along the reduction axes")
    result = _stable_reduction(array, axes, divisor=count)
    if keepdims:
        return _restore_reduced_axes(result, axes)
    return result


def scale_safe_sum(
    values: np.ndarray,
    *,
    axis=None,
    keepdims: bool = False,
) -> np.ndarray:
    """Sum finite values without overflowing a representable cancelled result."""

    array = np.asarray(values, dtype=np.float64)
    axes = _reduction_axes(axis, array.ndim)
    result = _stable_reduction(array, axes)
    if keepdims:
        return _restore_reduced_axes(result, axes)
    return result


def scale_safe_product(*factors: np.ndarray) -> np.ndarray:
    """Multiply broadcastable finite factors via mantissa/exponent arithmetic."""

    arrays = np.broadcast_arrays(*(np.asarray(factor, dtype=np.float64) for factor in factors))
    mantissa = np.ones(arrays[0].shape, dtype=np.float64)
    exponent = np.zeros(arrays[0].shape, dtype=np.int64)
    zero = np.zeros(arrays[0].shape, dtype=bool)
    for array in arrays:
        zero |= array == 0
        factor_mantissa, factor_exponent = np.frexp(array)
        mantissa *= factor_mantissa
        exponent += factor_exponent
    mantissa, adjustment = np.frexp(mantissa)
    exponent += adjustment
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.ldexp(mantissa, exponent)
    return np.where(zero, 0.0, result)


def scale_safe_product_sum(
    left: np.ndarray,
    right: np.ndarray,
    *,
    axis,
    divisor: int = 1,
) -> np.ndarray:
    """Sum pairwise products with optional exact pre-rounding division."""

    if isinstance(divisor, (bool, np.bool_)) or not isinstance(divisor, (int, np.integer)):
        raise TypeError("divisor must be a positive integer")
    if int(divisor) <= 0:
        raise ValueError("divisor must be a positive integer")

    return _stable_product_sum_reduction(left, right, axis, divisor=int(divisor))


def scale_safe_multi_product_sum(*factors: np.ndarray, axis) -> np.ndarray:
    """Exactly fuse factor multiplication with a reduction-axis sum."""

    return _stable_multi_product_sum_reduction(*factors, axis=axis)


def scale_safe_integrated_gradient(
    gradients: np.ndarray,
    baseline: np.ndarray,
    instance: np.ndarray,
    *,
    weights: np.ndarray,
    divisor: int,
) -> np.ndarray:
    """Fuse quadrature, endpoint difference, and multiplication exactly."""

    gradient_values = np.asarray(gradients, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    instance_values = np.asarray(instance, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        gradient_values.ndim < 1
        or gradient_values.shape[0] != weight_values.size
        or gradient_values.shape[1:] != baseline_values.shape
        or baseline_values.shape != instance_values.shape
    ):
        raise ValueError("integrated-gradient quadrature shapes are inconsistent")
    if isinstance(divisor, (bool, np.bool_)) or not isinstance(divisor, (int, np.integer)):
        raise TypeError("divisor must be a positive integer")
    if int(divisor) <= 0:
        raise ValueError("divisor must be a positive integer")
    if not (
        np.all(np.isfinite(gradient_values))
        and np.all(np.isfinite(baseline_values))
        and np.all(np.isfinite(instance_values))
        and np.all(np.isfinite(weight_values))
    ):
        raise ValueError("integrated-gradient quadrature values must be finite")

    flattened_gradients = gradient_values.reshape(gradient_values.shape[0], -1)
    flattened_baseline = baseline_values.reshape(-1)
    flattened_instance = instance_values.reshape(-1)
    result = np.empty(flattened_baseline.size, dtype=np.float64)
    with localcontext() as context:
        context.prec = 3500 + len(str(gradient_values.shape[0]))
        decimal_weights = [Decimal.from_float(float(value)) for value in weight_values]
        decimal_divisor = Decimal(int(divisor))
        for feature_index in range(result.size):
            weighted_sum = sum(
                (
                    Decimal.from_float(float(flattened_gradients[row, feature_index]))
                    * decimal_weights[row]
                    for row in range(gradient_values.shape[0])
                ),
                start=Decimal(0),
            )
            endpoint_delta = Decimal.from_float(
                float(flattened_instance[feature_index])
            ) - Decimal.from_float(float(flattened_baseline[feature_index]))
            exact = endpoint_delta * weighted_sum / decimal_divisor
            value = float(exact)
            if not np.isfinite(value) or (value == 0.0 and exact != 0):
                raise FloatingPointError("Integrated Gradients attribution is not representable")
            result[feature_index] = value
    return result.reshape(baseline_values.shape)


def scale_safe_spatial_mean_product_sum(
    activations: np.ndarray,
    gradients: np.ndarray,
) -> np.ndarray:
    """Fuse Grad-CAM's spatial gradient mean with its channel contraction.

    A per-channel mean can be smaller than the least binary64 subnormal even
    though combining several weighted channels produces a representable map.
    This rare fallback keeps the spatial sum, activation product, channel sum,
    and final division in exact Decimal arithmetic until the map cell's one
    binary64 rounding.
    """
    activation_values = np.asarray(activations, dtype=np.float64)
    gradient_values = np.asarray(gradients, dtype=np.float64)
    if (
        activation_values.shape != gradient_values.shape
        or activation_values.ndim != 4
        or not np.all(np.isfinite(activation_values))
        or not np.all(np.isfinite(gradient_values))
    ):
        raise ValueError("Grad-CAM activations and gradients must be paired finite 4-D arrays")

    batch_size, channels, height, width = activation_values.shape
    spatial_size = height * width
    result = np.empty((batch_size, height, width), dtype=np.float64)
    with localcontext() as context:
        context.prec = 3000 + len(str(channels * spatial_size))
        divisor = Decimal(spatial_size)
        for batch_index in range(batch_size):
            gradient_sums = [
                sum(
                    (
                        Decimal.from_float(float(value))
                        for value in gradient_values[batch_index, channel_index].reshape(-1)
                    ),
                    start=Decimal(0),
                )
                for channel_index in range(channels)
            ]
            for row_index in range(height):
                for column_index in range(width):
                    total = sum(
                        (
                            Decimal.from_float(
                                float(
                                    activation_values[
                                        batch_index,
                                        channel_index,
                                        row_index,
                                        column_index,
                                    ]
                                )
                            )
                            * gradient_sums[channel_index]
                            for channel_index in range(channels)
                        ),
                        start=Decimal(0),
                    )
                    exact = total / divisor
                    value = float(exact)
                    if not np.isfinite(value) or (value == 0.0 and exact != 0):
                        raise FloatingPointError("Grad-CAM map value is not representable")
                    result[batch_index, row_index, column_index] = value
    return result
