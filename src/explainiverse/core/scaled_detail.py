"""Versioned opt-in representation for exact details outside binary64 range."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Real
from typing import Any, Union

import numpy as np

SCALED_DETAIL_SCHEMA_VERSION = "explainiverse.scaled-detail.v1"
LEGACY_DETAIL_FORMAT = "float64"
# Retained compatibility selector: scaled-detail-v1 now distinguishes exact
# decimal values from canonical exact fractions instead of rounding ratios.
SCALED_DECIMAL_DETAIL_FORMAT = "scaled_decimal_v1"
_JS_MAX_SAFE_INTEGER = 2**53 - 1


def _unsafe_integer_float(value: float) -> bool:
    return value.is_integer() and not -_JS_MAX_SAFE_INTEGER <= value <= _JS_MAX_SAFE_INTEGER


class DetailRepresentationError(FloatingPointError):
    """A requested legacy detail element is not representable as binary64."""


def validate_detail_format(value: str) -> str:
    """Validate the opt-in detail representation selector."""

    if not isinstance(value, str):
        raise TypeError("detail_format must be a string")
    if value not in {LEGACY_DETAIL_FORMAT, SCALED_DECIMAL_DETAIL_FORMAT}:
        raise ValueError("detail_format must be 'float64' or 'scaled_decimal_v1'")
    return value


def _rounded_fraction(value: Fraction) -> Union[float, None]:
    """Return a finite binary64 rounding when Python can form one."""

    try:
        rounded = float(value)
    except OverflowError:
        return None
    return rounded if np.isfinite(rounded) else None


def _encode_exact_fraction(value: Fraction) -> Any:
    rounded = _rounded_fraction(value)
    if (
        rounded is not None
        and not (rounded == 0.0 and value != 0)
        and not _unsafe_integer_float(rounded)
        and Fraction.from_float(rounded) == value
    ):
        return rounded
    return {
        "exact_fraction": {
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
        }
    }


def _encode_exact_decimal(value: Decimal) -> Any:
    rounded = float(value)
    if (
        np.isfinite(rounded)
        and not (rounded == 0.0 and value != 0)
        and not _unsafe_integer_float(rounded)
        and Decimal.from_float(rounded) == value
    ):
        return rounded
    return {"exact_decimal": str(value)}


def _encode_scaled_scalar(value: Union[Decimal, Fraction, Real]) -> Any:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("scaled detail values must be real numbers, not booleans")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("scaled detail values must be finite")
        if value.is_zero() and value.is_signed():
            raise ValueError("scaled detail values must not contain negative zero")
        return _encode_exact_decimal(value)
    if isinstance(value, Fraction):
        return _encode_exact_fraction(value)
    if isinstance(value, Integral):
        return _encode_exact_fraction(Fraction(int(value), 1))
    if isinstance(value, np.floating):
        if value.dtype.itemsize > np.dtype(np.float64).itemsize:
            raise TypeError(
                "scaled detail NumPy floating scalars wider than binary64 are unsupported; "
                "convert the exact value to Decimal or Fraction"
            )
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("scaled detail values must be finite")
        if numeric == 0.0 and np.signbit(numeric):
            raise ValueError("scaled detail values must not contain negative zero")
        if _unsafe_integer_float(numeric):
            return _encode_exact_fraction(Fraction.from_float(numeric))
        return numeric
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("scaled detail values must be finite")
        if value == 0.0 and np.signbit(value):
            raise ValueError("scaled detail values must not contain negative zero")
        if _unsafe_integer_float(value):
            return _encode_exact_fraction(Fraction.from_float(value))
        return value
    if isinstance(value, Real):
        raise TypeError(
            "scaled detail real scalars must be Python int/float, Decimal, Fraction, "
            "or binary64-or-narrower NumPy floating values"
        )
    raise TypeError("scaled detail values must be finite real scalars")


def encode_scaled_detail(
    values: Sequence[Union[Decimal, Fraction, Real]],
    *,
    source_dtype: str = "float64",
) -> dict[str, Any]:
    """Encode exact values while retaining ordinary float entries unchanged.

    Native finite floats remain JSON numbers. Exact decimal values that cannot
    make a lossless binary64 round trip use ``exact_decimal``. Exact rational
    values use canonical numerator/denominator strings when binary64 cannot
    represent them, so a repeating decimal is never mislabeled as exact.
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("values must be a sequence of finite real scalars")
    if not isinstance(source_dtype, str) or not source_dtype.strip():
        raise ValueError("source_dtype must be a non-empty string")

    encoded: list[Any] = []
    for value in values:
        encoded.append(_encode_scaled_scalar(value))
    return {
        "schema_version": SCALED_DETAIL_SCHEMA_VERSION,
        "source_dtype": source_dtype.strip(),
        "values": encoded,
    }


def decode_scaled_detail(payload: Any) -> list[Union[float, Decimal, Fraction]]:
    """Validate and decode a scaled-detail-v1 payload without numeric loss."""

    if type(payload) is not dict:
        raise TypeError("scaled detail payload must be a plain dictionary")
    expected_fields = {"schema_version", "source_dtype", "values"}
    if set(payload) != expected_fields:
        raise ValueError("scaled detail payload must contain exactly the v1 schema fields")
    if payload["schema_version"] != SCALED_DETAIL_SCHEMA_VERSION:
        raise ValueError("unsupported scaled detail schema_version")
    if not isinstance(payload["source_dtype"], str) or not payload["source_dtype"].strip():
        raise ValueError("scaled detail source_dtype must be a non-empty string")
    if type(payload["values"]) is not list:
        raise TypeError("scaled detail values must be an ordinary list")

    decoded: list[Union[float, Decimal, Fraction]] = []
    for item in payload["values"]:
        if type(item) is float:
            if not np.isfinite(item) or (item == 0.0 and np.signbit(item)):
                raise ValueError("scaled detail float values must be finite and not negative zero")
            if _unsafe_integer_float(item):
                raise ValueError(
                    "scaled detail integer-valued floats must be within the JavaScript "
                    "safe-integer range"
                )
            decoded.append(item)
        elif type(item) is int and not isinstance(item, bool):
            if not -_JS_MAX_SAFE_INTEGER <= item <= _JS_MAX_SAFE_INTEGER:
                raise ValueError(
                    "scaled detail integer values must be within the JavaScript "
                    "safe-integer range"
                )
            decoded.append(float(item))
        elif type(item) is dict and set(item) == {"exact_decimal"}:
            text = item["exact_decimal"]
            if not isinstance(text, str) or not text:
                raise ValueError("exact_decimal must be a non-empty string")
            try:
                exact = Decimal(text)
            except Exception as exc:
                raise ValueError("exact_decimal is not a valid decimal") from exc
            if not exact.is_finite():
                raise ValueError("exact_decimal must be finite")
            if exact.is_zero() and exact.is_signed():
                raise ValueError("exact_decimal must not be negative zero")
            decoded.append(exact)
        elif type(item) is dict and set(item) == {"exact_fraction"}:
            components = item["exact_fraction"]
            if type(components) is not dict or set(components) != {
                "numerator",
                "denominator",
            }:
                raise ValueError("exact_fraction must contain exactly numerator and denominator")
            numerator_text = components["numerator"]
            denominator_text = components["denominator"]
            if type(numerator_text) is not str or type(denominator_text) is not str:
                raise TypeError("exact_fraction numerator and denominator must be strings")
            try:
                numerator = int(numerator_text)
                denominator = int(denominator_text)
            except ValueError as exc:
                raise ValueError("exact_fraction components must be base-10 integers") from exc
            if str(numerator) != numerator_text or str(denominator) != denominator_text:
                raise ValueError("exact_fraction components must use canonical integer strings")
            if denominator <= 0:
                raise ValueError("exact_fraction denominator must be positive")
            exact_fraction = Fraction(numerator, denominator)
            if exact_fraction.numerator != numerator or exact_fraction.denominator != denominator:
                raise ValueError("exact_fraction components must be reduced")
            decoded.append(exact_fraction)
        else:
            raise TypeError(
                "scaled detail values must be finite numbers, exact_decimal objects, "
                "or exact_fraction objects"
            )
    return decoded
