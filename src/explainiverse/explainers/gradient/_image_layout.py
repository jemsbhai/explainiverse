"""Explicit image-layout normalization shared by IG and CAM explainers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

_LAYOUTS = {"auto", "hw", "chw", "hwc", "nchw", "nhwc"}


def validate_image_layout(input_layout: str) -> str:
    """Validate a single-image layout declaration.

    ``auto`` is a rank convention, never a channel-size heuristic: two axes
    resolve to ``hw``, three to ``chw``, and four to ``nchw``. Callers with a
    channel-last representation must say so explicitly.
    """

    if not isinstance(input_layout, str):
        raise TypeError("input_layout must be 'auto', 'hw', 'chw', 'hwc', 'nchw', or 'nhwc'")
    normalized = input_layout.strip().lower()
    if normalized not in _LAYOUTS:
        raise ValueError("input_layout must be 'auto', 'hw', 'chw', 'hwc', 'nchw', or 'nhwc'")
    return normalized


def resolve_image_layout(value: np.ndarray, input_layout: str) -> str:
    """Resolve and validate the declared layout before model work."""

    layout = validate_image_layout(input_layout)
    if layout == "auto":
        inferred = {2: "hw", 3: "chw", 4: "nchw"}.get(value.ndim)
        if inferred is None:
            raise ValueError(
                "input_layout='auto' supports rank-2 HW, rank-3 CHW, or rank-4 "
                "NCHW inputs; declare a supported layout explicitly"
            )
        return inferred

    expected_rank = 2 if layout == "hw" else 3 if layout in {"chw", "hwc"} else 4
    if value.ndim != expected_rank:
        raise ValueError(
            f"input_layout={layout!r} requires a rank-{expected_rank} input; "
            f"got shape {value.shape}"
        )
    return layout


def image_to_nchw(value: np.ndarray, input_layout: str) -> Tuple[np.ndarray, str]:
    """Map one declared image representation to exactly one NCHW tensor."""

    layout = resolve_image_layout(value, input_layout)
    if any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"image dimensions must be positive; got {value.shape}")
    if layout == "hw":
        normalized = value[np.newaxis, np.newaxis, ...]
    elif layout == "chw":
        normalized = value[np.newaxis, ...]
    elif layout == "hwc":
        normalized = np.transpose(value, (2, 0, 1))[np.newaxis, ...]
    elif layout == "nchw":
        if value.shape[0] != 1:
            raise ValueError(
                "A single-image explanation requires an NCHW batch dimension of 1; "
                f"got {value.shape[0]}"
            )
        normalized = value
    else:  # nhwc
        if value.shape[0] != 1:
            raise ValueError(
                "A single-image explanation requires an NHWC batch dimension of 1; "
                f"got {value.shape[0]}"
            )
        normalized = np.transpose(value, (0, 3, 1, 2))
    return np.ascontiguousarray(normalized), layout


def gradient_from_nchw(
    gradient: np.ndarray, caller_shape: Tuple[int, ...], resolved_layout: str
) -> np.ndarray:
    """Map a one-row NCHW gradient back to the declared caller layout."""

    if gradient.ndim != 4 or gradient.shape[0] != 1:
        raise ValueError(
            "predict_with_gradients returned the wrong image gradient rank; "
            f"expected one NCHW row, got {gradient.shape}"
        )
    if resolved_layout == "hw":
        restored = gradient[0, 0]
    elif resolved_layout == "chw":
        restored = gradient[0]
    elif resolved_layout == "hwc":
        restored = np.transpose(gradient[0], (1, 2, 0))
    elif resolved_layout == "nchw":
        restored = gradient
    elif resolved_layout == "nhwc":
        restored = np.transpose(gradient, (0, 2, 3, 1))
    else:  # pragma: no cover - guarded by resolve_image_layout
        raise RuntimeError(f"Unknown resolved image layout {resolved_layout!r}")
    if tuple(restored.shape) != caller_shape:
        raise ValueError(
            "predict_with_gradients returned the wrong gradient shape; "
            f"expected caller shape {caller_shape}, got {restored.shape}"
        )
    return restored


def channel_axis_for_layout(resolved_layout: str) -> int | None:
    """Return the caller-visible channel axis for result metadata."""

    if resolved_layout == "hw":
        return None
    if resolved_layout in {"chw", "nchw"}:
        return 0 if resolved_layout == "chw" else 1
    if resolved_layout in {"hwc", "nhwc"}:
        return -1
    raise ValueError(f"Unknown resolved image layout {resolved_layout!r}")
