# src/explainiverse/evaluation/randomisation.py
"""
Randomisation diagnostics for explanations.

Implements:
- Model Parameter Randomisation Test — MPRT (Adebayo et al., 2018)
- Random Logit Test (Sixt et al., 2020)
- Smooth MPRT (Hedström et al., 2023)
- Efficient MPRT (Hedström et al., 2023)
- Data Randomisation Test (Adebayo et al., 2018)

These diagnostics measure whether explanations change under specified model,
output, or data randomisations. A score depends on the chosen similarity,
randomisation order, explainer, and data; it is not by itself proof of
faithfulness or unreliability.

Low-level score functions accept the attribution-array shapes documented by
each function. High-level model-randomisation APIs require supported PyTorch
models; callers must check the individual shape and target contracts.

Similarity Functions:
    Built-in similarity measures are dispatched via string keys:
    - "spearman": Spearman rank correlation (scipy.stats.spearmanr)
    - "pearson": Pearson correlation (scipy.stats.pearsonr)
    - "cosine": Cosine similarity (1 - scipy.spatial.distance.cosine)
    - "ssim": Structural Similarity Index for 2-D or channel-first ``(C,H,W)``
      maps (skimage.metrics — optional dep)
    - "mse": Negative Mean Squared Error (scipy/numpy)
    Custom callables f(a, b) -> float are also accepted.

References:
    Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
    & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html

    Sixt, L., Granz, M., & Landgraf, T. (2020). When Explanations Lie:
    Why Many Modified BP Attributions Fail. ICML.
    https://proceedings.mlr.press/v119/sixt20a.html

    Hedström, A., Weber, L., Lapuschkin, S., & Höhne, M. (2023).
    Sanity Checks Revisited: An Exploration to Repair the Model
    Parameter Randomisation Test. XAI in Action (xAI 2024).
    https://arxiv.org/abs/2401.06465
"""

import copy
import re
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple, TypedDict, Union

import numpy as np
from scipy import stats

from explainiverse.core.explanation import Explanation
from explainiverse.evaluation._utils import (
    _stable_cosine,
    _stable_mean,
    _stable_mean_square,
    _stable_pearson,
    _stable_spearman,
)

if TYPE_CHECKING:
    import torch


# =============================================================================
# Similarity Functions — Dispatcher and Built-ins
# =============================================================================

# Type alias for similarity functions: f(a, b) -> float
SimilarityFunc = Callable[[np.ndarray, np.ndarray], float]


def _finite_mean(values, context: str) -> float:
    result = float(_stable_mean(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{context} mean is not representable")
    return result


class MPRTResult(TypedDict):
    """Structured result shared by the MPRT variants."""

    layer_scores: List[float]
    layer_names: List[str]
    mean_score: float


class SmoothMPRTResult(MPRTResult, total=False):
    """MPRT core scores plus Smooth MPRT protocol-disclosure metadata."""

    variant: str
    randomisation_order: str
    nr_samples: int
    noise_magnitude: float
    smoothing_inputs_paired: bool
    rng_streams_split: bool
    clean_input_included: bool
    paper_default_order_used: bool
    paper_recommended_sample_count_met: bool
    paper_conformant_defaults_used: bool
    claim_scope: str


def _spearman_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation between two attribution vectors.

    Returns the correlation coefficient (range [-1, 1]). Correlation is
    mathematically undefined for fewer than two observations or when either
    vector is constant, so those cases raise instead of being silently scored
    as zero similarity.

    Args:
        a: First attribution vector (1-D, flattened).
        b: Second attribution vector (1-D, flattened).

    Returns:
        Spearman correlation coefficient.
    """
    if a.size < 2 or b.size < 2:
        raise ValueError("Spearman correlation requires at least two values per attribution.")
    if np.min(a) == np.max(a) or np.min(b) == np.max(b):
        raise ValueError("Spearman correlation is undefined for constant attributions.")
    try:
        return _stable_spearman(a, b)
    except ValueError as exc:
        raise ValueError("Spearman correlation is undefined for these attributions.") from exc


def _pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between two attribution vectors.

    Returns the correlation coefficient (range [-1, 1]). Correlation is
    undefined for fewer than two observations or constant inputs, so those
    cases raise instead of being silently scored as zero similarity.

    Args:
        a: First attribution vector (1-D, flattened).
        b: Second attribution vector (1-D, flattened).

    Returns:
        Pearson correlation coefficient.
    """
    if a.size < 2 or b.size < 2:
        raise ValueError("Pearson correlation requires at least two values per attribution.")
    try:
        return _stable_pearson(a, b)
    except ValueError as exc:
        raise ValueError("Pearson correlation is undefined for constant attributions.") from exc


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two attribution vectors.

    Computed as 1 - cosine_distance. Range [-1, 1], or [0, 1] when both
    nonzero attribution vectors are elementwise non-negative. It is undefined if either
    vector has zero norm, so that case raises instead of being silently scored
    as zero similarity.

    Args:
        a: First attribution vector (1-D, flattened).
        b: Second attribution vector (1-D, flattened).

    Returns:
        Cosine similarity.
    """
    try:
        return _stable_cosine(a, b)
    except ValueError as exc:
        raise ValueError("Cosine similarity is undefined for zero-norm attributions.") from exc


def _ssim_similarity(
    a: np.ndarray,
    b: np.ndarray,
    *,
    channel_axis: Optional[int] = None,
) -> float:
    """
    Structural Similarity Index (SSIM) between two attribution maps.

    Requires the package's scikit-image runtime dependency and 2-D or 3-D image maps.
    The built-in ``"ssim"`` similarity contract interprets every 3-D map as
    channel-first ``(C, H, W)``, matching the image-attribution layout emitted
    by the gradient explainers. Direct callers may pass another
    ``channel_axis`` (for example ``-1`` for ``(H, W, C)``). One-dimensional
    arrays are rejected; use Spearman or Pearson for the library's
    tabular-vector contract.

    Uses data_range computed from the union of both arrays to ensure
    consistent scaling.

    Args:
        a: First attribution map, either ``(H, W)`` or ``(C, H, W)`` by
            default.
        b: Second attribution map with the same shape and layout as ``a``.
        channel_axis: Channel dimension for 3-D inputs. ``None`` selects axis
            0, the library's channel-first default. It must remain ``None``
            for 2-D inputs.

    Returns:
        SSIM value in [-1, 1]. Higher = more similar.

    Raises:
        ImportError: If scikit-image is not installed.
        TypeError: If ``channel_axis`` is not an integer or ``None``.
        ValueError: If inputs are not 2-D or 3-D image maps, or if
            ``channel_axis`` is invalid for their rank.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        raise ImportError(
            "scikit-image is required for SSIM similarity. "
            "Install or repair Explainiverse with: pip install explainiverse"
        )

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"SSIM attribution shapes must match, got {a.shape} and {b.shape}.")
    if a.ndim not in (2, 3):
        raise ValueError(
            f"SSIM requires 2-D spatial maps or 3-D image maps, got shapes "
            f"{a.shape} and {b.shape}. Use 'spearman' or 'pearson' for 1-D "
            f"(tabular) data."
        )
    if a.ndim == 2:
        if channel_axis is not None:
            raise ValueError("channel_axis must be None for 2-D SSIM maps.")
        resolved_channel_axis = None
    else:
        if channel_axis is None:
            resolved_channel_axis = 0
        else:
            if isinstance(channel_axis, bool) or not isinstance(channel_axis, (int, np.integer)):
                raise TypeError("channel_axis must be an integer axis or None.")
            resolved_channel_axis = int(channel_axis)
            if resolved_channel_axis < -3 or resolved_channel_axis >= 3:
                raise ValueError("channel_axis must identify an axis of the 3-D SSIM maps.")
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError("SSIM attribution maps must be real-valued.")
    try:
        a = a.astype(np.float64, copy=False)
        b = b.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("SSIM attribution maps must be numeric.") from exc
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("SSIM attribution maps must contain only finite values.")

    spatial_axes = (
        tuple(range(a.ndim))
        if resolved_channel_axis is None
        else tuple(axis for axis in range(a.ndim) if axis != resolved_channel_axis % a.ndim)
    )
    minimum_spatial_extent = min(a.shape[axis] for axis in spatial_axes)
    if minimum_spatial_extent < 3:
        raise ValueError(
            "SSIM requires every spatial dimension to contain at least 3 values; "
            f"got spatial shape {tuple(a.shape[axis] for axis in spatial_axes)}."
        )
    # Own the backend contract instead of relying on scikit-image's changing
    # default. Seven is its historical default; smaller maps use the largest
    # valid odd window down to three.
    win_size = min(7, minimum_spatial_extent)
    if win_size % 2 == 0:
        win_size -= 1

    # SSIM is invariant to a shared positive scaling when data_range is scaled
    # with the maps. Normalize first so max-min cannot overflow for finite maps
    # spanning values near both ends of the float range.
    value_scale = float(max(np.max(np.abs(a)), np.max(np.abs(b))))
    if value_scale == 0.0:
        return 1.0
    scaled_a = a / value_scale
    scaled_b = b / value_scale
    data_range = float(
        max(np.max(scaled_a), np.max(scaled_b)) - min(np.min(scaled_a), np.min(scaled_b))
    )
    if data_range == 0.0:
        # A zero range over the union means both finite maps contain the same
        # single value. Different constant maps have a nonzero union range.
        return 1.0

    score = float(
        structural_similarity(
            scaled_a,
            scaled_b,
            data_range=data_range,
            channel_axis=resolved_channel_axis,
            win_size=win_size,
        )
    )
    # SSIM is strictly below one for unequal maps, but floating-point
    # round-off can collapse a value extremely close to one onto exactly one.
    if score == 1.0 and not np.array_equal(a, b):
        return float(np.nextafter(1.0, -np.inf))
    return score


def _mse_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Negative Mean Squared Error between two attribution vectors.

    Returns -MSE so that higher values = more similar (consistent with
    other similarity measures). Range: (-∞, 0]. A value of 0.0 means
    the attributions are identical.

    Args:
        a: First attribution vector (1-D, flattened).
        b: Second attribution vector (1-D, flattened).

    Returns:
        Negative MSE.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        differences = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if not np.all(np.isfinite(differences)):
        raise FloatingPointError("MSE residual is not representable")
    return -_stable_mean_square(differences)


# Registry of built-in similarity functions
_SIMILARITY_REGISTRY: Dict[str, SimilarityFunc] = {
    "spearman": _spearman_similarity,
    "pearson": _pearson_similarity,
    "cosine": _cosine_similarity,
    "ssim": _ssim_similarity,
    "mse": _mse_similarity,
}


def _resolve_similarity_func(
    similarity_func: Union[str, SimilarityFunc],
) -> SimilarityFunc:
    """
    Resolve a similarity function from a string key or callable.

    Args:
        similarity_func: Either a string key from the registry
            ("spearman", "pearson", "cosine", "ssim", "mse") or a
            callable f(a: np.ndarray, b: np.ndarray) -> float.

    Returns:
        A callable similarity function.

    Raises:
        ValueError: If string key is not recognised.
        TypeError: If argument is neither a string nor callable.
    """
    if isinstance(similarity_func, str):
        key = similarity_func.lower().strip()
        if key not in _SIMILARITY_REGISTRY:
            raise ValueError(
                f"Unknown similarity function '{similarity_func}'. "
                f"Available: {sorted(_SIMILARITY_REGISTRY.keys())}. "
                f"Or pass a callable f(a, b) -> float."
            )
        return _SIMILARITY_REGISTRY[key]
    if callable(similarity_func):
        return similarity_func
    raise TypeError(
        f"similarity_func must be a string or callable, " f"got {type(similarity_func).__name__}."
    )


def _compute_similarity(
    attr_a: np.ndarray,
    attr_b: np.ndarray,
    similarity_func: Union[str, SimilarityFunc],
) -> float:
    """
    Compute similarity between two attribution arrays.

    Handles flattening for non-SSIM measures (so that image attributions
    work with correlation-based measures) and shape validation.

    Args:
        attr_a: First attribution array (any shape).
        attr_b: Second attribution array (same shape as attr_a).
        similarity_func: Similarity measure (string key or callable).

    Returns:
        Similarity score (interpretation depends on the measure).

    Raises:
        ValueError: If shapes don't match.
    """
    if attr_a.shape != attr_b.shape:
        raise ValueError(
            f"Attribution shapes must match: got {attr_a.shape} " f"and {attr_b.shape}."
        )

    func = _resolve_similarity_func(similarity_func)

    # SSIM needs the original spatial structure; everything else uses flat
    func_name = similarity_func if isinstance(similarity_func, str) else ""
    if func_name.lower() == "ssim":
        return func(attr_a, attr_b)
    else:
        return func(attr_a.ravel(), attr_b.ravel())


# =============================================================================
# Attribution Extraction
# =============================================================================


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


def _extract_attribution_array(
    attributions: Union[np.ndarray, "Explanation"],
) -> np.ndarray:
    """
    Extract a numpy attribution array from various input types.

    Accepts:
        - numpy array directly (any shape — preserves spatial structure)
        - Explanation object (extracts feature_attributions values as 1-D)

    Args:
        attributions: Attribution values or Explanation object.

    Returns:
        numpy array of float64 attribution values.

    Raises:
        TypeError: If input is not a supported type.
        ValueError: If attributions are empty.
    """
    if isinstance(attributions, Explanation):
        attr_dict = attributions.explanation_data.get("feature_attributions", {})
        if not isinstance(attr_dict, Mapping) or not attr_dict:
            raise ValueError("No feature attributions found in Explanation.")
        feature_names = getattr(attributions, "feature_names", None)
        values = _ordered_explanation_attribution_values(attr_dict, feature_names)
        try:
            result = np.array(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Explanation feature attributions must be numeric.") from exc

    elif isinstance(attributions, np.ndarray):
        if np.iscomplexobj(attributions):
            raise TypeError("Attributions must contain real, not complex, values.")
        result = attributions.astype(np.float64)

    else:
        raise TypeError(f"Expected np.ndarray or Explanation, got {type(attributions).__name__}")

    if result.size == 0:
        raise ValueError("Attributions must not be empty.")
    if not np.all(np.isfinite(result)):
        raise ValueError("Attributions must contain only finite values.")
    return result


# =============================================================================
# PyTorch Model Helpers
# =============================================================================


def _validate_torch_available() -> None:
    """Raise ImportError if PyTorch is not installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch is required for high-level randomisation metrics. "
            "Install it with: pip install explainiverse[torch]"
        )


def _get_named_layers(
    model,
    layer_names: Optional[List[str]] = None,
) -> List[Tuple[str, "torch.nn.Module"]]:
    """
    Extract layers with learnable parameters from a PyTorch model.

    If layer_names is provided, only those layers are returned (in the
    given order). Otherwise, all layers with at least one parameter
    requiring grad are returned in module order.

    Args:
        model: PyTorch nn.Module.
        layer_names: Optional list of layer names to select. If None,
            auto-detect all layers with learnable parameters.

    Returns:
        List of (name, module) tuples.

    Raises:
        ValueError: If a requested layer_name is not found in the model.
    """
    if layer_names is not None:
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("layer_names must not contain duplicates.")
        # User-specified layers
        all_named = dict(model.named_modules())
        result = []
        for name in layer_names:
            if name not in all_named:
                raise ValueError(
                    f"Layer '{name}' not found in model. " f"Available: {sorted(all_named.keys())}"
                )
            module = all_named[name]
            has_direct_learnable_parameters = any(
                parameter.requires_grad for parameter in module.parameters(recurse=False)
            )
            if not has_direct_learnable_parameters:
                raise ValueError(f"Layer '{name}' has no direct learnable parameters to randomise.")
            if not callable(getattr(module, "reset_parameters", None)):
                raise ValueError(
                    f"Layer '{name}' does not expose reset_parameters(); "
                    "its initialization distribution cannot be verified."
                )
            result.append((name, module))
        return result

    # Auto-detect: leaf modules with at least one learnable parameter
    layers = []
    for name, module in model.named_modules():
        # Skip container modules (Sequential, ModuleList, etc.)
        if len(list(module.children())) > 0:
            continue
        # Check for learnable parameters
        has_params = any(p.requires_grad for p in module.parameters())
        if has_params:
            layers.append((name, module))

    return layers


def _validate_input_target_batch(x_batch: np.ndarray, y_batch: np.ndarray) -> None:
    """Validate the shared high-level randomisation-metric batch contract."""
    if not isinstance(x_batch, np.ndarray) or not isinstance(y_batch, np.ndarray):
        raise TypeError("x_batch and y_batch must be NumPy arrays.")
    if x_batch.ndim < 2:
        raise ValueError("x_batch must include a non-empty batch dimension and feature dimensions.")
    if y_batch.ndim != 1:
        raise ValueError("y_batch must be one-dimensional with shape (batch_size,).")
    if x_batch.shape[0] == 0:
        raise ValueError("x_batch and y_batch must not be empty.")
    if x_batch.shape[0] != y_batch.shape[0]:
        raise ValueError(
            f"x_batch and y_batch must have the same batch size; got "
            f"{x_batch.shape[0]} and {y_batch.shape[0]}."
        )
    if not np.all(np.isfinite(x_batch)):
        raise ValueError("x_batch must contain only finite values.")


def _randomise_layer_parameters(
    model,
    layer_name: str,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Reinitialise a single layer's parameters with random values.

    Calls the layer's own ``reset_parameters()`` implementation, which is the
    authoritative initialization contract for that module type. When ``rng``
    is supplied, the derived PyTorch seed is scoped with ``fork_rng`` so the
    caller's global CPU/CUDA RNG streams are restored afterwards.

    Operates in-place on the model.

    Args:
        model: PyTorch nn.Module (modified in-place).
        layer_name: Name of the layer to randomise (from named_modules).
        rng: Optional numpy random generator for reproducibility. If None,
            uses PyTorch's default random state.

    Raises:
        ValueError: If layer_name is not found in the model.
    """
    import torch

    all_named = dict(model.named_modules())
    if layer_name not in all_named:
        raise ValueError(
            f"Layer '{layer_name}' not found in model. " f"Available: {sorted(all_named.keys())}"
        )

    module = all_named[layer_name]

    direct_parameters = [
        parameter for parameter in module.parameters(recurse=False) if parameter.requires_grad
    ]
    if not direct_parameters:
        raise ValueError(f"Layer '{layer_name}' has no direct learnable parameters to randomise.")

    reset_parameters = getattr(module, "reset_parameters", None)
    if not callable(reset_parameters):
        raise ValueError(
            f"Layer '{layer_name}' does not expose reset_parameters(); "
            "its initialization distribution cannot be verified."
        )

    if rng is None:
        reset_parameters()
        return

    seed = int(rng.integers(0, 2**31))
    # torch.manual_seed seeds the CPU generator and *every* CUDA generator,
    # including CUDA devices unrelated to the randomized layer. Asking
    # fork_rng to snapshot all visible devices is therefore required even for
    # a CPU-resident module on a CUDA host.
    # ``fork_rng`` snapshots process-global default generators. Overlapping
    # forks in different threads can otherwise restore one another's seeded
    # intermediate state. Reuse the gradient subsystem's global lock domain so
    # every Explainiverse Torch RNG snapshot is serialized in one order.
    from explainiverse.explainers.gradient._model_state import adapter_model_operation_lock

    with adapter_model_operation_lock(model):
        with torch.random.fork_rng(devices=None, enabled=True):
            torch.manual_seed(seed)
            reset_parameters()


def _discrete_entropy(attributions: np.ndarray, n_bins: int = 100) -> float:
    """
    Compute histogram-based discrete Shannon entropy of attribution values.

    Used by Efficient MPRT (Hedström et al., 2023) to measure explanation
    complexity. Higher entropy = more uniform (complex/noisy) explanation.
    Lower entropy = more concentrated (simple/sparse) explanation.

    Attribution values (including their signs) are binned into ``n_bins``
    slots. The empirical frequency in each occupied bin is then used as the
    probability in Shannon's entropy. This is Equation 4 of Efficient MPRT;
    it is not entropy over normalised absolute feature magnitudes.

    Args:
        attributions: Attribution array (any shape, will be flattened).

    Returns:
        Shannon entropy in nats. Constant arrays have entropy 0.
    """
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool):
        raise TypeError("n_bins must be an integer")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    flat = np.asarray(attributions, dtype=np.float64).ravel()
    if flat.size == 0:
        raise ValueError("attributions must not be empty")
    if not np.all(np.isfinite(flat)):
        raise ValueError("attributions must contain only finite values")
    scale = float(np.max(np.abs(flat)))
    histogram, _ = np.histogram(flat if scale == 0.0 else flat / scale, bins=int(n_bins))
    occupied = histogram[histogram > 0].astype(np.float64)
    probabilities = occupied / occupied.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _add_noise_to_input(
    x: np.ndarray,
    noise_magnitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add Gaussian noise to an input for Smooth MPRT denoising.

    The noise standard deviation is scaled relative to the input's range:
        std = noise_magnitude * (x.max() - x.min())

    This follows Hedström et al. (2023), Section 3.1.

    Args:
        x: Input array (any shape).
        noise_magnitude: Fraction of input range used as noise std.
        rng: NumPy random generator.

    Returns:
        Noisy copy of x (same shape).
    """
    maximum = float(np.max(x))
    minimum = float(np.min(x))
    if not np.isfinite(maximum) or not np.isfinite(minimum):
        raise ValueError("x must contain only finite values")

    if noise_magnitude == 0.0:
        std = 0.0
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            data_range = maximum - minimum
            std = noise_magnitude * data_range
        if not np.isfinite(std):
            # A finite range can overflow before a small noise fraction is
            # applied. Distribute the multiplication so representable
            # standard deviations remain available (for example,
            # [-1e308, 1e308] at magnitude 1e-308 has std approximately 2).
            with np.errstate(over="ignore", invalid="ignore"):
                std = noise_magnitude * maximum - noise_magnitude * minimum
    if not np.isfinite(std):
        raise FloatingPointError("Smooth MPRT noise standard deviation is not representable")
    noise = rng.normal(0.0, std, size=x.shape)
    noisy = np.asarray(x) + noise
    if not np.all(np.isfinite(noisy)):
        raise FloatingPointError("Smooth MPRT produced non-finite perturbed inputs")
    return noisy


# =============================================================================
# MPRT — Model Parameter Randomisation Test (Adebayo et al., 2018)
# =============================================================================


def compute_mprt_score(
    original_attributions: Union[np.ndarray, "Explanation"],
    randomised_attributions_list: List[Union[np.ndarray, "Explanation"]],
    similarity_func: Union[str, SimilarityFunc] = "spearman",
    layer_names: Optional[List[str]] = None,
) -> MPRTResult:
    """
    Compute MPRT score from pre-computed attributions (low-level API).

    Compares original attributions against a list of attributions computed
    after successive layer randomisations. Each entry in
    ``randomised_attributions_list`` corresponds to one randomisation step
    (e.g., after randomising layer 1, then layers 1+2, etc.).

    The returned similarities describe sensitivity to the configured layer
    randomisations. A monotone trend is not guaranteed, and this diagnostic
    alone does not establish explanation quality.

    **Interpretation:**
        - Lower similarity means greater change under the configured intervention.
        - No universal quality threshold or required monotone trend is defined.

    Args:
        original_attributions: Attribution array from the original
            (fully trained) model. Shape: any (will be flattened for
            non-SSIM similarity measures).
        randomised_attributions_list: List of attribution arrays, one per
            randomisation step. Length = number of layers randomised.
        similarity_func: Similarity measure. One of "spearman"
            (default), "pearson", "cosine", "ssim", "mse", or a
            callable f(a, b) -> float. Three-dimensional SSIM maps use the
            channel-first ``(C, H, W)`` layout.
        layer_names: Optional list of layer names for labelling. If
            provided, must have same length as randomised_attributions_list.
            Used only for the returned dict keys.

    Returns:
        Dict with:
            - "layer_scores": List of similarity scores, one per layer.
            - "layer_names": List of layer name strings (if provided) or
              ["layer_0", "layer_1", ...].
            - "mean_score": Mean of layer_scores.

    Raises:
        ValueError: If randomised_attributions_list is empty or if
            layer_names length doesn't match.

    Example:
        >>> import numpy as np
        >>> original = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        >>> # After randomising top layer (small change)
        >>> rand_1 = np.array([0.85, 0.15, 0.45, 0.35, 0.18])
        >>> # After randomising top 2 layers (bigger change)
        >>> rand_2 = np.array([0.3, 0.6, 0.1, 0.7, 0.5])
        >>> result = compute_mprt_score(original, [rand_1, rand_2])
        >>> result["mean_score"]  # Mean similarity under this intervention

    References:
        Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
        & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    """
    if not randomised_attributions_list:
        raise ValueError("randomised_attributions_list must not be empty.")

    if layer_names is not None and len(layer_names) != len(randomised_attributions_list):
        raise ValueError(
            f"layer_names length ({len(layer_names)}) must match "
            f"randomised_attributions_list length ({len(randomised_attributions_list)})."
        )

    original = _extract_attribution_array(original_attributions)

    scores = []
    for rand_attr in randomised_attributions_list:
        rand = _extract_attribution_array(rand_attr)
        score = _finite_scalar_similarity(original, rand, similarity_func)
        scores.append(score)

    names = layer_names if layer_names is not None else [f"layer_{i}" for i in range(len(scores))]

    return {
        "layer_scores": scores,
        "layer_names": names,
        "mean_score": _finite_mean(scores, "MPRT layer score"),
    }


def compute_mprt(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    similarity_func: Union[str, SimilarityFunc] = "spearman",
    order: str = "cascading",
    layer_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> MPRTResult:
    """
    Model Parameter Randomisation Test (Adebayo et al., 2018).

    Measures explanation change while progressively randomising model layers.
    The result is a sensitivity diagnostic under this intervention, not a
    standalone explanation-quality verdict.

    **Algorithm:**
        1. Compute explanation for the original (trained) model.
        2. For each layer (in the specified order):
           a. Randomise the layer's parameters.
           b. Compute explanation for the (partially) randomised model.
           c. Compute similarity between original and new explanation.
        3. Return per-layer similarity scores and their mean.

    **Randomisation orders:**
        - ``"cascading"`` (default): Top-down. Randomise layer L, then
          L and L-1, then L, L-1, and L-2, etc. This is the original
          approach from Adebayo et al. (2018). Each step builds on
          the previous randomisation.
        - ``"independent"``: Randomise each layer independently,
          restoring the original model between steps. Tests each
          layer's contribution in isolation.
        - ``"bottom_up"``: Bottom-up cascading. Same as cascading but
          starting from the input layer.

    **Interpretation:**
        - Lower similarity means greater change under the configured randomisation.
        - The sequence and mean are descriptive; no universal "better" threshold
          is defined by this API.

    Args:
        model: PyTorch nn.Module. Will be deep-copied internally;
            the original model is never modified.
        x_batch: Input data, shape (batch_size, ...). If batch_size > 1,
            scores are averaged across samples.
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable with signature:
            ``explain_func(model, x, y) -> np.ndarray``
            where x is a single input and y is its label.
            Must return an attribution array (any shape).
        similarity_func: Similarity measure. One of "spearman"
            (default), "pearson", "cosine", "ssim", "mse", or a
            callable f(a, b) -> float. Three-dimensional SSIM maps use the
            channel-first ``(C, H, W)`` layout.
        order: Randomisation order. One of "cascading" (default),
            "independent", or "bottom_up".
        layer_names: Optional list of specific layer names to randomise.
            If None, auto-detects all layers with learnable parameters.
        seed: Random seed for reproducibility.

    Returns:
        Dict with:
            - "layer_scores": List of similarity scores per layer.
            - "layer_names": List of layer name strings.
            - "mean_score": Mean of layer_scores.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If order is not recognised or model has no layers.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> import numpy as np
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3)
        ... )
        >>> x = np.random.randn(1, 10).astype(np.float32)
        >>> y = np.array([0])
        >>> def explain_fn(model, x, y):
        ...     # Simple gradient-based explanation
        ...     x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        ...     out = model(x_t)
        ...     out[0, y].backward()
        ...     return x_t.grad.detach().numpy()
        >>> result = compute_mprt(model, x, y, explain_fn, seed=42)
        >>> result["mean_score"]  # Mean similarity under this intervention

    References:
        Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
        & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    """
    _validate_torch_available()
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not callable(explain_func):
        raise TypeError("explain_func must be callable.")
    _validate_input_target_batch(x_batch, y_batch)

    valid_orders = {"cascading", "independent", "bottom_up"}
    if order not in valid_orders:
        raise ValueError(f"Unknown order '{order}'. Must be one of {sorted(valid_orders)}.")

    rng = np.random.default_rng(_validated_seed(seed))

    # Deep copy to avoid modifying original model
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    # Get layers
    layers = _get_named_layers(model_copy, layer_names=layer_names)
    if not layers:
        raise ValueError("Model has no layers with learnable parameters.")

    detected_names = [name for name, _ in layers]

    # Determine layer order for randomisation
    if order == "cascading":
        # Top-down: last layer first
        randomisation_order = list(reversed(detected_names))
    elif order == "bottom_up":
        # Bottom-up: first layer first
        randomisation_order = list(detected_names)
    else:
        # Independent: order doesn't matter (each layer isolated)
        randomisation_order = list(detected_names)

    # Compute original explanations (one per sample)
    original_model = copy.deepcopy(model)
    original_model.eval()

    batch_size = x_batch.shape[0]

    original_attrs_list = []
    for i in range(batch_size):
        x_single = x_batch[i : i + 1]
        y_single = y_batch[i]
        attr = explain_func(original_model, x_single.copy(), y_single)
        original_attrs_list.append(_extract_attribution_array(attr))

    # For each layer, randomise and compute explanations
    all_layer_scores = []  # shape: (num_layers,)
    all_layer_names = []

    for step_idx, layer_name in enumerate(randomisation_order):
        if order == "independent":
            # Reset to original model for each layer
            model_copy = copy.deepcopy(model)
            model_copy.eval()

        # Randomise this layer (cascading/bottom_up accumulates)
        _randomise_layer_parameters(model_copy, layer_name, rng=rng)

        # Compute explanations for randomised model
        sample_scores = []
        for i in range(batch_size):
            x_single = x_batch[i : i + 1]
            y_single = y_batch[i]
            rand_attr = explain_func(model_copy, x_single.copy(), y_single)
            rand_attr = _extract_attribution_array(rand_attr)
            score = _finite_scalar_similarity(original_attrs_list[i], rand_attr, similarity_func)
            sample_scores.append(score)

        # Average across batch
        mean_score = _finite_mean(sample_scores, "MPRT sample score")
        all_layer_scores.append(mean_score)
        all_layer_names.append(layer_name)

    return {
        "layer_scores": all_layer_scores,
        "layer_names": all_layer_names,
        "mean_score": _finite_mean(all_layer_scores, "MPRT layer score"),
    }


def compute_batch_mprt(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    similarity_func: Union[str, SimilarityFunc] = "spearman",
    order: str = "cascading",
    layer_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[MPRTResult]:
    """
    Compute MPRT for each sample in a batch individually.

    Returns a list of MPRT result dicts, one per sample. This is useful
    when per-sample analysis is needed (e.g., to see which samples have
    explanations that are more/less sensitive to model randomisation).

    For a single aggregated score across the batch, use ``compute_mprt``
    which averages internally.

    Args:
        model: PyTorch nn.Module.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        similarity_func: Similarity measure (string or callable).
        order: Randomisation order ("cascading", "independent", "bottom_up").
        layer_names: Optional list of layer names to randomise.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, one per sample, each containing:
            - "layer_scores": Per-layer similarity scores.
            - "layer_names": Layer name strings.
            - "mean_score": Mean similarity across layers.

    References:
        Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
        & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    """
    _validate_input_target_batch(x_batch, y_batch)
    batch_size = x_batch.shape[0]
    results = []
    for i in range(batch_size):
        result = compute_mprt(
            model=model,
            x_batch=x_batch[i : i + 1],
            y_batch=y_batch[i : i + 1],
            explain_func=explain_func,
            similarity_func=similarity_func,
            order=order,
            layer_names=layer_names,
            seed=seed,
        )
        results.append(result)
    return results


# =============================================================================
# Random Logit Test (Sixt et al., 2020)
# =============================================================================


def _prepare_randomisation_model(model, argument_name: str):
    """Validate and copy a PyTorch model without changing caller state."""
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{argument_name} must be a torch.nn.Module.")
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    return model_copy


def _model_input_tensor(model, x_batch: np.ndarray):
    """Create a validation input on a model's device and floating dtype."""
    import torch

    reference_tensor = next(model.parameters(), None)
    if reference_tensor is None:
        reference_tensor = next(model.buffers(), None)

    # ``torch.as_tensor`` may alias a NumPy input. Validation executes
    # arbitrary caller-supplied modules, including legitimate in-place models,
    # so isolate the tensor before any forward pass can mutate caller state.
    x_tensor = torch.as_tensor(x_batch).detach().clone()
    if reference_tensor is not None:
        x_tensor = x_tensor.to(device=reference_tensor.device)
        if reference_tensor.is_floating_point():
            x_tensor = x_tensor.to(dtype=reference_tensor.dtype)
    return x_tensor


def _classification_output_width(model, x_batch: np.ndarray, argument_name: str) -> int:
    """Return the number of scalar model outputs per sample.

    Only tensors shaped ``(N, C)`` or ``(N,)`` are unambiguous for the
    output-index target contract used by these high-level metrics.
    """
    import torch

    x_tensor = _model_input_tensor(model, x_batch[:1])
    with torch.no_grad():
        output = model(x_tensor)

    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"{argument_name} must return a torch.Tensor, got " f"{type(output).__name__}."
        )
    if output.ndim == 1 and output.shape[0] == 1:
        width = 1
    elif output.ndim == 2 and output.shape[0] == 1:
        width = int(output.shape[1])
    else:
        raise ValueError(
            f"{argument_name} must return shape (batch_size, num_outputs) "
            f"or (batch_size,) for one output; got {tuple(output.shape)}."
        )
    if width < 1:
        raise ValueError(f"{argument_name} returned no output values per sample.")
    if not bool(torch.isfinite(output).all()):
        raise ValueError(f"{argument_name} output must contain only finite values.")
    return width


def _validate_output_index_targets(
    y_batch: np.ndarray,
    num_outputs: int,
    *,
    argument_name: str = "y_batch",
) -> np.ndarray:
    """Validate integer output indices and return a platform integer array."""
    if y_batch.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{argument_name} must contain integer output indices.")
    targets: np.ndarray = y_batch.astype(np.int64, copy=False)
    if np.any(targets < 0) or np.any(targets >= num_outputs):
        raise ValueError(
            f"{argument_name} values must be in [0, {num_outputs - 1}] for "
            f"the model's {num_outputs} outputs."
        )
    return targets


def _finite_scalar_similarity(
    attr_a: np.ndarray,
    attr_b: np.ndarray,
    similarity_func: Union[str, SimilarityFunc],
) -> float:
    """Compute a similarity and reject non-scalar or non-finite results."""
    score = _compute_similarity(attr_a, attr_b, similarity_func)
    score_array = np.asarray(score)
    if score_array.ndim != 0:
        raise ValueError("similarity_func must return one scalar score per sample.")
    value = float(score_array)
    if not np.isfinite(value):
        raise ValueError("similarity_func returned a non-finite score.")
    return value


def _validated_seed(seed: Optional[int]) -> Optional[int]:
    """Validate the shared non-negative integer seed contract."""
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be a non-negative integer or None.")
        if int(seed) < 0:
            raise ValueError("seed must be a non-negative integer or None.")
        seed = int(seed)
    return seed


def _random_logit_rng(seed: Optional[int]) -> np.random.Generator:
    """Create the metric-local generator under an explicit seed contract."""
    return np.random.default_rng(_validated_seed(seed))


def compute_random_logit_score(
    attr_true_class: Union[np.ndarray, "Explanation"],
    attr_random_class: Union[np.ndarray, "Explanation"],
    similarity_func: Union[str, SimilarityFunc] = "spearman",
) -> float:
    """
    Compute Random Logit score from pre-computed attributions (low-level API).

    Compares an explanation for a reference output against an explanation for
    a different output. Sixt et al. used the ground-truth output as the
    reference and SSIM as the comparison. This function leaves the reference
    implicit in the supplied attributions and makes similarity configurable;
    the default is raw, signed Spearman correlation, not the paper's exact
    SSIM configuration.

    **Interpretation:**
        - Lower similarity indicates greater output sensitivity under the
          selected similarity function.
        - High similarity indicates weak output sensitivity. This necessary
          sensitivity check does not by itself establish faithfulness.

    Args:
        attr_true_class: Attribution array for the supplied reference output.
        attr_random_class: Attribution array for a randomly chosen
            different class.
        similarity_func: Similarity measure. One of "spearman"
            (default), "pearson", "cosine", "ssim", "mse", or a
            callable f(a, b) -> float. Three-dimensional SSIM maps use the
            channel-first ``(C, H, W)`` layout.

    Returns:
        Finite scalar similarity score.

    Example:
        >>> import numpy as np
        >>> attr_true = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        >>> attr_rand = np.array([0.2, 0.8, 0.1, 0.6, 0.3])
        >>> score = compute_random_logit_score(attr_true, attr_rand)

    References:
        Sixt, L., Granz, M., & Landgraf, T. (2020). When Explanations Lie:
        Why Many Modified BP Attributions Fail. ICML.
    """
    a = _extract_attribution_array(attr_true_class)
    b = _extract_attribution_array(attr_random_class)
    return _finite_scalar_similarity(a, b, similarity_func)


def compute_random_logit(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    similarity_func: Union[str, SimilarityFunc] = "spearman",
    num_classes: Optional[int] = None,
    seed: Optional[int] = None,
) -> float:
    """
    Random Logit Test (Sixt et al., 2020).

    Compares the explanation for each *supplied* reference output index in
    ``y_batch`` against the explanation for a uniformly sampled different
    output index. Following Sixt et al., callers normally supply ground-truth
    class indices. Model predictions never replace those supplied targets.

    **Algorithm:**
        1. For each sample in the batch:
           a. Compute an explanation targeting the supplied output index y.
           b. Choose a random class y' ≠ y.
           c. Compute explanation targeting y'.
           d. Compute similarity between the two explanations.
        2. Return the mean similarity across all samples.

    Under this diagnostic, lower cross-output similarity is greater empirical
    output sensitivity. The API defines no universal pass threshold, and low
    similarity is not by itself evidence that an explanation is faithful.

    Sixt et al. compared image maps with SSIM. This implementation defaults to
    raw, signed Spearman correlation so it also has an explicit tabular-data
    contract; scores from that default are not numerically comparable to the
    paper's SSIM results.

    **Interpretation:**
        - Lower score indicates greater output sensitivity under the selected
          similarity function.
        - A score near 1 for correlation-like similarities indicates weak
          output sensitivity.

    Args:
        model: PyTorch nn.Module returning a tensor shaped ``(N, C)``. It is
            deep-copied and the caller's model is not modified.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Integer reference output indices, shape (batch_size,). The
            paper used ground-truth class indices; predictions are not inferred.
        explain_func: Callable with signature:
            ``explain_func(model, x, y) -> np.ndarray``
            where x is a single input (with batch dim) and y is a
            scalar target class label. Must return attributions.
        similarity_func: Similarity measure (string key or callable).
        num_classes: Number of separately attributable model outputs. If
            supplied, it must exactly match the model output width. If None,
            it is inferred from a ``(batch_size, num_outputs)`` tensor.
        seed: Non-negative integer seed for metric-local reproducibility, or
            None for nondeterministic sampling. The global NumPy RNG is not used.

    Returns:
        Arithmetic mean of the per-sample similarities. Use
        :func:`compute_batch_random_logit` for unaggregated scores.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If there are fewer than two explicit model outputs, a
            target is out of range, or the model output contract is ambiguous.

    Example:
        >>> import torch.nn as nn
        >>> import numpy as np
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        >>> x = np.random.randn(3, 10).astype(np.float32)
        >>> y = np.array([0, 2, 4])
        >>> def explain_fn(model, x, y):
        ...     import torch
        ...     x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        ...     out = model(x_t)
        ...     out[0, y].backward()
        ...     return x_t.grad.detach().numpy()
        >>> score = compute_random_logit(model, x, y, explain_fn, num_classes=5, seed=42)

    References:
        Sixt, L., Granz, M., & Landgraf, T. (2020). When Explanations Lie:
        Why Many Modified BP Attributions Fail. ICML.
    """
    _validate_torch_available()
    _validate_input_target_batch(x_batch, y_batch)
    if not callable(explain_func):
        raise TypeError("explain_func must be callable.")
    rng = _random_logit_rng(seed)
    model_eval = _prepare_randomisation_model(model, "model")
    inferred_num_classes = _classification_output_width(model_eval, x_batch, "model")
    if num_classes is not None:
        if isinstance(num_classes, (bool, np.bool_)) or not isinstance(
            num_classes, (int, np.integer)
        ):
            raise TypeError("num_classes must be an integer or None.")
        if int(num_classes) < 2:
            raise ValueError(f"num_classes must be >= 2 for Random Logit Test, got {num_classes}.")
        if int(num_classes) != inferred_num_classes:
            raise ValueError(
                f"num_classes={num_classes} does not match the model output "
                f"width {inferred_num_classes}."
            )
    num_classes = inferred_num_classes
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2 for Random Logit Test, got {num_classes}.")
    targets = _validate_output_index_targets(y_batch, num_classes)

    batch_size = x_batch.shape[0]
    scores = []

    for i in range(batch_size):
        x_single = x_batch[i : i + 1]
        y_true = int(targets[i])

        # Explanation for true class
        attr_true = explain_func(model_eval, x_single.copy(), y_true)
        attr_true = _extract_attribution_array(attr_true)

        # Sample a random different class
        # Uniformly draw one of C - 1 indices and skip the reference index.
        y_random = int(rng.integers(0, num_classes - 1))
        if y_random >= y_true:
            y_random += 1

        # Explanation for random class
        attr_random = explain_func(model_eval, x_single.copy(), y_random)
        attr_random = _extract_attribution_array(attr_random)

        score = _finite_scalar_similarity(attr_true, attr_random, similarity_func)
        scores.append(score)

    return _finite_mean(scores, "randomisation score")


def compute_batch_random_logit(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    similarity_func: Union[str, SimilarityFunc] = "spearman",
    num_classes: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Compute Random Logit Test for each sample individually.

    Returns per-sample similarity scores (one float per sample).
    For a single aggregated score, use ``compute_random_logit``.

    Args:
        model: PyTorch nn.Module.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Integer reference output indices, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        similarity_func: Similarity measure (string or callable).
        num_classes: Number of explicit outputs. If supplied, it must match the
            model output width exactly; if None, the width is inferred.
        seed: Non-negative integer seed for metric-local reproducibility, or
            None for nondeterministic sampling.

    Returns:
        Per-sample similarities in input order. Their arithmetic mean equals
        :func:`compute_random_logit` for identical arguments.

    References:
        Sixt, L., Granz, M., & Landgraf, T. (2020). When Explanations Lie:
        Why Many Modified BP Attributions Fail. ICML.
    """
    _validate_torch_available()
    _validate_input_target_batch(x_batch, y_batch)
    if not callable(explain_func):
        raise TypeError("explain_func must be callable.")
    rng = _random_logit_rng(seed)
    model_eval = _prepare_randomisation_model(model, "model")
    inferred_num_classes = _classification_output_width(model_eval, x_batch, "model")
    if num_classes is not None:
        if isinstance(num_classes, (bool, np.bool_)) or not isinstance(
            num_classes, (int, np.integer)
        ):
            raise TypeError("num_classes must be an integer or None.")
        if int(num_classes) < 2:
            raise ValueError(f"num_classes must be >= 2 for Random Logit Test, got {num_classes}.")
        if int(num_classes) != inferred_num_classes:
            raise ValueError(
                f"num_classes={num_classes} does not match the model output "
                f"width {inferred_num_classes}."
            )
    num_classes = inferred_num_classes
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2 for Random Logit Test, got {num_classes}.")
    targets = _validate_output_index_targets(y_batch, num_classes)

    batch_size = x_batch.shape[0]
    scores = []

    for i in range(batch_size):
        x_single = x_batch[i : i + 1]
        y_true = int(targets[i])

        attr_true = explain_func(model_eval, x_single.copy(), y_true)
        attr_true = _extract_attribution_array(attr_true)

        y_random = int(rng.integers(0, num_classes - 1))
        if y_random >= y_true:
            y_random += 1

        attr_random = explain_func(model_eval, x_single.copy(), y_random)
        attr_random = _extract_attribution_array(attr_random)

        score = _finite_scalar_similarity(attr_true, attr_random, similarity_func)
        scores.append(score)

    return scores


# =============================================================================
# Smooth MPRT (Hedström et al., 2023)
# =============================================================================


def compute_smooth_mprt(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    similarity_func: Union[str, SimilarityFunc] = "spearman",
    order: str = "bottom_up",
    layer_names: Optional[List[str]] = None,
    noise_magnitude: float = 0.1,
    nr_samples: int = 50,
    seed: Optional[int] = None,
) -> SmoothMPRTResult:
    """
    Smooth MPRT (Hedström et al., 2023).

    A denoised variant of the Model Parameter Randomisation Test that
    reduces the impact of gradient shattering noise by averaging
    explanations over multiple noisy input samples before computing
    similarity.

    **Algorithm:**
        1. For each sample x_i, generate N-1 noisy copies and retain the
           unperturbed input as the final sample:
           x_i^(k) = x_i + ε, where ε ~ N(0, σ²), σ = noise_magnitude * range(x_i)
        2. Compute the "smooth" original explanation as the mean of
           explanations over all noisy copies.
        3. For each layer (in the specified order):
           a. Randomise the layer's parameters.
           b. Compute the smooth explanation for the randomised model using
              the exact same noisy inputs used for the original model.
           c. Compute similarity between smooth original and smooth
              randomised explanations.
        4. Return per-layer similarity scores and their mean.

    Smoothing changes the estimator used before comparison. It does not
    guarantee lower variance, reliability, or explanation quality for a given
    model and dataset.

    **Interpretation:**
        Lower similarity means greater change under the configured randomisation;
        it is not a standalone quality ranking.

    Args:
        model: PyTorch nn.Module. Deep-copied internally.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        similarity_func: Similarity measure (string or callable).
        order: Randomisation order. The paper-conformant default is
            bottom-up cascading ("bottom_up").
        layer_names: Optional list of layer names to randomise.
        noise_magnitude: Fraction of input range used as noise std.
            Default 0.1 (10% of input range).
        nr_samples: Number of smoothing samples per input. Default 50. More
            samples require more explanation evaluations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with:
            - "layer_scores": List of similarity scores per layer.
            - "layer_names": List of layer name strings.
            - "mean_score": Mean of layer_scores.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If order is not recognised, nr_samples < 1,
            or noise_magnitude < 0.

    Example:
        >>> import torch.nn as nn
        >>> import numpy as np
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
        >>> x = np.random.randn(2, 10).astype(np.float32)
        >>> y = np.array([0, 1])
        >>> def explain_fn(model, x, y):
        ...     import torch
        ...     x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        ...     out = model(x_t)
        ...     out[0, y].backward()
        ...     return x_t.grad.detach().numpy()
        >>> result = compute_smooth_mprt(
        ...     model, x, y, explain_fn, nr_samples=5, seed=42
        ... )

    References:
        Hedström, A., Weber, L., Lapuschkin, S., & Höhne, M. (2023).
        Sanity Checks Revisited: An Exploration to Repair the Model
        Parameter Randomisation Test. XAI in Action.
    """
    _validate_torch_available()
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not callable(explain_func):
        raise TypeError("explain_func must be callable.")
    _validate_input_target_batch(x_batch, y_batch)

    valid_orders = {"cascading", "independent", "bottom_up"}
    if order not in valid_orders:
        raise ValueError(f"Unknown order '{order}'. Must be one of {sorted(valid_orders)}.")
    if isinstance(nr_samples, (bool, np.bool_)) or not isinstance(nr_samples, (int, np.integer)):
        raise TypeError("nr_samples must be a positive integer.")
    nr_samples = int(nr_samples)
    if nr_samples < 1:
        raise ValueError(f"nr_samples must be >= 1, got {nr_samples}.")
    if isinstance(noise_magnitude, (bool, np.bool_)) or not isinstance(
        noise_magnitude, (int, float, np.integer, np.floating)
    ):
        raise TypeError("noise_magnitude must be a finite non-negative number.")
    noise_magnitude = float(noise_magnitude)
    if not np.isfinite(noise_magnitude) or noise_magnitude < 0:
        raise ValueError(f"noise_magnitude must be >= 0, got {noise_magnitude}.")

    seed = _validated_seed(seed)
    # Smoothing draws and parameter initialisations must not consume one
    # another's random stream. In particular, changing the batch size must not
    # change the randomised models evaluated for an existing prefix.
    smoothing_seed, model_seed = np.random.SeedSequence(seed).spawn(2)
    smoothing_rng = np.random.default_rng(smoothing_seed)
    model_rng = np.random.default_rng(model_seed)

    model_copy = copy.deepcopy(model)
    model_copy.eval()

    layers = _get_named_layers(model_copy, layer_names=layer_names)
    if not layers:
        raise ValueError("Model has no layers with learnable parameters.")

    detected_names = [name for name, _ in layers]

    if order == "cascading":
        randomisation_order = list(reversed(detected_names))
    elif order == "bottom_up":
        randomisation_order = list(detected_names)
    else:
        randomisation_order = list(detected_names)

    # Compute smooth original explanations
    original_model = copy.deepcopy(model)
    original_model.eval()
    batch_size = x_batch.shape[0]

    def _smooth_explain(mdl, smoothing_inputs, y_single) -> np.ndarray:
        """Average explanations over a fixed, paired set of inputs."""
        samples: list[np.ndarray] = []
        expected_shape: Optional[tuple[int, ...]] = None
        for x_noisy in smoothing_inputs:
            # A third-party explainer is not allowed to mutate the stored
            # perturbation and thereby unpair later model comparisons.
            attr = explain_func(mdl, x_noisy.copy(), y_single)
            attr = _extract_attribution_array(attr)
            if expected_shape is None:
                expected_shape = attr.shape
            elif attr.shape != expected_shape:
                raise ValueError("Smooth MPRT explanation shape changed across smoothing samples")
            samples.append(attr.copy())
        if not samples:  # Defensive guard; nr_samples is validated as positive above.
            raise RuntimeError("Smooth MPRT produced no attribution samples")
        mean = _stable_mean(np.stack(samples, axis=0), axis=0)
        if not np.all(np.isfinite(mean)):
            raise FloatingPointError("Smooth MPRT attribution mean is not representable")
        return mean

    paired_smoothing_inputs: list[list[np.ndarray]] = []
    for i in range(batch_size):
        x_single = x_batch[i : i + 1]
        sample_inputs = [
            _add_noise_to_input(x_single, noise_magnitude, smoothing_rng)
            for _ in range(nr_samples - 1)
        ]
        # Retaining one clean input matches Equation 2 / the reference
        # implementation and makes nr_samples=1 exactly ordinary MPRT.
        sample_inputs.append(np.asarray(x_single).copy())
        paired_smoothing_inputs.append(sample_inputs)

    smooth_original_attrs = []
    for i in range(batch_size):
        y_single = y_batch[i]
        smooth_attr = _smooth_explain(original_model, paired_smoothing_inputs[i], y_single)
        smooth_original_attrs.append(smooth_attr)

    # For each layer, randomise and compute smooth explanations
    all_layer_scores = []
    all_layer_names = []

    for layer_name in randomisation_order:
        if order == "independent":
            model_copy = copy.deepcopy(model)
            model_copy.eval()

        _randomise_layer_parameters(model_copy, layer_name, rng=model_rng)

        sample_scores = []
        for i in range(batch_size):
            y_single = y_batch[i]
            smooth_rand = _smooth_explain(model_copy, paired_smoothing_inputs[i], y_single)
            score = _finite_scalar_similarity(
                smooth_original_attrs[i], smooth_rand, similarity_func
            )
            sample_scores.append(score)

        all_layer_scores.append(_finite_mean(sample_scores, "Smooth MPRT sample score"))
        all_layer_names.append(layer_name)

    return {
        "layer_scores": all_layer_scores,
        "layer_names": all_layer_names,
        "mean_score": _finite_mean(all_layer_scores, "Smooth MPRT layer score"),
        "variant": "smooth_mprt_paired",
        "randomisation_order": order,
        "nr_samples": nr_samples,
        "noise_magnitude": noise_magnitude,
        "smoothing_inputs_paired": True,
        "rng_streams_split": True,
        "clean_input_included": True,
        "paper_default_order_used": order == "bottom_up",
        "paper_recommended_sample_count_met": nr_samples >= 50,
        "paper_conformant_defaults_used": (
            order == "bottom_up" and nr_samples == 50 and noise_magnitude == 0.1
        ),
        "claim_scope": "smooth_model_parameter_randomisation_diagnostic",
    }


def compute_batch_smooth_mprt(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    similarity_func: Union[str, SimilarityFunc] = "spearman",
    order: str = "bottom_up",
    layer_names: Optional[List[str]] = None,
    noise_magnitude: float = 0.1,
    nr_samples: int = 50,
    seed: Optional[int] = None,
) -> List[SmoothMPRTResult]:
    """
    Compute Smooth MPRT for each sample in a batch individually.

    Args:
        model: PyTorch nn.Module.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        similarity_func: Similarity measure (string or callable).
        order: Randomisation order.
        layer_names: Optional layer names.
        noise_magnitude: Noise level for smoothing.
        nr_samples: Number of noisy samples.
        seed: Random seed.

    Returns:
        List of dicts, one per sample.

    References:
        Hedström, A., Weber, L., Lapuschkin, S., & Höhne, M. (2023).
        Sanity Checks Revisited. XAI in Action.
    """
    _validate_input_target_batch(x_batch, y_batch)
    batch_size = x_batch.shape[0]
    results = []
    for i in range(batch_size):
        result = compute_smooth_mprt(
            model=model,
            x_batch=x_batch[i : i + 1],
            y_batch=y_batch[i : i + 1],
            explain_func=explain_func,
            similarity_func=similarity_func,
            order=order,
            layer_names=layer_names,
            noise_magnitude=noise_magnitude,
            nr_samples=nr_samples,
            seed=seed,
        )
        results.append(result)
    return results


# =============================================================================
# Efficient MPRT (Hedström et al., 2023)
# =============================================================================


def compute_efficient_mprt(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    seed: Optional[int] = None,
    n_bins: int = 100,
) -> float:
    """
    Efficient MPRT (Hedström et al., 2023).

    An entropy-based reinterpretation of MPRT that avoids biased
    similarity measures. Instead of computing per-layer similarity
    scores, Efficient MPRT compares the entropy (complexity) of the
    original explanation against the entropy of the explanation from
    a fully randomised model.

    **Algorithm:**
        1. Compute explanation for the original model.
        2. Compute discrete Shannon entropy of the original explanation.
        3. Fully randomise ALL model parameters.
        4. Compute explanation for the fully randomised model.
        5. Compute entropy of the randomised explanation.
        6. Return the relative complexity increase:
           score = (entropy_random - entropy_original) / entropy_original.

    The returned value measures the relative rise in histogram entropy after
    parameter randomisation. It does not prove that an explanation captured
    model structure or that low/high entropy is intrinsically good.

    **Interpretation:**
        - Positive values mean histogram entropy increased after randomisation.
        - Zero means no relative entropy change; negative values mean entropy fell.
        - The API defines no universal quality threshold.

    Args:
        model: PyTorch nn.Module. Deep-copied internally.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        seed: Random seed for reproducibility.
        n_bins: Number of histogram slots used by the paper's discrete
            entropy complexity function.

    Returns:
        Mean relative histogram-entropy increase.

    Raises:
        ImportError: If PyTorch is not installed.

    Example:
        >>> import torch.nn as nn
        >>> import numpy as np
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
        >>> x = np.random.randn(2, 10).astype(np.float32)
        >>> y = np.array([0, 1])
        >>> def explain_fn(model, x, y):
        ...     import torch
        ...     x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        ...     out = model(x_t)
        ...     out[0, y].backward()
        ...     return x_t.grad.detach().numpy()
        >>> score = compute_efficient_mprt(model, x, y, explain_fn, seed=42)

    References:
        Hedström, A., Weber, L., Lapuschkin, S., & Höhne, M. (2023).
        Sanity Checks Revisited: An Exploration to Repair the Model
        Parameter Randomisation Test. XAI in Action.
    """
    _validate_torch_available()
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not callable(explain_func):
        raise TypeError("explain_func must be callable.")
    _validate_input_target_batch(x_batch, y_batch)
    rng = np.random.default_rng(_validated_seed(seed))
    _discrete_entropy(np.array([0.0]), n_bins=n_bins)
    batch_size = x_batch.shape[0]

    # Original model explanations
    original_model = copy.deepcopy(model)
    original_model.eval()

    # Fully randomised model
    random_model = copy.deepcopy(model)
    random_model.eval()
    layers = _get_named_layers(random_model)
    if not layers:
        raise ValueError("Model has no layers with learnable parameters to randomise.")
    for layer_name, _ in layers:
        _randomise_layer_parameters(random_model, layer_name, rng=rng)

    scores = []
    for i in range(batch_size):
        x_single = x_batch[i : i + 1]
        y_single = y_batch[i]

        # Original explanation entropy
        attr_orig = explain_func(original_model, x_single.copy(), y_single)
        attr_orig = _extract_attribution_array(attr_orig)
        entropy_orig = _discrete_entropy(attr_orig, n_bins=n_bins)

        # Randomised explanation entropy
        attr_rand = explain_func(random_model, x_single.copy(), y_single)
        attr_rand = _extract_attribution_array(attr_rand)
        if attr_rand.shape != attr_orig.shape:
            raise ValueError("Original and randomised attributions must have the same shape")
        entropy_rand = _discrete_entropy(attr_rand, n_bins=n_bins)

        if entropy_orig <= np.finfo(float).eps:
            raise ValueError(
                "Efficient MPRT relative rise is undefined because the "
                "original explanation histogram entropy is zero"
            )
        score = (entropy_rand - entropy_orig) / entropy_orig

        scores.append(float(score))

    return _finite_mean(scores, "randomisation score")


def compute_batch_efficient_mprt(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    seed: Optional[int] = None,
    n_bins: int = 100,
) -> List[float]:
    """
    Compute Efficient MPRT for each sample individually.

    Args:
        model: PyTorch nn.Module.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        seed: Random seed.
        n_bins: Number of histogram slots for discrete entropy.

    Returns:
        List of per-sample relative histogram-entropy increases.

    References:
        Hedström, A., Weber, L., Lapuschkin, S., & Höhne, M. (2023).
        Sanity Checks Revisited. XAI in Action.
    """
    _validate_torch_available()
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not callable(explain_func):
        raise TypeError("explain_func must be callable.")
    _validate_input_target_batch(x_batch, y_batch)
    rng = np.random.default_rng(_validated_seed(seed))
    _discrete_entropy(np.array([0.0]), n_bins=n_bins)
    batch_size = x_batch.shape[0]

    original_model = copy.deepcopy(model)
    original_model.eval()

    random_model = copy.deepcopy(model)
    random_model.eval()
    layers = _get_named_layers(random_model)
    if not layers:
        raise ValueError("Model has no layers with learnable parameters to randomise.")
    for layer_name, _ in layers:
        _randomise_layer_parameters(random_model, layer_name, rng=rng)

    scores = []
    for i in range(batch_size):
        x_single = x_batch[i : i + 1]
        y_single = y_batch[i]

        attr_orig = explain_func(original_model, x_single.copy(), y_single)
        attr_orig = _extract_attribution_array(attr_orig)
        entropy_orig = _discrete_entropy(attr_orig, n_bins=n_bins)

        attr_rand = explain_func(random_model, x_single.copy(), y_single)
        attr_rand = _extract_attribution_array(attr_rand)
        if attr_rand.shape != attr_orig.shape:
            raise ValueError("Original and randomised attributions must have the same shape")
        entropy_rand = _discrete_entropy(attr_rand, n_bins=n_bins)

        if entropy_orig <= np.finfo(float).eps:
            raise ValueError(
                "Efficient MPRT relative rise is undefined because the "
                "original explanation histogram entropy is zero"
            )
        score = (entropy_rand - entropy_orig) / entropy_orig

        scores.append(float(score))

    return scores


# =============================================================================
# Data Randomisation Test (Adebayo et al., 2018)
# =============================================================================


def _model_architecture_signature(model) -> tuple:
    """Build a strict module/parameter/buffer shape signature."""
    return tuple(
        (
            name,
            type(module),
            tuple(
                (parameter_name, tuple(parameter.shape))
                for parameter_name, parameter in module.named_parameters(recurse=False)
            ),
            tuple(
                (buffer_name, tuple(buffer.shape))
                for buffer_name, buffer in module.named_buffers(recurse=False)
            ),
        )
        for name, module in model.named_modules()
    )


def _prepare_data_randomisation_call(
    model_trained,
    model_random_labels,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
):
    """Validate the paper's same-architecture, same-target comparison."""
    _validate_input_target_batch(x_batch, y_batch)
    if not callable(explain_func):
        raise TypeError("explain_func must be callable.")

    model_a = _prepare_randomisation_model(model_trained, "model_trained")
    model_b = _prepare_randomisation_model(model_random_labels, "model_random_labels")
    if _model_architecture_signature(model_a) != _model_architecture_signature(model_b):
        raise ValueError(
            "model_trained and model_random_labels must have the same module "
            "architecture and parameter/buffer shapes."
        )

    width_a = _classification_output_width(model_a, x_batch, "model_trained")
    width_b = _classification_output_width(model_b, x_batch, "model_random_labels")
    if width_a != width_b:
        raise ValueError(
            "model_trained and model_random_labels must expose the same number "
            f"of outputs; got {width_a} and {width_b}."
        )
    targets = _validate_output_index_targets(y_batch, width_a)
    return model_a, model_b, targets


def compute_data_randomisation_score(
    attr_trained: Union[np.ndarray, "Explanation"],
    attr_random_labels: Union[np.ndarray, "Explanation"],
    similarity_func: Union[str, SimilarityFunc] = "spearman",
) -> float:
    """
    Compute Data Randomisation score from pre-computed attributions (low-level API).

    Compares an explanation from a model trained on true labels against an
    explanation from the same architecture trained on randomly permuted
    labels. Adebayo et al. reported several comparisons; the default here is
    their raw, signed Spearman-rank comparison. Absolute-value preprocessing,
    SSIM, and HOG similarity are not applied implicitly.

    **Interpretation:**
        - Lower similarity indicates greater sensitivity to label
          randomisation under the selected comparison.
        - High similarity indicates weak sensitivity. Passing this necessary
          check does not by itself establish explanation faithfulness.

    Args:
        attr_trained: Attribution array from the model trained on
            true labels.
        attr_random_labels: Attribution array from the model trained
            on randomised labels.
        similarity_func: Similarity measure. One of "spearman"
            (default), "pearson", "cosine", "ssim", "mse", or a
            callable f(a, b) -> float. Three-dimensional SSIM maps use the
            channel-first ``(C, H, W)`` layout.

    Returns:
        Finite scalar similarity score.

    Example:
        >>> import numpy as np
        >>> attr_true = np.array([0.9, 0.1, 0.5, 0.3, 0.2])
        >>> attr_rand = np.array([0.4, 0.3, 0.2, 0.5, 0.4])
        >>> score = compute_data_randomisation_score(attr_true, attr_rand)

    References:
        Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
        & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    """
    a = _extract_attribution_array(attr_trained)
    b = _extract_attribution_array(attr_random_labels)
    return _finite_scalar_similarity(a, b, similarity_func)


def compute_data_randomisation(
    model_trained,
    model_random_labels,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    similarity_func: Union[str, SimilarityFunc] = "spearman",
) -> float:
    """
    Data Randomisation Test (Adebayo et al., 2018).

    Tests whether an explanation method is sensitive to the relationship
    between training data and labels. Compares explanations from a model
    trained on true labels against a model trained on randomised
    (shuffled) labels.

    Unlike MPRT, which perturbs model parameters, this test asks whether an
    explanation changes after the instance-label relationship in training is
    broken. It is a necessary sensitivity check, not a sufficient proof that
    an explanation captures the data-generating process.

    **Important:** The caller must provide both models. The function verifies
    that their module structures and parameter/buffer shapes match, but cannot
    verify their training histories. The caller is responsible for ensuring
    that ``model_random_labels`` was trained on one fixed permutation of all
    training labels; merely supplying a different random initialisation does
    not perform the data-randomisation test.

    **Algorithm:**
        1. For each sample in the batch:
           a. Compute explanation from model_trained.
           b. Compute explanation from model_random_labels.
           c. Compute similarity between the two explanations.
        2. Return the mean similarity across all samples.

    **Interpretation:**
        - Lower score indicates greater sensitivity to the training-label
          permutation under the selected similarity function.
        - A score near 1 for correlation-like similarities indicates weak
          sensitivity; it is not a standalone faithfulness verdict.

    Args:
        model_trained: PyTorch nn.Module trained on true labels. Deep-copied;
            the caller's model is not modified.
        model_random_labels: PyTorch nn.Module trained on randomised
            labels (same architecture, same data, shuffled labels).
            Not modified.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Integer output indices, shape (batch_size,), used unchanged
            for both explanations. A one-output model therefore accepts only
            index 0; no class-sign convention is guessed.
        explain_func: Callable with signature:
            ``explain_func(model, x, y) -> np.ndarray``
            where x is a single input (with batch dim) and y is a
            scalar target class label.
        similarity_func: Similarity measure (string key or callable).

    Returns:
        Arithmetic mean of per-sample similarities. Use
        :func:`compute_batch_data_randomisation` for unaggregated scores.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If batches, target indices, model architectures, or output
            contracts are incompatible.

    Example:
        >>> import torch.nn as nn
        >>> import numpy as np
        >>> # Architecture illustration only: before evaluating, train model_true
        >>> # normally and fit model_rand on one fixed label permutation.
        >>> # Fresh initialisations alone are not valid inputs.
        >>> model_true = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
        >>> model_rand = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
        >>> x = np.random.randn(5, 10).astype(np.float32)
        >>> y = np.array([0, 1, 2, 0, 1])
        >>> def explain_fn(model, x, y):
        ...     import torch
        ...     x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        ...     out = model(x_t)
        ...     out[0, y].backward()
        ...     return x_t.grad.detach().numpy()
        >>> # score = compute_data_randomisation(
        >>> #     model_true, model_rand, x, y, explain_fn
        >>> # )

    References:
        Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
        & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    """
    _validate_torch_available()
    model_a, model_b, targets = _prepare_data_randomisation_call(
        model_trained, model_random_labels, x_batch, y_batch, explain_func
    )

    batch_size = x_batch.shape[0]
    scores = []

    for i in range(batch_size):
        x_single = x_batch[i : i + 1]
        y_single = int(targets[i])

        attr_trained = explain_func(model_a, x_single.copy(), y_single)
        attr_trained = _extract_attribution_array(attr_trained)

        attr_random = explain_func(model_b, x_single.copy(), y_single)
        attr_random = _extract_attribution_array(attr_random)

        score = _finite_scalar_similarity(attr_trained, attr_random, similarity_func)
        scores.append(score)

    return _finite_mean(scores, "randomisation score")


def compute_batch_data_randomisation(
    model_trained,
    model_random_labels,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    similarity_func: Union[str, SimilarityFunc] = "spearman",
) -> List[float]:
    """
    Compute Data Randomisation Test for each sample individually.

    Returns per-sample similarity scores.
    For a single aggregated score, use ``compute_data_randomisation``.

    Args:
        model_trained: PyTorch nn.Module trained on true labels.
        model_random_labels: PyTorch nn.Module trained on randomised labels.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Integer output indices, shape (batch_size,), used unchanged
            for both explanations.
        explain_func: Callable(model, x, y) -> np.ndarray.
        similarity_func: Similarity measure (string or callable).

    Returns:
        Per-sample similarities in input order. Their arithmetic mean equals
        :func:`compute_data_randomisation` for identical arguments.

    References:
        Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
        & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    """
    _validate_torch_available()
    model_a, model_b, targets = _prepare_data_randomisation_call(
        model_trained, model_random_labels, x_batch, y_batch, explain_func
    )

    batch_size = x_batch.shape[0]
    scores = []

    for i in range(batch_size):
        x_single = x_batch[i : i + 1]
        y_single = int(targets[i])

        attr_trained = explain_func(model_a, x_single.copy(), y_single)
        attr_trained = _extract_attribution_array(attr_trained)

        attr_random = explain_func(model_b, x_single.copy(), y_single)
        attr_random = _extract_attribution_array(attr_random)

        score = _finite_scalar_similarity(attr_trained, attr_random, similarity_func)
        scores.append(score)

    return scores
