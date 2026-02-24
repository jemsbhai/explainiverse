# src/explainiverse/evaluation/randomisation.py
"""
Randomisation evaluation metrics for explanations (Phase 5).

Implements:
- Model Parameter Randomisation Test — MPRT (Adebayo et al., 2018)
- Random Logit Test (Sixt et al., 2020)
- Smooth MPRT (Hedström et al., 2023)
- Efficient MPRT (Hedström et al., 2023)
- Data Randomisation Test (Adebayo et al., 2018)

These metrics evaluate whether explanations are sensitive to the model's
learned parameters, predicted class, and training data-label relationship.
A faithful explanation method should produce significantly different
explanations when the model or data is randomised. Methods that produce
similar explanations regardless of randomisation are unreliable.

All metrics support both tabular (1-D) and image (2-D/3-D) data.
They require PyTorch models for the high-level API, but also provide
low-level score functions that work on pre-computed attribution arrays.

Similarity Functions:
    Built-in similarity measures are dispatched via string keys:
    - "spearman": Spearman rank correlation (scipy.stats.spearmanr)
    - "pearson": Pearson correlation (scipy.stats.pearsonr)
    - "cosine": Cosine similarity (1 - scipy.spatial.distance.cosine)
    - "ssim": Structural Similarity Index (skimage.metrics — optional dep)
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
import warnings

import numpy as np
from typing import Union, Callable, List, Dict, Optional, Tuple

from scipy import stats
from scipy.spatial.distance import cosine as scipy_cosine_distance

from explainiverse.core.explanation import Explanation


# =============================================================================
# Similarity Functions — Dispatcher and Built-ins
# =============================================================================

# Type alias for similarity functions: f(a, b) -> float
SimilarityFunc = Callable[[np.ndarray, np.ndarray], float]


def _spearman_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation between two attribution vectors.

    Returns the correlation coefficient (range [-1, 1]). If either
    vector is constant (zero variance), returns 0.0 with a warning
    suppressed (this is expected when a fully randomised model
    produces near-uniform attributions).

    Args:
        a: First attribution vector (1-D, flattened).
        b: Second attribution vector (1-D, flattened).

    Returns:
        Spearman correlation coefficient.
    """
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
        corr, _ = stats.spearmanr(a, b)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation between two attribution vectors.

    Returns the correlation coefficient (range [-1, 1]). Returns 0.0
    for constant inputs.

    Args:
        a: First attribution vector (1-D, flattened).
        b: Second attribution vector (1-D, flattened).

    Returns:
        Pearson correlation coefficient.
    """
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
        corr, _ = stats.pearsonr(a, b)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two attribution vectors.

    Computed as 1 - cosine_distance. Range [-1, 1] (typically [0, 1]
    for non-negative attributions). Returns 0.0 if either vector is
    all zeros.

    Args:
        a: First attribution vector (1-D, flattened).
        b: Second attribution vector (1-D, flattened).

    Returns:
        Cosine similarity.
    """
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return 0.0
    dist = scipy_cosine_distance(a, b)
    if np.isnan(dist):
        return 0.0
    return float(1.0 - dist)


def _ssim_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between two attribution maps.

    Requires scikit-image (optional dependency). Only meaningful for
    2-D or 3-D (image) data. For 1-D (tabular) data, this will reshape
    into a 2-D array if possible, but Spearman/Pearson is recommended.

    Uses data_range computed from the union of both arrays to ensure
    consistent scaling.

    Args:
        a: First attribution map (2-D or 3-D).
        b: Second attribution map (2-D or 3-D).

    Returns:
        SSIM value in [-1, 1]. Higher = more similar.

    Raises:
        ImportError: If scikit-image is not installed.
        ValueError: If inputs are not at least 2-D.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        raise ImportError(
            "scikit-image is required for SSIM similarity. "
            "Install it with: pip install explainiverse[image]"
        )

    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(
            f"SSIM requires at least 2-D arrays, got shapes "
            f"{a.shape} and {b.shape}. Use 'spearman' or 'pearson' "
            f"for 1-D (tabular) data."
        )

    data_range = max(
        a.max() - a.min(),
        b.max() - b.min(),
    )
    if data_range < 1e-12:
        # Both arrays are effectively constant
        return 1.0

    # For multi-channel images (e.g. 3-D with channel dim), set channel_axis
    channel_axis = None
    if a.ndim == 3:
        channel_axis = -1  # assume channel-last; common after flattening batch

    return float(structural_similarity(
        a, b,
        data_range=data_range,
        channel_axis=channel_axis,
    ))


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
    return -float(np.mean((a - b) ** 2))


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
        f"similarity_func must be a string or callable, "
        f"got {type(similarity_func).__name__}."
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
            f"Attribution shapes must match: got {attr_a.shape} "
            f"and {attr_b.shape}."
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
    import torch.nn as nn

    if layer_names is not None:
        # User-specified layers
        all_named = dict(model.named_modules())
        result = []
        for name in layer_names:
            if name not in all_named:
                raise ValueError(
                    f"Layer '{name}' not found in model. "
                    f"Available: {sorted(all_named.keys())}"
                )
            result.append((name, all_named[name]))
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


def _randomise_layer_parameters(
    model,
    layer_name: str,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Reinitialise a single layer's parameters with random values.

    Uses the same distribution as PyTorch's default initialisation
    for the layer type (Kaiming uniform for Linear/Conv layers,
    zeros for biases). This matches the approach in Adebayo et al. (2018)
    where weights are "reinitialized" to destroy learned representations.

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
    import torch.nn as nn

    all_named = dict(model.named_modules())
    if layer_name not in all_named:
        raise ValueError(
            f"Layer '{layer_name}' not found in model. "
            f"Available: {sorted(all_named.keys())}"
        )

    module = all_named[layer_name]

    # Set seed if rng provided
    if rng is not None:
        seed = int(rng.integers(0, 2**31))
        torch.manual_seed(seed)

    # Reinitialise using PyTorch's standard init
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            if fan_in > 0:
                bound = 1.0 / (fan_in ** 0.5)
                nn.init.uniform_(module.bias, -bound, bound)
            else:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if module.running_mean is not None:
            module.running_mean.zero_()
        if module.running_var is not None:
            module.running_var.fill_(1)
    elif isinstance(module, (nn.Embedding,)):
        nn.init.normal_(module.weight)
    else:
        # Generic fallback: reinitialise all parameters with normal(0, 0.01)
        for param in module.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)


def _discrete_entropy(attributions: np.ndarray) -> float:
    """
    Compute discrete Shannon entropy of normalised absolute attributions.

    Used by Efficient MPRT (Hedström et al., 2023) to measure explanation
    complexity. Higher entropy = more uniform (complex/noisy) explanation.
    Lower entropy = more concentrated (simple/sparse) explanation.

    The attributions are first converted to a probability distribution
    by taking absolute values and normalising to sum to 1.

    Args:
        attributions: Attribution array (any shape, will be flattened).

    Returns:
        Shannon entropy in nats. Returns 0.0 if all attributions are zero.
    """
    flat = np.abs(attributions.ravel()).astype(np.float64)
    total = flat.sum()
    if total < 1e-12:
        return 0.0
    p = flat / total
    # Filter out zeros to avoid log(0)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


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
    data_range = x.max() - x.min()
    if data_range < 1e-12:
        data_range = 1.0
    std = noise_magnitude * data_range
    noise = rng.normal(0.0, std, size=x.shape)
    return x + noise


# =============================================================================
# MPRT — Model Parameter Randomisation Test (Adebayo et al., 2018)
# =============================================================================

def compute_mprt_score(
    original_attributions: Union[np.ndarray, "Explanation"],
    randomised_attributions_list: List[Union[np.ndarray, "Explanation"]],
    similarity_func: Union[str, SimilarityFunc] = "spearman",
    layer_names: Optional[List[str]] = None,
) -> Dict[str, Union[List[float], float]]:
    """
    Compute MPRT score from pre-computed attributions (low-level API).

    Compares original attributions against a list of attributions computed
    after successive layer randomisations. Each entry in
    ``randomised_attributions_list`` corresponds to one randomisation step
    (e.g., after randomising layer 1, then layers 1+2, etc.).

    A faithful explanation method should show decreasing similarity as more
    layers are randomised (i.e., the model deviates further from the
    trained state). An explanation that remains similar regardless of
    randomisation is not sensitive to the model's learned parameters.

    **Interpretation:**
        - Lower mean similarity = better (explanation is model-sensitive)
        - Monotonically decreasing layer_scores = ideal

    Args:
        original_attributions: Attribution array from the original
            (fully trained) model. Shape: any (will be flattened for
            non-SSIM similarity measures).
        randomised_attributions_list: List of attribution arrays, one per
            randomisation step. Length = number of layers randomised.
        similarity_func: Similarity measure. One of "spearman"
            (default), "pearson", "cosine", "ssim", "mse", or a
            callable f(a, b) -> float.
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
        >>> result["mean_score"]  # Lower = better

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
        score = _compute_similarity(original, rand, similarity_func)
        scores.append(score)

    names = layer_names if layer_names is not None else [
        f"layer_{i}" for i in range(len(scores))
    ]

    return {
        "layer_scores": scores,
        "layer_names": names,
        "mean_score": float(np.mean(scores)),
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
) -> Dict[str, Union[List[float], float]]:
    """
    Model Parameter Randomisation Test (Adebayo et al., 2018).

    Tests whether an explanation method is sensitive to the model's learned
    parameters by progressively randomising model layers and measuring how
    the explanation changes. A faithful explanation should change
    significantly when the model's parameters are destroyed.

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
        - Lower mean_score = better (explanation depends on model)
        - Decreasing layer_scores = ideal behaviour
        - Flat/high scores = explanation ignores model parameters (bad)

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
            callable f(a, b) -> float.
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
        >>> result["mean_score"]  # Lower = better

    References:
        Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
        & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    """
    _validate_torch_available()
    import torch

    valid_orders = {"cascading", "independent", "bottom_up"}
    if order not in valid_orders:
        raise ValueError(
            f"Unknown order '{order}'. Must be one of {sorted(valid_orders)}."
        )

    rng = np.random.default_rng(seed)

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
        x_single = x_batch[i:i+1]
        y_single = y_batch[i]
        attr = explain_func(original_model, x_single, y_single)
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
            x_single = x_batch[i:i+1]
            y_single = y_batch[i]
            rand_attr = explain_func(model_copy, x_single, y_single)
            rand_attr = _extract_attribution_array(rand_attr)
            score = _compute_similarity(
                original_attrs_list[i], rand_attr, similarity_func
            )
            sample_scores.append(score)

        # Average across batch
        mean_score = float(np.mean(sample_scores))
        all_layer_scores.append(mean_score)
        all_layer_names.append(layer_name)

    return {
        "layer_scores": all_layer_scores,
        "layer_names": all_layer_names,
        "mean_score": float(np.mean(all_layer_scores)),
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
) -> List[Dict[str, Union[List[float], float]]]:
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
    batch_size = x_batch.shape[0]
    results = []
    for i in range(batch_size):
        result = compute_mprt(
            model=model,
            x_batch=x_batch[i:i+1],
            y_batch=y_batch[i:i+1],
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

def compute_random_logit_score(
    attr_true_class: Union[np.ndarray, "Explanation"],
    attr_random_class: Union[np.ndarray, "Explanation"],
    similarity_func: Union[str, SimilarityFunc] = "spearman",
) -> float:
    """
    Compute Random Logit score from pre-computed attributions (low-level API).

    Compares the explanation for the true predicted class against the
    explanation for a randomly chosen different class. A faithful,
    class-discriminative explanation method should produce different
    explanations for different target classes.

    **Interpretation:**
        - Lower similarity = better (explanation is class-sensitive)
        - High similarity = explanation ignores which class is targeted (bad)

    Args:
        attr_true_class: Attribution array for the true/predicted class.
        attr_random_class: Attribution array for a randomly chosen
            different class.
        similarity_func: Similarity measure. One of "spearman"
            (default), "pearson", "cosine", "ssim", "mse", or a
            callable f(a, b) -> float.

    Returns:
        Similarity score (float). Lower = better.

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
    return _compute_similarity(a, b, similarity_func)


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

    Tests whether an explanation method produces class-discriminative
    explanations by comparing the explanation for the true target class
    against the explanation for a randomly chosen different class.

    **Algorithm:**
        1. For each sample in the batch:
           a. Compute explanation targeting the true class y.
           b. Choose a random class y' ≠ y.
           c. Compute explanation targeting y'.
           d. Compute similarity between the two explanations.
        2. Return the mean similarity across all samples.

    A faithful explanation should differ significantly when targeting
    different classes. Methods that produce similar explanations regardless
    of the target class fail this test.

    **Interpretation:**
        - Lower score = better (explanations are class-discriminative)
        - Score ≈ 1.0 = explanation ignores target class (bad)

    Args:
        model: PyTorch nn.Module. Not modified.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: True target labels, shape (batch_size,). Used as the
            "true" class for explanation.
        explain_func: Callable with signature:
            ``explain_func(model, x, y) -> np.ndarray``
            where x is a single input (with batch dim) and y is a
            scalar target class label. Must return attributions.
        similarity_func: Similarity measure (string key or callable).
        num_classes: Total number of classes. Required to sample a
            random alternative class. If None, inferred from
            ``model(x_batch)`` output dimension.
        seed: Random seed for reproducibility.

    Returns:
        Mean similarity score (float). Lower = better.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If num_classes < 2.

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
    import torch

    rng = np.random.default_rng(seed)
    model_eval = copy.deepcopy(model)
    model_eval.eval()

    # Infer num_classes if not provided
    if num_classes is None:
        with torch.no_grad():
            x_t = torch.tensor(x_batch[:1], dtype=torch.float32)
            out = model_eval(x_t)
            num_classes = out.shape[-1]

    if num_classes < 2:
        raise ValueError(
            f"num_classes must be >= 2 for Random Logit Test, got {num_classes}."
        )

    batch_size = x_batch.shape[0]
    scores = []

    for i in range(batch_size):
        x_single = x_batch[i:i+1]
        y_true = int(y_batch[i])

        # Explanation for true class
        attr_true = explain_func(model_eval, x_single, y_true)
        attr_true = _extract_attribution_array(attr_true)

        # Sample a random different class
        candidates = [c for c in range(num_classes) if c != y_true]
        y_random = int(rng.choice(candidates))

        # Explanation for random class
        attr_random = explain_func(model_eval, x_single, y_random)
        attr_random = _extract_attribution_array(attr_random)

        score = _compute_similarity(attr_true, attr_random, similarity_func)
        scores.append(score)

    return float(np.mean(scores))


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
        y_batch: True target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        similarity_func: Similarity measure (string or callable).
        num_classes: Total number of classes. If None, auto-detected.
        seed: Random seed for reproducibility.

    Returns:
        List of per-sample similarity scores. Lower = better.

    References:
        Sixt, L., Granz, M., & Landgraf, T. (2020). When Explanations Lie:
        Why Many Modified BP Attributions Fail. ICML.
    """
    _validate_torch_available()
    import torch

    rng = np.random.default_rng(seed)
    model_eval = copy.deepcopy(model)
    model_eval.eval()

    if num_classes is None:
        with torch.no_grad():
            x_t = torch.tensor(x_batch[:1], dtype=torch.float32)
            out = model_eval(x_t)
            num_classes = out.shape[-1]

    if num_classes < 2:
        raise ValueError(
            f"num_classes must be >= 2 for Random Logit Test, got {num_classes}."
        )

    batch_size = x_batch.shape[0]
    scores = []

    for i in range(batch_size):
        x_single = x_batch[i:i+1]
        y_true = int(y_batch[i])

        attr_true = explain_func(model_eval, x_single, y_true)
        attr_true = _extract_attribution_array(attr_true)

        candidates = [c for c in range(num_classes) if c != y_true]
        y_random = int(rng.choice(candidates))

        attr_random = explain_func(model_eval, x_single, y_random)
        attr_random = _extract_attribution_array(attr_random)

        score = _compute_similarity(attr_true, attr_random, similarity_func)
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
    order: str = "cascading",
    layer_names: Optional[List[str]] = None,
    noise_magnitude: float = 0.1,
    nr_samples: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, Union[List[float], float]]:
    """
    Smooth MPRT (Hedström et al., 2023).

    A denoised variant of the Model Parameter Randomisation Test that
    reduces the impact of gradient shattering noise by averaging
    explanations over multiple noisy input samples before computing
    similarity.

    **Algorithm:**
        1. For each sample x_i, generate N noisy copies:
           x_i^(k) = x_i + ε, where ε ~ N(0, σ²), σ = noise_magnitude * range(x_i)
        2. Compute the "smooth" original explanation as the mean of
           explanations over all noisy copies.
        3. For each layer (in the specified order):
           a. Randomise the layer's parameters.
           b. Compute the smooth explanation for the randomised model
              (average over the same N noisy samples).
           c. Compute similarity between smooth original and smooth
              randomised explanations.
        4. Return per-layer similarity scores and their mean.

    This addresses a key weakness of standard MPRT: gradient-based
    explanations can be noisy ("gradient shattering"), causing high
    variance in similarity measurements. Smooth MPRT produces more
    stable and reliable scores.

    **Interpretation:**
        Same as MPRT: lower mean_score = better.

    Args:
        model: PyTorch nn.Module. Deep-copied internally.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        similarity_func: Similarity measure (string or callable).
        order: Randomisation order ("cascading", "independent", "bottom_up").
        layer_names: Optional list of layer names to randomise.
        noise_magnitude: Fraction of input range used as noise std.
            Default 0.1 (10% of input range). From Hedström et al. (2023).
        nr_samples: Number of noisy samples per input for smoothing.
            Default 10. Higher = smoother but slower.
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
    import torch

    valid_orders = {"cascading", "independent", "bottom_up"}
    if order not in valid_orders:
        raise ValueError(
            f"Unknown order '{order}'. Must be one of {sorted(valid_orders)}."
        )
    if nr_samples < 1:
        raise ValueError(f"nr_samples must be >= 1, got {nr_samples}.")
    if noise_magnitude < 0:
        raise ValueError(f"noise_magnitude must be >= 0, got {noise_magnitude}.")

    rng = np.random.default_rng(seed)

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

    def _smooth_explain(mdl, x_single, y_single, noise_rng):
        """Average explanation over nr_samples noisy copies of x."""
        accum = None
        for _ in range(nr_samples):
            x_noisy = _add_noise_to_input(x_single, noise_magnitude, noise_rng)
            attr = explain_func(mdl, x_noisy, y_single)
            attr = _extract_attribution_array(attr)
            if accum is None:
                accum = attr.copy()
            else:
                accum += attr
        return accum / nr_samples

    smooth_original_attrs = []
    for i in range(batch_size):
        x_single = x_batch[i:i+1]
        y_single = y_batch[i]
        smooth_attr = _smooth_explain(original_model, x_single, y_single, rng)
        smooth_original_attrs.append(smooth_attr)

    # For each layer, randomise and compute smooth explanations
    all_layer_scores = []
    all_layer_names = []

    for layer_name in randomisation_order:
        if order == "independent":
            model_copy = copy.deepcopy(model)
            model_copy.eval()

        _randomise_layer_parameters(model_copy, layer_name, rng=rng)

        sample_scores = []
        for i in range(batch_size):
            x_single = x_batch[i:i+1]
            y_single = y_batch[i]
            smooth_rand = _smooth_explain(model_copy, x_single, y_single, rng)
            score = _compute_similarity(
                smooth_original_attrs[i], smooth_rand, similarity_func
            )
            sample_scores.append(score)

        all_layer_scores.append(float(np.mean(sample_scores)))
        all_layer_names.append(layer_name)

    return {
        "layer_scores": all_layer_scores,
        "layer_names": all_layer_names,
        "mean_score": float(np.mean(all_layer_scores)),
    }


def compute_batch_smooth_mprt(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    similarity_func: Union[str, SimilarityFunc] = "spearman",
    order: str = "cascading",
    layer_names: Optional[List[str]] = None,
    noise_magnitude: float = 0.1,
    nr_samples: int = 10,
    seed: Optional[int] = None,
) -> List[Dict[str, Union[List[float], float]]]:
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
    batch_size = x_batch.shape[0]
    results = []
    for i in range(batch_size):
        result = compute_smooth_mprt(
            model=model,
            x_batch=x_batch[i:i+1],
            y_batch=y_batch[i:i+1],
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
           score = (entropy_random - entropy_original) / entropy_max
           where entropy_max = ln(num_features).

    The intuition is: a faithful explanation should be relatively
    simple/structured (low entropy) for the trained model, but become
    more uniform/noisy (high entropy) when the model is randomised.
    A larger positive score indicates the explanation was meaningfully
    capturing model structure.

    **Interpretation:**
        - Higher score = better (explanation was model-sensitive)
        - Score ≈ 0 = explanation complexity unchanged (bad)
        - Negative score = original explanation was MORE noisy than
          random (very bad, suggests the method adds noise)

    Args:
        model: PyTorch nn.Module. Deep-copied internally.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        seed: Random seed for reproducibility.

    Returns:
        Mean relative complexity increase (float). Higher = better.

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
    import torch

    rng = np.random.default_rng(seed)
    batch_size = x_batch.shape[0]

    # Original model explanations
    original_model = copy.deepcopy(model)
    original_model.eval()

    # Fully randomised model
    random_model = copy.deepcopy(model)
    random_model.eval()
    layers = _get_named_layers(random_model)
    for layer_name, _ in layers:
        _randomise_layer_parameters(random_model, layer_name, rng=rng)

    scores = []
    for i in range(batch_size):
        x_single = x_batch[i:i+1]
        y_single = y_batch[i]

        # Original explanation entropy
        attr_orig = explain_func(original_model, x_single, y_single)
        attr_orig = _extract_attribution_array(attr_orig)
        entropy_orig = _discrete_entropy(attr_orig)

        # Randomised explanation entropy
        attr_rand = explain_func(random_model, x_single, y_single)
        attr_rand = _extract_attribution_array(attr_rand)
        entropy_rand = _discrete_entropy(attr_rand)

        # Maximum possible entropy for this dimensionality
        num_features = attr_orig.size
        entropy_max = np.log(num_features) if num_features > 1 else 1.0

        # Relative complexity increase
        if entropy_max < 1e-12:
            score = 0.0
        else:
            score = (entropy_rand - entropy_orig) / entropy_max

        scores.append(float(score))

    return float(np.mean(scores))


def compute_batch_efficient_mprt(
    model,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    explain_func: Callable,
    seed: Optional[int] = None,
) -> List[float]:
    """
    Compute Efficient MPRT for each sample individually.

    Args:
        model: PyTorch nn.Module.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        seed: Random seed.

    Returns:
        List of per-sample complexity increase scores. Higher = better.

    References:
        Hedström, A., Weber, L., Lapuschkin, S., & Höhne, M. (2023).
        Sanity Checks Revisited. XAI in Action.
    """
    _validate_torch_available()
    import torch

    rng = np.random.default_rng(seed)
    batch_size = x_batch.shape[0]

    original_model = copy.deepcopy(model)
    original_model.eval()

    random_model = copy.deepcopy(model)
    random_model.eval()
    layers = _get_named_layers(random_model)
    for layer_name, _ in layers:
        _randomise_layer_parameters(random_model, layer_name, rng=rng)

    scores = []
    for i in range(batch_size):
        x_single = x_batch[i:i+1]
        y_single = y_batch[i]

        attr_orig = explain_func(original_model, x_single, y_single)
        attr_orig = _extract_attribution_array(attr_orig)
        entropy_orig = _discrete_entropy(attr_orig)

        attr_rand = explain_func(random_model, x_single, y_single)
        attr_rand = _extract_attribution_array(attr_rand)
        entropy_rand = _discrete_entropy(attr_rand)

        num_features = attr_orig.size
        entropy_max = np.log(num_features) if num_features > 1 else 1.0

        if entropy_max < 1e-12:
            score = 0.0
        else:
            score = (entropy_rand - entropy_orig) / entropy_max

        scores.append(float(score))

    return scores


# =============================================================================
# Data Randomisation Test (Adebayo et al., 2018)
# =============================================================================

def compute_data_randomisation_score(
    attr_trained: Union[np.ndarray, "Explanation"],
    attr_random_labels: Union[np.ndarray, "Explanation"],
    similarity_func: Union[str, SimilarityFunc] = "spearman",
) -> float:
    """
    Compute Data Randomisation score from pre-computed attributions (low-level API).

    Compares the explanation from a model trained on true labels against
    the explanation from a model trained on randomised labels. A faithful
    explanation should differ significantly between the two models, since
    the model trained on random labels has learned no meaningful
    data-label relationship.

    **Interpretation:**
        - Lower similarity = better (explanation captures data structure)
        - High similarity = explanation ignores training data (bad)

    Args:
        attr_trained: Attribution array from the model trained on
            true labels.
        attr_random_labels: Attribution array from the model trained
            on randomised labels.
        similarity_func: Similarity measure. One of "spearman"
            (default), "pearson", "cosine", "ssim", "mse", or a
            callable f(a, b) -> float.

    Returns:
        Similarity score (float). Lower = better.

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
    return _compute_similarity(a, b, similarity_func)


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

    Unlike MPRT which tests sensitivity to model parameters, this test
    evaluates whether explanations capture meaningful data-label
    structure. A model trained on random labels memorises noise rather
    than learning real patterns, so a faithful explanation should look
    fundamentally different.

    **Important:** The user must provide both models. Training a model
    on random labels is computationally expensive and dataset-specific,
    so the library does not handle it internally. This design choice
    follows the principle that the library evaluates explanations, not
    trains models.

    **Algorithm:**
        1. For each sample in the batch:
           a. Compute explanation from model_trained.
           b. Compute explanation from model_random_labels.
           c. Compute similarity between the two explanations.
        2. Return the mean similarity across all samples.

    **Interpretation:**
        - Lower score = better (explanation captures data structure)
        - Score ≈ 1.0 = explanation ignores training data (bad)

    **Note:** This metric is excluded from Quantus due to the
    requirement of providing a retrained model. Explainiverse includes
    it as it completes the full Adebayo et al. (2018) evaluation
    framework and is critical for comprehensive sanity checking.

    Args:
        model_trained: PyTorch nn.Module trained on true labels.
            Not modified.
        model_random_labels: PyTorch nn.Module trained on randomised
            labels (same architecture, same data, shuffled labels).
            Not modified.
        x_batch: Input data, shape (batch_size, ...).
        y_batch: Target labels, shape (batch_size,). Used as the
            target class for both explanations.
        explain_func: Callable with signature:
            ``explain_func(model, x, y) -> np.ndarray``
            where x is a single input (with batch dim) and y is a
            scalar target class label.
        similarity_func: Similarity measure (string key or callable).

    Returns:
        Mean similarity score (float). Lower = better.

    Raises:
        ImportError: If PyTorch is not installed.

    Example:
        >>> import torch.nn as nn
        >>> import numpy as np
        >>> # Model trained normally
        >>> model_true = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
        >>> # Model trained on shuffled labels (different weights)
        >>> model_rand = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 3))
        >>> x = np.random.randn(5, 10).astype(np.float32)
        >>> y = np.array([0, 1, 2, 0, 1])
        >>> def explain_fn(model, x, y):
        ...     import torch
        ...     x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        ...     out = model(x_t)
        ...     out[0, y].backward()
        ...     return x_t.grad.detach().numpy()
        >>> score = compute_data_randomisation(
        ...     model_true, model_rand, x, y, explain_fn
        ... )

    References:
        Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
        & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    """
    _validate_torch_available()

    model_a = copy.deepcopy(model_trained)
    model_a.eval()
    model_b = copy.deepcopy(model_random_labels)
    model_b.eval()

    batch_size = x_batch.shape[0]
    scores = []

    for i in range(batch_size):
        x_single = x_batch[i:i+1]
        y_single = y_batch[i]

        attr_trained = explain_func(model_a, x_single, y_single)
        attr_trained = _extract_attribution_array(attr_trained)

        attr_random = explain_func(model_b, x_single, y_single)
        attr_random = _extract_attribution_array(attr_random)

        score = _compute_similarity(attr_trained, attr_random, similarity_func)
        scores.append(score)

    return float(np.mean(scores))


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
        y_batch: Target labels, shape (batch_size,).
        explain_func: Callable(model, x, y) -> np.ndarray.
        similarity_func: Similarity measure (string or callable).

    Returns:
        List of per-sample similarity scores. Lower = better.

    References:
        Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I. J., Hardt, M.,
        & Kim, B. (2018). Sanity Checks for Saliency Maps. NeurIPS.
    """
    _validate_torch_available()

    model_a = copy.deepcopy(model_trained)
    model_a.eval()
    model_b = copy.deepcopy(model_random_labels)
    model_b.eval()

    batch_size = x_batch.shape[0]
    scores = []

    for i in range(batch_size):
        x_single = x_batch[i:i+1]
        y_single = y_batch[i]

        attr_trained = explain_func(model_a, x_single, y_single)
        attr_trained = _extract_attribution_array(attr_trained)

        attr_random = explain_func(model_b, x_single, y_single)
        attr_random = _extract_attribution_array(attr_random)

        score = _compute_similarity(attr_trained, attr_random, similarity_func)
        scores.append(score)

    return scores
