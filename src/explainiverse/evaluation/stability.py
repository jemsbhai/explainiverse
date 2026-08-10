"""Verified stability entry points and compatibility wrappers.

Historically this module exposed two unrelated heuristics as ``RIS`` and
``ROS``. The written-paper/Quantus Relative Input/Output Stability contracts
are implemented in ``evaluation.robustness`` and require model-conditioned
perturbations. The wrappers below delegate to those implementations and reject
the old insufficient call contracts instead of silently returning mislabeled
scores.
"""

from typing import Callable, Dict, Optional, Union

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.evaluation.robustness import (
    _generate_perturbations_l2,
    _generate_perturbations_linf,
    _get_explanation_vector,
    _get_explanation_vector_and_target,
    _validate_batch,
    _validate_instance,
    _validate_norm_order,
    _validate_sampling_parameters,
    compute_relative_input_stability,
    compute_relative_output_stability,
)


def compute_ris(
    explainer: BaseExplainer,
    instance: np.ndarray,
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    seed: Optional[int] = None,
    *,
    model=None,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    target_class: Optional[int] = None,
) -> float:
    """Compute the paper/Quantus Relative Input Stability Equation 2 contract.

    ``model`` is required for the same-predicted-class constraint. It may be
    omitted only when ``explainer.model`` is available. The former routine in
    this module computed a mean absolute local-Lipschitz ratio and was not RIS.
    """
    resolved_model = model if model is not None else getattr(explainer, "model", None)
    if resolved_model is None:
        raise ValueError(
            "Equation 2 RIS requires model=... (or explainer.model) to enforce "
            "the same-predicted-class constraint."
        )
    result = compute_relative_input_stability(
        explainer,
        resolved_model,
        instance,
        n_perturbations=n_perturbations,
        noise_scale=noise_scale,
        norm_ord=norm_ord,
        epsilon_min=epsilon_min,
        aggregation="max",
        feature_types=feature_types,
        discrete_flip_prob=discrete_flip_prob,
        seed=seed,
        target_class=target_class,
    )
    if isinstance(result, dict):
        raise RuntimeError("RIS unexpectedly returned detail data")
    return float(result)


def compute_ros(
    explainer: BaseExplainer,
    model,
    instance: np.ndarray,
    reference_instances: Optional[np.ndarray] = None,
    n_neighbors: int = 5,
    prediction_threshold: float = 0.05,
    *,
    logit_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    norm_ord: Union[int, float] = 2,
    epsilon_min: float = 1e-7,
    feature_types: Optional[np.ndarray] = None,
    discrete_flip_prob: float = 0.03,
    seed: Optional[int] = None,
    target_class: Optional[int] = None,
) -> float:
    """Compute the paper/Quantus Relative Output Stability Equation 5 contract.

    A callable returning pre-softmax logits is required. The old
    ``reference_instances``/cosine-neighbour routine measured a different,
    noncanonical similarity heuristic and is deliberately unavailable.
    """
    if reference_instances is not None or n_neighbors != 5 or prediction_threshold != 0.05:
        raise ValueError(
            "reference_instances, n_neighbors, and prediction_threshold belong "
            "to the retired cosine-neighbour heuristic, not Equation 5 ROS."
        )
    if logit_fn is None:
        raise ValueError(
            "Equation 5 ROS requires logit_fn=... returning pre-softmax logits; "
            "probabilities are not interchangeable with Equation 5 logits."
        )
    result = compute_relative_output_stability(
        explainer,
        model,
        instance,
        logit_fn=logit_fn,
        n_perturbations=n_perturbations,
        noise_scale=noise_scale,
        norm_ord=norm_ord,
        epsilon_min=epsilon_min,
        aggregation="max",
        feature_types=feature_types,
        discrete_flip_prob=discrete_flip_prob,
        seed=seed,
        target_class=target_class,
    )
    if isinstance(result, dict):
        raise RuntimeError("ROS unexpectedly returned detail data")
    return float(result)


def compute_lipschitz_estimate(
    explainer: BaseExplainer,
    instance: np.ndarray,
    n_samples: int = 20,
    radius: float = 0.1,
    seed: Optional[int] = None,
    *,
    norm_ord: Union[int, float] = 2,
    perturb_norm: str = "l2",
    target_class: Optional[int] = None,
) -> float:
    """Estimate the anchor-point local Lipschitz constant.

    This is the Monte Carlo approximation to Alvarez-Melis & Jaakkola (2018,
    Equation 1): ``max ||E(x)-E(x')|| / ||x-x'||`` for neighbours of the
    fixed anchor ``x``. The former implementation compared arbitrary pairs of
    neighbours and therefore did not estimate the stated pointwise quantity.
    """
    instance = _validate_instance(instance)
    _validate_sampling_parameters(n_samples=n_samples, radius=radius)
    _validate_norm_order(norm_ord)
    if radius == 0:
        raise ValueError(
            "The local Lipschitz ratio is undefined at radius=0 because every "
            "sampled denominator is zero."
        )
    if perturb_norm not in {"l2", "linf"}:
        raise ValueError("perturb_norm must be 'l2' or 'linf'.")

    rng = np.random.default_rng(seed)
    generator = _generate_perturbations_l2 if perturb_norm == "l2" else _generate_perturbations_linf
    neighbours = generator(instance, radius, n_samples, rng)
    original, explained_target = _get_explanation_vector_and_target(
        explainer,
        instance,
        instance.size,
        target_class=target_class,
    )

    ratios = []
    for neighbour in neighbours:
        denominator = np.linalg.norm(instance - neighbour, ord=norm_ord)
        if denominator == 0:
            continue
        perturbed = _get_explanation_vector(
            explainer,
            neighbour,
            instance.size,
            target_class=target_class,
            expected_target=explained_target,
        )
        numerator = np.linalg.norm(original - perturbed, ord=norm_ord)
        ratios.append(float(numerator / denominator))
    if not ratios:
        raise ValueError("No nonzero perturbation denominator was sampled.")
    return float(np.max(ratios))


def compute_stability_metrics(
    explainer: BaseExplainer,
    model,
    instance: np.ndarray,
    background_data: Optional[np.ndarray] = None,
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    n_neighbors: int = 5,
    seed: Optional[int] = None,
    *,
    logit_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    target_class: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """Compute verified RIS, local Lipschitz, and optionally ROS contracts.

    ``background_data`` and ``n_neighbors`` are retired compatibility
    parameters. Passing a background pool would request the noncanonical
    cosine-neighbour heuristic and is rejected.
    """
    if background_data is not None or n_neighbors != 5:
        raise ValueError(
            "background_data/n_neighbors belong to the retired noncanonical "
            "ROS heuristic. Supply logit_fn for Equation 5 ROS."
        )
    _validate_sampling_parameters(
        n_samples=n_perturbations,
        noise_scale=noise_scale,
    )
    if noise_scale == 0:
        raise ValueError(
            "noise_scale=0 makes the included local Lipschitz denominator "
            "undefined; use a positive neighbourhood scale."
        )
    result: Dict[str, Optional[float]] = {
        "ris": compute_ris(
            explainer,
            instance,
            n_perturbations=n_perturbations,
            noise_scale=noise_scale,
            seed=seed,
            model=model,
            target_class=target_class,
        ),
        "lipschitz": compute_lipschitz_estimate(
            explainer,
            instance,
            n_samples=n_perturbations,
            radius=noise_scale,
            seed=seed,
            target_class=target_class,
        ),
        "ros": None,
    }
    if logit_fn is not None:
        result["ros"] = compute_ros(
            explainer,
            model,
            instance,
            logit_fn=logit_fn,
            n_perturbations=n_perturbations,
            noise_scale=noise_scale,
            seed=seed,
            target_class=target_class,
        )
    return result


def compute_batch_stability(
    explainer: BaseExplainer,
    model,
    X: np.ndarray,
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
    *,
    logit_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    target_class: Optional[int] = None,
) -> dict:
    """Compute verified stability summaries over a strict 2D batch."""
    X = _validate_batch(X)
    _validate_sampling_parameters(
        n_samples=n_perturbations,
        noise_scale=noise_scale,
    )
    if noise_scale == 0:
        raise ValueError(
            "noise_scale=0 makes the included local Lipschitz denominator "
            "undefined; use a positive neighbourhood scale."
        )
    n = len(X)
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)):
            raise TypeError("max_samples must be an integer or None.")
        if max_samples <= 0:
            raise ValueError("max_samples must be greater than zero.")
        n = min(n, max_samples)

    ris_scores = []
    lip_scores = []
    ros_scores = []
    for index in range(n):
        item_seed = seed + index if seed is not None else None
        ris_scores.append(
            compute_ris(
                explainer,
                X[index],
                n_perturbations=n_perturbations,
                noise_scale=noise_scale,
                seed=item_seed,
                model=model,
                target_class=target_class,
            )
        )
        lip_scores.append(
            compute_lipschitz_estimate(
                explainer,
                X[index],
                n_samples=n_perturbations,
                radius=noise_scale,
                seed=item_seed,
                target_class=target_class,
            )
        )
        if logit_fn is not None:
            ros_scores.append(
                compute_ros(
                    explainer,
                    model,
                    X[index],
                    logit_fn=logit_fn,
                    n_perturbations=n_perturbations,
                    noise_scale=noise_scale,
                    seed=item_seed,
                    target_class=target_class,
                )
            )

    def summarise_defined(scores):
        """Summarise finite per-row scores; no valid rows is explicitly undefined."""
        values = np.asarray(scores, dtype=float)
        defined = values[np.isfinite(values)]
        if defined.size == 0:
            return float("nan"), float("nan"), 0
        return float(np.mean(defined)), float(np.std(defined)), int(defined.size)

    mean_ris, std_ris, n_valid_ris = summarise_defined(ris_scores)
    mean_lipschitz, std_lipschitz, n_valid_lipschitz = summarise_defined(lip_scores)
    mean_ros, std_ros, n_valid_ros = summarise_defined(ros_scores)

    result = {
        "n_samples": n,
        "mean_ris": mean_ris,
        "std_ris": std_ris,
        "n_valid_ris": n_valid_ris,
        "mean_lipschitz": mean_lipschitz,
        "std_lipschitz": std_lipschitz,
        "n_valid_lipschitz": n_valid_lipschitz,
        "mean_ros": mean_ros,
        "std_ros": std_ros,
        "n_valid_ros": n_valid_ros,
    }
    return result


def compare_explainer_stability(
    explainers: Dict[str, BaseExplainer],
    model,
    X: np.ndarray,
    n_perturbations: int = 50,
    noise_scale: float = 0.05,
    max_samples: Optional[int] = 20,
    seed: Optional[int] = None,
    *,
    logit_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    target_class: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare verified stability summaries across explainers."""
    return {
        name: compute_batch_stability(
            explainer,
            model,
            X,
            n_perturbations=n_perturbations,
            noise_scale=noise_scale,
            max_samples=max_samples,
            seed=seed,
            logit_fn=logit_fn,
            target_class=target_class,
        )
        for name, explainer in explainers.items()
    }
