"""Bernoulli KL confidence bounds and deterministic top-m KL-LUCB.

This is a clean-room implementation of the bandit primitives used by
Algorithms 1 and 2 in Ribeiro, Singh, and Guestrin, *Anchors: High-Precision
Model-Agnostic Explanations* (AAAI 2018):
https://doi.org/10.1609/aaai.v32i1.11491

The authors' BSD-licensed reference implementation was consulted for algorithm
structure and the exploration rate, pinned at revision
``b1f5e6ca37428613723597e85c38558e8cd21c2e``:
https://github.com/marcotcr/anchor

Unlike the historical reference helpers, the inverse bounds here handle exact
Bernoulli endpoints and return numerically outward-conservative roots.  The
KL-LUCB routine owns no random state: callers supply a callback that returns the
number of Bernoulli successes from a requested arm and sample count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Optional, Sequence, Tuple

PullFunction = Callable[[int, int], int]


@dataclass(frozen=True)
class ArmStatistics:
    """Final sufficient statistics and simultaneous bounds for one arm."""

    index: int
    successes: int
    samples: int
    empirical_mean: Optional[float]
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class KLLUCBResult:
    """Result of one bounded top-m KL-LUCB invocation.

    ``max_samples`` and ``samples_drawn`` count only newly requested Bernoulli
    rows.  Any caller-supplied initial statistics are not charged again.
    ``selected_indices`` is empty when the budget cannot initialize every arm;
    otherwise it is the deterministic empirical top-m set, even if the budget
    expires before the PAC stopping rule is met.
    """

    selected_indices: Tuple[int, ...]
    statistics: Tuple[ArmStatistics, ...]
    samples_drawn: int
    max_samples: int
    rounds: int
    converged: bool
    budget_exhausted: bool
    termination_reason: str


def _validate_probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return numeric


def _validate_non_negative_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _validate_count(value: int, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    numeric = int(value)
    if numeric < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return numeric


def _stable_log_ratio(numerator: float, denominator: float) -> float:
    """Return ``log(numerator / denominator)`` without close-value cancellation."""
    difference = numerator - denominator
    if abs(difference) <= 0.5 * denominator:
        return math.log1p(difference / denominator)
    return math.log(numerator) - math.log(denominator)


def _log_one_minus_tail(value: float) -> float:
    """Return ``-log(1 - value) - value`` accurately for ``abs(value) <= 1/2``."""
    terms = []
    power = value * value
    for exponent in range(2, 65):
        term = power / exponent
        terms.append(term)
        partial = math.fsum(terms)
        if exponent > 2 and abs(term) <= math.ulp(partial):
            break
        power *= value
    return max(0.0, math.fsum(terms))


def bernoulli_kl(p: float, q: float) -> float:
    """Return ``KL(Bernoulli(p) || Bernoulli(q))`` with exact endpoints.

    The convention ``0 * log(0 / q) = 0`` is used.  A positive-probability
    event assigned zero probability by ``q`` yields positive infinity.
    """

    p_value = _validate_probability(p, name="p")
    q_value = _validate_probability(q, name="q")
    if p_value == q_value:
        return 0.0
    if q_value == 0.0:
        return math.inf
    if q_value == 1.0:
        return math.inf
    if p_value == 0.0:
        return -math.log1p(-q_value)
    if p_value == 1.0:
        return -math.log(q_value)
    difference = p_value - q_value
    relative_to_p = difference / p_value
    relative_to_complement = difference / (1.0 - p_value)
    if max(abs(relative_to_p), abs(relative_to_complement)) <= 0.5:
        # Expanding both logarithms around p cancels the first-order terms
        # symbolically.  The two remaining tails are non-negative, avoiding
        # the catastrophic subtraction that occurs close to p == q.
        return math.fsum(
            (
                p_value * _log_one_minus_tail(relative_to_p),
                (1.0 - p_value) * _log_one_minus_tail(-relative_to_complement),
            )
        )
    divergence = math.fsum(
        (
            p_value * _stable_log_ratio(p_value, q_value),
            (1.0 - p_value) * _stable_log_ratio(1.0 - p_value, 1.0 - q_value),
        )
    )
    # The exact divergence is non-negative.  Adjacent floats can leave a tiny
    # negative residue after the two opposite first-order terms cancel.
    return max(0.0, divergence)


def _kl_roundoff_tolerance(p: float, q: float, divergence: float) -> float:
    """Conservative absolute error allowance for a finite interior KL value."""
    if not math.isfinite(divergence):
        return 0.0
    difference = p - q
    relative_to_p = difference / p
    relative_to_complement = difference / (1.0 - p)
    if max(abs(relative_to_p), abs(relative_to_complement)) <= 0.5:
        scale = divergence
    else:
        scale = math.fsum(
            (
                abs(p * _stable_log_ratio(p, q)),
                abs((1.0 - p) * _stable_log_ratio(1.0 - p, 1.0 - q)),
            )
        )
    return 32.0 * math.ulp(1.0) * scale + 4.0 * math.ulp(divergence)


def _validate_binomial_statistics(successes: int, samples: int) -> tuple[int, int]:
    successes_value = _validate_count(successes, name="successes")
    samples_value = _validate_count(samples, name="samples")
    if successes_value > samples_value:
        raise ValueError("successes must not exceed samples")
    return successes_value, samples_value


def kl_lower_bound(successes: int, samples: int, beta: float) -> float:
    """Return an outward-conservative Bernoulli KL lower bound.

    For ``samples > 0``, the boundary solves
    ``KL(successes / samples, q) = beta / samples`` on the lower side of the
    empirical mean.  With no samples the confidence interval is ``[0, 1]``.
    """

    successes_value, samples_value = _validate_binomial_statistics(successes, samples)
    beta_value = _validate_non_negative_real(beta, name="beta")
    if samples_value == 0:
        return 0.0

    mean = successes_value / samples_value
    level = beta_value / samples_value
    if beta_value == 0.0 or mean == 0.0:
        return mean
    if mean == 1.0:
        root = math.exp(-level)
        return math.nextafter(root, 0.0) if root > 0.0 else 0.0
    if level == 0.0:
        # Positive beta can underflow only during division.  The true root is
        # then between the mean and its adjacent outward float.
        return math.nextafter(mean, 0.0)

    # ``outside`` is below the confidence set and ``inside`` is in it.  The
    # returned outside endpoint is at or below the mathematical root.
    outside = 0.0
    inside = mean
    for _ in range(100):
        midpoint = (outside + inside) / 2.0
        divergence = bernoulli_kl(mean, midpoint)
        tolerance = _kl_roundoff_tolerance(mean, midpoint, divergence)
        if divergence - tolerance > level:
            outside = midpoint
        else:
            inside = midpoint
    return math.nextafter(outside, 0.0)


def kl_upper_bound(successes: int, samples: int, beta: float) -> float:
    """Return an outward-conservative Bernoulli KL upper bound.

    For ``samples > 0``, the boundary solves
    ``KL(successes / samples, q) = beta / samples`` on the upper side of the
    empirical mean.  With no samples the confidence interval is ``[0, 1]``.
    """

    successes_value, samples_value = _validate_binomial_statistics(successes, samples)
    beta_value = _validate_non_negative_real(beta, name="beta")
    if samples_value == 0:
        return 1.0

    mean = successes_value / samples_value
    level = beta_value / samples_value
    if beta_value == 0.0 or mean == 1.0:
        return mean
    if mean == 0.0:
        root = -math.expm1(-level)
        return math.nextafter(root, 1.0) if root < 1.0 else 1.0
    if level == 0.0:
        # See the corresponding lower-bound branch above.
        return math.nextafter(mean, 1.0)

    # ``inside`` is in the confidence set and ``outside`` is above it.  The
    # returned outside endpoint is at or above the mathematical root.
    inside = mean
    outside = 1.0
    for _ in range(100):
        midpoint = (inside + outside) / 2.0
        divergence = bernoulli_kl(mean, midpoint)
        tolerance = _kl_roundoff_tolerance(mean, midpoint, divergence)
        if divergence - tolerance > level:
            outside = midpoint
        else:
            inside = midpoint
    return math.nextafter(outside, 1.0)


def compute_beta(n_arms: int, time_index: int, delta: float) -> float:
    """Return the KL-LUCB exploration rate used by the Anchors reference.

    The computation is performed in log space:

    ``beta(K, t, delta) = log(405.5 * K * t**1.1 / delta)``
    ``                     + log(log(405.5 * K * t**1.1 / delta))``.
    """

    arms_value = _validate_count(n_arms, name="n_arms", minimum=1)
    time_value = _validate_count(time_index, name="time_index", minimum=1)
    delta_value = _validate_probability(delta, name="delta")
    if delta_value in {0.0, 1.0}:
        raise ValueError("delta must be strictly between 0 and 1")

    log_argument = (
        math.log(405.5) + math.log(arms_value) + 1.1 * math.log(time_value) - math.log(delta_value)
    )
    return log_argument + math.log(log_argument)


def _validate_initial_statistics(
    n_arms: int,
    initial_successes: Optional[Sequence[int]],
    initial_samples: Optional[Sequence[int]],
) -> tuple[list[int], list[int]]:
    if (initial_successes is None) != (initial_samples is None):
        raise ValueError("initial_successes and initial_samples must be supplied together")
    if initial_successes is None:
        return [0] * n_arms, [0] * n_arms

    try:
        successes_values = list(initial_successes)
        sample_values = list(initial_samples)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("initial statistics must be integer sequences") from exc
    if len(successes_values) != n_arms or len(sample_values) != n_arms:
        raise ValueError("initial statistics must contain one entry per arm")

    validated_successes: list[int] = []
    validated_samples: list[int] = []
    for index, (successes, samples) in enumerate(zip(successes_values, sample_values)):
        try:
            successes_value, samples_value = _validate_binomial_statistics(successes, samples)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"invalid initial statistics for arm {index}: {exc}") from exc
        validated_successes.append(successes_value)
        validated_samples.append(samples_value)
    return validated_successes, validated_samples


def _pull_successes(pull_fn: PullFunction, arm_index: int, n_samples: int) -> int:
    result = pull_fn(arm_index, n_samples)
    if isinstance(result, bool) or not isinstance(result, Integral):
        raise TypeError("pull_fn must return an integer success count")
    successes = int(result)
    if successes < 0 or successes > n_samples:
        raise ValueError("pull_fn returned a success count outside the requested sample count")
    return successes


def _empirical_top_m(successes: Sequence[int], samples: Sequence[int], m: int) -> tuple[int, ...]:
    means = [successes[index] / samples[index] for index in range(len(samples))]
    selected = sorted(range(len(samples)), key=lambda index: (-means[index], index))[:m]
    return tuple(sorted(selected))


def _statistics(
    successes: Sequence[int],
    samples: Sequence[int],
    *,
    delta: float,
    time_index: int,
) -> tuple[ArmStatistics, ...]:
    if not samples:
        return ()
    beta = compute_beta(len(samples), time_index, delta)
    result = []
    for index, (arm_successes, arm_samples) in enumerate(zip(successes, samples)):
        result.append(
            ArmStatistics(
                index=index,
                successes=int(arm_successes),
                samples=int(arm_samples),
                empirical_mean=(float(arm_successes / arm_samples) if arm_samples > 0 else None),
                lower_bound=kl_lower_bound(arm_successes, arm_samples, beta),
                upper_bound=kl_upper_bound(arm_successes, arm_samples, beta),
            )
        )
    return tuple(result)


def kl_lucb_top_m(
    pull_fn: PullFunction,
    n_arms: int,
    m: int,
    *,
    max_samples: int,
    epsilon: float = 0.1,
    delta: float = 0.05,
    batch_size: int = 1,
    initial_successes: Optional[Sequence[int]] = None,
    initial_samples: Optional[Sequence[int]] = None,
) -> KLLUCBResult:
    """Identify a deterministic empirical top-m set with bounded KL-LUCB.

    At each round, the routine pulls the selected arm with the smallest lower
    bound and the unselected arm with the largest upper bound.  It converges
    when their bound gap is at most ``epsilon``.  Exact ties are resolved by
    lower arm index.

    ``max_samples`` is a hard budget on newly requested callback rows.  A
    paired LUCB round is never split: if fewer than two rows remain, the
    routine terminates without another callback.  For ``batch_size > 1``, a
    smaller paired batch may be used to consume the remaining even budget.
    """

    if not callable(pull_fn):
        raise TypeError("pull_fn must be callable")
    arms_value = _validate_count(n_arms, name="n_arms")
    m_value = _validate_count(m, name="m")
    budget = _validate_count(max_samples, name="max_samples")
    batch_value = _validate_count(batch_size, name="batch_size", minimum=1)
    epsilon_value = _validate_non_negative_real(epsilon, name="epsilon")
    if epsilon_value >= 1.0:
        raise ValueError("epsilon must be less than 1")
    delta_value = _validate_probability(delta, name="delta")
    if delta_value in {0.0, 1.0}:
        raise ValueError("delta must be strictly between 0 and 1")

    successes, samples = _validate_initial_statistics(
        arms_value,
        initial_successes,
        initial_samples,
    )

    if arms_value == 0:
        return KLLUCBResult((), (), 0, budget, 0, True, False, "no_arms")
    initial_time_index = max(1, sum(samples))
    effective_m = min(m_value, arms_value)
    if effective_m == 0:
        return KLLUCBResult(
            (),
            _statistics(
                successes,
                samples,
                delta=delta_value,
                time_index=initial_time_index,
            ),
            0,
            budget,
            0,
            True,
            False,
            "no_selection_requested",
        )
    if effective_m == arms_value:
        return KLLUCBResult(
            tuple(range(arms_value)),
            _statistics(
                successes,
                samples,
                delta=delta_value,
                time_index=initial_time_index,
            ),
            0,
            budget,
            0,
            True,
            False,
            "all_arms_selected",
        )

    samples_drawn = 0
    rounds = 0
    performed_initialization = not any(samples)
    for arm_index in range(arms_value):
        if samples[arm_index] > 0:
            continue
        if samples_drawn >= budget:
            return KLLUCBResult(
                (),
                _statistics(
                    successes,
                    samples,
                    delta=delta_value,
                    time_index=max(1, sum(samples)),
                ),
                samples_drawn,
                budget,
                rounds,
                False,
                True,
                "budget_exhausted_initialization",
            )
        successes[arm_index] += _pull_successes(pull_fn, arm_index, 1)
        samples[arm_index] += 1
        samples_drawn += 1

    # Paper Algorithm 1 initializes all arms at stage one, then advances the
    # shared stage once per paired observation.  Existing caller statistics do
    # not encode their pull history, so their total count is the conservative
    # inferred starting stage unless this invocation performed initialization.
    time_index = 1 if performed_initialization else max(1, sum(samples))
    while True:
        statistics = _statistics(
            successes,
            samples,
            delta=delta_value,
            time_index=time_index,
        )
        selected = _empirical_top_m(successes, samples, effective_m)
        selected_set = set(selected)
        unselected = tuple(index for index in range(arms_value) if index not in selected_set)

        weakest_selected = min(
            selected,
            key=lambda index: (statistics[index].lower_bound, index),
        )
        strongest_unselected = min(
            unselected,
            key=lambda index: (-statistics[index].upper_bound, index),
        )
        gap = (
            statistics[strongest_unselected].upper_bound - statistics[weakest_selected].lower_bound
        )
        if gap <= epsilon_value:
            return KLLUCBResult(
                selected,
                statistics,
                samples_drawn,
                budget,
                rounds,
                True,
                False,
                "converged",
            )

        remaining = budget - samples_drawn
        paired_batch = min(batch_value, remaining // 2)
        if paired_batch < 1:
            return KLLUCBResult(
                selected,
                statistics,
                samples_drawn,
                budget,
                rounds,
                False,
                True,
                "budget_exhausted",
            )

        for arm_index in (weakest_selected, strongest_unselected):
            successes[arm_index] += _pull_successes(pull_fn, arm_index, paired_batch)
            samples[arm_index] += paired_batch
            samples_drawn += paired_batch
        rounds += 1
        time_index += paired_batch
