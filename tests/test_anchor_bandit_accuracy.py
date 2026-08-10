"""Independent accuracy tests for the tabular Anchors bandit primitives."""

import math
from decimal import Decimal, localcontext
from typing import List, Tuple

import pytest
from scipy.optimize import brentq

from explainiverse.explainers.rule_based._anchor_bandit import (
    bernoulli_kl,
    compute_beta,
    kl_lower_bound,
    kl_lucb_top_m,
    kl_upper_bound,
)


def _oracle_bernoulli_kl(p: float, q: float) -> float:
    """Independent textbook formula used only by the inverse-bound oracle."""
    if p == q:
        return 0.0
    if q in {0.0, 1.0}:
        return math.inf
    first = 0.0 if p == 0.0 else p * math.log(p / q)
    second = 0.0 if p == 1.0 else (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return first + second


def _decimal_bernoulli_kl(p: float, q: float) -> Decimal:
    """High-precision KL of the exact input floats for side-of-root checks."""
    if p == q:
        return Decimal(0)
    if q in {0.0, 1.0}:
        return Decimal("Infinity")
    with localcontext() as context:
        context.prec = 100
        p_decimal = Decimal.from_float(p)
        q_decimal = Decimal.from_float(q)
        return (
            p_decimal * (p_decimal / q_decimal).ln()
            + (Decimal(1) - p_decimal) * ((Decimal(1) - p_decimal) / (Decimal(1) - q_decimal)).ln()
        )


def test_bernoulli_kl_handles_exact_endpoints() -> None:
    assert bernoulli_kl(0.0, 0.0) == 0.0
    assert bernoulli_kl(1.0, 1.0) == 0.0
    assert bernoulli_kl(0.0, 0.2) == pytest.approx(-math.log(0.8))
    assert bernoulli_kl(1.0, 0.2) == pytest.approx(-math.log(0.2))
    assert bernoulli_kl(0.5, 0.0) == math.inf
    assert bernoulli_kl(0.5, 1.0) == math.inf
    assert bernoulli_kl(0.3, 0.7) == pytest.approx(float(_decimal_bernoulli_kl(0.3, 0.7)))


def test_bernoulli_kl_is_nonnegative_for_adjacent_and_extreme_floats() -> None:
    p = 0.025469958543613225
    adjacent = math.nextafter(p, 1.0)

    assert bernoulli_kl(p, adjacent) >= 0.0
    assert bernoulli_kl(adjacent, p) >= 0.0
    assert math.isfinite(bernoulli_kl(math.ulp(0.0), 0.5))
    assert math.isfinite(bernoulli_kl(0.5, math.ulp(0.0)))


@pytest.mark.parametrize("successes", [0, 1])
def test_kl_bounds_are_vacuous_without_samples(successes: int) -> None:
    if successes:
        with pytest.raises(ValueError, match="must not exceed"):
            kl_lower_bound(successes, 0, 1.0)
        return
    assert kl_lower_bound(0, 0, 1.0) == 0.0
    assert kl_upper_bound(0, 0, 1.0) == 1.0


def test_kl_bounds_have_exact_endpoint_formulas() -> None:
    samples = 17
    beta = 3.2
    level = beta / samples

    zero_upper_root = 1.0 - math.exp(-level)
    assert kl_lower_bound(0, samples, beta) == 0.0
    assert kl_upper_bound(0, samples, beta) >= zero_upper_root
    assert kl_upper_bound(0, samples, beta) == pytest.approx(zero_upper_root)

    one_lower_root = math.exp(-level)
    assert kl_lower_bound(samples, samples, beta) <= one_lower_root
    assert kl_lower_bound(samples, samples, beta) == pytest.approx(one_lower_root)
    assert kl_upper_bound(samples, samples, beta) == 1.0


def test_positive_underflowing_confidence_level_still_moves_bounds_outward() -> None:
    beta = math.ulp(0.0)
    mean = 0.5

    assert beta / 2 == 0.0
    assert kl_lower_bound(1, 2, beta) == math.nextafter(mean, 0.0)
    assert kl_upper_bound(1, 2, beta) == math.nextafter(mean, 1.0)


@pytest.mark.parametrize(
    ("successes", "samples", "beta"),
    [
        (3, 10, 0.7),
        (7, 11, 2.3),
        (1, 9, 4.0),
        (8, 9, 0.1),
        (17, 138, 1.2053256864766832e-8),
        (3, 782, 0.08253864507357289),
        (257, 342, 5355.6444326344845),
    ],
)
def test_kl_bounds_match_scipy_brentq_oracle(
    successes: int,
    samples: int,
    beta: float,
) -> None:
    mean = successes / samples
    level = beta / samples

    def root_equation(q: float) -> float:
        return _oracle_bernoulli_kl(mean, q) - level

    expected_lower = brentq(
        root_equation,
        0.0,
        mean,
        xtol=math.ulp(0.0),
        rtol=4.0 * math.ulp(1.0),
    )
    expected_upper = brentq(
        root_equation,
        mean,
        1.0,
        xtol=math.ulp(0.0),
        rtol=4.0 * math.ulp(1.0),
    )
    actual_lower = kl_lower_bound(successes, samples, beta)
    actual_upper = kl_upper_bound(successes, samples, beta)

    # Brent's roots are independent numerical oracles. Production roots are
    # checked for outward conservatism through the defining KL inequality
    # because Brent's tolerance does not promise which side of the root wins.
    assert actual_lower == pytest.approx(expected_lower, abs=2e-12)
    assert actual_upper == pytest.approx(expected_upper, abs=2e-12)
    with localcontext() as context:
        context.prec = 100
        exact_level = Decimal.from_float(beta) / Decimal(samples)
        assert _decimal_bernoulli_kl(mean, actual_lower) >= exact_level
        assert _decimal_bernoulli_kl(mean, actual_upper) >= exact_level


def test_compute_beta_matches_log_space_definition() -> None:
    n_arms = 5
    time_index = 7
    delta = 0.05
    argument = 405.5 * n_arms * time_index**1.1 / delta
    expected = math.log(argument) + math.log(math.log(argument))

    assert compute_beta(n_arms, time_index, delta) == pytest.approx(expected)
    assert compute_beta(n_arms, time_index + 1, delta) > expected
    assert compute_beta(n_arms + 1, time_index, delta) > expected
    assert compute_beta(n_arms, time_index, delta / 2) > expected


def test_kl_lucb_handles_degenerate_arm_and_selection_counts_without_pulls() -> None:
    def unexpected_pull(_arm: int, _count: int) -> int:
        raise AssertionError("degenerate selections must not sample")

    no_arms = kl_lucb_top_m(unexpected_pull, 0, 0, max_samples=0)
    assert no_arms.selected_indices == ()
    assert no_arms.statistics == ()
    assert no_arms.converged
    assert no_arms.termination_reason == "no_arms"

    no_selection = kl_lucb_top_m(unexpected_pull, 3, 0, max_samples=0)
    assert no_selection.selected_indices == ()
    assert no_selection.converged
    assert no_selection.termination_reason == "no_selection_requested"

    one_arm = kl_lucb_top_m(unexpected_pull, 1, 1, max_samples=0)
    assert one_arm.selected_indices == (0,)
    assert one_arm.converged
    assert one_arm.termination_reason == "all_arms_selected"

    all_arms = kl_lucb_top_m(unexpected_pull, 2, 3, max_samples=0)
    assert all_arms.selected_indices == (0, 1)
    assert all_arms.converged
    assert all_arms.termination_reason == "all_arms_selected"
    assert all_arms.samples_drawn == 0


def test_kl_lucb_reports_budget_exhaustion_during_initialization() -> None:
    pulls: List[Tuple[int, int]] = []

    def pull(arm: int, count: int) -> int:
        pulls.append((arm, count))
        return 0

    result = kl_lucb_top_m(pull, 3, 1, max_samples=2)

    assert pulls == [(0, 1), (1, 1)]
    assert result.samples_drawn == 2
    assert result.selected_indices == ()
    assert result.budget_exhausted
    assert not result.converged
    assert result.termination_reason == "budget_exhausted_initialization"
    assert [stat.samples for stat in result.statistics] == [1, 1, 0]


def test_kl_lucb_never_splits_a_pair_or_overshoots_an_odd_budget() -> None:
    pulls: List[Tuple[int, int]] = []

    def pull(arm: int, count: int) -> int:
        pulls.append((arm, count))
        return count if arm == 0 else 0

    result = kl_lucb_top_m(
        pull,
        2,
        1,
        max_samples=7,
        epsilon=0.0,
        batch_size=3,
    )

    assert pulls == [(0, 1), (1, 1), (0, 2), (1, 2)]
    assert result.samples_drawn == 6
    assert result.samples_drawn <= result.max_samples
    assert result.rounds == 1
    assert result.budget_exhausted
    assert not result.converged
    assert result.selected_indices == (0,)


def test_kl_lucb_uses_the_paired_stage_index_for_confidence_bounds() -> None:
    def pull(arm: int, count: int) -> int:
        return count if arm == 0 else 0

    delta = 0.05
    result = kl_lucb_top_m(
        pull,
        2,
        1,
        max_samples=4,
        epsilon=0.0,
        delta=delta,
    )

    # One all-arm initialization stage plus one paired pull stage.
    beta = compute_beta(2, 2, delta)
    assert result.samples_drawn == 4
    assert result.rounds == 1
    assert result.statistics[0].lower_bound == kl_lower_bound(2, 2, beta)
    assert result.statistics[1].upper_bound == kl_upper_bound(0, 2, beta)


def test_kl_lucb_identifies_a_scripted_unique_best_arm() -> None:
    calls: List[Tuple[int, int]] = []

    def pull(arm: int, count: int) -> int:
        calls.append((arm, count))
        if arm == 0:
            return count
        if arm == 1:
            return count // 4
        return 0

    result = kl_lucb_top_m(
        pull,
        3,
        1,
        max_samples=20_000,
        epsilon=0.05,
        delta=0.1,
    )

    assert result.selected_indices == (0,)
    assert result.converged
    assert not result.budget_exhausted
    assert result.termination_reason == "converged"
    assert result.samples_drawn == sum(count for _, count in calls)
    assert result.samples_drawn <= result.max_samples
    assert result.statistics[0].empirical_mean == 1.0
    strongest_challenger = max(stat.upper_bound for stat in result.statistics[1:])
    assert result.statistics[0].lower_bound >= strongest_challenger - 0.05
    assert all(count == 1 for _, count in calls)


def test_kl_lucb_breaks_empirical_ties_by_arm_index() -> None:
    def unexpected_pull(_arm: int, _count: int) -> int:
        raise AssertionError("a zero new-sample budget must not pull")

    result = kl_lucb_top_m(
        unexpected_pull,
        3,
        1,
        max_samples=0,
        epsilon=0.0,
        initial_successes=[1, 1, 1],
        initial_samples=[2, 2, 2],
    )

    assert result.selected_indices == (0,)
    assert result.rounds == 0
    assert result.budget_exhausted
    assert not result.converged


def test_kl_lucb_breaks_critical_arm_ties_in_callback_order() -> None:
    pulls: List[Tuple[int, int]] = []

    def pull(arm: int, count: int) -> int:
        pulls.append((arm, count))
        return count

    result = kl_lucb_top_m(
        pull,
        3,
        1,
        max_samples=2,
        epsilon=0.0,
        initial_successes=[1, 1, 1],
        initial_samples=[2, 2, 2],
    )

    assert pulls == [(0, 1), (1, 1)]
    assert result.selected_indices == (0,)
    assert result.rounds == 1
    assert result.budget_exhausted


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"n_arms": -1, "m": 1, "max_samples": 1}, ValueError, "non-negative"),
        ({"n_arms": 2, "m": -1, "max_samples": 1}, ValueError, "non-negative"),
        ({"n_arms": 2, "m": 1, "max_samples": -1}, ValueError, "non-negative"),
        ({"n_arms": 2, "m": 1, "max_samples": 1, "delta": 0.0}, ValueError, "strictly"),
        ({"n_arms": 2, "m": 1, "max_samples": 1, "epsilon": 1.0}, ValueError, "less"),
        ({"n_arms": 2, "m": 1, "max_samples": 1, "batch_size": 0}, ValueError, "positive"),
    ],
)
def test_kl_lucb_rejects_invalid_contracts(
    kwargs: dict,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        kl_lucb_top_m(lambda _arm, count: count, **kwargs)
