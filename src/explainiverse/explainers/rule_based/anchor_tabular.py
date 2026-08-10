"""Confidence-certified tabular Anchors under an empirical background distribution.

This module implements the bottom-up KL-LUCB beam-search structure in Algorithms
1 and 2 of Ribeiro, Singh, and Guestrin, *Anchors: High-Precision
Model-Agnostic Explanations* (AAAI 2018).  Its verified scope is deliberately
narrow: finite continuous numeric tabular classification, query-consistent
one-sided empirical-quantile predicates, and an explicitly supplied empirical
background distribution.

The returned rule is certified only when its sequential Bernoulli KL lower
bound is strictly greater than the requested precision threshold.  A finite
sample budget can therefore produce a useful candidate without producing an
Anchor certificate.  Beam search is an epsilon-PAC search heuristic and does
not establish the globally maximum-coverage rule, causal sufficiency, or
sufficiency under a different data distribution.

Primary reference:
    https://doi.org/10.1609/aaai.v32i1.11491

The authors' BSD-licensed reference implementation was consulted for the
beam-search structure and query-consistent predicate construction at revision
``b1f5e6ca37428613723597e85c38558e8cd21c2e``:
    https://github.com/marcotcr/anchor
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import (
    as_real_array,
    ensure_classification_task,
    normalize_classifier_outputs,
    validate_name_sequence,
    validate_single_tabular_instance,
)
from explainiverse.explainers.rule_based._anchor_bandit import (
    compute_beta,
    kl_lower_bound,
    kl_lucb_top_m,
    kl_upper_bound,
)

_Operator = Literal["leq", "gt"]
_Rule = Tuple[int, ...]


@dataclass(frozen=True)
class _Predicate:
    index: int
    feature_index: int
    operator: _Operator
    cut: float
    label: str
    background_support: np.ndarray

    def applies(self, values: np.ndarray) -> np.ndarray:
        if self.operator == "leq":
            return np.asarray(values <= self.cut, dtype=bool)
        return np.asarray(values > self.cut, dtype=bool)

    def payload(self, feature_name: str) -> Dict[str, Any]:
        return {
            "predicate_index": int(self.index),
            "feature": feature_name,
            "feature_index": int(self.feature_index),
            "operator": "<=" if self.operator == "leq" else ">",
            "threshold": float(self.cut),
            "lower_bound": float(self.cut) if self.operator == "gt" else None,
            "upper_bound": float(self.cut) if self.operator == "leq" else None,
            "lower_inclusive": False if self.operator == "gt" else None,
            "upper_inclusive": True if self.operator == "leq" else None,
            "label": self.label,
        }


@dataclass
class _CandidateStatistics:
    successes: int = 0
    samples: int = 0

    @property
    def mean(self) -> Optional[float]:
        return self.successes / self.samples if self.samples else None


@dataclass
class _PredictionBudget:
    max_samples: int
    samples_used: int = 0
    prediction_calls: int = 0

    @property
    def remaining(self) -> int:
        return self.max_samples - self.samples_used


class AnchorTabularExplainer(BaseExplainer):
    r"""Paper-structured KL-LUCB Anchors for continuous numeric tables.

    ``background_data`` defines a uniform discrete empirical perturbation
    distribution :math:`D`.  A conditional draw from :math:`D(\cdot\mid A)`
    samples an entire background row uniformly from the rows satisfying rule
    :math:`A`; it never repairs individual columns or creates unseen feature
    combinations.  Coverage is the exact fraction of background rows satisfying
    a candidate rule.

    The per-explanation confidence budget is split conservatively between
    KL-LUCB beam selection across depths and time-uniform candidate
    certification across the finite predicate search space.  This makes the
    requested ``delta`` and the effective allocations visible and auditable.

    Args:
        model: Classification adapter with a ``predict`` method.
        background_data: Non-empty finite 2D numeric empirical reference data.
        feature_names: One non-empty display name per input column.
        class_names: One non-empty display name per normalized output column.
        threshold: Required conditional precision, strictly between 0 and 1.
        delta: Total requested error probability, strictly between 0 and 1.
        epsilon: KL-LUCB top-beam tolerance in ``[0, 1)``.
        beam_size: Number of candidates retained at each predicate depth.
        batch_size: Perturbation rows requested per adaptive pull.
        max_samples: Hard budget on perturbation prediction rows.  The one
            query prediction is reported separately and is not charged here.
        max_anchor_size: Maximum number of atomic cut predicates.  ``None``
            permits all generated predicates.
        discretizer: ``"quartile"`` or ``"decile"`` empirical cuts.
        random_state: Per-operation NumPy generator seed.
    """

    background_data: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    threshold: float
    delta: float
    epsilon: float
    beam_size: int
    batch_size: int
    max_samples: int
    max_anchor_size: Optional[int]
    discretizer: str
    random_state: int
    cuts: Dict[int, np.ndarray]

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: Sequence[str],
        class_names: Sequence[str],
        threshold: float = 0.95,
        delta: float = 0.05,
        epsilon: float = 0.1,
        beam_size: int = 4,
        batch_size: int = 1,
        max_samples: int = 100_000,
        max_anchor_size: Optional[int] = None,
        discretizer: str = "quartile",
        random_state: int = 42,
    ) -> None:
        super().__init__(model)
        ensure_classification_task(model, context="AnchorTabular")

        background = as_real_array(
            background_data,
            name="background_data",
            dtype=float,
            require_finite=True,
        )
        if background.ndim != 2 or background.shape[0] == 0:
            raise ValueError("background_data must be a non-empty 2D array")
        if background.shape[1] == 0:
            raise ValueError("background_data must contain at least one feature")
        self.background_data = np.array(background, dtype=float, copy=True)

        validated_features = validate_name_sequence(feature_names, name="feature_names")
        validated_classes = validate_name_sequence(class_names, name="class_names")
        assert validated_features is not None and validated_classes is not None
        if len(validated_features) != self.background_data.shape[1]:
            raise ValueError("feature_names length must match background_data columns")
        self.feature_names = validated_features
        self.class_names = validated_classes

        self.threshold = self._strict_probability(threshold, name="threshold")
        self.delta = self._strict_probability(delta, name="delta")
        self.epsilon = self._bounded_epsilon(epsilon)
        self.beam_size = self._positive_integer(beam_size, name="beam_size")
        self.batch_size = self._positive_integer(batch_size, name="batch_size")
        self.max_samples = self._positive_integer(max_samples, name="max_samples")
        if max_anchor_size is not None:
            self.max_anchor_size = self._positive_integer(max_anchor_size, name="max_anchor_size")
        else:
            self.max_anchor_size = None
        if discretizer not in {"quartile", "decile"}:
            raise ValueError("discretizer must be 'quartile' or 'decile'")
        self.discretizer = discretizer
        if isinstance(random_state, bool) or not isinstance(random_state, Integral):
            raise TypeError("random_state must be an integer")
        if int(random_state) < 0 or int(random_state) > 2**32 - 1:
            raise ValueError("random_state must be between 0 and 2**32 - 1")
        self.random_state = int(random_state)
        self.cuts = self._compute_cuts()

    @staticmethod
    def _strict_probability(value: object, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 < numeric < 1.0:
            raise ValueError(f"{name} must be finite and strictly between 0 and 1")
        return numeric

    @staticmethod
    def _bounded_epsilon(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("epsilon must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric < 1.0:
            raise ValueError("epsilon must be finite and in [0, 1)")
        return numeric

    @staticmethod
    def _positive_integer(value: int, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _compute_cuts(self) -> Dict[int, np.ndarray]:
        percentiles = [25, 50, 75] if self.discretizer == "quartile" else list(range(10, 100, 10))
        return {
            feature_index: np.unique(
                np.percentile(self.background_data[:, feature_index], percentiles)
            ).astype(float)
            for feature_index in range(self.background_data.shape[1])
        }

    @staticmethod
    def _format_cut(value: float) -> str:
        if float(value) == 0.0:
            return "0"
        return np.format_float_positional(float(value), unique=True, trim="-")

    def _generate_predicates(self, instance: np.ndarray) -> List[_Predicate]:
        predicates: List[_Predicate] = []
        for feature_index, feature_name in enumerate(self.feature_names):
            seen_supports: set[bytes] = set()
            values = self.background_data[:, feature_index]
            for cut in self.cuts[feature_index]:
                operator: _Operator = "leq" if instance[feature_index] <= cut else "gt"
                support = values <= cut if operator == "leq" else values > cut
                support = np.asarray(support, dtype=bool)
                if not np.any(support) or np.all(support):
                    continue
                signature = support.tobytes()
                if signature in seen_supports:
                    continue
                seen_supports.add(signature)
                symbol = "<=" if operator == "leq" else ">"
                predicates.append(
                    _Predicate(
                        index=len(predicates),
                        feature_index=feature_index,
                        operator=operator,
                        cut=float(cut),
                        label=f"{feature_name} {symbol} {self._format_cut(float(cut))}",
                        background_support=support,
                    )
                )
        return predicates

    def _predict_class_scores(self, data: np.ndarray) -> np.ndarray:
        return normalize_classifier_outputs(
            self.model,
            np.array(data, copy=True),
            context="AnchorTabular",
            class_names=self.class_names,
            require_probabilities=False,
            allow_label_predictions=True,
        )

    def _rule_support(self, rule: _Rule, predicates: Sequence[_Predicate]) -> np.ndarray:
        support = np.ones(self.background_data.shape[0], dtype=bool)
        for predicate_index in rule:
            support &= predicates[predicate_index].background_support
        return support

    def _coverage(self, rule: _Rule, predicates: Sequence[_Predicate]) -> float:
        return float(np.mean(self._rule_support(rule, predicates)))

    def _feature_support(
        self,
        rule: _Rule,
        feature_index: int,
        predicates: Sequence[_Predicate],
    ) -> np.ndarray:
        support = np.ones(self.background_data.shape[0], dtype=bool)
        for predicate_index in rule:
            predicate = predicates[predicate_index]
            if predicate.feature_index == feature_index:
                support &= predicate.background_support
        return support

    def _is_redundant_extension(
        self,
        parent: _Rule,
        extension_index: int,
        predicates: Sequence[_Predicate],
    ) -> bool:
        feature_index = predicates[extension_index].feature_index
        old_support = self._feature_support(parent, feature_index, predicates)
        new_support = old_support & predicates[extension_index].background_support
        return np.array_equal(old_support, new_support)

    def _conditional_sample(
        self,
        rule: _Rule,
        predicates: Sequence[_Predicate],
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        compatible_indices = np.flatnonzero(self._rule_support(rule, predicates))
        if compatible_indices.size == 0:
            raise RuntimeError("candidate predicate conjunction has zero empirical support")
        row_indices = rng.choice(compatible_indices, size=n_samples, replace=True)
        samples = self.background_data[row_indices].copy()
        for predicate_index in rule:
            predicate = predicates[predicate_index]
            if not np.all(predicate.applies(samples[:, predicate.feature_index])):
                raise RuntimeError("conditional sampler violated an anchor predicate")
        return samples

    def _pull_candidate(
        self,
        rule: _Rule,
        predicates: Sequence[_Predicate],
        n_samples: int,
        target_index: int,
        rng: np.random.Generator,
        budget: _PredictionBudget,
    ) -> int:
        if n_samples < 1 or n_samples > budget.remaining:
            raise ValueError("candidate pull exceeds the remaining perturbation budget")
        samples = self._conditional_sample(rule, predicates, n_samples, rng)
        scores = self._predict_class_scores(samples)
        successes = int(np.sum(np.argmax(scores, axis=1) == target_index))
        budget.samples_used += n_samples
        budget.prediction_calls += 1
        return successes

    @staticmethod
    def _candidate_bounds(
        statistics: _CandidateStatistics, certification_delta: float
    ) -> Tuple[float, float]:
        time_index = max(1, statistics.samples)
        beta = compute_beta(1, time_index, certification_delta)
        return (
            kl_lower_bound(statistics.successes, statistics.samples, beta),
            kl_upper_bound(statistics.successes, statistics.samples, beta),
        )

    def _certify_candidate(
        self,
        rule: _Rule,
        statistics: _CandidateStatistics,
        predicates: Sequence[_Predicate],
        target_index: int,
        certification_delta: float,
        rng: np.random.Generator,
        budget: _PredictionBudget,
    ) -> Tuple[str, float, float]:
        while True:
            lower, upper = self._candidate_bounds(statistics, certification_delta)
            if lower > self.threshold:
                return "certified", lower, upper
            if upper < self.threshold:
                return "rejected", lower, upper
            if budget.remaining == 0:
                return "budget_exhausted", lower, upper
            pull_size = min(self.batch_size, budget.remaining)
            statistics.successes += self._pull_candidate(
                rule,
                predicates,
                pull_size,
                target_index,
                rng,
                budget,
            )
            statistics.samples += pull_size

    @staticmethod
    def _is_better_certified(
        candidate: Tuple[_Rule, float, float],
        incumbent: Optional[Tuple[_Rule, float, float]],
    ) -> bool:
        if incumbent is None:
            return True
        rule, precision, coverage = candidate
        old_rule, old_precision, old_coverage = incumbent
        return (-coverage, len(rule), -precision, rule) < (
            -old_coverage,
            len(old_rule),
            -old_precision,
            old_rule,
        )

    @staticmethod
    def _is_better_fallback(
        candidate: Tuple[_Rule, float, float],
        incumbent: Optional[Tuple[_Rule, float, float]],
    ) -> bool:
        if incumbent is None:
            return True
        rule, precision, coverage = candidate
        old_rule, old_precision, old_coverage = incumbent
        return (-precision, -coverage, len(rule), rule) < (
            -old_precision,
            -old_coverage,
            len(old_rule),
            old_rule,
        )

    @staticmethod
    def _hypothesis_count(predicate_count: int, max_depth: int) -> int:
        if max_depth == predicate_count:
            return 1 << predicate_count
        return sum(math.comb(predicate_count, depth) for depth in range(max_depth + 1))

    def _confidence_allocations(
        self, predicate_count: int, max_depth: int
    ) -> Tuple[float, float, int]:
        hypotheses = max(1, self._hypothesis_count(predicate_count, max_depth))
        log_certification_delta = math.log(self.delta) - math.log(2.0) - math.log(hypotheses)
        if log_certification_delta < math.log(math.ulp(0.0)):
            raise ValueError(
                "candidate confidence allocation is not representable; "
                "set a smaller max_anchor_size"
            )
        certification_delta = math.exp(log_certification_delta)
        selection_delta = self.delta / (2.0 * max(1, max_depth))
        if certification_delta == 0.0 or not math.isfinite(certification_delta):
            raise ValueError(
                "candidate confidence allocation is not representable; "
                "set a smaller max_anchor_size"
            )
        return certification_delta, selection_delta, hypotheses

    def _search(
        self,
        instance: np.ndarray,
        target_index: int,
        predicates: Sequence[_Predicate],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        predicate_count = len(predicates)
        max_depth = min(
            predicate_count,
            predicate_count if self.max_anchor_size is None else self.max_anchor_size,
        )
        certification_delta, selection_delta, hypothesis_count = self._confidence_allocations(
            predicate_count, max_depth
        )
        budget = _PredictionBudget(self.max_samples)
        candidate_statistics: Dict[_Rule, _CandidateStatistics] = {(): _CandidateStatistics()}
        candidate_coverage: Dict[_Rule, float] = {(): 1.0}
        certification_bounds: Dict[_Rule, Tuple[float, float]] = {}
        best_certified: Optional[Tuple[_Rule, float, float]] = None
        best_fallback: Optional[Tuple[_Rule, float, float]] = None
        all_lucb_converged = True
        budget_exhausted = False
        depth_reached = 0

        empty_status, empty_lower, empty_upper = self._certify_candidate(
            (),
            candidate_statistics[()],
            predicates,
            target_index,
            certification_delta,
            rng,
            budget,
        )
        certification_bounds[()] = (empty_lower, empty_upper)
        empty_mean = candidate_statistics[()].mean
        assert empty_mean is not None
        best_fallback = ((), empty_mean, 1.0)
        if empty_status == "certified":
            best_certified = ((), empty_mean, 1.0)
            return {
                "rule": (),
                "statistics": candidate_statistics[()],
                "bounds": certification_bounds[()],
                "coverage": 1.0,
                "is_certified": True,
                "budget": budget,
                "budget_exhausted": False,
                "bounded_beam_search_completed": True,
                "all_lucb_converged": True,
                "depth_reached": 0,
                "termination_reason": "certified_empty_anchor",
                "certification_delta": certification_delta,
                "selection_delta": selection_delta,
                "hypothesis_count": hypothesis_count,
            }
        if empty_status == "budget_exhausted":
            budget_exhausted = True

        beam: List[_Rule] = [()]
        generated: set[_Rule] = {()}
        beam_search_completed = not budget_exhausted

        for depth in range(1, max_depth + 1):
            if budget_exhausted:
                beam_search_completed = False
                break
            depth_reached = depth
            best_coverage = best_certified[2] if best_certified is not None else -1.0
            candidates: List[_Rule] = []
            for parent in beam:
                for predicate_index in range(predicate_count):
                    if predicate_index in parent:
                        continue
                    candidate = tuple(sorted((*parent, predicate_index)))
                    if candidate in generated:
                        continue
                    generated.add(candidate)
                    if self._is_redundant_extension(parent, predicate_index, predicates):
                        continue
                    coverage = self._coverage(candidate, predicates)
                    if coverage <= 0.0 or coverage <= best_coverage:
                        continue
                    candidates.append(candidate)
                    candidate_statistics[candidate] = _CandidateStatistics()
                    candidate_coverage[candidate] = coverage

            if not candidates:
                break

            def pull_arm(arm_index: int, n_samples: int) -> int:
                rule = candidates[arm_index]
                successes = self._pull_candidate(
                    rule,
                    predicates,
                    n_samples,
                    target_index,
                    rng,
                    budget,
                )
                return successes

            lucb_result = kl_lucb_top_m(
                pull_arm,
                len(candidates),
                min(self.beam_size, len(candidates)),
                max_samples=budget.remaining,
                epsilon=self.epsilon,
                delta=selection_delta,
                batch_size=self.batch_size,
                initial_successes=[candidate_statistics[rule].successes for rule in candidates],
                initial_samples=[candidate_statistics[rule].samples for rule in candidates],
            )
            all_lucb_converged &= lucb_result.converged
            for arm_statistics in lucb_result.statistics:
                statistics = candidate_statistics[candidates[arm_statistics.index]]
                statistics.successes = arm_statistics.successes
                statistics.samples = arm_statistics.samples

            if not lucb_result.selected_indices:
                budget_exhausted = lucb_result.budget_exhausted or budget.remaining == 0
                beam_search_completed = False
                break

            beam = [candidates[index] for index in lucb_result.selected_indices]
            if lucb_result.budget_exhausted:
                # A paired LUCB step is atomic, so one row can remain even when
                # the configured budget cannot support the next valid step.
                budget_exhausted = True
                beam_search_completed = False
            for rule in beam:
                statistics = candidate_statistics[rule]
                status, lower, upper = self._certify_candidate(
                    rule,
                    statistics,
                    predicates,
                    target_index,
                    certification_delta,
                    rng,
                    budget,
                )
                certification_bounds[rule] = (lower, upper)
                mean = statistics.mean
                assert mean is not None
                fallback = (rule, mean, candidate_coverage[rule])
                if self._is_better_fallback(fallback, best_fallback):
                    best_fallback = fallback
                if status == "certified" and self._is_better_certified(fallback, best_certified):
                    best_certified = fallback
                if status == "budget_exhausted":
                    budget_exhausted = True
                    beam_search_completed = False
                    break
            if budget_exhausted:
                break

        selected = best_certified if best_certified is not None else best_fallback
        assert selected is not None
        selected_rule, selected_mean, selected_coverage = selected
        selected_bounds = certification_bounds.get(selected_rule)
        if selected_bounds is None:
            selected_bounds = self._candidate_bounds(
                candidate_statistics[selected_rule], certification_delta
            )
        is_certified = best_certified is not None and selected_rule == best_certified[0]
        if budget_exhausted:
            reason = "sample_budget_exhausted"
        elif is_certified:
            reason = "bounded_beam_search_completed_with_certified_anchor"
        else:
            reason = "bounded_beam_search_completed_without_certified_anchor"
        return {
            "rule": selected_rule,
            "statistics": candidate_statistics[selected_rule],
            "bounds": selected_bounds,
            "coverage": selected_coverage,
            "is_certified": is_certified,
            "budget": budget,
            "budget_exhausted": budget_exhausted,
            "bounded_beam_search_completed": beam_search_completed,
            "all_lucb_converged": all_lucb_converged,
            "depth_reached": depth_reached,
            "termination_reason": reason,
            "certification_delta": certification_delta,
            "selection_delta": selection_delta,
            "hypothesis_count": hypothesis_count,
            "selected_mean": selected_mean,
        }

    def explain(self, instance: np.ndarray, **kwargs: Any) -> Explanation:
        """Explain the model's fixed predicted output for exactly one row."""

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        query = np.array(
            validate_single_tabular_instance(
                instance,
                len(self.feature_names),
                dtype=float,
                require_finite=True,
            ),
            dtype=float,
            copy=True,
        )
        rng = np.random.default_rng(self.random_state)
        query_scores = self._predict_class_scores(query.reshape(1, -1))
        target_index = int(np.argmax(query_scores[0]))
        target_name = self.class_names[target_index]
        predicates = self._generate_predicates(query)
        search = self._search(query, target_index, predicates, rng)

        rule: _Rule = search["rule"]
        statistics: _CandidateStatistics = search["statistics"]
        lower, upper = search["bounds"]
        precision = statistics.mean
        if precision is None:
            precision = 0.0
        is_certified = bool(search["is_certified"] and lower > self.threshold)
        rule_conditions = [
            predicates[index].payload(self.feature_names[predicates[index].feature_index])
            for index in rule
        ]
        anchor_feature_indices = sorted({predicates[index].feature_index for index in rule})
        budget: _PredictionBudget = search["budget"]

        return Explanation(
            explainer_name="AnchorTabular",
            target_class=target_name,
            explanation_data={
                "rules": [condition["label"] for condition in rule_conditions],
                "rule_conditions": rule_conditions,
                "anchor_predicate_indices": list(rule),
                "anchor_feature_indices": anchor_feature_indices,
                "anchor_features": [self.feature_names[index] for index in anchor_feature_indices],
                "precision": float(precision),
                "empirical_precision": float(precision),
                "precision_lower_bound": float(lower),
                "precision_upper_bound": float(upper),
                "precision_sample_count": int(statistics.samples),
                "precision_success_count": int(statistics.successes),
                "coverage": float(search["coverage"]),
                "empirical_coverage": float(search["coverage"]),
                "precision_threshold": float(self.threshold),
                "delta": float(self.delta),
                "epsilon": float(self.epsilon),
                "effective_certification_delta_per_candidate": float(search["certification_delta"]),
                "effective_selection_delta_per_depth": float(search["selection_delta"]),
                "confidence_hypothesis_count": int(search["hypothesis_count"]),
                "is_certified_anchor": is_certified,
                "provides_high_probability_precision_guarantee": is_certified,
                "confidence_guarantee_scope": (
                    "Sequential Bernoulli KL bounds with a Bonferroni split of the requested "
                    "delta across beam depths and finite candidate hypotheses; requires IID "
                    "draws from the declared uniform empirical-joint distribution restricted "
                    "to rows satisfying the candidate rule and fixed deterministic model "
                    "predictions (or independent stationary prediction randomness)."
                ),
                "search_method": "paper_algorithms_1_2_kl_lucb_beam_search",
                "beam_size": int(self.beam_size),
                "batch_size": int(self.batch_size),
                "max_samples": int(self.max_samples),
                "max_anchor_size": self.max_anchor_size,
                "effective_max_anchor_size": int(
                    min(
                        len(predicates),
                        len(predicates) if self.max_anchor_size is None else self.max_anchor_size,
                    )
                ),
                "candidate_predicate_count": int(len(predicates)),
                "discretizer": self.discretizer,
                "background_row_count": int(self.background_data.shape[0]),
                "random_state": int(self.random_state),
                "perturbation_prediction_rows": int(budget.samples_used),
                "query_prediction_rows": 1,
                "total_prediction_rows": int(budget.samples_used + 1),
                "prediction_calls": int(budget.prediction_calls + 1),
                "budget_exhausted": bool(search["budget_exhausted"]),
                "budget_exhaustion_semantics": (
                    "the remaining perturbation-row budget could not support the next "
                    "required certification batch or atomic paired KL-LUCB pull"
                ),
                "bounded_beam_search_completed": bool(search["bounded_beam_search_completed"]),
                "bounded_search_scope": (
                    "bottom-up beam search limited by beam_size, max_anchor_size, finite "
                    "query-consistent predicates, and max_samples"
                ),
                "all_lucb_stages_converged": bool(search["all_lucb_converged"]),
                "search_depth_reached": int(search["depth_reached"]),
                "termination_reason": search["termination_reason"],
                "perturbation_distribution": (
                    "uniform_empirical_joint_background_conditioned_by_row_restriction"
                ),
                "conditional_sampling_semantics": (
                    "whole_background_rows_only; feature_dependence_preserved; no_column_repair"
                ),
                "coverage_semantics": "exact_fraction_of_empirical_background_rows",
                "target_output_index": int(target_index),
                "model_prediction": query_scores[0].copy(),
                "feature_attributions": {
                    name: float(index in anchor_feature_indices)
                    for index, name in enumerate(self.feature_names)
                },
                "feature_attribution_semantics": "anchor_membership_indicator",
                "causal_claim": False,
                "globally_maximum_coverage_claim": False,
            },
            feature_names=list(self.feature_names),
        )
