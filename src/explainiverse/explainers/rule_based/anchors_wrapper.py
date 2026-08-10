# src/explainiverse/explainers/rule_based/anchors_wrapper.py
"""Fixed-sample, approximate Anchors-style rule explanations.

This implementation discretizes tabular features and uses a bounded beam
search with fixed-size Monte Carlo precision estimates.  It is inspired by
Anchors, but it does *not* implement the paper's KL-LUCB sampling procedure or
its high-probability precision guarantee.  Results therefore report empirical
precision and empirical training-set coverage explicitly.

Reference for the canonical algorithm:
    Ribeiro, M.T., Singh, S., & Guestrin, C. (2018). Anchors: High-Precision
    Model-Agnostic Explanations. AAAI 2018.
"""

from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Set, Tuple

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


class AnchorsExplainer(BaseExplainer):
    """Approximate Anchors-style explainer for tabular classifiers.

    ``AnchorsExplainer`` is retained as the public class name for backwards
    compatibility.  The returned explanation is named
    ``"ApproximateAnchors"`` so it cannot be mistaken for the canonical
    confidence-bound algorithm.

    It generates if-then rules whose precision is estimated from a fixed
    number of perturbation samples.  Beam search keeps the most precise
    candidates and, among all sampled candidates that meet ``threshold``,
    returns the one with greatest empirical coverage.  Coverage ties prefer
    the shorter rule, then the higher empirical precision.

    Because this is a fixed-sample point estimate and a bounded beam search,
    it neither certifies the precision threshold nor guarantees the globally
    maximum-coverage rule.

    The algorithm:
    1. Discretizes continuous features into empirical bins
    2. Estimates candidate precision from ``n_samples`` perturbations
    3. Uses beam search to explore feature subsets
    4. Selects the sampled threshold-meeting candidate with highest empirical
       coverage

    Attributes:
        model: Model adapter with .predict() method
        training_data: Reference data for generating perturbations
        feature_names: List of feature names
        class_names: List of class names
        threshold: Minimum empirical precision estimate (default: 0.95)
        n_samples: Fixed number of samples per precision estimate (default: 1000)
        beam_size: Number of candidates in beam search (default: 4)
    """

    training_data: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    threshold: float
    n_samples: int
    beam_size: int
    max_anchor_size: int
    discretizer: str
    random_state: int
    rng: np.random.RandomState
    bins: Dict[int, np.ndarray]
    bin_labels: Dict[int, List[str]]

    def __init__(
        self,
        model,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        threshold: float = 0.95,
        n_samples: int = 1000,
        beam_size: int = 4,
        max_anchor_size: Optional[int] = None,
        discretizer: str = "quartile",
        random_state: int = 42,
    ) -> None:
        """
        Initialize the approximate Anchors-style explainer.

        Args:
            model: Model adapter with .predict() method
            training_data: Reference data (n_samples, n_features)
            feature_names: List of feature names
            class_names: List of class names
            threshold: Minimum empirical precision estimate for a candidate
            n_samples: Fixed number of perturbation samples per candidate
            beam_size: Number of candidates to keep in beam search
            max_anchor_size: Maximum number of conditions in anchor
            discretizer: How to discretize continuous features ("quartile", "decile")
            random_state: Random seed
        """
        super().__init__(model)
        ensure_classification_task(model, context="Approximate Anchors")
        self.training_data = as_real_array(
            training_data,
            name="training_data",
            dtype=float,
            require_finite=True,
        )
        validated_features = validate_name_sequence(feature_names, name="feature_names")
        validated_classes = validate_name_sequence(class_names, name="class_names")
        assert validated_features is not None and validated_classes is not None
        self.feature_names = validated_features
        self.class_names = validated_classes
        if self.training_data.ndim != 2 or self.training_data.shape[0] == 0:
            raise ValueError("training_data must be a non-empty 2D array")
        if self.training_data.shape[1] == 0:
            raise ValueError("training_data must contain at least one feature")
        if len(self.feature_names) != self.training_data.shape[1]:
            raise ValueError("feature_names length must match training_data columns")
        if not isinstance(threshold, Real) or isinstance(threshold, bool):
            raise TypeError("threshold must be a real number")
        if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be finite and between 0 and 1")
        for value, name in ((n_samples, "n_samples"), (beam_size, "beam_size")):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_anchor_size is not None and (
            not isinstance(max_anchor_size, Integral)
            or isinstance(max_anchor_size, bool)
            or max_anchor_size < 1
        ):
            raise ValueError("max_anchor_size must be a positive integer or None")
        if discretizer not in {"quartile", "decile"}:
            raise ValueError("discretizer must be 'quartile' or 'decile'")
        if not isinstance(random_state, Integral) or isinstance(random_state, bool):
            raise TypeError("random_state must be an integer")
        if random_state < 0 or random_state > 2**32 - 1:
            raise ValueError("random_state must be between 0 and 2**32 - 1")

        self.threshold = float(threshold)
        self.n_samples = int(n_samples)
        self.beam_size = int(beam_size)
        self.max_anchor_size = (
            len(self.feature_names) if max_anchor_size is None else int(max_anchor_size)
        )
        self.discretizer = discretizer
        self.random_state = int(random_state)
        self.rng = np.random.RandomState(self.random_state)

        # Pre-compute feature statistics for discretization
        self._compute_discretization()

    def _compute_discretization(self) -> None:
        """Pre-compute discretization bins for each feature."""
        self.bins = {}
        self.bin_labels = {}

        if self.discretizer == "quartile":
            percentiles = [25, 50, 75]
        elif self.discretizer == "decile":
            percentiles = list(range(10, 100, 10))
        for idx in range(self.training_data.shape[1]):
            values = self.training_data[:, idx]
            bins = np.percentile(values, percentiles)
            bins = np.unique(bins)  # Remove duplicates
            self.bins[idx] = bins

            self.bin_labels[idx] = [
                self._format_condition_label(idx, bin_idx) for bin_idx in range(len(bins) + 1)
            ]

    @staticmethod
    def _format_threshold(value: float) -> str:
        """Use a shortest round-trippable representation of an exact boundary."""
        value = float(value)
        if value == 0.0:
            return "0"
        return np.format_float_positional(value, unique=True, trim="-")

    def _format_condition_label(self, feature_idx: int, bin_idx: int) -> str:
        name = self.feature_names[feature_idx]
        bins = self.bins[feature_idx]
        if len(bins) == 0:
            return f"{name} = any"
        if bin_idx == 0:
            return f"{name} <= {self._format_threshold(bins[0])}"
        if bin_idx == len(bins):
            return f"{name} > {self._format_threshold(bins[-1])}"
        lower = self._format_threshold(bins[bin_idx - 1])
        upper = self._format_threshold(bins[bin_idx])
        return f"{lower} < {name} <= {upper}"

    def _condition_payload(self, feature_idx: int, bin_idx: int) -> Dict[str, Any]:
        """Return exact machine-readable bounds alongside the display rule."""
        bins = self.bins[feature_idx]
        lower = float(bins[bin_idx - 1]) if bin_idx > 0 and len(bins) else None
        upper = float(bins[bin_idx]) if bin_idx < len(bins) else None
        return {
            "feature": self.feature_names[feature_idx],
            "feature_index": int(feature_idx),
            "bin_index": int(bin_idx),
            "lower_bound": lower,
            "upper_bound": upper,
            "lower_inclusive": False if lower is not None else None,
            "upper_inclusive": True if upper is not None else None,
            "label": self._format_condition_label(feature_idx, bin_idx),
        }

    def _discretize_value(self, value: float, feature_idx: int) -> int:
        """Discretize a single value into a bin index."""
        bins = self.bins[feature_idx]
        if len(bins) == 0:
            return 0
        return int(np.searchsorted(bins, value))

    def _discretize_instance(self, instance: np.ndarray) -> np.ndarray:
        """Discretize an entire instance."""
        return np.array(
            [self._discretize_value(float(instance[i]), i) for i in range(len(instance))]
        )

    def _get_condition_label(self, feature_idx: int, bin_idx: int) -> str:
        """Get human-readable label for a condition."""
        labels = self.bin_labels[feature_idx]
        if bin_idx < len(labels):
            return labels[bin_idx]
        return f"{self.feature_names[feature_idx]} in bin {bin_idx}"

    def _generate_perturbations(
        self, instance: np.ndarray, anchor: List[int], n_samples: int
    ) -> np.ndarray:
        """
        Generate perturbation samples that respect the anchor conditions.

        Args:
            instance: Original instance
            anchor: List of feature indices that are fixed
            n_samples: Number of samples to generate

        Returns:
            Array of perturbed samples
        """
        perturbations = np.zeros((n_samples, len(instance)))

        # Discretize the instance
        disc_instance = self._discretize_instance(instance)

        for i in range(n_samples):
            # Start with random sample from training data
            sample_idx = self.rng.randint(len(self.training_data))
            sample = self.training_data[sample_idx].copy()

            # Fix anchor features to match the instance's bin
            for feat_idx in anchor:
                # Find values in training data that fall in the same bin
                target_bin = disc_instance[feat_idx]
                bins = self.bins[feat_idx]

                # Get values in the same bin from training data
                if len(bins) == 0:
                    # Use original value if no bins
                    sample[feat_idx] = instance[feat_idx]
                else:
                    # Sample from values in the same bin
                    feature_values = self.training_data[:, feat_idx]
                    in_bin = np.array(
                        [self._discretize_value(v, feat_idx) == target_bin for v in feature_values]
                    )
                    if np.any(in_bin):
                        valid_values = feature_values[in_bin]
                        sample[feat_idx] = self.rng.choice(valid_values)
                    else:
                        sample[feat_idx] = instance[feat_idx]

            perturbations[i] = sample

        return perturbations

    def _compute_precision(
        self, instance: np.ndarray, anchor: List[int], target_class: int
    ) -> Tuple[float, int]:
        """
        Compute a fixed-sample empirical precision estimate.

        Precision = P(prediction = target_class | anchor conditions hold)

        Returns:
            Tuple of (empirical precision, matching prediction count)
        """
        perturbations = self._generate_perturbations(instance, anchor, self.n_samples)
        predictions = self._predict_class_scores(perturbations)
        pred_classes = np.argmax(predictions, axis=1)

        matches = int(np.sum(pred_classes == target_class))
        precision = matches / len(pred_classes)

        return precision, matches

    def _predict_class_scores(self, X: np.ndarray) -> np.ndarray:
        """Return validated class columns under the Anchors score contract."""

        return normalize_classifier_outputs(
            self.model,
            np.asarray(X),
            context="Approximate Anchors",
            class_names=self.class_names,
            require_probabilities=False,
            allow_label_predictions=True,
        )

    def _compute_coverage(self, anchor: List[int], instance: np.ndarray) -> float:
        """
        Compute empirical coverage (fraction of training rows matching conditions).
        """
        disc_instance = self._discretize_instance(instance)

        matches = 0
        for sample in self.training_data:
            disc_sample = self._discretize_instance(sample)
            if all(disc_sample[i] == disc_instance[i] for i in anchor):
                matches += 1

        return matches / len(self.training_data)

    def _beam_search(
        self, instance: np.ndarray, target_class: int
    ) -> Tuple[List[int], float, float]:
        """
        Use bounded beam search to find the best sampled candidate.

        Candidates meeting the empirical precision threshold are ranked first
        by empirical coverage, then by shorter length, then by empirical
        precision.  If none meets the threshold, the best-effort candidate is
        ranked by precision, coverage, and shorter length, in that order.

        Returns:
            Tuple of (candidate features, empirical precision, empirical coverage)
        """
        n_features = len(instance)

        def is_better_valid(candidate, incumbent):
            if incumbent is None:
                return True
            anchor, precision, coverage = candidate
            old_anchor, old_precision, old_coverage = incumbent
            if coverage != old_coverage:
                return coverage > old_coverage
            if len(anchor) != len(old_anchor):
                return len(anchor) < len(old_anchor)
            if precision != old_precision:
                return precision > old_precision
            return tuple(anchor) < tuple(old_anchor)

        def is_better_fallback(candidate, incumbent):
            anchor, precision, coverage = candidate
            old_anchor, old_precision, old_coverage = incumbent
            if precision != old_precision:
                return precision > old_precision
            if coverage != old_coverage:
                return coverage > old_coverage
            if len(anchor) != len(old_anchor):
                return len(anchor) < len(old_anchor)
            return tuple(anchor) < tuple(old_anchor)

        # The empty rule is a real candidate: if the model already predicts
        # the target class often enough under the perturbation distribution,
        # no condition can improve on its coverage of one.
        empty_precision, _ = self._compute_precision(instance, [], target_class)
        empty_candidate: Tuple[List[int], float, float] = ([], empty_precision, 1.0)
        best_fallback = empty_candidate
        best_valid = empty_candidate if empty_precision >= self.threshold else None

        if best_valid is not None:
            return best_valid

        candidates = [empty_candidate]
        evaluated: Set[Tuple[int, ...]] = {()}
        max_depth = min(self.max_anchor_size, n_features)

        for _ in range(max_depth):
            new_candidates = []

            for anchor, _, _ in candidates:
                for feat_idx in range(n_features):
                    if feat_idx in anchor:
                        continue

                    # Canonical ordering prevents evaluating permutations such
                    # as [0, 1] and [1, 0] as separate rules.
                    new_anchor_tuple = tuple(sorted((*anchor, feat_idx)))
                    if new_anchor_tuple in evaluated:
                        continue
                    evaluated.add(new_anchor_tuple)
                    new_anchor = list(new_anchor_tuple)

                    precision, _ = self._compute_precision(instance, new_anchor, target_class)
                    coverage = self._compute_coverage(new_anchor, instance)
                    candidate = (new_anchor, precision, coverage)
                    new_candidates.append(candidate)

                    if is_better_fallback(candidate, best_fallback):
                        best_fallback = candidate
                    if precision >= self.threshold and is_better_valid(candidate, best_valid):
                        best_valid = candidate

            if not new_candidates:
                break

            # Beam membership is an approximation: prioritize precision, then
            # empirical coverage, with deterministic ordering for exact ties.
            new_candidates.sort(
                key=lambda candidate: (
                    -candidate[1],
                    -candidate[2],
                    len(candidate[0]),
                    tuple(candidate[0]),
                )
            )
            candidates = new_candidates[: self.beam_size]

        return best_valid if best_valid is not None else best_fallback

    def explain(self, instance: np.ndarray, **kwargs: Any) -> Explanation:
        """
        Generate an approximate Anchors-style explanation for an instance.

        Args:
            instance: The instance to explain (1D array)

        Returns:
            Explanation object with candidate rules and empirical diagnostics
        """
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        instance = validate_single_tabular_instance(
            instance,
            len(self.feature_names),
            dtype=float,
            require_finite=True,
        )

        # Make repeated calls reproducible for a fixed random_state.  The
        # estimate remains Monte Carlo and does not become a confidence bound.
        self.rng = np.random.RandomState(self.random_state)

        # Get the model's prediction
        predictions = self._predict_class_scores(instance.reshape(1, -1))
        target_class = int(np.argmax(predictions[0]))

        target_name = (
            self.class_names[target_class]
            if target_class < len(self.class_names)
            else f"class_{target_class}"
        )

        # Find anchor using beam search
        anchor_features, precision, coverage = self._beam_search(instance, target_class)

        # Convert to human-readable rules
        disc_instance = self._discretize_instance(instance)
        rule_conditions = [
            self._condition_payload(feat_idx, int(disc_instance[feat_idx]))
            for feat_idx in anchor_features
        ]
        rules = [condition["label"] for condition in rule_conditions]

        meets_threshold = precision >= self.threshold

        return Explanation(
            explainer_name="ApproximateAnchors",
            target_class=target_name,
            explanation_data={
                "rules": rules,
                "rule_conditions": rule_conditions,
                "precision": float(precision),
                "empirical_precision": float(precision),
                "coverage": float(coverage),
                "empirical_coverage": float(coverage),
                "precision_threshold": float(self.threshold),
                "meets_empirical_precision_threshold": bool(meets_threshold),
                "precision_sample_size": int(self.n_samples),
                "search_method": "fixed_sample_beam_search",
                "provides_high_probability_guarantee": False,
                "anchor_features": [self.feature_names[i] for i in anchor_features],
                "anchor_indices": anchor_features,
                # These values denote rule membership only. The bounded beam
                # search does not estimate a feature-importance ordering.
                "feature_attributions": {self.feature_names[i]: 1.0 for i in anchor_features},
                "feature_attribution_semantics": "anchor_membership_indicator",
            },
        )
