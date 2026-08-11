"""Constrained, gradient-free counterfactual search for tabular classifiers.

The public class name is retained for compatibility, but this implementation
does not claim to implement the DiCE optimization algorithm.  It performs a
deterministic multi-start search, projects candidates onto the declared tabular
domain, and returns only candidates whose predicted class is the requested
target.

Counterfactual explanations answer: "Which feasible feature changes would
change this classifier's prediction?"
"""

from numbers import Integral, Real
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import (
    as_real_array,
    ensure_classification_task,
    normalize_classifier_outputs,
    validate_name_sequence,
    validate_single_tabular_instance,
)


class CounterfactualExplainer(BaseExplainer):
    """Deterministic constrained search for tabular counterfactual examples.

    Continuous features are optimized inside their configured ranges.
    Categorical features can only take values observed in ``training_data``.
    Features omitted from both declarations remain fixed to the query value.
    The method requires classifier probabilities, either as ``(n, C)`` or a
    one-column/one-dimensional binary positive-class probability. Custom
    adapters whose one-dimensional probabilities can equal hard-label values
    such as 0 or 1 should declare
    ``prediction_output_kind = "probabilities"``; without that marker the two
    representations are mathematically indistinguishable.
    """

    training_data: np.ndarray
    feature_names: List[str]
    continuous_features: List[str]
    categorical_features: List[str]
    fixed_features: List[str]
    proximity_weight: float
    diversity_weight: float
    random_state: int
    feature_ranges: Dict[str, tuple]
    _name_to_index: Dict[str, int]
    _continuous_indices: np.ndarray
    _categorical_indices: np.ndarray
    _fixed_indices: np.ndarray
    _categorical_values: Dict[int, np.ndarray]
    scales: np.ndarray

    def __init__(
        self,
        model,
        training_data: np.ndarray,
        feature_names: List[str],
        continuous_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        feature_ranges: Optional[Dict[str, tuple]] = None,
        proximity_weight: float = 0.5,
        diversity_weight: float = 0.5,
        random_state: int = 42,
    ) -> None:
        super().__init__(model)
        ensure_classification_task(model, context="Counterfactual search")

        data = as_real_array(
            training_data,
            name="training_data",
            dtype=float,
            require_finite=True,
        )
        validated_names = validate_name_sequence(feature_names, name="feature_names")
        assert validated_names is not None
        names = validated_names
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("training_data must be a non-empty 2D array")
        if not names or len(names) != data.shape[1]:
            raise ValueError("feature_names length must match the training_data columns")
        for value, name in (
            (proximity_weight, "proximity_weight"),
            (diversity_weight, "diversity_weight"),
        ):
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(random_state, Integral) or isinstance(random_state, bool):
            raise TypeError("random_state must be an integer")
        if random_state < 0 or random_state > 2**32 - 1:
            raise ValueError("random_state must be between 0 and 2**32 - 1")

        categorical = [] if categorical_features is None else list(categorical_features)
        if continuous_features is None:
            continuous = [name for name in names if name not in categorical]
        else:
            continuous = list(continuous_features)
        for declaration, label in (
            (continuous, "continuous_features"),
            (categorical, "categorical_features"),
        ):
            if len(set(declaration)) != len(declaration):
                raise ValueError(f"{label} must not contain duplicates")
            unknown = set(declaration).difference(names)
            if unknown:
                raise ValueError(f"{label} contains unknown features: {sorted(unknown)}")
        overlap = set(continuous).intersection(categorical)
        if overlap:
            raise ValueError(
                "continuous_features and categorical_features overlap: " f"{sorted(overlap)}"
            )

        self.training_data = data
        self.feature_names = names
        self.continuous_features = continuous
        self.categorical_features = categorical
        declared = set(continuous).union(categorical)
        self.fixed_features = [name for name in names if name not in declared]
        self.proximity_weight = float(proximity_weight)
        self.diversity_weight = float(diversity_weight)
        self.random_state = int(random_state)

        ranges = {
            name: (float(np.min(data[:, index])), float(np.max(data[:, index])))
            for index, name in enumerate(names)
        }
        if feature_ranges is not None:
            unknown = set(feature_ranges).difference(names)
            if unknown:
                raise ValueError(f"feature_ranges contains unknown features: {sorted(unknown)}")
            ranges.update(feature_ranges)

        self.feature_ranges = {}
        for index, name in enumerate(names):
            bounds = ranges[name]
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"feature_ranges[{name!r}] must be a (min, max) pair")
            lower, upper = map(float, bounds)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError(f"Invalid range for feature {name!r}: {bounds!r}")
            if np.any(data[:, index] < lower) or np.any(data[:, index] > upper):
                raise ValueError(f"training_data for {name!r} falls outside its feature range")
            self.feature_ranges[name] = (lower, upper)

        self._name_to_index = {name: index for index, name in enumerate(names)}
        self._continuous_indices = np.array(
            [self._name_to_index[name] for name in continuous], dtype=int
        )
        self._categorical_indices = np.array(
            [self._name_to_index[name] for name in categorical], dtype=int
        )
        self._fixed_indices = np.array(
            [self._name_to_index[name] for name in self.fixed_features], dtype=int
        )
        self._categorical_values = {
            index: np.unique(data[:, index]) for index in self._categorical_indices
        }
        self._compute_scales()

        # Fail at construction if the model does not expose the score contract
        # required by the optimization objective.
        self._predict_probabilities(data[:1])

    def _compute_scales(self) -> None:
        """Compute non-zero range scales used by proximity and diversity."""
        scales = []
        for name in self.feature_names:
            lower, upper = self.feature_ranges[name]
            width = upper - lower
            scales.append(width if width > 0 else 1.0)
        self.scales = np.asarray(scales, dtype=float)

    def _predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Normalize the supported classifier probability representations."""
        matrix = as_real_array(
            X,
            name="model inputs",
            dtype=float,
            require_finite=True,
        )
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return normalize_classifier_outputs(
            self.model,
            matrix,
            context="Counterfactual search",
            class_names=getattr(self.model, "class_names", None),
            require_probabilities=True,
            allow_label_predictions=False,
        )

    def _get_target_class(self, instance: np.ndarray, desired_class: Optional[int] = None) -> int:
        """Resolve an output-column target from the original prediction."""
        probabilities = self._predict_probabilities(instance.reshape(1, -1))[0]
        current_class = int(np.argmax(probabilities))
        n_classes = len(probabilities)

        if desired_class is not None:
            if not isinstance(desired_class, Integral) or isinstance(desired_class, bool):
                raise TypeError("desired_class must be an integer output index")
            target = int(desired_class)
            if target < 0 or target >= n_classes:
                raise ValueError(f"desired_class must be between 0 and {n_classes - 1}")
            if target == current_class:
                raise ValueError("desired_class must differ from the original class")
            return target

        order = np.argsort(probabilities, kind="stable")[::-1]
        return int(next(index for index in order if index != current_class))

    def _proximity_loss(self, candidate: np.ndarray, original: np.ndarray) -> float:
        diff = (candidate - original) / self.scales
        return float(np.sum(diff**2))

    def _validity_loss(self, candidate: np.ndarray, target_class: int) -> float:
        probability = self._predict_probabilities(candidate.reshape(1, -1))[0, target_class]
        return float(-np.log(max(float(probability), 1e-12)))

    def _diversity_loss(self, candidates: List[np.ndarray]) -> float:
        """Negative mean pairwise normalized squared distance."""
        if len(candidates) < 2:
            return 0.0
        distances = []
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                diff = (candidates[left] - candidates[right]) / self.scales
                distances.append(float(np.sum(diff**2)))
        return -float(np.mean(distances))

    def _project(self, candidate: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Project a candidate onto ranges, categories, and fixed features."""
        projected = np.asarray(candidate, dtype=float).copy()
        for index, name in enumerate(self.feature_names):
            lower, upper = self.feature_ranges[name]
            projected[index] = np.clip(projected[index], lower, upper)
        for index in self._categorical_indices:
            allowed = self._categorical_values[int(index)]
            projected[index] = allowed[np.argmin(np.abs(allowed - projected[index]))]
        if self._fixed_indices.size:
            projected[self._fixed_indices] = original[self._fixed_indices]
        return projected

    def _validate_query_domain(self, query: np.ndarray) -> None:
        """Reject a query that is outside the declared feasible domain."""
        for index, name in enumerate(self.feature_names):
            lower, upper = self.feature_ranges[name]
            if query[index] < lower or query[index] > upper:
                raise ValueError(f"instance value for {name!r} is outside its feature range")
        for index in self._categorical_indices:
            allowed = self._categorical_values[int(index)]
            if not np.any(np.isclose(query[index], allowed, atol=1e-12, rtol=0.0)):
                name = self.feature_names[int(index)]
                raise ValueError(
                    f"instance value for categorical feature {name!r} was not "
                    "observed in training_data"
                )

    def _is_target(self, candidate: np.ndarray, target_class: int) -> bool:
        probabilities = self._predict_probabilities(candidate.reshape(1, -1))[0]
        return int(np.argmax(probabilities)) == target_class

    def _refine_toward_original(
        self,
        candidate: np.ndarray,
        original: np.ndarray,
        target_class: int,
        steps: int = 30,
    ) -> np.ndarray:
        """Binary-search continuous changes while retaining feasibility."""
        high = self._project(candidate, original)
        low = high.copy()
        if self._continuous_indices.size:
            low[self._continuous_indices] = original[self._continuous_indices]
        low = self._project(low, original)
        if self._is_target(low, target_class):
            return low

        for _ in range(steps):
            middle = high.copy()
            middle[self._continuous_indices] = (
                low[self._continuous_indices] + high[self._continuous_indices]
            ) / 2.0
            middle = self._project(middle, original)
            if self._is_target(middle, target_class):
                high = middle
            else:
                low = middle
        return high

    def _candidate_seeds(
        self,
        original: np.ndarray,
        target_class: int,
        max_attempts: int,
        rng: np.random.Generator,
    ):
        """Yield deterministic target exemplars followed by domain samples."""
        training_classes = np.argmax(self._predict_probabilities(self.training_data), axis=1)
        target_rows = self.training_data[training_classes == target_class]
        if len(target_rows):
            order = np.argsort(
                [
                    self._proximity_loss(self._project(row, original), original)
                    for row in target_rows
                ],
                kind="stable",
            )
            target_rows = target_rows[order]

        for attempt in range(max_attempts):
            if attempt < len(target_rows):
                seed = target_rows[attempt].copy()
            elif len(target_rows):
                seed = target_rows[int(rng.integers(len(target_rows)))].copy()
                if self._continuous_indices.size:
                    seed[self._continuous_indices] += rng.normal(
                        0.0,
                        0.15 * self.scales[self._continuous_indices],
                    )
            else:
                seed = original.copy()
                for index in self._continuous_indices:
                    lower, upper = self.feature_ranges[self.feature_names[int(index)]]
                    seed[index] = rng.uniform(lower, upper)
                for index in self._categorical_indices:
                    allowed = self._categorical_values[int(index)]
                    seed[index] = allowed[int(rng.integers(len(allowed)))]
            yield self._project(seed, original)

    def _optimize_seed(
        self,
        seed: np.ndarray,
        original: np.ndarray,
        target_class: int,
        existing: List[np.ndarray],
        max_iter: int,
    ) -> np.ndarray:
        if not self._continuous_indices.size:
            return seed

        indices = self._continuous_indices
        bounds = [self.feature_ranges[self.feature_names[int(index)]] for index in indices]

        def assemble(values: np.ndarray) -> np.ndarray:
            candidate = seed.copy()
            candidate[indices] = values
            return self._project(candidate, original)

        def objective(values: np.ndarray) -> float:
            candidate = assemble(values)
            loss = self._validity_loss(candidate, target_class)
            loss += self.proximity_weight * self._proximity_loss(candidate, original)
            if existing and self.diversity_weight:
                distances = []
                for prior in existing:
                    diff = (candidate - prior) / self.scales
                    distances.append(min(float(np.sum(diff**2)), 1.0))
                loss -= self.diversity_weight * float(np.mean(distances))
            return float(loss)

        result = minimize(
            objective,
            seed[indices],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter)},
        )
        return assemble(result.x)

    def _generate_diverse_counterfactuals(
        self,
        instance: np.ndarray,
        target_class: int,
        num_counterfactuals: int,
        max_attempts: int = 50,
        max_iter: int = 100,
    ) -> List[np.ndarray]:
        """Generate only feasible, target-valid, non-duplicate candidates."""
        rng = np.random.default_rng(self.random_state)
        counterfactuals: List[np.ndarray] = []
        for seed in self._candidate_seeds(instance, target_class, max_attempts, rng):
            optimized = self._optimize_seed(
                seed,
                instance,
                target_class,
                counterfactuals,
                max_iter,
            )
            options = (optimized, seed)
            valid = next(
                (option for option in options if self._is_target(option, target_class)),
                None,
            )
            if valid is None:
                continue
            valid = self._refine_toward_original(valid, instance, target_class)
            if not self._is_target(valid, target_class):
                continue
            if any(np.allclose(valid, prior, atol=1e-7, rtol=0.0) for prior in counterfactuals):
                continue
            counterfactuals.append(valid)
            if len(counterfactuals) == num_counterfactuals:
                break
        return counterfactuals

    def _target_name(self, target_class: int) -> str:
        class_names = getattr(self.model, "class_names", None)
        if class_names is not None and target_class < len(class_names):
            return str(class_names[target_class])
        return f"class_{target_class}"

    def explain(
        self,
        instance: np.ndarray,
        num_counterfactuals: int = 3,
        desired_class: Optional[int] = None,
        **kwargs: Any,
    ) -> Explanation:
        """Search for feasible counterfactuals in the requested output class."""
        query = validate_single_tabular_instance(
            instance,
            len(self.feature_names),
            dtype=float,
            require_finite=True,
        )
        self._validate_query_domain(query)
        if not isinstance(num_counterfactuals, Integral) or isinstance(num_counterfactuals, bool):
            raise TypeError("num_counterfactuals must be an integer")
        if num_counterfactuals < 1:
            raise ValueError("num_counterfactuals must be at least 1")

        max_attempts = kwargs.pop("max_attempts", 50)
        max_iter = kwargs.pop("max_iter", 100)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        for value, name in ((max_attempts, "max_attempts"), (max_iter, "max_iter")):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        original_probabilities = self._predict_probabilities(query.reshape(1, -1))[0]
        original_class = int(np.argmax(original_probabilities))
        target_class = self._get_target_class(query, desired_class)
        counterfactuals = self._generate_diverse_counterfactuals(
            query,
            target_class,
            int(num_counterfactuals),
            max_attempts=int(max_attempts),
            max_iter=int(max_iter),
        )

        all_changes = []
        predictions = []
        distances = []
        for candidate in counterfactuals:
            changes = {}
            for index, name in enumerate(self.feature_names):
                difference = candidate[index] - query[index]
                if abs(difference) > 1e-7:
                    changes[name] = {
                        "original": float(query[index]),
                        "counterfactual": float(candidate[index]),
                        "change": float(difference),
                    }
            all_changes.append(changes)
            probabilities = self._predict_probabilities(candidate.reshape(1, -1))[0]
            predictions.append(probabilities.tolist())
            distances.append(self._proximity_loss(candidate, query))

        action_magnitudes = None
        if counterfactuals:
            action_magnitudes = {}
            for index, name in enumerate(self.feature_names):
                changes = [
                    abs(candidate[index] - query[index]) / self.scales[index]
                    for candidate in counterfactuals
                ]
                action_magnitudes[name] = float(np.mean(changes))

        failure_reason = None
        if not counterfactuals:
            failure_reason = (
                "No valid target-class candidate was found within the declared "
                "domain and search budget."
            )

        search_succeeded = bool(counterfactuals)
        explanation_data = {
            "algorithm": "constrained_multistart_search",
            "is_dice_implementation": False,
            "claim_status": "quarantined",
            "promotion_requires_joint_proximity_diversity_optimization": True,
            "official_dice_parity_established": False,
            "counterfactuals": [candidate.tolist() for candidate in counterfactuals],
            "counterfactual_predictions": predictions,
            "changes": all_changes,
            "normalized_squared_distances": distances,
            "original_class": original_class,
            "original_probabilities": original_probabilities.tolist(),
            "target_class": target_class,
            "num_requested": int(num_counterfactuals),
            "num_generated": len(counterfactuals),
            "search_succeeded": search_succeeded,
            "all_counterfactuals_valid": search_succeeded
            and all(self._is_target(candidate, target_class) for candidate in counterfactuals),
            "continuous_features": list(self.continuous_features),
            "categorical_features": list(self.categorical_features),
            "fixed_features": list(self.fixed_features),
            "proximity_weight": self.proximity_weight,
            "diversity_weight": self.diversity_weight,
            "diversity_loss": self._diversity_loss(counterfactuals),
            "failure_reason": failure_reason,
        }
        if action_magnitudes is not None:
            explanation_data.update(
                {
                    "feature_attributions": action_magnitudes,
                    "feature_attribution_semantics": (
                        "mean_absolute_normalized_counterfactual_action"
                    ),
                }
            )

        return Explanation(
            explainer_name="Counterfactual",
            target_class=self._target_name(target_class),
            explanation_data=explanation_data,
        )
