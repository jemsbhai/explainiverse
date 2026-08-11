# src/explainiverse/explainers/example_based/protodash.py
"""
ProtoDash - Prototype Selection with Importance Weights.

ProtoDash greedily selects reference examples for its kernel objective and
assigns non-negative objective weights. Those weights are not causal
contributions and do not by themselves establish representativeness to a human.

The algorithm minimizes the Maximum Mean Discrepancy (MMD) between:
- The weighted combination of selected prototypes
- The target distribution (full dataset or specific instances)

The distribution-level objective is model-independent. Class-conditional
selection uses the labels supplied by the caller. RBF, linear, and cosine
kernels are supported under the contracts validated below.

Reference:
    Gurumoorthy, K.S., Dhurandhar, A., Cecchi, G., & Aggarwal, C. (2019).
    "Efficient Data Representation by Selecting Prototypes with Importance Weights"
    IEEE International Conference on Data Mining (ICDM).

Example:
    from explainiverse.explainers.example_based import ProtoDashExplainer

    # Dataset summarization
    explainer = ProtoDashExplainer(n_prototypes=10, kernel="rbf")
    result = explainer.find_prototypes(X_train)
    print(f"Prototype indices: {result.explanation_data['prototype_indices']}")
    print(f"Weights: {result.explanation_data['weights']}")

    # Explaining a prediction
    explainer = ProtoDashExplainer(model=adapter, n_prototypes=5)
    explanation = explainer.explain(test_instance, X_reference=X_train)
"""

from fractions import Fraction
from numbers import Integral, Real
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import (
    as_real_array,
    validate_name_sequence,
    validate_single_tabular_instance,
)


class ProtoDashExplainer(BaseExplainer):
    """
    ProtoDash explainer for prototype-based explanations.

    Selects reference examples (prototypes) by greedily optimizing the
    ProtoDash objective. Each selected example receives a non-negative weight.

    The algorithm greedily selects prototypes that minimize the Maximum
    Mean Discrepancy (MMD) between the weighted prototype set and the
    target, then optimizes the weights.

    Attributes:
        model: Optional model adapter (for prediction-based explanations)
        n_prototypes: Number of prototypes to select
        kernel: Kernel function type ("rbf", "linear", "cosine")
        kernel_width: Width parameter for RBF kernel (auto-computed if None)
        epsilon: Small constant for numerical stability

    Example:
        >>> explainer = ProtoDashExplainer(n_prototypes=5, kernel="rbf")
        >>> result = explainer.find_prototypes(X_train)
        >>> prototypes = X_train[result.explanation_data['prototype_indices']]
    """

    n_prototypes: int
    kernel: str
    kernel_width: Optional[float]
    epsilon: float
    optimize_weights: bool
    random_state: Optional[int]
    force_n_prototypes: bool

    def __init__(
        self,
        model=None,
        n_prototypes: int = 10,
        kernel: str = "rbf",
        kernel_width: Optional[float] = None,
        epsilon: float = 1e-10,
        optimize_weights: bool = True,
        random_state: Optional[int] = None,
        force_n_prototypes: bool = True,
    ) -> None:
        """
        Initialize the ProtoDash explainer.

        Args:
            model: Optional model adapter. If provided, can use model
                   predictions in the kernel computation for explanation.
            n_prototypes: Number of prototypes to select (default: 10).
            kernel: Kernel function type:
                - "rbf": Radial Basis Function (Gaussian) kernel
                - "linear": Linear kernel (dot product)
                - "cosine": Cosine similarity kernel
            kernel_width: Width (sigma) for RBF kernel. If None, uses
                         median heuristic based on pairwise distances.
            epsilon: Small constant for numerical stability (default: 1e-10).
            optimize_weights: If True, optimize weights after greedy selection.
                             If False, use weights from greedy selection only.
            random_state: Random seed for reproducibility.
            force_n_prototypes: If True (default), always select exactly
                               n_prototypes (or all available if fewer).
                               If False, may stop early when gain becomes
                               negative (original ProtoDash behavior).
        """
        super().__init__(model)

        if not isinstance(n_prototypes, Integral) or isinstance(n_prototypes, bool):
            raise TypeError("n_prototypes must be an integer")
        if n_prototypes < 1:
            raise ValueError("n_prototypes must be at least 1")
        if not isinstance(kernel, str):
            raise TypeError("kernel must be a string")
        if kernel_width is not None:
            if not isinstance(kernel_width, Real) or isinstance(kernel_width, (bool, np.bool_)):
                raise TypeError("kernel_width must be a real number or None")
            if not np.isfinite(kernel_width) or kernel_width <= 0:
                raise ValueError("kernel_width must be a positive finite number")
        if not isinstance(epsilon, Real) or isinstance(epsilon, (bool, np.bool_)):
            raise TypeError("epsilon must be a real number")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a positive finite number")
        for value, name in (
            (optimize_weights, "optimize_weights"),
            (force_n_prototypes, "force_n_prototypes"),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean")
        if random_state is not None and (
            not isinstance(random_state, Integral) or isinstance(random_state, bool)
        ):
            raise TypeError("random_state must be an integer or None")
        if random_state is not None and (random_state < 0 or random_state > 2**32 - 1):
            raise ValueError("random_state must be between 0 and 2**32 - 1 or None")

        self.n_prototypes = int(n_prototypes)
        self.kernel = kernel.lower()
        self.kernel_width = None if kernel_width is None else float(kernel_width)
        self.epsilon = float(epsilon)
        self.optimize_weights = bool(optimize_weights)
        self.random_state = None if random_state is None else int(random_state)
        self.force_n_prototypes = bool(force_n_prototypes)

        if self.kernel not in ["rbf", "linear", "cosine"]:
            raise ValueError(f"Unknown kernel '{kernel}'. Supported: 'rbf', 'linear', 'cosine'")

    def _compute_kernel_width(self, X: np.ndarray) -> float:
        """
        Compute kernel width using median heuristic.

        The median heuristic sets sigma = median of pairwise distances,
        which is a common rule of thumb for RBF kernels.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Kernel width (sigma) value
        """
        # Subsample for efficiency if dataset is large
        n_samples = X.shape[0]
        if n_samples > 1000:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=1000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Compute pairwise distances
        distances = cdist(X_sample, X_sample, metric="euclidean")

        # Get median of non-zero distances
        mask = distances > 0
        if np.any(mask):
            median_dist = float(np.median(distances[mask]))
        else:
            median_dist = 1.0

        return float(max(median_dist, self.epsilon))

    def _compute_kernel(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None, kernel_width: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute kernel matrix between X and Y.

        Args:
            X: First data matrix of shape (n_samples_X, n_features)
            Y: Second data matrix of shape (n_samples_Y, n_features).
               If None, computes K(X, X).
            kernel_width: Override kernel width for RBF kernel.

        Returns:
            Kernel matrix of shape (n_samples_X, n_samples_Y)
        """
        if Y is None:
            Y = X

        if self.kernel == "rbf":
            sigma = self.kernel_width if kernel_width is None else kernel_width
            if sigma is None:
                sigma = self._compute_kernel_width(X)

            # K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
            sq_dists = cdist(X, Y, metric="sqeuclidean")
            K = np.exp(-sq_dists / (2 * sigma**2))

        elif self.kernel == "linear":
            # K(x, y) = x · y
            K = X @ Y.T

        elif self.kernel == "cosine":
            # K(x, y) = (x · y) / (||x|| * ||y||)
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + self.epsilon)
            Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + self.epsilon)
            K = X_norm @ Y_norm.T

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        return K

    def _greedy_prototype_selection(
        self,
        K_ref_ref: np.ndarray,
        K_ref_target: np.ndarray,
        n_prototypes: int,
        force_n_prototypes: bool = True,
    ) -> Tuple[List[int], np.ndarray]:
        """
        ProtoDash greedy prototype selection with iterative weight optimization.

        Implements the algorithm from:
        Gurumoorthy et al., 2019 - "Efficient Data Representation by Selecting
        Prototypes with Importance Weights" (ICDM)

        The algorithm solves:
            min_w  (1/2) w^T K w - w^T μ
            s.t.   w >= 0

        where μ_j = mean(K(x_j, target_points)) is the mean kernel similarity
        of candidate j to all target points.

        At each iteration:
        1. Compute gradient gain for each unselected candidate
        2. Select the candidate with maximum positive gain
        3. Re-optimize weights over all selected prototypes

        Args:
            K_ref_ref: Kernel matrix K(reference, reference) of shape (n_ref, n_ref)
            K_ref_target: Kernel matrix K(reference, target) of shape (n_ref, n_target)
            n_prototypes: Number of prototypes to select
            force_n_prototypes: If True, always select n_prototypes even if gain
                               becomes negative. If False, stop when no positive gain.

        Returns:
            Tuple of (prototype_indices, weights)
        """
        n_ref = K_ref_ref.shape[0]

        # μ_j = mean kernel similarity of candidate j to target distribution
        # This is the linear term in the QP objective
        mu = K_ref_target.mean(axis=1)

        # Track selected prototypes and their optimized weights
        selected_indices: List[int] = []
        # Full weight vector (sparse, only selected indices are non-zero)
        weights = np.zeros(n_ref)

        for iteration in range(min(n_prototypes, n_ref)):
            # Compute gradient gain for each candidate
            # For the objective L(w) = (1/2) w^T K w - w^T μ
            # Gradient: ∇L = K w - μ
            # Gain for adding point j (currently w_j = 0): gain_j = μ_j - (Kw)_j
            # We want to maximize gain, which means minimizing the objective

            gradient = K_ref_ref @ weights - mu  # objective gradient
            gains = -gradient  # mu - K w (negative gradient)

            # Mask already selected indices
            gains_masked = gains.copy()
            gains_masked[selected_indices] = -np.inf

            # Select candidate with maximum gain
            best_idx = int(np.argmax(gains_masked))
            best_gain = gains_masked[best_idx]

            # Early stopping check (only if not forcing n_prototypes)
            if not force_n_prototypes and best_gain <= self.epsilon:
                break

            selected_indices.append(best_idx)

            # Re-optimize weights over all selected prototypes
            # Solve: min_w (1/2) w^T K_ss w - w^T μ_s, s.t. w >= 0
            # where K_ss is kernel matrix restricted to selected indices
            # and μ_s is mu restricted to selected indices

            selected_arr = np.array(selected_indices)
            K_selected = K_ref_ref[np.ix_(selected_arr, selected_arr)]
            mu_selected = mu[selected_arr]

            # Optimize weights for selected prototypes
            w_selected = self._optimize_weights_qp(K_selected, mu_selected)

            # Update full weight vector
            weights = np.zeros(n_ref)
            weights[selected_arr] = w_selected

        # Return only the selected indices and their weights
        if len(selected_indices) == 0:
            return [], np.array([])

        final_weights = weights[np.array(selected_indices)]
        return selected_indices, final_weights

    def _optimize_weights_qp(
        self, K: np.ndarray, mu: np.ndarray, normalize: bool = False
    ) -> np.ndarray:
        """
        Optimize prototype weights via constrained quadratic programming.

        Solves:
            min_w  (1/2) w^T K w - w^T μ
            s.t.   w >= 0
                   (optional) sum(w) = 1

        Uses scipy.optimize.minimize with SLSQP method.

        Args:
            K: Kernel matrix between selected prototypes (m x m)
            mu: Mean kernel similarity to target for each prototype (m,)
            normalize: If True, constrain weights to sum to 1

        Returns:
            Optimized non-negative weights
        """
        m = K.shape[0]

        if m == 0:
            return np.array([])

        if m == 1:
            # Single prototype: optimal weight is μ/K if K > 0
            if K[0, 0] > self.epsilon:
                w = max(mu[0] / K[0, 0], 0)
            else:
                w = 1.0
            return np.array([w]) if not normalize else np.array([1.0])

        # Add small regularization for numerical stability
        K_reg = K + self.epsilon * np.eye(m)

        # Objective: (1/2) w^T K w - w^T μ
        def objective(w):
            return 0.5 * w @ K_reg @ w - w @ mu

        def gradient(w):
            return K_reg @ w - mu

        # Initial guess: equal weights
        w0 = np.ones(m) / m

        # Bounds: w >= 0
        bounds = [(0, None) for _ in range(m)]

        # Constraints
        constraints = []
        if normalize:
            constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(f"ProtoDash weight optimization failed: {result.message}")

        weights = result.x

        # Ensure non-negativity (numerical cleanup)
        weights = np.maximum(weights, 0)

        return weights

    def _optimize_weights(
        self, K_proto_proto: np.ndarray, K_proto_target: np.ndarray, initial_weights: np.ndarray
    ) -> np.ndarray:
        """
        Final canonical weight optimization for selected prototypes.

        Solves the same QP as _optimize_weights_qp but uses the
        mean kernel to target as the linear term.

        Args:
            K_proto_proto: Kernel matrix between prototypes (m x m)
            K_proto_target: Kernel matrix from prototypes to target (m x n_target)
            initial_weights: Initial weights from greedy selection

        Returns:
            Optimized non-negative objective weights. These are intentionally
            unnormalized, matching the ProtoDash optimization problem.
        """
        n_proto = K_proto_proto.shape[0]

        if n_proto == 0:
            return np.array([])

        # Target: mean kernel to target points
        mu = K_proto_target.mean(axis=1)

        # Use the QP solver
        return self._optimize_weights_qp(K_proto_proto, mu, normalize=False)

    def _scale_safe_weight_normalization(
        self, objective_weights: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Normalize finite non-negative weights without summation overflow."""
        objective_weights = as_real_array(
            objective_weights,
            name="objective_weights",
            dtype=float,
            require_finite=True,
        )
        if objective_weights.size == 0:
            return objective_weights, False
        if np.any(objective_weights < 0):
            raise ValueError("objective_weights must be non-negative")

        scale = float(np.max(objective_weights))
        if scale == 0.0:
            return np.zeros_like(objective_weights), False

        scaled_weights = objective_weights / scale
        scaled_total = float(np.sum(scaled_weights))
        if scale <= self.epsilon / scaled_total:
            return np.zeros_like(objective_weights), False
        return scaled_weights / scaled_total, True

    def _normalized_display_weights(self, objective_weights: np.ndarray) -> np.ndarray:
        """Normalize positive mass; preserve zero mass instead of inventing weights."""
        normalized, _ = self._scale_safe_weight_normalization(objective_weights)
        return normalized

    def _normalized_weights_defined(self, objective_weights: np.ndarray) -> bool:
        """Return whether canonical weights have enough mass to normalize."""
        _, is_defined = self._scale_safe_weight_normalization(objective_weights)
        return is_defined

    @staticmethod
    def _evaluate_objective(
        mu: np.ndarray,
        kernel_matrix: np.ndarray,
        objective_weights: np.ndarray,
    ) -> float:
        """Evaluate the ProtoDash objective without serializing overflow artifacts."""
        for values, name in (
            (mu, "mu"),
            (kernel_matrix, "kernel_matrix"),
            (objective_weights, "objective_weights"),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")

        try:
            with np.errstate(over="raise", under="raise", invalid="raise"):
                result = float(
                    mu @ objective_weights
                    - 0.5 * objective_weights @ kernel_matrix @ objective_weights
                )
        except FloatingPointError:
            # The two terms may overflow separately even when exact cancellation
            # leaves a representable result. Re-evaluate that exceptional path
            # from the exact binary-float rationals before deciding to fail.
            weights_exact = [Fraction.from_float(float(value)) for value in objective_weights]
            linear = sum(
                Fraction.from_float(float(value)) * weight
                for value, weight in zip(mu, weights_exact)
            )
            quadratic = sum(
                left_weight * Fraction.from_float(float(kernel_matrix[row, column])) * right_weight
                for row, left_weight in enumerate(weights_exact)
                for column, right_weight in enumerate(weights_exact)
            )
            exact_result = linear - quadratic / 2
            try:
                result = float(exact_result)
            except OverflowError as exc:
                raise ValueError(
                    "ProtoDash objective is not representable as a finite float64 value"
                ) from exc
            if result == 0.0 and exact_result != 0:
                raise ValueError(
                    "ProtoDash objective is not representable as a nonzero float64 value"
                )

        if not np.isfinite(result):
            raise ValueError("ProtoDash objective is not representable as a finite float64 value")
        return result

    def find_prototypes(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        target_class: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        return_mmd: bool = False,
    ) -> Explanation:
        """
        Select examples by the implemented ProtoDash kernel objective.

        Greedily selects candidates from ``X`` and optimizes non-negative
        objective weights. If ``y`` and ``target_class`` are supplied, both
        candidates and the empirical target distribution are restricted to
        that class. No human-representativeness claim follows from the
        objective value alone.

        Args:
            X: Data matrix of shape (n_samples, n_features).
            y: Optional labels. If provided with target_class, selects
               prototypes only from that class.
            target_class: If provided with y, only consider examples
                         from this class as candidates.
            feature_names: Optional list of feature names.
            return_mmd: If True, include MMD score in explanation.

        Returns:
            Explanation object containing:
                - prototype_indices: Indices of selected prototypes in X
                - weights: Normalized display weight for each prototype
                - prototypes: The actual prototype data points
                - mmd_score: (optional) Final MMD between prototypes and data
        """
        X = as_real_array(X, name="X", dtype=np.float64, require_finite=True)
        if not isinstance(return_mmd, (bool, np.bool_)):
            raise TypeError("return_mmd must be a boolean")
        return_mmd = bool(return_mmd)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        n_samples, n_features = X.shape
        validated_names = validate_name_sequence(
            feature_names,
            name="feature_names",
            allow_none=True,
        )
        if validated_names is not None and len(validated_names) != n_features:
            raise ValueError("feature_names length must match the columns of X")
        feature_names = validated_names
        if target_class is not None and y is None:
            raise ValueError("y is required when target_class is provided")
        if y is not None:
            y = np.asarray(y)
            if y.ndim != 1 or len(y) != n_samples:
                raise ValueError("y must be a 1D array aligned with X")

        # Filter by class if specified
        if y is not None and target_class is not None:
            class_mask = y == target_class
            X_candidates = X[class_mask]
            X_target = X_candidates
            original_indices = np.where(class_mask)[0]
        else:
            X_candidates = X
            X_target = X
            original_indices = np.arange(n_samples)

        n_candidates = X_candidates.shape[0]
        n_proto = min(self.n_prototypes, n_candidates)

        if n_proto == 0:
            raise ValueError("No candidate examples available for prototype selection.")

        effective_kernel_width = self.kernel_width
        if self.kernel == "rbf" and effective_kernel_width is None:
            effective_kernel_width = self._compute_kernel_width(X_candidates)

        # Compute kernel matrices
        # K(candidates, candidates) for prototype selection
        # K(candidates, target distribution). In class-conditional mode the
        # target distribution is the same requested class, not the full data.
        K_cand_cand = self._compute_kernel(X_candidates, X_candidates, effective_kernel_width)
        K_cand_target = self._compute_kernel(X_candidates, X_target, effective_kernel_width)

        # Greedy prototype selection
        local_indices, greedy_weights = self._greedy_prototype_selection(
            K_cand_cand, K_cand_target, n_proto, self.force_n_prototypes
        )

        # Convert to original indices
        prototype_indices = [int(original_indices[i]) for i in local_indices]

        # Optimize canonical (unnormalized) objective weights if requested.
        if self.optimize_weights and len(local_indices) > 1:
            # Get kernel matrices for selected prototypes
            proto_local_idx = np.array(local_indices)
            K_proto_proto = K_cand_cand[np.ix_(proto_local_idx, proto_local_idx)]
            K_proto_target = K_cand_target[proto_local_idx, :]

            objective_weights = self._optimize_weights(
                K_proto_proto, K_proto_target, greedy_weights
            )
        else:
            objective_weights = greedy_weights.copy()
        weights = self._normalized_display_weights(objective_weights)
        normalized_weights_defined = self._normalized_weights_defined(objective_weights)

        # Build explanation data
        explanation_data = {
            "prototype_indices": prototype_indices,
            "weights": weights.tolist(),
            "objective_weights": objective_weights.tolist(),
            "weight_semantics": (
                "normalized_relative_objective_weights"
                if normalized_weights_defined
                else "undefined_zero_objective_weight_mass"
            ),
            "normalized_weights_defined": normalized_weights_defined,
            "objective_weight_semantics": "unnormalized_protodash_weights",
            "prototypes": X[prototype_indices].tolist(),
            "n_prototypes": len(prototype_indices),
            "kernel": self.kernel,
            "kernel_width": effective_kernel_width if self.kernel == "rbf" else None,
            "candidate_count": int(n_candidates),
            "target_distribution_size": int(len(X_target)),
            "target_distribution": ("requested_class" if target_class is not None else "dataset"),
        }

        if feature_names:
            explanation_data["feature_names"] = feature_names

        # Compute MMD if requested
        if return_mmd:
            explanation_data["mmd_weight_space"] = "normalized_relative_weights"
            explanation_data["mmd_defined"] = normalized_weights_defined
            if normalized_weights_defined:
                # Distributional MMD requires a probability measure over the
                # selected support. Canonical zero-mass ProtoDash weights do
                # not define one and must not be replaced with uniform mass.
                proto_idx_local = np.array(local_indices)
                K_pp = K_cand_cand[np.ix_(proto_idx_local, proto_idx_local)]
                K_pt = K_cand_target[proto_idx_local, :]
                K_tt = self._compute_kernel(X_target, X_target, effective_kernel_width)
                w = np.array(weights)
                mmd_sq = w @ K_pp @ w - 2 * w @ K_pt.mean(axis=1) + K_tt.mean()
                mmd = np.sqrt(max(mmd_sq, 0))
                explanation_data["mmd_score"] = float(mmd)
            else:
                explanation_data["mmd_undefined_reason"] = (
                    "objective_weights_have_zero_normalizable_mass"
                )

        if local_indices:
            selected = np.asarray(local_indices, dtype=int)
            K_selected = K_cand_cand[np.ix_(selected, selected)]
            mu_selected = K_cand_target[selected].mean(axis=1)
            explanation_data["protodash_objective"] = self._evaluate_objective(
                mu_selected,
                K_selected,
                objective_weights,
            )

        # Determine label
        if target_class is not None:
            label_name = f"class_{target_class}"
        else:
            label_name = "dataset"

        return Explanation(
            explainer_name="ProtoDash", target_class=label_name, explanation_data=explanation_data
        )

    def explain(
        self,
        instance: np.ndarray,
        X_reference: np.ndarray,
        feature_names: Optional[List[str]] = None,
        use_predictions: bool = False,
        return_similarity: bool = True,
    ) -> Explanation:
        """
        Select reference examples relative to one instance.

        Runs the implemented ProtoDash objective with the instance as the
        empirical target distribution. The selected set need not equal the
        individually highest-k kernel similarities because the objective also
        accounts for interactions among selected candidates.

        Args:
            instance: Instance to explain (1D array of shape n_features).
            X_reference: Reference dataset to select prototypes from
                        (shape: n_samples, n_features).
            feature_names: Optional list of feature names.
            use_predictions: If True and model is provided, include model
                            predictions in the similarity computation.
            return_similarity: If True, include similarity scores.

        Returns:
            Explanation object containing prototype indices and weights.
        """
        X_reference = as_real_array(
            X_reference,
            name="X_reference",
            dtype=np.float64,
            require_finite=True,
        )

        for value, name in (
            (use_predictions, "use_predictions"),
            (return_similarity, "return_similarity"),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean")
        use_predictions = bool(use_predictions)
        return_similarity = bool(return_similarity)

        if X_reference.ndim == 1:
            X_reference = X_reference.reshape(1, -1)
        if X_reference.ndim != 2 or X_reference.shape[0] == 0:
            raise ValueError("X_reference must be a non-empty 2D array")
        instance = validate_single_tabular_instance(
            instance,
            X_reference.shape[1],
            dtype=np.float64,
            require_finite=True,
        )
        n_ref, n_features = X_reference.shape
        validated_names = validate_name_sequence(
            feature_names,
            name="feature_names",
            allow_none=True,
        )
        if validated_names is not None and len(validated_names) != n_features:
            raise ValueError("feature_names length must match the X_reference columns")
        feature_names = validated_names
        n_proto = min(self.n_prototypes, n_ref)

        if use_predictions and self.model is None:
            raise ValueError("use_predictions=True requires a model")

        # If using predictions and model is available, augment features
        if use_predictions and self.model is not None:
            # Get predictions for instance and reference
            instance_pred = as_real_array(
                self.model.predict(instance.reshape(1, -1)),
                name="model predictions",
                dtype=float,
                require_finite=True,
            ).reshape(1, -1)
            ref_preds = as_real_array(
                self.model.predict(X_reference),
                name="model predictions",
                dtype=float,
                require_finite=True,
            )
            if ref_preds.ndim == 1:
                ref_preds = ref_preds.reshape(-1, 1)
            if ref_preds.ndim != 2 or ref_preds.shape[0] != n_ref:
                raise ValueError("model.predict returned predictions with invalid shape")
            # Augment features with predictions
            instance_aug = np.concatenate([instance, instance_pred.reshape(-1)])
            X_ref_aug = np.hstack([X_reference, ref_preds])
            similarity_space = "input_plus_model_prediction"
        else:
            instance_aug = instance
            X_ref_aug = X_reference
            similarity_space = "input"

        effective_kernel_width = self.kernel_width
        if self.kernel == "rbf" and effective_kernel_width is None:
            effective_kernel_width = self._compute_kernel_width(X_ref_aug)

        # Compute kernel matrices
        # K(reference, reference) for prototype selection
        # K(reference, instance) as target
        K_ref_ref = self._compute_kernel(X_ref_aug, X_ref_aug, effective_kernel_width)
        K_ref_instance = self._compute_kernel(
            X_ref_aug, instance_aug.reshape(1, -1), effective_kernel_width
        )

        # Greedy prototype selection
        prototype_indices, greedy_weights = self._greedy_prototype_selection(
            K_ref_ref, K_ref_instance, n_proto, self.force_n_prototypes
        )

        # Optimize weights
        if self.optimize_weights and len(prototype_indices) > 1:
            proto_idx = np.array(prototype_indices)
            K_proto_proto = K_ref_ref[np.ix_(proto_idx, proto_idx)]
            K_proto_instance = K_ref_instance[proto_idx, :]

            objective_weights = self._optimize_weights(
                K_proto_proto, K_proto_instance, greedy_weights
            )
        else:
            objective_weights = greedy_weights.copy()
        weights = self._normalized_display_weights(objective_weights)
        normalized_weights_defined = self._normalized_weights_defined(objective_weights)

        # Build explanation data
        explanation_data = {
            "prototype_indices": [int(i) for i in prototype_indices],
            "weights": weights.tolist(),
            "objective_weights": objective_weights.tolist(),
            "weight_semantics": (
                "normalized_relative_objective_weights"
                if normalized_weights_defined
                else "undefined_zero_objective_weight_mass"
            ),
            "normalized_weights_defined": normalized_weights_defined,
            "objective_weight_semantics": "unnormalized_protodash_weights",
            "prototypes": X_reference[prototype_indices].tolist(),
            "n_prototypes": len(prototype_indices),
            "kernel": self.kernel,
            "kernel_width": effective_kernel_width if self.kernel == "rbf" else None,
            "instance": instance.tolist(),
            "kernel_input_space": similarity_space,
        }

        if feature_names:
            explanation_data["feature_names"] = feature_names

        # Add similarity scores
        if return_similarity:
            K_instance_proto = self._compute_kernel(
                instance_aug.reshape(1, -1),
                X_ref_aug[prototype_indices],
                effective_kernel_width,
            ).flatten()
            explanation_data["similarity_scores"] = K_instance_proto.tolist()
            explanation_data["similarity_space"] = similarity_space

        if prototype_indices:
            selected = np.asarray(prototype_indices, dtype=int)
            K_selected = K_ref_ref[np.ix_(selected, selected)]
            mu_selected = K_ref_instance[selected].mean(axis=1)
            explanation_data["protodash_objective"] = self._evaluate_objective(
                mu_selected,
                K_selected,
                objective_weights,
            )

        # Add model predictions if available
        if self.model is not None:
            instance_pred = as_real_array(
                self.model.predict(instance.reshape(1, -1)),
                name="model predictions",
                require_finite=True,
            )
            proto_preds = as_real_array(
                self.model.predict(X_reference[prototype_indices]),
                name="model predictions",
                require_finite=True,
            )

            explanation_data["instance_prediction"] = instance_pred.tolist()
            explanation_data["prototype_predictions"] = proto_preds.tolist()

        return Explanation(
            explainer_name="ProtoDash",
            target_class="instance_explanation",
            explanation_data=explanation_data,
        )

    def explain_batch(
        self, X: np.ndarray, X_reference: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> List[Explanation]:
        """
        Explain multiple instances.

        Args:
            X: Instances to explain (n_instances, n_features).
            X_reference: Reference dataset for prototype selection.
            feature_names: Optional feature names.

        Returns:
            List of Explanation objects, one per instance.
        """
        X = as_real_array(X, name="X", dtype=np.float64, require_finite=True)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")

        return [self.explain(X[i], X_reference, feature_names) for i in range(X.shape[0])]

    def find_criticisms(
        self,
        X: np.ndarray,
        prototype_indices: List[int],
        n_criticisms: int = 5,
        feature_names: Optional[List[str]] = None,
    ) -> Explanation:
        """
        Rank non-prototype rows by a library-defined kernel-witness score.

        The compatibility name ``criticisms`` refers here to a deterministic
        difference between mean kernel similarity to all rows and mean kernel
        similarity to the selected prototypes. It does not establish that a
        row is unusual, an edge case, or poorly represented for a downstream
        purpose, and it does not implement MMD-Critic's regularized subset
        optimization.

        Args:
            X: Full dataset.
            prototype_indices: Indices of already-selected prototypes.
            n_criticisms: Number of criticisms to find.
            feature_names: Optional feature names.

        Returns:
            Explanation with selected indices and their library-defined
            ``unusualness_scores`` compatibility field.
        """
        X = as_real_array(X, name="X", dtype=np.float64, require_finite=True)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        n_samples = X.shape[0]

        prototype_indices = list(prototype_indices)
        if not prototype_indices:
            raise ValueError("prototype_indices must contain at least one prototype")
        if len(set(prototype_indices)) != len(prototype_indices):
            raise ValueError("prototype_indices must be unique")
        if any(
            not isinstance(index, Integral)
            or isinstance(index, bool)
            or index < 0
            or index >= n_samples
            for index in prototype_indices
        ):
            raise ValueError("prototype_indices contains an invalid index")
        if not isinstance(n_criticisms, Integral) or isinstance(n_criticisms, bool):
            raise TypeError("n_criticisms must be an integer")
        if n_criticisms < 0:
            raise ValueError("n_criticisms must be non-negative")
        validated_names = validate_name_sequence(
            feature_names,
            name="feature_names",
            allow_none=True,
        )
        if validated_names is not None and len(validated_names) != X.shape[1]:
            raise ValueError("feature_names length must match the columns of X")
        feature_names = validated_names
        n_crit = min(n_criticisms, n_samples - len(prototype_indices))

        if n_crit <= 0:
            return Explanation(
                explainer_name="ProtoDash_Criticisms",
                target_class="criticisms",
                explanation_data={
                    "criticism_indices": [],
                    "unusualness_scores": [],
                    "criticisms": [],
                    "algorithm": "kernel_witness_ranking",
                    "is_mmd_critic_implementation": False,
                },
            )

        effective_kernel_width = self.kernel_width
        if self.kernel == "rbf" and effective_kernel_width is None:
            effective_kernel_width = self._compute_kernel_width(X)

        # Compute kernel from all points to prototypes
        X_proto = X[prototype_indices]
        K_all_proto = self._compute_kernel(X, X_proto, effective_kernel_width)

        # Compute the two mean-kernel terms used by this library diagnostic.
        mean_sim_to_protos = K_all_proto.mean(axis=1)

        # The all-row mean includes the row's self-similarity.
        K_all_all = self._compute_kernel(X, X, effective_kernel_width)
        mean_sim_to_all = K_all_all.mean(axis=1)

        # Retain the historical output name for compatibility. This difference
        # is descriptive and is not a calibrated unusualness measure.
        unusualness = mean_sim_to_all - mean_sim_to_protos

        # Exclude prototypes from consideration
        unusualness[prototype_indices] = -np.inf

        # Select top criticisms
        criticism_indices = np.argsort(unusualness)[-n_crit:][::-1].tolist()
        criticism_scores = unusualness[criticism_indices].tolist()

        return Explanation(
            explainer_name="ProtoDash_Criticisms",
            target_class="criticisms",
            explanation_data={
                "criticism_indices": criticism_indices,
                "unusualness_scores": criticism_scores,
                "criticisms": X[criticism_indices].tolist(),
                "n_criticisms": len(criticism_indices),
                "feature_names": feature_names,
                "algorithm": "kernel_witness_ranking",
                "is_mmd_critic_implementation": False,
                "kernel_width": (effective_kernel_width if self.kernel == "rbf" else None),
            },
        )

    def get_prototype_summary(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        include_criticisms: bool = True,
        n_criticisms: int = 5,
    ) -> Dict:
        """
        Return prototype selection with optional kernel-witness rows.

        This combines two library outputs; it does not claim completeness as a
        data summary or evaluation of coverage.

        Args:
            X: Dataset to summarize.
            y: Optional labels.
            feature_names: Optional feature names.
            include_criticisms: Whether to also find criticisms.
            n_criticisms: Number of criticisms if including them.

        Returns:
            Dictionary with prototypes, weights, and optionally criticisms.
        """
        if not isinstance(include_criticisms, (bool, np.bool_)):
            raise TypeError("include_criticisms must be a boolean")
        include_criticisms = bool(include_criticisms)
        # Find prototypes
        proto_exp = self.find_prototypes(X, y, feature_names=feature_names, return_mmd=True)

        result = {
            "prototypes": proto_exp.explanation_data,
        }

        # Find criticisms if requested
        if include_criticisms:
            crit_exp = self.find_criticisms(
                X, proto_exp.explanation_data["prototype_indices"], n_criticisms, feature_names
            )
            result["criticisms"] = crit_exp.explanation_data

        return result
