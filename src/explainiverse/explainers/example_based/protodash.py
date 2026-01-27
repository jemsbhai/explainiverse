# src/explainiverse/explainers/example_based/protodash.py
"""
ProtoDash - Prototype Selection with Importance Weights.

ProtoDash selects a small set of prototypical examples from a dataset
that best represent the data distribution or explain model predictions.
Each prototype is assigned an importance weight indicating its contribution.

The algorithm minimizes the Maximum Mean Discrepancy (MMD) between:
- The weighted combination of selected prototypes
- The target distribution (full dataset or specific instances)

Key Features:
- Works with any model type (or no model at all for data summarization)
- Provides interpretable weights for each prototype
- Supports multiple kernel functions (RBF, linear, cosine)
- Can explain individual predictions or summarize entire datasets
- Class-conditional prototype selection

Use Cases:
1. Dataset Summarization: "These 10 examples represent the entire dataset"
2. Prediction Explanation: "This prediction is similar to examples A, B, C"
3. Model Debugging: "The model relies heavily on these training examples"
4. Data Compression: Reduce dataset while preserving distribution

Reference:
    Gurumoorthy, K.S., Dhurandhar, A., Cecchi, G., & Aggarwal, C. (2019).
    "Efficient Data Representation by Selecting Prototypes with Importance Weights"
    IEEE International Conference on Data Mining (ICDM).
    
    Also based on:
    Kim, B., Khanna, R., & Koyejo, O. (2016).
    "Examples are not Enough, Learn to Criticize! Criticism for Interpretability"
    NeurIPS 2016.

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

import numpy as np
from typing import List, Optional, Union, Callable, Tuple, Dict
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class ProtoDashExplainer(BaseExplainer):
    """
    ProtoDash explainer for prototype-based explanations.
    
    Selects representative examples (prototypes) from a reference dataset
    that best explain a target distribution or individual predictions.
    Each prototype is assigned an importance weight.
    
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
    
    def __init__(
        self,
        model=None,
        n_prototypes: int = 10,
        kernel: str = "rbf",
        kernel_width: Optional[float] = None,
        epsilon: float = 1e-10,
        optimize_weights: bool = True,
        random_state: Optional[int] = None,
        force_n_prototypes: bool = True
    ):
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
        
        self.n_prototypes = n_prototypes
        self.kernel = kernel.lower()
        self.kernel_width = kernel_width
        self.epsilon = epsilon
        self.optimize_weights = optimize_weights
        self.random_state = random_state
        self.force_n_prototypes = force_n_prototypes
        
        if self.kernel not in ["rbf", "linear", "cosine"]:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Supported: 'rbf', 'linear', 'cosine'"
            )
        
        # Cache for kernel matrix
        self._kernel_matrix_cache = None
        self._reference_data_hash = None
    
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
            if self.random_state is not None:
                np.random.seed(self.random_state)
            indices = np.random.choice(n_samples, size=1000, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Compute pairwise distances
        distances = cdist(X_sample, X_sample, metric='euclidean')
        
        # Get median of non-zero distances
        mask = distances > 0
        if np.any(mask):
            median_dist = np.median(distances[mask])
        else:
            median_dist = 1.0
        
        return max(median_dist, self.epsilon)
    
    def _compute_kernel(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        kernel_width: Optional[float] = None
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
            sigma = kernel_width or self.kernel_width
            if sigma is None:
                sigma = self._compute_kernel_width(X)
            
            # K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
            sq_dists = cdist(X, Y, metric='sqeuclidean')
            K = np.exp(-sq_dists / (2 * sigma ** 2))
            
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
        force_n_prototypes: bool = True
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
        selected_indices = []
        # Full weight vector (sparse, only selected indices are non-zero)
        weights = np.zeros(n_ref)
        
        for iteration in range(min(n_prototypes, n_ref)):
            # Compute gradient gain for each candidate
            # For the objective L(w) = (1/2) w^T K w - w^T μ
            # Gradient: ∇L = K w - μ
            # Gain for adding point j (currently w_j = 0): gain_j = μ_j - (Kw)_j
            # We want to maximize gain, which means minimizing the objective
            
            gradient = K_ref_ref @ weights - mu  # ∇L
            gains = -gradient  # gain = μ - Kw (negative gradient = descent direction)
            
            # Mask already selected indices
            gains_masked = gains.copy()
            gains_masked[selected_indices] = -np.inf
            
            # Select candidate with maximum gain
            best_idx = np.argmax(gains_masked)
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
        self,
        K: np.ndarray,
        mu: np.ndarray,
        normalize: bool = False
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
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        weights = result.x
        
        # Ensure non-negativity (numerical cleanup)
        weights = np.maximum(weights, 0)
        
        return weights
    
    def _optimize_weights(
        self,
        K_proto_proto: np.ndarray,
        K_proto_target: np.ndarray,
        initial_weights: np.ndarray
    ) -> np.ndarray:
        """
        Final weight optimization for selected prototypes.
        
        This is called after greedy selection to do a final refinement
        of weights, optionally with normalization for interpretability.
        
        Solves the same QP as _optimize_weights_qp but uses the
        mean kernel to target as the linear term.
        
        Args:
            K_proto_proto: Kernel matrix between prototypes (m x m)
            K_proto_target: Kernel matrix from prototypes to target (m x n_target)
            initial_weights: Initial weights from greedy selection
            
        Returns:
            Optimized weights (non-negative, optionally normalized)
        """
        n_proto = K_proto_proto.shape[0]
        
        if n_proto == 0:
            return np.array([])
        
        if n_proto == 1:
            return np.array([1.0])  # Single prototype gets weight 1
        
        # Target: mean kernel to target points
        mu = K_proto_target.mean(axis=1)
        
        # Use the QP solver
        weights = self._optimize_weights_qp(K_proto_proto, mu, normalize=False)
        
        # Normalize for interpretability (weights sum to 1)
        weight_sum = weights.sum()
        if weight_sum > self.epsilon:
            weights = weights / weight_sum
        else:
            # Fallback to equal weights if optimization failed
            weights = np.ones(n_proto) / n_proto
        
        return weights
    
    def find_prototypes(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        target_class: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        return_mmd: bool = False
    ) -> Explanation:
        """
        Find prototypes that summarize a dataset.
        
        Selects a small set of examples from X that best represent
        the data distribution. If y is provided, can select prototypes
        for a specific class.
        
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
                - weights: Importance weight for each prototype
                - prototypes: The actual prototype data points
                - mmd_score: (optional) Final MMD between prototypes and data
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        
        # Filter by class if specified
        if y is not None and target_class is not None:
            y = np.asarray(y)
            class_mask = (y == target_class)
            X_candidates = X[class_mask]
            original_indices = np.where(class_mask)[0]
        else:
            X_candidates = X
            original_indices = np.arange(n_samples)
        
        n_candidates = X_candidates.shape[0]
        n_proto = min(self.n_prototypes, n_candidates)
        
        if n_proto == 0:
            raise ValueError("No candidate examples available for prototype selection.")
        
        # Auto-compute kernel width if needed
        if self.kernel == "rbf" and self.kernel_width is None:
            self.kernel_width = self._compute_kernel_width(X_candidates)
        
        # Compute kernel matrices
        # K(candidates, candidates) for prototype selection
        # K(candidates, X) for representing the full distribution
        K_cand_cand = self._compute_kernel(X_candidates, X_candidates)
        K_cand_all = self._compute_kernel(X_candidates, X)
        
        # Greedy prototype selection
        local_indices, greedy_weights = self._greedy_prototype_selection(
            K_cand_cand, K_cand_all, n_proto, self.force_n_prototypes
        )
        
        # Convert to original indices
        prototype_indices = [int(original_indices[i]) for i in local_indices]
        
        # Optimize weights if requested
        if self.optimize_weights and len(local_indices) > 1:
            # Get kernel matrices for selected prototypes
            proto_local_idx = np.array(local_indices)
            K_proto_proto = K_cand_cand[np.ix_(proto_local_idx, proto_local_idx)]
            K_proto_all = K_cand_all[proto_local_idx, :]
            
            weights = self._optimize_weights(K_proto_proto, K_proto_all, greedy_weights)
        else:
            # Normalize greedy weights for interpretability
            weights = greedy_weights.copy()
            weight_sum = weights.sum()
            if weight_sum > self.epsilon:
                weights = weights / weight_sum
            elif len(weights) > 0:
                weights = np.ones(len(weights)) / len(weights)
        
        # Build explanation data
        explanation_data = {
            "prototype_indices": prototype_indices,
            "weights": weights.tolist(),
            "prototypes": X[prototype_indices].tolist(),
            "n_prototypes": len(prototype_indices),
            "kernel": self.kernel,
            "kernel_width": self.kernel_width if self.kernel == "rbf" else None,
        }
        
        if feature_names:
            explanation_data["feature_names"] = feature_names
        
        # Compute MMD if requested
        if return_mmd:
            proto_idx_local = np.array(local_indices)
            K_pp = K_cand_cand[np.ix_(proto_idx_local, proto_idx_local)]
            K_pa = K_cand_all[proto_idx_local, :]
            K_aa = self._compute_kernel(X, X)
            
            # MMD^2 = w^T K_pp w - 2 * w^T K_pa.mean() + K_aa.mean()
            w = np.array(weights)
            mmd_sq = w @ K_pp @ w - 2 * w @ K_pa.mean(axis=1) + K_aa.mean()
            mmd = np.sqrt(max(mmd_sq, 0))
            
            explanation_data["mmd_score"] = float(mmd)
        
        # Determine label
        if target_class is not None:
            label_name = f"class_{target_class}"
        else:
            label_name = "dataset"
        
        return Explanation(
            explainer_name="ProtoDash",
            target_class=label_name,
            explanation_data=explanation_data
        )
    
    def explain(
        self,
        instance: np.ndarray,
        X_reference: np.ndarray,
        feature_names: Optional[List[str]] = None,
        use_predictions: bool = False,
        return_similarity: bool = True
    ) -> Explanation:
        """
        Explain a prediction by finding similar prototypes.
        
        Finds prototypes from the reference set that are most similar
        to the given instance, providing a "this is like..." explanation.
        
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
        instance = np.asarray(instance, dtype=np.float64).flatten()
        X_reference = np.asarray(X_reference, dtype=np.float64)
        
        if X_reference.ndim == 1:
            X_reference = X_reference.reshape(1, -1)
        
        n_ref, n_features = X_reference.shape
        n_proto = min(self.n_prototypes, n_ref)
        
        # Auto-compute kernel width if needed
        if self.kernel == "rbf" and self.kernel_width is None:
            self.kernel_width = self._compute_kernel_width(X_reference)
        
        # If using predictions and model is available, augment features
        if use_predictions and self.model is not None:
            # Get predictions for instance and reference
            instance_pred = self.model.predict(instance.reshape(1, -1)).flatten()
            ref_preds = self.model.predict(X_reference)
            
            # Augment features with predictions
            instance_aug = np.concatenate([instance, instance_pred])
            X_ref_aug = np.hstack([X_reference, ref_preds])
        else:
            instance_aug = instance
            X_ref_aug = X_reference
        
        # Compute kernel matrices
        # K(reference, reference) for prototype selection
        # K(reference, instance) as target
        K_ref_ref = self._compute_kernel(X_ref_aug, X_ref_aug)
        K_ref_instance = self._compute_kernel(X_ref_aug, instance_aug.reshape(1, -1))
        
        # Greedy prototype selection
        prototype_indices, greedy_weights = self._greedy_prototype_selection(
            K_ref_ref, K_ref_instance, n_proto, self.force_n_prototypes
        )
        
        # Optimize weights
        if self.optimize_weights and len(prototype_indices) > 1:
            proto_idx = np.array(prototype_indices)
            K_proto_proto = K_ref_ref[np.ix_(proto_idx, proto_idx)]
            K_proto_instance = K_ref_instance[proto_idx, :]
            
            weights = self._optimize_weights(K_proto_proto, K_proto_instance, greedy_weights)
        else:
            # Normalize greedy weights for interpretability
            weights = greedy_weights.copy()
            weight_sum = weights.sum()
            if weight_sum > self.epsilon:
                weights = weights / weight_sum
            elif len(weights) > 0:
                weights = np.ones(len(weights)) / len(weights)
        
        # Build explanation data
        explanation_data = {
            "prototype_indices": [int(i) for i in prototype_indices],
            "weights": weights.tolist(),
            "prototypes": X_reference[prototype_indices].tolist(),
            "n_prototypes": len(prototype_indices),
            "kernel": self.kernel,
            "kernel_width": self.kernel_width if self.kernel == "rbf" else None,
            "instance": instance.tolist(),
        }
        
        if feature_names:
            explanation_data["feature_names"] = feature_names
        
        # Add similarity scores
        if return_similarity:
            K_instance_proto = self._compute_kernel(
                instance.reshape(1, -1),
                X_reference[prototype_indices]
            ).flatten()
            explanation_data["similarity_scores"] = K_instance_proto.tolist()
        
        # Add model predictions if available
        if self.model is not None:
            instance_pred = self.model.predict(instance.reshape(1, -1))
            proto_preds = self.model.predict(X_reference[prototype_indices])
            
            explanation_data["instance_prediction"] = instance_pred.tolist()
            explanation_data["prototype_predictions"] = proto_preds.tolist()
        
        return Explanation(
            explainer_name="ProtoDash",
            target_class="instance_explanation",
            explanation_data=explanation_data
        )
    
    def explain_batch(
        self,
        X: np.ndarray,
        X_reference: np.ndarray,
        feature_names: Optional[List[str]] = None
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
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        return [
            self.explain(X[i], X_reference, feature_names)
            for i in range(X.shape[0])
        ]
    
    def find_criticisms(
        self,
        X: np.ndarray,
        prototype_indices: List[int],
        n_criticisms: int = 5,
        feature_names: Optional[List[str]] = None
    ) -> Explanation:
        """
        Find criticisms - examples not well-represented by prototypes.
        
        Criticisms are data points that are furthest from the prototype
        representation, highlighting unusual or edge-case examples.
        
        This implements the criticism selection from MMD-Critic (Kim et al., 2016).
        
        Args:
            X: Full dataset.
            prototype_indices: Indices of already-selected prototypes.
            n_criticisms: Number of criticisms to find.
            feature_names: Optional feature names.
            
        Returns:
            Explanation with criticism indices and their "unusualness" scores.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        prototype_indices = list(prototype_indices)
        n_crit = min(n_criticisms, n_samples - len(prototype_indices))
        
        if n_crit <= 0:
            return Explanation(
                explainer_name="ProtoDash_Criticisms",
                target_class="criticisms",
                explanation_data={
                    "criticism_indices": [],
                    "unusualness_scores": [],
                    "criticisms": []
                }
            )
        
        # Auto-compute kernel width if needed
        if self.kernel == "rbf" and self.kernel_width is None:
            self.kernel_width = self._compute_kernel_width(X)
        
        # Compute kernel from all points to prototypes
        X_proto = X[prototype_indices]
        K_all_proto = self._compute_kernel(X, X_proto)
        
        # For each point, compute its "witness function" value
        # High values = well-represented by prototypes
        # Low values = not well-represented (criticisms)
        
        # Mean kernel distance to prototypes
        mean_sim_to_protos = K_all_proto.mean(axis=1)
        
        # Mean kernel value to all other points (density estimate)
        K_all_all = self._compute_kernel(X, X)
        mean_sim_to_all = K_all_all.mean(axis=1)
        
        # Unusualness = difference between expected similarity and prototype similarity
        # Points with high unusualness are criticisms
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
                "feature_names": feature_names
            }
        )
    
    def get_prototype_summary(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        include_criticisms: bool = True,
        n_criticisms: int = 5
    ) -> Dict:
        """
        Generate a complete prototype-based summary of a dataset.
        
        Combines prototype selection with optional criticisms for a
        complete data summary.
        
        Args:
            X: Dataset to summarize.
            y: Optional labels.
            feature_names: Optional feature names.
            include_criticisms: Whether to also find criticisms.
            n_criticisms: Number of criticisms if including them.
            
        Returns:
            Dictionary with prototypes, weights, and optionally criticisms.
        """
        # Find prototypes
        proto_exp = self.find_prototypes(X, y, feature_names=feature_names, return_mmd=True)
        
        result = {
            "prototypes": proto_exp.explanation_data,
        }
        
        # Find criticisms if requested
        if include_criticisms:
            crit_exp = self.find_criticisms(
                X,
                proto_exp.explanation_data["prototype_indices"],
                n_criticisms,
                feature_names
            )
            result["criticisms"] = crit_exp.explanation_data
        
        return result
