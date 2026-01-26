# src/explainiverse/explainers/global_explainers/sage.py
"""
SAGE (Shapley Additive Global importancE) Explainer.

SAGE extends SHAP to provide global feature importance by computing
the expected Shapley value across all samples. This gives a theoretically
grounded global importance measure.

Reference:
    Covert, I., Lundberg, S., & Lee, S.I. (2020). Understanding Global Feature
    Contributions with Additive Importance Measures. NeurIPS 2020.
"""

import numpy as np
from typing import List, Optional, Callable
from sklearn.metrics import accuracy_score, mean_squared_error
from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class SAGEExplainer(BaseExplainer):
    """
    SAGE: Shapley Additive Global importancE.
    
    Computes global feature importance using Shapley values, averaging
    contributions across all samples. Unlike permutation importance,
    SAGE accounts for feature interactions.
    
    Attributes:
        model: Model adapter with .predict() method
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        n_permutations: Number of permutation samples for approximation
        loss_fn: Loss function (default: accuracy for classification)
    """
    
    def __init__(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_permutations: int = 100,
        loss_fn: Optional[Callable] = None,
        task: str = "classification",
        random_state: int = 42
    ):
        """
        Initialize the SAGE explainer.
        
        Args:
            model: Model adapter with .predict() method
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: List of feature names
            n_permutations: Number of permutations for approximation
            loss_fn: Custom loss function (lower is better)
            task: "classification" or "regression"
            random_state: Random seed
        """
        super().__init__(model)
        self.X = np.array(X)
        self.y = np.array(y)
        self.feature_names = list(feature_names)
        self.n_permutations = n_permutations
        self.task = task
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        if loss_fn is None:
            if task == "classification":
                self.loss_fn = lambda y_true, y_pred: 1.0 - accuracy_score(
                    y_true, np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred
                )
            else:
                self.loss_fn = lambda y_true, y_pred: mean_squared_error(y_true, y_pred)
        else:
            self.loss_fn = loss_fn
    
    def _compute_loss(self, X_masked: np.ndarray) -> float:
        """Compute loss on masked data."""
        predictions = self.model.predict(X_masked)
        return self.loss_fn(self.y, predictions)
    
    def _marginal_contribution(
        self,
        feature_idx: int,
        feature_order: List[int],
        position: int
    ) -> float:
        """
        Compute marginal contribution of a feature given a feature ordering.
        
        The marginal contribution is the change in loss when adding the feature
        to the set of features that come before it in the ordering.
        """
        n_samples, n_features = self.X.shape
        
        # Features before this one in the ordering
        features_before = set(feature_order[:position])
        features_with = features_before | {feature_idx}
        
        # Create masked versions
        X_without = self.X.copy()
        X_with = self.X.copy()
        
        # Mask features NOT in the respective sets by replacing with random samples
        for j in range(n_features):
            if j not in features_before:
                # Shuffle this feature (marginalizing out)
                shuffle_idx = self.rng.permutation(n_samples)
                X_without[:, j] = self.X[shuffle_idx, j]
            
            if j not in features_with:
                shuffle_idx = self.rng.permutation(n_samples)
                X_with[:, j] = self.X[shuffle_idx, j]
        
        loss_without = self._compute_loss(X_without)
        loss_with = self._compute_loss(X_with)
        
        # Marginal contribution = reduction in loss
        return loss_without - loss_with
    
    def explain(self, **kwargs) -> Explanation:
        """
        Compute SAGE values for all features.
        
        Uses permutation sampling to approximate the Shapley values.
        
        Returns:
            Explanation object with global feature importance (SAGE values)
        """
        n_features = len(self.feature_names)
        sage_values = np.zeros(n_features)
        
        for _ in range(self.n_permutations):
            # Random feature ordering
            order = self.rng.permutation(n_features).tolist()
            
            for position, feature_idx in enumerate(order):
                contribution = self._marginal_contribution(
                    feature_idx, order, position
                )
                sage_values[feature_idx] += contribution
        
        # Average over permutations
        sage_values /= self.n_permutations
        
        # Create attribution dict
        attributions = {
            fname: float(sage_values[i])
            for i, fname in enumerate(self.feature_names)
        }
        
        return Explanation(
            explainer_name="SAGE",
            target_class="global",
            explanation_data={
                "feature_attributions": attributions,
                "n_permutations": self.n_permutations,
                "task": self.task
            }
        )
