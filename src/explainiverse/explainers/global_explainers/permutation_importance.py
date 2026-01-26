# src/explainiverse/explainers/global_explainers/permutation_importance.py
"""
Permutation Feature Importance Explainer.

Measures feature importance by measuring the decrease in model performance
when a feature's values are randomly shuffled.

Reference:
    Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
"""

import numpy as np
from typing import List, Optional, Callable
from sklearn.metrics import accuracy_score

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class PermutationImportanceExplainer(BaseExplainer):
    """
    Global explainer based on permutation feature importance.
    
    Measures how much the model's performance decreases when each feature
    is randomly shuffled, breaking the relationship between the feature
    and the target.
    
    Attributes:
        model: Model adapter with .predict() method
        X: Feature matrix for evaluation
        y: True labels
        feature_names: List of feature names
        n_repeats: Number of times to permute each feature
        scoring_fn: Function to compute score (higher is better)
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10,
        scoring_fn: Optional[Callable] = None,
        random_state: int = 42
    ):
        """
        Initialize the Permutation Importance explainer.
        
        Args:
            model: Model adapter with .predict() method
            X: Feature matrix (n_samples, n_features)
            y: True labels (n_samples,)
            feature_names: List of feature names
            n_repeats: Number of permutation repeats per feature
            scoring_fn: Custom scoring function (default: accuracy)
            random_state: Random seed
        """
        super().__init__(model)
        self.X = np.array(X)
        self.y = np.array(y)
        self.feature_names = feature_names
        self.n_repeats = n_repeats
        self.scoring_fn = scoring_fn or self._default_scorer
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def _default_scorer(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Default scoring function: accuracy for classification."""
        if y_pred.ndim == 2:
            y_pred = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred)
    
    def _compute_baseline_score(self) -> float:
        """Compute model performance on unperturbed data."""
        predictions = self.model.predict(self.X)
        return self.scoring_fn(self.y, predictions)
    
    def _permute_feature(self, X: np.ndarray, feature_idx: int) -> np.ndarray:
        """Create a copy of X with one feature permuted."""
        X_permuted = X.copy()
        self.rng.shuffle(X_permuted[:, feature_idx])
        return X_permuted
    
    def explain(self, **kwargs) -> Explanation:
        """
        Compute permutation feature importance.
        
        Returns:
            Explanation object with:
                - feature_attributions: dict of {feature_name: importance}
                - std: dict of {feature_name: std across repeats}
                - baseline_score: original model score
        """
        baseline_score = self._compute_baseline_score()
        
        importances = {}
        stds = {}
        
        for idx, fname in enumerate(self.feature_names):
            scores = []
            
            for _ in range(self.n_repeats):
                X_permuted = self._permute_feature(self.X, idx)
                predictions = self.model.predict(X_permuted)
                score = self.scoring_fn(self.y, predictions)
                scores.append(score)
            
            # Importance = drop in score when feature is permuted
            importance = baseline_score - np.mean(scores)
            importances[fname] = float(importance)
            stds[fname] = float(np.std(scores))
        
        return Explanation(
            explainer_name="PermutationImportance",
            target_class="global",
            explanation_data={
                "feature_attributions": importances,
                "std": stds,
                "baseline_score": baseline_score
            }
        )
