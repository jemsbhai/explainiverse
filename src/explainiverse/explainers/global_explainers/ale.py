# src/explainiverse/explainers/global_explainers/ale.py
"""
Accumulated Local Effects (ALE) Explainer.

ALE plots are an alternative to Partial Dependence Plots that are unbiased
when features are correlated. They measure how the prediction changes locally
when the feature value changes.

Reference:
    Apley, D.W. & Zhu, J. (2020). Visualizing the Effects of Predictor Variables
    in Black Box Supervised Learning Models. Journal of the Royal Statistical Society
    Series B, 82(4), 1059-1086.
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class ALEExplainer(BaseExplainer):
    """
    Accumulated Local Effects (ALE) explainer.
    
    Unlike PDP, ALE avoids extrapolation issues when features are correlated
    by using local differences rather than marginal averages.
    
    Attributes:
        model: Model adapter with .predict() method
        X: Training/reference data
        feature_names: List of feature names
        n_bins: Number of bins for computing ALE
    """
    
    def __init__(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        n_bins: int = 20
    ):
        """
        Initialize the ALE explainer.
        
        Args:
            model: Model adapter with .predict() method
            X: Reference dataset (n_samples, n_features)
            feature_names: List of feature names
            n_bins: Number of bins for ALE computation
        """
        super().__init__(model)
        self.X = np.array(X)
        self.feature_names = list(feature_names)
        self.n_bins = n_bins
    
    def _get_feature_idx(self, feature: Union[int, str]) -> int:
        """Convert feature name to index if needed."""
        if isinstance(feature, str):
            return self.feature_names.index(feature)
        return feature
    
    def _compute_quantile_bins(self, values: np.ndarray) -> np.ndarray:
        """
        Compute bin edges using quantiles to ensure similar sample sizes per bin.
        """
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(values, percentiles)
        # Remove duplicate edges
        bin_edges = np.unique(bin_edges)
        return bin_edges
    
    def _compute_ale_1d(
        self,
        feature_idx: int,
        target_class: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 1D ALE for a single feature.
        
        Args:
            feature_idx: Index of the feature
            target_class: Class index for which to compute ALE
            
        Returns:
            Tuple of (bin_centers, ale_values, bin_edges)
        """
        values = self.X[:, feature_idx]
        bin_edges = self._compute_quantile_bins(values)
        
        if len(bin_edges) < 2:
            # Not enough unique values
            return np.array([np.mean(values)]), np.array([0.0]), bin_edges
        
        # Compute local effects for each bin
        local_effects = []
        
        for i in range(len(bin_edges) - 1):
            lower, upper = bin_edges[i], bin_edges[i + 1]
            
            # Find samples in this bin
            if i == len(bin_edges) - 2:
                # Include upper bound in last bin
                in_bin = (values >= lower) & (values <= upper)
            else:
                in_bin = (values >= lower) & (values < upper)
            
            if not np.any(in_bin):
                local_effects.append(0.0)
                continue
            
            X_bin = self.X[in_bin]
            
            # Compute predictions at bin edges
            X_lower = X_bin.copy()
            X_lower[:, feature_idx] = lower
            
            X_upper = X_bin.copy()
            X_upper[:, feature_idx] = upper
            
            pred_lower = self.model.predict(X_lower)
            pred_upper = self.model.predict(X_upper)
            
            # Extract target class predictions
            if pred_lower.ndim == 2:
                pred_lower = pred_lower[:, target_class]
                pred_upper = pred_upper[:, target_class]
            
            # Local effect = average difference
            effect = np.mean(pred_upper - pred_lower)
            local_effects.append(effect)
        
        # Accumulate effects
        ale_values = np.cumsum(local_effects)
        
        # Center around zero (mean-center)
        ale_values = ale_values - np.mean(ale_values)
        
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_centers, ale_values, bin_edges
    
    def explain(
        self,
        feature: Union[int, str],
        target_class: int = 1,
        **kwargs
    ) -> Explanation:
        """
        Compute ALE for a specified feature.
        
        Args:
            feature: Feature index or name
            target_class: Class index for which to compute ALE
            
        Returns:
            Explanation object with ALE values
        """
        idx = self._get_feature_idx(feature)
        bin_centers, ale_values, bin_edges = self._compute_ale_1d(idx, target_class)
        
        feature_name = self.feature_names[idx]
        
        return Explanation(
            explainer_name="ALE",
            target_class=f"class_{target_class}",
            explanation_data={
                "ale_values": ale_values.tolist(),
                "bin_centers": bin_centers.tolist(),
                "bin_edges": bin_edges.tolist(),
                "feature": feature_name,
                "feature_attributions": {
                    feature_name: float(np.max(ale_values) - np.min(ale_values))
                }
            }
        )
    
    def explain_all(self, target_class: int = 1) -> List[Explanation]:
        """
        Compute ALE for all features.
        
        Args:
            target_class: Class index for which to compute ALE
            
        Returns:
            List of Explanation objects, one per feature
        """
        return [
            self.explain(idx, target_class)
            for idx in range(len(self.feature_names))
        ]
