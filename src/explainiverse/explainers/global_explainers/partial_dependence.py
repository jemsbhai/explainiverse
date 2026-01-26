# src/explainiverse/explainers/global_explainers/partial_dependence.py
"""
Partial Dependence Plot (PDP) Explainer.

Shows the marginal effect of one or two features on the predicted outcome,
averaging over the values of all other features.

Reference:
    Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine.
    Annals of Statistics, 29(5), 1189-1232.
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class PartialDependenceExplainer(BaseExplainer):
    """
    Partial Dependence Plot (PDP) explainer.
    
    Computes the average prediction for each value of the feature(s) of interest,
    marginalizing over all other features. This shows the relationship between
    the feature and the predicted outcome.
    
    Attributes:
        model: Model adapter with .predict() method
        X: Training/reference data
        feature_names: List of feature names
        grid_resolution: Number of points in the PDP grid
        percentile_range: Range of percentiles to use for grid (default: 5-95)
    """
    
    def __init__(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        grid_resolution: int = 50,
        percentile_range: Tuple[float, float] = (5, 95)
    ):
        """
        Initialize the PDP explainer.
        
        Args:
            model: Model adapter with .predict() method
            X: Reference dataset (n_samples, n_features)
            feature_names: List of feature names
            grid_resolution: Number of grid points for each feature
            percentile_range: Tuple of (min_percentile, max_percentile) for grid
        """
        super().__init__(model)
        self.X = np.array(X)
        self.feature_names = list(feature_names)
        self.grid_resolution = grid_resolution
        self.percentile_range = percentile_range
    
    def _get_feature_idx(self, feature: Union[int, str]) -> int:
        """Convert feature name to index if needed."""
        if isinstance(feature, str):
            return self.feature_names.index(feature)
        return feature
    
    def _create_grid(self, feature_idx: int) -> np.ndarray:
        """Create a grid of values for a feature."""
        values = self.X[:, feature_idx]
        grid = np.linspace(
            np.percentile(values, self.percentile_range[0]),
            np.percentile(values, self.percentile_range[1]),
            self.grid_resolution
        )
        return grid
    
    def _compute_pdp_1d(self, feature_idx: int, target_class: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 1D partial dependence for a single feature.
        
        Args:
            feature_idx: Index of the feature
            target_class: Class index for which to compute PDP
            
        Returns:
            Tuple of (grid_values, pdp_values)
        """
        grid = self._create_grid(feature_idx)
        pdp_values = []
        
        for value in grid:
            X_temp = self.X.copy()
            X_temp[:, feature_idx] = value
            
            predictions = self.model.predict(X_temp)
            
            # Handle multi-class predictions
            if predictions.ndim == 2:
                avg_pred = np.mean(predictions[:, target_class])
            else:
                avg_pred = np.mean(predictions)
            
            pdp_values.append(avg_pred)
        
        return grid, np.array(pdp_values)
    
    def _compute_pdp_2d(
        self,
        feature_idx1: int,
        feature_idx2: int,
        target_class: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D partial dependence for feature interaction.
        
        Args:
            feature_idx1: Index of first feature
            feature_idx2: Index of second feature
            target_class: Class index for which to compute PDP
            
        Returns:
            Tuple of (grid1, grid2, pdp_values_2d)
        """
        grid1 = self._create_grid(feature_idx1)
        grid2 = self._create_grid(feature_idx2)
        
        pdp_values = np.zeros((len(grid1), len(grid2)))
        
        for i, val1 in enumerate(grid1):
            for j, val2 in enumerate(grid2):
                X_temp = self.X.copy()
                X_temp[:, feature_idx1] = val1
                X_temp[:, feature_idx2] = val2
                
                predictions = self.model.predict(X_temp)
                
                if predictions.ndim == 2:
                    avg_pred = np.mean(predictions[:, target_class])
                else:
                    avg_pred = np.mean(predictions)
                
                pdp_values[i, j] = avg_pred
        
        return grid1, grid2, pdp_values
    
    def explain(
        self,
        features: List[Union[int, str, Tuple[int, int]]],
        target_class: int = 1,
        **kwargs
    ) -> Explanation:
        """
        Compute partial dependence for specified features.
        
        Args:
            features: List of feature indices/names or tuples for interactions
            target_class: Class index for which to compute PDP
            
        Returns:
            Explanation object with PDP values and grids
        """
        pdp_results = {}
        grid_results = {}
        
        for feature in features:
            if isinstance(feature, tuple):
                # 2D interaction
                idx1 = self._get_feature_idx(feature[0])
                idx2 = self._get_feature_idx(feature[1])
                
                grid1, grid2, pdp = self._compute_pdp_2d(idx1, idx2, target_class)
                
                key = f"{self.feature_names[idx1]}_x_{self.feature_names[idx2]}"
                pdp_results[key] = pdp.tolist()
                grid_results[key] = {"grid1": grid1.tolist(), "grid2": grid2.tolist()}
            else:
                # 1D PDP
                idx = self._get_feature_idx(feature)
                grid, pdp = self._compute_pdp_1d(idx, target_class)
                
                key = self.feature_names[idx]
                pdp_results[key] = pdp.tolist()
                grid_results[key] = grid.tolist()
        
        return Explanation(
            explainer_name="PartialDependence",
            target_class=f"class_{target_class}",
            explanation_data={
                "pdp_values": pdp_results,
                "grid_values": grid_results,
                "features_analyzed": [str(f) for f in features],
                "interaction": any(isinstance(f, tuple) for f in features)
            }
        )
