# src/explainiverse/core/explanation.py
"""
Unified container for explanation results.

The Explanation class provides a standardized format for all explainer outputs,
enabling consistent handling across different explanation methods.
"""

from typing import Dict, List, Optional, Any


class Explanation:
    """
    Unified container for explanation results.
    
    Attributes:
        explainer_name: Name of the explainer that generated this explanation
        target_class: The class/output being explained
        explanation_data: Dictionary containing explanation details
            (e.g., feature_attributions, heatmaps, rules)
        feature_names: Optional list of feature names for index resolution
        metadata: Optional additional metadata about the explanation
    
    Example:
        >>> explanation = Explanation(
        ...     explainer_name="LIME",
        ...     target_class="cat",
        ...     explanation_data={"feature_attributions": {"fur": 0.8, "whiskers": 0.6}},
        ...     feature_names=["fur", "whiskers", "tail", "ears"]
        ... )
        >>> print(explanation.get_top_features(k=2))
        [('fur', 0.8), ('whiskers', 0.6)]
    """

    def __init__(
        self,
        explainer_name: str,
        target_class: str,
        explanation_data: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an Explanation object.
        
        Args:
            explainer_name: Name of the explainer (e.g., "LIME", "SHAP")
            target_class: The target class or output being explained
            explanation_data: Dictionary containing the explanation details.
                Common keys include:
                - "feature_attributions": Dict[str, float] mapping feature names to importance
                - "attributions_raw": List[float] of raw attribution values
                - "heatmap": np.ndarray for image explanations
                - "rules": List of rule strings for rule-based explanations
            feature_names: Optional list of feature names. If provided, enables
                index-based lookup in evaluation metrics.
            metadata: Optional additional metadata (e.g., computation time, parameters)
        """
        self.explainer_name = explainer_name
        self.target_class = target_class
        self.explanation_data = explanation_data
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.metadata = metadata or {}

    def __repr__(self):
        n_features = len(self.feature_names) if self.feature_names else "N/A"
        return (
            f"Explanation(explainer='{self.explainer_name}', "
            f"target='{self.target_class}', "
            f"keys={list(self.explanation_data.keys())}, "
            f"n_features={n_features})"
        )

    def get_attributions(self) -> Optional[Dict[str, float]]:
        """
        Get feature attributions if available.
        
        Returns:
            Dictionary mapping feature names to attribution values,
            or None if not available.
        """
        return self.explanation_data.get("feature_attributions")
    
    def get_top_features(self, k: int = 5, absolute: bool = True) -> List[tuple]:
        """
        Get the top-k most important features.
        
        Args:
            k: Number of top features to return
            absolute: If True, rank by absolute value of attribution
            
        Returns:
            List of (feature_name, attribution_value) tuples sorted by importance
        """
        attributions = self.get_attributions()
        if not attributions:
            return []
        
        if absolute:
            sorted_items = sorted(
                attributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        else:
            sorted_items = sorted(
                attributions.items(),
                key=lambda x: x[1],
                reverse=True
            )
        
        return sorted_items[:k]
    
    def get_feature_index(self, feature_name: str) -> Optional[int]:
        """
        Get the index of a feature by name.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Index of the feature, or None if not found or feature_names not set
        """
        if self.feature_names is None:
            return None
        try:
            return self.feature_names.index(feature_name)
        except ValueError:
            return None

    def plot(self, plot_type: str = 'bar', **kwargs):
        """
        Visualize the explanation.
        
        Args:
            plot_type: Type of plot ('bar', 'waterfall', 'heatmap')
            **kwargs: Additional arguments passed to the plotting function
            
        Note:
            This is a placeholder for future visualization integration.
        """
        print(
            f"[plot: {plot_type}] Plotting explanation for {self.target_class} "
            f"from {self.explainer_name}."
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert explanation to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the explanation
        """
        return {
            "explainer_name": self.explainer_name,
            "target_class": self.target_class,
            "explanation_data": self.explanation_data,
            "feature_names": self.feature_names,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Explanation":
        """
        Create an Explanation from a dictionary.
        
        Args:
            data: Dictionary with explanation data
            
        Returns:
            Explanation instance
        """
        return cls(
            explainer_name=data["explainer_name"],
            target_class=data["target_class"],
            explanation_data=data["explanation_data"],
            feature_names=data.get("feature_names"),
            metadata=data.get("metadata", {})
        )
