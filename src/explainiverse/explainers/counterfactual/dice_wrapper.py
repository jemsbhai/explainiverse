# src/explainiverse/explainers/counterfactual/dice_wrapper.py
"""
Counterfactual Explainer - DiCE-style diverse counterfactual explanations.

Counterfactual explanations answer "What minimal changes would flip the prediction?"

Reference:
    Mothilal, R.K., Sharma, A., & Tan, C. (2020). Explaining Machine Learning
    Classifiers through Diverse Counterfactual Explanations. FAT* 2020.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from scipy.optimize import minimize
from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class CounterfactualExplainer(BaseExplainer):
    """
    Counterfactual explainer using gradient-free optimization.
    
    Generates minimal perturbations that change the model's prediction
    to a desired class (or just a different class).
    
    Attributes:
        model: Model adapter with .predict() method
        training_data: Reference data for constraints
        feature_names: List of feature names
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names
    """
    
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
        random_state: int = 42
    ):
        """
        Initialize the Counterfactual explainer.
        
        Args:
            model: Model adapter with .predict() method
            training_data: Reference data (n_samples, n_features)
            feature_names: List of feature names
            continuous_features: Features that can take continuous values
            categorical_features: Features with discrete values
            feature_ranges: Dict of {feature_name: (min, max)} constraints
            proximity_weight: Weight for proximity loss (closer to original)
            diversity_weight: Weight for diversity among counterfactuals
            random_state: Random seed
        """
        super().__init__(model)
        self.training_data = np.array(training_data)
        self.feature_names = list(feature_names)
        self.continuous_features = continuous_features or feature_names
        self.categorical_features = categorical_features or []
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Compute feature ranges from data if not provided
        if feature_ranges:
            self.feature_ranges = feature_ranges
        else:
            self.feature_ranges = {}
            for idx, name in enumerate(feature_names):
                values = self.training_data[:, idx]
                self.feature_ranges[name] = (float(np.min(values)), float(np.max(values)))
        
        # Compute feature scales for normalization
        self._compute_scales()
    
    def _compute_scales(self):
        """Compute scaling factors for each feature."""
        self.scales = np.zeros(len(self.feature_names))
        for idx, name in enumerate(self.feature_names):
            min_val, max_val = self.feature_ranges.get(name, (0, 1))
            scale = max_val - min_val
            self.scales[idx] = scale if scale > 0 else 1.0
    
    def _get_target_class(
        self,
        instance: np.ndarray,
        desired_class: Optional[int] = None
    ) -> int:
        """Determine the target class for the counterfactual."""
        predictions = self.model.predict(instance.reshape(1, -1))
        
        if predictions.ndim == 2:
            current_class = np.argmax(predictions[0])
            n_classes = predictions.shape[1]
        else:
            current_class = int(predictions[0] > 0.5)
            n_classes = 2
        
        if desired_class is not None:
            return desired_class
        
        # Default: flip to any other class
        if n_classes == 2:
            return 1 - current_class
        else:
            # For multi-class, pick the second most likely class
            probs = predictions[0]
            sorted_classes = np.argsort(probs)[::-1]
            return int(sorted_classes[1]) if sorted_classes[0] == current_class else int(sorted_classes[0])
    
    def _proximity_loss(self, cf: np.ndarray, original: np.ndarray) -> float:
        """Compute normalized distance between counterfactual and original."""
        diff = (cf - original) / self.scales
        return float(np.sum(diff ** 2))
    
    def _validity_loss(self, cf: np.ndarray, target_class: int) -> float:
        """Compute loss for achieving the target class."""
        predictions = self.model.predict(cf.reshape(1, -1))
        
        if predictions.ndim == 2:
            target_prob = predictions[0, target_class]
            return -np.log(target_prob + 1e-10)
        else:
            if target_class == 1:
                return -np.log(predictions[0] + 1e-10)
            else:
                return -np.log(1 - predictions[0] + 1e-10)
    
    def _diversity_loss(self, cfs: List[np.ndarray]) -> float:
        """Compute diversity loss (encourage different counterfactuals)."""
        if len(cfs) < 2:
            return 0.0
        
        total_dist = 0.0
        count = 0
        for i in range(len(cfs)):
            for j in range(i + 1, len(cfs)):
                diff = (cfs[i] - cfs[j]) / self.scales
                total_dist += np.sum(diff ** 2)
                count += 1
        
        return -total_dist / count if count > 0 else 0.0
    
    def _generate_single_counterfactual(
        self,
        instance: np.ndarray,
        target_class: int,
        max_iter: int = 100
    ) -> Optional[np.ndarray]:
        """
        Generate a single counterfactual using optimization.
        """
        # Start from a random perturbation of the instance
        cf = instance.copy()
        cf += self.rng.randn(len(cf)) * 0.1 * self.scales
        
        # Clip to valid ranges
        for idx, name in enumerate(self.feature_names):
            min_val, max_val = self.feature_ranges.get(name, (-np.inf, np.inf))
            cf[idx] = np.clip(cf[idx], min_val, max_val)
        
        def objective(x):
            validity = self._validity_loss(x, target_class)
            proximity = self._proximity_loss(x, instance)
            return validity + self.proximity_weight * proximity
        
        # Define bounds
        bounds = []
        for idx, name in enumerate(self.feature_names):
            min_val, max_val = self.feature_ranges.get(name, (-np.inf, np.inf))
            bounds.append((min_val, max_val))
        
        # Optimize
        result = minimize(
            objective,
            cf,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        cf_result = result.x
        
        # Check if valid (prediction changed)
        predictions = self.model.predict(cf_result.reshape(1, -1))
        if predictions.ndim == 2:
            pred_class = np.argmax(predictions[0])
        else:
            pred_class = int(predictions[0] > 0.5)
        
        if pred_class == target_class:
            return cf_result
        return None
    
    def _generate_diverse_counterfactuals(
        self,
        instance: np.ndarray,
        target_class: int,
        num_counterfactuals: int,
        max_attempts: int = 50
    ) -> List[np.ndarray]:
        """
        Generate multiple diverse counterfactuals.
        """
        counterfactuals = []
        attempts = 0
        
        while len(counterfactuals) < num_counterfactuals and attempts < max_attempts:
            # Add some randomization to encourage diversity
            self.rng = np.random.RandomState(self.random_state + attempts)
            
            cf = self._generate_single_counterfactual(instance, target_class)
            
            if cf is not None:
                # Check if it's diverse enough from existing CFs
                is_diverse = True
                for existing_cf in counterfactuals:
                    diff = np.abs(cf - existing_cf) / self.scales
                    if np.max(diff) < 0.1:  # Too similar
                        is_diverse = False
                        break
                
                if is_diverse:
                    counterfactuals.append(cf)
            
            attempts += 1
        
        return counterfactuals
    
    def explain(
        self,
        instance: np.ndarray,
        num_counterfactuals: int = 3,
        desired_class: Optional[int] = None,
        **kwargs
    ) -> Explanation:
        """
        Generate counterfactual explanations.
        
        Args:
            instance: The instance to explain (1D array)
            num_counterfactuals: Number of diverse counterfactuals to generate
            desired_class: Target class (default: flip to different class)
            
        Returns:
            Explanation object with counterfactuals and changes
        """
        instance = np.array(instance).flatten()
        target_class = self._get_target_class(instance, desired_class)
        
        # Get original prediction
        original_pred = self.model.predict(instance.reshape(1, -1))
        if original_pred.ndim == 2:
            original_class = int(np.argmax(original_pred[0]))
        else:
            original_class = int(original_pred[0] > 0.5)
        
        # Generate counterfactuals
        counterfactuals = self._generate_diverse_counterfactuals(
            instance, target_class, num_counterfactuals
        )
        
        # Compute changes for each counterfactual
        all_changes = []
        for cf in counterfactuals:
            changes = {}
            for idx, name in enumerate(self.feature_names):
                diff = cf[idx] - instance[idx]
                if abs(diff) > 1e-6:
                    changes[name] = {
                        "original": float(instance[idx]),
                        "counterfactual": float(cf[idx]),
                        "change": float(diff)
                    }
            all_changes.append(changes)
        
        # Compute feature importance based on average change magnitude
        feature_importance = {}
        for idx, name in enumerate(self.feature_names):
            total_change = 0.0
            for cf in counterfactuals:
                total_change += abs(cf[idx] - instance[idx]) / self.scales[idx]
            feature_importance[name] = total_change / max(len(counterfactuals), 1)
        
        return Explanation(
            explainer_name="Counterfactual",
            target_class=f"class_{target_class}",
            explanation_data={
                "counterfactuals": [cf.tolist() for cf in counterfactuals],
                "changes": all_changes,
                "original_class": original_class,
                "target_class": target_class,
                "num_generated": len(counterfactuals),
                "feature_attributions": feature_importance
            }
        )
