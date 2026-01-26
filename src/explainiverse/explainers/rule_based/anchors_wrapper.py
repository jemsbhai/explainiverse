# src/explainiverse/explainers/rule_based/anchors_wrapper.py
"""
Anchors Explainer - High-precision rule-based explanations.

Anchors are if-then rules that "anchor" a prediction, meaning the prediction
remains the same regardless of other features' values (with high probability).

Reference:
    Ribeiro, M.T., Singh, S., & Guestrin, C. (2018). Anchors: High-Precision
    Model-Agnostic Explanations. AAAI 2018.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from itertools import combinations
from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation


class AnchorsExplainer(BaseExplainer):
    """
    Anchors explainer for rule-based explanations.
    
    Generates if-then rules that explain individual predictions with
    high precision (the rule holds with high probability). Uses beam search
    to efficiently explore the space of possible anchors, optimizing for
    both precision (rule reliability) and coverage (rule generality).
    
    The algorithm:
    1. Discretizes continuous features into interpretable bins
    2. Uses beam search to find minimal feature subsets (anchors)
    3. Evaluates precision via perturbation sampling
    4. Returns the shortest anchor meeting the precision threshold
    
    Attributes:
        model: Model adapter with .predict() method
        training_data: Reference data for generating perturbations
        feature_names: List of feature names
        class_names: List of class names
        threshold: Minimum precision for the anchor (default: 0.95)
        n_samples: Number of samples for precision estimation (default: 1000)
        beam_size: Number of candidates in beam search (default: 4)
    """
    
    def __init__(
        self,
        model,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        threshold: float = 0.95,
        n_samples: int = 1000,
        beam_size: int = 4,
        max_anchor_size: int = None,
        discretizer: str = "quartile",
        random_state: int = 42
    ):
        """
        Initialize the Anchors explainer.
        
        Args:
            model: Model adapter with .predict() method
            training_data: Reference data (n_samples, n_features)
            feature_names: List of feature names
            class_names: List of class names
            threshold: Minimum precision for a valid anchor
            n_samples: Number of perturbation samples
            beam_size: Number of candidates to keep in beam search
            max_anchor_size: Maximum number of conditions in anchor
            discretizer: How to discretize continuous features ("quartile", "decile")
            random_state: Random seed
        """
        super().__init__(model)
        self.training_data = np.array(training_data)
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.threshold = threshold
        self.n_samples = n_samples
        self.beam_size = beam_size
        self.max_anchor_size = max_anchor_size or len(feature_names)
        self.discretizer = discretizer
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Pre-compute feature statistics for discretization
        self._compute_discretization()
    
    def _compute_discretization(self):
        """Pre-compute discretization bins for each feature."""
        self.bins = {}
        self.bin_labels = {}
        
        if self.discretizer == "quartile":
            percentiles = [25, 50, 75]
        elif self.discretizer == "decile":
            percentiles = list(range(10, 100, 10))
        else:
            percentiles = [25, 50, 75]
        
        for idx in range(self.training_data.shape[1]):
            values = self.training_data[:, idx]
            bins = np.percentile(values, percentiles)
            bins = np.unique(bins)  # Remove duplicates
            self.bins[idx] = bins
            
            # Create human-readable labels
            labels = []
            if len(bins) == 0:
                labels = [f"{self.feature_names[idx]} = any"]
            else:
                labels.append(f"{self.feature_names[idx]} <= {bins[0]:.2f}")
                for i in range(len(bins) - 1):
                    labels.append(f"{bins[i]:.2f} < {self.feature_names[idx]} <= {bins[i+1]:.2f}")
                labels.append(f"{self.feature_names[idx]} > {bins[-1]:.2f}")
            self.bin_labels[idx] = labels
    
    def _discretize_value(self, value: float, feature_idx: int) -> int:
        """Discretize a single value into a bin index."""
        bins = self.bins[feature_idx]
        if len(bins) == 0:
            return 0
        return int(np.searchsorted(bins, value))
    
    def _discretize_instance(self, instance: np.ndarray) -> np.ndarray:
        """Discretize an entire instance."""
        return np.array([
            self._discretize_value(instance[i], i)
            for i in range(len(instance))
        ])
    
    def _get_condition_label(self, feature_idx: int, bin_idx: int) -> str:
        """Get human-readable label for a condition."""
        labels = self.bin_labels[feature_idx]
        if bin_idx < len(labels):
            return labels[bin_idx]
        return f"{self.feature_names[feature_idx]} in bin {bin_idx}"
    
    def _generate_perturbations(
        self,
        instance: np.ndarray,
        anchor: List[int],
        n_samples: int
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
                    in_bin = np.array([
                        self._discretize_value(v, feat_idx) == target_bin
                        for v in feature_values
                    ])
                    if np.any(in_bin):
                        valid_values = feature_values[in_bin]
                        sample[feat_idx] = self.rng.choice(valid_values)
                    else:
                        sample[feat_idx] = instance[feat_idx]
            
            perturbations[i] = sample
        
        return perturbations
    
    def _compute_precision(
        self,
        instance: np.ndarray,
        anchor: List[int],
        target_class: int
    ) -> Tuple[float, int]:
        """
        Compute the precision of an anchor.
        
        Precision = P(prediction = target_class | anchor conditions hold)
        
        Returns:
            Tuple of (precision, coverage_count)
        """
        perturbations = self._generate_perturbations(
            instance, anchor, self.n_samples
        )
        
        predictions = self.model.predict(perturbations)
        
        if predictions.ndim == 2:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        matches = np.sum(pred_classes == target_class)
        precision = matches / len(pred_classes)
        
        return precision, matches
    
    def _compute_coverage(self, anchor: List[int], instance: np.ndarray) -> float:
        """
        Compute the coverage of an anchor (fraction of data matching conditions).
        """
        disc_instance = self._discretize_instance(instance)
        
        matches = 0
        for sample in self.training_data:
            disc_sample = self._discretize_instance(sample)
            if all(disc_sample[i] == disc_instance[i] for i in anchor):
                matches += 1
        
        return matches / len(self.training_data)
    
    def _beam_search(
        self,
        instance: np.ndarray,
        target_class: int
    ) -> Tuple[List[int], float, float]:
        """
        Use beam search to find the best anchor.
        
        Returns:
            Tuple of (anchor_features, precision, coverage)
        """
        n_features = len(instance)
        
        # Start with empty anchor
        candidates = [
            ([], 1.0, 1.0)  # (anchor, precision, coverage)
        ]
        
        best_anchor = ([], 0.0, 1.0)
        
        for _ in range(self.max_anchor_size):
            new_candidates = []
            
            for anchor, _, _ in candidates:
                # Try adding each unused feature
                for feat_idx in range(n_features):
                    if feat_idx in anchor:
                        continue
                    
                    new_anchor = anchor + [feat_idx]
                    precision, _ = self._compute_precision(
                        instance, new_anchor, target_class
                    )
                    coverage = self._compute_coverage(new_anchor, instance)
                    
                    new_candidates.append((new_anchor, precision, coverage))
                    
                    # Check if this is a valid anchor
                    if precision >= self.threshold:
                        if coverage > best_anchor[2] or \
                           (coverage == best_anchor[2] and len(new_anchor) < len(best_anchor[0])):
                            best_anchor = (new_anchor, precision, coverage)
            
            if not new_candidates:
                break
            
            # Keep top candidates by precision (prefer smaller anchors for ties)
            new_candidates.sort(
                key=lambda x: (x[1], -len(x[0]), x[2]),
                reverse=True
            )
            candidates = new_candidates[:self.beam_size]
            
            # Early stopping if we found a good anchor
            if best_anchor[1] >= self.threshold:
                # Check if we can improve coverage
                can_improve = any(c[1] >= self.threshold and c[2] > best_anchor[2] 
                                  for c in candidates)
                if not can_improve:
                    break
        
        if best_anchor[0]:
            return best_anchor
        elif candidates:
            # Return best candidate even if below threshold
            return max(candidates, key=lambda x: x[1])
        else:
            return ([], 0.0, 1.0)
    
    def explain(self, instance: np.ndarray, **kwargs) -> Explanation:
        """
        Generate an anchor explanation for the given instance.
        
        Args:
            instance: The instance to explain (1D array)
            
        Returns:
            Explanation object with anchor rules
        """
        instance = np.array(instance).flatten()
        
        # Get the model's prediction
        predictions = self.model.predict(instance.reshape(1, -1))
        if predictions.ndim == 2:
            target_class = np.argmax(predictions[0])
        else:
            target_class = int(predictions[0])
        
        target_name = self.class_names[target_class] if target_class < len(self.class_names) else f"class_{target_class}"
        
        # Find anchor using beam search
        anchor_features, precision, coverage = self._beam_search(instance, target_class)
        
        # Convert to human-readable rules
        disc_instance = self._discretize_instance(instance)
        rules = [
            self._get_condition_label(feat_idx, int(disc_instance[feat_idx]))
            for feat_idx in anchor_features
        ]
        
        return Explanation(
            explainer_name="Anchors",
            target_class=target_name,
            explanation_data={
                "rules": rules,
                "precision": float(precision),
                "coverage": float(coverage),
                "anchor_features": [self.feature_names[i] for i in anchor_features],
                "anchor_indices": anchor_features,
                "feature_attributions": {
                    self.feature_names[i]: 1.0 / (idx + 1)
                    for idx, i in enumerate(anchor_features)
                } if anchor_features else {}
            }
        )
