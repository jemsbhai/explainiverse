# src/explainiverse/explainers/gradient/tcav.py
"""
TCAV - Testing with Concept Activation Vectors.

TCAV provides human-interpretable explanations by quantifying how much
a model's predictions are influenced by high-level concepts. Instead of
attributing importance to individual features, TCAV explains which
human-understandable concepts (e.g., "striped", "furry") are important
for a model's predictions.

Key Components:
    - Concept Activation Vectors (CAVs): Learned direction in activation space
      that separates concept examples from random examples
    - Directional Derivatives: Gradient of model output in CAV direction
    - TCAV Score: Fraction of inputs where concept positively influences prediction
    - Statistical Testing: Significance against random concepts

Reference:
    Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., &
    Sayres, R. (2018). Interpretability Beyond Feature Attribution:
    Quantitative Testing with Concept Activation Vectors (TCAV).
    ICML 2018. https://arxiv.org/abs/1711.11279

Example:
    from explainiverse.explainers.gradient import TCAVExplainer
    from explainiverse.adapters import PyTorchAdapter

    adapter = PyTorchAdapter(model, task="classification")

    explainer = TCAVExplainer(
        model=adapter,
        layer_name="layer3",
        class_names=["zebra", "horse", "dog"]
    )

    # Learn a concept (e.g., "striped")
    explainer.learn_concept(
        concept_name="striped",
        concept_examples=striped_images,
        negative_examples=random_images
    )

    # Compute TCAV score for target class
    result = explainer.compute_tcav_score(
        test_inputs=test_images,
        target_class=0,  # zebra
        concept_name="striped"
    )
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from collections import defaultdict

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation

# Check if sklearn is available for linear classifier
try:
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Check if scipy is available for statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _check_dependencies():
    """Check required dependencies for TCAV."""
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for TCAV. "
            "Install it with: pip install scikit-learn"
        )
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for TCAV statistical testing. "
            "Install it with: pip install scipy"
        )


class ConceptActivationVector:
    """
    Represents a learned Concept Activation Vector (CAV).
    
    A CAV is the normal vector to the hyperplane that separates
    concept examples from random (negative) examples in the
    activation space of a neural network layer.
    
    Attributes:
        concept_name: Human-readable name of the concept
        layer_name: Name of the layer this CAV was trained on
        vector: The CAV direction (normal to separating hyperplane)
        classifier: The trained linear classifier
        accuracy: Classification accuracy on held-out data (Python float)
        metadata: Additional training information
    """
    
    def __init__(
        self,
        concept_name: str,
        layer_name: str,
        vector: np.ndarray,
        classifier: Any,
        accuracy: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.concept_name = concept_name
        self.layer_name = layer_name
        self.vector = vector / np.linalg.norm(vector)  # Normalize
        self.classifier = classifier
        # Ensure accuracy is Python float, not numpy.float64
        # sklearn.metrics.accuracy_score returns numpy.float64
        self.accuracy = float(accuracy)
        self.metadata = metadata or {}
    
    def __repr__(self):
        return (f"CAV(concept='{self.concept_name}', "
                f"layer='{self.layer_name}', "
                f"accuracy={self.accuracy:.3f})")


class TCAVExplainer(BaseExplainer):
    """
    TCAV (Testing with Concept Activation Vectors) explainer.
    
    TCAV explains model predictions using high-level human concepts
    rather than low-level features. It quantifies how sensitive a
    model's predictions are to specific concepts.
    
    The TCAV score for a concept C and class k is the fraction of
    inputs of class k for which the model's prediction increases
    when moving in the direction of concept C.
    
    Attributes:
        model: Model adapter with layer access (PyTorchAdapter)
        layer_name: Target layer for activation extraction
        class_names: List of class names
        concepts: Dictionary of learned CAVs
        random_concepts: Dictionary of random CAVs for statistical testing
    """
    
    def __init__(
        self,
        model,
        layer_name: str,
        class_names: Optional[List[str]] = None,
        cav_classifier: str = "logistic",
        random_seed: int = 42
    ):
        """
        Initialize the TCAV explainer.
        
        Args:
            model: A model adapter with get_layer_output() and
                   get_layer_gradients() methods. Use PyTorchAdapter.
            layer_name: Name of the layer to extract activations from.
                       Use model.list_layers() to see available layers.
            class_names: List of class names for the model's outputs.
            cav_classifier: Type of linear classifier for CAV training:
                           - "logistic": Logistic Regression (default)
                           - "sgd": SGD Classifier (faster for large data)
            random_seed: Random seed for reproducibility.
        """
        _check_dependencies()
        super().__init__(model)
        
        # Validate model capabilities
        if not hasattr(model, 'get_layer_output'):
            raise TypeError(
                "Model adapter must have get_layer_output() method. "
                "Use PyTorchAdapter for PyTorch models."
            )
        if not hasattr(model, 'get_layer_gradients'):
            raise TypeError(
                "Model adapter must have get_layer_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )
        
        self.layer_name = layer_name
        self.class_names = list(class_names) if class_names else None
        self.cav_classifier = cav_classifier
        self.random_seed = random_seed
        
        # Storage for learned concepts
        self.concepts: Dict[str, ConceptActivationVector] = {}
        self.random_concepts: Dict[str, List[ConceptActivationVector]] = defaultdict(list)
        
        # Validate layer exists
        if hasattr(model, 'list_layers'):
            available_layers = model.list_layers()
            if layer_name not in available_layers:
                raise ValueError(
                    f"Layer '{layer_name}' not found. "
                    f"Available layers: {available_layers}"
                )
    
    def _get_activations(self, inputs: np.ndarray) -> np.ndarray:
        """
        Extract activations from the target layer.
        
        Args:
            inputs: Input data (n_samples, ...)
            
        Returns:
            Flattened activations (n_samples, n_features)
        """
        activations = self.model.get_layer_output(inputs, self.layer_name)
        
        # Flatten activations if multi-dimensional (e.g., CNN feature maps)
        if activations.ndim > 2:
            # Global average pooling for spatial dimensions
            # Shape: (batch, channels, height, width) -> (batch, channels)
            activations = activations.mean(axis=tuple(range(2, activations.ndim)))
        
        return activations
    
    def _get_gradients_wrt_layer(
        self,
        inputs: np.ndarray,
        target_class: int
    ) -> np.ndarray:
        """
        Get gradients of output w.r.t. layer activations.
        
        Args:
            inputs: Input data
            target_class: Target class index
            
        Returns:
            Gradients w.r.t. layer activations (n_samples, n_features)
        """
        activations, gradients = self.model.get_layer_gradients(
            inputs, self.layer_name, target_class=target_class
        )
        
        # Flatten gradients if multi-dimensional
        if gradients.ndim > 2:
            gradients = gradients.mean(axis=tuple(range(2, gradients.ndim)))
        
        return gradients
    
    def _train_cav(
        self,
        concept_activations: np.ndarray,
        negative_activations: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, Any, float]:
        """
        Train a CAV (linear classifier) to separate concept from negative examples.
        
        Args:
            concept_activations: Activations for concept examples
            negative_activations: Activations for negative examples
            test_size: Fraction of data to use for accuracy estimation
            
        Returns:
            Tuple of (cav_vector, classifier, accuracy)
            Note: accuracy is returned as Python float, not numpy.float64
        """
        np.random.seed(self.random_seed)
        
        # Prepare training data
        X = np.vstack([concept_activations, negative_activations])
        y = np.array([1] * len(concept_activations) + [0] * len(negative_activations))
        
        # Split for accuracy estimation
        if test_size > 0 and len(X) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_seed, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y
        
        # Train classifier
        if self.cav_classifier == "sgd":
            classifier = SGDClassifier(
                loss='hinge',
                max_iter=1000,
                random_state=self.random_seed,
                n_jobs=-1
            )
        else:  # logistic
            classifier = LogisticRegression(
                max_iter=1000,
                random_state=self.random_seed,
                n_jobs=-1,
                solver='lbfgs'
            )
        
        classifier.fit(X_train, y_train)
        
        # Compute accuracy and convert to Python float
        # accuracy_score returns numpy.float64
        accuracy = float(accuracy_score(y_test, classifier.predict(X_test)))
        
        # Extract CAV (normal vector to separating hyperplane)
        # For linear classifiers, this is the coefficient vector
        cav_vector = classifier.coef_.flatten()
        
        return cav_vector, classifier, accuracy
    
    def learn_concept(
        self,
        concept_name: str,
        concept_examples: np.ndarray,
        negative_examples: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        min_accuracy: float = 0.6
    ) -> ConceptActivationVector:
        """
        Learn a Concept Activation Vector from examples.
        
        The CAV is the direction in activation space that separates
        concept examples from negative (non-concept) examples.
        
        Args:
            concept_name: Human-readable name for the concept.
            concept_examples: Examples that exhibit the concept.
                            Shape: (n_concept, ...) matching model input.
            negative_examples: Examples that don't exhibit the concept.
                             If None, random noise is used (not recommended).
                             Shape: (n_negative, ...) matching model input.
            test_size: Fraction of data to hold out for accuracy estimation.
            min_accuracy: Minimum accuracy for CAV to be considered valid.
                         Low accuracy suggests concept isn't linearly separable.
        
        Returns:
            The learned ConceptActivationVector.
            
        Raises:
            ValueError: If CAV accuracy is below min_accuracy threshold.
        """
        concept_examples = np.array(concept_examples)
        
        if negative_examples is None:
            # Generate random negative examples (not recommended)
            import warnings
            warnings.warn(
                "No negative examples provided. Using random noise. "
                "For meaningful CAVs, provide actual non-concept examples.",
                UserWarning
            )
            negative_examples = np.random.randn(*concept_examples.shape).astype(np.float32)
        else:
            negative_examples = np.array(negative_examples)
        
        # Extract activations
        concept_acts = self._get_activations(concept_examples)
        negative_acts = self._get_activations(negative_examples)
        
        # Train CAV (accuracy is already Python float from _train_cav)
        cav_vector, classifier, accuracy = self._train_cav(
            concept_acts, negative_acts, test_size
        )
        
        if accuracy < min_accuracy:
            raise ValueError(
                f"CAV accuracy ({accuracy:.3f}) is below threshold ({min_accuracy}). "
                f"The concept '{concept_name}' may not be linearly separable in "
                f"layer '{self.layer_name}'. Consider using a different layer "
                f"or providing more/better examples."
            )
        
        # Create CAV object (accuracy is already Python float)
        cav = ConceptActivationVector(
            concept_name=concept_name,
            layer_name=self.layer_name,
            vector=cav_vector,
            classifier=classifier,
            accuracy=accuracy,
            metadata={
                "n_concept_examples": int(len(concept_examples)),
                "n_negative_examples": int(len(negative_examples)),
                "test_size": float(test_size)
            }
        )
        
        # Store the CAV
        self.concepts[concept_name] = cav
        
        return cav
    
    def learn_random_concepts(
        self,
        negative_examples: np.ndarray,
        n_random: int = 10,
        concept_name_prefix: str = "_random"
    ) -> List[ConceptActivationVector]:
        """
        Learn random CAVs for statistical significance testing.
        
        Random CAVs are trained by splitting random examples into
        two arbitrary groups. They serve as a baseline to test
        whether a concept's TCAV score is significantly different
        from random.
        
        Args:
            negative_examples: Pool of examples to sample from.
            n_random: Number of random CAVs to train.
            concept_name_prefix: Prefix for random concept names.
            
        Returns:
            List of random CAVs.
        """
        negative_examples = np.array(negative_examples)
        random_cavs = []
        
        # Get all activations
        all_acts = self._get_activations(negative_examples)
        n_samples = len(all_acts)
        
        for i in range(n_random):
            np.random.seed(self.random_seed + i)
            
            # Randomly split into two groups
            indices = np.random.permutation(n_samples)
            split_point = n_samples // 2
            
            group1_acts = all_acts[indices[:split_point]]
            group2_acts = all_acts[indices[split_point:]]
            
            # Train CAV on arbitrary split
            try:
                cav_vector, classifier, accuracy = self._train_cav(
                    group1_acts, group2_acts, test_size=0.0
                )
                
                cav = ConceptActivationVector(
                    concept_name=f"{concept_name_prefix}_{i}",
                    layer_name=self.layer_name,
                    vector=cav_vector,
                    classifier=classifier,
                    accuracy=accuracy,  # Already Python float from _train_cav
                    metadata={"random_seed": int(self.random_seed + i)}
                )
                
                random_cavs.append(cav)
            except Exception:
                # Skip failed random CAVs
                continue
        
        # Store random CAVs
        self.random_concepts[concept_name_prefix] = random_cavs
        
        return random_cavs
    
    def compute_directional_derivative(
        self,
        inputs: np.ndarray,
        cav: ConceptActivationVector,
        target_class: int
    ) -> np.ndarray:
        """
        Compute directional derivative of predictions in CAV direction.
        
        The directional derivative measures how the model's output for
        the target class changes when moving in the CAV direction.
        
        S_C,k(x) = ∇h_l,k(x) · v_C
        
        where h_l,k is the model's logit for class k at layer l,
        and v_C is the CAV direction.
        
        Args:
            inputs: Input data (n_samples, ...)
            cav: The Concept Activation Vector
            target_class: Target class index
            
        Returns:
            Directional derivatives as numpy array (n_samples,)
            Note: Returns numpy array for efficient computation;
                  individual values should be converted to float if needed.
        """
        # Get gradients w.r.t. layer activations
        gradients = self._get_gradients_wrt_layer(inputs, target_class)
        
        # Compute dot product with CAV
        # S_C,k(x) = ∇h_l,k(x) · v_C
        directional_derivatives = np.dot(gradients, cav.vector)
        
        return directional_derivatives
    
    def compute_tcav_score(
        self,
        test_inputs: np.ndarray,
        target_class: int,
        concept_name: str,
        return_derivatives: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute TCAV score for a concept and target class.
        
        The TCAV score is the fraction of test inputs for which
        the model's prediction for the target class increases
        when moving in the concept direction.
        
        TCAV_C,k = |{x : S_C,k(x) > 0}| / |X|
        
        A score > 0.5 indicates the concept positively influences
        the prediction, while < 0.5 indicates negative influence.
        
        Args:
            test_inputs: Test examples to compute TCAV score over.
            target_class: Target class index.
            concept_name: Name of the concept (must be learned first).
            return_derivatives: If True, also return the directional derivatives.
            
        Returns:
            TCAV score as Python float in [0, 1].
            If return_derivatives=True, returns (score, derivatives) where
            score is Python float and derivatives is numpy array.
        """
        if concept_name not in self.concepts:
            raise ValueError(
                f"Concept '{concept_name}' not found. "
                f"Available concepts: {list(self.concepts.keys())}. "
                f"Use learn_concept() first."
            )
        
        test_inputs = np.array(test_inputs)
        cav = self.concepts[concept_name]
        
        # Compute directional derivatives
        derivatives = self.compute_directional_derivative(
            test_inputs, cav, target_class
        )
        
        # TCAV score = fraction with positive derivative
        # np.mean returns numpy.float64, convert to Python float
        tcav_score = float(np.mean(derivatives > 0))
        
        if return_derivatives:
            return tcav_score, derivatives
        return tcav_score
    
    def statistical_significance_test(
        self,
        test_inputs: np.ndarray,
        target_class: int,
        concept_name: str,
        n_random: int = 10,
        negative_examples: Optional[np.ndarray] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test statistical significance of TCAV score against random concepts.
        
        Performs a two-sided t-test comparing the concept's TCAV score
        against the distribution of random TCAV scores.
        
        Args:
            test_inputs: Test examples to compute TCAV scores over.
            target_class: Target class index.
            concept_name: Name of the concept to test.
            n_random: Number of random concepts for comparison.
            negative_examples: Examples for training random concepts.
                             If None, uses test_inputs (not ideal).
            alpha: Significance level for the test.
            
        Returns:
            Dictionary containing (all values are Python native types):
            - tcav_score: The concept's TCAV score (float)
            - random_scores: List of random TCAV scores (list of float)
            - random_mean: Mean of random scores (float)
            - random_std: Std of random scores (float)
            - t_statistic: t-statistic from t-test (float)
            - p_value: p-value from t-test (float)
            - significant: Whether the result is significant at level alpha (bool)
            - effect_size: Cohen's d effect size (float)
            - alpha: The significance level used (float)
        """
        test_inputs = np.array(test_inputs)
        
        # Compute concept TCAV score (already Python float from compute_tcav_score)
        concept_score = self.compute_tcav_score(
            test_inputs, target_class, concept_name
        )
        
        # Train random CAVs if not already done
        if negative_examples is None:
            negative_examples = test_inputs
        
        random_prefix = f"_random_{concept_name}_{target_class}"
        
        if random_prefix not in self.random_concepts or \
           len(self.random_concepts[random_prefix]) < n_random:
            self.learn_random_concepts(
                negative_examples,
                n_random=n_random,
                concept_name_prefix=random_prefix
            )
        
        random_cavs = self.random_concepts[random_prefix][:n_random]
        
        # Compute random TCAV scores
        random_scores = []
        for random_cav in random_cavs:
            derivatives = self.compute_directional_derivative(
                test_inputs, random_cav, target_class
            )
            # Convert to Python float immediately
            random_score = float(np.mean(derivatives > 0))
            random_scores.append(random_score)
        
        # Convert to numpy array for statistical computations
        random_scores_array = np.array(random_scores)
        
        # Perform one-sample t-test against concept score
        # scipy.stats.ttest_1samp returns numpy scalars
        t_stat_np, p_value_np = stats.ttest_1samp(random_scores_array, concept_score)
        
        # Convert scipy/numpy results to Python native types
        t_stat = float(t_stat_np)
        p_value = float(p_value_np)
        
        # Compute effect size (Cohen's d)
        random_std = float(random_scores_array.std())
        random_mean = float(random_scores_array.mean())
        
        if random_std > 0:
            effect_size = float((concept_score - random_mean) / random_std)
        else:
            # Handle zero std case
            if concept_score != random_mean:
                effect_size = float('inf') if concept_score > random_mean else float('-inf')
            else:
                effect_size = 0.0
        
        # Compute significance as Python bool (not numpy.bool_)
        significant = bool(p_value < alpha)
        
        return {
            "tcav_score": concept_score,  # Already Python float
            "random_scores": random_scores,  # Already list of Python floats
            "random_mean": random_mean,
            "random_std": random_std,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": significant,
            "effect_size": effect_size,
            "alpha": float(alpha)
        }
    
    def explain(
        self,
        test_inputs: np.ndarray,
        target_class: Optional[int] = None,
        concept_names: Optional[List[str]] = None,
        run_significance_test: bool = False,
        negative_examples: Optional[np.ndarray] = None,
        n_random: int = 10
    ) -> Explanation:
        """
        Generate TCAV explanation for test inputs.
        
        Computes TCAV scores for all (or specified) concepts
        and optionally runs statistical significance tests.
        
        Args:
            test_inputs: Input examples to explain.
            target_class: Target class to explain. If None, uses
                         the most common predicted class.
            concept_names: List of concepts to include. If None,
                          uses all learned concepts.
            run_significance_test: Whether to run statistical tests.
            negative_examples: Examples for random CAVs (for significance test).
            n_random: Number of random concepts for significance test.
            
        Returns:
            Explanation object with TCAV scores for each concept.
        """
        test_inputs = np.array(test_inputs)
        
        if len(self.concepts) == 0:
            raise ValueError(
                "No concepts learned. Use learn_concept() first."
            )
        
        # Determine target class
        if target_class is None:
            predictions = self.model.predict(test_inputs)
            if predictions.ndim > 1:
                target_class = int(np.argmax(np.bincount(
                    np.argmax(predictions, axis=1)
                )))
            else:
                target_class = 0
        
        # Determine concepts to analyze
        if concept_names is None:
            concept_names = list(self.concepts.keys())
        
        # Compute TCAV scores for each concept
        tcav_scores = {}
        significance_results = {}
        
        for concept_name in concept_names:
            if concept_name not in self.concepts:
                continue
            
            # compute_tcav_score returns Python float
            score, derivatives = self.compute_tcav_score(
                test_inputs, target_class, concept_name,
                return_derivatives=True
            )
            
            tcav_scores[concept_name] = {
                "score": score,  # Already Python float
                "cav_accuracy": self.concepts[concept_name].accuracy,  # Already Python float
                "positive_count": int(np.sum(derivatives > 0)),
                "total_count": int(len(derivatives))
            }
            
            # Optionally run significance test
            if run_significance_test:
                neg_examples = negative_examples if negative_examples is not None else test_inputs
                # statistical_significance_test returns dict with Python native types
                sig_result = self.statistical_significance_test(
                    test_inputs, target_class, concept_name,
                    n_random=n_random,
                    negative_examples=neg_examples
                )
                significance_results[concept_name] = sig_result
        
        # Determine class name
        if self.class_names and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}"
        
        explanation_data = {
            "tcav_scores": tcav_scores,
            "target_class": int(target_class),
            "n_test_inputs": int(len(test_inputs)),
            "layer_name": self.layer_name,
            "concepts_analyzed": list(concept_names)  # Ensure it's a list
        }
        
        if run_significance_test:
            explanation_data["significance_tests"] = significance_results
        
        return Explanation(
            explainer_name="TCAV",
            target_class=label_name,
            explanation_data=explanation_data
        )
    
    def explain_batch(
        self,
        X: np.ndarray,
        target_class: Optional[int] = None,
        **kwargs
    ) -> List[Explanation]:
        """
        TCAV typically explains batches together, not individually.
        
        For TCAV, it makes more sense to analyze a batch of inputs
        together to compute meaningful TCAV scores. This method
        returns a single explanation for the batch.
        
        Args:
            X: Batch of inputs.
            target_class: Target class to explain.
            **kwargs: Additional arguments passed to explain().
            
        Returns:
            List containing a single Explanation for the batch.
        """
        return [self.explain(X, target_class=target_class, **kwargs)]
    
    def get_most_influential_concepts(
        self,
        test_inputs: np.ndarray,
        target_class: int,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get the most influential concepts for the target class.
        
        Ranks concepts by how much they positively influence
        the model's prediction for the target class.
        
        Args:
            test_inputs: Test examples.
            target_class: Target class index.
            top_k: Number of top concepts to return.
            
        Returns:
            List of (concept_name, tcav_score) tuples, sorted by score descending.
            All scores are Python floats.
        """
        scores = []
        
        for concept_name in self.concepts:
            # compute_tcav_score returns Python float
            score = self.compute_tcav_score(
                test_inputs, target_class, concept_name
            )
            scores.append((concept_name, score))
        
        # Sort by score (higher = more positive influence)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def compare_concepts(
        self,
        test_inputs: np.ndarray,
        target_classes: List[int],
        concept_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Compare TCAV scores across multiple target classes.
        
        Useful for understanding which concepts are important
        for different classes.
        
        Args:
            test_inputs: Test examples.
            target_classes: List of class indices to compare.
            concept_names: Concepts to analyze (default: all).
            
        Returns:
            Dictionary mapping concept names to {class_idx: tcav_score}.
            All scores are Python floats.
        """
        if concept_names is None:
            concept_names = list(self.concepts.keys())
        
        results: Dict[str, Dict[int, float]] = {}
        
        for concept_name in concept_names:
            results[concept_name] = {}
            for class_idx in target_classes:
                # compute_tcav_score returns Python float
                score = self.compute_tcav_score(
                    test_inputs, class_idx, concept_name
                )
                results[concept_name][class_idx] = score
        
        return results
    
    def list_concepts(self) -> List[str]:
        """List all learned concept names."""
        return list(self.concepts.keys())
    
    def get_concept(self, concept_name: str) -> ConceptActivationVector:
        """Get a specific CAV by name."""
        if concept_name not in self.concepts:
            raise ValueError(f"Concept '{concept_name}' not found.")
        return self.concepts[concept_name]
    
    def remove_concept(self, concept_name: str) -> None:
        """Remove a learned concept."""
        if concept_name in self.concepts:
            del self.concepts[concept_name]
