# src/explainiverse/explainers/gradient/tcav.py
"""
TCAV - Testing with Concept Activation Vectors.

TCAV estimates directional sensitivity to user-defined concept vectors at a
chosen bottleneck. Instead of attributing scores to input features, it reports
how often the selected output's directional derivative is positive. A TCAV
score does not by itself prove interpretability or causal concept influence.
The paper-defined score differentiates a class logit. Adapters that expose a
different score space are retained as explicitly identified variants; set
``require_logit_scores=True`` to reject those variants.

Implemented components:
    - Concept Activation Vectors (CAVs): Learned direction in activation space
      that separates concept examples from random examples
    - Directional Derivatives: Gradient of model output in CAV direction
    - TCAV Score: Fraction of inputs with a positive selected-output
      directional derivative along the learned CAV
    - Repeated-run diagnostic: a two-sided Welch test comparing concept-vs-
      random TCAV scores with random-vs-random TCAV scores. The returned
      threshold decision is per-concept, uncorrected, and not presented as a
      confirmatory significance guarantee.

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

import copy
from itertools import combinations
from numbers import Integral
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, overload

import numpy as np

from explainiverse.core.explainer import BaseExplainer, synchronized_explainer_method
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence
from explainiverse.explainers.gradient._input import scale_safe_product_sum
from explainiverse.explainers.gradient._model_state import preserve_adapter_model_eval

# Check if sklearn is available for linear classifier
try:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

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
    """Check the dependency required for learning a linear CAV."""
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for TCAV. Install it with: pip install scikit-learn"
        )


def _check_scipy():
    """Check the optional dependency used only by repeated-run inference."""
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for TCAV repeated-run inference. "
            "Install it with: pip install scipy"
        )


def _normalize_cav_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize one finite non-zero CAV without norm under/overflow."""

    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise ValueError("CAV vector must be non-zero")
    scaled = vector / scale
    scaled_norm_squared = float(scale_safe_product_sum(scaled, scaled, axis=0))
    scaled_norm = float(np.sqrt(scaled_norm_squared))
    if not np.isfinite(scaled_norm) or scaled_norm == 0.0:
        raise ValueError("CAV vector must have a finite non-zero direction")
    normalized = scaled / scaled_norm
    if not np.all(np.isfinite(normalized)):
        raise ValueError("CAV vector normalization produced non-finite values")
    return normalized


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
        accuracy: Classification accuracy on held-out data when possible, or
            training data for very small sets (identified in metadata)
        metadata: Additional training information
    """

    _concept_name: str
    _layer_name: str
    _vector: np.ndarray
    _classifier: Any
    _accuracy: float
    _metadata: Dict[str, Any]
    _sealed: bool

    __slots__ = (
        "_concept_name",
        "_layer_name",
        "_vector",
        "_classifier",
        "_accuracy",
        "_metadata",
        "_sealed",
    )

    def __init__(
        self,
        concept_name: str,
        layer_name: str,
        vector: np.ndarray,
        classifier: Any,
        accuracy: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if not isinstance(layer_name, str) or not layer_name.strip():
            raise ValueError("layer_name must be a non-empty string")

        vector_array = as_real_array(
            vector,
            name="CAV vector",
            dtype=np.float64,
            require_finite=True,
        )
        if vector_array.ndim != 1 or vector_array.size == 0:
            raise ValueError("CAV vector must be a non-empty one-dimensional array")
        normalized_vector = _normalize_cav_vector(vector_array)

        accuracy_value = float(accuracy)
        if not np.isfinite(accuracy_value) or not 0.0 <= accuracy_value <= 1.0:
            raise ValueError("accuracy must be a finite value in [0, 1]")

        try:
            owned_classifier = copy.deepcopy(classifier)
            owned_metadata = copy.deepcopy(dict(metadata) if metadata is not None else {})
        except Exception as error:
            raise TypeError(
                "CAV classifier and metadata must support defensive copying so learned "
                "concept state cannot alias caller-owned mutable objects"
            ) from error
        owned_vector = normalized_vector.copy()
        owned_vector.setflags(write=False)
        object.__setattr__(self, "_concept_name", concept_name)
        object.__setattr__(self, "_layer_name", layer_name)
        object.__setattr__(self, "_vector", owned_vector)
        object.__setattr__(self, "_classifier", owned_classifier)
        object.__setattr__(self, "_accuracy", accuracy_value)
        object.__setattr__(self, "_metadata", owned_metadata)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ConceptActivationVector snapshots are immutable")
        object.__setattr__(self, name, value)

    @property
    def concept_name(self) -> str:
        return self._concept_name

    @property
    def layer_name(self) -> str:
        return self._layer_name

    @property
    def vector(self) -> np.ndarray:
        # A read-only copy prevents ``setflags(write=True)`` from exposing the
        # internally owned CAV direction.
        snapshot = self._vector.copy()
        snapshot.setflags(write=False)
        return snapshot

    @property
    def classifier(self) -> Any:
        return copy.deepcopy(self._classifier)

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(copy.deepcopy(self._metadata))

    def clone(self) -> "ConceptActivationVector":
        """Return a fully independent, bit-exact public snapshot."""

        try:
            owned_classifier = copy.deepcopy(self._classifier)
            owned_metadata = copy.deepcopy(self._metadata)
        except Exception as error:
            raise TypeError(
                "CAV classifier and metadata must support defensive copying so learned "
                "concept state cannot alias caller-owned mutable objects"
            ) from error
        owned_vector = self._vector.copy()
        owned_vector.setflags(write=False)
        snapshot = object.__new__(ConceptActivationVector)
        object.__setattr__(snapshot, "_concept_name", self._concept_name)
        object.__setattr__(snapshot, "_layer_name", self._layer_name)
        object.__setattr__(snapshot, "_vector", owned_vector)
        object.__setattr__(snapshot, "_classifier", owned_classifier)
        object.__setattr__(snapshot, "_accuracy", self._accuracy)
        object.__setattr__(snapshot, "_metadata", owned_metadata)
        object.__setattr__(snapshot, "_sealed", True)
        return snapshot

    def __repr__(self):
        return (
            f"CAV(concept='{self.concept_name}', "
            f"layer='{self.layer_name}', "
            f"accuracy={self.accuracy:.3f})"
        )


class TCAVExplainer(BaseExplainer):
    """
    TCAV (Testing with Concept Activation Vectors) explainer.

    TCAV explains a class-level set of examples using high-level human
    concepts rather than low-level input features. Its score is a global
    aggregate over the supplied target-class example set, not a local
    explanation of one row.

    The TCAV score for a concept C and class k is the fraction of
    inputs of class k for which the model's prediction increases
    when moving in the direction of concept C.

    Attributes:
        model: Model adapter with layer access (PyTorchAdapter)
        layer_name: Target layer for activation extraction
        class_names: List of class names
        concepts: Dictionary of learned CAVs
        random_concepts: Dictionary of learned random-baseline CAVs
    """

    def __init__(
        self,
        model,
        layer_name: str,
        class_names: Optional[List[str]] = None,
        cav_classifier: str = "logistic",
        random_seed: int = 42,
        require_logit_scores: bool = False,
        layer_occurrence: Optional[int] = None,
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
                           - "sgd": SGD Classifier
            random_seed: Random seed for reproducibility.
            require_logit_scores: If True, enforce the paper-defined class-
                logit derivative contract. If False, other adapter-selected
                score spaces remain available but are labeled as variants in
                returned metadata.
            layer_occurrence: Explicit zero-based execution occurrence when
                the named layer is reused in one forward pass. ``None`` keeps
                the adapter's fail-closed repeated-layer default.
        """
        _check_dependencies()
        super().__init__(model)

        # Validate model capabilities
        if not hasattr(model, "get_layer_output"):
            raise TypeError(
                "Model adapter must have get_layer_output() method. "
                "Use PyTorchAdapter for PyTorch models."
            )
        if not hasattr(model, "get_layer_gradients"):
            raise TypeError(
                "Model adapter must have get_layer_gradients() method. "
                "Use PyTorchAdapter for PyTorch models."
            )

        if hasattr(model, "task") and model.task != "classification":
            raise ValueError("TCAVExplainer currently supports classification models only")
        if not isinstance(layer_name, str) or not layer_name.strip():
            raise ValueError("layer_name must be a non-empty string")
        if cav_classifier not in {"logistic", "sgd"}:
            raise ValueError("cav_classifier must be 'logistic' or 'sgd'")
        random_seed = self._validate_random_seed(random_seed)
        if not isinstance(require_logit_scores, (bool, np.bool_)):
            raise TypeError("require_logit_scores must be a boolean")
        if layer_occurrence is not None:
            if not isinstance(layer_occurrence, Integral) or isinstance(layer_occurrence, bool):
                raise TypeError("layer_occurrence must be a non-negative integer or None")
            if int(layer_occurrence) < 0:
                raise ValueError("layer_occurrence must be a non-negative integer or None")
        validated_classes = validate_name_sequence(
            class_names,
            name="class_names",
            allow_none=True,
        )

        self.layer_name: str = layer_name
        self.class_names: Optional[List[str]] = validated_classes
        self.cav_classifier: str = cav_classifier
        self.random_seed: int = random_seed
        self.require_logit_scores: bool = bool(require_logit_scores)
        self.layer_occurrence: Optional[int] = (
            None if layer_occurrence is None else int(layer_occurrence)
        )
        self.last_target_score_space: Optional[str] = None
        self.last_tcav_variant: Optional[str] = None
        self.last_tcav_is_canonical: Optional[bool] = None

        if self.require_logit_scores:
            declared_space = getattr(model, "raw_model_output_space", "unspecified")
            if declared_space != "logit":
                raise ValueError(
                    "Canonical TCAV requires an adapter that declares raw class-logit outputs"
                )
            requested_space = getattr(model, "gradient_output", "model")
            if requested_space != "model":
                raise ValueError(
                    "Canonical TCAV requires gradient_output='model' for class-logit derivatives"
                )

        # Storage for learned concepts
        self._concepts: Dict[str, ConceptActivationVector] = {}
        self._random_concepts: Dict[str, Tuple[ConceptActivationVector, ...]] = {}
        # Repeated TCAV runs must retrain the named concept against multiple
        # random counterexample sets. Keep the already-extracted positive
        # bottleneck activations private so inference never substitutes the
        # class test inputs for concept examples.
        self._concept_activations: Dict[str, np.ndarray] = {}

        # Validate layer exists
        if hasattr(model, "list_layers"):
            available_layers = model.list_layers()
            if layer_name not in available_layers:
                raise ValueError(
                    f"Layer '{layer_name}' not found. " f"Available layers: {available_layers}"
                )

    @property
    def concepts(self) -> Mapping[str, ConceptActivationVector]:
        """Immutable, defensively copied snapshot of learned concepts."""

        with self._instance_lock:
            return MappingProxyType({name: cav.clone() for name, cav in self._concepts.items()})

    @property
    def random_concepts(self) -> Mapping[str, Tuple[ConceptActivationVector, ...]]:
        """Immutable, defensively copied snapshot of random CAV collections."""

        with self._instance_lock:
            return MappingProxyType(
                {
                    name: tuple(cav.clone() for cav in cavs)
                    for name, cavs in self._random_concepts.items()
                }
            )

    @staticmethod
    def _validate_random_seed(value, *, name: str = "random_seed") -> int:
        """Normalize a seed accepted by NumPy RandomState and scikit-learn."""
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        seed = int(value)
        maximum = int(np.iinfo(np.uint32).max)
        if seed < 0 or seed > maximum:
            raise ValueError(f"{name} must be in [0, {maximum}]")
        return seed

    def _derived_seed(self, offset: int) -> int:
        """Derive a reproducible RandomState seed without range overflow."""
        modulus = int(np.iinfo(np.uint32).max) + 1
        return (self.random_seed + int(offset)) % modulus

    def _get_activations(self, inputs: np.ndarray) -> np.ndarray:
        """
        Extract activations from the target layer.

        Args:
            inputs: Input data (n_samples, ...)

        Returns:
            Flattened activations (n_samples, n_features)
        """
        input_array = as_real_array(
            inputs,
            name="inputs",
            require_finite=True,
        )
        if input_array.ndim == 0 or len(input_array) == 0:
            raise ValueError("inputs must contain at least one example")

        with preserve_adapter_model_eval(self.model, preserve_gradients=False):
            occurrence_kwargs = (
                {} if self.layer_occurrence is None else {"occurrence": self.layer_occurrence}
            )
            activations = as_real_array(
                self.model.get_layer_output(
                    input_array,
                    self.layer_name,
                    **occurrence_kwargs,
                ),
                name="bottleneck activations",
                dtype=np.float64,
                require_finite=True,
            )
        if activations.ndim == 0 or activations.shape[0] != len(input_array):
            raise ValueError("get_layer_output() must return one activation tensor per input")
        # The reference TCAV implementation reshapes each complete bottleneck
        # tensor to a vector. Pooling changes both the CAV space and its
        # directional derivative, so it is deliberately not done here.
        activations = activations.reshape(len(input_array), -1)
        if activations.shape[1] == 0 or not np.all(np.isfinite(activations)):
            raise ValueError("bottleneck activations must be non-empty and finite")
        return activations

    def _get_gradients_wrt_layer(self, inputs: np.ndarray, target_class: int) -> np.ndarray:
        """
        Get gradients of output w.r.t. layer activations.

        Args:
            inputs: Input data
            target_class: Target class index

        Returns:
            Gradients w.r.t. layer activations (n_samples, n_features)
        """
        input_array = as_real_array(
            inputs,
            name="inputs",
            require_finite=True,
        )
        if input_array.ndim == 0 or len(input_array) == 0:
            raise ValueError("inputs must contain at least one example")

        with preserve_adapter_model_eval(self.model):
            occurrence_kwargs = (
                {} if self.layer_occurrence is None else {"occurrence": self.layer_occurrence}
            )
            _, gradients = self.model.get_layer_gradients(
                input_array,
                self.layer_name,
                target_class=target_class,
                **occurrence_kwargs,
            )
        score_space = getattr(self.model, "last_gradient_output_space", None) or "adapter_defined"
        declared_raw_space = getattr(self.model, "raw_model_output_space", "unspecified")
        canonical = bool(score_space == "model" and declared_raw_space == "logit")
        if canonical:
            variant = "canonical_class_logit_tcav"
        elif score_space == "prediction":
            variant = "prediction_space_directional_derivative_variant"
        else:
            variant = "adapter_score_directional_derivative_variant"
        self.last_target_score_space = str(score_space)
        self.last_tcav_variant = variant
        self.last_tcav_is_canonical = canonical
        if self.require_logit_scores and not canonical:
            raise ValueError(
                "Canonical TCAV requires effective class-logit gradients; "
                f"the adapter used {score_space!r} scores"
            )
        gradients = as_real_array(
            gradients,
            name="bottleneck gradients",
            dtype=np.float64,
            require_finite=True,
        )
        if gradients.ndim == 0 or gradients.shape[0] != len(input_array):
            raise ValueError("get_layer_gradients() must return one gradient tensor per input")
        gradients = gradients.reshape(len(input_array), -1)
        if gradients.shape[1] == 0 or not np.all(np.isfinite(gradients)):
            raise ValueError("bottleneck gradients must be non-empty and finite")
        return gradients

    def _train_cav(
        self,
        concept_activations: np.ndarray,
        negative_activations: np.ndarray,
        test_size: float = 0.2,
        random_seed: Optional[int] = None,
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
        seed = self.random_seed if random_seed is None else self._validate_random_seed(random_seed)
        if not 0.0 <= float(test_size) < 1.0:
            raise ValueError("test_size must be in [0, 1)")

        concept_activations = as_real_array(
            concept_activations,
            name="concept activations",
            dtype=np.float64,
            require_finite=True,
        )
        negative_activations = as_real_array(
            negative_activations,
            name="negative activations",
            dtype=np.float64,
            require_finite=True,
        )
        if concept_activations.ndim != 2 or negative_activations.ndim != 2:
            raise ValueError("CAV activations must be two-dimensional")
        if concept_activations.shape[1] != negative_activations.shape[1]:
            raise ValueError("Concept and negative activations must have equal dimensions")
        if len(concept_activations) < 2 or len(negative_activations) < 2:
            raise ValueError("At least two examples per CAV group are required")
        if not np.all(np.isfinite(concept_activations)) or not np.all(
            np.isfinite(negative_activations)
        ):
            raise ValueError("CAV activations must contain only finite values")

        # Match the reference implementation's balanced training set. A local
        # RNG makes subsampling reproducible without mutating NumPy's process-
        # global random state.
        n_per_group = min(len(concept_activations), len(negative_activations))
        rng = np.random.RandomState(seed)
        concept_indices = rng.permutation(len(concept_activations))[:n_per_group]
        negative_indices = rng.permutation(len(negative_activations))[:n_per_group]
        X = np.vstack(
            [concept_activations[concept_indices], negative_activations[negative_indices]]
        )
        y = np.array([1] * n_per_group + [0] * n_per_group)

        # Split for accuracy estimation whenever a stratified split can leave
        # at least one member of each class on both sides. Tiny requested
        # holdouts that contain fewer than two total rows cannot estimate both
        # classes, so those datasets fall back to explicitly labelled training
        # accuracy instead of presenting a resubstitution score as held out.
        n_test = int(np.ceil(len(X) * float(test_size))) if test_size > 0 else 0
        if test_size > 0 and n_test >= 2:
            if n_test < 2 or len(X) - n_test < 2:
                raise ValueError(
                    "test_size must leave at least one example from each class "
                    "in both train and test partitions"
                )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
            accuracy_evaluation = "held_out"
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y
            accuracy_evaluation = "training"

        # Train classifier
        if self.cav_classifier == "sgd":
            classifier = SGDClassifier(loss="hinge", max_iter=1000, random_state=seed, n_jobs=-1)
        else:
            classifier = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")

        classifier.fit(X_train, y_train)

        # Compute accuracy and convert to Python float
        # accuracy_score returns numpy.float64
        accuracy = float(accuracy_score(y_test, classifier.predict(X_test)))
        classifier._explainiverse_accuracy_evaluation = accuracy_evaluation
        classifier._explainiverse_accuracy_effective_test_size = (
            float(len(y_test) / len(y)) if accuracy_evaluation == "held_out" else 0.0
        )
        classifier._explainiverse_training_class_counts = {
            "negative": int(np.sum(y_train == 0)),
            "concept": int(np.sum(y_train == 1)),
        }
        classifier._explainiverse_accuracy_class_counts = {
            "negative": int(np.sum(y_test == 0)),
            "concept": int(np.sum(y_test == 1)),
        }

        # Extract CAV (normal vector to separating hyperplane)
        # For linear classifiers, this is the coefficient vector
        cav_vector = np.asarray(classifier.coef_, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(cav_vector)) or np.max(np.abs(cav_vector)) == 0:
            raise ValueError("Linear classifier did not learn a finite non-zero CAV")

        return cav_vector, classifier, accuracy

    @synchronized_explainer_method
    def learn_concept(
        self,
        concept_name: str,
        concept_examples: np.ndarray,
        negative_examples: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        min_accuracy: float = 0.6,
    ) -> ConceptActivationVector:
        """
        Learn a Concept Activation Vector from examples.

        The CAV is the direction in activation space that separates
        concept examples from negative (non-concept) examples.

        Args:
            concept_name: Human-readable name for the concept.
            concept_examples: Examples that exhibit the concept.
                            Shape: (n_concept, ...) matching model input.
            negative_examples: Counterexamples that do not exhibit the concept.
                This is required; synthetic input noise is not assumed to be
                an in-distribution random concept.
            test_size: Fraction of data to hold out for accuracy estimation.
            min_accuracy: Minimum accuracy for CAV to be considered valid.
                         This threshold is applied to the reported training or
                         held-out linear-classifier accuracy.

        Returns:
            The learned ConceptActivationVector.

        Raises:
            ValueError: If CAV accuracy is below min_accuracy threshold.
        """
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if negative_examples is None:
            raise ValueError(
                "negative_examples is required; TCAV does not treat synthetic "
                "input noise as a verified random concept"
            )
        if not 0.0 <= float(min_accuracy) <= 1.0:
            raise ValueError("min_accuracy must be in [0, 1]")
        if not 0.0 <= float(test_size) < 1.0:
            raise ValueError("test_size must be in [0, 1)")

        concept_examples = as_real_array(
            concept_examples,
            name="concept_examples",
            require_finite=True,
        )
        negative_examples = as_real_array(
            negative_examples,
            name="negative_examples",
            require_finite=True,
        )
        if concept_examples.ndim == 0 or negative_examples.ndim == 0:
            raise ValueError("Concept and negative examples need a batch dimension")

        # Extract activations
        concept_acts = self._get_activations(concept_examples)
        negative_acts = self._get_activations(negative_examples)

        # Train CAV (accuracy is already Python float from _train_cav)
        cav_vector, classifier, accuracy = self._train_cav(concept_acts, negative_acts, test_size)

        if accuracy < min_accuracy:
            raise ValueError(
                f"CAV accuracy ({accuracy:.3f}) is below threshold ({min_accuracy}). "
                f"The concept '{concept_name}' may not be linearly separable in "
                f"layer '{self.layer_name}'. Consider using a different layer "
                f"or providing different or additional examples."
            )

        n_per_group = min(len(concept_acts), len(negative_acts))
        accuracy_evaluation = classifier._explainiverse_accuracy_evaluation
        cav = ConceptActivationVector(
            concept_name=concept_name,
            layer_name=self.layer_name,
            vector=cav_vector,
            classifier=classifier,
            accuracy=accuracy,
            metadata={
                "n_concept_examples": int(len(concept_examples)),
                "n_negative_examples": int(len(negative_examples)),
                "n_balanced_examples_per_group": int(n_per_group),
                "classifier_training_examples_by_label": (
                    classifier._explainiverse_training_class_counts
                ),
                "accuracy_examples_by_label": (classifier._explainiverse_accuracy_class_counts),
                "test_size": float(test_size),
                "accuracy_evaluation": accuracy_evaluation,
                "accuracy_effective_test_size": (
                    classifier._explainiverse_accuracy_effective_test_size
                ),
                "activation_space": "flattened_full_bottleneck",
                "cav_direction": "toward_concept_label",
                "layer_occurrence": self.layer_occurrence,
            },
        )

        # Store the CAV
        self._concepts[concept_name] = cav
        self._concept_activations[concept_name] = concept_acts.copy()

        return cav.clone()

    @synchronized_explainer_method
    def learn_random_concepts(
        self,
        negative_examples: np.ndarray,
        n_random: int = 10,
        concept_name_prefix: str = "_random",
    ) -> List[ConceptActivationVector]:
        """
        Learn random-partition CAVs for exploratory baseline diagnostics.

        Each CAV is trained from a fresh random split of one pool. These
        overlapping splits are not the independent repeated sets used by
        :meth:`statistical_significance_test`.

        Args:
            negative_examples: Pool of examples to sample from.
            n_random: Number of random CAVs to train.
            concept_name_prefix: Prefix for random concept names.

        Returns:
            List of random CAVs.
        """
        if not isinstance(n_random, Integral) or isinstance(n_random, bool):
            raise TypeError("n_random must be an integer")
        if int(n_random) < 1:
            raise ValueError("n_random must be at least 1")
        if not isinstance(concept_name_prefix, str) or not concept_name_prefix:
            raise ValueError("concept_name_prefix must be a non-empty string")

        negative_examples = as_real_array(
            negative_examples,
            name="negative_examples",
            require_finite=True,
        )
        random_cavs = []

        # Get all activations
        all_acts = self._get_activations(negative_examples)
        n_samples = len(all_acts)
        if n_samples < 4:
            raise ValueError("At least four examples are required for random splits")

        for i in range(int(n_random)):
            seed = self._derived_seed(i)
            rng = np.random.RandomState(seed)
            # Randomly split into two groups
            indices = rng.permutation(n_samples)
            split_point = n_samples // 2

            group1_acts = all_acts[indices[:split_point]]
            group2_acts = all_acts[indices[split_point:]]

            cav_vector, classifier, accuracy = self._train_cav(
                group1_acts,
                group2_acts,
                test_size=0.0,
                random_seed=seed,
            )
            cav = ConceptActivationVector(
                concept_name=f"{concept_name_prefix}_{i}",
                layer_name=self.layer_name,
                vector=cav_vector,
                classifier=classifier,
                accuracy=accuracy,
                metadata={
                    "random_seed": int(seed),
                    "baseline_semantics": "overlapping_random_partition_diagnostic",
                    "activation_space": "flattened_full_bottleneck",
                    "accuracy_evaluation": "training",
                    "layer_occurrence": self.layer_occurrence,
                },
            )
            random_cavs.append(cav)

        # Store random CAVs
        self._random_concepts[concept_name_prefix] = tuple(random_cavs)

        return [cav.clone() for cav in random_cavs]

    def _validate_target_class(self, target_class: int) -> int:
        """Normalize one fixed target-score index for a complete TCAV run."""
        if not isinstance(target_class, Integral) or isinstance(target_class, bool):
            raise TypeError("target_class must be an integer output index")
        target_index = int(target_class)
        if target_index < 0:
            raise ValueError("target_class must be non-negative")
        if self.class_names is not None and target_index >= len(self.class_names):
            raise ValueError(f"target_class must be in [0, {len(self.class_names) - 1}]")
        return target_index

    @synchronized_explainer_method
    def compute_directional_derivative(
        self, inputs: np.ndarray, cav: ConceptActivationVector, target_class: int
    ) -> np.ndarray:
        """
        Compute directional derivative of predictions in CAV direction.

        The directional derivative measures how the model's output for
        the target class changes when moving in the CAV direction.

        S_C,k(x) = ∇h_l,k(x) · v_C

        where h_l,k is the adapter-selected fixed score for class k at layer l,
        and v_C is the CAV direction. With PyTorchAdapter, the effective score
        space is exposed as ``last_gradient_output_space``.

        Args:
            inputs: Input data (n_samples, ...)
            cav: The Concept Activation Vector
            target_class: Target class index

        Returns:
            Directional derivatives as a NumPy array with shape ``(n_samples,)``.
        """
        # Get gradients w.r.t. layer activations
        if not isinstance(cav, ConceptActivationVector):
            raise TypeError("cav must be a ConceptActivationVector")
        if cav.layer_name != self.layer_name:
            raise ValueError(f"CAV layer {cav.layer_name!r} does not match {self.layer_name!r}")
        cav_occurrence = cav.metadata.get("layer_occurrence")
        if cav_occurrence != self.layer_occurrence:
            raise ValueError(
                f"CAV layer occurrence {cav_occurrence!r} does not match "
                f"{self.layer_occurrence!r}"
            )
        target_index = self._validate_target_class(target_class)
        gradients = self._get_gradients_wrt_layer(inputs, target_index)
        if gradients.shape[1] != cav.vector.size:
            raise ValueError(
                f"CAV dimension {cav.vector.size} does not match flattened "
                f"bottleneck-gradient dimension {gradients.shape[1]}"
            )

        # Compute dot product with CAV
        # S_C,k(x) = ∇h_l,k(x) · v_C
        directional_derivatives = scale_safe_product_sum(gradients, cav.vector, axis=1)
        if not np.all(np.isfinite(directional_derivatives)):
            raise FloatingPointError("TCAV directional derivatives must be finite")

        return directional_derivatives

    @overload
    def compute_tcav_score(
        self,
        test_inputs: np.ndarray,
        target_class: int,
        concept_name: str,
        return_derivatives: Literal[False] = False,
    ) -> float: ...

    @overload
    def compute_tcav_score(
        self,
        test_inputs: np.ndarray,
        target_class: int,
        concept_name: str,
        return_derivatives: Literal[True],
    ) -> Tuple[float, np.ndarray]: ...

    @overload
    def compute_tcav_score(
        self,
        test_inputs: np.ndarray,
        target_class: int,
        concept_name: str,
        return_derivatives: bool,
    ) -> Union[float, Tuple[float, np.ndarray]]: ...

    @synchronized_explainer_method
    def compute_tcav_score(
        self,
        test_inputs: np.ndarray,
        target_class: int,
        concept_name: str,
        return_derivatives: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute TCAV score for a concept and target class.

        The TCAV score is the fraction of supplied inputs with a positive
        directional derivative of the selected target score along the CAV.

        TCAV_C,k = |{x : S_C,k(x) > 0}| / |X|

        A score above 0.5 means positive derivatives occurred on more than
        half of the supplied inputs; it is not a causal-effect threshold.

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
        if not isinstance(return_derivatives, (bool, np.bool_)):
            raise TypeError("return_derivatives must be a boolean")
        if concept_name not in self._concepts:
            raise ValueError(
                f"Concept '{concept_name}' not found. "
                f"Available concepts: {list(self._concepts.keys())}. "
                f"Use learn_concept() first."
            )

        test_inputs = as_real_array(
            test_inputs,
            name="test_inputs",
            require_finite=True,
        )
        cav = self._concepts[concept_name]

        # Compute directional derivatives
        derivatives = self.compute_directional_derivative(test_inputs, cav, target_class)

        # TCAV score = fraction with positive derivative
        # np.mean returns numpy.float64, convert to Python float
        tcav_score = float(np.mean(derivatives > 0))

        if return_derivatives:
            return tcav_score, derivatives
        return tcav_score

    def _prepare_random_example_sets(
        self,
        negative_examples: Optional[np.ndarray],
        random_example_sets: Optional[Sequence[np.ndarray]],
        n_random: int,
    ) -> Tuple[List[np.ndarray], str, int]:
        """Return explicit or reproducibly partitioned random concept sets."""
        if random_example_sets is not None and negative_examples is not None:
            raise ValueError("Provide either negative_examples or random_example_sets, not both")

        if random_example_sets is not None:
            if isinstance(random_example_sets, np.ndarray):
                raise TypeError(
                    "random_example_sets must be a sequence of separate batches; "
                    "use negative_examples for one pool"
                )
            sets = [
                as_real_array(
                    values,
                    name="random example set",
                    require_finite=True,
                )
                for values in random_example_sets
            ]
            if len(sets) < n_random:
                raise ValueError(f"At least {n_random} random_example_sets are required")
            sets = sets[:n_random]
            if any(values.ndim == 0 or len(values) < 2 for values in sets):
                raise ValueError("Each random example set must contain at least two examples")
            return sets, "explicit_sets", int(min(len(values) for values in sets))

        if negative_examples is None:
            raise ValueError(
                "negative_examples or random_example_sets is required; test_inputs "
                "are never reused as the random-concept baseline"
            )
        pool = as_real_array(
            negative_examples,
            name="negative_examples",
            require_finite=True,
        )
        if pool.ndim == 0:
            raise ValueError("negative_examples needs a batch dimension")
        group_size = len(pool) // n_random
        if group_size < 2:
            raise ValueError(
                "negative_examples must contain at least two disjoint examples "
                "per requested random run"
            )
        rng = np.random.RandomState(self.random_seed)
        indices = rng.permutation(len(pool))[: group_size * n_random]
        sets = [pool[indices[i * group_size : (i + 1) * group_size]] for i in range(n_random)]
        return sets, "disjoint_partition", int(group_size)

    @staticmethod
    def _welch_test(
        concept_scores: np.ndarray,
        random_scores: np.ndarray,
    ) -> Tuple[float, float]:
        """Run Welch's test, including defined limits for constant samples."""
        concept_variance = float(np.var(concept_scores, ddof=1))
        random_variance = float(np.var(random_scores, ddof=1))
        difference = float(np.mean(concept_scores) - np.mean(random_scores))
        concept_term = concept_variance / len(concept_scores)
        random_term = random_variance / len(random_scores)
        standard_error_squared = concept_term + random_term
        if standard_error_squared == 0.0:
            if difference == 0.0:
                return 0.0, 1.0
            return (float("inf") if difference > 0 else float("-inf")), 0.0

        t_statistic = difference / np.sqrt(standard_error_squared)
        degrees_of_freedom = standard_error_squared**2 / (
            concept_term**2 / (len(concept_scores) - 1) + random_term**2 / (len(random_scores) - 1)
        )
        p_value = 2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom)
        return float(t_statistic), float(p_value)

    @staticmethod
    def _standardized_mean_difference(
        concept_scores: np.ndarray,
        random_scores: np.ndarray,
    ) -> float:
        """Return pooled-SD standardized mean difference for two samples."""
        n_concept = len(concept_scores)
        n_random = len(random_scores)
        pooled_variance = (
            (n_concept - 1) * np.var(concept_scores, ddof=1)
            + (n_random - 1) * np.var(random_scores, ddof=1)
        ) / (n_concept + n_random - 2)
        difference = float(np.mean(concept_scores) - np.mean(random_scores))
        if pooled_variance == 0.0:
            if difference == 0.0:
                return 0.0
            return float("inf") if difference > 0 else float("-inf")
        return float(difference / np.sqrt(pooled_variance))

    @synchronized_explainer_method
    def statistical_significance_test(
        self,
        test_inputs: np.ndarray,
        target_class: int,
        concept_name: str,
        n_random: int = 10,
        negative_examples: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        random_example_sets: Optional[Sequence[np.ndarray]] = None,
        cav_test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Compare repeated concept TCAV scores with random TCAV scores.

        One concept CAV is trained against each random counterexample set.
        Random-baseline CAVs are trained for every pair of those random sets.
        Their TCAV-score distributions are compared by a two-sided Welch
        t-test. ``significant`` is a compatibility alias for an *uncorrected,
        per-concept* threshold decision. It is a TCAV-style diagnostic, not a
        guarantee of confirmatory significance: runs share the positive
        concept set and random-vs-random pairs reuse random sets.

        ``random_example_sets`` is the clearest contract when independently
        assembled random concepts are available. A single ``negative_examples``
        pool is reproducibly split into disjoint groups and the required IID
        assumption is reported in the result.
        """
        _check_scipy()
        if concept_name not in self._concepts or concept_name not in self._concept_activations:
            raise ValueError(f"Concept {concept_name!r} was not learned with learn_concept()")
        if not isinstance(n_random, Integral) or isinstance(n_random, bool):
            raise TypeError("n_random must be an integer")
        n_random = int(n_random)
        if n_random < 3:
            raise ValueError("n_random must be at least 3 to estimate both score distributions")
        if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be a finite value strictly between 0 and 1")
        if not np.isfinite(cav_test_size) or not 0.0 <= float(cav_test_size) < 1.0:
            raise ValueError("cav_test_size must be a finite value in [0, 1)")

        target_index = self._validate_target_class(target_class)
        test_inputs = as_real_array(
            test_inputs,
            name="test_inputs",
            require_finite=True,
        )
        if test_inputs.ndim == 0 or len(test_inputs) == 0:
            raise ValueError("test_inputs must contain at least one example")
        example_sets, set_source, set_size = self._prepare_random_example_sets(
            negative_examples,
            random_example_sets,
            n_random,
        )
        random_activation_sets = [self._get_activations(values) for values in example_sets]
        concept_activations = self._concept_activations[concept_name]

        concept_scores: List[float] = []
        concept_cav_accuracies: List[float] = []
        concept_cav_accuracy_evaluations: List[str] = []
        for run_index, random_activations in enumerate(random_activation_sets):
            vector, classifier, accuracy = self._train_cav(
                concept_activations,
                random_activations,
                test_size=float(cav_test_size),
                random_seed=self._derived_seed(run_index),
            )
            cav = ConceptActivationVector(
                concept_name=concept_name,
                layer_name=self.layer_name,
                vector=vector,
                classifier=classifier,
                accuracy=accuracy,
                metadata={
                    "run_index": int(run_index),
                    "run_type": "concept_vs_random",
                    "accuracy_evaluation": classifier._explainiverse_accuracy_evaluation,
                    "accuracy_requested_test_size": float(cav_test_size),
                    "accuracy_effective_test_size": (
                        classifier._explainiverse_accuracy_effective_test_size
                    ),
                    "layer_occurrence": self.layer_occurrence,
                },
            )
            concept_cav_accuracies.append(float(accuracy))
            concept_cav_accuracy_evaluations.append(classifier._explainiverse_accuracy_evaluation)
            concept_scores.append(
                float(
                    np.mean(self.compute_directional_derivative(test_inputs, cav, target_index) > 0)
                )
            )

        random_scores: List[float] = []
        random_cav_accuracies: List[float] = []
        random_cav_accuracy_evaluations: List[str] = []
        random_cavs: List[ConceptActivationVector] = []
        for pair_index, (left_index, right_index) in enumerate(combinations(range(n_random), 2)):
            vector, classifier, accuracy = self._train_cav(
                random_activation_sets[left_index],
                random_activation_sets[right_index],
                test_size=float(cav_test_size),
                random_seed=self._derived_seed(n_random + pair_index),
            )
            cav = ConceptActivationVector(
                concept_name=f"_random_{left_index}_vs_{right_index}",
                layer_name=self.layer_name,
                vector=vector,
                classifier=classifier,
                accuracy=accuracy,
                metadata={
                    "run_type": "random_vs_random",
                    "random_set_indices": [int(left_index), int(right_index)],
                    "accuracy_evaluation": classifier._explainiverse_accuracy_evaluation,
                    "accuracy_requested_test_size": float(cav_test_size),
                    "accuracy_effective_test_size": (
                        classifier._explainiverse_accuracy_effective_test_size
                    ),
                    "layer_occurrence": self.layer_occurrence,
                },
            )
            random_cavs.append(cav)
            random_cav_accuracies.append(float(accuracy))
            random_cav_accuracy_evaluations.append(classifier._explainiverse_accuracy_evaluation)
            random_scores.append(
                float(
                    np.mean(self.compute_directional_derivative(test_inputs, cav, target_index) > 0)
                )
            )

        random_prefix = f"_random_{concept_name}_{target_index}"
        self._random_concepts[random_prefix] = tuple(random_cavs)
        concept_array = np.asarray(concept_scores, dtype=np.float64)
        random_array = np.asarray(random_scores, dtype=np.float64)
        t_statistic, p_value = self._welch_test(concept_array, random_array)
        concept_mean = float(np.mean(concept_array))
        random_mean = float(np.mean(random_array))
        concept_std = float(np.std(concept_array, ddof=1))
        random_std = float(np.std(random_array, ddof=1))
        effect_size = self._standardized_mean_difference(concept_array, random_array)
        significant = bool(p_value < float(alpha))

        return {
            "tcav_score": concept_mean,
            "concept_scores": [float(score) for score in concept_scores],
            "concept_cav_accuracies": concept_cav_accuracies,
            "concept_cav_accuracy_evaluations": concept_cav_accuracy_evaluations,
            "concept_mean": concept_mean,
            "concept_std": concept_std,
            "random_scores": [float(score) for score in random_scores],
            "random_cav_accuracies": random_cav_accuracies,
            "random_cav_accuracy_evaluations": random_cav_accuracy_evaluations,
            "random_mean": random_mean,
            "random_std": random_std,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "significant": significant,
            "significant_at_alpha_uncorrected": significant,
            "effect_size": effect_size,
            "alpha": float(alpha),
            "test_method": "two-sided Welch t-test",
            "multiple_comparisons_corrected": False,
            "supports_confirmatory_significance_claim": False,
            "random_set_source": set_source,
            "random_set_size": set_size,
            "n_concept_runs": int(len(concept_scores)),
            "n_random_baseline_runs": int(len(random_scores)),
            "cav_accuracy_requested_test_size": float(cav_test_size),
            "target_score_space": getattr(
                self.model, "last_gradient_output_space", "adapter_defined"
            ),
            "tcav_variant": self.last_tcav_variant,
            "canonical_class_logit_tcav": bool(self.last_tcav_is_canonical),
            "inference_assumptions": [
                "random example sets represent the null concept distribution",
                "Welch test independence is an approximation because runs reuse sets",
                "reported p-value is per-concept and uncorrected",
                "test inputs are caller-supplied target-class examples",
            ],
        }

    @synchronized_explainer_method
    def explain(
        self,
        test_inputs: np.ndarray,
        target_class: Optional[int] = None,
        concept_names: Optional[List[str]] = None,
        run_significance_test: bool = False,
        negative_examples: Optional[np.ndarray] = None,
        n_random: int = 10,
        random_example_sets: Optional[Sequence[np.ndarray]] = None,
        alpha: float = 0.05,
        significance_cav_test_size: float = 0.2,
    ) -> Explanation:
        """
        Generate one aggregate TCAV explanation for a target-class input set.

        Computes TCAV scores for all (or specified) concepts
        and optionally runs statistical significance tests. The returned score
        is global to the supplied set; it is not a per-instance local score.

        Args:
            test_inputs: Input examples to explain.
            target_class: Target class to explain. If None, uses
                         the most common predicted class.
            concept_names: List of concepts to include. If None,
                          uses all learned concepts.
            run_significance_test: Whether to run statistical tests.
            negative_examples: One random-concept pool that will be split into
                disjoint counterexample sets for the repeated-run diagnostic.
            n_random: Number of concept-vs-random runs. All pairwise
                random-vs-random baselines are also trained.
            random_example_sets: Explicit random-concept example batches. Use
                this instead of ``negative_examples`` when independently
                assembled sets are available.
            alpha: Per-concept, uncorrected threshold for the compatibility
                ``significant`` field.
            significance_cav_test_size: Requested held-out fraction used to
                estimate every repeated-run CAV classifier's accuracy. Very
                small datasets that cannot form a stratified holdout are
                explicitly reported as training-accuracy evaluations.

        Returns:
            Explanation object with TCAV scores for each concept.
        """
        if not isinstance(run_significance_test, (bool, np.bool_)):
            raise TypeError("run_significance_test must be a boolean")
        if concept_names is not None:
            validated_concepts = validate_name_sequence(concept_names, name="concept_names")
            assert validated_concepts is not None
            concept_names = validated_concepts

        test_inputs = as_real_array(
            test_inputs,
            name="test_inputs",
            require_finite=True,
        )
        if test_inputs.ndim == 0 or len(test_inputs) == 0:
            raise ValueError("test_inputs must contain at least one example")

        if len(self._concepts) == 0:
            raise ValueError("No concepts learned. Use learn_concept() first.")

        # Validate the score width even for an explicit target so class labels
        # cannot silently describe a different output than the differentiated
        # score.
        with preserve_adapter_model_eval(self.model, preserve_gradients=False):
            predictions = as_real_array(
                self.model.predict(test_inputs),
                name="classification predictions",
                dtype=np.float64,
                require_finite=True,
            )
        if predictions.ndim != 2 or predictions.shape[0] != len(test_inputs):
            raise ValueError("TCAV requires a two-dimensional classification score matrix")
        if predictions.shape[1] == 1:
            if np.any((predictions < 0) | (predictions > 1)):
                raise ValueError("A one-column prediction must contain P(class 1) in [0, 1]")
            predictions = np.column_stack((1.0 - predictions[:, 0], predictions[:, 0]))
        if predictions.shape[1] < 2 or not np.all(np.isfinite(predictions)):
            raise ValueError("Classification predictions must be finite class scores")
        if self.class_names is not None and len(self.class_names) != predictions.shape[1]:
            raise ValueError("class_names length does not match the model prediction width")

        if target_class is None:
            predicted_classes = np.argmax(predictions, axis=1)
            target_class = int(np.argmax(np.bincount(predicted_classes)))
        target_class = self._validate_target_class(target_class)
        if target_class >= predictions.shape[1]:
            raise ValueError(f"target_class must be in [0, {predictions.shape[1] - 1}]")

        # Determine concepts to analyze
        if concept_names is None:
            concept_names = list(self._concepts.keys())
        else:
            missing = [name for name in concept_names if name not in self._concepts]
            if missing:
                raise ValueError(f"Unknown concepts requested: {missing}")

        # Compute TCAV scores for each concept
        tcav_scores = {}
        significance_results = {}

        for concept_name in concept_names:
            # compute_tcav_score returns Python float
            score, derivatives = self.compute_tcav_score(
                test_inputs, target_class, concept_name, return_derivatives=True
            )

            tcav_scores[concept_name] = {
                "score": score,  # Already Python float
                "cav_accuracy": self._concepts[concept_name].accuracy,  # Already Python float
                "positive_count": int(np.sum(derivatives > 0)),
                "total_count": int(len(derivatives)),
                "target_score_space": self.last_target_score_space,
                "tcav_variant": self.last_tcav_variant,
                "canonical_class_logit_tcav": bool(self.last_tcav_is_canonical),
            }

            # Optionally run significance test
            if run_significance_test:
                sig_result = self.statistical_significance_test(
                    test_inputs,
                    target_class,
                    concept_name,
                    n_random=n_random,
                    negative_examples=negative_examples,
                    random_example_sets=random_example_sets,
                    alpha=alpha,
                    cav_test_size=significance_cav_test_size,
                )
                significance_results[concept_name] = sig_result

        # Determine class name
        if self.class_names is not None and target_class is not None:
            label_name = self.class_names[target_class]
        else:
            label_name = f"class_{target_class}"

        explanation_data = {
            "tcav_scores": tcav_scores,
            "target_class": int(target_class),
            "n_test_inputs": int(len(test_inputs)),
            "layer_name": self.layer_name,
            "layer_occurrence": self.layer_occurrence,
            "concepts_analyzed": list(concept_names),
            "activation_space": "flattened_full_bottleneck",
            "tcav_score_definition": (
                "fraction_of_positive_class_logit_directional_derivatives"
                if self.last_tcav_is_canonical
                else "fraction_of_positive_adapter_score_directional_derivatives"
            ),
            "target_score_fixed_for_batch": True,
            "test_input_class_membership": "caller_supplied_not_validated",
            "aggregate_scope": "global_target_class_input_set",
            "returns_per_instance_scores": False,
            "target_score_space": self.last_target_score_space,
            "tcav_variant": self.last_tcav_variant,
            "canonical_class_logit_tcav": bool(self.last_tcav_is_canonical),
        }

        if run_significance_test:
            explanation_data["significance_tests"] = significance_results

        return Explanation(
            explainer_name="TCAV",
            target_class=label_name,
            explanation_data=explanation_data,
            metadata={
                "explanation_scope": "global",
                "aggregate_unit": "target_class_input_set",
                "tcav_variant": self.last_tcav_variant,
                "canonical_class_logit_tcav": bool(self.last_tcav_is_canonical),
                "target_score_space": self.last_target_score_space,
                "layer_occurrence": self.layer_occurrence,
            },
        )

    @synchronized_explainer_method
    def explain_batch(
        self, X: np.ndarray, target_class: Optional[int] = None, **kwargs
    ) -> List[Explanation]:
        """
        Compute one global aggregate TCAV explanation for an input set.

        The returned score is the fraction of positive target-score
        directional derivatives across the supplied batch.

        Args:
            X: Batch of inputs.
            target_class: Target class to explain.
            **kwargs: Additional arguments passed to explain().

        Returns:
            List containing a single Explanation for the batch.
        """
        return [self.explain(X, target_class=target_class, **kwargs)]

    @synchronized_explainer_method
    def get_most_influential_concepts(
        self, test_inputs: np.ndarray, target_class: int, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Return concepts with the highest TCAV scores for the target class.

        This is a ranking of the configured directional-derivative statistic,
        not a causal influence ranking.

        Args:
            test_inputs: Test examples.
            target_class: Target class index.
            top_k: Number of top concepts to return.

        Returns:
            List of (concept_name, tcav_score) tuples, sorted by score descending.
            All scores are Python floats.
        """
        if not isinstance(top_k, Integral) or isinstance(top_k, bool) or int(top_k) < 1:
            raise ValueError("top_k must be a positive integer")
        target_class = self._validate_target_class(target_class)
        scores = []

        for concept_name in self._concepts:
            # compute_tcav_score returns Python float
            score = self.compute_tcav_score(test_inputs, target_class, concept_name)
            scores.append((concept_name, score))

        # Sort by the reported TCAV score.
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[: int(top_k)]

    @synchronized_explainer_method
    def compare_concepts(
        self,
        test_inputs: np.ndarray,
        target_classes: List[int],
        concept_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[int, float]]:
        """
        Compare TCAV scores across multiple target classes.

        The result compares directional-derivative score fractions; it does
        not by itself establish concept importance.

        Args:
            test_inputs: Test examples.
            target_classes: List of class indices to compare.
            concept_names: Concepts to analyze (default: all).

        Returns:
            Dictionary mapping concept names to {class_idx: tcav_score}.
            All scores are Python floats.
        """
        if concept_names is None:
            concept_names = list(self._concepts.keys())
        else:
            missing = [name for name in concept_names if name not in self._concepts]
            if missing:
                raise ValueError(f"Unknown concepts requested: {missing}")
        normalized_targets = [self._validate_target_class(value) for value in target_classes]

        results: Dict[str, Dict[int, float]] = {}

        for concept_name in concept_names:
            results[concept_name] = {}
            for class_idx in normalized_targets:
                # compute_tcav_score returns Python float
                score = self.compute_tcav_score(test_inputs, class_idx, concept_name)
                results[concept_name][class_idx] = score

        return results

    @synchronized_explainer_method
    def list_concepts(self) -> List[str]:
        """List all learned concept names."""
        return list(self._concepts.keys())

    @synchronized_explainer_method
    def get_concept(self, concept_name: str) -> ConceptActivationVector:
        """Get a specific CAV by name."""
        if concept_name not in self._concepts:
            raise ValueError(f"Concept '{concept_name}' not found.")
        return self._concepts[concept_name].clone()

    @synchronized_explainer_method
    def remove_concept(self, concept_name: str) -> None:
        """Remove a learned concept."""
        if concept_name in self._concepts:
            del self._concepts[concept_name]
            self._concept_activations.pop(concept_name, None)
