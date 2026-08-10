# src/explainiverse/core/registry.py
"""
ExplainerRegistry - metadata and construction registry for XAI methods.

This module provides:
- Registration of explainers with validated metadata
- Filtering/discovery by scope, model type, data type, task type
- Instantiation with caller-supplied constructor arguments
- Decorator-based registration
- Metadata-only compatibility ordering (historical ``recommend`` API)

Example usage:
    from explainiverse.core.registry import default_registry, ExplainerMeta

    # List available explainers
    print(default_registry.list_explainers())

    # Filter by criteria
    local_tabular = default_registry.filter(scope="local", data_type="tabular")

    # Create an explainer
    explainer = default_registry.create("lime", model=adapter, training_data=X, ...)

    # Register a custom explainer
    @default_registry.register_decorator(
        name="my_explainer",
        meta=ExplainerMeta(scope="local", description="My custom explainer")
    )
    class MyExplainer(BaseExplainer):
        ...
"""

import copy
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Dict, List, Optional, Type

from explainiverse.core.explainer import BaseExplainer


@dataclass
class ExplainerMeta:
    """
    Metadata for explainer discovery and compatibility filtering.

    Attributes:
        scope: "local" (instance-level) or "global" (model-level)
        model_types: List of compatible model types ("any", "tree", "linear", "neural", "ensemble")
        data_types: List of compatible data types ("tabular", "image", "text", "time_series")
        task_types: List of compatible tasks ("classification", "regression")
        description: Human-readable description of the explainer
        paper_reference: Citation for the original paper
        complexity: Optional informal implementation-scaling note. It is not a
            proven complexity bound unless ``complexity_verified`` is true.
        requires_training_data: Whether the explainer needs training data
        supports_batching: Whether the class exposes a callable ``explain_batch``
        claim_status: Audit status: "verified", "quarantined", or "unverified"
        claim_scope: Human-readable boundary of the verified/allowed claim
    """

    scope: str  # "local" or "global"
    model_types: List[str] = field(default_factory=lambda: ["any"])
    data_types: List[str] = field(default_factory=lambda: ["tabular"])
    task_types: List[str] = field(default_factory=lambda: ["classification", "regression"])
    description: str = ""
    paper_reference: Optional[str] = None
    complexity: Optional[str] = None
    complexity_verified: bool = False
    requires_training_data: bool = False
    supports_batching: bool = False
    claim_status: str = "unverified"
    claim_scope: str = "Implementation has not completed an accuracy audit."

    _VALID_SCOPES = {"local", "global"}
    _VALID_MODEL_TYPES = {"any", "tree", "linear", "neural", "ensemble"}
    _VALID_DATA_TYPES = {"tabular", "image", "text", "time_series"}
    _VALID_TASK_TYPES = {"classification", "regression"}
    _VALID_CLAIM_STATUSES = {"verified", "quarantined", "unverified"}

    def __post_init__(self) -> None:
        if self.scope not in self._VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(self._VALID_SCOPES)}.")
        self._validate_categories("model_types", self.model_types, self._VALID_MODEL_TYPES)
        self._validate_categories("data_types", self.data_types, self._VALID_DATA_TYPES)
        self._validate_categories("task_types", self.task_types, self._VALID_TASK_TYPES)
        if not isinstance(self.description, str):
            raise TypeError("description must be a string.")
        for name in ("paper_reference", "complexity"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TypeError(f"{name} must be None or a non-empty string.")
        for name in ("complexity_verified", "requires_training_data", "supports_batching"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        if self.complexity_verified and self.complexity is None:
            raise ValueError("complexity_verified=True requires a complexity note.")
        if self.claim_status not in self._VALID_CLAIM_STATUSES:
            raise ValueError(f"claim_status must be one of {sorted(self._VALID_CLAIM_STATUSES)}.")
        if not isinstance(self.claim_scope, str) or not self.claim_scope.strip():
            raise TypeError("claim_scope must be a non-empty string.")
        if (
            self.claim_status != "unverified"
            and self.claim_scope.strip() == "Implementation has not completed an accuracy audit."
        ):
            raise ValueError(
                f"claim_status={self.claim_status!r} requires an explicit audited claim_scope."
            )

    @staticmethod
    def _validate_categories(name: str, values: Any, allowed: set[str]) -> None:
        if not isinstance(values, list) or not values:
            raise TypeError(f"{name} must be a non-empty list.")
        if any(not isinstance(value, str) for value in values):
            raise TypeError(f"{name} must contain only strings.")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must not contain duplicates.")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown {name}: {sorted(unknown)}; allowed: {sorted(allowed)}.")

    def matches(
        self,
        scope: Optional[str] = None,
        model_type: Optional[str] = None,
        data_type: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> bool:
        """Check if this metadata matches the given criteria."""
        self._validate_match_criteria(scope, model_type, data_type, task_type)

        if scope is not None and self.scope != scope:
            return False

        if model_type is not None:
            if "any" not in self.model_types and model_type not in self.model_types:
                return False

        if data_type is not None:
            if data_type not in self.data_types:
                return False

        if task_type is not None:
            if task_type not in self.task_types:
                return False

        return True

    @classmethod
    def _validate_match_criteria(
        cls,
        scope: Optional[str] = None,
        model_type: Optional[str] = None,
        data_type: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> None:
        """Validate public compatibility-query values independent of registry state."""
        criteria = (
            ("scope", scope, cls._VALID_SCOPES),
            ("model_type", model_type, cls._VALID_MODEL_TYPES),
            ("data_type", data_type, cls._VALID_DATA_TYPES),
            ("task_type", task_type, cls._VALID_TASK_TYPES),
        )
        for name, value, allowed in criteria:
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError(f"{name} must be a string or None.")
                if value not in allowed:
                    raise ValueError(f"{name} must be one of {sorted(allowed)}.")


class ExplainerRegistry:
    """
    Registry for Explainiverse explainer entries.

    Provides:
    - Registration (programmatic and decorator-based)
    - Discovery and filtering
    - Instantiation with dependency injection
    - Metadata-only compatibility ordering
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        explainer_class: Type[BaseExplainer],
        meta: ExplainerMeta,
        override: bool = False,
    ) -> None:
        """
        Register an explainer class with metadata.

        Args:
            name: Unique identifier for the explainer (e.g., "lime", "shap")
            explainer_class: The explainer class (must inherit from BaseExplainer)
            meta: Metadata describing the explainer's capabilities
            override: If True, allows overwriting existing registration

        Raises:
            ValueError: If name is already registered and override=False
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Explainer name must be a non-empty string.")
        if not isinstance(override, bool):
            raise TypeError("override must be a boolean.")
        if not isinstance(explainer_class, type) or not issubclass(explainer_class, BaseExplainer):
            raise TypeError("explainer_class must inherit from BaseExplainer.")
        if not isinstance(meta, ExplainerMeta):
            raise TypeError("meta must be an ExplainerMeta instance.")

        # ExplainerMeta remains mutable for compatibility. Revalidate the
        # isolated snapshot so mutations made after construction cannot bypass
        # the registry's metadata invariants.
        validated_meta = copy.deepcopy(meta)
        validated_meta.__post_init__()
        if validated_meta.supports_batching and not callable(
            getattr(explainer_class, "explain_batch", None)
        ):
            raise ValueError(
                "supports_batching=True requires a callable explain_batch method "
                f"on {explainer_class.__name__}."
            )
        if name in self._registry and not override:
            raise ValueError(
                f"Explainer '{name}' is already registered. Use override=True to replace."
            )

        # Metadata is part of the registry's public trust boundary.  Keep an
        # isolated validated snapshot so callers cannot mutate a registered
        # claim through the object they originally supplied.
        self._registry[name] = {"class": explainer_class, "meta": validated_meta}

    def unregister(self, name: str) -> None:
        """
        Remove an explainer from the registry.

        Args:
            name: The explainer name to remove

        Raises:
            KeyError: If the explainer is not registered
        """
        if name not in self._registry:
            raise KeyError(f"Explainer '{name}' is not registered.")
        del self._registry[name]

    def get(self, name: str) -> Dict[str, Any]:
        """
        Get the explainer class and metadata by name.

        Args:
            name: The explainer name

        Returns:
            Dict with "class" and a defensive copy under "meta"

        Raises:
            KeyError: If the explainer is not registered
        """
        if name not in self._registry:
            raise KeyError(
                f"Explainer '{name}' is not registered. Available: {list(self._registry.keys())}"
            )
        entry = self._registry[name]
        return {
            "class": entry["class"],
            "meta": copy.deepcopy(entry["meta"]),
        }

    def get_meta(self, name: str) -> ExplainerMeta:
        """
        Get just the metadata for an explainer.

        Args:
            name: The explainer name

        Returns:
            Defensive copy of the registered ExplainerMeta instance

        Raises:
            KeyError: If the explainer is not registered
        """
        return self.get(name)["meta"]

    def list_explainers(self, with_meta: bool = False) -> Any:
        """
        List all registered explainers.

        Args:
            with_meta: If True, return dict with metadata; if False, return list of names

        Returns:
            List of names or dict of {name: {"class": ..., "meta": ...}},
            with defensive metadata copies
        """
        if with_meta:
            return {
                name: {
                    "class": entry["class"],
                    "meta": copy.deepcopy(entry["meta"]),
                }
                for name, entry in self._registry.items()
            }
        return list(self._registry.keys())

    def filter(
        self,
        scope: Optional[str] = None,
        model_type: Optional[str] = None,
        data_type: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[str]:
        """
        Filter explainers by criteria.

        Args:
            scope: "local" or "global"
            model_type: "any", "tree", "linear", "neural", "ensemble"
            data_type: "tabular", "image", "text", "time_series"
            task_type: "classification" or "regression"

        Returns:
            List of matching explainer names
        """
        ExplainerMeta._validate_match_criteria(scope, model_type, data_type, task_type)
        results = []
        for name, entry in self._registry.items():
            meta: ExplainerMeta = entry["meta"]
            if meta.matches(scope, model_type, data_type, task_type):
                results.append(name)
        return results

    def create(self, name: str, **kwargs) -> BaseExplainer:
        """
        Instantiate an explainer by name with the given arguments.

        Args:
            name: The explainer name
            **kwargs: Arguments to pass to the explainer constructor

        Returns:
            Instantiated explainer

        Raises:
            KeyError: If the explainer is not registered
        """
        entry = self.get(name)
        explainer_class = entry["class"]
        return explainer_class(**kwargs)

    def register_decorator(
        self, name: str, meta: ExplainerMeta
    ) -> Callable[[Type[BaseExplainer]], Type[BaseExplainer]]:
        """
        Decorator for registering an explainer class.

        Usage:
            @registry.register_decorator(
                name="my_explainer",
                meta=ExplainerMeta(scope="local")
            )
            class MyExplainer(BaseExplainer):
                ...

        Args:
            name: Unique identifier for the explainer
            meta: Metadata describing the explainer

        Returns:
            Decorator function that registers the class and returns it unchanged
        """

        def decorator(cls: Type[BaseExplainer]) -> Type[BaseExplainer]:
            self.register(name, cls, meta)
            return cls

        return decorator

    def summary(self) -> str:
        """
        Generate a human-readable summary of all registered explainers.

        Returns:
            Formatted string summary
        """
        lines = ["=" * 60, "Explainiverse - Registered Explainers", "=" * 60, ""]

        # Group by scope
        local = []
        global_ = []

        for name, entry in self._registry.items():
            meta: ExplainerMeta = entry["meta"]
            info = (
                f"  {name} [{meta.claim_status}]: "
                f"{meta.description or '(no description)'}\n"
                f"    claim scope: {meta.claim_scope}"
            )
            if meta.scope == "local":
                local.append(info)
            else:
                global_.append(info)

        if local:
            lines.append("LOCAL EXPLAINERS (instance-level):")
            lines.extend(local)
            lines.append("")

        if global_:
            lines.append("GLOBAL EXPLAINERS (model-level):")
            lines.extend(global_)
            lines.append("")

        lines.append(f"Total: {len(self._registry)} explainers")
        lines.append("=" * 60)

        return "\n".join(lines)

    def recommend(
        self,
        model_type: Optional[str] = None,
        data_type: Optional[str] = None,
        task_type: Optional[str] = None,
        scope_preference: Optional[str] = None,
        max_results: int = 5,
    ) -> List[str]:
        """
        Return a deterministic metadata-only ordering of compatible entries.

        The historical method name is retained for compatibility. This function
        does **not** rank explanation quality, empirical performance, runtime, or
        suitability for a particular dataset. It also does not filter by
        ``claim_status``; callers must inspect :meth:`get_meta` before selecting an
        unverified or quarantined entry.

        Args:
            model_type: The type of model being explained
            data_type: The type of data
            task_type: The ML task type
            scope_preference: Preferred scope ("local" or "global")
            max_results: Maximum number of recommendations

        Returns:
            Compatible names ordered by the requested scope/model metadata and
            then registration order.
        """
        if isinstance(max_results, bool) or not isinstance(max_results, Integral):
            raise TypeError("max_results must be a non-negative integer.")
        if max_results < 0:
            raise ValueError("max_results must be non-negative.")
        if scope_preference is not None and scope_preference not in ExplainerMeta._VALID_SCOPES:
            raise ValueError(
                f"scope_preference must be one of {sorted(ExplainerMeta._VALID_SCOPES)}."
            )

        candidates = self.filter(model_type=model_type, data_type=data_type, task_type=task_type)

        # Order only by explicit compatibility metadata. Documentation or a paper
        # citation is not evidence that one implementation is better than another.
        scored = []
        for name in candidates:
            meta = self.get_meta(name)
            score = 0

            # Prefer matching scope
            if scope_preference and meta.scope == scope_preference:
                score += 10

            # Prefer specific model types over "any"
            if model_type and model_type in meta.model_types:
                score += 5

            scored.append((name, score))

        # Python's sort is stable, so equal metadata scores retain registration order.
        scored.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in scored[:max_results]]


# =============================================================================
# Default Global Registry
# =============================================================================


def _create_default_registry() -> ExplainerRegistry:
    """Create and populate the default global registry."""
    from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
    from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
    from explainiverse.explainers.attribution.treeshap_wrapper import TreeShapExplainer
    from explainiverse.explainers.counterfactual.dice_wrapper import CounterfactualExplainer
    from explainiverse.explainers.example_based.protodash import ProtoDashExplainer
    from explainiverse.explainers.global_explainers.ale import ALEExplainer
    from explainiverse.explainers.global_explainers.partial_dependence import (
        PartialDependenceExplainer,
    )
    from explainiverse.explainers.global_explainers.permutation_importance import (
        PermutationImportanceExplainer,
    )
    from explainiverse.explainers.global_explainers.sage import SAGEExplainer
    from explainiverse.explainers.gradient.cam_variants import (
        AblationCAMExplainer,
        EigenCAMExplainer,
        EigenGradCAMExplainer,
        GradCAMElementWiseExplainer,
        HiResCAMExplainer,
        LayerCAMExplainer,
        ScoreCAMExplainer,
        XGradCAMExplainer,
    )
    from explainiverse.explainers.gradient.deeplift import DeepLIFTExplainer, DeepLIFTShapExplainer
    from explainiverse.explainers.gradient.gradcam import GradCAMExplainer
    from explainiverse.explainers.gradient.integrated_gradients import IntegratedGradientsExplainer
    from explainiverse.explainers.gradient.lrp import LRPExplainer
    from explainiverse.explainers.gradient.saliency import SaliencyExplainer
    from explainiverse.explainers.gradient.smoothgrad import SmoothGradExplainer
    from explainiverse.explainers.gradient.tcav import TCAVExplainer
    from explainiverse.explainers.rule_based.anchor_tabular import AnchorTabularExplainer
    from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer

    registry = ExplainerRegistry()

    # =========================================================================
    # Local Explainers (instance-level)
    # =========================================================================

    # Register LIME
    registry.register(
        name="lime",
        explainer_class=LimeExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="Tabular LIME using the official lime.lime_tabular backend",
            paper_reference="Ribeiro et al., 2016 - 'Why Should I Trust You?'",
            complexity="O(n_samples * n_features)",
            requires_training_data=True,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Official lime.lime_tabular backend for tabular class-probability models "
                "and single-output regression; the API returns one output explanation, "
                "requires top_labels=1, and accepts an explicit classification output index; "
                "mode follows explicit model-task semantics, and coefficients are local-surrogate "
                "weights in LIME's interpretable/discretized representation, not causal effects."
            ),
        ),
    )

    # Register SHAP (KernelSHAP)
    registry.register(
        name="shap",
        explainer_class=ShapExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="SHapley Additive exPlanations (KernelSHAP)",
            paper_reference="Lundberg & Lee, 2017 - 'A Unified Approach to Interpreting Model Predictions'",
            complexity="O(2^n_features) approximated",
            requires_training_data=True,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Official KernelSHAP for finite sample-first numerical outputs with explicit "
                "or model-derived task semantics, sampling, regularisation, link, and "
                "output-space metadata; class_names are display-only, ambiguous multi-output "
                "models require an explicit task, and multi-output regression requires an "
                "output index. The single-explanation API requires top_labels=1."
            ),
        ),
    )

    # Register the shap.TreeExplainer adapter.
    registry.register(
        name="treeshap",
        explainer_class=TreeShapExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["tree", "ensemble"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="TreeSHAP via shap.TreeExplainer with explicit output/perturbation scope",
            paper_reference="Lundberg et al., 2018 - 'Consistent Individualized Feature Attribution for Tree Ensembles'",
            complexity="O(TLD^2) - polynomial in tree depth",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "shap.TreeExplainer for supported single-target sklearn/XGBoost trees with "
                "explicit output targeting and output-space contracts; regression is "
                "single-output, interventional explanations require background data, and "
                "interaction values are supported only with tree_path_dependent perturbation."
            ),
        ),
    )

    # Register the canonical continuous-tabular Anchors implementation.
    registry.register(
        name="anchor_tabular",
        explainer_class=AnchorTabularExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification"],
            description="Continuous-tabular Anchors with KL-LUCB candidate selection",
            paper_reference=(
                "Ribeiro et al., 2018 - 'Anchors: High-Precision "
                "Model-Agnostic Explanations' (Algorithms 1-2 scope)"
            ),
            complexity="Budget-bounded adaptive sampling within a bounded beam search",
            requires_training_data=True,
            supports_batching=False,
            claim_status="verified",
            claim_scope=(
                "Paper Algorithms 1-2-style KL-LUCB candidate selection for finite continuous "
                "numeric tabular inputs with query-consistent one-sided quartile/decile threshold "
                "predicates under the uniform empirical joint distribution of an explicitly "
                "supplied background table; conditional draws restrict to whole satisfying "
                "rows and preserve observed feature dependence. The confidence statement "
                "assumes fixed deterministic model predictions (or independent stationary "
                "prediction randomness). Each explanation fixes the model's predicted output column; "
                "class_names are display-only and their length is validated against output width. "
                "Only candidates with strict lower_bound > threshold are certified, and budget "
                "exhaustion is returned explicitly as uncertified. Bounded beam search does not "
                "guarantee a globally maximum-coverage anchor or causal sufficiency."
            ),
        ),
    )

    # Register Anchors
    registry.register(
        name="anchors",
        explainer_class=AnchorsExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification"],
            description="Approximate fixed-sample Anchors-style beam-search heuristic",
            paper_reference="Inspired by Ribeiro et al., 2018; does not implement Anchors' KL-LUCB guarantee",
            complexity="O(beam_size * n_features * n_samples)",
            requires_training_data=True,
            supports_batching=False,
            claim_status="quarantined",
            claim_scope=(
                "Fixed-sample bounded-beam Anchors-style heuristic with empirical precision/"
                "coverage and exact structured bin bounds; no KL-LUCB sampling, "
                "high-probability precision guarantee, or globally maximum-coverage claim."
            ),
        ),
    )

    # Register constrained counterfactual search (historical public name)
    registry.register(
        name="counterfactual",
        explainer_class=CounterfactualExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification"],
            description="Constrained multi-start counterfactual search heuristic",
            paper_reference="Inspired by Mothilal et al., 2020; not the DiCE algorithm",
            complexity="O(n_counterfactuals * optimization_steps)",
            requires_training_data=True,
            supports_batching=False,
            claim_status="quarantined",
            claim_scope=(
                "Constrained multi-start heuristic returning only feasible target-class "
                "candidates found within the search budget; an empty result is a search "
                "failure, not proof of infeasibility, and action magnitudes are emitted only "
                "for successful searches; this is not DiCE."
            ),
        ),
    )

    # Register Integrated Gradients (for neural networks)
    registry.register(
        name="integrated_gradients",
        explainer_class=IntegratedGradientsExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["tabular", "image"],
            task_types=["classification", "regression"],
            description="Straight-line path-gradient quadrature for neural networks (requires PyTorch)",
            paper_reference="Sundararajan et al., 2017 - 'Axiomatic Attribution for Deep Networks' (ICML)",
            complexity="O(n_steps * forward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Finite real PyTorch tabular/image inputs with an exact configured or "
                "inferred input shape, exact finite adapter gradients, a fixed output "
                "target, explicit score-space contracts, and operation-local random-baseline "
                "generation that does not mutate NumPy's global RNG. Implementation "
                "verification is CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register Grad-CAM
    registry.register(
        name="gradcam",
        explainer_class=GradCAMExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description="Grad-CAM channel-mean-gradient weighting over spatial activations",
            paper_reference="Selvaraju et al., ICCV 2017, Grad-CAM, arXiv:1610.02391",
            complexity="O(forward_pass + backward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Canonical Grad-CAM equations 1-2 for one finite image, one 4-D "
                "spatial target-layer output, one fixed scalar output target, and an "
                "explicit CHW/HWC layout whenever automatic layout inference is ambiguous. "
                "Implementation verification is CPU-only; CUDA is outside the audited "
                "device scope."
            ),
        ),
    )

    # Register DeepLIFT (for neural networks)
    registry.register(
        name="deeplift",
        explainer_class=DeepLIFTExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="DeepLIFT - reference-based attributions via activation differences (requires PyTorch)",
            paper_reference="Shrikumar et al., 2017 - 'Learning Important Features Through Propagating Activation Differences' (ICML)",
            complexity="O(forward_pass + backward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Captum DeepLift on one-dimensional flat feature vectors and explicitly "
                "supported PyTorch graph/module contracts. Implementation verification is "
                "CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register DeepSHAP (DeepLIFT + SHAP)
    registry.register(
        name="deepshap",
        explainer_class=DeepLIFTShapExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description=(
                "Expected DeepLIFT contributions over background samples that approximate "
                "SHAP values (requires PyTorch)"
            ),
            paper_reference="Lundberg & Lee, 2017 - combines DeepLIFT with SHAP",
            complexity="O(n_background * forward_pass)",
            requires_training_data=True,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Captum DeepLiftShap over validated flat background samples and supported "
                "graph/module contracts; SHAP-value language is approximate and subject to "
                "Captum's assumptions. Implementation verification is CPU-only; CUDA is "
                "outside the audited device scope."
            ),
        ),
    )

    # Register SmoothGrad (for neural networks)
    registry.register(
        name="smoothgrad",
        explainer_class=SmoothGradExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="SmoothGrad - configured-noise-averaged PyTorch input gradients",
            paper_reference="Smilkov et al., 2017 - 'SmoothGrad: removing noise by adding noise' (ICML Workshop)",
            complexity="O(n_samples * gradient evaluation)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Configured-noise-averaged PyTorch gradients for one-dimensional tabular "
                "vectors, with a fixed output target and per-call local NumPy Generator; "
                "integer random_state repeats the perturbation sequence. Implementation "
                "verification is CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register Saliency Maps (for neural networks)
    registry.register(
        name="saliency",
        explainer_class=SaliencyExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="Saliency Maps - gradient-based feature attribution (requires PyTorch)",
            paper_reference="Simonyan et al., 2014 - 'Deep Inside Convolutional Networks' (ICLR Workshop)",
            complexity="O(forward_pass + backward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "One-dimensional finite flat feature vectors with exactly one declared "
                "feature name per input dimension, a fixed integer output target, and "
                "explicit adapter score-space metadata. Implementation verification is "
                "CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register TCAV (Concept-based explanations for neural networks)
    registry.register(
        name="tcav",
        explainer_class=TCAVExplainer,
        meta=ExplainerMeta(
            scope="global",
            model_types=["neural"],
            data_types=["tabular", "image"],
            task_types=["classification"],
            description="Global target-class-set TCAV concept-sensitivity aggregate",
            paper_reference="Kim et al., 2018 - 'Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors' (ICML)",
            complexity="O(n_concepts * n_test_inputs * forward_pass)",
            requires_training_data=True,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Global fraction of positive directional derivatives over a caller-supplied "
                "target-class input set in flattened bottleneck space. require_logit_scores=True "
                "enforces canonical class-logit TCAV; other adapter score spaces are explicitly "
                "labeled variants. CAV accuracies use configurable held-out evaluation when the "
                "sample size permits it and label tiny-set training fallbacks explicitly; "
                "repeated-run statistics are diagnostic and nonconfirmatory. Implementation "
                "verification is CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register LRP (Layer-wise Relevance Propagation)
    registry.register(
        name="lrp",
        explainer_class=LRPExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["tabular", "image"],
            task_types=["classification", "regression"],
            description="LRP - Layer-wise Relevance Propagation for decomposition-based attributions (requires PyTorch)",
            paper_reference="Bach et al., 2015 - 'On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation' (PLOS ONE)",
            complexity="O(n_layers * forward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Captum-backed epsilon, gamma, and z-plus rules plus a constrained analytical "
                "alpha-beta graph subset. For a one-logit classifier, class 0 is supported only "
                "by the sign-equivariant epsilon rule; asymmetric rules and composites fail "
                "instead of applying an undeclared post-hoc sign convention. Implementation "
                "verification is CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register HiResCAM
    registry.register(
        name="hirescam",
        explainer_class=HiResCAMExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description="HiResCAM channel-summed elementwise activation-gradient map",
            paper_reference="Draelos & Carin, Medical Image Analysis 2021, HiResCAM",
            complexity="O(forward_pass + backward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Canonical elementwise activation-gradient formula for one 4-D spatial "
                "layer and fixed scalar target. No architecture-wide theorem is asserted; "
                "the paper's guarantee is limited to a CNN ending in one fully connected layer. "
                "Implementation verification is CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register XGrad-CAM
    registry.register(
        name="xgradcam",
        explainer_class=XGradCAMExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description="XGrad-CAM activation-normalized gradient channel weighting",
            paper_reference="Fu et al., BMVC 2020, XGrad-CAM",
            complexity="O(forward_pass + backward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Canonical equations 7-8 for one 4-D spatial layer and fixed scalar "
                "target; undefined nonzero channels with zero activation sum are rejected. "
                "No unconditional sensitivity or conservation guarantee is asserted. "
                "Implementation verification is CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register LayerCAM
    registry.register(
        name="layercam",
        explainer_class=LayerCAMExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description="LayerCAM positive-spatial-gradient activation map",
            paper_reference="Jiang et al., IEEE TIP 2021, LayerCAM",
            complexity="O(forward_pass + backward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Canonical positive-gradient-times-activation formula for one compatible "
                "4-D spatial layer and one fixed scalar target. Implementation verification "
                "is CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register Eigen-CAM
    registry.register(
        name="eigencam",
        explainer_class=EigenCAMExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description="Eigen-CAM class-agnostic raw-activation SVD projection",
            paper_reference="Muhammad & Yeasin, IJCNN 2020, Eigen-CAM",
            complexity="O(forward_pass + SVD)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Canonical uncentered SVD projection of one 4-D spatial activation tensor; "
                "the method is class-agnostic, rejects explicit targets, and discloses its "
                "deterministic SVD sign convention. Implementation verification is CPU-only; "
                "CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register Score-CAM
    registry.register(
        name="scorecam",
        explainer_class=ScoreCAMExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description="Score-CAM paper Algorithm-1 raw-output/channel-softmax variant",
            paper_reference="Wang et al., CVPR Workshops 2020, Score-CAM",
            complexity="O(n_channels * forward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Verified transcription of paper Algorithm 1: normalized activation masks, "
                "raw model-output target scores, and softmax across channels. This differs "
                "from section 3.2 and the authors' released post-softmax probability-weighting "
                "code. Paper logit-space match is asserted only when the adapter declares "
                "raw_model_output_space='logit'; one-logit binary target expansion is unsupported. "
                "Implementation verification is CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register the pytorch-grad-cam EigenGradCAM library variant
    registry.register(
        name="eigengradcam",
        explainer_class=EigenGradCAMExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description=(
                "pytorch-grad-cam EigenGradCAM library variant using centered SVD of "
                "gradient-weighted activations"
            ),
            paper_reference=(
                "jacobgil/pytorch-grad-cam, eigen_grad_cam.py library implementation; "
                "not attributed to Muhammad & Yeasin's Eigen-CAM paper"
            ),
            complexity="O(forward_pass + backward_pass + SVD)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="quarantined",
            claim_scope=(
                "Verified only against the pytorch-grad-cam centered-SVD library operation; "
                "it is not attributed to the Eigen-CAM paper or another primary method paper. "
                "Implementation verification is CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # Register the pytorch-grad-cam GradCAMElementWise library variant
    registry.register(
        name="gradcam_elementwise",
        explainer_class=GradCAMElementWiseExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description=(
                "pytorch-grad-cam GradCAMElementWise library variant using per-element "
                "rectified gradient-activation products"
            ),
            paper_reference=(
                "jacobgil/pytorch-grad-cam, grad_cam_elementwise.py element-wise library "
                "implementation"
            ),
            complexity="O(forward_pass + backward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="quarantined",
            claim_scope=(
                "Verified only against the pytorch-grad-cam per-element rectification "
                "operation; it is not attributed to the Grad-CAM paper or another primary "
                "method paper. Implementation verification is CPU-only; CUDA is outside the "
                "audited device scope."
            ),
        ),
    )

    # Register Ablation-CAM
    registry.register(
        name="ablationcam",
        explainer_class=AblationCAMExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description="Ablation-CAM target-layer channel-ablation map using raw class scores",
            paper_reference="Desai & Ramaswamy, WACV 2020, Ablation-CAM",
            complexity="O(n_channels * forward_pass)",
            requires_training_data=False,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Canonical target-layer channel zeroing with raw output scores for spatial "
                "PyTorch models whose target module runs once per forward, whose raw outputs "
                "map one-to-one to target indices, and whose original target score is nonzero; "
                "one-logit binary adapters are unsupported. Implementation verification is "
                "CPU-only; CUDA is outside the audited device scope."
            ),
        ),
    )

    # =========================================================================
    # Global Explainers (model-level)
    # =========================================================================

    # Register Permutation Importance
    registry.register(
        name="permutation_importance",
        explainer_class=PermutationImportanceExplainer,
        meta=ExplainerMeta(
            scope="global",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="Feature importance via permutation-based performance degradation",
            paper_reference="Breiman, 2001 - 'Random Forests' (Machine Learning)",
            complexity="O(n_features * n_repeats * n_samples)",
            requires_training_data=True,
            supports_batching=False,
            claim_status="verified",
            claim_scope="Explicit classification-accuracy or regression-R2 score-drop semantics.",
        ),
    )

    # Register Partial Dependence
    registry.register(
        name="partial_dependence",
        explainer_class=PartialDependenceExplainer,
        meta=ExplainerMeta(
            scope="global",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="Empirical partial-dependence averages of a selected prediction output",
            paper_reference="Friedman, 2001 - 'Greedy Function Approximation' (Annals of Statistics)",
            complexity="O(grid_resolution * n_samples)",
            requires_training_data=True,
            supports_batching=False,
            claim_status="verified",
            claim_scope="Empirical tabular PDP with explicit output targeting and grid semantics.",
        ),
    )

    # Register ALE
    registry.register(
        name="ale",
        explainer_class=ALEExplainer,
        meta=ExplainerMeta(
            scope="global",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="Continuous first-order Accumulated Local Effects with empirical quantile bins",
            paper_reference="Apley & Zhu, 2020 - 'Visualizing the Effects of Predictor Variables' (JRSS-B)",
            complexity="O(n_bins * n_samples)",
            requires_training_data=True,
            supports_batching=False,
            claim_status="verified",
            claim_scope=(
                "ALEPlot-compatible continuous first-order ALE with inverse-empirical-CDF "
                "(R type-1) quantile edges and empirical weighted centering; nominal-feature "
                "ALE is unsupported, and curve_range is descriptive rather than an attribution."
            ),
        ),
    )

    # Register SAGE
    registry.register(
        name="sage",
        explainer_class=SAGEExplainer,
        meta=ExplainerMeta(
            scope="global",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="Shapley Additive Global importancE - global feature importance via Shapley values",
            paper_reference="Covert et al., 2020 - 'Understanding Global Feature Contributions' (NeurIPS)",
            complexity="O(n_permutations * n_features * n_samples)",
            requires_training_data=True,
            supports_batching=False,
            claim_status="verified",
            claim_scope=(
                "Permutation-sampled marginal-imputer SAGE loss game: zero-one loss for "
                "single-target classification with output columns mapped through model.classes_, "
                "MSE for regression, or caller-supplied loss; efficiency is reported against "
                "null-minus-full loss."
            ),
        ),
    )

    # =========================================================================
    # Example-Based Explainers
    # =========================================================================

    # Register ProtoDash
    registry.register(
        name="protodash",
        explainer_class=ProtoDashExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="ProtoDash - prototype selection with importance weights for example-based explanations",
            paper_reference="Gurumoorthy et al., 2019 - 'Efficient Data Representation by Selecting Prototypes' (ICDM)",
            complexity="O(n_prototypes * n_samples^2)",
            requires_training_data=True,
            supports_batching=True,
            claim_status="verified",
            claim_scope=(
                "Canonical ProtoDash maximum-gradient greedy support selection with "
                "nonnegative QP objective weights over supported tabular kernels; normalized "
                "display weights are derived, and the criticism helper is a separate "
                "non-MMD-Critic diagnostic."
            ),
        ),
    )

    return registry


# Lazy initialization to avoid circular imports
_default_registry: Optional[ExplainerRegistry] = None


def get_default_registry() -> ExplainerRegistry:
    """Get the default global registry (lazy initialization)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = _create_default_registry()
    return _default_registry


# For convenience, expose as module-level variable
# This will be initialized on first access
class _LazyRegistry:
    """Lazy proxy for the default registry."""

    def __getattr__(self, name):
        return getattr(get_default_registry(), name)

    def __contains__(self, item):
        return item in get_default_registry().list_explainers()


default_registry = _LazyRegistry()
