# src/explainiverse/core/registry.py
"""
ExplainerRegistry - A plugin system for XAI methods.

This module provides a flexible registry that allows:
- Registration of explainers with rich metadata
- Filtering/discovery by scope, model type, data type, task type
- Easy instantiation with dependency injection
- Decorator-based registration for clean syntax
- Recommendations based on use case

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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type, Callable
from explainiverse.core.explainer import BaseExplainer


@dataclass
class ExplainerMeta:
    """
    Metadata for an explainer, used for discovery and recommendations.
    
    Attributes:
        scope: "local" (instance-level) or "global" (model-level)
        model_types: List of compatible model types ("any", "tree", "linear", "neural", "ensemble")
        data_types: List of compatible data types ("tabular", "image", "text", "time_series")
        task_types: List of compatible tasks ("classification", "regression")
        description: Human-readable description of the explainer
        paper_reference: Citation for the original paper
        complexity: Computational complexity (e.g., "O(n)", "O(n^2)")
        requires_training_data: Whether the explainer needs training data
        supports_batching: Whether the explainer can process batches efficiently
    """
    scope: str  # "local" or "global"
    model_types: List[str] = field(default_factory=lambda: ["any"])
    data_types: List[str] = field(default_factory=lambda: ["tabular"])
    task_types: List[str] = field(default_factory=lambda: ["classification", "regression"])
    description: str = ""
    paper_reference: Optional[str] = None
    complexity: Optional[str] = None
    requires_training_data: bool = False
    supports_batching: bool = False
    
    def matches(
        self,
        scope: Optional[str] = None,
        model_type: Optional[str] = None,
        data_type: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> bool:
        """Check if this metadata matches the given criteria."""
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


class ExplainerRegistry:
    """
    Central registry for all explainers in Explainiverse.
    
    Provides:
    - Registration (programmatic and decorator-based)
    - Discovery and filtering
    - Instantiation with dependency injection
    - Recommendations based on use case
    """
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        explainer_class: Type[BaseExplainer],
        meta: ExplainerMeta,
        override: bool = False
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
        if name in self._registry and not override:
            raise ValueError(f"Explainer '{name}' is already registered. Use override=True to replace.")
        
        self._registry[name] = {
            "class": explainer_class,
            "meta": meta
        }
    
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
            Dict with "class" and "meta" keys
            
        Raises:
            KeyError: If the explainer is not registered
        """
        if name not in self._registry:
            raise KeyError(f"Explainer '{name}' is not registered. Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def get_meta(self, name: str) -> ExplainerMeta:
        """
        Get just the metadata for an explainer.
        
        Args:
            name: The explainer name
            
        Returns:
            ExplainerMeta instance
            
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
            List of names or dict of {name: {"class": ..., "meta": ...}}
        """
        if with_meta:
            return dict(self._registry)
        return list(self._registry.keys())
    
    def filter(
        self,
        scope: Optional[str] = None,
        model_type: Optional[str] = None,
        data_type: Optional[str] = None,
        task_type: Optional[str] = None
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
        self,
        name: str,
        meta: ExplainerMeta
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
            info = f"  {name}: {meta.description or '(no description)'}"
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
        max_results: int = 5
    ) -> List[str]:
        """
        Recommend explainers based on criteria.
        
        This is a smarter version of filter() that ranks results
        by compatibility and preference.
        
        Args:
            model_type: The type of model being explained
            data_type: The type of data
            task_type: The ML task type
            scope_preference: Preferred scope ("local" or "global")
            max_results: Maximum number of recommendations
            
        Returns:
            List of recommended explainer names, ranked by relevance
        """
        candidates = self.filter(
            model_type=model_type,
            data_type=data_type,
            task_type=task_type
        )
        
        # Score candidates
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
            
            # Prefer explainers with documentation
            if meta.description:
                score += 1
            if meta.paper_reference:
                score += 2
            
            scored.append((name, score))
        
        # Sort by score descending
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
    from explainiverse.explainers.rule_based.anchors_wrapper import AnchorsExplainer
    from explainiverse.explainers.global_explainers.permutation_importance import PermutationImportanceExplainer
    from explainiverse.explainers.global_explainers.partial_dependence import PartialDependenceExplainer
    from explainiverse.explainers.global_explainers.ale import ALEExplainer
    from explainiverse.explainers.global_explainers.sage import SAGEExplainer
    from explainiverse.explainers.counterfactual.dice_wrapper import CounterfactualExplainer
    from explainiverse.explainers.gradient.integrated_gradients import IntegratedGradientsExplainer
    from explainiverse.explainers.gradient.gradcam import GradCAMExplainer
    
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
            data_types=["tabular", "text", "image"],
            task_types=["classification", "regression"],
            description="Local Interpretable Model-agnostic Explanations",
            paper_reference="Ribeiro et al., 2016 - 'Why Should I Trust You?'",
            complexity="O(n_samples * n_features)",
            requires_training_data=True,
            supports_batching=False
        )
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
            supports_batching=True
        )
    )
    
    # Register TreeSHAP (optimized for tree models)
    registry.register(
        name="treeshap",
        explainer_class=TreeShapExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["tree", "ensemble"],
            data_types=["tabular"],
            task_types=["classification", "regression"],
            description="TreeSHAP - exact SHAP values for tree-based models (RandomForest, XGBoost, etc.)",
            paper_reference="Lundberg et al., 2018 - 'Consistent Individualized Feature Attribution for Tree Ensembles'",
            complexity="O(TLD^2) - polynomial in tree depth",
            requires_training_data=False,
            supports_batching=True
        )
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
            description="High-precision rule-based explanations using beam search",
            paper_reference="Ribeiro et al., 2018 - 'Anchors: High-Precision Model-Agnostic Explanations' (AAAI)",
            complexity="O(beam_size * n_features * n_samples)",
            requires_training_data=True,
            supports_batching=False
        )
    )
    
    # Register Counterfactual (DiCE-style)
    registry.register(
        name="counterfactual",
        explainer_class=CounterfactualExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["any"],
            data_types=["tabular"],
            task_types=["classification"],
            description="Diverse counterfactual explanations via gradient-free optimization",
            paper_reference="Mothilal et al., 2020 - 'Explaining ML Classifiers through Diverse Counterfactual Explanations' (FAT*)",
            complexity="O(n_counterfactuals * optimization_steps)",
            requires_training_data=True,
            supports_batching=False
        )
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
            description="Integrated Gradients - axiomatic attributions for neural networks (requires PyTorch)",
            paper_reference="Sundararajan et al., 2017 - 'Axiomatic Attribution for Deep Networks' (ICML)",
            complexity="O(n_steps * forward_pass)",
            requires_training_data=False,
            supports_batching=True
        )
    )
    
    # Register GradCAM (for CNNs)
    registry.register(
        name="gradcam",
        explainer_class=GradCAMExplainer,
        meta=ExplainerMeta(
            scope="local",
            model_types=["neural"],
            data_types=["image"],
            task_types=["classification"],
            description="GradCAM/GradCAM++ - visual explanations for CNNs via gradient-weighted activations (requires PyTorch)",
            paper_reference="Selvaraju et al., 2017 - 'Grad-CAM: Visual Explanations from Deep Networks' (ICCV)",
            complexity="O(forward_pass + backward_pass)",
            requires_training_data=False,
            supports_batching=True
        )
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
            supports_batching=False
        )
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
            description="Marginal effect of features on predictions (PDP)",
            paper_reference="Friedman, 2001 - 'Greedy Function Approximation' (Annals of Statistics)",
            complexity="O(grid_resolution * n_samples)",
            requires_training_data=True,
            supports_batching=True
        )
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
            description="Accumulated Local Effects - unbiased alternative to PDP for correlated features",
            paper_reference="Apley & Zhu, 2020 - 'Visualizing the Effects of Predictor Variables' (JRSS-B)",
            complexity="O(n_bins * n_samples)",
            requires_training_data=True,
            supports_batching=True
        )
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
            supports_batching=False
        )
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
