# src/explainiverse/engine/suite.py
"""
ExplanationSuite - Multi-explainer comparison and evaluation.

Provides utilities for running multiple explainers on the same instances
and comparing their outputs.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class ExplanationSuite:
    """
    Run and compare multiple explainers on the same instances.
    
    This class provides a unified interface for:
    - Running multiple explainers on a single instance
    - Comparing attribution scores side-by-side
    - Suggesting the best explainer based on model/task characteristics
    - Evaluating explainers using ROAR (Remove And Retrain)
    
    Example:
        >>> from explainiverse import ExplanationSuite, SklearnAdapter
        >>> suite = ExplanationSuite(
        ...     model=adapter,
        ...     explainer_configs=[
        ...         ("lime", {"training_data": X_train, "feature_names": fnames, "class_names": cnames}),
        ...         ("shap", {"background_data": X_train[:50], "feature_names": fnames, "class_names": cnames}),
        ...     ]
        ... )
        >>> results = suite.run(X_test[0])
        >>> suite.compare()
    """

    def __init__(
        self,
        model,
        explainer_configs: List[Tuple[str, Dict[str, Any]]],
        data_meta: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ExplanationSuite.
        
        Args:
            model: A model adapter (e.g., SklearnAdapter, PyTorchAdapter)
            explainer_configs: List of (explainer_name, kwargs) tuples.
                The explainer_name should match a registered explainer in
                the default_registry (e.g., "lime", "shap", "treeshap").
            data_meta: Optional metadata about the task, scope, or preference.
                Can include "task" ("classification" or "regression").
        """
        self.model = model
        self.configs = explainer_configs
        self.data_meta = data_meta or {}
        self.explanations: Dict[str, Any] = {}
        self._registry = None

    def _get_registry(self):
        """Lazy load the registry to avoid circular imports."""
        if self._registry is None:
            from explainiverse.core.registry import default_registry
            self._registry = default_registry
        return self._registry

    def run(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Run all configured explainers on a single instance.
        
        Args:
            instance: Input instance to explain (1D numpy array)
            
        Returns:
            Dictionary mapping explainer names to Explanation objects
        """
        instance = np.asarray(instance)
        registry = self._get_registry()
        
        for name, params in self.configs:
            try:
                explainer = registry.create(name, model=self.model, **params)
                explanation = explainer.explain(instance)
                self.explanations[name] = explanation
            except Exception as e:
                print(f"[ExplanationSuite] Warning: Failed to run {name}: {e}")
                continue
        
        return self.explanations

    def compare(self) -> None:
        """
        Print attribution scores side-by-side for comparison.
        """
        if not self.explanations:
            print("No explanations to compare. Run suite.run(instance) first.")
            return
        
        # Collect all feature names across explanations
        all_keys = set()
        for explanation in self.explanations.values():
            attrs = explanation.explanation_data.get("feature_attributions", {})
            all_keys.update(attrs.keys())

        print("\nSide-by-Side Comparison:")
        print("-" * 60)
        
        # Header
        header = ["Feature"] + list(self.explanations.keys())
        print(" | ".join(f"{h:>15}" for h in header))
        print("-" * 60)
        
        # Rows
        for key in sorted(all_keys):
            row = [f"{key:>15}"]
            for name in self.explanations:
                value = self.explanations[name].explanation_data.get(
                    "feature_attributions", {}
                ).get(key, None)
                if value is not None:
                    row.append(f"{value:>15.4f}")
                else:
                    row.append(f"{'—':>15}")
            print(" | ".join(row))

    def suggest_best(self) -> str:
        """
        Suggest the best explainer based on model type and task characteristics.
        
        Returns:
            Name of the suggested explainer
        """
        task = self.data_meta.get("task", "unknown")
        model = self.model.model if hasattr(self.model, 'model') else self.model

        # 1. Regression: SHAP preferred due to consistent output
        if task == "regression":
            return "shap"

        # 2. Model with predict_proba → SHAP handles probabilistic outputs well
        if hasattr(model, "predict_proba"):
            try:
                # Check output dimensions
                if hasattr(model, 'n_features_in_'):
                    test_input = np.zeros((1, model.n_features_in_))
                    output = self.model.predict(test_input)
                    if output.shape[1] > 2:
                        return "shap"  # Multi-class, SHAP more stable
                    else:
                        return "lime"  # Binary, both are okay
            except Exception:
                return "shap"

        # 3. Tree-based models → prefer TreeSHAP
        model_type_str = str(type(model)).lower()
        if any(tree_type in model_type_str for tree_type in ['tree', 'forest', 'xgb', 'lgbm', 'catboost']):
            return "treeshap"

        # 4. Neural networks → prefer gradient methods
        if 'torch' in model_type_str or 'keras' in model_type_str or 'tensorflow' in model_type_str:
            return "integrated_gradients"

        # 5. Default fallback
        return "lime"

    def evaluate_roar(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        top_k: int = 2,
        model_class=None,
        model_kwargs: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Evaluate each explainer using ROAR (Remove And Retrain).
        
        ROAR measures explanation quality by retraining the model after
        removing the top-k important features identified by each explainer.
        A larger accuracy drop indicates more faithful explanations.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            top_k: Number of features to mask
            model_class: Model constructor with .fit() and .predict()
                        If None, uses the same type as self.model.model
            model_kwargs: Optional keyword args for new model instance

        Returns:
            Dict mapping explainer names to accuracy drops
        """
        from explainiverse.evaluation.metrics import compute_roar

        model_kwargs = model_kwargs or {}

        # Default to type(self.model.model) if not provided
        if model_class is None:
            raw_model = self.model.model if hasattr(self.model, 'model') else self.model
            model_class = type(raw_model)

        roar_scores = {}

        for name, explanation in self.explanations.items():
            print(f"[ROAR] Evaluating explainer: {name}")
            try:
                roar = compute_roar(
                    model_class=model_class,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    explanations=[explanation],
                    top_k=top_k,
                    model_kwargs=model_kwargs
                )
                roar_scores[name] = roar
            except Exception as e:
                print(f"[ROAR] Failed for {name}: {e}")
                roar_scores[name] = 0.0

        return roar_scores
    
    def get_explanation(self, name: str):
        """
        Get a specific explanation by explainer name.
        
        Args:
            name: Name of the explainer
            
        Returns:
            Explanation object or None if not found
        """
        return self.explanations.get(name)
    
    def list_explainers(self) -> List[str]:
        """
        List all configured explainer names.
        
        Returns:
            List of explainer names
        """
        return [name for name, _ in self.configs]
    
    def list_completed(self) -> List[str]:
        """
        List explainers that have been run successfully.
        
        Returns:
            List of explainer names with results
        """
        return list(self.explanations.keys())
