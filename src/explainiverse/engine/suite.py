# src/explainiverse/engine/suite.py

from explainiverse.core.explanation import Explanation
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer


class ExplanationSuite:
    """
    Runs multiple explainers on a single instance and compares their outputs.
    """

    def __init__(self, model, explainer_configs, data_meta=None):
        """
        Args:
            model: a model adapter (e.g., SklearnAdapter)
            explainer_configs: list of (name, kwargs) tuples for explainers
            data_meta: optional metadata about the task, scope, or preference
        """
        self.model = model
        self.configs = explainer_configs
        self.data_meta = data_meta or {}
        self.explanations = {}

    def run(self, instance):
        """
        Run all configured explainers on a single instance.
        """
        for name, params in self.configs:
            explainer = self._load_explainer(name, **params)
            explanation = explainer.explain(instance)
            self.explanations[name] = explanation
        return self.explanations

    def compare(self):
        """
        Print attribution scores side-by-side.
        """
        keys = set()
        for explanation in self.explanations.values():
            keys.update(explanation.explanation_data.get("feature_attributions", {}).keys())

        print("\nSide-by-Side Comparison:")
        for key in sorted(keys):
            row = [f"{key}"]
            for name in self.explanations:
                value = self.explanations[name].explanation_data.get("feature_attributions", {}).get(key, "—")
                row.append(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
            print(" | ".join(row))

    def suggest_best(self):
        """
        Suggest the best explainer based on model type, output structure, and task metadata.
        """
        if "task" in self.data_meta:
            task = self.data_meta["task"]
        else:
            task = "unknown"

        model = self.model.model

        # 1. Regression: SHAP preferred due to consistent output
        if task == "regression":
            return "shap"

        # 2. Model with `predict_proba` → SHAP handles probabilistic outputs well
        if hasattr(model, "predict_proba"):
            try:
                output = self.model.predict([[0] * model.n_features_in_])
                if output.shape[1] > 2:
                    return "shap"  # Multi-class, SHAP more stable
                else:
                    return "lime"  # Binary, both are okay
            except Exception:
                return "shap"

        # 3. Tree-based models → prefer SHAP (TreeSHAP if available)
        if "tree" in str(type(model)).lower():
            return "shap"

        # 4. Default fallback
        return "lime"

    def _load_explainer(self, name, **kwargs):
        if name == "lime":
            return LimeExplainer(model=self.model, **kwargs)
        elif name == "shap":
            return ShapExplainer(model=self.model, **kwargs)
        else:
            raise ValueError(f"Unknown explainer: {name}")