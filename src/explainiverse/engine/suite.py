# src/explainiverse/engine/suite.py

from explainiverse.core.explanation import Explanation
from explainiverse.explainers.attribution.lime_wrapper import LimeExplainer
from explainiverse.explainers.attribution.shap_wrapper import ShapExplainer
from explainiverse.evaluation.metrics import compute_roar
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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
        


    def evaluate_roar(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        top_k: int = 2,
        model_class=None,
        model_kwargs: dict = None
    ):
        """
        Evaluate each explainer using ROAR (Remove And Retrain).
        
        Args:
            X_train, y_train: training data
            X_test, y_test: test data
            top_k: number of features to mask
            model_class: model constructor with .fit() and .predict() (default: same as current model)
            model_kwargs: optional keyword args for new model instance

        Returns:
            Dict of {explainer_name: accuracy drop (baseline - retrained)}
        """
        from explainiverse.evaluation.metrics import compute_roar

        model_kwargs = model_kwargs or {}

        # Default to type(self.model.model) if not provided
        if model_class is None:
            model_class = type(self.model.model)

        roar_scores = {}

        for name, explanation in self.explanations.items():
            print(f"[ROAR] Evaluating explainer: {name}")
            roar = compute_roar(
                model_class=model_class,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                explanations=[explanation],  # single-instance for now
                top_k=top_k,
                model_kwargs=model_kwargs
            )
            roar_scores[name] = roar

        return roar_scores