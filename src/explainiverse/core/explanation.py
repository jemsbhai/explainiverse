# src/explainiverse/core/explanation.py

class Explanation:
    """
    Unified container for explanation results.
    """

    def __init__(self, explainer_name: str, target_class: str, explanation_data: dict):
        self.explainer_name = explainer_name
        self.target_class = target_class
        self.explanation_data = explanation_data  # e.g., {'feature_attributions': {...}}

    def __repr__(self):
        return (f"Explanation(explainer='{self.explainer_name}', "
                f"target='{self.target_class}', "
                f"keys={list(self.explanation_data.keys())})")

    def plot(self, type='bar'):
        """
        Visualizes the explanation.
        This will later integrate with a proper visualization backend.
        """
        print(f"[plot: {type}] Plotting explanation for {self.target_class} "
              f"from {self.explainer_name}.")
