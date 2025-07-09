# tests/test_all.py
import sys
import os
sys.path.insert(0, os.path.abspath(".")) 

from tests.test_sklearn_adapter import (
    test_sklearn_adapter_prediction,
    test_predict_proba_shape_and_range,
    test_predict_fallback_without_proba
)

from tests.test_lime_explainer import (
    test_lime_explainer_target_and_shape,
    test_lime_explainer_num_features_range,
    test_lime_explainer_top_labels,
    test_lime_explainer_plot_does_not_crash
)

def run_all_tests():
    print("Running SklearnAdapter tests...")
    test_sklearn_adapter_prediction()
    test_predict_proba_shape_and_range()
    test_predict_fallback_without_proba()

    print("Running LimeExplainer tests...")
    test_lime_explainer_target_and_shape()
    test_lime_explainer_num_features_range()
    test_lime_explainer_top_labels()
    test_lime_explainer_plot_does_not_crash()

    print(" All core tests passed.")


if __name__ == "__main__":
    run_all_tests()