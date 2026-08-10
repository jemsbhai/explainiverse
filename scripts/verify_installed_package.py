"""Smoke-test an installed Explainiverse distribution as a consumer.

Run this script with the interpreter from an isolated wheel/sdist environment.
It deliberately imports the installed package rather than the checkout source.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import pkgutil

import numpy as np

import explainiverse
import explainiverse.evaluation as evaluation
from explainiverse import BaseExplainer, Explanation, default_registry


def _import_every_package_module() -> None:
    discovered = list(
        pkgutil.walk_packages(explainiverse.__path__, prefix=f"{explainiverse.__name__}.")
    )
    if not discovered:
        raise AssertionError("the installed distribution exposes no package modules")
    for module in discovered:
        importlib.import_module(module.name)


class _SmokeExplainer(BaseExplainer):
    def __init__(self) -> None:
        super().__init__(model=None)

    def explain(self, instance: np.ndarray) -> Explanation:
        values = np.asarray(instance)
        return Explanation(
            explainer_name="installed-package-smoke",
            target_class="output",
            explanation_data={
                "feature_attributions": {
                    f"f{index}": float(value) for index, value in enumerate(values)
                }
            },
            feature_names=[f"f{index}" for index in range(values.size)],
        )


def main() -> None:
    installed_version = importlib.metadata.version("explainiverse")
    if installed_version != explainiverse.__version__:
        raise AssertionError(
            f"metadata version {installed_version!r} != package version "
            f"{explainiverse.__version__!r}"
        )

    _import_every_package_module()

    explainer_names = default_registry.list_explainers()
    if not explainer_names:
        raise AssertionError("the installed explainer registry is empty")
    for name in explainer_names:
        default_registry.get_meta(name)

    metric_registry = evaluation.default_metric_registry
    metric_registry.validate_inventory(evaluation.__all__)
    if not metric_registry.list_metrics():
        raise AssertionError("the installed metric registry is empty")

    score = evaluation.compute_sparseness(
        _SmokeExplainer(), np.asarray([1.0, 0.0], dtype=np.float64)
    )
    if not math.isfinite(score):
        raise AssertionError("the installed metric smoke test returned a non-finite value")

    print(
        f"Explainiverse {installed_version}: imported all modules, "
        f"{len(explainer_names)} explainers, "
        f"{len(metric_registry.list_metrics())} metrics"
    )


if __name__ == "__main__":
    main()
