"""Integrity checks for the installed public Python namespace."""

from importlib import import_module
from importlib.metadata import version

import pytest

import explainiverse

PUBLIC_MODULES = (
    "explainiverse",
    "explainiverse.adapters",
    "explainiverse.core",
    "explainiverse.engine",
    "explainiverse.evaluation",
    "explainiverse.explainers",
    "explainiverse.explainers.attribution",
    "explainiverse.explainers.counterfactual",
    "explainiverse.explainers.example_based",
    "explainiverse.explainers.global_explainers",
    "explainiverse.explainers.gradient",
    "explainiverse.explainers.rule_based",
)


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_every_declared_public_export_resolves_once(module_name):
    module = import_module(module_name)
    exports = getattr(module, "__all__", None)
    assert isinstance(exports, list), module_name
    assert len(exports) == len(set(exports)), module_name
    for export in exports:
        assert isinstance(export, str) and export
        assert hasattr(module, export), f"{module_name}.{export} is declared but absent"


def test_anchor_exports_keep_canonical_and_compatibility_classes_distinct():
    from explainiverse.explainers import AnchorsExplainer, AnchorTabularExplainer
    from explainiverse.explainers.rule_based import AnchorsExplainer as RuleAnchorsExplainer
    from explainiverse.explainers.rule_based import (
        AnchorTabularExplainer as RuleAnchorTabularExplainer,
    )

    assert AnchorTabularExplainer is RuleAnchorTabularExplainer
    assert AnchorsExplainer is RuleAnchorsExplainer
    assert AnchorTabularExplainer is not AnchorsExplainer


def test_runtime_version_matches_installed_distribution_metadata():
    assert explainiverse.__version__ == version("explainiverse")


def test_default_registry_has_one_entry_per_name_and_no_gradcam_plus_plus_alias():
    registry = explainiverse.get_default_registry()
    names = registry.list_explainers()
    assert len(names) == len(set(names))
    assert "gradcam++" not in names
    assert "gradcamplusplus" not in names
    for name in names:
        entry = registry.get(name)
        returned_meta = registry.get_meta(name)
        assert entry["meta"] == returned_meta
        assert entry["meta"] is not returned_meta
        assert entry["class"].__name__
