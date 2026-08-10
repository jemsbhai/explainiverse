"""Failure and claim-scope contracts for ``ExplanationSuite``."""

import numpy as np
import pytest

from explainiverse.core.explanation import Explanation
from explainiverse.core.registry import ExplainerMeta
from explainiverse.engine.suite import ExplanationSuite


class _StaticExplainer:
    def __init__(self, result):
        self.result = result

    def explain(self, _instance):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _StaticRegistry:
    def __init__(self, result):
        self.result = result

    def create(self, _name, **_kwargs):
        return _StaticExplainer(self.result)


def _valid_explanation(value=1.0, *, target="target", comparison_contract=None):
    metadata = {}
    if comparison_contract is not None:
        metadata["comparison_contract"] = comparison_contract
    return Explanation(
        "static",
        target,
        {"feature_attributions": {"x": value}},
        feature_names=["x"],
        metadata=metadata,
    )


def _named_explanation(name, value, *, comparison_contract, feature_names=None):
    return Explanation(
        "static",
        "target",
        {"feature_attributions": {name: value}},
        feature_names=feature_names,
        metadata={"comparison_contract": comparison_contract},
    )


class _ReferenceExplainer:
    def __init__(self, **_kwargs):
        pass

    def explain(self, _instance, X_reference):
        return _valid_explanation(float(np.asarray(X_reference).shape[0]))


class _GlobalExplainer:
    def __init__(self, **_kwargs):
        pass

    def explain(self, _dataset):
        return _valid_explanation()


class _RequiredConstructorExplainer:
    def __init__(self, model, required_token):
        del model, required_token

    def explain(self, _instance):
        return _valid_explanation()


class _MetadataRegistry:
    def __init__(self):
        self.entries = {
            "reference": {
                "class": _ReferenceExplainer,
                "meta": ExplainerMeta(
                    scope="local",
                    claim_status="verified",
                    claim_scope="Test-only local contract.",
                ),
            },
            "global": {
                "class": _GlobalExplainer,
                "meta": ExplainerMeta(
                    scope="global",
                    claim_status="verified",
                    claim_scope="Test-only global contract.",
                ),
            },
            "required_constructor": {
                "class": _RequiredConstructorExplainer,
                "meta": ExplainerMeta(
                    scope="local",
                    claim_status="verified",
                    claim_scope="Test-only constructor contract.",
                ),
            },
        }

    def get(self, name):
        return self.entries[name]

    def get_meta(self, name):
        return self.entries[name]["meta"]

    def create(self, name, **kwargs):
        return self.entries[name]["class"](**kwargs)


@pytest.mark.parametrize(
    "configs,error_type",
    [
        ([], ValueError),
        (["lime"], TypeError),
        ([("", {})], ValueError),
        ([("lime", [])], TypeError),
        ([("lime", {}), ("lime", {})], ValueError),
    ],
)
def test_constructor_rejects_ambiguous_configs(configs, error_type):
    with pytest.raises(error_type):
        ExplanationSuite(None, configs)


@pytest.mark.parametrize(
    "instance",
    [np.array(1.0), np.array([]), np.array([1.0, np.nan]), ["not", "numeric"]],
)
def test_run_requires_one_nonempty_finite_numeric_array(instance):
    suite = ExplanationSuite(None, [("static", {})])
    suite._registry = _StaticRegistry(_valid_explanation())
    with pytest.raises((TypeError, ValueError)):
        suite.run(instance)


def test_run_surfaces_explainer_failure_and_clears_stale_results():
    suite = ExplanationSuite(None, [("static", {})])
    suite.explanations = {"stale": _valid_explanation()}
    suite._registry = _StaticRegistry(RuntimeError("deliberate explainer failure"))

    with pytest.raises(RuntimeError, match="deliberate explainer failure"):
        suite.run(np.array([1.0]))
    assert suite.explanations == {}


def test_invalid_run_input_also_clears_stale_results():
    suite = ExplanationSuite(None, [("static", {})])
    suite.explanations = {"stale": _valid_explanation()}
    suite._registry = _StaticRegistry(_valid_explanation())

    with pytest.raises(ValueError, match="at least one dimension"):
        suite.run(np.array(1.0))
    assert suite.explanations == {}


def test_suite_delegates_exact_multidimensional_shape_to_concrete_explainer():
    result = _valid_explanation()
    suite = ExplanationSuite(None, [("static", {})])
    suite._registry = _StaticRegistry(result)

    assert suite.run(np.ones((3, 4, 4))) == {"static": result}


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_run_preserves_numeric_dtype_and_isolates_the_caller_array(dtype):
    seen = []
    result = _valid_explanation()

    class CapturingExplainer:
        def explain(self, instance):
            seen.append(instance)
            instance[...] = 0
            return result

    class CapturingRegistry:
        def create(self, _name, **_kwargs):
            return CapturingExplainer()

    original = np.array([1, 2], dtype=dtype)
    suite = ExplanationSuite(None, [("capture", {})])
    suite._registry = CapturingRegistry()
    suite.run(original)

    assert seen[0].dtype == dtype
    np.testing.assert_array_equal(original, np.array([1, 2], dtype=dtype))


@pytest.mark.parametrize(
    "instance",
    [np.array([True, False]), np.array([1 + 2j]), np.array([1], dtype=object)],
)
def test_run_rejects_bool_complex_and_object_arrays(instance):
    suite = ExplanationSuite(None, [("static", {})])
    suite._registry = _StaticRegistry(_valid_explanation())
    with pytest.raises(TypeError, match="numeric|real|boolean"):
        suite.run(instance)


def test_run_rejects_non_explanation_results():
    suite = ExplanationSuite(None, [("static", {})])
    suite._registry = _StaticRegistry({"feature_attributions": {"x": 1.0}})
    with pytest.raises(TypeError, match="requires an Explanation"):
        suite.run(np.array([1.0]))


def test_run_replaces_previous_results_atomically():
    suite = ExplanationSuite(None, [("static", {})])
    result = _valid_explanation(2.0)
    suite.explanations = {"stale": _valid_explanation()}
    suite._registry = _StaticRegistry(result)

    assert suite.run(np.array([1.0])) == {"static": result}
    assert "stale" not in suite.explanations


def test_instance_runner_requires_and_forwards_method_specific_call_arguments():
    reference = np.ones((3, 2))
    registry = _MetadataRegistry()
    suite = ExplanationSuite(None, [("reference", {})])
    suite._registry = registry

    assert suite.suggest_compatible() == []
    assert suite.suggest_compatible(for_instance_run=False) == ["reference"]
    with pytest.raises(ValueError, match="explainer_call_kwargs"):
        suite.run(np.array([1.0, 2.0]))

    configured = ExplanationSuite(
        None,
        [("reference", {})],
        explainer_call_kwargs={"reference": {"X_reference": reference}},
    )
    configured._registry = registry
    assert configured.suggest_compatible() == ["reference"]
    result = configured.run(np.array([1.0, 2.0]))
    assert result["reference"].get_attributions() == {"x": 3.0}

    override = np.ones((5, 2))
    result = configured.run(
        np.array([1.0, 2.0]),
        call_kwargs_by_explainer={"reference": {"X_reference": override}},
    )
    assert result["reference"].get_attributions() == {"x": 5.0}


def test_instance_runner_rejects_global_scope_and_listing_excludes_it():
    suite = ExplanationSuite(None, [("global", {})])
    suite._registry = _MetadataRegistry()

    assert suite.suggest_compatible() == []
    assert suite.suggest_compatible(for_instance_run=False) == ["global"]
    with pytest.raises(ValueError, match="scope='global'"):
        suite.run(np.array([1.0]))


def test_instance_listing_and_run_require_complete_constructor_configuration():
    registry = _MetadataRegistry()
    suite = ExplanationSuite(None, [("required_constructor", {})])
    suite._registry = registry

    assert suite.suggest_compatible() == []
    assert suite.suggest_compatible(for_instance_run=False) == ["required_constructor"]
    with pytest.raises(ValueError, match="required constructor arguments"):
        suite.run(np.array([1.0]))

    configured = ExplanationSuite(
        None,
        [("required_constructor", {"required_token": "configured"})],
    )
    configured._registry = registry
    assert configured.suggest_compatible() == ["required_constructor"]
    assert set(configured.run(np.array([1.0]))) == {"required_constructor"}


@pytest.mark.parametrize(
    "call_kwargs,error_type",
    [
        ({"unknown": {}}, ValueError),
        ({"reference": None}, TypeError),
        ({"reference": {"instance": np.array([0.0])}}, ValueError),
    ],
)
def test_run_validates_per_call_overrides(call_kwargs, error_type):
    suite = ExplanationSuite(None, [("reference", {})])
    suite._registry = _MetadataRegistry()
    with pytest.raises(error_type):
        suite.run(np.array([1.0]), call_kwargs_by_explainer=call_kwargs)


def test_compatibility_listing_excludes_quarantined_and_unverified_by_default():
    background_data = np.zeros((2, 1))
    training_data_args = {
        "training_data": background_data,
        "feature_names": ["x"],
        "class_names": ["zero", "one"],
    }
    anchor_tabular_args = {
        "background_data": background_data,
        "feature_names": ["x"],
        "class_names": ["zero", "one"],
    }
    suite = ExplanationSuite(
        None,
        [
            ("anchors", training_data_args),
            ("anchor_tabular", anchor_tabular_args),
            ("lime", training_data_args),
            ("gradcam", {}),
        ],
        data_meta={"data_type": "tabular", "task": "classification"},
    )

    assert suite.suggest_compatible() == ["anchor_tabular", "lime"]
    assert suite.suggest_compatible(include_statuses=("verified", "quarantined")) == [
        "anchors",
        "anchor_tabular",
        "lime",
    ]


def test_canonical_anchor_suite_requires_empirical_background_configuration():
    suite = ExplanationSuite(
        None,
        [
            (
                "anchor_tabular",
                {"feature_names": ["x"], "class_names": ["zero", "one"]},
            )
        ],
        data_meta={"data_type": "tabular", "task": "classification"},
    )

    assert suite.suggest_compatible() == []
    assert suite.suggest_compatible(for_instance_run=False) == ["anchor_tabular"]
    with pytest.raises(ValueError, match="required constructor arguments"):
        suite.run(np.array([1.0]))


def test_canonical_anchor_suite_constructs_with_background_data_and_runs_locally():
    class ConstantClassifier:
        task = "classification"

        @staticmethod
        def predict(X):
            rows = np.asarray(X).shape[0]
            return np.tile(np.array([[0.1, 0.9]]), (rows, 1))

    suite = ExplanationSuite(
        ConstantClassifier(),
        [
            (
                "anchor_tabular",
                {
                    "background_data": np.arange(4.0).reshape(-1, 1),
                    "feature_names": ["x"],
                    "class_names": ["zero", "one"],
                    "threshold": 0.5,
                    "delta": 0.2,
                    "max_samples": 200,
                    "random_state": 7,
                },
            )
        ],
        data_meta={"data_type": "tabular", "task": "classification"},
    )

    explanation = suite.run(np.array([1.5]))["anchor_tabular"]
    assert explanation.explainer_name == "AnchorTabular"
    assert explanation.target_class == "one"
    assert explanation.explanation_data["is_certified_anchor"] is True
    assert explanation.explanation_data["budget_exhausted"] is False


def test_historical_best_name_warns_and_returns_only_first_verified_compatible():
    lime_args = {
        "training_data": np.zeros((2, 1)),
        "feature_names": ["x"],
        "class_names": ["zero", "one"],
    }
    suite = ExplanationSuite(
        None,
        [("anchors", {}), ("lime", lime_args), ("shap", {})],
        data_meta={"data_type": "tabular", "task": "classification"},
    )
    with pytest.warns(FutureWarning, match="cannot establish explainer quality"):
        assert suite.suggest_best() == "lime"


def test_historical_best_name_refuses_unverified_or_quarantined_only_configs():
    suite = ExplanationSuite(None, [("anchors", {}), ("eigengradcam", {})])
    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError, match="No configured explainer"):
            suite.suggest_best()


@pytest.mark.parametrize("max_results", [-1, True, 1.5])
def test_compatibility_listing_validates_limit(max_results):
    suite = ExplanationSuite(None, [("lime", {})])
    with pytest.raises((TypeError, ValueError), match="max_results"):
        suite.suggest_compatible(max_results=max_results)


def test_compare_rejects_missing_or_nonfinite_attribution_mapping(capsys):
    suite = ExplanationSuite(None, [("static", {})])
    suite.explanations = {"static": Explanation("bad", "target", {})}
    with pytest.raises(ValueError, match="feature_attributions mapping"):
        suite.compare()

    suite.explanations = {
        "static": Explanation("bad", "target", {"feature_attributions": {0: 1.0}})
    }
    with pytest.raises(TypeError, match="string feature-attribution keys"):
        suite.compare()

    suite.explanations = {"static": _valid_explanation(np.nan)}
    with pytest.raises(ValueError, match="must be finite"):
        suite.compare()

    for invalid_value in (True, None, "1", np.array([1.0])):
        suite.explanations = {"static": _valid_explanation(invalid_value)}
        with pytest.raises(TypeError, match="real numeric scalar"):
            suite.compare()

    suite.explanations = {"static": Explanation("empty", "target", {"feature_attributions": {}})}
    with pytest.raises(ValueError, match="at least one feature attribution"):
        suite.compare()
    capsys.readouterr()


def test_compare_requires_matching_explicit_contracts_and_targets(capsys):
    suite = ExplanationSuite(None, [("static", {})])
    suite.explanations = {
        "first": _valid_explanation(1.0),
        "second": _valid_explanation(2.0),
    }
    with pytest.raises(ValueError, match="comparison_contract"):
        suite.compare()
    with pytest.warns(RuntimeWarning, match="descriptively only"):
        suite.compare(allow_incommensurate=True)

    contract = "signed-feature-attribution:model-output:class-1"
    suite.explanations = {
        "first": _valid_explanation(1.0, comparison_contract=contract),
        "second": _valid_explanation(2.0, comparison_contract=contract),
    }
    suite.compare()

    suite.explanations["second"] = _valid_explanation(
        2.0,
        target="other",
        comparison_contract=contract,
    )
    with pytest.raises(ValueError, match="different explained targets"):
        suite.compare()
    capsys.readouterr()


def test_compare_rejects_disjoint_feature_spaces_even_with_matching_contract(capsys):
    contract = "caller-asserted:signed-feature-attribution:model-output"
    suite = ExplanationSuite(None, [("static", {})])
    suite.explanations = {
        "first": _named_explanation(
            "age", 1.0, comparison_contract=contract, feature_names=["age"]
        ),
        "second": _named_explanation(
            "income", 2.0, comparison_contract=contract, feature_names=["income"]
        ),
    }

    with pytest.raises(ValueError, match="different ordered feature identities"):
        suite.compare()
    with pytest.warns(RuntimeWarning, match="descriptively only"):
        suite.compare(allow_incommensurate=True)
    capsys.readouterr()


def test_compare_rejects_feature_names_that_do_not_match_attribution_keys():
    suite = ExplanationSuite(None, [("static", {})])
    suite.explanations = {
        "static": _named_explanation(
            "age",
            1.0,
            comparison_contract="caller-asserted:test",
            feature_names=["income"],
        )
    }

    with pytest.raises(ValueError, match="do not exactly match"):
        suite.compare()


def test_compare_requires_the_same_feature_order():
    contract = "caller-asserted:test"
    suite = ExplanationSuite(None, [("static", {})])
    suite.explanations = {
        "first": Explanation(
            "static",
            "target",
            {"feature_attributions": {"age": 1.0, "income": 2.0}},
            feature_names=["age", "income"],
            metadata={"comparison_contract": contract},
        ),
        "second": Explanation(
            "static",
            "target",
            {"feature_attributions": {"income": 2.0, "age": 1.0}},
            feature_names=["income", "age"],
            metadata={"comparison_contract": contract},
        ),
    }

    with pytest.raises(ValueError, match="different ordered feature identities"):
        suite.compare()
