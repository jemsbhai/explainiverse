"""Accuracy tests for the deliberately minimal explainer and adapter ABCs."""

import pytest

from explainiverse.adapters.base_adapter import BaseModelAdapter
from explainiverse.core.explainer import BaseExplainer


class _EchoExplainer(BaseExplainer):
    def explain(self, *args, **kwargs):
        return args, kwargs


class _GlobalExplainer(BaseExplainer):
    """A valid explainer with no call-time instance or target argument."""

    def explain(self):
        return "global explanation"


class _DelegatingExplainer(BaseExplainer):
    def explain(self, *args, **kwargs):
        return super().explain(*args, **kwargs)


class _EchoAdapter(BaseModelAdapter):
    def predict(self, data):
        return data


class _DelegatingAdapter(BaseModelAdapter):
    def predict(self, data):
        return super().predict(data)


def test_explainer_base_and_missing_override_are_abstract():
    with pytest.raises(TypeError, match="abstract"):
        BaseExplainer(object())

    class MissingExplain(BaseExplainer):
        pass

    with pytest.raises(TypeError, match="abstract"):
        MissingExplain(object())


def test_explainer_retains_model_identity_without_mutating_it():
    model = {"state": [1, 2]}
    before = {"state": [1, 2]}

    explainer = _EchoExplainer(model)

    assert explainer.model is model
    assert model == before


def test_explainer_allows_model_independent_implementations():
    explainer = _GlobalExplainer(None)

    assert explainer.model is None
    assert explainer.explain() == "global explanation"


def test_explainer_base_does_not_imply_batching_or_target_support():
    assert not hasattr(BaseExplainer, "explain_batch")
    with pytest.raises(TypeError):
        _GlobalExplainer(None).explain(object())
    with pytest.raises(TypeError):
        _GlobalExplainer(None).explain(target_class=0)


def test_explainer_abstract_fallback_fails_explicitly():
    with pytest.raises(NotImplementedError, match="DelegatingExplainer.*explain"):
        _DelegatingExplainer(object()).explain("input")


def test_adapter_base_and_missing_override_are_abstract():
    with pytest.raises(TypeError, match="abstract"):
        BaseModelAdapter(object())

    class MissingPredict(BaseModelAdapter):
        pass

    with pytest.raises(TypeError, match="abstract"):
        MissingPredict(object())


def test_adapter_requires_a_model_but_retains_the_exact_reference():
    with pytest.raises(ValueError, match="model must not be None"):
        _EchoAdapter(None)

    model = {"weights": [1.0]}
    adapter = _EchoAdapter(model)

    assert adapter.model is model
    assert model == {"weights": [1.0]}


def test_adapter_copies_feature_names_without_mutating_the_caller():
    source_names = ["age", "income"]
    adapter = _EchoAdapter(object(), feature_names=source_names)

    assert adapter.feature_names == source_names
    assert adapter.feature_names is not source_names

    source_names.append("postcode")
    assert adapter.feature_names == ["age", "income"]

    adapter.feature_names.append("balance")
    assert source_names == ["age", "income", "postcode"]


def test_adapter_materializes_a_feature_name_iterable_once():
    names = (name for name in ["left", "right"])

    adapter = _EchoAdapter(object(), feature_names=names)

    assert adapter.feature_names == ["left", "right"]
    assert list(names) == []


@pytest.mark.parametrize(
    ("feature_names", "error_type", "message"),
    [
        ("feature", TypeError, "iterable of strings"),
        (3, TypeError, "iterable of strings"),
        (["feature", 2], TypeError, "only strings"),
        (["feature", ""], ValueError, "non-empty"),
        (["feature", "feature"], ValueError, "unique"),
    ],
)
def test_adapter_rejects_ambiguous_feature_name_metadata(feature_names, error_type, message):
    with pytest.raises(error_type, match=message):
        _EchoAdapter(object(), feature_names=feature_names)


def test_adapter_base_does_not_coerce_inputs_or_imply_targets():
    adapter = _EchoAdapter(object())
    data = [[1.0, 2.0]]

    assert adapter.predict(data) is data
    with pytest.raises(TypeError):
        adapter.predict(data, target=0)


def test_adapter_abstract_fallback_fails_explicitly():
    with pytest.raises(NotImplementedError, match="DelegatingAdapter.*predict"):
        _DelegatingAdapter(object()).predict([[1.0]])
