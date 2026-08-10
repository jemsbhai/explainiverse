"""Capability-claim and validation tests for the explainer registry."""

import pytest

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.core.registry import ExplainerMeta, ExplainerRegistry, get_default_registry
from explainiverse.engine.suite import ExplanationSuite


class _MinimalExplainer(BaseExplainer):
    def explain(self, instance, **kwargs):
        del instance, kwargs
        return Explanation("minimal", "target", {})


class _BatchExplainer(_MinimalExplainer):
    def explain_batch(self, instances, **kwargs):
        return [self.explain(instance, **kwargs) for instance in instances]


@pytest.mark.parametrize("scope", ["", "LOCAL", "other"])
def test_metadata_rejects_unknown_scope(scope):
    with pytest.raises(ValueError, match="scope"):
        ExplainerMeta(scope=scope)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_types", "any"),
        ("model_types", ["unknown"]),
        ("data_types", []),
        ("data_types", ["tabular", "tabular"]),
        ("task_types", ["classification", 3]),
    ],
)
def test_metadata_rejects_malformed_capability_taxonomy(field, value):
    kwargs = {field: value}
    with pytest.raises((TypeError, ValueError), match=field):
        ExplainerMeta(scope="local", **kwargs)


@pytest.mark.parametrize("claim_status", ["complete", "canonical", ""])
def test_metadata_rejects_unknown_claim_status(claim_status):
    with pytest.raises(ValueError, match="claim_status"):
        ExplainerMeta(scope="local", claim_status=claim_status)


@pytest.mark.parametrize("claim_status", ["verified", "quarantined"])
def test_audited_status_requires_an_explicit_claim_scope(claim_status):
    with pytest.raises(ValueError, match="explicit audited claim_scope"):
        ExplainerMeta(scope="local", claim_status=claim_status)


@pytest.mark.parametrize("claim_status", ["verified", "quarantined"])
def test_audited_status_rejects_whitespace_padded_default_claim_scope(claim_status):
    with pytest.raises(ValueError, match="explicit audited claim_scope"):
        ExplainerMeta(
            scope="local",
            claim_status=claim_status,
            claim_scope="  Implementation has not completed an accuracy audit.\n",
        )


@pytest.mark.parametrize(
    ("criteria", "error_type"),
    [
        ({"scope": "LOCAL"}, ValueError),
        ({"scope": 1}, TypeError),
        ({"model_type": "neurla"}, ValueError),
        ({"model_type": ["neural"]}, TypeError),
        ({"data_type": "audio"}, ValueError),
        ({"data_type": True}, TypeError),
        ({"task_type": "ranking"}, ValueError),
        ({"task_type": {"classification"}}, TypeError),
    ],
)
def test_matching_and_filtering_reject_invalid_taxonomy_queries(criteria, error_type):
    meta = ExplainerMeta(scope="local")
    with pytest.raises(error_type):
        meta.matches(**criteria)

    # Validation must not depend on the registry containing an entry.
    with pytest.raises(error_type):
        ExplainerRegistry().filter(**criteria)


def test_suite_surfaces_invalid_data_meta_taxonomy():
    suite = ExplanationSuite(
        model=None,
        # The typo must surface even though this quarantined entry would be
        # excluded by the default status filter before metadata matching.
        explainer_configs=[("anchors", {})],
        data_meta={"model_type": "neurla"},
    )

    with pytest.raises(ValueError, match="model_type"):
        suite.suggest_compatible(for_instance_run=False)


def test_verified_complexity_requires_an_actual_note():
    with pytest.raises(ValueError, match="complexity"):
        ExplainerMeta(scope="local", complexity_verified=True)


@pytest.mark.parametrize("bad_class", [object, object(), lambda: None])
def test_registry_rejects_non_explainer_classes(bad_class):
    registry = ExplainerRegistry()
    with pytest.raises(TypeError, match="inherit from BaseExplainer"):
        registry.register("bad", bad_class, ExplainerMeta(scope="local"))


def test_registry_rejects_false_batch_capability_claim():
    registry = ExplainerRegistry()
    with pytest.raises(ValueError, match="explain_batch"):
        registry.register(
            "bad_batch",
            _MinimalExplainer,
            ExplainerMeta(scope="local", supports_batching=True),
        )


def test_registry_accepts_real_batch_capability():
    registry = ExplainerRegistry()
    registry.register(
        "batch",
        _BatchExplainer,
        ExplainerMeta(scope="local", supports_batching=True),
    )
    assert registry.get_meta("batch").supports_batching is True


@pytest.mark.parametrize(
    ("field", "mutated_value", "error_type"),
    [
        ("scope", "dataset", ValueError),
        ("model_types", ["any", "foundation"], ValueError),
        ("claim_status", "complete", ValueError),
        ("supports_batching", "yes", TypeError),
    ],
)
def test_registration_revalidates_mutated_metadata(field, mutated_value, error_type):
    registry = ExplainerRegistry()
    meta = ExplainerMeta(scope="local")
    setattr(meta, field, mutated_value)

    with pytest.raises(error_type):
        registry.register("mutated", _MinimalExplainer, meta)
    assert registry.list_explainers() == []


def test_registry_metadata_is_an_isolated_validated_snapshot():
    registry = ExplainerRegistry()
    supplied = ExplainerMeta(
        scope="local",
        model_types=["any"],
        claim_status="quarantined",
        claim_scope="Compatibility implementation only.",
    )
    registry.register("isolated", _MinimalExplainer, supplied)

    supplied.claim_status = "verified"
    supplied.model_types.append("tree")

    stored = registry.get_meta("isolated")
    assert stored.claim_status == "quarantined"
    assert stored.model_types == ["any"]


def test_registry_reads_do_not_expose_mutable_internal_metadata():
    registry = ExplainerRegistry()
    registry.register(
        "isolated",
        _MinimalExplainer,
        ExplainerMeta(
            scope="local",
            claim_status="quarantined",
            claim_scope="Compatibility implementation only.",
        ),
    )

    registry.get("isolated")["meta"].claim_status = "verified"
    registry.get_meta("isolated").model_types.append("tree")
    registry.list_explainers(with_meta=True)["isolated"]["meta"].data_types.append("image")

    stored = registry.get_meta("isolated")
    assert stored.claim_status == "quarantined"
    assert stored.model_types == ["any"]
    assert stored.data_types == ["tabular"]


@pytest.mark.parametrize("name", ["", "   ", 3])
def test_registry_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="name"):
        ExplainerRegistry().register(name, _MinimalExplainer, ExplainerMeta(scope="local"))


@pytest.mark.parametrize("max_results", [-1, True, 2.5])
def test_recommend_validates_result_limit(max_results):
    registry = ExplainerRegistry()
    with pytest.raises((TypeError, ValueError), match="max_results"):
        registry.recommend(max_results=max_results)


def test_recommend_is_metadata_only_and_does_not_reward_citations():
    registry = ExplainerRegistry()
    registry.register(
        "first",
        _MinimalExplainer,
        ExplainerMeta(scope="local"),
    )
    registry.register(
        "documented_later",
        _MinimalExplainer,
        ExplainerMeta(
            scope="local",
            description="long description",
            paper_reference="A citation is not implementation evidence",
        ),
    )

    assert registry.recommend(scope_preference="local") == [
        "first",
        "documented_later",
    ]


def test_registry_summary_surfaces_claim_status_and_scope():
    registry = ExplainerRegistry()
    registry.register(
        "pending",
        _MinimalExplainer,
        ExplainerMeta(scope="local", claim_scope="Formula audit pending."),
    )

    summary = registry.summary()
    assert "pending [unverified]" in summary
    assert "claim scope: Formula audit pending." in summary


def test_default_registry_batch_flags_match_public_methods():
    registry = get_default_registry()
    for name, entry in registry.list_explainers(with_meta=True).items():
        has_batch_method = callable(getattr(entry["class"], "explain_batch", None))
        assert entry["meta"].supports_batching is has_batch_method, name


def test_default_registry_does_not_overclaim_quarantined_algorithms():
    registry = get_default_registry()

    lime = registry.get_meta("lime")
    assert lime.data_types == ["tabular"]
    assert lime.claim_status == "verified"

    anchors = registry.get_meta("anchors")
    assert anchors.claim_status == "quarantined"
    assert "no KL-LUCB" in anchors.claim_scope

    counterfactual = registry.get_meta("counterfactual")
    assert counterfactual.claim_status == "quarantined"
    assert "not DiCE" in counterfactual.claim_scope


def test_canonical_and_compatibility_anchor_entries_coexist_with_distinct_claims():
    registry = get_default_registry()

    assert {"anchor_tabular", "anchors"} <= set(registry.list_explainers())
    canonical_entry = registry.get("anchor_tabular")
    compatibility_entry = registry.get("anchors")
    canonical = canonical_entry["meta"]
    compatibility = compatibility_entry["meta"]

    assert canonical_entry["class"].__name__ == "AnchorTabularExplainer"
    assert compatibility_entry["class"].__name__ == "AnchorsExplainer"
    assert canonical_entry["class"] is not compatibility_entry["class"]
    assert canonical.scope == "local"
    assert canonical.model_types == ["any"]
    assert canonical.data_types == ["tabular"]
    assert canonical.task_types == ["classification"]
    assert canonical.requires_training_data is True
    assert canonical.supports_batching is False
    assert canonical.claim_status == "verified"
    assert compatibility.scope == "local"
    assert compatibility.data_types == ["tabular"]
    assert compatibility.task_types == ["classification"]
    assert compatibility.claim_status == "quarantined"

    canonical_scope = canonical.claim_scope
    assert "Algorithms 1-2-style KL-LUCB" in canonical_scope
    assert "finite continuous numeric tabular inputs" in canonical_scope
    assert "query-consistent one-sided quartile/decile threshold predicates" in canonical_scope
    assert "uniform empirical joint distribution" in canonical_scope
    assert "conditional draws restrict to whole satisfying rows" in canonical_scope
    assert "preserve observed feature dependence" in canonical_scope
    assert "fixed deterministic model predictions" in canonical_scope
    assert "fixes the model's predicted output column" in canonical_scope
    assert "class_names are display-only" in canonical_scope
    assert "strict lower_bound > threshold" in canonical_scope
    assert "budget exhaustion is returned explicitly as uncertified" in canonical_scope
    assert "does not guarantee a globally maximum-coverage anchor" in canonical_scope
    assert "causal sufficiency" in canonical_scope
    assert "no KL-LUCB" in compatibility.claim_scope


def test_default_registry_gradient_claims_match_actual_data_and_aggregate_contracts():
    registry = get_default_registry()

    deeplift = registry.get_meta("deeplift")
    assert deeplift.data_types == ["tabular"]
    assert "one-dimensional flat feature vectors" in deeplift.claim_scope

    deepshap = registry.get_meta("deepshap")
    assert deepshap.data_types == ["tabular"]
    assert "approximate" in deepshap.description.lower()
    assert "SHAP-value language is approximate" in deepshap.claim_scope

    saliency = registry.get_meta("saliency")
    assert saliency.data_types == ["tabular"]
    assert "exactly one declared feature name" in saliency.claim_scope

    tcav = registry.get_meta("tcav")
    assert tcav.scope == "global"
    assert "Global fraction" in tcav.claim_scope
    assert "require_logit_scores=True" in tcav.claim_scope

    integrated_gradients = registry.get_meta("integrated_gradients")
    assert integrated_gradients.data_types == ["tabular", "image"]
    assert "exact configured or inferred input shape" in integrated_gradients.claim_scope

    lrp = registry.get_meta("lrp")
    assert "class 0 is supported only" in lrp.claim_scope
    assert "asymmetric rules and composites fail" in lrp.claim_scope


def test_default_registry_tabular_claims_capture_remediated_boundaries():
    registry = get_default_registry()

    assert "mode follows explicit model-task semantics" in registry.get_meta("lime").claim_scope
    assert "class_names are display-only" in registry.get_meta("shap").claim_scope
    assert "interaction values are supported only" in registry.get_meta("treeshap").claim_scope
    assert "mapped through model.classes_" in registry.get_meta("sage").claim_scope
    assert "maximum-gradient greedy support selection" in registry.get_meta("protodash").claim_scope
    assert "empty result is a search failure" in registry.get_meta("counterfactual").claim_scope


@pytest.mark.parametrize(
    ("name", "claim_status", "claim_scope"),
    [
        (
            "gradcam",
            "verified",
            "Canonical Grad-CAM equations 1-2 for one finite image, one 4-D "
            "spatial target-layer output, one fixed scalar output target, and an "
            "explicit CHW/HWC layout whenever automatic layout inference is ambiguous.",
        ),
        (
            "hirescam",
            "verified",
            "Canonical elementwise activation-gradient formula for one 4-D spatial "
            "layer and fixed scalar target. No architecture-wide theorem is asserted; "
            "the paper's guarantee is limited to a CNN ending in one fully connected layer.",
        ),
        (
            "xgradcam",
            "verified",
            "Canonical equations 7-8 for one 4-D spatial layer and fixed scalar "
            "target; undefined nonzero channels with zero activation sum are rejected. "
            "No unconditional sensitivity or conservation guarantee is asserted.",
        ),
        (
            "layercam",
            "verified",
            "Canonical positive-gradient-times-activation formula for one compatible "
            "4-D spatial layer and one fixed scalar target.",
        ),
        (
            "eigencam",
            "verified",
            "Canonical uncentered SVD projection of one 4-D spatial activation tensor; "
            "the method is class-agnostic, rejects explicit targets, and discloses its "
            "deterministic SVD sign convention.",
        ),
        (
            "scorecam",
            "verified",
            "Verified transcription of paper Algorithm 1: normalized activation masks, "
            "raw model-output target scores, and softmax across channels. This differs "
            "from section 3.2 and the authors' released post-softmax probability-weighting "
            "code. Paper logit-space match is asserted only when the adapter declares "
            "raw_model_output_space='logit'; one-logit binary target expansion is unsupported.",
        ),
        (
            "eigengradcam",
            "quarantined",
            "Verified only against the pytorch-grad-cam centered-SVD library operation; "
            "it is not attributed to the Eigen-CAM paper or another primary method paper.",
        ),
        (
            "gradcam_elementwise",
            "quarantined",
            "Verified only against the pytorch-grad-cam per-element rectification "
            "operation; it is not attributed to the Grad-CAM paper or another primary "
            "method paper.",
        ),
        (
            "ablationcam",
            "verified",
            "Canonical target-layer channel zeroing with raw output scores for spatial "
            "PyTorch models whose target module runs once per forward, whose raw outputs "
            "map one-to-one to target indices, and whose original target score is nonzero; "
            "one-logit binary adapters are unsupported.",
        ),
    ],
)
def test_default_registry_cam_claims_match_audited_boundaries(name, claim_status, claim_scope):
    meta = get_default_registry().get_meta(name)
    assert meta.claim_status == claim_status
    assert meta.claim_scope == (
        f"{claim_scope} Implementation verification is CPU-only; "
        "CUDA is outside the audited device scope."
    )

    neutral_text = f"{meta.description} {meta.paper_reference}".lower()
    for unsupported_claim in ("verification pending", "faithful", "axiom-based", "accurate"):
        assert unsupported_claim not in neutral_text


def test_default_registry_quarantines_library_cams_and_omits_gradcamplusplus():
    registry = get_default_registry()
    for name in ("eigengradcam", "gradcam_elementwise"):
        meta = registry.get_meta(name)
        assert "library variant" in meta.description
        assert "pytorch-grad-cam" in meta.paper_reference

    registered = registry.list_explainers()
    assert "gradcam++" not in registered
    assert "gradcamplusplus" not in registered


def test_default_registry_neural_claims_are_explicitly_cpu_verified():
    registry = get_default_registry()
    for name in registry.list_explainers():
        meta = registry.get_meta(name)
        if "neural" in meta.model_types:
            assert "Implementation verification is CPU-only" in meta.claim_scope
            assert "CUDA is outside the audited device scope" in meta.claim_scope


def test_metadata_defaults_are_isolated_between_instances():
    first = ExplainerMeta(scope="local")
    second = ExplainerMeta(scope="local")
    first.model_types.append("tree")
    assert second.model_types == ["any"]
