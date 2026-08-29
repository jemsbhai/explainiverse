"""Exact integrity contract for repository-wide expected test skips."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

POLICY_PATH = Path(__file__).with_name("conftest.py")
POLICY_SPEC = importlib.util.spec_from_file_location(
    "explainiverse_repository_skip_policy",
    POLICY_PATH,
)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
skip_policy = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(skip_policy)

EXPECTED_SKIP_REASONS: dict[str, str | tuple[str, ...]] = {
    "tests/test_bugfixes_v093.py::TestBug1DeviceMismatch::test_get_model_device_cuda": (
        "CUDA not available"
    ),
    "tests/test_bugfixes_v093.py::TestBug1DeviceMismatch::test_prepare_input_tensor_device_cuda": (
        "CUDA not available"
    ),
    "tests/test_bugfixes_v093.py::TestBug1DeviceMismatch::test_lrp_explain_cuda_model": (
        "CUDA not available"
    ),
    "tests/test_bugfixes_v093.py::TestBug1DeviceMismatch::test_lrp_cnn_explain_cuda": (
        "CUDA not available"
    ),
    "tests/test_treeshap_accuracy.py::test_xgboost_31_vector_intercept_is_preserved_and_additive": (
        "vector-valued base_score was introduced in 3.1"
    ),
    "tests/test_lambda_github_controller.py::test_provider_intent_late_destination_race_preserves_source_and_never_falls_back": (
        "Windows no-replace publication race"
    ),
    "tests/test_lambda_github_controller.py::test_journal_reserve_handle_denies_second_writer_for_full_lifetime_on_windows": (
        "Windows share-mode contract"
    ),
    "tests/test_lambda_github_controller.py::test_windows_atomic_publication_denies_source_write_delete_and_rebind": (
        "Windows held-handle publication contract"
    ),
    "tests/test_lambda_github_controller.py::test_windows_recovery_sidecar_and_journal_publication_hold_source_identity": (
        "Windows held-handle recovery publication contract"
    ),
    "tests/test_lambda_github_controller.py::test_windows_atomic_open_failure_closes_full_width_native_handle": (
        "Windows native handle-width contract"
    ),
    "tests/test_lambda_operator_cli.py::test_preloader_holds_verified_trees_without_write_or_delete_sharing": (
        "Windows held-tree semantics are native"
    ),
    "tests/test_lambda_operator_cli.py::test_operator_toolchain_is_exact_path_byte_version_owner_and_signer_pinned": (
        "requires the exact reviewed Windows release host toolchain"
    ),
    "tests/test_lambda_operator_cli.py::test_preloader_validates_owner_private_receipt_root_before_third_party_import": (
        "Windows owner-private ACL is native"
    ),
    "tests/test_lambda_operator_cli.py::test_windows_launcher_rejects_unguarded_parent_before_reading_secret": (
        "native launcher is Windows-only"
    ),
    "tests/test_lambda_operator_cli.py::test_windows_launcher_delivers_secret_and_post_plan_confirmation_without_exposure": (
        "native inherited HANDLE contract is Windows-only",
        "requires a freshly prepared pinned-runtime clean-source fixture",
    ),
}


def _recorded_unexpected_skip(nodeid: str, reason: str) -> list[tuple[str, str]]:
    skip_policy._UNEXPECTED_SKIPS.clear()
    report = SimpleNamespace(
        skipped=True,
        nodeid=nodeid,
        longrepr=f"Skipped: {reason}",
    )
    skip_policy.pytest_runtest_logreport(report)
    return list(skip_policy._UNEXPECTED_SKIPS)


def test_expected_skip_inventory_and_reason_sets_are_exact() -> None:
    assert skip_policy._EXPECTED_SKIPS == EXPECTED_SKIP_REASONS


def test_every_exact_expected_skip_reason_is_accepted() -> None:
    for nodeid, expected in EXPECTED_SKIP_REASONS.items():
        reasons = (expected,) if isinstance(expected, str) else expected
        for reason in reasons:
            assert _recorded_unexpected_skip(nodeid, reason) == []


def test_unlisted_node_and_near_miss_reason_are_rejected() -> None:
    nodeid = (
        "tests/test_lambda_operator_cli.py::"
        "test_windows_launcher_delivers_secret_and_post_plan_confirmation_without_exposure"
    )
    assert _recorded_unexpected_skip(nodeid, "requires some clean-source fixture")
    assert _recorded_unexpected_skip(
        "tests/test_repository_skip_policy.py::test_unknown_skip",
        "native inherited HANDLE contract is Windows-only",
    )
