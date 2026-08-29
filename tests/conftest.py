"""Repository-wide test-integrity hooks."""

from __future__ import annotations

from typing import Any

import pytest

# Skips are part of the tested contract, not an unbounded escape hatch. CUDA
# behavior is exercised only on GPU runners. The XGBoost case covers a feature
# introduced in 3.1 while Python 3.10 intentionally supports XGBoost <3.1.
# Native release-control tests remain required on their declared host OS; the
# exact node/reason bindings below make those platform skips reviewable.
_EXPECTED_SKIPS: dict[str, str | tuple[str, ...]] = {
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

_UNEXPECTED_SKIPS: list[tuple[str, str]] = []


def _reported_skip_reason(longrepr: Any) -> str | None:
    """Extract only pytest's observed runtime-skip tuple representation."""
    if (
        type(longrepr) is not tuple
        or len(longrepr) != 3
        or type(longrepr[0]) is not str
        or not longrepr[0]
        or type(longrepr[1]) is not int
        or longrepr[1] <= 0
        or type(longrepr[2]) is not str
        or not longrepr[2].startswith("Skipped: ")
    ):
        return None
    reason = longrepr[2].removeprefix("Skipped: ")
    return reason or None


def pytest_runtest_logreport(report: Any) -> None:
    if not report.skipped:
        return
    reported_reason = _reported_skip_reason(report.longrepr)
    expected = _EXPECTED_SKIPS.get(report.nodeid)
    expected_reasons = (expected,) if isinstance(expected, str) else expected
    if expected_reasons is None or reported_reason not in expected_reasons:
        _UNEXPECTED_SKIPS.append((report.nodeid, str(report.longrepr)))


def pytest_collectreport(report: Any) -> None:
    """Treat collection-time import skips as missing test coverage."""
    if report.skipped:
        _UNEXPECTED_SKIPS.append((report.nodeid, str(report.longrepr)))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    if not _UNEXPECTED_SKIPS:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "unexpected skips")
        for nodeid, reason in _UNEXPECTED_SKIPS:
            reporter.write_line(f"{nodeid}: {reason}")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
