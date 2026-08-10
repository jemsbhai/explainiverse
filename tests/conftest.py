"""Repository-wide test-integrity hooks."""

from __future__ import annotations

from typing import Any

import pytest

# Skips are part of the tested contract, not an unbounded escape hatch. CUDA
# behavior is exercised only on GPU runners. The XGBoost case covers a feature
# introduced in 3.1 while Python 3.10 intentionally supports XGBoost <3.1.
_EXPECTED_SKIPS = {
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
}

_UNEXPECTED_SKIPS: list[tuple[str, str]] = []


def pytest_runtest_logreport(report: Any) -> None:
    if not report.skipped:
        return
    reason = str(report.longrepr)
    expected_reason = _EXPECTED_SKIPS.get(report.nodeid)
    if expected_reason is None or expected_reason not in reason:
        _UNEXPECTED_SKIPS.append((report.nodeid, reason))


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
