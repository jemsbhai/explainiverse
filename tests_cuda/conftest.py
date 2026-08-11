"""CUDA release tests fail the session for every skipped test or collection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

_UNEXPECTED_SKIPS: list[tuple[str, str]] = []
_MANIFEST = Path(__file__).with_name("expected-nodeids.txt")


def _load_expected_nodeids(path: Path = _MANIFEST) -> list[str]:
    nodeids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not nodeids or len(nodeids) != len(set(nodeids)):
        raise ValueError("CUDA expected-node manifest must be non-empty and duplicate-free")
    return nodeids


def _validate_expected_nodeids(actual: list[str], expected: list[str]) -> None:
    if actual == expected:
        return
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    order_mismatch = not missing and not extra and actual != expected
    raise ValueError(
        "CUDA release collection differs from the reviewed manifest: "
        f"expected={len(expected)}, actual={len(actual)}, missing={missing!r}, "
        f"extra={extra!r}, order_mismatch={order_mismatch}"
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    del session
    _UNEXPECTED_SKIPS.clear()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("EXPLAINIVERSE_ENFORCE_CUDA_MANIFEST") != "1":
        return
    expected = _load_expected_nodeids()
    actual = [item.nodeid.replace("\\", "/") for item in items]
    try:
        _validate_expected_nodeids(actual, expected)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"CUDA release manifest: exact {len(actual)} test nodes collected")


def pytest_runtest_logreport(report: Any) -> None:
    if report.skipped:
        _UNEXPECTED_SKIPS.append((report.nodeid, str(report.longrepr)))


def pytest_collectreport(report: Any) -> None:
    if report.skipped:
        _UNEXPECTED_SKIPS.append((report.nodeid, str(report.longrepr)))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    if not _UNEXPECTED_SKIPS:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "unexpected CUDA skips")
        for nodeid, reason in _UNEXPECTED_SKIPS:
            reporter.write_line(f"{nodeid}: {reason}")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
