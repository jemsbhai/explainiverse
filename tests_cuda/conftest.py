"""CUDA release tests fail the session for every skipped test or collection."""

from __future__ import annotations

from typing import Any

import pytest

_UNEXPECTED_SKIPS: list[tuple[str, str]] = []


def pytest_sessionstart(session: pytest.Session) -> None:
    del session
    _UNEXPECTED_SKIPS.clear()


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
