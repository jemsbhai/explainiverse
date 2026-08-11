"""The hardware release suite cannot turn missing coverage into a green skip."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cuda_release_conftest", ROOT / "tests_cuda" / "conftest.py"
)
assert SPEC is not None and SPEC.loader is not None
policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy
SPEC.loader.exec_module(policy)


class _PluginManager:
    def get_plugin(self, name):
        del name
        return None


def _session():
    return SimpleNamespace(
        config=SimpleNamespace(pluginmanager=_PluginManager()),
        exitstatus=pytest.ExitCode.OK,
    )


def test_runtime_skip_forces_the_cuda_session_to_fail():
    session = _session()
    policy.pytest_sessionstart(session)
    policy.pytest_runtest_logreport(
        SimpleNamespace(
            skipped=True, nodeid="tests_cuda/test_example.py::test_gpu", longrepr="no GPU"
        )
    )

    policy.pytest_sessionfinish(session, pytest.ExitCode.OK)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_collection_skip_forces_the_cuda_session_to_fail():
    session = _session()
    policy.pytest_sessionstart(session)
    policy.pytest_collectreport(
        SimpleNamespace(
            skipped=True, nodeid="tests_cuda/test_example.py", longrepr="missing import"
        )
    )

    policy.pytest_sessionfinish(session, pytest.ExitCode.OK)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_zero_skips_preserves_the_existing_cuda_session_result():
    session = _session()
    policy.pytest_sessionstart(session)

    policy.pytest_sessionfinish(session, pytest.ExitCode.OK)

    assert session.exitstatus == pytest.ExitCode.OK
