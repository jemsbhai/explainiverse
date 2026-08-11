"""Quantus parity tests are an exact audited exception to the floor lane."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_quantus_partition.py"
SPEC = importlib.util.spec_from_file_location("validate_quantus_partition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
partition = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = partition
SPEC.loader.exec_module(partition)
MANIFEST = ROOT / ".github" / "constraints" / "quantus-reference-tests.txt"


def test_repository_quantus_partition_matches_the_reviewed_manifest():
    partition.validate_partition(ROOT / "tests", MANIFEST)


def _write_test(root: Path, source: str, manifest_entries=()):
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_metric.py").write_text(source, encoding="utf-8")
    manifest = root / "manifest.txt"
    manifest.write_text("\n".join(manifest_entries) + "\n", encoding="utf-8")
    return tests, manifest


def test_unmarked_quantus_import_is_rejected(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "def test_reference():\n    import quantus\n",
    )
    with pytest.raises(ValueError, match="missing quantus_reference marker"):
        partition.validate_partition(tests, manifest)


def test_module_scope_quantus_import_is_rejected_even_when_test_is_marked(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "import quantus\n\nimport pytest\n\n@pytest.mark.quantus_reference\ndef test_reference():\n    assert quantus\n",
        ["tests/test_metric.py::test_reference"],
    )
    with pytest.raises(ValueError, match="module scope"):
        partition.validate_partition(tests, manifest)


def test_unmarked_call_to_quantus_loading_helper_is_rejected(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "import pytest\n\ndef _quantus_reference():\n    return pytest.importorskip('quantus')\n\ndef test_reference():\n    _quantus_reference()\n",
    )
    with pytest.raises(ValueError, match="without quantus_reference marker"):
        partition.validate_partition(tests, manifest)


def test_manifest_drift_is_rejected_in_both_directions(tmp_path):
    source = (
        "import pytest\n\n@pytest.mark.quantus_reference\n"
        "def test_reference():\n    pytest.importorskip('quantus')\n"
    )
    tests, manifest = _write_test(tmp_path, source)
    with pytest.raises(ValueError, match="manifest is missing"):
        partition.validate_partition(tests, manifest)
    manifest.write_text("tests/test_metric.py::test_stale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing.*stale"):
        partition.validate_partition(tests, manifest)


def test_manifest_must_be_sorted_and_unique(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "tests/test_z.py::test_z\ntests/test_a.py::test_a\ntests/test_a.py::test_a\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sorted.*unique"):
        partition.load_manifest(manifest)
