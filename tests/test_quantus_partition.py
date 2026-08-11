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


def _write_support_file(tests: Path, name: str, source: str) -> None:
    (tests / name).write_text(source, encoding="utf-8")


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


def test_module_scope_dynamic_import_alias_is_rejected(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "import importlib as imports\n"
        "loader = imports.import_module\n"
        "module_name = 'quantus'\n"
        "module_alias = module_name\n"
        "backend = loader(module_alias)\n\n"
        "def test_plain():\n"
        "    assert backend\n",
    )

    with pytest.raises(ValueError, match="module scope"):
        partition.validate_partition(tests, manifest)


def test_dynamic_import_uses_module_level_string_aliases(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "import importlib\n"
        "QUANTUS_MODULE = 'quantus'\n\n"
        "def test_reference():\n"
        "    loader = importlib.import_module\n"
        "    loader(QUANTUS_MODULE)\n",
    )

    with pytest.raises(ValueError, match="missing quantus_reference marker"):
        partition.validate_partition(tests, manifest)


def test_unmarked_call_to_quantus_loading_helper_is_rejected(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "import pytest\n\ndef _quantus_reference():\n    return pytest.importorskip('quantus')\n\ndef test_reference():\n    _quantus_reference()\n",
    )
    with pytest.raises(ValueError, match="without quantus_reference marker"):
        partition.validate_partition(tests, manifest)


def test_all_python_files_and_unused_quantus_fixtures_are_audited(tmp_path):
    tests, manifest = _write_test(tmp_path, "def test_plain():\n    pass\n")
    _write_support_file(
        tests,
        "conftest.py",
        "import importlib as imports\n"
        "import pytest\n\n"
        "@pytest.fixture\n"
        "def quantus_backend():\n"
        "    loader = imports.import_module\n"
        "    return loader('quantus')\n",
    )

    with pytest.raises(ValueError, match="Quantus-loading helper/fixture is not referenced"):
        partition.validate_partition(tests, manifest)


def test_unmarked_fixture_request_and_import_module_alias_are_rejected(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "def test_reference(quantus_backend):\n    assert quantus_backend\n",
    )
    _write_support_file(
        tests,
        "conftest.py",
        "from importlib import import_module as load\n"
        "import pytest\n\n"
        "@pytest.fixture\n"
        "def quantus_backend():\n"
        "    return load(name='quantus')\n",
    )

    with pytest.raises(ValueError, match="without quantus_reference marker"):
        partition.validate_partition(tests, manifest)


def test_marked_test_can_request_an_aliased_quantus_fixture(tmp_path):
    node_id = "tests/test_metric.py::test_reference"
    tests, manifest = _write_test(
        tmp_path,
        "import pytest\n\n"
        "@pytest.mark.quantus_reference\n"
        "def test_reference(quantus_backend):\n"
        "    assert quantus_backend\n",
        [node_id],
    )
    _write_support_file(
        tests,
        "conftest.py",
        "from pytest import fixture as fixture_alias\n\n"
        "@fixture_alias(name='quantus_backend')\n"
        "def _backend():\n"
        "    loader = __import__\n"
        "    return loader('quantus')\n",
    )

    partition.validate_partition(tests, manifest)


def test_class_method_node_id_and_dunder_import_alias_are_supported(tmp_path):
    source = (
        "import pytest\n\n"
        "class TestReference:\n"
        "    @pytest.mark.quantus_reference\n"
        "    def test_backend(self):\n"
        "        loader = __import__\n"
        "        loader('quantus.metrics')\n"
    )
    node_id = "tests/test_metric.py::TestReference::test_backend"
    tests, manifest = _write_test(tmp_path, source, [node_id])

    marked, violations = partition.discover_partition(tests)

    assert marked == [node_id]
    assert violations == []
    partition.validate_partition(tests, manifest)


def test_simple_helper_and_importorskip_aliases_preserve_the_marked_partition(tmp_path):
    source = (
        "import pytest\n"
        "from pytest import importorskip as optional_import\n\n"
        "def _load_reference():\n"
        "    return optional_import(modname='quantus')\n\n"
        "load_alias = _load_reference\n\n"
        "@pytest.mark.quantus_reference\n"
        "def test_reference():\n"
        "    assert load_alias()\n"
    )
    node_id = "tests/test_metric.py::test_reference"
    tests, manifest = _write_test(tmp_path, source, [node_id])

    partition.validate_partition(tests, manifest)


def test_quantus_loading_autouse_fixture_is_rejected(tmp_path):
    tests, manifest = _write_test(tmp_path, "def test_plain():\n    pass\n")
    _write_support_file(
        tests,
        "conftest.py",
        "import pytest\n\n"
        "@pytest.fixture(autouse=True)\n"
        "def quantus_backend():\n"
        "    return pytest.importorskip('quantus')\n",
    )

    with pytest.raises(ValueError, match="autouse fixtures"):
        partition.validate_partition(tests, manifest)


def test_request_getfixturevalue_cannot_hide_an_unmarked_quantus_fixture(tmp_path):
    node_id = "tests/test_metric.py::test_reference"
    tests, manifest = _write_test(
        tmp_path,
        "import pytest\n\n"
        "@pytest.mark.quantus_reference\n"
        "def test_reference(quantus_backend):\n    assert quantus_backend\n\n"
        "def test_plain(request):\n"
        "    request.getfixturevalue('quantus_backend')\n",
        [node_id],
    )
    _write_support_file(
        tests,
        "conftest.py",
        "import pytest\n\n"
        "@pytest.fixture\n"
        "def quantus_backend():\n"
        "    return pytest.importorskip('quantus')\n",
    )

    with pytest.raises(ValueError, match="without quantus_reference marker"):
        partition.validate_partition(tests, manifest)


def test_class_usefixtures_cannot_hide_an_unmarked_quantus_fixture(tmp_path):
    node_id = "tests/test_metric.py::test_reference"
    tests, manifest = _write_test(
        tmp_path,
        "import pytest\n\n"
        "@pytest.mark.quantus_reference\n"
        "def test_reference(quantus_backend):\n    assert quantus_backend\n\n"
        "@pytest.mark.usefixtures('quantus_backend')\n"
        "class TestPlain:\n"
        "    def test_plain(self):\n"
        "        pass\n",
        [node_id],
    )
    _write_support_file(
        tests,
        "conftest.py",
        "import pytest\n\n"
        "@pytest.fixture\n"
        "def quantus_backend():\n"
        "    return pytest.importorskip('quantus')\n",
    )

    with pytest.raises(ValueError, match="without quantus_reference marker"):
        partition.validate_partition(tests, manifest)


def test_dynamic_getattr_import_cannot_bypass_the_partition(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "import importlib\n\n"
        "def test_plain():\n"
        "    getattr(importlib, 'import_module')('quantus')\n",
    )

    with pytest.raises(ValueError, match="missing quantus_reference marker"):
        partition.validate_partition(tests, manifest)


def test_parameterised_import_wrapper_cannot_bypass_the_partition(tmp_path):
    node_id = "tests/test_metric.py::test_reference"
    tests, manifest = _write_test(
        tmp_path,
        "import importlib\n"
        "import pytest\n\n"
        "def load_module(name):\n"
        "    return importlib.import_module(name)\n\n"
        "@pytest.mark.quantus_reference\n"
        "def test_reference():\n"
        "    import quantus\n"
        "    assert quantus\n\n"
        "def test_plain():\n"
        "    load_module('quantus')\n",
        [node_id],
    )

    with pytest.raises(ValueError, match="without quantus_reference marker"):
        partition.validate_partition(tests, manifest)


def test_unrelated_quantus_reference_decorator_is_not_a_pytest_marker(tmp_path):
    tests, _ = _write_test(
        tmp_path,
        "def quantus_reference(function):\n"
        "    return function\n\n"
        "@quantus_reference\n"
        "def test_plain():\n"
        "    import quantus\n",
    )

    marked, violations = partition.discover_partition(tests)

    assert marked == []
    assert any("missing quantus_reference marker" in value for value in violations)


def test_real_marker_alias_and_module_pytestmark_are_supported(tmp_path):
    node_id = "tests/test_metric.py::test_reference"
    tests, manifest = _write_test(
        tmp_path,
        "import pytest\n\n"
        "reference = pytest.mark.quantus_reference\n"
        "pytestmark = reference\n\n"
        "def test_reference():\n"
        "    import quantus\n",
        [node_id],
    )

    partition.validate_partition(tests, manifest)


def test_nearest_conftest_fixture_override_prevents_a_false_positive(tmp_path):
    tests = tmp_path / "tests"
    (tests / "sub").mkdir(parents=True)
    (tests / "conftest.py").write_text(
        "import pytest\n\n"
        "@pytest.fixture\n"
        "def quantus_backend():\n"
        "    return pytest.importorskip('quantus')\n",
        encoding="utf-8",
    )
    (tests / "test_reference.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.quantus_reference\n"
        "def test_reference(quantus_backend):\n"
        "    assert quantus_backend\n",
        encoding="utf-8",
    )
    (tests / "sub" / "conftest.py").write_text(
        "import pytest\n\n" "@pytest.fixture\n" "def quantus_backend():\n" "    return object()\n",
        encoding="utf-8",
    )
    (tests / "sub" / "test_plain.py").write_text(
        "def test_plain(quantus_backend):\n" "    assert quantus_backend\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "tests/test_reference.py::test_reference\n",
        encoding="utf-8",
    )

    partition.validate_partition(tests, manifest)


def test_dormant_nested_import_and_rebound_loader_do_not_create_false_positives(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "import importlib\n\n"
        "def test_plain():\n"
        "    def never_called():\n"
        "        import quantus\n"
        "    loader = importlib.import_module\n"
        "    loader = lambda name: None\n"
        "    loader('quantus')\n",
    )

    partition.validate_partition(tests, manifest)


def test_called_nested_import_and_loader_use_before_rebinding_are_rejected(tmp_path):
    tests, manifest = _write_test(
        tmp_path,
        "import importlib\n\n"
        "def test_plain():\n"
        "    loader = importlib.import_module\n"
        "    loader('quantus')\n"
        "    loader = lambda name: None\n"
        "    def load_nested():\n"
        "        import quantus\n"
        "    load_nested()\n",
    )

    with pytest.raises(ValueError, match="missing quantus_reference marker"):
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
