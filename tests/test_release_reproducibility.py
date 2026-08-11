import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison = _load_script(
    "compare_distribution_artifacts",
    ROOT / "scripts" / "compare_distribution_artifacts.py",
)
environment = _load_script(
    "record_release_environment",
    ROOT / "scripts" / "record_release_environment.py",
)
compare_distribution_directories = comparison.compare_distribution_directories
release_environment = environment.release_environment


def _artifact(directory: Path, name: str, content: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def test_independent_distribution_inventories_must_be_byte_identical(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        _artifact(directory, "example-1.0-py3-none-any.whl", b"wheel")
        _artifact(directory, "example-1.0.tar.gz", b"sdist")

    report = compare_distribution_directories(first, second)

    assert report["schema_version"] == 1
    assert report["comparison"] == "byte-identical"
    assert report["reproducible"] is True
    assert report["first"] == report["second"]


def test_distribution_comparison_reports_hash_difference(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _artifact(first, "example-1.0-py3-none-any.whl", b"first")
    _artifact(second, "example-1.0-py3-none-any.whl", b"second")

    with pytest.raises(ValueError, match="not byte-identical") as exc_info:
        compare_distribution_directories(first, second)

    assert hashlib.sha256(b"first").hexdigest() in str(exc_info.value)
    assert hashlib.sha256(b"second").hexdigest() in str(exc_info.value)


def test_distribution_comparison_rejects_missing_artifact_name(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _artifact(first, "example-1.0-py3-none-any.whl", b"wheel")
    _artifact(second, "example-1.0.tar.gz", b"sdist")

    with pytest.raises(ValueError, match="filename sets differ"):
        compare_distribution_directories(first, second)


def test_release_environment_records_pip_runner_tools_and_lock(monkeypatch, tmp_path):
    requirements = _artifact(tmp_path, "release-tools.txt", b"poetry==2.3.2\n")
    monkeypatch.setenv("ImageOS", "ubuntu24")
    monkeypatch.setenv("ImageVersion", "20260801.1")
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setattr(
        "record_release_environment.importlib.metadata.version",
        lambda package: {
            "poetry": "2.3.2",
            "twine": "6.2.0",
            "cyclonedx-bom": "7.2.1",
        }[package],
    )

    payload = release_environment(requirements)

    assert payload["schema_version"] == 1
    assert payload["bootstrap_pip"].startswith("pip ")
    assert payload["runner"]["ImageOS"] == "ubuntu24"
    assert payload["runner"]["ImageVersion"] == "20260801.1"
    assert payload["runner"]["RUNNER_OS"] == "Linux"
    assert payload["release_tools"] == {
        "poetry": "2.3.2",
        "twine": "6.2.0",
        "cyclonedx-bom": "7.2.1",
    }
    assert (
        payload["requirements"]["sha256"] == hashlib.sha256(requirements.read_bytes()).hexdigest()
    )
    json.dumps(payload, allow_nan=False)


def test_release_tool_graph_is_fully_pinned_and_hashed_for_linux_python312():
    lock_path = ROOT / ".github" / "requirements" / "release-tools.txt"
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    assert "--python-version 3.12 --python-platform x86_64-manylinux_2_28" in lines[1]

    requirement_indexes = [
        index for index, line in enumerate(lines) if line and not line.startswith((" ", "#"))
    ]
    assert requirement_indexes
    for position, index in enumerate(requirement_indexes):
        requirement = lines[index]
        assert "==" in requirement
        end = (
            requirement_indexes[position + 1]
            if position + 1 < len(requirement_indexes)
            else len(lines)
        )
        assert any("--hash=sha256:" in line for line in lines[index + 1 : end])

    rendered = "\n".join(lines)
    assert "poetry==2.3.2" in rendered
    assert "twine==6.2.0" in rendered
    assert "cyclonedx-bom==7.2.1" in rendered


def test_reproducibility_workflow_uses_two_clean_runners_and_no_publish_step():
    workflow = (ROOT / ".github" / "workflows" / "artifact-reproducibility.yml").read_text(
        encoding="utf-8"
    )

    assert "build-id: [one, two]" in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "--require-hashes" in workflow
    assert "--only-binary=:all:" in workflow
    assert "SOURCE_DATE_EPOCH" in workflow
    assert "Artifact byte reproducibility" in workflow
    assert "compare_distribution_artifacts.py" in workflow
    assert "gh-action-pypi-publish" not in workflow
    assert "gh release create" not in workflow


def test_publish_build_uses_the_same_hashed_tools_and_records_its_environment():
    workflow = (ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04" in workflow
    assert 'BOOTSTRAP_PIP_VERSION: "26.2.1"' in workflow
    assert "--require-hashes" in workflow
    assert "--only-binary=:all:" in workflow
    assert "--requirement .github/requirements/release-tools.txt" in workflow
    assert "record_release_environment.py" in workflow
    assert "provenance/release-environment.json" in workflow
    assert "SOURCE_DATE_EPOCH" in workflow
    assert "PYTHONHASHSEED=0" in workflow
    assert '"poetry==2.3.2"' not in workflow
    assert '"twine==6.2.0"' not in workflow
    assert '"cyclonedx-bom==7.2.1"' not in workflow
