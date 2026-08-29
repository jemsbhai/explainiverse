import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
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
compare_release_environments = environment.compare_release_environments


def _compare_release_environments(first, second, *, run_id="123456", run_attempt="1"):
    return compare_release_environments(
        first,
        second,
        expected_run_id=run_id,
        expected_run_attempt=run_attempt,
    )


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
    _artifact(first, "example-1.0.tar.gz", b"sdist")
    _artifact(second, "example-1.0.tar.gz", b"sdist")

    with pytest.raises(ValueError, match="not byte-identical") as exc_info:
        compare_distribution_directories(first, second)

    assert hashlib.sha256(b"first").hexdigest() in str(exc_info.value)
    assert hashlib.sha256(b"second").hexdigest() in str(exc_info.value)


def test_distribution_comparison_rejects_missing_artifact_name(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _artifact(first, "example-1.0-py3-none-any.whl", b"wheel")
    _artifact(first, "example-1.0.tar.gz", b"sdist")
    _artifact(second, "example-1.0-py3-none-any.whl", b"wheel")
    _artifact(second, "renamed-1.0.tar.gz", b"sdist")

    with pytest.raises(ValueError, match="filename sets differ"):
        compare_distribution_directories(first, second)


@pytest.mark.parametrize("artifact_name", ["example-1.0-py3-none-any.whl", "example-1.0.tar.gz"])
def test_distribution_comparison_requires_exactly_one_wheel_and_one_sdist(tmp_path, artifact_name):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _artifact(first, artifact_name, b"same")
    _artifact(second, artifact_name, b"same")

    with pytest.raises(ValueError, match="exactly one wheel and one source distribution"):
        compare_distribution_directories(first, second)


def test_distribution_comparison_rejects_multiple_wheels_or_sdists(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        _artifact(directory, "example-1.0-py3-none-any.whl", b"wheel")
        _artifact(directory, "example-1.0-py2-none-any.whl", b"other-wheel")
        _artifact(directory, "example-1.0.tar.gz", b"sdist")

    with pytest.raises(ValueError, match="exactly one wheel and one source distribution"):
        compare_distribution_directories(first, second)


def test_distribution_comparison_cli_writes_failure_report_before_nonzero_exit(
    monkeypatch, tmp_path
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    report_path = tmp_path / "provenance" / "reproducibility.json"
    for directory, wheel in ((first, b"first"), (second, b"second")):
        _artifact(directory, "example-1.0-py3-none-any.whl", wheel)
        _artifact(directory, "example-1.0.tar.gz", b"sdist")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_distribution_artifacts.py",
            str(first),
            str(second),
            "--report",
            str(report_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        comparison.main()

    assert exc_info.value.code == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reproducible"] is False
    assert (
        report["first"]["example-1.0-py3-none-any.whl"]["sha256"]
        == hashlib.sha256(b"first").hexdigest()
    )
    assert (
        report["second"]["example-1.0-py3-none-any.whl"]["sha256"]
        == hashlib.sha256(b"second").hexdigest()
    )
    assert "not byte-identical" in report["error"]


def test_release_environment_records_pip_runner_tools_and_lock(monkeypatch, tmp_path):
    requirements = _artifact(tmp_path, "release-tools.txt", b"poetry==2.3.2\n")
    monkeypatch.setenv("ImageOS", "ubuntu24")
    monkeypatch.setenv("ImageVersion", "20260801.1")
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("RUNNER_NAME", "GitHub Actions 1")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_RUN_ID", "123456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("REPRODUCIBILITY_BUILD_ID", "one")
    monkeypatch.setenv("REPRODUCIBILITY_JOB_INDEX", "0")
    monkeypatch.setenv("REPRODUCIBILITY_JOB_TOTAL", "2")
    monkeypatch.setenv("REPRODUCIBILITY_RUNNER_ENVIRONMENT", "github-hosted")
    monkeypatch.setattr(
        "record_release_environment.importlib.metadata.version",
        lambda package: {
            "poetry": "2.3.2",
            "twine": "6.2.0",
            "cyclonedx-bom": "7.2.1",
        }[package],
    )

    payload = release_environment(requirements, build_identity="123456:1:one")

    assert payload["schema_version"] == 1
    assert payload["bootstrap_pip"].startswith("pip ")
    assert payload["bootstrap_pip_version"]
    assert payload["build_identity"] == "123456:1:one"
    assert payload["runner"]["ImageOS"] == "ubuntu24"
    assert payload["runner"]["ImageVersion"] == "20260801.1"
    assert payload["runner"]["RUNNER_OS"] == "Linux"
    assert payload["source"]["commit"]
    assert payload["source"]["commit_timestamp"].isdigit()
    assert payload["platform_identity"]["system"]
    assert payload["platform_identity"]["machine"]
    assert payload["release_tools"] == {
        "poetry": "2.3.2",
        "twine": "6.2.0",
        "cyclonedx-bom": "7.2.1",
    }
    assert (
        payload["requirements"]["sha256"] == hashlib.sha256(requirements.read_bytes()).hexdigest()
    )
    json.dumps(payload, allow_nan=False)


def _recorded_environment(build_identity="123456:1:one"):
    slot = build_identity.rsplit(":", 1)[-1] if isinstance(build_identity, str) else "one"
    job_index = {"one": "0", "two": "1"}.get(slot, "0")
    return {
        "schema_version": 1,
        "build_identity": build_identity,
        "python": {
            "executable": "/opt/hostedtoolcache/Python/3.12/bin/python",
            "implementation": "CPython",
            "version": "3.12.12",
        },
        "bootstrap_pip": "pip 26.2.1 from /opt/pip (python 3.12)",
        "bootstrap_pip_version": "26.2.1",
        "platform": "Linux-6.11-host-specific",
        "platform_identity": {"system": "Linux", "machine": "x86_64"},
        "source": {"commit": "a" * 40, "commit_timestamp": "1786415675"},
        "release_tools": {
            "poetry": "2.3.2",
            "twine": "6.2.0",
            "cyclonedx-bom": "7.2.1",
        },
        "requirements": {
            "path": ".github/requirements/release-tools.txt",
            "sha256": "b" * 64,
        },
        "runner": {
            "GITHUB_ACTION": "__run_3",
            "GITHUB_ACTIONS": "true",
            "GITHUB_JOB": "independent-build",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_REPOSITORY": "jemsbhai/explainiverse",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "123456",
            "GITHUB_SHA": "a" * 40,
            "GITHUB_WORKFLOW": "Artifact reproducibility",
            "ImageOS": "ubuntu24",
            "ImageVersion": "20260801.1",
            "RUNNER_ARCH": "X64",
            "RUNNER_NAME": f"GitHub Actions {int(job_index) + 1}",
            "RUNNER_OS": "Linux",
            "REPRODUCIBILITY_BUILD_ID": slot,
            "REPRODUCIBILITY_JOB_INDEX": job_index,
            "REPRODUCIBILITY_JOB_TOTAL": "2",
            "REPRODUCIBILITY_RUNNER_ENVIRONMENT": "github-hosted",
        },
    }


def test_environment_comparison_matches_stable_identity_and_proves_distinct_builds():
    first = _recorded_environment("123456:1:one")
    second = _recorded_environment("123456:1:two")
    second["runner"]["RUNNER_NAME"] = "GitHub Actions 2"
    second["platform"] = "Linux-6.11-another-host"
    second["python"]["executable"] = "/another/stable/python/path"

    report = _compare_release_environments(first, second)

    assert report["compatible"] is True
    assert report["comparison"] == "stable-environment-identity"
    assert report["source_workflow"] == {"run_id": "123456", "run_attempt": "1"}
    assert report["distinct_build_identities"] == {
        "first": "123456:1:one",
        "second": "123456:1:two",
    }
    assert report["hosted_runner_jobs"] == {
        "first": {
            "runner_name": "GitHub Actions 1",
            "matrix_slot": "one",
            "matrix_job_index": "0",
        },
        "second": {
            "runner_name": "GitHub Actions 2",
            "matrix_slot": "two",
            "matrix_job_index": "1",
        },
    }
    assert report["matching_identity"]["runner.ImageVersion"] == "20260801.1"
    assert report["first"] == first
    assert report["second"] == second


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["python"].update(version="3.13.0"), "python.version"),
        (
            lambda value: value.update(bootstrap_pip_version="26.2.2"),
            "bootstrap_pip_version",
        ),
        (
            lambda value: value["platform_identity"].update(machine="aarch64"),
            "platform_identity.machine",
        ),
        (lambda value: value["source"].update(commit="c" * 40), "source"),
        (
            lambda value: value["release_tools"].update(poetry="2.3.3"),
            "release_tools",
        ),
        (
            lambda value: value["requirements"].update(sha256="d" * 64),
            "requirements",
        ),
        (
            lambda value: value["runner"].update(GITHUB_RUN_ID="654321"),
            "runner.GITHUB_RUN_ID",
        ),
        (
            lambda value: value["runner"].update(GITHUB_SHA="e" * 40),
            "runner.GITHUB_SHA",
        ),
        (
            lambda value: value["runner"].update(ImageVersion="20260808.1"),
            "runner.ImageVersion",
        ),
        (
            lambda value: value["runner"].update(RUNNER_OS="Windows"),
            "runner.RUNNER_OS",
        ),
    ],
)
def test_environment_comparison_rejects_stable_identity_drift(mutation, match):
    first = _recorded_environment("123456:1:one")
    second = deepcopy(_recorded_environment("123456:1:two"))
    mutation(second)

    with pytest.raises(ValueError, match=match):
        _compare_release_environments(first, second)


def test_environment_comparison_rejects_reused_or_missing_build_identity():
    first = _recorded_environment("123456:1:one")
    second = _recorded_environment("123456:1:one")
    with pytest.raises(ValueError, match="second build identity must use slot 'two'"):
        _compare_release_environments(first, second)

    second["build_identity"] = None
    with pytest.raises(ValueError, match="second release environment has no build identity"):
        _compare_release_environments(first, second)


@pytest.mark.parametrize("identity", ["forged-two", "123456:1:three", "123456:1:two:extra"])
def test_environment_comparison_rejects_unstructured_build_identity(identity):
    first = _recorded_environment("123456:1:one")
    second = _recorded_environment(identity)

    with pytest.raises(ValueError, match=r"RUN_ID:RUN_ATTEMPT:one\|two"):
        _compare_release_environments(first, second)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("GITHUB_RUN_ID", "654321", "run_id does not match"),
        ("GITHUB_RUN_ATTEMPT", "2", "run_attempt does not match"),
        ("REPRODUCIBILITY_BUILD_ID", "one", "slot does not match"),
        ("REPRODUCIBILITY_JOB_INDEX", "0", "matrix job index"),
        ("REPRODUCIBILITY_JOB_TOTAL", "3", "exactly two jobs"),
        ("REPRODUCIBILITY_RUNNER_ENVIRONMENT", "self-hosted", "GitHub-hosted runner"),
        ("GITHUB_ACTIONS", "false", "not bound to GitHub Actions"),
    ],
)
def test_environment_comparison_binds_build_identity_to_hosted_matrix_job(field, value, match):
    first = _recorded_environment("123456:1:one")
    second = _recorded_environment("123456:1:two")
    second["runner"][field] = value

    with pytest.raises(ValueError, match=match):
        _compare_release_environments(first, second)


def test_environment_comparison_retains_but_does_not_require_unique_runner_display_names():
    first = _recorded_environment("123456:1:one")
    second = _recorded_environment("123456:1:two")
    second["runner"]["RUNNER_NAME"] = first["runner"]["RUNNER_NAME"]

    report = _compare_release_environments(first, second)

    assert report["hosted_runner_jobs"]["first"]["runner_name"] == "GitHub Actions 1"
    assert report["hosted_runner_jobs"]["second"]["runner_name"] == "GitHub Actions 1"
    assert (
        report["distinct_build_identities"]["first"]
        != report["distinct_build_identities"]["second"]
    )


def test_environment_comparison_binds_recorded_commit_to_github_sha():
    first = _recorded_environment("123456:1:one")
    second = _recorded_environment("123456:1:two")
    first["source"]["commit"] = second["source"]["commit"] = "c" * 40
    first["runner"]["GITHUB_SHA"] = "d" * 40
    second["runner"]["GITHUB_SHA"] = "d" * 40

    with pytest.raises(ValueError, match="first source commit does not match runner GITHUB_SHA"):
        _compare_release_environments(first, second)


@pytest.mark.parametrize(
    ("run_id", "run_attempt", "match"),
    [
        ("654321", "1", "GITHUB_RUN_ID does not match the current workflow run"),
        ("123456", "2", "GITHUB_RUN_ATTEMPT does not match the current workflow attempt"),
        ("0", "1", "expected run ID must be a positive decimal string"),
        ("123456", "0", "expected run attempt must be a positive decimal string"),
    ],
)
def test_environment_comparison_rejects_manifests_from_a_different_attempt(
    run_id, run_attempt, match
):
    first = _recorded_environment("123456:1:one")
    second = _recorded_environment("123456:1:two")

    with pytest.raises(ValueError, match=match):
        _compare_release_environments(
            first,
            second,
            run_id=run_id,
            run_attempt=run_attempt,
        )


def test_environment_comparison_cli_writes_both_complete_manifests(monkeypatch, tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    output_path = tmp_path / "comparison.json"
    first = _recorded_environment("123456:1:one")
    second = _recorded_environment("123456:1:two")
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_release_environment.py",
            "--compare",
            str(first_path),
            str(second_path),
            "--expected-run-id",
            "123456",
            "--expected-run-attempt",
            "1",
            "--output",
            str(output_path),
        ],
    )

    environment.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["first"] == first
    assert report["second"] == second


def test_environment_comparison_cli_writes_failure_report_before_nonzero_exit(
    monkeypatch, tmp_path
):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    output_path = tmp_path / "provenance" / "comparison.json"
    first = _recorded_environment("123456:1:one")
    second = _recorded_environment("123456:1:two")
    second["runner"]["ImageVersion"] = "different-image"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_release_environment.py",
            "--compare",
            str(first_path),
            str(second_path),
            "--expected-run-id",
            "123456",
            "--expected-run-attempt",
            "1",
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        environment.main()

    assert exc_info.value.code == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["compatible"] is False
    assert "runner.ImageVersion" in report["error"]
    assert report["first"] == first
    assert report["second"] == second


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
    assert 'SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"' in workflow
    assert "export SOURCE_DATE_EPOCH" in workflow
    assert 'export SOURCE_DATE_EPOCH="$(' not in workflow
    assert "python -m twine check ./dist/*" in workflow
    assert "Artifact byte reproducibility" in workflow
    assert "compare_distribution_artifacts.py" in workflow
    assert (
        '--build-identity "$GITHUB_RUN_ID:$GITHUB_RUN_ATTEMPT:$REPRODUCIBILITY_BUILD_ID"'
        in workflow
    )
    assert "REPRODUCIBILITY_JOB_INDEX: ${{ strategy.job-index }}" in workflow
    assert "REPRODUCIBILITY_JOB_TOTAL: ${{ strategy.job-total }}" in workflow
    assert "REPRODUCIBILITY_RUNNER_ENVIRONMENT: ${{ runner.environment }}" in workflow
    assert "--compare" in workflow
    assert '--expected-run-id "$GITHUB_RUN_ID"' in workflow
    assert '--expected-run-attempt "$GITHUB_RUN_ATTEMPT"' in workflow
    assert "release-environment-comparison.json" in workflow
    assert "release-environment-one.json" in workflow
    assert "release-environment-two.json" in workflow
    assert workflow.index("cp independent/one") < workflow.index("--compare")
    assert workflow.index("cp independent/two") < workflow.index("--compare")
    assert "Require stable hosted-runner environment identity\n        if: always()" in workflow
    assert "Require byte-identical wheel and source distribution\n        if: always()" in workflow
    assert "path: provenance/*" in workflow
    assert "Archive reproducibility report\n        if: always()" in workflow
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
    assert 'SOURCE_DATE_EPOCH="$(git -C release-source log -1 --format=%ct)"' in workflow
    assert "export SOURCE_DATE_EPOCH" in workflow
    assert 'export SOURCE_DATE_EPOCH="$(' not in workflow
    assert "python -m twine check ./dist/*" in workflow
    assert "PYTHONHASHSEED=0" in workflow
    assert "reproducibility_run_attempt=$(" in workflow
    assert "provenance/reproducibility-source-run.json" in workflow
    assert '--expected-run-id "$reproducibility_run_id"' in workflow
    assert '--expected-run-attempt "$reproducibility_run_attempt"' in workflow
    assert '"poetry==2.3.2"' not in workflow
    assert '"twine==6.2.0"' not in workflow
    assert '"cyclonedx-bom==7.2.1"' not in workflow
