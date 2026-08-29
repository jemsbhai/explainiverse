from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, TypeVar, cast

import pytest

from scripts import release_external_controls as controls
from scripts.release_gpu_jit_lambda_controller import (
    ControllerError,
    EvidenceJournal,
    GitHubResponse,
)
from scripts.release_gpu_jit_lambda_controller.controller import SealedControllerResources
from scripts.release_gpu_jit_lambda_live import (
    ContractError,
    build_immutable_plan,
    create_evidence_directory,
    load_runtime_bundle,
)
from scripts.release_gpu_jit_lambda_operator import bootstrap as operator_bootstrap
from scripts.release_gpu_jit_lambda_operator import boundary, build_source_worktree_manifest
from scripts.release_gpu_jit_lambda_operator import cli as operator
from scripts.release_gpu_jit_lambda_operator import install_windows_python, install_windows_runtime
from scripts.release_gpu_jit_lambda_operator import preloader as operator_preloader
from scripts.release_gpu_jit_lambda_operator import windows_launcher

NOW = datetime.now(timezone.utc)
HEAD = "a" * 40
T = TypeVar("T")


def _append_and_return(events: list[str], event: str, value: T) -> T:
    events.append(event)
    return value


def _repository_inventory_record() -> dict[str, Any]:
    return {
        "repository": "jemsbhai/explainiverse",
        "absolute_root": "C:\\fixture\\repository",
        "origin_url": boundary.EXPECTED_ORIGIN_URL,
        "head_sha": HEAD,
        "tree_object_sha": "b" * 40,
        "tree_inventory_sha256": "c" * 64,
        "clean_tracked_and_untracked": True,
        "supplied_ref": "refs/heads/main",
        "remote_object_type": "commit",
        "remote_object_sha": HEAD,
        "remote_target_sha": HEAD,
        "remote_ref_response_sha256": "d" * 64,
        "annotated_tag_response_sha256": None,
        "critical_sources": {},
        "git_configuration": {
            "system_config_disabled": True,
            "system_config_path": "nul",
            "global_config_path": "nul",
            "system_attributes_disabled": True,
            "repository_fsmonitor_overridden_false": True,
            "repository_untracked_cache_overridden_false": True,
            "pager_disabled": True,
            "terminal_prompt_disabled": True,
            "optional_locks_disabled": True,
        },
    }


def _controller_resources() -> SealedControllerResources:
    return SealedControllerResources.from_files_for_tests()


def _operator_resources() -> operator.OperatorSealedResources:
    controller = _controller_resources()
    runtime_bundle = load_runtime_bundle(
        Path("scripts/release_gpu_jit_lambda_runtime").resolve(strict=True)
    )
    return operator.OperatorSealedResources(
        controller=controller,
        runtime_bundle=runtime_bundle,
        binding={
            "policy_sha256": controller.policy_sha256,
            "controller_source_sha256": controller.controller_source_sha256,
            "runtime_bundle_sha256": runtime_bundle.sha256,
        },
    )


def test_source_seal_forces_lf_checkout_bytes_for_every_tracked_text_file() -> None:
    assert Path(".gitattributes").read_bytes() == b"* text=auto eol=lf\n"


def _plan() -> Any:
    return build_immutable_plan(
        head_sha=HEAD,
        lifecycle_nonce="b" * 32,
        created_at_unix=int(NOW.timestamp()) - 60,
        expires_at_unix=int(NOW.timestamp()) + 1800,
        current_public_ipv4_cidr="8.8.8.8/32",
        region_description="Illinois, USA",
        image_id="image-fixture-001",
        image_created_time="2025-01-01T00:00:00Z",
        image_description="fixture",
        image_name="lambda-stack-fixture",
        image_family="lambda-stack-22-04",
        image_version="1",
        image_updated_time="2025-01-02T00:00:00Z",
        instance_type_description="NVIDIA A100 80 GB SXM4",
        gpu_description="NVIDIA A100 80 GB SXM4",
        price_cents_per_hour=200,
        vcpus=30,
        memory_gib=200,
        storage_gib=1400,
        ssh_key_name="preexisting-fixture-key",
        ssh_public_key_sha256="c" * 64,
        baseline_file_systems_sha256="d" * 64,
        original_global_rules=[
            {
                "protocol": "icmp",
                "source_network": "0.0.0.0/0",
                "description": "fixture baseline",
            }
        ],
        host_key_fingerprint="SHA256:" + "A" * 43,
        runtime_bundle_sha256="e" * 64,
    )


def _app_capture(
    captured_at: datetime, *, variant: str = ""
) -> tuple[dict[str, Any], dict[str, bytes]]:
    policy, _ = controls.load_policy(
        Path(".github/release-control-policy.json").resolve(strict=True)
    )
    apps = policy["release_runner_authority"]["installed_apps"]
    timestamp = captured_at.astimezone(timezone.utc).isoformat()
    installations = sorted(deepcopy(apps["expected_installations"]), key=lambda item: item["id"])
    roles: list[tuple[str, int | None]] = [("installation-list", None)]
    roles.extend(("installation-configure", item["id"]) for item in installations)
    roles.extend(
        ("permission-update", item["id"])
        for item in installations
        if item["permission_update_requested"]
    )
    evidence: list[dict[str, Any]] = []
    pages: dict[str, bytes] = {}
    for index, (kind, installation_id) in enumerate(roles):
        suffix = "list" if installation_id is None else str(installation_id)
        if kind == "installation-list":
            source_url = "https://github.com/settings/installations"
        elif kind == "installation-configure":
            source_url = f"https://github.com/settings/installations/{installation_id}"
        else:
            source_url = (
                f"https://github.com/settings/installations/{installation_id}/permissions/update"
            )
        item: dict[str, Any] = {
            "filename": f"capture-{kind}-{suffix}.txt",
            "kind": kind,
            "installation_id": installation_id,
            "source_url": source_url,
            "captured_at": timestamp,
            "media_type": "text/plain; charset=utf-8",
            "full_page": True,
        }
        raw = (
            controls._app_evidence_header(item)
            + f"full owner-authenticated page capture: {item['filename']}\n"
            + (f"variant={variant}-{index}\n" if variant else "")
        ).encode("utf-8")
        item.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        evidence.append(item)
        pages[item["filename"]] = raw
    return (
        {
            "schema_version": 1,
            "repository": "jemsbhai/explainiverse",
            "captured_at": timestamp,
            "capture_principal": "jemsbhai",
            "source_url": apps["source_url"],
            "coverage_complete": True,
            "installations": installations,
            "evidence": evidence,
        },
        pages,
    )


def _write_capture_source(
    root: Path, capture: Mapping[str, Any], pages: Mapping[str, bytes]
) -> Any:
    root = root.resolve()
    receipt = create_evidence_directory(root)
    pages_root = root / "pages"
    pages_root.mkdir()
    capture_path = root / "capture.json"
    capture_path.write_bytes(boundary.canonical_json(capture))
    for filename, raw in pages.items():
        (pages_root / filename).write_bytes(raw)
    return receipt


def _publish(
    receipt: Any,
    source_root: Path,
    *,
    phase: str = "pull-request",
    ordinal: int = 1,
    generation: int = 1,
    captured_at: datetime = NOW,
    variant: str = "",
    nonce: str | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    capture, pages = _app_capture(captured_at, variant=variant)
    staging_receipt = _write_capture_source(source_root, capture, pages)
    try:
        boundary.publish_app_capture_generation(
            receipt,
            controller_resources=_controller_resources(),
            staging_receipt=staging_receipt,
            phase=phase,
            ordinal=ordinal,
            generation=generation,
            publication_nonce=nonce or f"{ordinal:08x}{generation:024x}",
            now=captured_at,
        )
    finally:
        staging_receipt.close()
    return capture, pages


def _bind_stale_archive_fixture(
    inbox: boundary.AppCaptureInbox,
    observed: list[dict[str, Any]] | None = None,
) -> None:
    def archive(
        classified_at: str,
        generation: Mapping[str, Any],
        pages: Mapping[str, bytes],
    ) -> Mapping[str, Any]:
        assert set(pages) == {item["filename"] for item in generation["pages"]}
        assert all(
            len(pages[item["filename"]]) == item["bytes"]
            and hashlib.sha256(pages[item["filename"]]).hexdigest() == item["sha256"]
            for item in generation["pages"]
        )
        identity = {
            "phase": inbox.phase,
            "ordinal": generation["ordinal"],
            "generation": generation["generation"],
            "publication_nonce": generation["publication_nonce"],
            "ready_marker_sha256": generation["ready_marker_sha256"],
            "capture_json_sha256": generation["capture_json_sha256"],
            "classified_at": classified_at,
        }
        identity_sha256 = hashlib.sha256(boundary.canonical_json(identity)).hexdigest()
        material = {
            "schema_version": 1,
            "kind": "explainiverse-installed-app-stale-raw-archive",
            **identity,
            "archive_identity_sha256": identity_sha256,
            "archive_directory": f"installed-app-pages/{identity_sha256}",
            "files": generation["pages"],
            "all_pages_exclusive_single_link": True,
        }
        receipt = {
            **material,
            "archive_evidence_sha256": hashlib.sha256(
                boundary.canonical_json(material)
            ).hexdigest(),
        }
        if observed is not None:
            observed.append({"receipt": receipt, "pages": dict(pages)})
        return receipt

    inbox.bind_stale_archive_sink(archive)


def test_parser_defaults_to_inspect_and_has_no_raw_secret_or_publish_surface() -> None:
    parser = operator.build_parser()
    assert parser.parse_args([]).action == "inspect"
    with pytest.raises(SystemExit):
        parser.parse_args(["--lambda-api-key", "forbidden"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--jit-config", "forbidden"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--publish-workflow", "publish-pypi.yml"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--capture-json", "unsafe-loose-source.json"])
    source = Path(operator.__file__).read_text(encoding="utf-8")
    assert "workflow run" not in source
    assert "actions/workflows/" not in source
    assert "subprocess" not in source
    assert "dispatch_release_recovery" in source
    assert "reconcile_release_recovery_dispatch" in source


def test_operator_readme_uses_only_pinned_cp313_byte_sealed_invocations() -> None:
    readme = Path("scripts/release_gpu_jit_lambda_operator/README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    assert "cp312" not in lowered
    assert "cpython 3.12" not in lowered
    assert "-m scripts.release_gpu_jit_lambda_operator" not in readme
    assert "$OperatorPython -I -S -B -c $Shim" in readme
    assert "--operator-target $Target" in readme
    assert "-Target windows-launcher" in readme
    assert "parent_provenance_authenticated=false" in readme
    assert "security_authority_derived_from_declaration=false" in readme
    assert "unauthenticated and non-authoritative" in readme
    assert "$BootstrapWheelhouse" in readme and "$RuntimeWheelhouse" in readme
    assert "--find-links $BootstrapWheelhouse" in readme
    assert "--dest $RuntimeWheelhouse" in readme
    assert "--wheelhouse $RuntimeWheelhouse" in readme
    assert "$Wheelhouse" not in readme
    assert "gpu_8x_a100_80gb_sxm4" in readme
    assert "us-midwest-1" in readme and "Illinois, USA" in readme
    assert "lambda-stack-22-04" in readme
    assert "us-east-1" not in readme


def test_static_tools_exclude_only_the_exact_byte_sealed_preloader_shim() -> None:
    configuration = Path("pyproject.toml").read_text(encoding="utf-8")
    relative = "scripts/release_gpu_jit_lambda_operator/preloader_shim.py"
    assert (
        "extend-exclude = '^/scripts/release_gpu_jit_lambda_operator/" "preloader_shim\\.py$'"
    ) in configuration
    assert f'skip_glob = ["{relative}"]' in configuration
    assert f'extend-exclude = ["{relative}"]' in configuration
    assert (
        "exclude = ['^scripts[\\\\/]release_gpu_jit_lambda_operator" "[\\\\/]preloader_shim\\.py$']"
    ) in configuration
    assert "explicit_package_bases = true" in configuration
    shim = Path(relative).read_bytes()
    assert (
        hashlib.sha256(shim).hexdigest()
        == "22bb14f6e5fed4e7c5456f62e11569c9e5a0846ad1428854e545b2e4c1c979aa"
    )
    python_files = {
        path.as_posix()
        for root in (Path("src"), Path("tests"), Path("scripts"))
        for path in root.rglob("*.py")
    }
    expected = {relative}
    assert {
        path
        for path in python_files
        if re.fullmatch(
            r"^/scripts/release_gpu_jit_lambda_operator/preloader_shim\.py$",
            f"/{path}",
        )
    } == expected
    assert {
        path
        for path in python_files
        if fnmatch.fnmatch(
            path,
            "scripts/release_gpu_jit_lambda_operator/preloader_shim.py",
        )
    } == expected
    assert {
        path
        for path in python_files
        if path == "scripts/release_gpu_jit_lambda_operator/preloader_shim.py"
    } == expected
    assert {
        path
        for path in python_files
        if re.fullmatch(
            r"^scripts[\\/]release_gpu_jit_lambda_operator[\\/]preloader_shim\.py$",
            path,
        )
    } == expected


def test_release_operations_bind_exact_midwest_target_without_fallback() -> None:
    runbook = Path("docs/RELEASE_OPERATIONS.md").read_text(encoding="utf-8")
    capacity = runbook.split("## CUDA capacity acceptance", 1)[1]
    assert "gpu_8x_a100_80gb_sxm4" in capacity
    assert "us-midwest-1" in capacity and "Illinois, USA" in capacity
    assert "lambda-stack-22-04" in capacity
    assert "hard stop rather than a fallback" in capacity


def test_main_rejects_bypassing_secure_entrypoint(capsys: pytest.CaptureFixture[str]) -> None:
    assert operator.main([]) == 2
    assert "secure_entrypoint_required" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("action", "handler_name"),
    (
        ("inspect", "inspect"),
        ("create-app-inbox", "create_app_inbox"),
        ("create-app-staging", "create_app_staging"),
        ("publish-app-capture", "publish_app_capture"),
        ("execute", "execute"),
        ("resume-abort", "resume_abort"),
        ("dispatch-release-recovery", "dispatch_release_recovery"),
        ("transport-self-test", "transport_self_test"),
    ),
)
def test_main_routes_every_explicit_action_only_to_its_named_handler(
    action: str,
    handler_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def handler(_: Any, **__: Any) -> dict[str, Any]:
        calls.append(handler_name)
        return {"schema_version": 1, "kind": f"fixture-{action}"}

    outputs: list[Mapping[str, Any]] = []
    monkeypatch.setattr(operator, handler_name, handler)
    monkeypatch.setattr(operator, "_sealed_resources", lambda *_: _operator_resources())
    monkeypatch.setattr(operator, "_emit", lambda value, **_: outputs.append(value))
    assert (
        operator.main(
            ["--action", action],
            environment_receipt={"scrubbed": True},
            launch_receipt={"secure": True},
            captured_resources={"captured": True},
        )
        == 0
    )
    assert calls == [handler_name]
    assert outputs == [{"schema_version": 1, "kind": f"fixture-{action}"}]


def test_environment_scrub_never_records_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "highly-sensitive-value")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret-proxy.invalid")
    monkeypatch.setenv("SAFE_OPERATOR_FIXTURE", "retained")
    receipt = boundary.scrub_process_environment()
    serialized = boundary.canonical_json(receipt)
    assert b"highly-sensitive-value" not in serialized
    assert b"secret-proxy" not in serialized
    assert "GH_TOKEN" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert os.environ["SAFE_OPERATOR_FIXTURE"] == "retained"


def test_anonymous_confirmation_rejects_regular_file_and_requires_exact_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forbidden.txt"
    path.write_text("a" * 64 + "\n", encoding="ascii")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(boundary.OperatorError, match="anonymous_transport_rejected"):
            boundary.validate_anonymous_fd(descriptor, context="fixture")
    finally:
        os.close(descriptor)

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"a" * 64 + b"\n")
    os.close(write_fd)
    try:
        receipt = boundary.read_plan_confirmation(read_fd, expected_sha256="a" * 64)
        assert receipt["confirmation_read_after_plan"] is True
    finally:
        os.close(read_fd)


def test_direct_module_entrypoint_is_rejected_without_verified_preloader() -> None:
    root = Path.cwd().resolve()
    module = "scripts.release_gpu_jit_lambda_operator"
    rejected = subprocess.run(
        [sys.executable, "-m", module, "--repository-root", str(root), "--help"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert rejected.returncode == 2
    assert b"secure_launch_preloader_receipt_missing" in rejected.stderr
    isolated_bypass = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-m",
            module,
            "--repository-root",
            str(root),
            "--help",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert isolated_bypass.returncode != 0
    assert b"Fail-closed Explainiverse" not in isolated_bypass.stdout


def test_reviewed_shim_matches_tracked_bytes_and_rejects_preloader_drift() -> None:
    root = Path.cwd().resolve()
    shim_path = root / "scripts" / "release_gpu_jit_lambda_operator" / "preloader_shim.py"
    shim_raw = shim_path.read_bytes()
    assert shim_raw.decode("utf-8") == windows_launcher.PRELOADER_SHIM
    shim_sha256 = hashlib.sha256(shim_raw).hexdigest()
    assert shim_sha256 == "22bb14f6e5fed4e7c5456f62e11569c9e5a0846ad1428854e545b2e4c1c979aa"
    preloader = root / "scripts" / "release_gpu_jit_lambda_operator" / "preloader.py"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            windows_launcher.PRELOADER_SHIM,
            shim_sha256,
            str(preloader),
            "0" * 64,
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert b"operator_preloader_shim_digest_rejected" in result.stderr


def test_source_manifest_builder_uses_only_exact_staged_index_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = shutil.which("git")
    assert git is not None
    git_path = Path(git).resolve(strict=True)
    monkeypatch.setattr(build_source_worktree_manifest, "GIT_PATH", git_path)
    monkeypatch.setattr(
        build_source_worktree_manifest,
        "GIT_SHA256",
        hashlib.sha256(git_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(build_source_worktree_manifest, "GIT_RUNTIME_PATH", git_path)
    monkeypatch.setattr(
        build_source_worktree_manifest,
        "GIT_RUNTIME_SHA256",
        hashlib.sha256(git_path.read_bytes()).hexdigest(),
    )
    root = (tmp_path / "index-fixture").resolve()
    root.mkdir()
    subprocess.run([git, "init"], cwd=root, check=True, stdout=subprocess.PIPE)
    operator_root = root / "scripts" / "release_gpu_jit_lambda_operator"
    operator_root.mkdir(parents=True)
    preloader_path = operator_root / "preloader.py"
    preloader_path.write_bytes(b'SOURCE_MANIFEST_SHA256 = "' + b"0" * 64 + b'"\nVALUE = "staged"\n')
    manifest_path = operator_root / "source-worktree-manifest.json"
    manifest_path.write_bytes(b"{}\n")
    payload_path = root / "package" / "payload.py"
    payload_path.parent.mkdir()
    staged_payload = b"VALUE = 'staged-index'\n"
    payload_path.write_bytes(staged_payload)
    (root / ".gitignore").write_bytes(b"ignored-*\n")
    subprocess.run([git, "add", "--all"], cwd=root, check=True)

    payload_path.write_bytes(b"VALUE = 'dirty-worktree'\n")
    (root / "ignored-residue").write_bytes(b"must-not-enter-manifest")
    (root / "untracked-residue").write_bytes(b"must-not-enter-manifest")
    value = build_source_worktree_manifest.build(root)
    assert value["source"] == "exact-staged-index-blobs"
    assert value["runtime_git_dependency"] is False
    assert set(value["excluded_paths"]) == {
        build_source_worktree_manifest.MANIFEST_RELATIVE,
        build_source_worktree_manifest.PRELOADER_RELATIVE,
    }
    assert build_source_worktree_manifest.MANIFEST_RELATIVE not in value["files"]
    assert build_source_worktree_manifest.PRELOADER_RELATIVE not in value["files"]
    assert "ignored-residue" not in value["files"]
    assert "untracked-residue" not in value["files"]
    assert (
        value["files"]["package/payload.py"]["sha256"] == hashlib.sha256(staged_payload).hexdigest()
    )

    output = (tmp_path / "published-manifest.json").resolve()
    sealed = (tmp_path / "sealed-preloader.py").resolve()
    assert (
        build_source_worktree_manifest.main(
            [
                "--repository-root",
                str(root),
                "--output",
                str(output),
                "--sealed-preloader-output",
                str(sealed),
            ]
        )
        == 0
    )
    raw = output.read_bytes()
    assert raw == boundary.canonical_json(value)
    expected_slot = f'SOURCE_MANIFEST_SHA256 = "{hashlib.sha256(raw).hexdigest()}"\n'.encode(
        "ascii"
    )
    assert expected_slot in sealed.read_bytes()
    with pytest.raises(ValueError, match="source_manifest_output_rejected"):
        build_source_worktree_manifest.main(
            [
                "--repository-root",
                str(root),
                "--output",
                str(output),
                "--sealed-preloader-output",
                str(tmp_path / "second-preloader.py"),
            ]
        )


def test_preloader_full_source_inventory_rejects_every_non_git_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = (tmp_path / "source").resolve()
    root.mkdir()
    (root / ".git").mkdir()
    required = {
        "scripts/release_gpu_jit_lambda_operator/bootstrap.py",
        "scripts/release_gpu_jit_lambda_operator/__main__.py",
        "scripts/release_gpu_jit_lambda_operator/windows_launcher.py",
        "scripts/release_gpu_jit_lambda_operator/receipt_contract.py",
        "scripts/release_gpu_jit_lambda_controller/controller.py",
        "scripts/release_gpu_jit_lambda_controller/driver.py",
        "scripts/release_gpu_jit_lambda_live/adapter.py",
        "scripts/release_gpu_jit_lambda_runtime/runtime_contract.py",
        "scripts/release_external_controls.py",
        "scripts/verify_release_recovery.py",
        ".github/release-control-policy.json",
        ".github/workflows/cuda-ci.yml",
        ".github/workflows/publish-pypi.yml",
        ".github/workflows/recover-github-release.yml",
        "poetry.lock",
        "pyproject.toml",
    }
    shim_relative = "scripts/release_gpu_jit_lambda_operator/preloader_shim.py"
    required.add(shim_relative)
    files: dict[str, bytes] = {}
    for relative in sorted(required):
        if relative == shim_relative:
            raw = Path("scripts/release_gpu_jit_lambda_operator/preloader_shim.py").read_bytes()
        else:
            raw = f"fixture:{relative}\n".encode("utf-8")
        path = root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files[relative] = raw
    preloader_raw = b"fixture sealed preloader\n"
    preloader_path = root / operator_preloader.PRELOADER_RELATIVE
    preloader_path.parent.mkdir(parents=True, exist_ok=True)
    preloader_path.write_bytes(preloader_raw)
    directories: set[str] = set()
    for relative in files:
        parent = Path(relative).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    manifest_files = {
        relative: {
            "mode": "100644",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha": hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest(),
        }
        for relative, raw in sorted(files.items())
    }
    rows = [
        f"{relative}\t100644\t{item['bytes']}\t{item['sha256']}\t{item['git_blob_sha']}\n".encode(
            "utf-8"
        )
        for relative, item in sorted(manifest_files.items())
    ]
    manifest = {
        "schema_version": 1,
        "kind": "explainiverse-operator-source-worktree-manifest",
        "excluded_paths": [
            operator_preloader.SOURCE_MANIFEST_RELATIVE,
            operator_preloader.PRELOADER_RELATIVE,
        ],
        "files": manifest_files,
        "directories": sorted(directories),
        "file_count": len(manifest_files),
        "directory_count": len(directories),
        "file_inventory_sha256": hashlib.sha256(b"".join(rows)).hexdigest(),
        "source": "exact-staged-index-blobs",
        "runtime_git_dependency": False,
    }
    manifest_raw = boundary.canonical_json(manifest)
    (root / operator_preloader.SOURCE_MANIFEST_RELATIVE).write_bytes(manifest_raw)
    monkeypatch.setattr(
        operator_preloader, "SOURCE_MANIFEST_SHA256", hashlib.sha256(manifest_raw).hexdigest()
    )
    shim_receipt = {"preloader_sha256": hashlib.sha256(preloader_raw).hexdigest()}
    snapshot, captured = operator_preloader._source_snapshot(
        [], root=root, expected_head_sha="a" * 40, shim_receipt=shim_receipt
    )
    assert snapshot["runtime_git_dependency"] is False
    assert required.issubset(captured)
    (root / "ignored-or-untracked-residue.tmp").write_bytes(b"unexpected")
    with pytest.raises(SystemExit):
        operator_preloader._source_snapshot(
            [], root=root, expected_head_sha="a" * 40, shim_receipt=shim_receipt
        )
    assert "preloader_source_file_or_directory_set_rejected" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "nt", reason="Windows held-tree semantics are native")
def test_preloader_holds_verified_trees_without_write_or_delete_sharing(tmp_path: Path) -> None:
    root = (tmp_path / "held").resolve()
    root.mkdir()
    target = root / "target.bin"
    target.write_bytes(b"bound")
    replacement = root / "replacement.bin"
    replacement.write_bytes(b"replacement")
    held = operator_preloader._HeldWindowsTrees((root,))
    try:
        assert held.mapping["held_before_third_party_site_or_third_party_native_import"] is True
        assert held.mapping["write_share_allowed"] is False
        assert held.mapping["delete_share_allowed"] is False
        with pytest.raises(OSError):
            target.write_bytes(b"drift")
        with pytest.raises(OSError):
            os.replace(replacement, target)
    finally:
        held.close()
    target.write_bytes(b"after-close")
    assert target.read_bytes() == b"after-close"


def test_wheel_and_python_tree_verifiers_reject_drift_or_orphans(tmp_path: Path) -> None:
    site = (tmp_path / "site").resolve()
    site.mkdir()
    (site / "package.py").write_bytes(b"VALUE = 1\n")
    site_manifest = {
        "files": {
            "package.py": {
                "bytes": 10,
                "sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
            }
        },
        "directories": [],
    }
    assert operator_bootstrap.verify_site_tree(site, site_manifest)["file_count"] == 1
    (site / "sitecustomize.py").write_bytes(b"raise RuntimeError\n")
    with pytest.raises(operator_bootstrap.BootstrapError, match="site_file_set_or_bytes_rejected"):
        operator_bootstrap.verify_site_tree(site, site_manifest)

    runtime_root = (tmp_path / "python-runtime").resolve()
    runtime_root.mkdir()
    (runtime_root / "python.exe").write_bytes(b"pinned-fixture")
    python_manifest = {
        "files": {
            "python.exe": {
                "bytes": len(b"pinned-fixture"),
                "sha256": hashlib.sha256(b"pinned-fixture").hexdigest(),
            }
        }
    }
    assert operator_bootstrap.verify_python_tree(runtime_root, python_manifest)["file_count"] == 1
    (runtime_root / "orphan.dll").write_bytes(b"unowned")
    with pytest.raises(
        operator_bootstrap.BootstrapError, match="python_file_set_or_bytes_rejected"
    ):
        operator_bootstrap.verify_python_tree(runtime_root, python_manifest)


def test_setup_helpers_are_absolute_script_safe_and_reject_manifest_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path.cwd().resolve()
    for relative in (
        "scripts/release_gpu_jit_lambda_operator/install_windows_python.py",
        "scripts/release_gpu_jit_lambda_operator/install_windows_runtime.py",
    ):
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(root / relative), "--help"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")

    substituted = (tmp_path / "substituted.json").resolve()
    substituted.write_bytes(boundary.canonical_json({"schema_version": 1}))
    wheelhouse = (tmp_path / "wheelhouse").resolve()
    wheelhouse.mkdir()
    with pytest.raises(ValueError, match="runtime_manifest_digest_rejected"):
        install_windows_runtime.install(
            wheelhouse,
            substituted,
            (tmp_path / "runtime-output").resolve(),
        )

    archive = (tmp_path / install_windows_python.ARCHIVE_FILENAME).resolve()
    with zipfile.ZipFile(archive, "w") as fixture:
        fixture.writestr("python.exe", b"fixture")
    raw_archive = archive.read_bytes()
    monkeypatch.setattr(install_windows_python, "ARCHIVE_BYTES", len(raw_archive))
    monkeypatch.setattr(
        install_windows_python, "ARCHIVE_SHA256", hashlib.sha256(raw_archive).hexdigest()
    )
    with pytest.raises(ValueError, match="python_manifest_digest_rejected"):
        install_windows_python.install(
            archive,
            substituted,
            (tmp_path / "python-output").resolve(),
        )


def test_inspect_persists_canonical_no_replace_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = (tmp_path / "operator-inspection").resolve()
    inventory = {
        "schema_version": 1,
        "kind": boundary.INVENTORY_KIND,
        "repository": {"supplied_ref": "refs/heads/main"},
    }
    monkeypatch.setattr(operator, "capture_inventory", lambda **_: inventory)
    args = SimpleNamespace(
        action="inspect",
        phase="final-main",
        supplied_ref="refs/heads/main",
        expected_head_sha=HEAD,
        repository_root=str(Path.cwd().resolve()),
        operator_python_root=str(tmp_path.resolve()),
        operator_site_root=str(tmp_path.resolve()),
        git_executable=str(Path(sys.executable).resolve()),
        gh_executable=str(Path(sys.executable).resolve()),
        ssh_executable=str(Path(sys.executable).resolve()),
        inspection_evidence_directory=str(evidence),
    )
    result = operator.inspect(
        cast(Any, args),
        environment_receipt={"scrubbed": True},
        launch_receipt={"sealed": True},
    )
    receipt_path = Path(result["inspection_receipt"])
    raw = receipt_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result["inspection_receipt_sha256"]
    assert raw == boundary.canonical_json(json.loads(raw))
    assert result["crash_safe_no_replace"] is True
    with pytest.raises(boundary.OperatorError, match="already_exists"):
        operator.inspect(
            cast(Any, args),
            environment_receipt={"scrubbed": True},
            launch_receipt={"sealed": True},
        )


def test_immutable_plan_recovery_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    plan = _plan()
    evidence = (tmp_path / "evidence").resolve()
    receipt = create_evidence_directory(evidence)
    journal = EvidenceJournal(receipt, plan_sha256=plan.sha256)
    journal.record("immutable-plan", plan.to_mapping())
    rebuilt = boundary.read_recovery_plan(receipt, expected_plan_sha256=plan.sha256)
    assert rebuilt.to_mapping() == plan.to_mapping()
    tampered = plan.to_mapping()
    tampered["target"]["price_cents_per_hour"] += 1
    with pytest.raises((ContractError, boundary.OperatorError)):
        boundary.immutable_plan_from_mapping(tampered, expected_sha256=plan.sha256)
    journal.close()


def test_app_inbox_skips_stale_then_accepts_fresh_generation(tmp_path: Path) -> None:
    inbox_root = (tmp_path / "inbox").resolve()
    receipt = create_evidence_directory(inbox_root)
    try:
        _publish(
            receipt,
            tmp_path / "source-stale",
            captured_at=NOW - timedelta(minutes=11),
            generation=1,
            variant="stale",
        )
        fresh, _ = _publish(
            receipt,
            tmp_path / "source-fresh",
            captured_at=NOW,
            generation=2,
            variant="fresh",
        )
        inbox = boundary.AppCaptureInbox(
            receipt,
            _controller_resources(),
            phase="pull-request",
            poll_limit=2,
            poll_seconds=0,
            sleep=lambda _: None,
            clock=lambda: NOW,
        )
        archived: list[dict[str, Any]] = []
        _bind_stale_archive_fixture(inbox, archived)
        observed, reader = inbox()
        assert observed == fresh
        assert reader(fresh["evidence"][0]["filename"])
        assert inbox.to_public_mapping()["stale_generation_count"] == 1
        assert len(archived) == 1
        assert archived[0]["receipt"]["classified_at"] == NOW.isoformat()
        assert archived[0]["pages"]
    finally:
        receipt.close()


def test_app_inbox_rejects_regressing_classification_clock_across_generations(
    tmp_path: Path,
) -> None:
    receipt = create_evidence_directory((tmp_path / "inbox").resolve())
    try:
        _publish(
            receipt,
            tmp_path / "source-stale",
            captured_at=NOW - timedelta(minutes=11),
            generation=1,
            variant="stale-before-clock-rollback",
        )
        _publish(
            receipt,
            tmp_path / "source-fresh",
            captured_at=NOW - timedelta(seconds=2),
            generation=2,
            variant="fresh-after-clock-rollback",
        )
        classified_times = iter((NOW, NOW - timedelta(seconds=1)))
        inbox = boundary.AppCaptureInbox(
            receipt,
            _controller_resources(),
            phase="pull-request",
            poll_limit=2,
            poll_seconds=0,
            sleep=lambda _: None,
            clock=lambda: next(classified_times),
        )
        archived: list[dict[str, Any]] = []
        _bind_stale_archive_fixture(inbox, archived)
        with pytest.raises(
            boundary.OperatorError, match="app_capture_classification_time_regressed"
        ):
            inbox()
        assert inbox.to_public_mapping()["stale_generation_count"] == 1
        assert inbox.to_public_mapping()["accepted_capture_count"] == 0
        assert archived[0]["receipt"]["classified_at"] == NOW.isoformat()
    finally:
        receipt.close()


def test_app_inbox_rejects_classification_before_capture_within_provider_clock_skew(
    tmp_path: Path,
) -> None:
    receipt = create_evidence_directory((tmp_path / "inbox").resolve())
    try:
        _publish(
            receipt,
            tmp_path / "source-future",
            captured_at=NOW + timedelta(seconds=1),
            variant="future-within-controller-clock-skew",
        )
        inbox = boundary.AppCaptureInbox(
            receipt,
            _controller_resources(),
            phase="pull-request",
            poll_limit=1,
            poll_seconds=0,
            sleep=lambda _: None,
            clock=lambda: NOW,
        )
        _bind_stale_archive_fixture(inbox, [])
        with pytest.raises(boundary.OperatorError, match="app_capture_classified_before_capture"):
            inbox()
        assert inbox.to_public_mapping()["accepted_capture_count"] == 0
        assert inbox.to_public_mapping()["stale_generation_count"] == 0
    finally:
        receipt.close()


def test_app_inbox_never_advances_stale_generation_without_durable_raw_archive(
    tmp_path: Path,
) -> None:
    inbox_root = (tmp_path / "inbox").resolve()
    receipt = create_evidence_directory(inbox_root)
    try:
        _publish(
            receipt,
            tmp_path / "source-stale",
            captured_at=NOW - timedelta(minutes=11),
            variant="stale",
        )
        inbox = boundary.AppCaptureInbox(
            receipt,
            _controller_resources(),
            phase="pull-request",
            poll_limit=1,
            poll_seconds=0,
            sleep=lambda _: None,
            clock=lambda: NOW,
        )
        with pytest.raises(boundary.OperatorError, match="stale_archive_sink_missing"):
            inbox()
        assert inbox.to_public_mapping()["stale_generation_count"] == 0
    finally:
        receipt.close()


def test_app_inbox_archives_stale_raw_pages_through_real_evidence_journal(
    tmp_path: Path,
) -> None:
    inbox_receipt = create_evidence_directory((tmp_path / "inbox").resolve())
    evidence_receipt = create_evidence_directory((tmp_path / "evidence").resolve())
    journal = EvidenceJournal(evidence_receipt, plan_sha256=_plan().sha256)
    resources = _controller_resources()
    try:
        _publish(
            inbox_receipt,
            tmp_path / "source-stale",
            captured_at=NOW - timedelta(minutes=11),
            generation=1,
            variant="stale-real-archive",
        )
        fresh, _ = _publish(
            inbox_receipt,
            tmp_path / "source-fresh",
            captured_at=NOW,
            generation=2,
            variant="fresh-after-stale",
        )
        inbox = boundary.AppCaptureInbox(
            inbox_receipt,
            resources,
            phase="pull-request",
            poll_limit=2,
            poll_seconds=0,
            sleep=lambda _: None,
            clock=lambda: NOW,
        )
        inbox.bind_stale_archive_sink(
            lambda classified_at, generation, pages: (
                journal.archive_stale_installed_app_capture(
                    phase="pull-request",
                    classified_at=classified_at,
                    generation_receipt=generation,
                    evidence_pages=pages,
                    controller_resources=resources,
                )
            )
        )
        assert inbox()[0] == fresh
        consumed_generations = inbox._consumed_generations
        assert consumed_generations is not None
        stale_archive = consumed_generations[0]["stale_archive"]
        archive_root = Path(evidence_receipt.absolute_path) / stale_archive["archive_directory"]
        assert archive_root.is_dir()
        assert {
            child.name: hashlib.sha256(child.read_bytes()).hexdigest()
            for child in archive_root.iterdir()
        } == {item["filename"]: item["sha256"] for item in stale_archive["files"]}
        journal_entry = next(
            Path(evidence_receipt.absolute_path).glob("*-installed-app-stale-raw-archive.json")
        )
        envelope = json.loads(journal_entry.read_bytes())
        assert envelope["payload"] == stale_archive
    finally:
        journal.close()
        inbox_receipt.close()


def test_app_inbox_final_inventory_truthfully_binds_retained_accepted_sources(
    tmp_path: Path,
) -> None:
    inbox_root = (tmp_path / "inbox").resolve()
    receipt = create_evidence_directory(inbox_root)
    try:
        _publish(
            receipt,
            tmp_path / "source-stale",
            ordinal=1,
            generation=1,
            captured_at=NOW - timedelta(minutes=11),
            variant="stale",
        )
        first, _ = _publish(
            receipt,
            tmp_path / "source-first",
            ordinal=1,
            generation=2,
            captured_at=NOW,
            variant="first",
        )
        second, _ = _publish(
            receipt,
            tmp_path / "source-second",
            ordinal=2,
            captured_at=NOW,
            variant="second",
        )
        inbox = boundary.AppCaptureInbox(
            receipt,
            _controller_resources(),
            phase="pull-request",
            poll_limit=2,
            poll_seconds=0,
            sleep=lambda _: None,
            clock=lambda: NOW,
        )
        archived: list[dict[str, Any]] = []
        _bind_stale_archive_fixture(inbox, archived)
        assert inbox()[0] == first
        assert inbox()[0] == second
        final_inventory = inbox.validate_consumed()
        material = dict(final_inventory)
        evidence_sha256 = material.pop("evidence_sha256")
        assert hashlib.sha256(boundary.canonical_json(material)).hexdigest() == evidence_sha256
        assert final_inventory["accepted_generation_count"] == 2
        assert final_inventory["stale_generation_count"] == 1
        assert [
            (item["ordinal"], item["generation"], item["classification"])
            for item in final_inventory["consumed_generations"]
        ] == [(1, 1, "stale"), (1, 2, "accepted"), (2, 1, "accepted")]
        assert all(
            item["pages_inventory_sha256"]
            == hashlib.sha256(boundary.canonical_json(item["pages"])).hexdigest()
            for item in final_inventory["consumed_generations"]
        )
        assert "capture_evidence_sha256" not in final_inventory["consumed_generations"][0]
        assert final_inventory["consumed_generations"][0]["stale_archive"] == archived[0]["receipt"]
        assert all(
            item["capture_evidence_sha256"] for item in final_inventory["consumed_generations"][1:]
        )
        assert all(
            item["classified_at"] == NOW.isoformat()
            for item in final_inventory["consumed_generations"]
        )
        assert final_inventory["accepted_source_generations_retained"] is True
        assert final_inventory["unobserved_residue_present"] is False
        assert final_inventory["file_count"] == len(final_inventory["files"])
        assert final_inventory["directory_count"] == len(final_inventory["directories"])
        assert {path.name for path in inbox_root.iterdir()} == {
            "capture-01-000001",
            "ready-01-000001.json",
            "capture-01-000002",
            "ready-01-000002.json",
            "capture-02-000001",
            "ready-02-000001.json",
        }
        (inbox_root / "unexpected.txt").write_bytes(b"residue")
        with pytest.raises(boundary.OperatorError, match="final_inbox_inventory_drift"):
            inbox.validate_consumed()
    finally:
        receipt.close()


def test_app_capture_publisher_rejects_valid_but_noncanonical_source_order(
    tmp_path: Path,
) -> None:
    inbox_root = (tmp_path / "inbox").resolve()
    receipt = create_evidence_directory(inbox_root)
    capture, pages = _app_capture(NOW, variant="unsorted")
    capture["installations"] = list(reversed(capture["installations"]))
    capture["evidence"] = list(reversed(capture["evidence"]))
    staging_receipt = _write_capture_source(tmp_path / "source", capture, pages)
    try:
        with pytest.raises(boundary.OperatorError, match="app_capture_source_not_normalized"):
            boundary.publish_app_capture_generation(
                receipt,
                controller_resources=_controller_resources(),
                staging_receipt=staging_receipt,
                phase="pull-request",
                ordinal=1,
                generation=1,
                publication_nonce="d" * 32,
                now=NOW,
            )
        assert tuple(inbox_root.iterdir()) == ()
    finally:
        staging_receipt.close()
        receipt.close()


def test_app_inbox_rejects_stale_generation_replayed_under_new_nonce(tmp_path: Path) -> None:
    inbox_root = (tmp_path / "inbox").resolve()
    receipt = create_evidence_directory(inbox_root)
    try:
        capture, pages = _publish(
            receipt,
            tmp_path / "source-stale",
            captured_at=NOW - timedelta(minutes=11),
            generation=1,
            variant="stale",
        )
        duplicate_source = tmp_path / "source-duplicate"
        staging_receipt = _write_capture_source(duplicate_source, capture, pages)
        try:
            boundary.publish_app_capture_generation(
                receipt,
                controller_resources=_controller_resources(),
                staging_receipt=staging_receipt,
                phase="pull-request",
                ordinal=1,
                generation=2,
                publication_nonce="f" * 32,
                now=NOW - timedelta(minutes=11),
            )
        finally:
            staging_receipt.close()
        inbox = boundary.AppCaptureInbox(
            receipt,
            _controller_resources(),
            phase="pull-request",
            poll_limit=2,
            poll_seconds=0,
            sleep=lambda _: None,
            clock=lambda: NOW,
        )
        _bind_stale_archive_fixture(inbox)
        with pytest.raises(boundary.OperatorError, match="app_capture_json_replayed"):
            inbox()
    finally:
        receipt.close()


def test_app_inbox_rejects_raw_page_replay_across_distinct_capture(tmp_path: Path) -> None:
    inbox_root = (tmp_path / "inbox").resolve()
    receipt = create_evidence_directory(inbox_root)
    try:
        first, first_pages = _publish(
            receipt,
            tmp_path / "source-first",
            ordinal=1,
            generation=1,
            captured_at=NOW,
            variant="first",
        )
        second, second_pages = _app_capture(NOW, variant="second")
        reused_name = first["evidence"][0]["filename"]
        second_pages[reused_name] = first_pages[reused_name]
        second_item = next(item for item in second["evidence"] if item["filename"] == reused_name)
        second_item["bytes"] = len(second_pages[reused_name])
        second_item["sha256"] = hashlib.sha256(second_pages[reused_name]).hexdigest()
        staging_receipt = _write_capture_source(tmp_path / "source-second", second, second_pages)
        try:
            boundary.publish_app_capture_generation(
                receipt,
                controller_resources=_controller_resources(),
                staging_receipt=staging_receipt,
                phase="pull-request",
                ordinal=2,
                generation=1,
                publication_nonce="e" * 32,
                now=NOW,
            )
        finally:
            staging_receipt.close()
        inbox = boundary.AppCaptureInbox(
            receipt,
            _controller_resources(),
            phase="pull-request",
            poll_limit=1,
            poll_seconds=0,
            sleep=lambda _: None,
            clock=lambda: NOW,
        )
        inbox()
        with pytest.raises(boundary.OperatorError, match="app_capture_page_replayed"):
            inbox()
    finally:
        receipt.close()


def test_publisher_ready_marker_is_last_no_replace_and_crash_is_unready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox_root = (tmp_path / "inbox").resolve()
    receipt = create_evidence_directory(inbox_root)
    capture, pages = _app_capture(NOW, variant="crash")
    staging_receipt = _write_capture_source(tmp_path / "source", capture, pages)
    original = boundary._publish_no_replace

    def crash_before_ready(*_: Any, **__: Any) -> None:
        raise KeyboardInterrupt("simulated crash")

    monkeypatch.setattr(boundary, "_publish_no_replace", crash_before_ready)
    try:
        with pytest.raises(KeyboardInterrupt):
            boundary.publish_app_capture_generation(
                receipt,
                controller_resources=_controller_resources(),
                staging_receipt=staging_receipt,
                phase="pull-request",
                ordinal=1,
                generation=1,
                publication_nonce="1" * 32,
                now=NOW,
            )
        assert (inbox_root / "capture-01-000001").is_dir()
        assert not (inbox_root / "ready-01-000001.json").exists()
        inbox = boundary.AppCaptureInbox(
            receipt,
            _controller_resources(),
            phase="pull-request",
            poll_limit=1,
            poll_seconds=0,
            sleep=lambda _: None,
            clock=lambda: NOW,
        )
        with pytest.raises(boundary.OperatorError, match="fresh_app_capture_timeout"):
            inbox()
        monkeypatch.setattr(boundary, "_publish_no_replace", original)
        with pytest.raises(boundary.OperatorError, match="generation_already_exists"):
            boundary.publish_app_capture_generation(
                receipt,
                controller_resources=_controller_resources(),
                staging_receipt=staging_receipt,
                phase="pull-request",
                ordinal=1,
                generation=1,
                publication_nonce="2" * 32,
                now=NOW,
            )
        with pytest.raises(
            boundary.OperatorError,
            match="prior_generation_incomplete_requires_fresh_inbox",
        ):
            boundary.publish_app_capture_generation(
                receipt,
                controller_resources=_controller_resources(),
                staging_receipt=staging_receipt,
                phase="pull-request",
                ordinal=1,
                generation=2,
                publication_nonce="3" * 32,
                now=NOW,
            )
    finally:
        staging_receipt.close()
        receipt.close()


def test_app_publisher_requires_a_separate_secure_staging_receipt(tmp_path: Path) -> None:
    inbox_root = (tmp_path / "inbox").resolve()
    receipt = create_evidence_directory(inbox_root)
    try:
        with pytest.raises(boundary.OperatorError, match="not_disjoint"):
            boundary.publish_app_capture_generation(
                receipt,
                controller_resources=_controller_resources(),
                staging_receipt=receipt,
                phase="pull-request",
                ordinal=1,
                generation=1,
                publication_nonce="4" * 32,
                now=NOW,
            )
    finally:
        receipt.close()


@pytest.mark.parametrize(
    "context",
    (
        "evidence_directory",
        "app_capture_inbox",
        "app_capture_staging",
        "final_main_evidence_directory",
    ),
)
@pytest.mark.parametrize("direction", ("inside", "ancestor"))
def test_operator_security_roots_must_be_disjoint_from_repository_in_both_directions(
    tmp_path: Path, direction: str, context: str
) -> None:
    repository = (tmp_path / "outer" / "repository").resolve()
    repository.mkdir(parents=True)
    candidate = (
        (repository / "evidence").resolve()
        if direction == "inside"
        else repository.parent.resolve()
    )
    with pytest.raises(
        boundary.OperatorError,
        match=f"{context}_not_disjoint_from_repository",
    ):
        boundary.ensure_path_outside_repository(str(candidate), str(repository), context=context)


def test_app_publisher_rejects_staging_nested_under_inbox(tmp_path: Path) -> None:
    inbox_root = (tmp_path / "inbox").resolve()
    inbox_receipt = create_evidence_directory(inbox_root)
    staging_receipt = create_evidence_directory((inbox_root / "staging").resolve())
    try:
        with pytest.raises(boundary.OperatorError, match="not_disjoint"):
            boundary.publish_app_capture_generation(
                inbox_receipt,
                controller_resources=_controller_resources(),
                staging_receipt=staging_receipt,
                phase="pull-request",
                ordinal=1,
                generation=1,
                publication_nonce="6" * 32,
                now=NOW,
            )
    finally:
        staging_receipt.close()
        inbox_receipt.close()


def test_app_ready_publication_never_replaces_a_raced_marker(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    destination = root / "ready-01-000001.json"
    destination.write_bytes(b"raced-value\n")
    temporary = root / ".ready-pending"
    with pytest.raises(boundary.OperatorError, match="ready_already_exists"):
        boundary._publish_no_replace(temporary, destination, b"new-value\n")
    assert destination.read_bytes() == b"raced-value\n"
    assert not temporary.exists()


def test_app_capture_cli_creates_separate_secure_directories_and_publishes(
    tmp_path: Path,
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    inbox = (tmp_path / "inbox").resolve()
    staging = (tmp_path / "staging").resolve()
    common = {
        "repository_root": str(repository),
        "app_capture_inbox": str(inbox),
        "app_capture_staging": str(staging),
    }
    inbox_result = operator.create_app_inbox(
        cast(Any, SimpleNamespace(**common)),
        environment_receipt={"scrubbed": True},
        launch_receipt={"secure": True},
    )
    staging_result = operator.create_app_staging(
        cast(Any, SimpleNamespace(**common)),
        environment_receipt={"scrubbed": True},
        launch_receipt={"secure": True},
    )
    assert inbox_result["receipt"]["owner_private"] is True
    assert staging_result["receipt"]["owner_private"] is True
    assert inbox_result["receipt"]["receipt_sha256"] != staging_result["receipt"]["receipt_sha256"]
    capture, pages = _app_capture(datetime.now(timezone.utc), variant="cli")
    pages_root = staging / "pages"
    pages_root.mkdir()
    (staging / "capture.json").write_bytes(boundary.canonical_json(capture))
    for filename, raw in pages.items():
        (pages_root / filename).write_bytes(raw)
    publish_args = SimpleNamespace(
        **common,
        phase="pull-request",
        app_capture_inbox_receipt_sha256=inbox_result["receipt"]["receipt_sha256"],
        app_capture_staging_receipt_sha256=staging_result["receipt"]["receipt_sha256"],
        capture_ordinal=1,
        capture_generation=1,
        capture_publication_nonce="5" * 32,
    )
    published = operator.publish_app_capture(
        cast(Any, publish_args),
        environment_receipt={"scrubbed": True},
        launch_receipt={"secure": True},
        resources=_operator_resources(),
    )
    assert published["ready_marker_published_last"] is True
    assert published["staging_directory_separate"] is True
    assert (inbox / published["ready_marker"]).is_file()


@pytest.mark.skipif(
    os.name != "nt"
    or not all(
        Path(str(item["absolute_path"])).is_file()
        for item in boundary.PINNED_WINDOWS_EXECUTABLES.values()
    ),
    reason="requires the exact reviewed Windows release host toolchain",
)
def test_operator_toolchain_is_exact_path_byte_version_owner_and_signer_pinned() -> None:
    expected = boundary.PINNED_WINDOWS_EXECUTABLES
    observed = boundary.executable_inventory(
        git_executable=str(expected["git"]["absolute_path"]),
        gh_executable=str(expected["gh"]["absolute_path"]),
        ssh_executable=str(expected["ssh"]["absolute_path"]),
    )
    for name in ("git", "gh", "ssh"):
        assert observed[name]["pinned_reviewed_identity"] is True
        assert observed[name]["sha256"] == expected[name]["sha256"]
        assert observed[name]["version"] == expected[name]["version"]
        assert observed[name]["acl"]["owner_sid"] == expected[name]["owner_sid"]
        assert observed[name]["acl"]["unprivileged_write_ace_present"] is False
        assert observed[name]["authenticode"] == {
            "status": "Valid",
            "subject": expected[name]["authenticode_subject"],
            "thumbprint": expected[name]["authenticode_thumbprint"],
        }
    assert (
        observed["git"]["resolved_runtime"]["absolute_path"]
        == expected["git"]["runtime_absolute_path"]
    )
    assert observed["git"]["resolved_runtime"]["sha256"] == expected["git"]["runtime_sha256"]
    alternate_git = Path(r"C:\Program Files\Git\mingw64\bin\git.exe")
    if alternate_git.is_file():
        with pytest.raises(boundary.OperatorError, match="git_pinned_identity_drift"):
            boundary.executable_inventory(
                git_executable=str(alternate_git.resolve(strict=True)),
                gh_executable=str(expected["gh"]["absolute_path"]),
                ssh_executable=str(expected["ssh"]["absolute_path"]),
            )


@pytest.mark.skipif(os.name != "nt", reason="Windows owner-private ACL is native")
def test_preloader_validates_owner_private_receipt_root_before_third_party_import(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "private-root").resolve()
    receipt = create_evidence_directory(root)
    try:
        mapping = operator_preloader._windows_owner_private_acl(
            root, context="fixture_private_root"
        )
        assert mapping["inheritance_protected"] is True
        assert mapping["ace_count"] == 3
        assert mapping["validated_before_third_party_site_or_third_party_native_import"] is True
        assert mapping["pinned_stdlib_native_modules_loaded_before_hold"] is True
    finally:
        receipt.close()


def test_repository_binding_requires_clean_tree_exact_origin_and_live_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = shutil.which("git")
    assert git is not None
    root = (tmp_path / "clean-worktree").resolve()
    root.mkdir()
    subprocess.run([git, "init"], cwd=root, check=True, stdout=subprocess.PIPE)
    subprocess.run([git, "config", "user.email", "fixture@example.invalid"], cwd=root, check=True)
    subprocess.run([git, "config", "user.name", "Fixture"], cwd=root, check=True)
    subprocess.run([git, "config", "core.autocrlf", "false"], cwd=root, check=True)
    subprocess.run(
        [git, "remote", "add", "origin", boundary.EXPECTED_ORIGIN_URL], cwd=root, check=True
    )
    source = root / "source.py"
    source.write_bytes(b"VALUE = 1\n")
    subprocess.run([git, "add", "source.py"], cwd=root, check=True)
    subprocess.run([git, "commit", "-m", "fixture"], cwd=root, check=True, stdout=subprocess.PIPE)
    head = (
        subprocess.run([git, "rev-parse", "HEAD"], cwd=root, check=True, stdout=subprocess.PIPE)
        .stdout.decode("ascii")
        .strip()
    )
    fsmonitor_sentinel = (tmp_path / "ambient-fsmonitor-ran").resolve()
    if os.name == "nt":
        fsmonitor = (tmp_path / "ambient-fsmonitor.sh").resolve()
        fsmonitor.write_text(
            f"#!/bin/sh\nprintf invoked > '{fsmonitor_sentinel.as_posix()}'\nexit 0\n",
            encoding="utf-8",
        )
    else:
        fsmonitor = (tmp_path / "ambient-fsmonitor.sh").resolve()
        fsmonitor.write_text(
            f"#!/bin/sh\nprintf invoked > '{fsmonitor_sentinel}'\nexit 0\n",
            encoding="ascii",
        )
        fsmonitor.chmod(0o700)
    subprocess.run([git, "config", "core.fsmonitor", fsmonitor.as_posix()], cwd=root, check=True)
    subprocess.run(
        [git, "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert fsmonitor_sentinel.is_file(), "fixture must prove ambient fsmonitor is executable"
    fsmonitor_sentinel.unlink()
    hostile_config = (tmp_path / "ambient-gitconfig").resolve()
    hostile_config.write_text(f"[core]\n\tfsmonitor = {fsmonitor.as_posix()}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(hostile_config))
    monkeypatch.setattr(boundary, "CRITICAL_SOURCE_PATHS", ("source.py",))

    class GitHub:
        def __init__(self, remote_sha: str = head) -> None:
            self.remote_sha = remote_sha

        def request(self, method: str, path: str, body: Any = None) -> GitHubResponse:
            value = {
                "ref": "refs/heads/main",
                "object": {"type": "commit", "sha": self.remote_sha},
            }
            return GitHubResponse(
                method,
                path,
                200,
                bytearray(json.dumps(value).encode("ascii")),
                "f" * 64,
            )

    observed = boundary.repository_inventory(
        repository_root=str(root),
        git_executable=str(Path(git).resolve()),
        github=GitHub(),  # type: ignore[arg-type]
        expected_head_sha=head,
        supplied_ref="refs/heads/main",
    )
    assert observed["origin_url"] == boundary.EXPECTED_ORIGIN_URL
    assert observed["git_configuration"]["repository_fsmonitor_overridden_false"] is True
    assert not fsmonitor_sentinel.exists()
    with pytest.raises(boundary.OperatorError, match="local_head_sha_drift"):
        boundary.repository_inventory(
            repository_root=str(root),
            git_executable=str(Path(git).resolve()),
            github=GitHub(),  # type: ignore[arg-type]
            expected_head_sha="b" * 40,
            supplied_ref="refs/heads/main",
        )
    with pytest.raises(boundary.OperatorError, match="remote_head_sha_drift"):
        boundary.repository_inventory(
            repository_root=str(root),
            git_executable=str(Path(git).resolve()),
            github=GitHub("b" * 40),  # type: ignore[arg-type]
            expected_head_sha=head,
            supplied_ref="refs/heads/main",
        )
    subprocess.run([git, "update-index", "--skip-worktree", "source.py"], cwd=root, check=True)
    try:
        with pytest.raises(boundary.OperatorError, match="index_nonordinary_flag_rejected"):
            boundary.repository_inventory(
                repository_root=str(root),
                git_executable=str(Path(git).resolve()),
                github=GitHub(),  # type: ignore[arg-type]
                expected_head_sha=head,
                supplied_ref="refs/heads/main",
            )
    finally:
        subprocess.run(
            [git, "update-index", "--no-skip-worktree", "source.py"], cwd=root, check=True
        )
    (root / "untracked.txt").write_text("drift", encoding="ascii")
    with pytest.raises(boundary.OperatorError, match="worktree_not_clean"):
        boundary.repository_inventory(
            repository_root=str(root),
            git_executable=str(Path(git).resolve()),
            github=GitHub(),  # type: ignore[arg-type]
            expected_head_sha=head,
            supplied_ref="refs/heads/main",
        )


def test_module_resolution_rejects_shadow_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shadow = (tmp_path / "cryptography.py").resolve()
    shadow.write_text("# shadow\n", encoding="ascii")
    monkeypatch.setattr(
        boundary.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(shadow), submodule_search_locations=None),
    )
    with pytest.raises(boundary.OperatorError, match="outside_distribution"):
        boundary._module_resolution(
            "cryptography",
            "cryptography",
            site_root=tmp_path.resolve(),
            manifest={
                "archives": [
                    {
                        "distribution": "cryptography",
                        "filename": "cryptography.whl",
                    }
                ],
                "files": {},
            },
        )


def test_publication_acceptance_can_only_come_from_closed_journal_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    evidence = (tmp_path / "final-evidence").resolve()
    evidence.mkdir()
    fake_receipt = SimpleNamespace(close=lambda: None)
    accepted = SimpleNamespace(head_sha=HEAD)
    observed: dict[str, Any] = {}
    monkeypatch.setattr(operator, "reopen_evidence_directory", lambda *a, **k: fake_receipt)

    def load(receipt: Any, **kwargs: Any) -> Any:
        observed.update(kwargs)
        assert receipt is fake_receipt
        return accepted

    monkeypatch.setattr(operator.EvidenceJournal, "load_final_main_acceptance", load)
    args = SimpleNamespace(
        final_main_evidence_directory=str(evidence),
        final_main_evidence_receipt_sha256="b" * 64,
        final_main_plan_sha256="c" * 64,
        final_main_journal_sha256="d" * 64,
    )
    resources = _operator_resources()
    assert (
        operator._publication_acceptance(
            cast(Any, args),
            repository_root=str(repository),
            expected_head_sha=HEAD,
            resources=resources,
        )
        is accepted
    )
    assert observed["final_control_plane_plan_sha256"] == "c" * 64
    assert observed["final_journal_sha256"] == "d" * 64
    assert observed["controller_resources"] is resources.controller


def test_publication_rejects_new_evidence_nested_under_final_main_before_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    final_main = (tmp_path / "final-main").resolve()
    final_main.mkdir()
    inbox = (tmp_path / "inbox").resolve()
    inbox.mkdir()
    nested_new_evidence = (final_main / "publication").resolve()
    monkeypatch.setattr(operator, "_phase_inputs", lambda _: ("publication", "tag", HEAD))
    monkeypatch.setattr(
        operator,
        "_common_paths",
        lambda _: (str(repository), "git", "gh", "ssh"),
    )
    monkeypatch.setattr(
        operator,
        "_load_and_revalidate_inventory",
        lambda *a, **k: ({}, {}, "a" * 64),
    )
    monkeypatch.setattr(operator, "_validate_runtime_root", lambda *a, **k: repository)
    monkeypatch.setattr(
        operator,
        "LiveGates",
        lambda *a, **k: pytest.fail("gates constructed before directory disjointness"),
    )
    args = SimpleNamespace(
        runtime_root=str(repository),
        evidence_directory=str(nested_new_evidence),
        app_capture_inbox=str(inbox),
        final_main_evidence_directory=str(final_main),
    )
    with pytest.raises(boundary.OperatorError, match="evidence_directory_not_disjoint"):
        operator.execute(
            cast(Any, args),
            environment_receipt={"scrubbed": True},
            launch_receipt={"windows_handle_transport": True, "inherited_handle_count": 2},
            resources=_operator_resources(),
        )


@pytest.mark.parametrize("failure_mode", ("success", "run-failure", "abort-failure"))
def test_execute_orders_confirmation_before_gates_and_always_restores_on_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    events: list[str] = []
    # The receipt contract has its own exhaustive producer/consumer tests;
    # this fixture isolates lifecycle ordering and cleanup.
    monkeypatch.setattr(operator, "validate_operator_preflight", lambda *a, **k: {})
    resources = _operator_resources()
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    inbox_path = (tmp_path / "inbox").resolve()
    inbox_path.mkdir()
    evidence_path = (tmp_path / "evidence").resolve()
    runtime_root = (repository / "runtime").resolve()
    runtime_root.mkdir()
    key_path = (tmp_path / "access-key").resolve()
    key_path.write_text("fixture", encoding="ascii")

    class Receipt:
        def __init__(self, path: Path, digest: str) -> None:
            self.absolute_path = str(path)
            self.receipt_sha256 = digest
            self.closed = False

        def validate(self) -> Mapping[str, Any]:
            events.append("receipt-validate")
            return {"owner_private": True}

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append("receipt-close")

    inbox_receipt = Receipt(inbox_path, "1" * 64)
    evidence_receipt = Receipt(evidence_path, "2" * 64)

    class Inbox:
        phase = "pull-request"

        def __init__(self, receipt: Receipt, *_: Any, **__: Any) -> None:
            self.receipt = receipt
            self.calls = 0

        def __call__(self) -> tuple[Mapping[str, Any], Any]:
            self.calls += 1
            events.append(f"capture-{self.calls}")
            return {"capture": self.calls}, lambda _: b"page"

        def bind_stale_archive_sink(self, sink: Any) -> None:
            assert callable(sink)
            events.append("stale-archive-bound")

        def validate_consumed(self) -> None:
            events.append("inbox-consumed")
            assert self.calls == 2

        def to_public_mapping(self) -> dict[str, Any]:
            return {"accepted_capture_count": self.calls}

        def close(self) -> None:
            events.append("inbox-close")
            self.receipt.close()

    class Client:
        def close(self) -> None:
            events.append("client-close")

    client = Client()

    class Discovery:
        def to_public_mapping(self) -> dict[str, Any]:
            return {"fresh": True}

    class Identity:
        destroyed = False

        def destroy(self) -> None:
            self.destroyed = True
            events.append("identity-destroy")

    identity = Identity()

    class Access:
        closed = False

        def close(self) -> None:
            self.closed = True
            events.append("access-close")

    access = Access()
    plan = SimpleNamespace(
        sha256="3" * 64,
        head_sha=HEAD,
        lifecycle_nonce="4" * 32,
        ssh_public_key_sha256="5" * 64,
        to_mapping=lambda: {"head_sha": HEAD},
    )

    class Journal:
        def __init__(self, receipt: Receipt, *, plan_sha256: str) -> None:
            assert receipt is evidence_receipt and plan_sha256 == plan.sha256
            self.receipt = receipt
            self.closed = False
            events.append("journal-create")

        def record(self, label: str, _: Mapping[str, Any]) -> str:
            events.append(label)
            return "6" * 64

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append("journal-close")
                self.receipt.close()

    class Driver:
        def __init__(self, *positional: Any, **keyword: Any) -> None:
            assert positional[4] is resources.runtime_bundle
            self.identity = positional[3]
            self.journal = positional[5]
            self.access = keyword["access_identity"]
            self.state = "created"
            events.append("driver-create")

        def _close(self) -> None:
            self.journal.close()
            self.access.close()
            self.identity.destroy()

        def provision(self) -> None:
            self.state = "provisioned"
            events.append("provision")

        def run_phase(self, _: str, *, app_capture_supplier: Any, **__: Any) -> Any:
            events.append("run-phase")
            app_capture_supplier()
            app_capture_supplier()
            if failure_mode != "success":
                events.append("base-exception")
                raise KeyboardInterrupt("fixture")
            return SimpleNamespace(
                head_sha=HEAD,
                run_id=811,
                final_evidence_sha256="7" * 64,
            )

        def teardown(self) -> str:
            events.append("teardown")
            self.state = "restored"
            self._close()
            return "8" * 64

        def abort(self) -> str:
            events.append("abort")
            if failure_mode == "abort-failure":
                raise ControllerError("fixture-abort-before-close")
            self.state = "restored"
            self._close()
            return "9" * 64

    inventory: dict[str, Any] = {
        "executables": {
            name: {"absolute_path": name, "sha256": name * 4}
            for name in ("git", "gh", "ssh", "python")
        },
        "repository": _repository_inventory_record(),
    }
    monkeypatch.setattr(
        operator, "_phase_inputs", lambda _: ("pull-request", "refs/heads/candidate", HEAD)
    )
    monkeypatch.setattr(
        operator,
        "_common_paths",
        lambda _: (str(repository), "git", "gh", "ssh"),
    )
    monkeypatch.setattr(
        operator,
        "_load_and_revalidate_inventory",
        lambda *a, **k: ({}, inventory, "a" * 64),
    )
    monkeypatch.setattr(operator, "_validate_runtime_root", lambda *a, **k: runtime_root)
    monkeypatch.setattr(
        operator, "ensure_path_outside_repository", lambda value, *a, **k: Path(value)
    )
    monkeypatch.setattr(operator, "canonical_existing_file", lambda value, **k: Path(value))
    monkeypatch.setattr(
        operator,
        "validate_anonymous_fd",
        lambda *a, **k: {"regular_file": False, "anonymous": True},
    )
    monkeypatch.setattr(
        operator, "close_owned_fd", lambda value: events.append(f"fd-close-{value}")
    )
    monkeypatch.setattr(operator, "reopen_evidence_directory", lambda *a, **k: inbox_receipt)
    monkeypatch.setattr(operator, "AppCaptureInbox", Inbox)
    monkeypatch.setattr(
        operator.LambdaHttpClient,
        "from_secret_fd",
        staticmethod(lambda _: _append_and_return(events, "secret-read", client)),
    )
    monkeypatch.setattr(
        operator,
        "capture_action_time_discovery",
        lambda *a, **k: _append_and_return(events, "discovery", Discovery()),
    )
    monkeypatch.setattr(operator, "generate_ephemeral_host_identity", lambda: identity)
    monkeypatch.setattr(
        operator,
        "build_plan_from_discovery",
        lambda *a, **k: _append_and_return(events, "plan-build", plan),
    )
    monkeypatch.setattr(operator, "_emit", lambda _: events.append("plan-emit"))
    monkeypatch.setattr(
        operator,
        "read_plan_confirmation",
        lambda *a, **k: _append_and_return(events, "confirmation", {"confirmed": True}),
    )
    monkeypatch.setattr(
        operator,
        "_revalidate_locked_posture",
        lambda *a, **k: events.append("posture-revalidate"),
    )
    monkeypatch.setattr(
        operator,
        "LiveGates",
        lambda *a: _append_and_return(events, "gates", SimpleNamespace()),
    )
    monkeypatch.setattr(operator, "LambdaLiveAdapter", lambda *a: SimpleNamespace())
    monkeypatch.setattr(
        operator,
        "create_evidence_directory",
        lambda _: _append_and_return(events, "evidence-create", evidence_receipt),
    )
    monkeypatch.setattr(
        operator,
        "_write_operator_preflight",
        lambda *a, **k: _append_and_return(events, "preflight-write", ("preflight.json", "b" * 64)),
    )
    monkeypatch.setattr(operator, "EvidenceJournal", Journal)
    monkeypatch.setattr(operator, "capture_access_identity", lambda *a, **k: access)
    monkeypatch.setattr(operator, "_locked_github", lambda _: object())
    monkeypatch.setattr(operator, "_locked_ssh", lambda *a: object())
    monkeypatch.setattr(operator, "ReleaseGpuController", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(operator, "LiveReleaseDriver", Driver)
    args = SimpleNamespace(
        runtime_root=str(runtime_root),
        evidence_directory=str(evidence_path),
        app_capture_inbox=str(inbox_path),
        app_capture_inbox_receipt_sha256=inbox_receipt.receipt_sha256,
        ssh_access_key=str(key_path),
        ssh_key_name="fixture-key",
        image_id="fixture-image",
        controller_public_ipv4_cidr="8.8.8.8/32",
        lifecycle_nonce="c" * 32,
        plan_lifetime_seconds=1800,
        lambda_api_key_fd=101,
        plan_confirmation_fd=102,
        prior_accepted_cuda_runner_nonce=None,
        preflight_run_id=None,
        cuda_run_id=None,
        app_capture_poll_limit=2,
        app_capture_poll_seconds=0,
        observation_poll_limit=2,
        dispatch_poll_limit=None,
    )
    launch = {"windows_handle_transport": True, "inherited_handle_count": 2}
    if failure_mode == "run-failure":
        with pytest.raises(KeyboardInterrupt, match="fixture"):
            operator.execute(
                cast(Any, args),
                environment_receipt={"scrubbed": True},
                launch_receipt=launch,
                resources=resources,
            )
        assert "abort" in events and "teardown" not in events
        assert events.index("base-exception") < events.index("abort") < events.index("client-close")
    elif failure_mode == "abort-failure":
        with pytest.raises(operator.OperatorError, match="operator_abort_failed") as caught:
            operator.execute(
                cast(Any, args),
                environment_receipt={"scrubbed": True},
                launch_receipt=launch,
                resources=resources,
            )
        assert isinstance(caught.value.__cause__, ControllerError)
        assert str(caught.value.__cause__) == "fixture-abort-before-close"
        assert events.index("abort") < events.index("client-close") < events.index("journal-close")
    else:
        result = operator.execute(
            cast(Any, args),
            environment_receipt={"scrubbed": True},
            launch_receipt=launch,
            resources=resources,
        )
        assert result["provider_and_github_restored"] is True
        assert "teardown" in events and "abort" not in events
    assert events.index("confirmation") < events.index("posture-revalidate") < events.index("gates")
    assert events.index("gates") < events.index("evidence-create") < events.index("provision")
    assert events.index("operator-preflight-binding") < events.index("run-phase")
    assert events.index("client-close") < events.index("inbox-close")
    assert identity.destroyed is True and access.closed is True and evidence_receipt.closed is True


def test_production_cli_uses_only_preloader_captured_runtime_bytes() -> None:
    source = Path(operator.__file__).read_text(encoding="utf-8")
    assert "load_runtime_bundle(" not in source
    assert "runtime_bundle_from_captured_files(" in source
    assert "resources.runtime_bundle" in source


def test_preloader_source_receipt_archives_reconstructable_canonical_manifest() -> None:
    source = Path(operator_preloader.__file__).read_text(encoding="utf-8")
    assert '"source_manifest": manifest' in source
    assert '"source_manifest_sha256": SOURCE_MANIFEST_SHA256' in source
    assert '"source_manifest_inventory_sha256": manifest["file_inventory_sha256"]' in source


def test_operator_preflight_archives_exact_full_executable_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operator, "validate_operator_preflight", lambda *a, **k: {})
    executables = {
        name: {
            "absolute_path": f"C:\\fixture\\{name}.exe",
            "sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
            "posture": {"fixture": name},
        }
        for name in ("git", "gh", "ssh", "python")
    }
    repository = _repository_inventory_record()
    inventory = {"executables": executables, "repository": repository}
    value = operator._preflight_mapping(
        plan=SimpleNamespace(
            sha256="a" * 64,
            head_sha=HEAD,
            lifecycle_nonce="b" * 32,
            to_mapping=lambda: {},
        ),
        discovery=SimpleNamespace(to_public_mapping=lambda: {"fresh": True}),
        inspection_receipt_sha256="c" * 64,
        inventory=inventory,
        environment_receipt={"scrubbed": True},
        launch_receipt={"sealed": True},
        lambda_fd_receipt={"anonymous": True},
        confirmation_receipt={"confirmed": True},
        app_inbox=cast(
            Any,
            SimpleNamespace(phase="final-main", to_public_mapping=lambda: {"held": True}),
        ),
        final_acceptance=None,
        resources=_operator_resources(),
    )
    assert value["executables"] == executables
    assert value["executables"] is not executables
    assert value["repository"] == repository
    assert value["repository"] is not repository
    assert value["repository"]["git_configuration"] == repository["git_configuration"]
    assert value["inventory"] == inventory
    assert value["inventory"] is not inventory
    assert (
        value["inventory_sha256"]
        == hashlib.sha256(boundary.canonical_json(value["inventory"])).hexdigest()
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda repository: repository.pop("git_configuration"),
            "operator_preflight_repository_inventory_rejected",
        ),
        (
            lambda repository: repository["git_configuration"].update(
                repository_fsmonitor_overridden_false=False
            ),
            "operator_preflight_git_configuration_rejected",
        ),
        (
            lambda repository: repository["git_configuration"].update(
                system_config_path="/dev/null",
                global_config_path="/dev/null",
            ),
            "operator_preflight_git_configuration_rejected",
        ),
        (
            lambda repository: repository.update(remote_target_sha="f" * 40),
            "operator_preflight_repository_binding_rejected",
        ),
    ),
)
def test_operator_preflight_rejects_incomplete_or_unbound_repository_security(
    mutation: Any, error: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operator, "validate_operator_preflight", lambda *a, **k: {})
    repository = _repository_inventory_record()
    mutation(repository)
    inventory = {
        "executables": {
            name: {
                "absolute_path": f"C:\\fixture\\{name}.exe",
                "sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
                "posture": {"fixture": name},
            }
            for name in ("git", "gh", "ssh", "python")
        },
        "repository": repository,
    }
    with pytest.raises(boundary.OperatorError, match=error):
        operator._preflight_mapping(
            plan=SimpleNamespace(
                sha256="a" * 64,
                head_sha=HEAD,
                lifecycle_nonce="b" * 32,
                to_mapping=lambda: {},
            ),
            discovery=SimpleNamespace(to_public_mapping=lambda: {"fresh": True}),
            inspection_receipt_sha256="c" * 64,
            inventory=inventory,
            environment_receipt={"scrubbed": True},
            launch_receipt={"sealed": True},
            lambda_fd_receipt={"anonymous": True},
            confirmation_receipt={"confirmed": True},
            app_inbox=cast(
                Any,
                SimpleNamespace(phase="final-main", to_public_mapping=lambda: {"held": True}),
            ),
            final_acceptance=None,
            resources=_operator_resources(),
        )


def test_recovery_preflight_reconstructs_full_inventory_and_security_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operator, "validate_operator_preflight", lambda *a, **k: {})
    executables = {
        name: {
            "absolute_path": f"C:\\fixture\\{name}.exe",
            "sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
            "posture": {"fixture": name},
        }
        for name in ("git", "gh", "ssh", "python")
    }
    inventory = {"executables": executables, "repository": _repository_inventory_record()}
    plan_sha256 = "a" * 64
    inspection_sha256 = "c" * 64
    value = operator._preflight_mapping(
        plan=SimpleNamespace(
            sha256=plan_sha256,
            head_sha=HEAD,
            lifecycle_nonce="b" * 32,
            to_mapping=lambda: {},
        ),
        discovery=SimpleNamespace(to_public_mapping=lambda: {"fresh": True}),
        inspection_receipt_sha256=inspection_sha256,
        inventory=inventory,
        environment_receipt={"scrubbed": True},
        launch_receipt={"sealed": True},
        lambda_fd_receipt={"anonymous": True},
        confirmation_receipt={"confirmed": True},
        app_inbox=cast(
            Any,
            SimpleNamespace(phase="final-main", to_public_mapping=lambda: {"held": True}),
        ),
        final_acceptance=None,
        resources=_operator_resources(),
    )

    def publish(root: Path, payload: Mapping[str, Any]) -> Any:
        root.mkdir()
        raw = boundary.canonical_json(payload)
        digest = hashlib.sha256(raw).hexdigest()
        (root / f"operator-preflight-{digest}.json").write_bytes(raw)
        return SimpleNamespace(absolute_path=str(root)), digest

    receipt, digest = publish(tmp_path / "accepted", value)
    assert operator._validate_recovery_preflight(
        receipt,
        plan_sha256=plan_sha256,
        inventory=inventory,
        inspection_receipt_sha256=inspection_sha256,
    ) == {"filename": f"operator-preflight-{digest}.json", "sha256": digest}

    tampered = deepcopy(value)
    tampered["repository"]["git_configuration"]["repository_fsmonitor_overridden_false"] = False
    tampered_receipt, _ = publish(tmp_path / "tampered", tampered)
    with pytest.raises(
        boundary.OperatorError, match="recovery_operator_preflight_security_binding_rejected"
    ):
        operator._validate_recovery_preflight(
            tampered_receipt,
            plan_sha256=plan_sha256,
            inventory=inventory,
            inspection_receipt_sha256=inspection_sha256,
        )


def test_recovery_journal_barrier_archives_pending_repair_before_strict_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    pending = {
        "schema_version": 1,
        "kind": "explainiverse-journal-publish-recovery",
        "sidecar_filename": "journal-publish-recovery-fixture.json",
        "recovered_entries": [{"sequence": 17}],
    }
    record_sha256 = "7" * 64

    class Receipt:
        def __init__(self, ordinal: int) -> None:
            self.ordinal = ordinal
            self.closed = False

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append(f"receipt-close-{self.ordinal}")

    class Journal:
        def __init__(self, receipt: Receipt, ordinal: int) -> None:
            self.receipt = receipt
            self.ordinal = ordinal
            self.closed = False
            self._pending = dict(pending) if ordinal == 1 else None

        @property
        def interrupted_publish_recovery(self) -> dict[str, Any] | None:
            events.append(f"pending-read-{self.ordinal}")
            return deepcopy(self._pending)

        def record_interrupted_publish_recovery(self) -> str:
            events.append("record-interrupted-publish-recovery")
            assert self.ordinal == 1 and self._pending == pending
            self._pending = None
            return record_sha256

        @property
        def last_evidence_sha256(self) -> str:
            events.append(f"tail-read-{self.ordinal}")
            return record_sha256

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append(f"journal-close-{self.ordinal}")
                self.receipt.close()

    receipts = [Receipt(1), Receipt(2)]
    journals = [Journal(receipts[0], 1), Journal(receipts[1], 2)]

    def reopen_receipt(*_: Any, **__: Any) -> Receipt:
        receipt = receipts.pop(0)
        events.append(f"receipt-open-{receipt.ordinal}")
        return receipt

    def reopen_journal(receipt: Receipt, **_: Any) -> Journal:
        journal = journals.pop(0)
        assert journal.receipt is receipt
        events.append(f"journal-open-{journal.ordinal}")
        return journal

    monkeypatch.setattr(operator, "reopen_evidence_directory", reopen_receipt)
    monkeypatch.setattr(operator.EvidenceJournal, "reopen_for_recovery", reopen_journal)

    receipt, journal, recovered, recorded, tail = operator._reopen_recovery_journal_barrier(
        tmp_path / "evidence",
        receipt_sha256="a" * 64,
        plan_sha256="b" * 64,
    )
    try:
        assert recovered == pending
        assert recorded == record_sha256
        assert tail == record_sha256
        assert receipt.ordinal == 2 and journal.ordinal == 2
        assert events == [
            "receipt-open-1",
            "journal-open-1",
            "pending-read-1",
            "record-interrupted-publish-recovery",
            "tail-read-1",
            "journal-close-1",
            "receipt-close-1",
            "receipt-open-2",
            "journal-open-2",
            "pending-read-2",
            "tail-read-2",
        ]
    finally:
        journal.close()


def test_recovery_journal_barrier_reopens_when_recursive_record_is_already_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    tail_sha256 = "8" * 64

    class Receipt:
        def __init__(self, ordinal: int) -> None:
            self.ordinal = ordinal
            self.closed = False

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append(f"receipt-close-{self.ordinal}")

    class Journal:
        def __init__(self, receipt: Receipt, ordinal: int) -> None:
            self.receipt = receipt
            self.ordinal = ordinal
            self.closed = False

        @property
        def interrupted_publish_recovery(self) -> None:
            events.append(f"pending-read-{self.ordinal}")
            return None

        def record_interrupted_publish_recovery(self) -> str:
            raise AssertionError("an already-complete recursive record must not be recorded twice")

        @property
        def last_evidence_sha256(self) -> str:
            events.append(f"tail-read-{self.ordinal}")
            return tail_sha256

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append(f"journal-close-{self.ordinal}")
                self.receipt.close()

    receipts = [Receipt(1), Receipt(2)]
    journals = [Journal(receipts[0], 1), Journal(receipts[1], 2)]

    def reopen_receipt(*_: Any, **__: Any) -> Receipt:
        receipt = receipts.pop(0)
        events.append(f"receipt-open-{receipt.ordinal}")
        return receipt

    def reopen_journal(receipt: Receipt, **_: Any) -> Journal:
        journal = journals.pop(0)
        assert journal.receipt is receipt
        events.append(f"journal-open-{journal.ordinal}")
        return journal

    monkeypatch.setattr(operator, "reopen_evidence_directory", reopen_receipt)
    monkeypatch.setattr(operator.EvidenceJournal, "reopen_for_recovery", reopen_journal)

    receipt, journal, recovered, recorded, tail = operator._reopen_recovery_journal_barrier(
        tmp_path / "evidence",
        receipt_sha256="a" * 64,
        plan_sha256="b" * 64,
    )
    try:
        assert recovered is None
        assert recorded is None
        assert tail == tail_sha256
        assert receipt.ordinal == 2 and journal.ordinal == 2
        assert events == [
            "receipt-open-1",
            "journal-open-1",
            "pending-read-1",
            "tail-read-1",
            "journal-close-1",
            "receipt-close-1",
            "receipt-open-2",
            "journal-open-2",
            "pending-read-2",
            "tail-read-2",
        ]
    finally:
        journal.close()


def test_recovery_journal_barrier_capacity_failure_closes_without_second_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Receipt:
        closed = False

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append("receipt-close")

    receipt = Receipt()

    class Journal:
        interrupted_publish_recovery = {"recovered_entries": [{"sequence": 999}]}

        def record_interrupted_publish_recovery(self) -> str:
            events.append("record-interrupted-publish-recovery")
            raise ControllerError("journal_sequence_capacity_exhausted")

        @property
        def last_evidence_sha256(self) -> str:
            raise AssertionError("capacity failure must precede tail acceptance")

        def close(self) -> None:
            events.append("journal-close")
            receipt.close()

    receipt_calls = 0

    def reopen_receipt(*_: Any, **__: Any) -> Receipt:
        nonlocal receipt_calls
        receipt_calls += 1
        events.append("receipt-open")
        return receipt

    monkeypatch.setattr(operator, "reopen_evidence_directory", reopen_receipt)
    monkeypatch.setattr(
        operator.EvidenceJournal,
        "reopen_for_recovery",
        lambda *a, **k: _append_and_return(events, "journal-open", Journal()),
    )

    with pytest.raises(ControllerError, match="journal_sequence_capacity_exhausted"):
        operator._reopen_recovery_journal_barrier(
            tmp_path / "evidence",
            receipt_sha256="a" * 64,
            plan_sha256="b" * 64,
        )
    assert receipt_calls == 1
    assert events == [
        "receipt-open",
        "journal-open",
        "record-interrupted-publish-recovery",
        "journal-close",
        "receipt-close",
    ]


@pytest.mark.parametrize("second_pending", (False, True))
def test_recovery_journal_barrier_rejects_strict_second_reopen_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_pending: bool,
) -> None:
    events: list[str] = []
    receipts: list[Any] = []

    class Receipt:
        def __init__(self, ordinal: int) -> None:
            self.ordinal = ordinal
            self.closed = False

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append(f"receipt-close-{self.ordinal}")

    class Journal:
        def __init__(self, receipt: Receipt, ordinal: int) -> None:
            self.receipt = receipt
            self.ordinal = ordinal
            self.closed = False

        @property
        def interrupted_publish_recovery(self) -> dict[str, Any] | None:
            if self.ordinal == 2 and second_pending:
                return {"recovered_entries": [{"sequence": 18}]}
            return None

        @property
        def last_evidence_sha256(self) -> str:
            return ("9" if self.ordinal == 1 else "8") * 64

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append(f"journal-close-{self.ordinal}")
                self.receipt.close()

    journals: list[Journal] = []

    def reopen_receipt(*_: Any, **__: Any) -> Receipt:
        receipt = Receipt(len(receipts) + 1)
        receipts.append(receipt)
        return receipt

    def reopen_journal(receipt: Receipt, **_: Any) -> Journal:
        journal = Journal(receipt, len(journals) + 1)
        journals.append(journal)
        return journal

    monkeypatch.setattr(operator, "reopen_evidence_directory", reopen_receipt)
    monkeypatch.setattr(operator.EvidenceJournal, "reopen_for_recovery", reopen_journal)

    with pytest.raises(
        boundary.OperatorError,
        match="interrupted_publish_recovery_strict_reopen_rejected",
    ):
        operator._reopen_recovery_journal_barrier(
            tmp_path / "evidence",
            receipt_sha256="a" * 64,
            plan_sha256="b" * 64,
        )
    assert len(receipts) == 2 and all(receipt.closed for receipt in receipts)
    assert len(journals) == 2 and all(journal.closed for journal in journals)
    assert events == [
        "journal-close-1",
        "receipt-close-1",
        "journal-close-2",
        "receipt-close-2",
    ]


@pytest.mark.parametrize("abort_fails_before_close", (False, True))
def test_resume_abort_is_cleanup_only_and_closes_owned_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    abort_fails_before_close: bool,
) -> None:
    events: list[str] = []
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    evidence = (tmp_path / "evidence").resolve()
    evidence.mkdir()
    plan = SimpleNamespace(head_sha=HEAD, sha256="a" * 64)

    class Receipt:
        closed = False

        def close(self) -> None:
            self.closed = True
            events.append("receipt-close")

    receipt = Receipt()

    class Journal:
        closed = False
        interrupted_publish_recovery = None

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                events.append("journal-close")
                receipt.close()

    journal = Journal()

    class Client:
        def close(self) -> None:
            events.append("client-close")

    class Driver:
        def abort(self) -> str:
            events.append("abort")
            if abort_fails_before_close:
                raise ControllerError("fixture-resume-abort-before-close")
            journal.close()
            return "b" * 64

    class DriverFactory:
        @staticmethod
        def resume_for_abort(*positional: Any, **keyword: Any) -> Driver:
            events.append("resume-for-abort")
            assert positional[2] is plan and positional[3] is journal
            assert keyword["observation_poll_limit"] == 3
            return Driver()

    class Controller:
        def __init__(self, _: Any, remote: Any, **__: Any) -> None:
            assert isinstance(remote, operator._AbortOnlyRemote)
            events.append("cleanup-controller")

    monkeypatch.setattr(
        operator, "_phase_inputs", lambda _: ("publication", "refs/tags/v0.15.0", HEAD)
    )
    monkeypatch.setattr(
        operator,
        "_common_paths",
        lambda _: (str(repository), "git", "gh", "ssh"),
    )
    monkeypatch.setattr(
        operator,
        "_load_and_revalidate_inventory",
        lambda *a, **k: _append_and_return(events, "inventory-revalidate", ({}, {}, "c" * 64)),
    )
    monkeypatch.setattr(operator, "ensure_path_outside_repository", lambda *a, **k: evidence)
    monkeypatch.setattr(
        operator,
        "validate_anonymous_fd",
        lambda *a, **k: {"regular_file": False, "anonymous": True},
    )
    monkeypatch.setattr(
        operator, "close_owned_fd", lambda value: events.append(f"fd-close-{value}")
    )
    monkeypatch.setattr(
        operator,
        "read_recovery_plan",
        lambda *a, **k: _append_and_return(events, "plan-read", plan),
    )
    monkeypatch.setattr(operator, "_validate_recovery_preflight", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(
        operator, "_revalidate_locked_posture", lambda *a, **k: events.append("posture-revalidate")
    )
    monkeypatch.setattr(
        operator.LambdaHttpClient,
        "from_secret_fd",
        staticmethod(lambda _: _append_and_return(events, "secret-read", Client())),
    )
    monkeypatch.setattr(
        operator,
        "_reopen_recovery_journal_barrier",
        lambda *a, **k: _append_and_return(
            events,
            "journal-reopen-barrier",
            (receipt, journal, None, None, "f" * 64),
        ),
    )
    monkeypatch.setattr(
        operator,
        "LiveGates",
        lambda *a: _append_and_return(events, "cleanup-gates", SimpleNamespace()),
    )
    monkeypatch.setattr(operator, "LambdaLiveAdapter", lambda *a: SimpleNamespace())
    monkeypatch.setattr(operator, "_locked_github", lambda _: object())
    monkeypatch.setattr(operator, "ReleaseGpuController", Controller)
    monkeypatch.setattr(operator, "LiveReleaseDriver", DriverFactory)
    args = SimpleNamespace(
        confirm_plan_sha256=plan.sha256,
        evidence_directory=str(evidence),
        evidence_directory_receipt_sha256="d" * 64,
        lambda_api_key_fd=103,
        observation_poll_limit=3,
    )
    if abort_fails_before_close:
        with pytest.raises(ControllerError, match="fixture-resume-abort-before-close"):
            operator.resume_abort(
                cast(Any, args),
                environment_receipt={"scrubbed": True},
                launch_receipt={"windows_handle_transport": True, "inherited_handle_count": 1},
                resources=_operator_resources(),
            )
    else:
        result = operator.resume_abort(
            cast(Any, args),
            environment_receipt={"scrubbed": True},
            launch_receipt={"windows_handle_transport": True, "inherited_handle_count": 1},
            resources=_operator_resources(),
        )
        assert result["cleanup_only"] is True
        assert result["remote_execution_available"] is False
        assert result["interrupted_publish_recovery"] is None
    assert (
        events.index("journal-reopen-barrier")
        < events.index("plan-read")
        < events.index("inventory-revalidate")
        < events.index("secret-read")
    )
    assert events.index("posture-revalidate") < events.index("cleanup-gates")
    assert events.index("resume-for-abort") < events.index("abort") < events.index("client-close")
    if abort_fails_before_close:
        assert events.index("client-close") < events.index("journal-close")
    else:
        assert events.index("journal-close") < events.index("client-close")
    assert journal.closed is True and receipt.closed is True


def test_resume_abort_rejects_recovery_plan_before_external_observation_or_secret_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    evidence = (tmp_path / "evidence").resolve()
    evidence.mkdir()
    calls: list[str] = []

    class Receipt:
        def close(self) -> None:
            calls.append("receipt-close")

    receipt = Receipt()

    class Journal:
        interrupted_publish_recovery = None

        def close(self) -> None:
            calls.append("journal-close")
            receipt.close()

    monkeypatch.setattr(
        operator, "_phase_inputs", lambda _: ("publication", "refs/tags/v0.15.0", HEAD)
    )
    monkeypatch.setattr(
        operator,
        "_common_paths",
        lambda _: (str(repository), "git", "gh", "ssh"),
    )
    monkeypatch.setattr(operator, "ensure_path_outside_repository", lambda *a, **k: evidence)
    monkeypatch.setattr(
        operator,
        "validate_anonymous_fd",
        lambda *a, **k: {"regular_file": False, "anonymous": True},
    )
    monkeypatch.setattr(operator, "close_owned_fd", lambda *_: None)
    journal = Journal()
    monkeypatch.setattr(
        operator,
        "_reopen_recovery_journal_barrier",
        lambda *a, **k: _append_and_return(
            calls,
            "journal-reopen-barrier",
            (receipt, journal, None, None, "f" * 64),
        ),
    )

    def reject_plan(*_: Any, **__: Any) -> Any:
        calls.append("plan-loader")
        raise ControllerError("recovery_plan_rejected_fixture")

    monkeypatch.setattr(operator, "read_recovery_plan", reject_plan)
    monkeypatch.setattr(
        operator,
        "_load_and_revalidate_inventory",
        lambda *a, **k: pytest.fail("inventory/GitHub observation preceded plan rejection"),
    )
    monkeypatch.setattr(
        operator.LambdaHttpClient,
        "from_secret_fd",
        staticmethod(lambda _: pytest.fail("secret read preceded plan rejection")),
    )
    args = SimpleNamespace(
        confirm_plan_sha256="a" * 64,
        evidence_directory=str(evidence),
        evidence_directory_receipt_sha256="d" * 64,
        lambda_api_key_fd=103,
        observation_poll_limit=3,
    )
    with pytest.raises(ControllerError, match="recovery_plan_rejected_fixture"):
        operator.resume_abort(
            cast(Any, args),
            environment_receipt={"scrubbed": True},
            launch_receipt={"windows_handle_transport": True, "inherited_handle_count": 1},
            resources=_operator_resources(),
        )
    assert calls[:2] == ["journal-reopen-barrier", "plan-loader"]
    assert "journal-close" in calls and "receipt-close" in calls


def test_recovery_dispatch_source_contains_four_state_fail_closed_contract() -> None:
    source = Path(operator.__file__).read_text(encoding="utf-8")
    assert 'tail.state == "source-unrecorded"' in source
    assert 'tail.state == "pending-operator-settlement"' in source
    assert 'tail.state == "pending-intent"' in source
    assert 'tail.state in {"complete", "pending-intent"}' in source
    assert source.count("recovery_source=publication_source") == 2
    assert "EvidenceJournal.build_publication_recovery_operator_settlement" in source
    assert "publication_source = load_publication_source()" in source
    assert "progress=record_progress" in source
    assert 'workflow_completion_verified": False' in source


@pytest.mark.parametrize(
    ("initial_state", "expected_decision", "load_states"),
    (
        (
            "source-unrecorded",
            "dispatch",
            ("source-unrecorded", "complete", "pending-operator-settlement", "complete"),
        ),
        ("complete", "dispatch", ("complete", "pending-operator-settlement", "complete")),
        (
            "pending-intent",
            "reconcile",
            ("pending-intent", "pending-operator-settlement", "complete"),
        ),
        (
            "pending-operator-settlement",
            "local-only",
            ("pending-operator-settlement", "complete"),
        ),
    ),
)
def test_recovery_dispatch_executes_exact_loader_sealed_crash_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_state: str,
    expected_decision: str,
    load_states: tuple[str, ...],
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    evidence = (tmp_path / "evidence").resolve()
    evidence.mkdir()
    events: list[tuple[str, Mapping[str, Any]]] = []
    calls: list[str] = []
    pending = {"head_sha": HEAD, "source_run_id": 771, "request_nonce": "1" * 16}
    settlement = {
        "plan_sha256": "d" * 64,
        "head_sha": HEAD,
        "source_run_id": 771,
        "mode": (
            "response-loss-reconciled"
            if expected_decision == "reconcile"
            else "mutation-response-observed"
        ),
        "recovery_dispatch_evidence_sha256": "a" * 64,
        "publication_recovery_source_evidence_sha256": "8" * 64,
        "publication_journal_sha256": "9" * 64,
        "recovery_run_id": 990,
        "request_nonce_archived_in_controller_receipt": True,
        "raw_dispatch_bypass_used": False,
        "pending_intent_replayed": False,
        "workflow_completion_verified": False,
        "no_republish_verified": False,
    }

    class Tail:
        def __init__(self, state: str, *, final: bool) -> None:
            self.state = state
            self.pending_intent = dict(pending) if state == "pending-intent" else None
            self.pending_operator_settlement = (
                dict(settlement) if state == "pending-operator-settlement" else None
            )
            self.last_operator_settlement = dict(settlement) if final else None
            self.completed_run_ids = (990,) if final else ()
            self.completed_request_nonces = ("2" * 16,) if final else ()

    class Source:
        head_sha = HEAD
        run_id = 771
        control_plane_plan_sha256 = "d" * 64
        publication_journal_sha256 = "9" * 64
        evidence_directory_receipt_sha256 = "e" * 64
        evidence_sha256 = "8" * 64

        def __init__(self, state: str, *, final: bool = False) -> None:
            self.recovery_tail = Tail(state, final=final)

        def to_mapping(self) -> dict[str, Any]:
            return {
                "schema_version": 1,
                "kind": "explainiverse-publication-recovery-source",
                "head_sha": HEAD,
                "run_id": 771,
                "evidence_sha256": "8" * 64,
            }

    class Journal:
        def __init__(self) -> None:
            self.sequence = 998 if initial_state == "pending-operator-settlement" else 0
            self.interrupted_publish_recovery = {
                "classification": "canonical-next-envelope-published"
            }

        def require_capacity(self, entries: int) -> None:
            calls.append(f"capacity-{entries}")
            assert entries in {1, 2, 3, 4}
            assert self.sequence + entries <= 999

        def pending_recovery_dispatch_intent(self) -> Mapping[str, Any] | None:
            calls.append("pending")
            return dict(pending) if initial_state == "pending-intent" else None

        def record(self, label: str, payload: Mapping[str, Any]) -> str:
            self.sequence += 1
            events.append((label, dict(payload)))
            return "f" * 64

        def close(self) -> None:
            calls.append("journal-close")

    journal = Journal()
    source_loads = iter(
        Source(state, final=index == len(load_states) - 1)
        for index, state in enumerate(load_states)
    )

    class Controller:
        def __init__(self, *_: Any, **__: Any) -> None:
            calls.append("controller-created")

        def reconcile_release_recovery_dispatch(
            self,
            intent: Mapping[str, Any],
            *,
            recovery_source: Source,
            poll_limit: int,
            progress: Any,
        ) -> Any:
            calls.append("reconcile")
            assert intent == pending and poll_limit == 7
            assert recovery_source.recovery_tail.state == "pending-intent"
            progress("github-recovery-dispatch-settled", {"run_id": 990})
            return SimpleNamespace(evidence_sha256="a" * 64, run_id=990, run_attempt=1)

        def dispatch_release_recovery(
            self,
            *,
            head_sha: str,
            source_run_id: int,
            recovery_request_nonce: str,
            recovery_source: Source,
            poll_limit: int,
            progress: Any,
        ) -> Any:
            calls.append("dispatch")
            assert head_sha == HEAD and source_run_id == 771 and poll_limit == 7
            assert re.fullmatch(r"[0-9a-f]{16}", recovery_request_nonce)
            assert recovery_source.recovery_tail.state == "complete"
            progress(
                "github-recovery-dispatch-intent",
                {
                    "head_sha": head_sha,
                    "source_run_id": source_run_id,
                    "request_nonce": recovery_request_nonce,
                },
            )
            progress("github-recovery-dispatch-settled", {"run_id": 990})
            return SimpleNamespace(evidence_sha256="a" * 64, run_id=990, run_attempt=1)

    receipt = SimpleNamespace(close=lambda: calls.append("receipt-close"))
    inventory: dict[str, Any] = {}
    monkeypatch.setattr(operator, "_phase_inputs", lambda _: ("publication", "tag", HEAD))
    monkeypatch.setattr(
        operator,
        "_common_paths",
        lambda _: (str(repository), "git", "gh", "ssh"),
    )
    monkeypatch.setattr(
        operator,
        "_load_and_revalidate_inventory",
        lambda *a, **k: _append_and_return(
            calls, "inventory-revalidate", ({}, inventory, "c" * 64)
        ),
    )
    monkeypatch.setattr(operator, "ensure_path_outside_repository", lambda *a, **k: evidence)
    monkeypatch.setattr(
        operator, "read_recovery_plan", lambda *a, **k: SimpleNamespace(head_sha=HEAD)
    )
    monkeypatch.setattr(
        operator.EvidenceJournal,
        "load_publication_recovery_source",
        lambda *a, **k: _append_and_return(calls, "source-loader", next(source_loads)),
    )
    monkeypatch.setattr(
        operator.EvidenceJournal,
        "build_publication_recovery_operator_settlement",
        lambda source, receipt: _append_and_return(
            calls,
            "settlement-builder",
            (
                pytest.fail("builder received unsealed tail")
                if source.recovery_tail.state != "pending-operator-settlement"
                else dict(settlement)
            ),
        ),
    )
    monkeypatch.setattr(operator, "_validate_recovery_preflight", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(operator, "_revalidate_locked_posture", lambda *a, **k: None)
    monkeypatch.setattr(
        operator,
        "_reopen_recovery_journal_barrier",
        lambda *a, **k: _append_and_return(
            calls,
            "journal-reopen-barrier",
            (
                receipt,
                journal,
                {"classification": "canonical-next-envelope-published"},
                "7" * 64,
                "7" * 64,
            ),
        ),
    )
    monkeypatch.setattr(
        operator,
        "_locked_github",
        lambda _: _append_and_return(calls, "github-transport-created", object()),
    )
    monkeypatch.setattr(operator, "ReleaseGpuController", Controller)
    args = SimpleNamespace(
        confirm_plan_sha256="d" * 64,
        publication_journal_sha256="9" * 64,
        source_run_id=771,
        recovery_poll_limit=7,
        evidence_directory=str(evidence),
        evidence_directory_receipt_sha256="e" * 64,
    )
    result = operator.dispatch_release_recovery(
        cast(Any, args),
        environment_receipt={"scrubbed": True},
        launch_receipt={"secure": True},
        resources=_operator_resources(),
    )
    assert calls.index("journal-reopen-barrier") < calls.index("source-loader")
    assert calls.index("source-loader") < calls.index("inventory-revalidate")
    assert ("reconcile" in calls) is (expected_decision == "reconcile")
    assert ("dispatch" in calls) is (expected_decision == "dispatch")
    assert ("controller-created" in calls) is (expected_decision != "local-only")
    assert ("github-transport-created" in calls) is (expected_decision != "local-only")
    assert ("settlement-builder" in calls) is (expected_decision != "local-only")
    assert events[-1][0] == "operator-release-recovery-dispatch-settled"
    if initial_state == "source-unrecorded":
        assert events[0][0] == "operator-publication-recovery-source"
    assert result["workflow_completion_verified"] is False
    assert result["no_republish_verified"] is False
    assert result["interrupted_publish_recovery"] == {
        "classification": "canonical-next-envelope-published"
    }
    assert result["mode"] == settlement["mode"]
    assert result["recovery_run_id"] == settlement["recovery_run_id"]
    if initial_state == "pending-operator-settlement":
        assert journal.sequence == 999
    expected_capacity = {
        "source-unrecorded": ["capacity-4", "capacity-3"],
        "complete": ["capacity-3"],
        "pending-intent": ["capacity-2"],
        "pending-operator-settlement": ["capacity-1"],
    }
    assert [call for call in calls if call.startswith("capacity-")] == expected_capacity[
        initial_state
    ]
    with pytest.raises(StopIteration):
        next(source_loads)


def test_recovery_local_settlement_at_sequence_999_rejects_before_any_append_or_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    evidence = (tmp_path / "evidence").resolve()
    evidence.mkdir()
    calls: list[str] = []
    settlement = {
        "plan_sha256": "d" * 64,
        "head_sha": HEAD,
        "source_run_id": 771,
        "mode": "mutation-response-observed",
        "recovery_dispatch_evidence_sha256": "a" * 64,
        "publication_recovery_source_evidence_sha256": "8" * 64,
        "publication_journal_sha256": "9" * 64,
        "recovery_run_id": 990,
        "request_nonce_archived_in_controller_receipt": True,
        "raw_dispatch_bypass_used": False,
        "pending_intent_replayed": False,
        "workflow_completion_verified": False,
        "no_republish_verified": False,
    }
    source = SimpleNamespace(
        head_sha=HEAD,
        run_id=771,
        control_plane_plan_sha256="d" * 64,
        publication_journal_sha256="9" * 64,
        evidence_directory_receipt_sha256="e" * 64,
        evidence_sha256="8" * 64,
        recovery_tail=SimpleNamespace(
            state="pending-operator-settlement",
            pending_operator_settlement=settlement,
        ),
    )

    class Journal:
        sequence = 999
        interrupted_publish_recovery = None

        def require_capacity(self, entries: int) -> None:
            calls.append(f"capacity-{entries}")
            assert entries == 1 and self.sequence + entries == 1000
            raise ControllerError("journal_sequence_capacity_exhausted")

        def record(self, *_: Any, **__: Any) -> str:
            raise AssertionError("capacity failure must precede journal append")

        def close(self) -> None:
            calls.append("journal-close")

    receipt = SimpleNamespace(close=lambda: calls.append("receipt-close"))
    monkeypatch.setattr(operator, "_phase_inputs", lambda _: ("publication", "tag", HEAD))
    monkeypatch.setattr(
        operator,
        "_common_paths",
        lambda _: (str(repository), "git", "gh", "ssh"),
    )
    monkeypatch.setattr(
        operator,
        "_load_and_revalidate_inventory",
        lambda *a, **k: ({}, {}, "c" * 64),
    )
    monkeypatch.setattr(operator, "ensure_path_outside_repository", lambda *a, **k: evidence)
    monkeypatch.setattr(
        operator,
        "read_recovery_plan",
        lambda *a, **k: SimpleNamespace(head_sha=HEAD),
    )
    monkeypatch.setattr(
        operator.EvidenceJournal,
        "load_publication_recovery_source",
        lambda *a, **k: source,
    )
    monkeypatch.setattr(operator, "_validate_recovery_preflight", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(operator, "_revalidate_locked_posture", lambda *a, **k: None)
    journal = Journal()
    monkeypatch.setattr(
        operator,
        "_reopen_recovery_journal_barrier",
        lambda *a, **k: (receipt, journal, None, None, "f" * 64),
    )
    monkeypatch.setattr(
        operator,
        "_locked_github",
        lambda _: pytest.fail("capacity failure must precede transport construction"),
    )
    monkeypatch.setattr(
        operator,
        "ReleaseGpuController",
        lambda *a, **k: pytest.fail("capacity failure must precede controller construction"),
    )
    args = SimpleNamespace(
        confirm_plan_sha256="d" * 64,
        publication_journal_sha256="9" * 64,
        source_run_id=771,
        recovery_poll_limit=7,
        evidence_directory=str(evidence),
        evidence_directory_receipt_sha256="e" * 64,
    )
    with pytest.raises(ControllerError, match="journal_sequence_capacity_exhausted"):
        operator.dispatch_release_recovery(
            cast(Any, args),
            environment_receipt={"scrubbed": True},
            launch_receipt={"secure": True},
            resources=_operator_resources(),
        )
    assert calls == ["capacity-1", "journal-close"]


@pytest.mark.parametrize(
    "loader_error",
    (
        "publication_source_anchor_not_lifecycle_restored",
        "publication_source_dispatch_binding_rejected",
        "publication_source_journal_anchor_missing",
    ),
)
def test_recovery_dispatch_rejects_partial_wrong_source_or_tail_drift_before_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader_error: str,
) -> None:
    repository = (tmp_path / "repo").resolve()
    repository.mkdir()
    evidence = (tmp_path / "evidence").resolve()
    evidence.mkdir()
    receipt = SimpleNamespace(close=lambda: None)
    calls: list[str] = []
    monkeypatch.setattr(operator, "_phase_inputs", lambda _: ("publication", "tag", HEAD))
    monkeypatch.setattr(
        operator,
        "_common_paths",
        lambda _: (str(repository), "git", "gh", "ssh"),
    )
    monkeypatch.setattr(
        operator,
        "_load_and_revalidate_inventory",
        lambda *a, **k: ({}, {}, "a" * 64),
    )
    monkeypatch.setattr(operator, "ensure_path_outside_repository", lambda *a, **k: evidence)
    monkeypatch.setattr(
        operator, "read_recovery_plan", lambda *a, **k: SimpleNamespace(head_sha=HEAD)
    )

    def reject(*_: Any, **__: Any) -> Any:
        raise ControllerError(loader_error)

    monkeypatch.setattr(operator.EvidenceJournal, "load_publication_recovery_source", reject)

    class Journal:
        interrupted_publish_recovery = None

        def close(self) -> None:
            calls.append("journal-close")

    journal = Journal()
    monkeypatch.setattr(
        operator,
        "_reopen_recovery_journal_barrier",
        lambda *a, **k: _append_and_return(
            calls,
            "journal-reopen-barrier",
            (receipt, journal, None, None, "f" * 64),
        ),
    )
    args = SimpleNamespace(
        confirm_plan_sha256="b" * 64,
        publication_journal_sha256="c" * 64,
        source_run_id=771,
        recovery_poll_limit=7,
        evidence_directory=str(evidence),
        evidence_directory_receipt_sha256="d" * 64,
    )
    with pytest.raises(ControllerError, match=loader_error):
        operator.dispatch_release_recovery(
            cast(Any, args),
            environment_receipt={"scrubbed": True},
            launch_receipt={"secure": True},
            resources=_operator_resources(),
        )
    assert calls == ["journal-reopen-barrier", "journal-close"]


def test_windows_console_reader_uses_wide_typed_api_and_restores_mode() -> None:
    class FakeConsole:
        def __init__(self) -> None:
            self.modes: list[int] = []

        def get_std_handle(self, _: Any) -> int:
            return 1234

        def get_console_mode(self, _: Any, result: Any) -> int:
            result._obj.value = 7
            return 1

        def set_console_mode(self, _: Any, mode: Any) -> int:
            self.modes.append(int(mode.value))
            return 1

        def read_console_w(self, _: Any, buffer: Any, __: Any, count: Any, ___: Any) -> int:
            value = "fixture-console-key\r\n"
            for index, character in enumerate(value):
                buffer._obj[index] = character
            count._obj.value = len(value)
            return 1

    fake = FakeConsole()
    prompt = io.StringIO()
    secret = windows_launcher._console_secret(api=fake, prompt_stream=prompt)
    try:
        assert secret == b"fixture-console-key"
        assert fake.modes == [3, 7]
        assert "not echoed" in prompt.getvalue()
    finally:
        windows_launcher._zero(secret)


def test_windows_launcher_pipe_write_uses_zero_copy_secret_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = bytearray(b"fixture-secret")
    observations: list[tuple[type[Any], Any]] = []

    def write(_: int, chunk: Any) -> int:
        observations.append((type(chunk), chunk.obj))
        return len(chunk)

    monkeypatch.setattr(windows_launcher.os, "write", write)
    windows_launcher._write_all(123, secret)
    assert observations == [(memoryview, secret)]
    windows_launcher._zero(secret)


@pytest.mark.parametrize(
    "override",
    (
        "--lambda-api-key-fd=3",
        "--plan-confirmation-fd=4",
        "--lambda-api-key-handle=12",
        "--plan-confirmation-handle=16",
        "--windows-launcher-parent-receipt={}",
    ),
)
def test_windows_launcher_rejects_reserved_transport_equals_overrides(override: str) -> None:
    with pytest.raises(windows_launcher.LauncherError, match="transport_override_rejected"):
        windows_launcher._parse_launcher(
            ["--action", "transport-self-test", "--repository-root", str(Path.cwd()), override]
        )


@pytest.mark.skipif(os.name != "nt", reason="native launcher is Windows-only")
def test_windows_launcher_rejects_unguarded_parent_before_reading_secret() -> None:
    root = Path.cwd().resolve()
    secret = b"must-not-be-consumed-or-logged"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.release_gpu_jit_lambda_operator.windows_launcher",
            "--launcher-secret-source",
            "stdin-pipe",
            "--action",
            "transport-self-test",
            "--repository-root",
            str(root),
            "--transport-self-test-nonce",
            "8" * 32,
        ],
        cwd=root,
        input=secret + b"\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert b"launcher_preloader_receipt_missing" in result.stderr
    assert secret not in result.stdout + result.stderr


@pytest.mark.skipif(os.name != "nt", reason="native inherited HANDLE contract is Windows-only")
def test_windows_launcher_delivers_secret_and_post_plan_confirmation_without_exposure(
    tmp_path: Path,
) -> None:
    required_environment = {
        name: os.environ.get(name)
        for name in (
            "EXPLAINIVERSE_OPERATOR_E2E_REPOSITORY_ROOT",
            "EXPLAINIVERSE_OPERATOR_E2E_PYTHON_ROOT",
            "EXPLAINIVERSE_OPERATOR_E2E_SITE_ROOT",
            "EXPLAINIVERSE_OPERATOR_E2E_PYTHON_INSTALL_RECEIPT",
            "EXPLAINIVERSE_OPERATOR_E2E_PYTHON_INSTALL_RECEIPT_SHA256",
            "EXPLAINIVERSE_OPERATOR_E2E_SITE_INSTALL_RECEIPT",
            "EXPLAINIVERSE_OPERATOR_E2E_SITE_INSTALL_RECEIPT_SHA256",
            "EXPLAINIVERSE_OPERATOR_E2E_EXPECTED_HEAD",
            "EXPLAINIVERSE_OPERATOR_E2E_PRELOADER_SHA256",
        )
    }
    if not all(required_environment.values()):
        pytest.skip("requires a freshly prepared pinned-runtime clean-source fixture")
    root = Path(str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_REPOSITORY_ROOT"])).resolve()
    python_root = Path(
        str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_PYTHON_ROOT"])
    ).resolve()
    site_root = Path(str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_SITE_ROOT"])).resolve()
    preloader = root / "scripts" / "release_gpu_jit_lambda_operator" / "preloader.py"
    hostile = (tmp_path / "hostile-pythonpath").resolve()
    hostile.mkdir()
    sentinel = hostile / "sitecustomize-ran"
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    secret = b"fixture-lambda-key-never-log"
    nonce = "9" * 32
    command = [
        str(python_root / "python.exe"),
        "-I",
        "-S",
        "-B",
        "-c",
        windows_launcher.PRELOADER_SHIM,
        hashlib.sha256(windows_launcher.PRELOADER_SHIM.encode("utf-8")).hexdigest(),
        str(preloader),
        str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_PRELOADER_SHA256"]),
        "--operator-target",
        "windows-launcher",
        "--launcher-secret-source",
        "stdin-pipe",
        "--action",
        "transport-self-test",
        "--repository-root",
        str(root),
        "--operator-python-root",
        str(python_root),
        "--operator-site-root",
        str(site_root),
        "--operator-python-install-receipt",
        str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_PYTHON_INSTALL_RECEIPT"]),
        "--operator-python-install-receipt-sha256",
        str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_PYTHON_INSTALL_RECEIPT_SHA256"]),
        "--operator-site-install-receipt",
        str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_SITE_INSTALL_RECEIPT"]),
        "--operator-site-install-receipt-sha256",
        str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_SITE_INSTALL_RECEIPT_SHA256"]),
        "--expected-head-sha",
        str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_EXPECTED_HEAD"]),
        "--transport-self-test-nonce",
        nonce,
    ]
    process = subprocess.Popen(
        command,
        cwd=Path(
            str(required_environment["EXPLAINIVERSE_OPERATOR_E2E_PYTHON_INSTALL_RECEIPT"])
        ).parent,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "GH_TOKEN": "ambient-token-must-not-reach-child",
            "PYTHONPATH": str(hostile),
        },
    )
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(secret + b"\n")
    process.stdin.flush()
    plan_line = process.stdout.readline()
    if not plan_line:
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait(timeout=30)
        pytest.fail(
            f"native launcher exited {return_code} before plan: " + stderr.decode(errors="replace")
        )
    plan = json.loads(plan_line)
    assert plan["kind"] == "explainiverse-lambda-plan-awaiting-confirmation"
    process.stdin.write(plan["plan_sha256"].encode("ascii") + b"\n")
    process.stdin.flush()
    process.stdin.close()
    remaining = process.stdout.read()
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait(timeout=30)
    output = plan_line + remaining + stderr
    assert return_code == 0, output.decode(errors="replace")
    assert secret not in output
    assert b"ambient-token-must-not-reach-child" not in output
    result = json.loads(remaining.splitlines()[-1])
    assert result["lambda_secret_received"] is True
    assert result["plan_confirmation"]["confirmation_read_after_plan"] is True
    source_receipt = result["secure_launch"]["preloader"]["source"]
    archived_manifest = source_receipt["source_manifest"]
    assert (
        hashlib.sha256(boundary.canonical_json(archived_manifest)).hexdigest()
        == source_receipt["source_manifest_sha256"]
    )
    manifest_rows = b"".join(
        (
            f"{name}\t{item['mode']}\t{item['bytes']}\t{item['sha256']}\t"
            f"{item['git_blob_sha']}\n"
        ).encode("utf-8")
        for name, item in sorted(archived_manifest["files"].items())
    )
    assert (
        hashlib.sha256(manifest_rows).hexdigest()
        == source_receipt["source_manifest_inventory_sha256"]
    )
    declaration = result["secure_launch"]["windows_launcher_parent_declaration"]
    assert declaration["parent_provenance_authenticated"] is False
    assert declaration["security_authority_derived_from_declaration"] is False
    assert declaration["child_revalidated_handle_transport_and_sealed_resources"] is True
    assert not sentinel.exists()
    command_text = " ".join(command).encode("utf-8")
    assert secret not in command_text
