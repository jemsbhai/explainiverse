from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from scripts.release_gpu_jit_lambda_operator import receipt_contract as contract

HEAD = "a" * 40
LIFECYCLE_NONCE = "c" * 32
INSPECTION_SHA256 = "d" * 64
ROOT = PureWindowsPath(r"C:\fixture\repository")
PYTHON_ROOT = PureWindowsPath(r"C:\fixture\python")
SITE_ROOT = PureWindowsPath(r"C:\fixture\site")
WORKING_ROOT = PureWindowsPath(r"C:\fixture\receipts\python")
OWNER_SID = "S-1-5-21-1000"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _live_canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _blob(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode("ascii") + value).hexdigest()


def _source_fixture(
    phase: str,
    *,
    policy_sha256: str | None = None,
    controller_source_sha256: str | None = None,
    runtime_bundle_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    excluded = {contract.SOURCE_MANIFEST_RELATIVE, contract.PRELOADER_RELATIVE}
    files: dict[str, dict[str, Any]] = {}
    for index, relative in enumerate(sorted(contract.CRITICAL_SOURCE_PATHS - excluded), 1):
        raw = f"fixture:{index}:{relative}\n".encode()
        files[relative] = {
            "mode": "100644",
            "bytes": len(raw),
            "sha256": _sha(raw),
            "git_blob_sha": _blob(raw),
        }
    files[contract.SHIM_RELATIVE]["sha256"] = contract.PRELOADER_SHIM_SHA256
    if policy_sha256 is not None:
        files[".github/release-control-policy.json"]["sha256"] = policy_sha256
    if controller_source_sha256 is not None:
        files["scripts/release_gpu_jit_lambda_controller/controller.py"][
            "sha256"
        ] = controller_source_sha256
    directories: set[str] = set()
    rows: list[bytes] = []
    for relative, item in sorted(files.items()):
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
        rows.append(
            (
                f"{relative}\t{item['mode']}\t{item['bytes']}\t"
                f"{item['sha256']}\t{item['git_blob_sha']}\n"
            ).encode()
        )
    manifest = {
        "schema_version": 1,
        "kind": "explainiverse-operator-source-worktree-manifest",
        "excluded_paths": [contract.SOURCE_MANIFEST_RELATIVE, contract.PRELOADER_RELATIVE],
        "files": files,
        "directories": sorted(directories),
        "file_count": len(files),
        "directory_count": len(directories),
        "file_inventory_sha256": _sha(b"".join(rows)),
        "source": "exact-staged-index-blobs",
        "runtime_git_dependency": False,
    }
    manifest_raw = _canonical(manifest)
    preloader_raw = b"x" * 4096
    preloader_sha256 = _sha(preloader_raw)
    critical = {
        name: {
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "git_blob_sha": item["git_blob_sha"],
        }
        for name, item in files.items()
    }
    critical[contract.SOURCE_MANIFEST_RELATIVE] = {
        "bytes": len(manifest_raw),
        "sha256": _sha(manifest_raw),
        "git_blob_sha": _blob(manifest_raw),
    }
    critical[contract.PRELOADER_RELATIVE] = {
        "bytes": len(preloader_raw),
        "sha256": preloader_sha256,
        "git_blob_sha": _blob(preloader_raw),
    }
    complete = dict(files)
    complete[contract.SOURCE_MANIFEST_RELATIVE] = {
        "mode": "100644",
        **critical[contract.SOURCE_MANIFEST_RELATIVE],
    }
    complete[contract.PRELOADER_RELATIVE] = {
        "mode": "100644",
        **critical[contract.PRELOADER_RELATIVE],
    }
    tree_raw = b"".join(
        f"{item['mode']} blob {item['git_blob_sha']}\t{name}\0".encode()
        for name, item in sorted(complete.items())
    )
    repository = {
        "repository": contract.REPOSITORY,
        "absolute_root": str(ROOT),
        "origin_url": contract.ORIGIN_URL,
        "head_sha": HEAD,
        "tree_object_sha": contract._git_tree_object_sha(complete),
        "tree_inventory_sha256": _sha(tree_raw),
        "clean_tracked_and_untracked": True,
        "supplied_ref": contract.PHASE_REFS[phase],
        "remote_object_type": "tag" if phase == "publication" else "commit",
        "remote_object_sha": "e" * 40 if phase == "publication" else HEAD,
        "remote_target_sha": HEAD,
        "remote_ref_response_sha256": "f" * 64,
        "annotated_tag_response_sha256": "0" * 64 if phase == "publication" else None,
        "critical_sources": critical,
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
    captured = {
        name: item
        for name, item in files.items()
        if name.startswith(
            (
                "scripts/release_gpu_jit_lambda_controller/",
                "scripts/release_gpu_jit_lambda_live/",
                "scripts/release_gpu_jit_lambda_operator/",
                "scripts/release_gpu_jit_lambda_runtime/",
            )
        )
        or name
        in {
            ".github/release-control-policy.json",
            ".github/workflows/cuda-ci.yml",
            ".github/workflows/publish-pypi.yml",
            ".github/workflows/recover-github-release.yml",
            "poetry.lock",
            "pyproject.toml",
            "scripts/release_external_controls.py",
            "scripts/verify_release_recovery.py",
        }
    }
    captured_rows = b"".join(
        f"{name}\t{item['bytes']}\t{item['sha256']}\n".encode()
        for name, item in sorted(captured.items())
    )
    source_material = {
        "schema_version": 1,
        "kind": "explainiverse-operator-clean-source-preload",
        "repository_root": str(ROOT),
        "origin_url": contract.ORIGIN_URL,
        "head_sha": HEAD,
        "head_and_origin_verified_during_credential_free_inventory": False,
        "source_manifest": manifest,
        "source_manifest_sha256": _sha(manifest_raw),
        "source_manifest_inventory_sha256": manifest["file_inventory_sha256"],
        "tracked_and_untracked_clean": True,
        "runtime_git_dependency": False,
        "preloader_path": str(ROOT / PurePosixPath(contract.PRELOADER_RELATIVE)),
        "preloader_sha256": preloader_sha256,
        "captured_module_count": len(captured),
        "captured_module_inventory_sha256": _sha(captured_rows),
        "project_modules_execute_from_captured_bytes": True,
        "arguments_sha256": "1" * 64,
    }
    source = {**source_material, "evidence_sha256": _sha(_canonical(source_material))}
    resources = {
        "policy_sha256": files[".github/release-control-policy.json"]["sha256"],
        "controller_source_sha256": files[
            "scripts/release_gpu_jit_lambda_controller/controller.py"
        ]["sha256"],
        "runtime_bundle_sha256": runtime_bundle_sha256 or "2" * 64,
    }
    return repository, source, resources


def _executable_fixture() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("git", "gh", "ssh"):
        expected = contract.PINNED_EXECUTABLES[name]
        acl = {
            "owner_sid": expected["owner_sid"],
            "expected_owner_sid": expected["owner_sid"],
            "unprivileged_write_ace_present": False,
            "dacl_ace_count": 3,
            "dacl_inventory_sha256": "3" * 64,
        }
        signature = {
            "status": "Valid",
            "subject": expected["authenticode_subject"],
            "thumbprint": expected["authenticode_thumbprint"],
        }
        row: dict[str, Any] = {
            "absolute_path": expected["absolute_path"],
            "sha256": expected["sha256"],
            "version": expected["version"],
            "regular_file": True,
            "symlink_or_reparse": False,
            "path_lookup_used": False,
            "hardlink_count": 1,
            "pinned_reviewed_identity": True,
            "acl": acl,
            "authenticode": signature,
            "authenticode_validated_by_pinned_helper": True,
        }
        if name == "git":
            row["resolved_runtime"] = {
                "absolute_path": expected["runtime_absolute_path"],
                "sha256": expected["runtime_sha256"],
                "version": expected["version"],
                "hardlink_count": 1,
                "acl": acl,
                "authenticode": signature,
            }
        result[name] = row
    result["python"] = {
        "absolute_path": str(PYTHON_ROOT / "python.exe"),
        "sha256": contract.PYTHON_EXECUTABLE_SHA256,
        "version": "Python 3.13.15",
        "regular_file": True,
        "symlink_or_reparse": False,
        "path_lookup_used": False,
        "hardlink_count": 1,
        "pinned_runtime_manifest_authority": True,
    }
    return result


def _directory_receipt(seed: str) -> tuple[dict[str, Any], dict[str, Any]]:
    captured_at = "2026-08-28T12:00:00+00:00"
    acl_material = {
        "owner_sid": OWNER_SID,
        "current_user_sid": OWNER_SID,
        "inheritance_protected": True,
        "child_inheritance_enabled": True,
        "aces": sorted(
            [
                {
                    "sid": sid,
                    "access": "allow",
                    "rights": "full-control",
                    "mask": 2_032_127,
                    "ace_flags": 3,
                }
                for sid in (OWNER_SID, "S-1-5-18", "S-1-5-32-544")
            ],
            key=lambda item: str(item["sid"]),
        ),
        "security_descriptor_sha256": seed * 64,
        "security_descriptor_bytes": 128,
    }
    acl = {
        **acl_material,
        "captured_at": captured_at,
        "evidence_sha256": _sha(_canonical(acl_material)),
    }
    public = {
        "captured_at": captured_at,
        "receipt_sha256": _sha(f"receipt:{seed}".encode()),
        "absolute_path_redacted": True,
        "directory_identity_recorded": True,
        "no_reparse_or_symlink": True,
        "owner_private": True,
        "acl": acl,
    }
    validation = {
        "validated_at": "2026-08-28T12:01:00+00:00",
        "receipt_sha256": public["receipt_sha256"],
        "absolute_path_redacted": True,
        "directory_identity_recorded": True,
        "no_reparse_or_symlink": True,
        "owner_private": True,
        "acl_evidence_sha256": acl["evidence_sha256"],
    }
    return public, validation


def _runtime_trees() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "python_root": str(PYTHON_ROOT),
            "file_count": 34,
            "directory_count": 0,
            "file_inventory_sha256": contract.PYTHON_FILE_INVENTORY_SHA256,
            "official_archive_sha256": contract.PYTHON_ARCHIVE_SHA256,
            "untracked_files_or_directories_present": False,
            "all_runtime_bytes_match_official_archive": True,
        },
        {
            "site_root": str(SITE_ROOT),
            "file_count": 756,
            "directory_count": 113,
            "file_inventory_sha256": contract.SITE_FILE_INVENTORY_SHA256,
            "untracked_files_or_directories_present": False,
            "bytecode_present": False,
            "all_importable_bytes_match_verified_wheels": True,
        },
    )


def _revalidation() -> dict[str, Any]:
    python_tree, site_tree = _runtime_trees()
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-enabled-environment-revalidation",
        "python_manifest_sha256": contract.PYTHON_MANIFEST_SHA256,
        "python_archive_sha256": contract.PYTHON_ARCHIVE_SHA256,
        "python_tree": python_tree,
        "manifest_sha256": contract.SITE_MANIFEST_SHA256,
        "archive_set_sha256": contract.SITE_ARCHIVE_SET_SHA256,
        "runtime_requirements_sha256": contract.RUNTIME_REQUIREMENTS_SHA256,
        "bootstrap_requirements_sha256": contract.BOOTSTRAP_REQUIREMENTS_SHA256,
        "site_tree": site_tree,
        "activation_paths": [
            str(SITE_ROOT),
            str(SITE_ROOT / "win32"),
            str(SITE_ROOT / "win32" / "lib"),
            str(SITE_ROOT / "pythonwin"),
        ],
        "site_processing_disabled": True,
        "pth_executed_by_cpython": False,
    }


def _dependency_fixture() -> dict[str, Any]:
    revalidation = _revalidation()
    revalidation_sha = _sha(_canonical(revalidation))
    distributions = {
        name: {
            "distribution": name,
            "version": expected["version"],
            "archive_filename": expected["archive_filename"],
            "archive_sha256": expected["archive_sha256"],
            "file_count": expected["file_count"],
            "total_bytes": expected["total_bytes"],
            "inventory_sha256": expected["inventory_sha256"],
            "actual_files_hashed": True,
            "actual_tree_revalidation_sha256": revalidation_sha,
            "record_metadata_trusted": False,
            "wheel_archive_manifest_authoritative": True,
            "bytecode_excluded": True,
            "locked_version": expected["version"],
            "source_wheel_sha256": expected["archive_sha256"],
        }
        for name, expected in contract.LOCKED_DISTRIBUTIONS.items()
    }
    return {
        "schema_version": 1,
        "kind": "explainiverse-operator-dependency-inventory",
        "target": "CPython 3.13.15 Windows AMD64",
        "lock": {
            "relative_path": contract.RUNTIME_LOCK_RELATIVE,
            "absolute_path": str(ROOT / PurePosixPath(contract.RUNTIME_LOCK_RELATIVE)),
            "bytes": contract.RUNTIME_REQUIREMENTS_BYTES,
            "sha256": contract.RUNTIME_REQUIREMENTS_SHA256,
            "require_hashes": True,
            "wheels_only": True,
        },
        "distributions": distributions,
        "installed_distribution_set_exact": True,
        "startup_pth": {
            "allowed_files": contract.ALLOWED_PTH_FILES,
            "unexpected_files_present": False,
            "all_startup_files_hashed": True,
        },
        "site_manifest": {
            "relative_path": contract.SITE_MANIFEST_RELATIVE,
            "bytes": contract.SITE_MANIFEST_BYTES,
            "sha256": contract.SITE_MANIFEST_SHA256,
        },
        "wheel_derived_site_manifest": revalidation,
    }


def _interpreter_fixture() -> dict[str, Any]:
    paths = [
        PYTHON_ROOT / "python313.zip",
        PYTHON_ROOT,
        SITE_ROOT,
        SITE_ROOT / "win32",
        SITE_ROOT / "win32" / "lib",
        SITE_ROOT / "pythonwin",
    ]
    path_rows: list[dict[str, Any]] = []
    for path in paths:
        row = {
            "absolute_path": str(path),
            "path_sha256": _sha(str(path).encode()),
            "kind": "file" if path.name == "python313.zip" else "directory",
        }
        if path.name == "python313.zip":
            row["content_sha256"] = contract.PYTHON_ZIP_SHA256
        path_rows.append(row)
    resolutions = {}
    for module, distribution in {
        "_cffi_backend": "cffi",
        "cffi": "cffi",
        "cryptography": "cryptography",
        "pycparser": "pycparser",
        "win32api": "pywin32",
        "win32security": "pywin32",
    }.items():
        resolutions[module] = {
            "module": module,
            "distribution": distribution,
            "origin": str(SITE_ROOT / f"{module}.py"),
            "origin_sha256": _sha(module.encode()),
            "search_roots": [],
            "distribution_root": str(SITE_ROOT),
            "distribution_root_sha256": _sha(str(SITE_ROOT).encode()),
            "origin_present_in_hashed_distribution_inventory": True,
        }
    return {
        "secure_flags": {
            "isolated": True,
            "safe_path": True,
            "ignore_environment": True,
            "no_user_site": True,
            "no_site": True,
            "dont_write_bytecode": True,
        },
        "working_directory": str(PureWindowsPath(r"C:\fixture\operator-working")),
        "sys_path_first": str(paths[0]),
        "sys_path": path_rows,
        "sys_path_sha256": _sha(_canonical(path_rows)),
        "repository_present_in_sys_path": False,
        "prefixes": {
            name: {
                "absolute_path": str(PYTHON_ROOT),
                "path_sha256": _sha(str(PYTHON_ROOT).encode()),
            }
            for name in ("prefix", "base_prefix")
        },
        "site_package_roots": [
            {
                "absolute_path": str(SITE_ROOT),
                "path_sha256": _sha(str(SITE_ROOT).encode()),
            }
        ],
        "module_resolutions": resolutions,
        "pinned_runtime_and_site_revalidation": _revalidation(),
    }


def _preloader_fixture(
    phase: str,
    repository: dict[str, Any],
    source: dict[str, Any],
    resources: dict[str, str],
    executables: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    environment = {
        "schema_version": 1,
        "kind": "operator-environment-scrub",
        "removed_name_count": 0,
        "removed_names_sha256": _sha(b""),
        "removed_values_observed": False,
        "ambient_credentials_retained": False,
        "ambient_proxies_retained": False,
    }
    directory_pairs = [_directory_receipt(seed) for seed in ("1", "2", "3", "4")]
    python_install: dict[str, Any] = {
        "schema_version": 1,
        "kind": "explainiverse-operator-python-runtime-installed",
        "python_runtime_root": str(PYTHON_ROOT),
        "archive_sha256": contract.PYTHON_ARCHIVE_SHA256,
        "manifest_sha256": contract.PYTHON_MANIFEST_SHA256,
        "file_count": 34,
        "directory_count": 0,
        "file_inventory_sha256": contract.PYTHON_FILE_INVENTORY_SHA256,
        "owner_private_acl_applied_before_children": True,
        "site_processing_disabled_by_embeddable_pth": True,
        "untracked_files_or_directories_present": False,
        "crash_recovery": "discard-partial-directory-and-create-a-new-path",
    }
    site_install: dict[str, Any] = {
        "schema_version": 1,
        "kind": "explainiverse-operator-runtime-installed",
        "runtime_root": str(SITE_ROOT),
        "manifest_sha256": contract.SITE_MANIFEST_SHA256,
        "file_count": 756,
        "directory_count": 113,
        "file_inventory_sha256": contract.SITE_FILE_INVENTORY_SHA256,
        "owner_private_acl_applied_before_children": True,
        "pip_present_in_runtime": False,
        "record_files_present": False,
        "generated_scripts_present": False,
        "bytecode_present": False,
        "crash_recovery": "discard-partial-directory-and-create-a-new-path",
    }
    python_tree, site_tree = _runtime_trees()
    bootstrap = {
        "schema_version": 1,
        "kind": "explainiverse-operator-pre-site-bootstrap",
        "python_manifest_sha256": contract.PYTHON_MANIFEST_SHA256,
        "python_archive_sha256": contract.PYTHON_ARCHIVE_SHA256,
        "python_tree": python_tree,
        "manifest_sha256": contract.SITE_MANIFEST_SHA256,
        "archive_set_sha256": contract.SITE_ARCHIVE_SET_SHA256,
        "runtime_requirements_sha256": contract.RUNTIME_REQUIREMENTS_SHA256,
        "bootstrap_requirements_sha256": contract.BOOTSTRAP_REQUIREMENTS_SHA256,
        "base_python_executable": executables["python"]["absolute_path"],
        "base_python_executable_sha256": contract.PYTHON_EXECUTABLE_SHA256,
        "preactivation": {
            "working_directory": str(WORKING_ROOT),
            "sys_path_sha256": "4" * 64,
            "only_base_stdlib_roots": True,
        },
        "site_tree": site_tree,
        "activation_paths": _revalidation()["activation_paths"],
        "site_processing_disabled": True,
        "pth_executed_by_cpython": False,
        "verified_pywin32_bootstrap_imported_after_verification": True,
    }
    early_acl = {}
    for name, pair in zip(
        ("python_root", "site_root", "python_receipt_root", "site_receipt_root"),
        directory_pairs,
        strict=True,
    ):
        public = pair[0]
        early_acl[name] = {
            "owner_sid": OWNER_SID,
            "inheritance_protected": True,
            "child_inheritance_enabled": True,
            "allowed_sids": sorted([OWNER_SID, "S-1-5-18", "S-1-5-32-544"]),
            "ace_count": 3,
            "rights": "full-control",
            "security_descriptor_sha256": public["acl"]["security_descriptor_sha256"],
            "security_descriptor_bytes": 128,
            "validated_before_third_party_site_or_third_party_native_import": True,
            "pinned_stdlib_native_modules_loaded_before_hold": True,
        }
    early_material = {
        "schema_version": 1,
        "kind": "explainiverse-operator-early-runtime-boundary",
        "acl": early_acl,
        "held_trees": {
            "root_count": 4,
            "held_handle_count": (
                1
                + int(python_install["file_count"])
                + int(python_install["directory_count"])
                + 1
                + int(site_install["file_count"])
                + int(site_install["directory_count"])
                + 2
                + 2
            ),
            "write_share_allowed": False,
            "delete_share_allowed": False,
            "read_share_allowed": True,
            "held_before_third_party_site_or_third_party_native_import": True,
        },
        "all_runtime_and_receipt_roots_owner_private": True,
        "all_runtime_and_receipt_paths_held_without_write_or_delete_share": True,
        "validated_before_third_party_site_or_third_party_native_import": True,
        "pinned_official_python_runtime_is_the_pre_hold_trust_boundary": True,
        "working_directory": str(WORKING_ROOT),
        "working_directory_repository_disjoint": True,
    }
    early = {**early_material, "evidence_sha256": _sha(_canonical(early_material))}
    sealed = {
        "schema_version": 1,
        "kind": "explainiverse-operator-sealed-resource-binding",
        **resources,
        "runtime_file_sha256": {
            name: repository["critical_sources"][f"scripts/release_gpu_jit_lambda_runtime/{name}"][
                "sha256"
            ]
            for name in contract.RUNTIME_BUNDLE_NAMES
        },
        "captured_before_project_import": True,
        "live_repository_reopen_permitted": False,
    }
    preloader_material = {
        "schema_version": 1,
        "kind": "explainiverse-operator-isolated-preloader",
        "shim": {
            "schema_version": 1,
            "kind": "explainiverse-operator-preloader-shim",
            "preloader_path": source["preloader_path"],
            "preloader_bytes": repository["critical_sources"][contract.PRELOADER_RELATIVE]["bytes"],
            "preloader_sha256": source["preloader_sha256"],
            "shim_sha256": contract.PRELOADER_SHIM_SHA256,
            "stable_descriptor_read": True,
            "compiled_verified_bytes_without_reopen": True,
        },
        "source": source,
        "bootstrap": bootstrap,
        "python_runtime_directory_receipt": directory_pairs[0][0],
        "python_runtime_validation": directory_pairs[0][1],
        "runtime_site_directory_receipt": directory_pairs[1][0],
        "runtime_site_validation": directory_pairs[1][1],
        "python_install_receipt": python_install,
        "python_install_receipt_sha256": _sha(_canonical(python_install)),
        "python_install_directory_receipt": directory_pairs[2][0],
        "python_install_directory_validation": directory_pairs[2][1],
        "site_install_receipt": site_install,
        "site_install_receipt_sha256": _sha(_canonical(site_install)),
        "site_install_directory_receipt": directory_pairs[3][0],
        "site_install_directory_validation": directory_pairs[3][1],
        "environment": environment,
        "early_runtime_boundary": early,
        "sealed_resources": sealed,
        "working_directory": str(WORKING_ROOT),
        "working_directory_is_python_install_receipt_directory": True,
        "isolated": True,
        "safe_path": True,
        "site_disabled": True,
        "bytecode_disabled": True,
        "repository_absent_from_sys_path": True,
        "project_imports_from_captured_bytes": True,
    }
    preloader = {
        **preloader_material,
        "evidence_sha256": _sha(_canonical(preloader_material)),
    }
    return preloader, environment


def _immutable_plan_fixture(resources: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "explainiverse-lambda-live-plan",
        "openapi": {
            "openapi_version": "3.1.0",
            "api_version": "1.10.0",
            "document_sha256": "2e00f2884d043fa2377a1a6f898eba4b81d8b0c4546d5d98079c7faa4451ba8f",
            "production_origin": "https://cloud.lambda.ai",
        },
        "repository": contract.REPOSITORY,
        "head_sha": HEAD,
        "lifecycle_nonce": LIFECYCLE_NONCE,
        "created_at_unix": 1_787_918_400,
        "expires_at_unix": 1_787_920_200,
        "controller_source": "8.8.8.8/32",
        "target": {
            "instance_type_name": "gpu_8x_a100_80gb_sxm4",
            "instance_type_description": "NVIDIA A100 80 GB SXM4",
            "gpu_description": "NVIDIA A100 80 GB SXM4",
            "physical_gpu_count": 8,
            "architecture": "x86_64",
            "price_cents_per_hour": 200,
            "vcpus": 30,
            "memory_gib": 200,
            "storage_gib": 1400,
            "region_name": contract.TARGET_REGION,
            "region_description": contract.TARGET_REGION_DESCRIPTION,
            "image": {
                "id": "image-fixture",
                "created_time": "2026-01-01T00:00:00+00:00",
                "description": "fixture",
                "name": "lambda-stack-fixture",
                "family": "lambda-stack-22-04",
                "version": "1",
                "updated_time": "2026-01-02T00:00:00+00:00",
                "architecture": "x86_64",
                "region_name": contract.TARGET_REGION,
            },
        },
        "ssh_access": {
            "key_name": "fixture-key",
            "public_key_sha256": "7" * 64,
            "ephemeral_host_key_fingerprint": "SHA256:" + "A" * 43,
        },
        "remote_runtime": {
            "bundle_sha256": resources["runtime_bundle_sha256"],
            "bundle_files": list(contract.RUNTIME_BUNDLE_NAMES),
            "install_root": contract.REMOTE_RUNTIME_ROOT,
            "fixed_cloud_init_wait_command": list(contract.FIXED_CLOUD_INIT_WAIT_COMMAND),
            "fixed_preflight_command": list(contract.FIXED_PREFLIGHT_COMMAND),
            "fixed_command": list(contract.FIXED_REMOTE_COMMAND),
            "fixed_command_contains_dynamic_or_plan_values": False,
        },
        "baseline_file_systems_sha256": "8" * 64,
        "original_global_rules": [
            {
                "protocol": "icmp",
                "source_network": "0.0.0.0/0",
                "description": "fixture",
            }
        ],
        "desired_global_and_instance_rules": [
            {
                "protocol": "tcp",
                "port_range": [22, 22],
                "source_network": "8.8.8.8/32",
                "description": f"Explainiverse {LIFECYCLE_NONCE} controller SSH",
            }
        ],
        "ownership_tags": [
            {"key": "explainiverse-lifecycle-nonce", "value": LIFECYCLE_NONCE},
            {"key": "explainiverse-owner", "value": contract.REPOSITORY},
            {"key": "explainiverse-purpose", "value": "stable-release-cuda"},
            {"key": "explainiverse-source-sha", "value": HEAD},
        ],
        "mutation_order": list(contract.MUTATION_ORDER),
        "secret_transport": {
            "lambda_api_key": "anonymous-fd-or-stdin-only",
            "github_jit_config": "anonymous-fd-or-stdin-only",
            "host_private_key": "in-memory-cloud-init-only",
        },
        "production_authorized": False,
        "provider_mutation_authorized": False,
        "live_go": False,
    }


def _discovery_fixture() -> dict[str, Any]:
    operations = (
        ("instances", "/api/v1/instances"),
        ("file_systems", "/api/v1/file-systems"),
        ("ssh_keys", "/api/v1/ssh-keys"),
        ("instance_types", "/api/v1/instance-types"),
        ("images", "/api/v1/images"),
        ("regions", "/api/v1/regions"),
        ("global_firewall", "/api/v1/firewall-rulesets/global"),
        ("firewall_rulesets", "/api/v1/firewall-rulesets"),
    )
    payload_digests = {
        operation: _sha(f"payload:{operation}".encode()) for operation, _ in operations
    }
    response_bindings = [
        {
            "operation": operation,
            "method": "GET",
            "path": path,
            "request_sha256": _sha(f"request:{operation}".encode()),
            "request_body_sha256": None,
            "response_body_sha256": _sha(f"response:{operation}".encode()),
            "status_code": 200,
            "content_type": "application/json",
        }
        for index, (operation, path) in enumerate(operations, 1)
    ]
    snapshot_sha256 = _sha(
        _live_canonical({"payload_digests": payload_digests, "bindings": response_bindings})
    )
    target = {
        "instance_type_name": "gpu_8x_a100_80gb_sxm4",
        "instance_type_description": "NVIDIA A100 80 GB SXM4",
        "gpu_description": "NVIDIA A100 80 GB SXM4",
        "price_cents_per_hour": 200,
        "vcpus": 30,
        "memory_gib": 200,
        "storage_gib": 1400,
        "gpus": 8,
        "architecture": "x86_64",
        "capacity_region": contract.TARGET_REGION,
        "region_description": contract.TARGET_REGION_DESCRIPTION,
    }
    images = [
        {
            "id": "image-fixture",
            "created_time": "2026-01-01T00:00:00+00:00",
            "updated_time": "2026-01-02T00:00:00+00:00",
            "name": "lambda-stack-fixture",
            "description": "fixture",
            "family": "lambda-stack-22-04",
            "version": "1",
            "architecture": "x86_64",
            "region": {
                "name": contract.TARGET_REGION,
                "description": contract.TARGET_REGION_DESCRIPTION,
            },
        }
    ]
    ssh = {"key_name": "fixture-key", "public_key_sha256": "7" * 64}
    rules = [
        {
            "protocol": "icmp",
            "source_network": "0.0.0.0/0",
            "description": "fixture",
        }
    ]
    binding_material = {
        "snapshot_sha256": snapshot_sha256,
        "region_description": target["region_description"],
        "instance_type_description": target["instance_type_description"],
        "gpu_description": target["gpu_description"],
        "price_cents_per_hour": target["price_cents_per_hour"],
        "vcpus": target["vcpus"],
        "memory_gib": target["memory_gib"],
        "storage_gib": target["storage_gib"],
        "images": images,
        "ssh_key_name": ssh["key_name"],
        "ssh_public_key_sha256": ssh["public_key_sha256"],
        "baseline_file_systems_sha256": "8" * 64,
        "original_global_rules": rules,
    }
    return {
        "snapshot_sha256": snapshot_sha256,
        "binding_sha256": _sha(_live_canonical(binding_material)),
        "payload_digests": payload_digests,
        "response_bindings": response_bindings,
        "zero_instances": True,
        "zero_firewall_rulesets": True,
        "target": target,
        "image_candidates": images,
        "ssh_access": ssh,
        "baseline_file_systems_sha256": "8" * 64,
        "original_global_rules": rules,
    }


def operator_preflight_fixture(
    phase: str,
    *,
    policy_sha256: str | None = None,
    controller_source_sha256: str | None = None,
    runtime_bundle_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository, source, resources = _source_fixture(
        phase,
        policy_sha256=policy_sha256,
        controller_source_sha256=controller_source_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
    )
    immutable_plan = _immutable_plan_fixture(resources)
    plan_sha256 = _sha(_live_canonical(immutable_plan))
    executables = _executable_fixture()
    preloader, environment = _preloader_fixture(phase, repository, source, resources, executables)
    secure_launch = {
        "schema_version": 1,
        "kind": "operator-secure-interpreter-launch",
        "isolated": True,
        "safe_path": True,
        "ignore_environment": True,
        "no_user_site": True,
        "no_site": True,
        "dont_write_bytecode": True,
        "invocation": "pinned-python -I -S -B -c <byte-sealing-shim>",
        "working_directory": str(WORKING_ROOT),
        "repository_absent_from_sys_path": True,
        "sys_path_sha256": "9" * 64,
        "site_processing_disabled": True,
        "preloader": preloader,
        "controller_imported_after_launch_validation": True,
        "windows_handle_transport": True,
        "inherited_handle_count": 2,
        "handles_distinct": True,
        "file_type_pipe": True,
        "child_handles_made_noninheritable": True,
        "raw_handle_values_archived": False,
        "secret_values_in_argv": False,
        "secret_values_in_environment": False,
        "windows_launcher_parent_declaration": {
            "receipt_sha256": "a" * 64,
            "preloader_metadata_matched": True,
            "parent_provenance_authenticated": False,
            "security_authority_derived_from_declaration": False,
            "child_revalidated_handle_transport_and_sealed_resources": True,
        },
    }
    inventory = {
        "schema_version": 1,
        "kind": "explainiverse-lambda-operator-inventory",
        "python_implementation": "CPython",
        "interpreter_runtime": _interpreter_fixture(),
        "executables": executables,
        "dependencies": _dependency_fixture(),
        "repository": repository,
    }
    inventory_sha256 = _sha(_canonical(inventory))
    inbox = {
        "phase": phase,
        "expected_capture_count": contract.PHASE_CAPTURE_COUNTS[phase],
        "accepted_capture_count": 0,
        "stale_generation_count": 0,
        "stale_generations_sha256": _sha(_canonical([])),
        "owner_private_directory_receipt_sha256": "b" * 64,
        "on_demand_before_each_jit": True,
        "ready_marker_no_replace_required": True,
        "raw_pages_archived_by_driver": True,
    }
    anonymous = {
        "kind": "anonymous-pipe",
        "descriptor_owned_by_operator": True,
        "current_user_owner_verified": True,
        "regular_file": False,
        "terminal": False,
        "value_archived": False,
    }
    acceptance = (
        {
            "loader_verified": True,
            "evidence_sha256": "c" * 64,
            "head_sha": HEAD,
            "run_id": 123,
        }
        if phase == "publication"
        else None
    )
    preflight = {
        "schema_version": 1,
        "kind": "explainiverse-lambda-operator-preflight",
        "plan_sha256": plan_sha256,
        "head_sha": HEAD,
        "lifecycle_nonce": LIFECYCLE_NONCE,
        "discovery": _discovery_fixture(),
        "inspection_receipt_sha256": INSPECTION_SHA256,
        "inventory": inventory,
        "inventory_sha256": inventory_sha256,
        "executables": executables,
        "repository": repository,
        "environment": environment,
        "secure_launch": secure_launch,
        "lambda_secret_transport": anonymous,
        "plan_confirmation": {
            **anonymous,
            "confirmed_plan_sha256": plan_sha256,
            "confirmation_exact_line": True,
            "confirmation_read_after_plan": True,
        },
        "app_capture_inbox": inbox,
        "final_main_acceptance": acceptance,
        "live_gates_not_constructed_before_confirmation": True,
        "direct_publication_dispatch_exposed": False,
    }
    expected = {
        "expected_immutable_plan": immutable_plan,
        "expected_phase": phase,
        "expected_head_sha": HEAD,
        "expected_ref": contract.PHASE_REFS[phase],
        "expected_plan_sha256": plan_sha256,
        "expected_lifecycle_nonce": LIFECYCLE_NONCE,
        "expected_inspection_receipt_sha256": INSPECTION_SHA256,
        "expected_inventory_sha256": inventory_sha256,
        "expected_policy_sha256": resources["policy_sha256"],
        "expected_controller_source_sha256": resources["controller_source_sha256"],
        "expected_runtime_bundle_sha256": resources["runtime_bundle_sha256"],
    }
    return preflight, expected
