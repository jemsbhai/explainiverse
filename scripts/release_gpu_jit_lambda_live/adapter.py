"""Fail-closed live adapter for one disposable Lambda Cloud GPU instance.

This module deliberately owns only the provider boundary and the secret
transport primitives shared with the remote runner executor.  It does not
create API keys, dispatch GitHub workflows, register runners, or execute SSH.

The implementation is intentionally narrow:

* Lambda Cloud OpenAPI 1.10.0 and its exact production origin are pinned.
* The only allowed compute target is ``gpu_8x_a100_80gb_sxm4`` in
  ``us-midwest-1`` (Illinois, USA). H100 and every filesystem attachment are
  rejected.
* A Lambda API key and a GitHub encoded JIT configuration can enter only from
  stdin or another anonymous file descriptor.  Secret values have redacted
  representations and are never placed in argv, environment variables,
  evidence, or exception text.
* Every provider mutation consumes a fresh, single-use, validated prestate
  receipt.  Mutations are never retried.  A transport or response ambiguity
  must be resolved through a new inventory observation.
* A per-lifecycle Ed25519 SSH host identity is generated in memory.  Its
  private material appears only in the TLS-protected launch request's
  cloud-init ``ssh_keys`` payload and is best-effort zeroized afterward.

No live action occurs merely by importing this module or constructing a plan.
``production_authorized`` and ``provider_mutation_authorized`` are separate
action-time gates and both must bind the exact immutable plan digest.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import ssl
import stat
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, NoReturn, Sequence

OPENAPI_VERSION = "3.1.0"
LAMBDA_API_VERSION = "1.10.0"
OPENAPI_SHA256 = "2e00f2884d043fa2377a1a6f898eba4b81d8b0c4546d5d98079c7faa4451ba8f"
PRODUCTION_ORIGIN = "https://cloud.lambda.ai"
API_PREFIX = "/api/v1"

TARGET_INSTANCE_TYPE = "gpu_8x_a100_80gb_sxm4"
TARGET_REGION = "us-midwest-1"
TARGET_REGION_DESCRIPTION = "Illinois, USA"
TARGET_ARCHITECTURE = "x86_64"
TARGET_GPU_COUNT = 8
TARGET_IMAGE_FAMILY = "lambda-stack-22-04"
REPOSITORY = "jemsbhai/explainiverse"
RUNTIME_BUNDLE_NAMES = ("__init__.py", "bootstrap.py", "executor.py", "runtime_contract.py")
REMOTE_RUNTIME_ROOT = "/opt/explainiverse/bin/release_gpu_jit_lambda_runtime"
FIXED_REMOTE_COMMAND = (
    "/usr/bin/sudo",
    "-n",
    "--",
    "/usr/bin/python3",
    "-B",
    f"{REMOTE_RUNTIME_ROOT}/bootstrap.py",
)
FIXED_CLOUD_INIT_WAIT_COMMAND = (
    "/usr/bin/sudo",
    "-n",
    "--",
    "/usr/bin/cloud-init",
    "status",
    "--wait",
)
FIXED_PREFLIGHT_COMMAND = (
    "/usr/bin/sudo",
    "-n",
    "--",
    "/usr/bin/python3",
    "-B",
    f"{REMOTE_RUNTIME_ROOT}/executor.py",
    "probe-host",
)

MAX_API_KEY_BYTES = 4096
MAX_JIT_CONFIG_BYTES = 1_048_576
MIN_JIT_CONFIG_BYTES = 100
SECRET_FD_READ_SECONDS = 30
MAX_PROVIDER_RESPONSE_BYTES = 1_048_576
PROVIDER_TIMEOUT_SECONDS = 20
PROVIDER_MIN_REQUEST_INTERVAL_SECONDS = 1.0
RUNTIME_FRAME_WRITE_SECONDS = 30
MAX_OBSERVATION_WINDOW_SECONDS = 30
PRESTATE_FRESHNESS_SECONDS = 45
MAX_PLAN_LIFETIME_SECONDS = 4 * 60 * 60
HOST_PREFLIGHT_FRESHNESS_SECONDS = 300
MAX_PROVIDER_CLOCK_SKEW_SECONDS = 300

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
RESOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
IMAGE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@-]{2,255}\Z")
SSH_PUBLIC_RE = re.compile(r"ssh-ed25519 ([A-Za-z0-9+/]+={0,2})(?: [^\r\n]+)?\Z")
SSH_FINGERPRINT_RE = re.compile(r"SHA256:[A-Za-z0-9+/]{43}\Z")
GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-" r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z"
)

EXPECTED_HOST_GPU_PRODUCT = "NVIDIA A100-SXM4-80GB"
EXPECTED_RUNNER_IMAGE_REFERENCE = (
    "ghcr.io/actions/actions-runner@"
    "sha256:a1919047b038c38871d667c58cfdc7a878452711ab1212fb6036188f27a7ab16"
)
EXPECTED_RUNNER_IMAGE_MANIFEST = (
    "sha256:a1919047b038c38871d667c58cfdc7a878452711ab1212fb6036188f27a7ab16"
)
EXPECTED_RUNNER_IMAGE_CONFIG = (
    "sha256:bd6fe162bb4ab4821daa8d694e20d779865618825d30c94342a0228b89947305"
)
EXPECTED_RUNNER_IMAGE_PLATFORM = "linux/amd64"
EXPECTED_RUNNER_VERSION = "2.336.0"
EXPECTED_RUNNER_COMMIT = "98aabcd429c4e8402406c56ce2d26387fed3b9ce"
EXPECTED_NODE20_VERSION = "v20.20.2"
EXPECTED_NODE20_SHA256 = "6295488653f0d93b0a157841746fef7e72cc4328cfb60c4bbe0ca2668a836ffd"

READ_OPERATIONS: tuple[tuple[str, str], ...] = (
    ("instances", f"{API_PREFIX}/instances"),
    ("file_systems", f"{API_PREFIX}/file-systems"),
    ("ssh_keys", f"{API_PREFIX}/ssh-keys"),
    ("instance_types", f"{API_PREFIX}/instance-types"),
    ("images", f"{API_PREFIX}/images"),
    ("regions", f"{API_PREFIX}/regions"),
    ("global_firewall", f"{API_PREFIX}/firewall-rulesets/global"),
    ("firewall_rulesets", f"{API_PREFIX}/firewall-rulesets"),
)

MUTATION_PATHS = MappingProxyType(
    {
        "restrict_global": ("PATCH", f"{API_PREFIX}/firewall-rulesets/global"),
        "create_ruleset": ("POST", f"{API_PREFIX}/firewall-rulesets"),
        "launch": ("POST", f"{API_PREFIX}/instance-operations/launch"),
        "terminate": ("POST", f"{API_PREFIX}/instance-operations/terminate"),
        "delete_ruleset": ("DELETE", f"{API_PREFIX}/firewall-rulesets/{{id}}"),
        "restore_global": ("PATCH", f"{API_PREFIX}/firewall-rulesets/global"),
    }
)

MUTATION_PRESTATE = MappingProxyType(
    {
        "restrict_global": "baseline",
        "create_ruleset": "global_restricted",
        "launch": "ruleset_ready",
        "terminate": "instance_bound",
        "delete_ruleset": "instance_absent",
        "restore_global": "ruleset_absent",
    }
)


class ContractError(RuntimeError):
    """A stable fail-closed contract violation with no secret-bearing text."""


class TransportFailure(ContractError):
    """A read-only provider request failed without a usable bound response."""


class AmbiguousMutation(ContractError):
    """A one-shot mutation may have taken effect and must be inventoried."""

    def __init__(self, operation: str, request_sha256: str, reason_code: str) -> None:
        super().__init__(f"ambiguous_{operation}_{reason_code}")
        self.operation = operation
        self.request_sha256 = request_sha256
        self.reason_code = reason_code


def _fail(code: str) -> NoReturn:
    raise ContractError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _strict_json_loads(value: bytes) -> Any:
    """Decode RFC JSON while rejecting duplicate keys and non-finite numbers."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = item
        return result

    def reject_constant(_: str) -> NoReturn:
        raise ValueError("non_finite_json_number")

    return json.loads(value, object_pairs_hook=object_pairs, parse_constant=reject_constant)


def _sha256(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_keys(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{context}_not_object")
    _require(set(value) == keys, f"{context}_keys_rejected")
    return value


def _required_keys(
    value: Any, required: set[str], allowed: set[str], context: str
) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{context}_not_object")
    actual = set(value)
    _require(required <= actual, f"{context}_required_keys_missing")
    _require(actual <= allowed, f"{context}_unknown_keys_rejected")
    return value


def _text(value: Any, context: str, *, allow_empty: bool = False) -> str:
    _require(type(value) is str, f"{context}_not_text")
    _require(allow_empty or bool(value), f"{context}_empty")
    _require("\x00" not in value and "\r" not in value and "\n" not in value, f"{context}_control")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{context}_not_integer")
    return value


def _utc_timestamp(value: str, context: str) -> datetime:
    _text(value, context)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"{context}_invalid")
    _require(parsed.tzinfo is not None, f"{context}_timezone_missing")
    return parsed.astimezone(timezone.utc)


def _ed25519_public_key_blob(value: Any, context: str) -> bytes:
    public_key = _text(value, context)
    match = SSH_PUBLIC_RE.fullmatch(public_key)
    _require(match is not None, f"{context}_not_ed25519")
    assert match is not None
    try:
        key_blob = base64.b64decode(match.group(1), validate=True)
    except (ValueError, binascii.Error):
        _fail(f"{context}_encoding_rejected")
    expected_prefix = (
        struct.pack(">I", len(b"ssh-ed25519")) + b"ssh-ed25519" + struct.pack(">I", 32)
    )
    _require(
        len(key_blob) == len(expected_prefix) + 32 and key_blob.startswith(expected_prefix),
        f"{context}_blob_rejected",
    )
    return key_blob


def _canonical_ed25519_public_key(value: Any, context: str) -> str:
    key_blob = _ed25519_public_key_blob(value, context)
    return "ssh-ed25519 " + base64.b64encode(key_blob).decode("ascii")


class SecretBuffer:
    """A mutable, best-effort zeroizable secret with a permanently redacted repr."""

    __slots__ = ("_value", "_destroyed", "_label")

    def __init__(self, value: bytes | bytearray, *, label: str) -> None:
        _require(bool(value), f"{label}_empty")
        self._value = bytearray(value)
        self._destroyed = False
        self._label = label

    def __repr__(self) -> str:
        return f"SecretBuffer(label={self._label!r}, value=<redacted>)"

    def __enter__(self) -> SecretBuffer:
        _require(not self._destroyed, f"{self._label}_destroyed")
        return self

    def __exit__(self, *_: object) -> None:
        self.destroy()

    def copy_bytes(self) -> bytes:
        _require(not self._destroyed, f"{self._label}_destroyed")
        return bytes(self._value)

    def view(self) -> memoryview:
        _require(not self._destroyed, f"{self._label}_destroyed")
        return memoryview(self._value)

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def destroy(self) -> None:
        if self._destroyed:
            return
        for index in range(len(self._value)):
            self._value[index] = 0
        self._value.clear()
        self._destroyed = True


def _read_secret_fd(fd: int, *, maximum: int, label: str) -> SecretBuffer:
    _require(type(fd) is int and (fd == 0 or fd >= 3), f"{label}_fd_rejected")
    descriptor_mode = os.fstat(fd).st_mode
    _require(not stat.S_ISREG(descriptor_mode), f"{label}_regular_file_rejected")
    _require(not os.isatty(fd), f"{label}_terminal_rejected")
    try:
        os.set_blocking(fd, False)
    except OSError:
        _fail(f"{label}_nonblocking_unavailable")
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + SECRET_FD_READ_SECONDS
    while True:
        try:
            chunk = os.read(fd, min(65_536, maximum + 1 - total))
        except BlockingIOError:
            if time.monotonic() >= deadline:
                for item in chunks:
                    del item
                _fail(f"{label}_read_timeout")
            time.sleep(0.01)
            continue
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            for item in chunks:
                # Immutable Python byte strings cannot be reliably zeroized.
                del item
            _fail(f"{label}_too_large")
    value = b"".join(chunks)
    _require(bool(value), f"{label}_empty")
    return SecretBuffer(value, label=label)


def read_jit_config_from_fd(fd: int) -> SecretBuffer:
    """Read one opaque GitHub JIT config from stdin/anonymous FD only."""

    secret = _read_secret_fd(fd, maximum=MAX_JIT_CONFIG_BYTES, label="jit_config")
    try:
        encoded = secret.copy_bytes()
        _require(len(encoded) >= MIN_JIT_CONFIG_BYTES, "jit_config_too_small")
        _require(b"\x00" not in encoded, "jit_config_nul_rejected")
        # GitHub currently emits base64.  Validate without decoding or exposing it.
        base64.b64decode(encoded, validate=True)
        return secret
    except ContractError:
        secret.destroy()
        raise
    except (ValueError, binascii.Error):
        secret.destroy()
        raise ContractError("jit_config_encoding_rejected") from None


RUNTIME_FRAME_MAGIC = b"EXJIT01\n"
RUNTIME_FRAME_VERSION = 1
RUNTIME_FRAME_FLAGS = 0
RUNTIME_FRAME_HEADER = struct.Struct(">8sHHII32s32s")


@dataclass(frozen=True)
class RuntimeFrameReceipt:
    version: int
    flags: int
    plan_bytes: int
    jit_config_bytes: int
    plan_sha256: str
    jit_config_sha256: str
    header_sha256: str

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "magic": RUNTIME_FRAME_MAGIC.decode("ascii").rstrip("\n"),
            "version": self.version,
            "flags": self.flags,
            "header_bytes": RUNTIME_FRAME_HEADER.size,
            "plan_bytes": self.plan_bytes,
            "jit_config_bytes": self.jit_config_bytes,
            "plan_sha256": self.plan_sha256,
            "jit_config_sha256": self.jit_config_sha256,
            "header_sha256": self.header_sha256,
            "trailing_bytes_permitted": False,
            "remote_argv_contains_plan_or_jit_values": False,
        }


def write_runtime_frame_and_close(
    output_fd: int, *, canonical_plan: bytes, jit_config: SecretBuffer
) -> RuntimeFrameReceipt:
    """Write the exact remote-bootstrap frame to anonymous SSH stdin.

    The destination is always closed and the local JIT buffer is always
    destroyed, including on a short write or broken pipe.  The remote bootstrap
    is responsible for requiring EOF immediately after the declared payloads.
    """

    _require(type(output_fd) is int and output_fd >= 3, "runtime_output_fd_rejected")
    jit_view: memoryview | None = None
    try:
        mode = os.fstat(output_fd).st_mode
        _require(not stat.S_ISREG(mode), "runtime_output_regular_file_rejected")
        _require(type(canonical_plan) is bytes, "runtime_plan_not_bytes")
        _require(
            1 <= len(canonical_plan) <= MAX_JIT_CONFIG_BYTES,
            "runtime_plan_size_rejected",
        )
        try:
            parsed_plan = _strict_json_loads(canonical_plan)
        except (UnicodeDecodeError, ValueError):
            _fail("runtime_plan_json_rejected")
        _require(type(parsed_plan) is dict, "runtime_plan_not_object")
        _require(_canonical_json(parsed_plan) == canonical_plan, "runtime_plan_not_canonical")
        jit_view = jit_config.view()
        _require(
            MIN_JIT_CONFIG_BYTES <= len(jit_view) <= MAX_JIT_CONFIG_BYTES,
            "runtime_jit_size_rejected",
        )
        plan_digest = hashlib.sha256(canonical_plan).digest()
        jit_digest = hashlib.sha256(jit_view).digest()
        header = RUNTIME_FRAME_HEADER.pack(
            RUNTIME_FRAME_MAGIC,
            RUNTIME_FRAME_VERSION,
            RUNTIME_FRAME_FLAGS,
            len(canonical_plan),
            len(jit_view),
            plan_digest,
            jit_digest,
        )

        if os.name == "nt":
            import msvcrt

            msvcrt.setmode(output_fd, os.O_BINARY)  # type: ignore[attr-defined]
        os.set_blocking(output_fd, False)
        write_deadline = time.monotonic() + RUNTIME_FRAME_WRITE_SECONDS

        def write_all(value: bytes | memoryview) -> None:
            offset = 0
            while offset < len(value):
                try:
                    written = os.write(output_fd, value[offset:])
                except BlockingIOError:
                    _require(
                        time.monotonic() < write_deadline,
                        "runtime_frame_write_timeout",
                    )
                    time.sleep(0.01)
                    continue
                if written <= 0:
                    _fail("runtime_frame_short_write")
                offset += written

        write_all(header)
        write_all(canonical_plan)
        write_all(jit_view)
        return RuntimeFrameReceipt(
            version=RUNTIME_FRAME_VERSION,
            flags=RUNTIME_FRAME_FLAGS,
            plan_bytes=len(canonical_plan),
            jit_config_bytes=len(jit_view),
            plan_sha256=plan_digest.hex(),
            jit_config_sha256=jit_digest.hex(),
            header_sha256=_sha256(header),
        )
    except (BrokenPipeError, OSError):
        raise ContractError("runtime_frame_transport_failure") from None
    finally:
        if jit_view is not None:
            jit_view.release()
        jit_config.destroy()
        os.close(output_fd)


def _validate_a100_api_description(value: Any) -> str:
    """Bind the live API string while rejecting any non-A100-80GB-SXM4 target."""

    description = _text(value, "gpu_description")
    normalized = re.sub(r"[^A-Z0-9]+", " ", description.upper()).split()
    _require("A100" in normalized, "gpu_description_not_a100")
    _require("80" in normalized and "GB" in normalized, "gpu_description_not_80gb")
    _require("SXM4" in normalized, "gpu_description_not_sxm4")
    _require("H100" not in normalized, "gpu_description_h100_rejected")
    return description


@dataclass(frozen=True)
class FirewallRule:
    protocol: str
    port_range: tuple[int, int] | None
    source_network: str
    description: str

    def __post_init__(self) -> None:
        _require(self.protocol in {"tcp", "udp", "icmp"}, "firewall_protocol_rejected")
        network = ipaddress.ip_network(self.source_network, strict=True)
        _require(type(network) is ipaddress.IPv4Network, "firewall_source_not_ipv4")
        _text(self.description, "firewall_description", allow_empty=True)
        _require(len(self.description) <= 128, "firewall_description_too_long")
        if self.protocol == "icmp":
            _require(self.port_range is None, "icmp_port_range_rejected")
        else:
            port_range = self.port_range
            if type(port_range) is not tuple or len(port_range) != 2:
                _fail("port_range_missing")
            lower, upper = port_range
            _require(
                type(lower) is int and type(upper) is int and 1 <= lower <= upper <= 65535,
                "port_range_rejected",
            )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "protocol": self.protocol,
            "source_network": self.source_network,
            "description": self.description,
        }
        if self.port_range is not None:
            result["port_range"] = list(self.port_range)
        return result

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> FirewallRule:
        mapping = _required_keys(
            value,
            {"protocol", "source_network", "description"},
            {"protocol", "port_range", "source_network", "description"},
            context,
        )
        raw_range = mapping.get("port_range")
        port_range: tuple[int, int] | None
        if raw_range is None:
            port_range = None
        else:
            _require(
                type(raw_range) is list and len(raw_range) == 2,
                f"{context}_port_range_rejected",
            )
            port_range = (
                _integer(raw_range[0], f"{context}_port_min", minimum=1),
                _integer(raw_range[1], f"{context}_port_max", minimum=1),
            )
        return cls(
            protocol=_text(mapping["protocol"], f"{context}_protocol"),
            port_range=port_range,
            source_network=_text(mapping["source_network"], f"{context}_source"),
            description=_text(mapping["description"], f"{context}_description", allow_empty=True),
        )


@dataclass(frozen=True)
class RuntimeBundle:
    """Exact public remote runtime source, kept separate from every secret."""

    files: tuple[tuple[str, bytes], ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require(
            tuple(name for name, _ in self.files) == RUNTIME_BUNDLE_NAMES,
            "runtime_bundle_names_or_order_rejected",
        )
        total = 0
        for name, content in self.files:
            _require(type(content) is bytes and bool(content), f"runtime_bundle_{name}_empty")
            _require(len(content) <= 512_000, f"runtime_bundle_{name}_too_large")
            total += len(content)
        _require(total <= 900_000, "runtime_bundle_too_large")

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        for name, content in self.files:
            encoded_name = name.encode("ascii")
            digest.update(struct.pack(">H", len(encoded_name)))
            digest.update(encoded_name)
            digest.update(struct.pack(">Q", len(content)))
            digest.update(content)
        return digest.hexdigest()


def load_runtime_bundle(root: str | os.PathLike[str]) -> RuntimeBundle:
    """Read the exact four-file audited runtime bundle without path traversal."""

    directory = Path(root)
    _require(directory.is_absolute(), "runtime_bundle_root_not_absolute")
    _require(directory.is_dir() and not directory.is_symlink(), "runtime_bundle_root_rejected")
    files: list[tuple[str, bytes]] = []
    for name in RUNTIME_BUNDLE_NAMES:
        path = directory / name
        _require(path.is_file() and not path.is_symlink(), f"runtime_bundle_{name}_rejected")
        files.append((name, path.read_bytes()))
    return RuntimeBundle(tuple(files))


def runtime_bundle_from_captured_files(
    files: Mapping[str, bytes],
    *,
    expected_bundle_sha256: str,
) -> RuntimeBundle:
    """Build the runtime only from preloader-held bytes, never repository paths."""

    _require(
        type(files) is dict
        and set(files) == set(RUNTIME_BUNDLE_NAMES)
        and type(expected_bundle_sha256) is str
        and SHA256_RE.fullmatch(expected_bundle_sha256) is not None,
        "captured_runtime_bundle_mapping_rejected",
    )
    bundle = RuntimeBundle(tuple((name, bytes(files[name])) for name in RUNTIME_BUNDLE_NAMES))
    _require(
        bundle.sha256 == expected_bundle_sha256,
        "captured_runtime_bundle_digest_mismatch",
    )
    return bundle


@dataclass(frozen=True)
class ImmutablePlan:
    """Exact action-time target.  Gate flags remain false in the artifact."""

    head_sha: str
    lifecycle_nonce: str
    created_at_unix: int
    expires_at_unix: int
    current_public_ipv4_cidr: str
    region_description: str
    image_id: str
    image_created_time: str
    image_description: str
    image_name: str
    image_family: str
    image_version: str
    image_updated_time: str
    instance_type_description: str
    gpu_description: str
    price_cents_per_hour: int
    vcpus: int
    memory_gib: int
    storage_gib: int
    ssh_key_name: str
    ssh_public_key_sha256: str
    baseline_file_systems_sha256: str
    original_global_rules: tuple[FirewallRule, ...]
    host_key_fingerprint: str
    runtime_bundle_sha256: str
    production_authorized: bool = False
    provider_mutation_authorized: bool = False

    def __post_init__(self) -> None:
        _require(COMMIT_RE.fullmatch(self.head_sha) is not None, "head_sha_rejected")
        _require(NONCE_RE.fullmatch(self.lifecycle_nonce) is not None, "lifecycle_nonce_rejected")
        _require(type(self.created_at_unix) is int, "plan_created_at_rejected")
        _require(type(self.expires_at_unix) is int, "plan_expires_at_rejected")
        _require(self.expires_at_unix > self.created_at_unix, "plan_time_order_rejected")
        _require(
            self.expires_at_unix - self.created_at_unix <= MAX_PLAN_LIFETIME_SECONDS,
            "plan_lifetime_rejected",
        )
        network = ipaddress.ip_network(self.current_public_ipv4_cidr, strict=True)
        _require(
            type(network) is ipaddress.IPv4Network and network.prefixlen == 32,
            "controller_source_not_ipv4_32",
        )
        _require(network.network_address.is_global, "controller_source_not_public")
        _require(
            self.region_description == TARGET_REGION_DESCRIPTION,
            "region_description_rejected",
        )
        _require(IMAGE_ID_RE.fullmatch(self.image_id) is not None, "image_id_rejected")
        image_created = _utc_timestamp(self.image_created_time, "image_created_time")
        _text(self.image_description, "image_description", allow_empty=True)
        _text(self.image_name, "image_name")
        _text(self.image_family, "image_family")
        _require(
            self.image_family == TARGET_IMAGE_FAMILY,
            "image_family_not_lambda_stack_22_04",
        )
        _text(self.image_version, "image_version")
        image_updated = _utc_timestamp(self.image_updated_time, "image_updated_time")
        _require(image_created <= image_updated, "image_timestamp_order_rejected")
        _require(
            image_updated
            <= datetime.now(timezone.utc) + timedelta(seconds=MAX_PROVIDER_CLOCK_SKEW_SECONDS),
            "image_timestamp_in_future",
        )
        _text(self.instance_type_description, "instance_type_description")
        _validate_a100_api_description(self.gpu_description)
        _integer(self.price_cents_per_hour, "price_cents_per_hour", minimum=1)
        _integer(self.vcpus, "vcpus", minimum=1)
        _integer(self.memory_gib, "memory_gib", minimum=1)
        _integer(self.storage_gib, "storage_gib", minimum=1)
        _text(self.ssh_key_name, "ssh_key_name")
        _require(
            SHA256_RE.fullmatch(self.ssh_public_key_sha256) is not None,
            "ssh_public_key_sha256_rejected",
        )
        _require(
            SHA256_RE.fullmatch(self.baseline_file_systems_sha256) is not None,
            "baseline_file_systems_sha256_rejected",
        )
        _require(bool(self.original_global_rules), "original_global_rules_empty")
        _require(
            SSH_FINGERPRINT_RE.fullmatch(self.host_key_fingerprint) is not None,
            "host_key_fingerprint_rejected",
        )
        _require(
            SHA256_RE.fullmatch(self.runtime_bundle_sha256) is not None,
            "runtime_bundle_sha256_rejected",
        )
        _require(self.production_authorized is False, "plan_production_gate_must_be_false")
        _require(
            self.provider_mutation_authorized is False,
            "plan_provider_gate_must_be_false",
        )

    @property
    def ruleset_name(self) -> str:
        return f"explainiverse-{self.lifecycle_nonce}"

    @property
    def instance_name(self) -> str:
        return f"explainiverse-{self.lifecycle_nonce}"

    @property
    def desired_firewall_rules(self) -> tuple[FirewallRule, ...]:
        return (
            FirewallRule(
                protocol="tcp",
                port_range=(22, 22),
                source_network=self.current_public_ipv4_cidr,
                description=f"Explainiverse {self.lifecycle_nonce} controller SSH",
            ),
        )

    @property
    def ownership_tags(self) -> tuple[tuple[str, str], ...]:
        return (
            ("explainiverse-lifecycle-nonce", self.lifecycle_nonce),
            ("explainiverse-owner", REPOSITORY),
            ("explainiverse-purpose", "stable-release-cuda"),
            ("explainiverse-source-sha", self.head_sha),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "explainiverse-lambda-live-plan",
            "openapi": {
                "openapi_version": OPENAPI_VERSION,
                "api_version": LAMBDA_API_VERSION,
                "document_sha256": OPENAPI_SHA256,
                "production_origin": PRODUCTION_ORIGIN,
            },
            "repository": REPOSITORY,
            "head_sha": self.head_sha,
            "lifecycle_nonce": self.lifecycle_nonce,
            "created_at_unix": self.created_at_unix,
            "expires_at_unix": self.expires_at_unix,
            "controller_source": self.current_public_ipv4_cidr,
            "target": {
                "instance_type_name": TARGET_INSTANCE_TYPE,
                "instance_type_description": self.instance_type_description,
                "gpu_description": self.gpu_description,
                "physical_gpu_count": TARGET_GPU_COUNT,
                "architecture": TARGET_ARCHITECTURE,
                "price_cents_per_hour": self.price_cents_per_hour,
                "vcpus": self.vcpus,
                "memory_gib": self.memory_gib,
                "storage_gib": self.storage_gib,
                "region_name": TARGET_REGION,
                "region_description": self.region_description,
                "image": {
                    "id": self.image_id,
                    "created_time": self.image_created_time,
                    "description": self.image_description,
                    "name": self.image_name,
                    "family": self.image_family,
                    "version": self.image_version,
                    "updated_time": self.image_updated_time,
                    "architecture": TARGET_ARCHITECTURE,
                    "region_name": TARGET_REGION,
                },
            },
            "ssh_access": {
                "key_name": self.ssh_key_name,
                "public_key_sha256": self.ssh_public_key_sha256,
                "ephemeral_host_key_fingerprint": self.host_key_fingerprint,
            },
            "remote_runtime": {
                "bundle_sha256": self.runtime_bundle_sha256,
                "bundle_files": list(RUNTIME_BUNDLE_NAMES),
                "install_root": REMOTE_RUNTIME_ROOT,
                "fixed_cloud_init_wait_command": list(FIXED_CLOUD_INIT_WAIT_COMMAND),
                "fixed_preflight_command": list(FIXED_PREFLIGHT_COMMAND),
                "fixed_command": list(FIXED_REMOTE_COMMAND),
                "fixed_command_contains_dynamic_or_plan_values": False,
            },
            "baseline_file_systems_sha256": self.baseline_file_systems_sha256,
            "original_global_rules": [rule.to_mapping() for rule in self.original_global_rules],
            "desired_global_and_instance_rules": [
                rule.to_mapping() for rule in self.desired_firewall_rules
            ],
            "ownership_tags": [{"key": key, "value": value} for key, value in self.ownership_tags],
            "mutation_order": list(MUTATION_PATHS),
            "secret_transport": {
                "lambda_api_key": "anonymous-fd-or-stdin-only",
                "github_jit_config": "anonymous-fd-or-stdin-only",
                "host_private_key": "in-memory-cloud-init-only",
            },
            "production_authorized": self.production_authorized,
            "provider_mutation_authorized": self.provider_mutation_authorized,
            "live_go": False,
        }

    @property
    def sha256(self) -> str:
        return _sha256(_canonical_json(self.to_mapping()))


def build_immutable_plan(
    *,
    head_sha: str,
    lifecycle_nonce: str,
    created_at_unix: int,
    expires_at_unix: int,
    current_public_ipv4_cidr: str,
    region_description: str,
    image_id: str,
    image_created_time: str,
    image_description: str,
    image_name: str,
    image_family: str,
    image_version: str,
    image_updated_time: str,
    instance_type_description: str,
    gpu_description: str,
    price_cents_per_hour: int,
    vcpus: int,
    memory_gib: int,
    storage_gib: int,
    ssh_key_name: str,
    ssh_public_key_sha256: str,
    baseline_file_systems_sha256: str,
    original_global_rules: Sequence[Mapping[str, Any]],
    host_key_fingerprint: str,
    runtime_bundle_sha256: str,
) -> ImmutablePlan:
    """Construct a digestable plan from an already captured read-only inventory."""

    return ImmutablePlan(
        head_sha=head_sha,
        lifecycle_nonce=lifecycle_nonce,
        created_at_unix=created_at_unix,
        expires_at_unix=expires_at_unix,
        current_public_ipv4_cidr=current_public_ipv4_cidr,
        region_description=region_description,
        image_id=image_id,
        image_created_time=image_created_time,
        image_description=image_description,
        image_name=image_name,
        image_family=image_family,
        image_version=image_version,
        image_updated_time=image_updated_time,
        instance_type_description=instance_type_description,
        gpu_description=gpu_description,
        price_cents_per_hour=price_cents_per_hour,
        vcpus=vcpus,
        memory_gib=memory_gib,
        storage_gib=storage_gib,
        ssh_key_name=ssh_key_name,
        ssh_public_key_sha256=ssh_public_key_sha256,
        baseline_file_systems_sha256=baseline_file_systems_sha256,
        original_global_rules=tuple(
            FirewallRule.from_mapping(rule, f"original_global_rule_{index}")
            for index, rule in enumerate(original_global_rules, 1)
        ),
        host_key_fingerprint=host_key_fingerprint,
        runtime_bundle_sha256=runtime_bundle_sha256,
    )


@dataclass(frozen=True)
class LiveGates:
    production_authorized: bool
    provider_mutation_authorized: bool
    immutable_plan_sha256: str

    def validate(self, plan: ImmutablePlan, *, require_current: bool) -> None:
        _require(self.production_authorized is True, "production_gate_closed")
        _require(
            self.provider_mutation_authorized is True,
            "provider_mutation_gate_closed",
        )
        _require(
            SHA256_RE.fullmatch(self.immutable_plan_sha256) is not None,
            "gate_plan_sha256_rejected",
        )
        _require(self.immutable_plan_sha256 == plan.sha256, "gate_plan_digest_mismatch")
        if require_current:
            now = int(time.time())
            _require(plan.created_at_unix <= now <= plan.expires_at_unix, "plan_not_current")


@dataclass(frozen=True)
class HostIdentity:
    """Ephemeral host identity; its repr intentionally excludes every key byte."""

    _private_openssh: SecretBuffer = field(repr=False)
    _public_openssh: str = field(repr=False)
    fingerprint: str

    def __post_init__(self) -> None:
        _require(type(self._private_openssh) is SecretBuffer, "host_private_buffer_rejected")
        key_blob = _ed25519_public_key_blob(self._public_openssh, "host_public_key")
        calculated = "SHA256:" + base64.b64encode(hashlib.sha256(key_blob).digest()).decode(
            "ascii"
        ).rstrip("=")
        _require(calculated == self.fingerprint, "host_fingerprint_mismatch")
        private = self._private_openssh.copy_bytes()
        _require(
            private.startswith(b"-----BEGIN OPENSSH PRIVATE KEY-----\n")
            and private.endswith(b"-----END OPENSSH PRIVATE KEY-----\n")
            and b"\x00" not in private,
            "host_private_key_format_rejected",
        )

    def __repr__(self) -> str:
        return f"HostIdentity(fingerprint={self.fingerprint!r}, key_material=<redacted>)"

    @property
    def destroyed(self) -> bool:
        return self._private_openssh.destroyed

    def cloud_init(self, runtime_bundle: RuntimeBundle) -> SecretBuffer:
        _require(not self.destroyed, "host_identity_destroyed")
        private_text = self._private_openssh.copy_bytes().decode("ascii")
        indented = "\n".join(f"    {line}" for line in private_text.rstrip("\n").splitlines())
        lines = [
            "#cloud-config\n"
            "ssh_deletekeys: true\n"
            "ssh_genkeytypes: []\n"
            "ssh_keys:\n"
            "  ed25519_private: |\n"
            f"{indented}\n"
            f"  ed25519_public: {self._public_openssh}\n"
            "write_files:\n"
        ]
        for name, content in runtime_bundle.files:
            lines.extend(
                [
                    f"  - path: {REMOTE_RUNTIME_ROOT}/{name}\n",
                    "    owner: root:root\n",
                    "    permissions: '0444'\n",
                    "    encoding: b64\n",
                    f"    content: {base64.b64encode(content).decode('ascii')}\n",
                ]
            )
        lines.extend(
            [
                "runcmd:\n",
                f"  - [chown, 'root:root', '{REMOTE_RUNTIME_ROOT}']\n",
                f"  - [chmod, '0555', '{REMOTE_RUNTIME_ROOT}']\n",
            ]
        )
        cloud_config = "".join(lines)
        _require(len(cloud_config.encode("ascii")) <= 1_048_576, "cloud_init_too_large")
        return SecretBuffer(cloud_config.encode("ascii"), label="cloud_init_host_key")

    def known_hosts(self, public_ipv4: str) -> str:
        address = ipaddress.ip_address(public_ipv4)
        _require(type(address) is ipaddress.IPv4Address, "instance_ip_not_ipv4")
        _require(address.is_global, "instance_ip_not_public")
        # OpenSSH's canonical known_hosts form for the default port 22 is the
        # plain host/IP. Bracketed ``[host]:port`` form is for non-default ports.
        return f"{address.compressed} {self._public_openssh}\n"

    def destroy(self) -> None:
        self._private_openssh.destroy()


def generate_ephemeral_host_identity() -> HostIdentity:
    """Generate an Ed25519 OpenSSH host key without touching the filesystem."""

    try:
        from cryptography.hazmat.primitives import serialization  # type: ignore[import-not-found]
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore[import-not-found]
            Ed25519PrivateKey,
        )
    except ImportError:
        raise ContractError("cryptography_dependency_unavailable") from None

    private_key = Ed25519PrivateKey.generate()
    private_openssh = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_openssh = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )
        .decode("ascii")
    )
    key_blob = _ed25519_public_key_blob(public_openssh, "generated_host_public_key")
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key_blob).digest()).decode(
        "ascii"
    ).rstrip("=")
    return HostIdentity(
        _private_openssh=SecretBuffer(private_openssh, label="host_private_key"),
        _public_openssh=public_openssh,
        fingerprint=fingerprint,
    )


def _derive_access_public_key(raw_private_key: bytes) -> tuple[str, str, str]:
    try:
        from cryptography.hazmat.primitives import serialization  # type: ignore[import-not-found]
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore[import-not-found]
            Ed25519PrivateKey,
        )
    except ImportError:
        raise ContractError("cryptography_dependency_unavailable") from None
    try:
        private_key = serialization.load_ssh_private_key(raw_private_key, password=None)
    except (TypeError, ValueError):
        _fail("ssh_access_identity_private_key_rejected")
    _require(
        isinstance(private_key, Ed25519PrivateKey),
        "ssh_access_identity_not_ed25519",
    )
    public_key = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )
        .decode("ascii")
    )
    canonical = _canonical_ed25519_public_key(public_key, "ssh_access_identity_public_key")
    key_blob = _ed25519_public_key_blob(canonical, "ssh_access_identity_public_key")
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key_blob).digest()).decode(
        "ascii"
    ).rstrip("=")
    return canonical, _sha256(canonical.encode("ascii")), fingerprint


def _windows_access_identity_acl(path: Path) -> dict[str, Any]:
    try:
        import ntsecuritycon  # type: ignore[import-not-found,import-untyped]
        import win32api  # type: ignore[import-not-found,import-untyped]
        import win32con  # type: ignore[import-not-found,import-untyped]
        import win32security  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        raise ContractError("windows_acl_dependency_unavailable") from None

    requested = win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
    try:
        descriptor = win32security.GetFileSecurity(str(path), requested)
        owner = win32security.ConvertSidToStringSid(descriptor.GetSecurityDescriptorOwner())
        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
        current_user = win32security.ConvertSidToStringSid(
            win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        )
        control = descriptor.GetSecurityDescriptorControl()[0]
        dacl = descriptor.GetSecurityDescriptorDacl()
    except Exception:
        raise ContractError("ssh_access_identity_acl_query_failed") from None
    _require(owner == current_user, "ssh_access_identity_owner_rejected")
    _require(dacl is not None, "ssh_access_identity_dacl_missing")
    _require(
        bool(control & win32security.SE_DACL_PROTECTED),
        "ssh_access_identity_dacl_inheritance_enabled",
    )
    allowed_sids = {current_user, "S-1-5-18", "S-1-5-32-544"}
    aces: list[dict[str, Any]] = []
    try:
        for index in range(dacl.GetAceCount()):
            header, mask, sid = dacl.GetAce(index)
            ace_type, ace_flags = header
            normalized_sid = win32security.ConvertSidToStringSid(sid)
            _require(
                ace_type == win32security.ACCESS_ALLOWED_ACE_TYPE,
                "ssh_access_identity_nonallow_ace_rejected",
            )
            _require(ace_flags == 0, "ssh_access_identity_ace_flags_rejected")
            _require(
                normalized_sid in allowed_sids,
                "ssh_access_identity_trustee_rejected",
            )
            _require(
                mask == ntsecuritycon.FILE_ALL_ACCESS,
                "ssh_access_identity_rights_rejected",
            )
            aces.append(
                {
                    "sid": normalized_sid,
                    "access": "allow",
                    "rights": "full-control",
                    "mask": mask,
                    "ace_flags": ace_flags,
                }
            )
        raw_descriptor = bytes(memoryview(descriptor))
    except ContractError:
        raise
    except Exception:
        raise ContractError("ssh_access_identity_acl_parse_failed") from None
    _require(
        len(aces) == 3 and {item["sid"] for item in aces} == allowed_sids,
        "ssh_access_identity_dacl_not_exact",
    )
    material = {
        "owner_sid": owner,
        "current_user_sid": current_user,
        "inheritance_protected": True,
        "aces": sorted(aces, key=lambda item: str(item["sid"])),
        "security_descriptor_sha256": _sha256(raw_descriptor),
        "security_descriptor_bytes": len(raw_descriptor),
    }
    return {
        **material,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "evidence_sha256": _sha256(_canonical_json(material)),
    }


def _open_access_identity(path: Path) -> tuple[Any, tuple[Any, ...], bytes, dict[str, Any]]:
    if os.name == "nt":
        try:
            import win32con  # type: ignore[import-not-found,import-untyped]
            import win32file  # type: ignore[import-not-found,import-untyped]
        except ImportError:
            raise ContractError("windows_file_identity_dependency_unavailable") from None
        try:
            handle = win32file.CreateFile(
                str(path),
                win32con.GENERIC_READ,
                win32con.FILE_SHARE_READ,
                None,
                win32con.OPEN_EXISTING,
                0x00200000,  # FILE_FLAG_OPEN_REPARSE_POINT
                None,
            )
            info = win32file.GetFileInformationByHandle(handle)
            attributes = int(info[0])
            size = (int(info[5]) << 32) | int(info[6])
            links = int(info[7])
            final_path = str(win32file.GetFinalPathNameByHandle(handle, 0))
            if final_path.startswith("\\\\?\\"):
                final_path = final_path[4:]
            _require(
                os.path.normcase(os.path.abspath(final_path)) == os.path.normcase(str(path)),
                "ssh_access_identity_handle_path_mismatch",
            )
            _require(
                attributes & 0x00000400 == 0,
                "ssh_access_identity_reparse_point_rejected",
            )
            _require(links == 1, "ssh_access_identity_link_count_rejected")
            _require(1 <= size <= 65_536, "ssh_access_identity_size_rejected")
            win32file.SetFilePointer(handle, 0, win32con.FILE_BEGIN)
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                _, chunk = win32file.ReadFile(handle, min(65_536, remaining))
                _require(bool(chunk), "ssh_access_identity_short_read")
                chunks.append(bytes(chunk))
                remaining -= len(chunk)
            raw = b"".join(chunks)
            win32file.SetFilePointer(handle, 0, win32con.FILE_BEGIN)
            identity = (int(info[4]), int(info[8]), int(info[9]), links, size, attributes)
            acl = _windows_access_identity_acl(path)
            return handle, identity, raw, acl
        except ContractError:
            try:
                handle.Close()
            except (AttributeError, UnboundLocalError):
                pass
            raise
        except Exception:
            try:
                handle.Close()
            except (AttributeError, UnboundLocalError):
                pass
            raise ContractError("ssh_access_identity_open_failed") from None

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        _require(stat.S_ISREG(opened.st_mode), "ssh_access_identity_not_regular")
        _require(opened.st_nlink == 1, "ssh_access_identity_link_count_rejected")
        _require(
            opened.st_uid == os.geteuid(),  # type: ignore[attr-defined]
            "ssh_access_identity_owner_rejected",
        )
        _require(
            stat.S_IMODE(opened.st_mode) == 0o600,
            "ssh_access_identity_permissions_open",
        )
        _require(1 <= opened.st_size <= 65_536, "ssh_access_identity_size_rejected")
        raw = b""
        while len(raw) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(raw))
            _require(bool(chunk), "ssh_access_identity_short_read")
            raw += chunk
        os.lseek(descriptor, 0, os.SEEK_SET)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_uid,
            stat.S_IMODE(opened.st_mode),
        )
        acl_material = {
            "owner_uid": opened.st_uid,
            "current_user_uid": os.geteuid(),  # type: ignore[attr-defined]
            "mode": "0600",
            "single_link": True,
        }
        acl = {
            **acl_material,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "evidence_sha256": _sha256(_canonical_json(acl_material)),
        }
        return descriptor, identity, raw, acl
    except ContractError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise
    except OSError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise ContractError("ssh_access_identity_open_failed") from None


@dataclass(frozen=True)
class AccessIdentityReceipt:
    """Sealed existing SSH access identity; private path/digest never enter evidence."""

    public_key_sha256: str
    public_key_fingerprint: str
    private_file_bytes: int
    acl: Mapping[str, Any]
    _absolute_path: str = field(repr=False)
    _private_file_sha256: str = field(repr=False)
    _file_identity: tuple[Any, ...] = field(repr=False)
    _handle: Any = field(repr=False, compare=False)
    _closed: bool = field(default=False, repr=False, compare=False)

    @property
    def absolute_path(self) -> str:
        _require(not self._closed, "ssh_access_identity_closed")
        return self._absolute_path

    @property
    def closed(self) -> bool:
        return self._closed

    def _read_held_bytes(self) -> tuple[tuple[Any, ...], bytes]:
        _require(not self._closed, "ssh_access_identity_closed")
        if os.name == "nt":
            import win32con  # type: ignore[import-not-found]
            import win32file  # type: ignore[import-not-found]

            info = win32file.GetFileInformationByHandle(self._handle)
            size = (int(info[5]) << 32) | int(info[6])
            identity = (
                int(info[4]),
                int(info[8]),
                int(info[9]),
                int(info[7]),
                size,
                int(info[0]),
            )
            win32file.SetFilePointer(self._handle, 0, win32con.FILE_BEGIN)
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                _, chunk = win32file.ReadFile(self._handle, min(65_536, remaining))
                _require(bool(chunk), "ssh_access_identity_short_read")
                chunks.append(bytes(chunk))
                remaining -= len(chunk)
            win32file.SetFilePointer(self._handle, 0, win32con.FILE_BEGIN)
            return identity, b"".join(chunks)
        opened = os.fstat(self._handle)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_uid,
            stat.S_IMODE(opened.st_mode),
        )
        os.lseek(self._handle, 0, os.SEEK_SET)
        raw = b""
        while len(raw) < opened.st_size:
            chunk = os.read(self._handle, opened.st_size - len(raw))
            _require(bool(chunk), "ssh_access_identity_short_read")
            raw += chunk
        os.lseek(self._handle, 0, os.SEEK_SET)
        return identity, raw

    def validate(self, *, expected_public_key_sha256: str) -> dict[str, Any]:
        _require(
            SHA256_RE.fullmatch(expected_public_key_sha256) is not None,
            "ssh_access_identity_expected_public_digest_rejected",
        )
        identity, raw = self._read_held_bytes()
        _require(identity == self._file_identity, "ssh_access_identity_file_identity_drift")
        _require(
            len(raw) == self.private_file_bytes and _sha256(raw) == self._private_file_sha256,
            "ssh_access_identity_private_file_drift",
        )
        _, public_digest, fingerprint = _derive_access_public_key(raw)
        _require(
            public_digest == self.public_key_sha256 == expected_public_key_sha256
            and fingerprint == self.public_key_fingerprint,
            "ssh_access_identity_public_key_drift",
        )
        if os.name == "nt":
            import win32file  # type: ignore[import-not-found]

            final_path = str(win32file.GetFinalPathNameByHandle(self._handle, 0))
            if final_path.startswith("\\\\?\\"):
                final_path = final_path[4:]
            _require(
                os.path.normcase(os.path.abspath(final_path))
                == os.path.normcase(self._absolute_path),
                "ssh_access_identity_handle_path_mismatch",
            )
            acl = _windows_access_identity_acl(Path(self._absolute_path))
        else:
            opened = os.fstat(self._handle)
            try:
                path_state = os.stat(self._absolute_path, follow_symlinks=False)
            except OSError:
                raise ContractError("ssh_access_identity_path_unavailable") from None
            path_identity = (
                path_state.st_dev,
                path_state.st_ino,
                path_state.st_nlink,
                path_state.st_size,
                path_state.st_uid,
                stat.S_IMODE(path_state.st_mode),
            )
            _require(
                stat.S_ISREG(path_state.st_mode) and path_identity == self._file_identity,
                "ssh_access_identity_path_identity_drift",
            )
            acl_material = {
                "owner_uid": opened.st_uid,
                "current_user_uid": os.geteuid(),  # type: ignore[attr-defined]
                "mode": "0600",
                "single_link": opened.st_nlink == 1,
            }
            acl = {
                **acl_material,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "evidence_sha256": _sha256(_canonical_json(acl_material)),
            }
        initial_acl = {key: value for key, value in self.acl.items() if key != "captured_at"}
        current_acl = {key: value for key, value in acl.items() if key != "captured_at"}
        _require(current_acl == initial_acl, "ssh_access_identity_acl_drift")
        return {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "public_key_sha256": public_digest,
            "public_key_fingerprint": fingerprint,
            "private_file_bytes": len(raw),
            "private_digest_recorded": True,
            "absolute_path_redacted": True,
            "file_identity_recorded": True,
            "single_link": True,
            "no_reparse_or_symlink": True,
            "acl_evidence_sha256": str(self.acl["evidence_sha256"]),
        }

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "captured_at": self.acl["captured_at"],
            "public_key_sha256": self.public_key_sha256,
            "public_key_fingerprint": self.public_key_fingerprint,
            "key_type": "ssh-ed25519",
            "private_file_bytes": self.private_file_bytes,
            "private_digest_recorded": True,
            "absolute_path_redacted": True,
            "file_identity_recorded": True,
            "single_link": True,
            "no_reparse_or_symlink": True,
            "acl": dict(self.acl),
        }

    def close(self) -> None:
        if self._closed:
            return
        if os.name == "nt":
            self._handle.Close()
        else:
            os.close(self._handle)
        object.__setattr__(self, "_closed", True)


def capture_access_identity(
    path: str | os.PathLike[str], *, expected_public_key_sha256: str
) -> AccessIdentityReceipt:
    """Seal one owner-private Ed25519 access key and hold its file identity."""

    _require(
        SHA256_RE.fullmatch(expected_public_key_sha256) is not None,
        "ssh_access_identity_expected_public_digest_rejected",
    )
    candidate = Path(path)
    _require(candidate.is_absolute(), "ssh_access_identity_path_not_absolute")
    resolved = candidate.resolve(strict=True)
    _require(candidate == resolved, "ssh_access_identity_path_not_canonical")
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        "ssh_access_identity_file_rejected",
    )
    handle, file_identity, raw, acl = _open_access_identity(candidate)
    try:
        _, public_digest, fingerprint = _derive_access_public_key(raw)
        _require(
            public_digest == expected_public_key_sha256,
            "ssh_access_identity_public_key_mismatch",
        )
        receipt = AccessIdentityReceipt(
            public_key_sha256=public_digest,
            public_key_fingerprint=fingerprint,
            private_file_bytes=len(raw),
            acl=MappingProxyType(dict(acl)),
            _absolute_path=str(candidate),
            _private_file_sha256=_sha256(raw),
            _file_identity=file_identity,
            _handle=handle,
        )
        receipt.validate(expected_public_key_sha256=expected_public_key_sha256)
        return receipt
    except BaseException:
        if os.name == "nt":
            handle.Close()
        else:
            os.close(handle)
        raise


def _windows_evidence_directory_acl(path: Path) -> dict[str, Any]:
    try:
        import ntsecuritycon  # type: ignore[import-not-found,import-untyped]
        import win32api  # type: ignore[import-not-found,import-untyped]
        import win32con  # type: ignore[import-not-found,import-untyped]
        import win32security  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        raise ContractError("windows_acl_dependency_unavailable") from None

    requested = win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
    try:
        descriptor = win32security.GetFileSecurity(str(path), requested)
        owner = win32security.ConvertSidToStringSid(descriptor.GetSecurityDescriptorOwner())
        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
        current_user = win32security.ConvertSidToStringSid(
            win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        )
        control = descriptor.GetSecurityDescriptorControl()[0]
        dacl = descriptor.GetSecurityDescriptorDacl()
    except Exception:
        raise ContractError("evidence_directory_acl_query_failed") from None
    _require(owner == current_user, "evidence_directory_owner_rejected")
    _require(dacl is not None, "evidence_directory_dacl_missing")
    _require(
        bool(control & win32security.SE_DACL_PROTECTED),
        "evidence_directory_dacl_inheritance_enabled",
    )
    allowed_sids = {current_user, "S-1-5-18", "S-1-5-32-544"}
    inheritance_flags = win32security.OBJECT_INHERIT_ACE | win32security.CONTAINER_INHERIT_ACE
    aces: list[dict[str, Any]] = []
    try:
        for index in range(dacl.GetAceCount()):
            header, mask, sid = dacl.GetAce(index)
            ace_type, ace_flags = header
            normalized_sid = win32security.ConvertSidToStringSid(sid)
            _require(
                ace_type == win32security.ACCESS_ALLOWED_ACE_TYPE,
                "evidence_directory_nonallow_ace_rejected",
            )
            _require(
                ace_flags == inheritance_flags,
                "evidence_directory_ace_flags_rejected",
            )
            _require(
                normalized_sid in allowed_sids,
                "evidence_directory_trustee_rejected",
            )
            _require(
                mask == ntsecuritycon.FILE_ALL_ACCESS,
                "evidence_directory_rights_rejected",
            )
            aces.append(
                {
                    "sid": normalized_sid,
                    "access": "allow",
                    "rights": "full-control",
                    "mask": mask,
                    "ace_flags": ace_flags,
                }
            )
        raw_descriptor = bytes(memoryview(descriptor))
    except ContractError:
        raise
    except Exception:
        raise ContractError("evidence_directory_acl_parse_failed") from None
    _require(
        len(aces) == 3 and {item["sid"] for item in aces} == allowed_sids,
        "evidence_directory_dacl_not_exact",
    )
    material = {
        "owner_sid": owner,
        "current_user_sid": current_user,
        "inheritance_protected": True,
        "child_inheritance_enabled": True,
        "aces": sorted(aces, key=lambda item: str(item["sid"])),
        "security_descriptor_sha256": _sha256(raw_descriptor),
        "security_descriptor_bytes": len(raw_descriptor),
    }
    return {
        **material,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "evidence_sha256": _sha256(_canonical_json(material)),
    }


def _create_windows_evidence_directory(path: Path) -> None:
    try:
        import ntsecuritycon  # type: ignore[import-not-found,import-untyped]
        import win32api  # type: ignore[import-not-found,import-untyped]
        import win32con  # type: ignore[import-not-found,import-untyped]
        import win32file  # type: ignore[import-not-found,import-untyped]
        import win32security  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        raise ContractError("windows_acl_dependency_unavailable") from None

    try:
        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
        current_user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        dacl = win32security.ACL()
        inheritance_flags = win32security.OBJECT_INHERIT_ACE | win32security.CONTAINER_INHERIT_ACE
        for sid in (
            current_user,
            win32security.ConvertStringSidToSid("S-1-5-18"),
            win32security.ConvertStringSidToSid("S-1-5-32-544"),
        ):
            dacl.AddAccessAllowedAceEx(
                win32security.ACL_REVISION_DS,
                inheritance_flags,
                ntsecuritycon.FILE_ALL_ACCESS,
                sid,
            )
        descriptor = win32security.SECURITY_DESCRIPTOR()
        descriptor.SetSecurityDescriptorOwner(current_user, False)
        descriptor.SetSecurityDescriptorDacl(True, dacl, False)
        descriptor.SetSecurityDescriptorControl(
            win32security.SE_DACL_PROTECTED,
            win32security.SE_DACL_PROTECTED,
        )
        attributes = win32security.SECURITY_ATTRIBUTES()
        attributes.bInheritHandle = 0
        attributes.SECURITY_DESCRIPTOR = descriptor
        win32file.CreateDirectory(str(path), attributes)
    except Exception:
        raise ContractError("evidence_directory_secure_create_failed") from None


def _open_evidence_directory(
    path: Path,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    identity: tuple[Any, ...]
    if os.name == "nt":
        try:
            import win32con  # type: ignore[import-not-found,import-untyped]
            import win32file  # type: ignore[import-not-found,import-untyped]
        except ImportError:
            raise ContractError("windows_file_identity_dependency_unavailable") from None
        try:
            handle = win32file.CreateFile(
                str(path),
                win32con.GENERIC_READ,
                win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
                None,
                win32con.OPEN_EXISTING,
                0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
                None,
            )
            info = win32file.GetFileInformationByHandle(handle)
            attributes = int(info[0])
            final_path = str(win32file.GetFinalPathNameByHandle(handle, 0))
            if final_path.startswith("\\\\?\\"):
                final_path = final_path[4:]
            _require(
                os.path.normcase(os.path.abspath(final_path)) == os.path.normcase(str(path)),
                "evidence_directory_handle_path_mismatch",
            )
            _require(
                bool(attributes & 0x00000010),
                "evidence_directory_not_directory",
            )
            _require(
                attributes & 0x00000400 == 0,
                "evidence_directory_reparse_point_rejected",
            )
            identity = (int(info[4]), int(info[8]), int(info[9]), attributes)
            return handle, identity, _windows_evidence_directory_acl(path)
        except ContractError:
            try:
                handle.Close()
            except (AttributeError, UnboundLocalError):
                pass
            raise
        except Exception:
            try:
                handle.Close()
            except (AttributeError, UnboundLocalError):
                pass
            raise ContractError("evidence_directory_open_failed") from None

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        _require(stat.S_ISDIR(opened.st_mode), "evidence_directory_not_directory")
        _require(
            opened.st_uid == os.geteuid(),  # type: ignore[attr-defined]
            "evidence_directory_owner_rejected",
        )
        _require(
            stat.S_IMODE(opened.st_mode) == 0o700,
            "evidence_directory_permissions_open",
        )
        identity = (opened.st_dev, opened.st_ino)
        acl_material = {
            "owner_uid": opened.st_uid,
            "current_user_uid": os.geteuid(),  # type: ignore[attr-defined]
            "mode": "0700",
        }
        acl = {
            **acl_material,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "evidence_sha256": _sha256(_canonical_json(acl_material)),
        }
        return descriptor, identity, acl
    except ContractError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise
    except OSError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise ContractError("evidence_directory_open_failed") from None


@dataclass(frozen=True)
class EvidenceDirectoryReceipt:
    """Held identity and exact owner-private ACL for one evidence directory."""

    receipt_sha256: str
    acl: Mapping[str, Any]
    _absolute_path: str = field(repr=False)
    _directory_identity: tuple[Any, ...] = field(repr=False)
    _handle: Any = field(repr=False, compare=False)
    _closed: bool = field(default=False, repr=False, compare=False)

    @property
    def absolute_path(self) -> str:
        _require(not self._closed, "evidence_directory_receipt_closed")
        return self._absolute_path

    @property
    def closed(self) -> bool:
        return self._closed

    def validate(self) -> dict[str, Any]:
        _require(not self._closed, "evidence_directory_receipt_closed")
        _require(
            SHA256_RE.fullmatch(self.receipt_sha256) is not None,
            "evidence_directory_receipt_sha256_rejected",
        )
        identity: tuple[Any, ...]
        if os.name == "nt":
            import win32file  # type: ignore[import-not-found]

            info = win32file.GetFileInformationByHandle(self._handle)
            attributes = int(info[0])
            identity = (int(info[4]), int(info[8]), int(info[9]), attributes)
            final_path = str(win32file.GetFinalPathNameByHandle(self._handle, 0))
            if final_path.startswith("\\\\?\\"):
                final_path = final_path[4:]
            _require(
                os.path.normcase(os.path.abspath(final_path))
                == os.path.normcase(self._absolute_path),
                "evidence_directory_handle_path_mismatch",
            )
            acl = _windows_evidence_directory_acl(Path(self._absolute_path))
        else:
            opened = os.fstat(self._handle)
            identity = (opened.st_dev, opened.st_ino)
            try:
                path_state = os.stat(self._absolute_path, follow_symlinks=False)
            except OSError:
                raise ContractError("evidence_directory_path_unavailable") from None
            _require(
                stat.S_ISDIR(path_state.st_mode)
                and (path_state.st_dev, path_state.st_ino) == self._directory_identity,
                "evidence_directory_path_identity_drift",
            )
            acl_material = {
                "owner_uid": opened.st_uid,
                "current_user_uid": os.geteuid(),  # type: ignore[attr-defined]
                "mode": f"{stat.S_IMODE(opened.st_mode):04o}",
            }
            acl = {
                **acl_material,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "evidence_sha256": _sha256(_canonical_json(acl_material)),
            }
        _require(identity == self._directory_identity, "evidence_directory_identity_drift")
        initial_acl = {key: value for key, value in self.acl.items() if key != "captured_at"}
        current_acl = {key: value for key, value in acl.items() if key != "captured_at"}
        _require(current_acl == initial_acl, "evidence_directory_acl_drift")
        expected_receipt_sha256 = _sha256(
            _canonical_json(
                _evidence_directory_receipt_material(
                    Path(self._absolute_path),
                    self._directory_identity,
                    str(self.acl.get("evidence_sha256")),
                )
            )
        )
        _require(
            self.receipt_sha256 == expected_receipt_sha256,
            "evidence_directory_receipt_binding_rejected",
        )
        return {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "receipt_sha256": self.receipt_sha256,
            "absolute_path_redacted": True,
            "directory_identity_recorded": True,
            "no_reparse_or_symlink": True,
            "owner_private": True,
            "acl_evidence_sha256": str(self.acl["evidence_sha256"]),
        }

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "captured_at": self.acl["captured_at"],
            "receipt_sha256": self.receipt_sha256,
            "absolute_path_redacted": True,
            "directory_identity_recorded": True,
            "no_reparse_or_symlink": True,
            "owner_private": True,
            "acl": dict(self.acl),
        }

    def close(self) -> None:
        if self._closed:
            return
        if os.name == "nt":
            self._handle.Close()
        else:
            os.close(self._handle)
        object.__setattr__(self, "_closed", True)


def _evidence_directory_receipt_material(
    path: Path,
    directory_identity: tuple[Any, ...],
    acl_evidence_sha256: str,
) -> dict[str, Any]:
    _require(
        SHA256_RE.fullmatch(acl_evidence_sha256) is not None,
        "evidence_directory_acl_evidence_sha256_rejected",
    )
    return {
        "schema_version": 1,
        "kind": "explainiverse-evidence-directory-receipt",
        "canonical_path_sha256": _sha256(os.path.normcase(str(path)).encode("utf-8")),
        "directory_identity": list(directory_identity),
        "acl_evidence_sha256": acl_evidence_sha256,
    }


def _capture_evidence_directory(path: Path) -> EvidenceDirectoryReceipt:
    handle, directory_identity, acl = _open_evidence_directory(path)
    try:
        material = _evidence_directory_receipt_material(
            path,
            directory_identity,
            str(acl["evidence_sha256"]),
        )
        receipt = EvidenceDirectoryReceipt(
            receipt_sha256=_sha256(_canonical_json(material)),
            acl=MappingProxyType(dict(acl)),
            _absolute_path=str(path),
            _directory_identity=directory_identity,
            _handle=handle,
        )
        receipt.validate()
        return receipt
    except BaseException:
        if os.name == "nt":
            handle.Close()
        else:
            os.close(handle)
        raise


def create_evidence_directory(path: str | os.PathLike[str]) -> EvidenceDirectoryReceipt:
    """Atomically create and hold one exact owner-private evidence directory."""

    candidate = Path(path)
    _require(candidate.is_absolute(), "evidence_directory_path_not_absolute")
    resolved = candidate.resolve(strict=False)
    _require(candidate == resolved, "evidence_directory_path_not_canonical")
    parent = candidate.parent
    _require(
        parent.resolve(strict=True) == parent and parent.is_dir() and not parent.is_symlink(),
        "evidence_directory_parent_rejected",
    )
    _require(not candidate.exists(), "evidence_directory_path_exists")
    created = False
    try:
        if os.name == "nt":
            _create_windows_evidence_directory(candidate)
        else:
            os.mkdir(candidate, 0o700)
            os.chmod(candidate, 0o700, follow_symlinks=False)
        created = True
        return _capture_evidence_directory(candidate)
    except BaseException:
        if created:
            try:
                os.rmdir(candidate)
            except OSError:
                pass
        raise


def reopen_evidence_directory(
    path: str | os.PathLike[str], *, expected_receipt_sha256: str
) -> EvidenceDirectoryReceipt:
    """Reopen an interrupted evidence directory only under its original binding."""

    _require(
        SHA256_RE.fullmatch(expected_receipt_sha256) is not None,
        "evidence_directory_expected_receipt_rejected",
    )
    candidate = Path(path)
    _require(candidate.is_absolute(), "evidence_directory_path_not_absolute")
    resolved = candidate.resolve(strict=True)
    _require(candidate == resolved, "evidence_directory_path_not_canonical")
    _require(
        candidate.is_dir() and not candidate.is_symlink(),
        "evidence_directory_path_rejected",
    )
    receipt = _capture_evidence_directory(candidate)
    if receipt.receipt_sha256 != expected_receipt_sha256:
        receipt.close()
        raise ContractError("evidence_directory_receipt_mismatch")
    return receipt


@dataclass(frozen=True)
class StrictSshBinding:
    argv_prefix: tuple[str, ...]
    known_hosts: str
    known_hosts_path: str | None
    known_hosts_sha256: str
    evidence_directory_acl_receipt_sha256: str | None
    host_fingerprint: str
    remote_mode: str
    remote_command: tuple[str, ...]

    def to_public_mapping(self) -> dict[str, Any]:
        redacted_argv = list(self.argv_prefix)
        identity_index = redacted_argv.index("-i") + 1
        redacted_argv[identity_index] = "<redacted-existing-access-identity-file>"
        return {
            "argv_prefix": redacted_argv,
            "access_identity_file_redacted": True,
            "known_hosts": self.known_hosts,
            "known_hosts_path": self.known_hosts_path,
            "known_hosts_sha256": self.known_hosts_sha256,
            "evidence_directory_acl_receipt_sha256": self.evidence_directory_acl_receipt_sha256,
            "host_fingerprint": self.host_fingerprint,
            "trust_on_first_use": False,
            "remote_mode": self.remote_mode,
            "fixed_remote_command": list(self.remote_command),
        }


@dataclass(frozen=True)
class KnownHostsFileReceipt:
    absolute_path: str
    content_sha256: str
    evidence_directory_acl_receipt_sha256: str
    public_ipv4: str
    host_fingerprint: str

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "absolute_path": self.absolute_path,
            "content_sha256": self.content_sha256,
            "evidence_directory_acl_receipt_sha256": self.evidence_directory_acl_receipt_sha256,
            "public_ipv4": self.public_ipv4,
            "host_fingerprint": self.host_fingerprint,
            "content_is_public": True,
        }


def _write_exclusive_bytes(destination: Path, payload: bytes) -> None:
    _require(destination.is_absolute(), "public_file_path_not_absolute")
    _require(destination.parent.is_dir(), "public_file_parent_missing")
    _require(not destination.exists(), "public_file_path_exists")
    temporary = destination.parent / (f".{destination.name}.pending-{secrets.token_hex(16)}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    published = False
    try:
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            _require(count > 0, "public_file_short_write")
            written += count
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if os.name == "nt":
            try:
                import win32file  # type: ignore[import-not-found,import-untyped]
            except ImportError:
                raise ContractError("windows_atomic_publish_dependency_unavailable") from None
            # No REPLACE_EXISTING flag: a raced destination fails closed.  The
            # write-through flag waits for the move to reach durable storage.
            win32file.MoveFileEx(
                str(temporary),
                str(destination),
                win32file.MOVEFILE_WRITE_THROUGH,
            )
            published = True
        else:
            # A same-filesystem hard link publishes without overwriting a
            # raced destination.  Both names point at fully synced bytes.
            os.link(temporary, destination, follow_symlinks=False)
            published = True
            temporary.unlink()
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except OSError:
            pass
        if published:
            _require(
                destination.is_file() and destination.read_bytes() == payload,
                "public_file_atomic_publish_ambiguous",
            )
            raise ContractError("public_file_durability_unconfirmed") from None
        raise ContractError("public_file_atomic_publish_failed") from None
    _require(destination.is_file(), "public_file_atomic_publish_missing")
    _require(destination.read_bytes() == payload, "public_file_atomic_publish_drift")
    if os.name != "nt":
        _require(stat.S_IMODE(destination.stat().st_mode) == 0o600, "public_file_mode_drift")


def write_public_known_hosts(
    path: str | os.PathLike[str],
    *,
    identity: HostIdentity,
    public_ipv4: str,
    evidence_directory_receipt: EvidenceDirectoryReceipt,
) -> KnownHostsFileReceipt:
    """Create a dedicated public known_hosts file without overwrite or TOFU."""

    _require(
        type(evidence_directory_receipt) is EvidenceDirectoryReceipt,
        "evidence_directory_receipt_type_rejected",
    )
    evidence_directory_receipt.validate()
    destination = Path(path)
    _require(destination.is_absolute(), "known_hosts_path_not_absolute")
    resolved_destination = destination.resolve(strict=False)
    _require(destination == resolved_destination, "known_hosts_path_not_canonical")
    destination = resolved_destination
    _require(
        destination.parent == Path(evidence_directory_receipt.absolute_path),
        "known_hosts_outside_evidence_directory",
    )
    content = identity.known_hosts(public_ipv4).encode("ascii")
    _write_exclusive_bytes(destination, content)
    evidence_directory_receipt.validate()
    return KnownHostsFileReceipt(
        absolute_path=str(destination),
        content_sha256=_sha256(content),
        evidence_directory_acl_receipt_sha256=evidence_directory_receipt.receipt_sha256,
        public_ipv4=ipaddress.ip_address(public_ipv4).compressed,
        host_fingerprint=identity.fingerprint,
    )


def build_strict_ssh_binding(
    *,
    identity: HostIdentity,
    public_ipv4: str,
    access_identity_file: str,
    known_hosts_fd: int | None = None,
    known_hosts_file: KnownHostsFileReceipt | None = None,
    remote_mode: str,
    user: str = "ubuntu",
) -> StrictSshBinding:
    """Return an executable SSH argv with exact host-key and fixed command."""

    _require(
        (known_hosts_fd is None) != (known_hosts_file is None),
        "known_hosts_transport_not_exactly_one",
    )
    _require(
        remote_mode in {"cloud-init", "preflight", "run"},
        "ssh_remote_mode_rejected",
    )
    remote_command = {
        "cloud-init": FIXED_CLOUD_INIT_WAIT_COMMAND,
        "preflight": FIXED_PREFLIGHT_COMMAND,
        "run": FIXED_REMOTE_COMMAND,
    }[remote_mode]
    identity_path = Path(access_identity_file)
    _require(identity_path.is_absolute(), "ssh_identity_path_not_absolute")
    _require(identity_path.is_file(), "ssh_identity_file_missing")
    mode = identity_path.stat().st_mode
    if os.name != "nt":
        _require(mode & (stat.S_IRWXG | stat.S_IRWXO) == 0, "ssh_identity_permissions_open")
    known_hosts = identity.known_hosts(public_ipv4)
    known_hosts_path: str | None = None
    acl_receipt: str | None = None
    if known_hosts_file is not None:
        path = Path(known_hosts_file.absolute_path)
        _require(path.is_absolute(), "known_hosts_path_not_absolute")
        _require(path.is_file() and not path.is_symlink(), "known_hosts_file_rejected")
        content = path.read_bytes()
        _require(content == known_hosts.encode("ascii"), "known_hosts_content_mismatch")
        _require(_sha256(content) == known_hosts_file.content_sha256, "known_hosts_digest_mismatch")
        _require(
            known_hosts_file.host_fingerprint == identity.fingerprint,
            "known_hosts_fingerprint_mismatch",
        )
        _require(
            known_hosts_file.public_ipv4 == ipaddress.ip_address(public_ipv4).compressed,
            "known_hosts_ip_mismatch",
        )
        _require(
            SHA256_RE.fullmatch(known_hosts_file.evidence_directory_acl_receipt_sha256) is not None,
            "known_hosts_acl_receipt_rejected",
        )
        if os.name != "nt":
            _require(stat.S_IMODE(path.stat().st_mode) == 0o600, "known_hosts_mode_drift")
        known_hosts_reference = str(path)
        known_hosts_path = str(path)
        acl_receipt = known_hosts_file.evidence_directory_acl_receipt_sha256
    else:
        if type(known_hosts_fd) is not int or known_hosts_fd < 3:
            _fail("known_hosts_fd_rejected")
        _require(os.name != "nt", "windows_known_hosts_fd_rejected")
        _require(not stat.S_ISREG(os.fstat(known_hosts_fd).st_mode), "known_hosts_fd_regular")
        known_hosts_reference = f"/dev/fd/{known_hosts_fd}"
    target = f"{user}@{ipaddress.ip_address(public_ipv4).compressed}"
    return StrictSshBinding(
        argv_prefix=(
            "ssh",
            "-T",
            "-F",
            os.devnull,
            "-i",
            str(identity_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "IdentityAgent=none",
            "-o",
            "RequestTTY=no",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "HostKeyAlgorithms=ssh-ed25519",
            "-o",
            f"UserKnownHostsFile={known_hosts_reference}",
            "-o",
            f"GlobalKnownHostsFile={os.devnull}",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            "CheckHostIP=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ConnectTimeout=20",
            "-p",
            "22",
            target,
            *remote_command,
        ),
        known_hosts=known_hosts,
        known_hosts_path=known_hosts_path,
        known_hosts_sha256=_sha256(known_hosts.encode("ascii")),
        evidence_directory_acl_receipt_sha256=acl_receipt,
        host_fingerprint=identity.fingerprint,
        remote_mode=remote_mode,
        remote_command=remote_command,
    )


@dataclass(frozen=True)
class ProviderRequest:
    operation: str
    method: str
    path: str
    mutating: bool
    body: bytearray | None = field(default=None, repr=False)
    sensitive_body: bool = False
    timeout_seconds: int = PROVIDER_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        _text(self.operation, "request_operation")
        _require(self.method in {"GET", "POST", "PATCH", "DELETE"}, "request_method_rejected")
        _require(self.path.startswith(f"{API_PREFIX}/"), "request_path_rejected")
        _require("?" not in self.path and "#" not in self.path, "request_path_suffix_rejected")
        _require(type(self.mutating) is bool, "request_mutating_not_bool")
        _require(type(self.sensitive_body) is bool, "request_sensitive_not_bool")
        read_paths = dict(READ_OPERATIONS)
        if not self.mutating:
            _require(self.operation in read_paths, "read_operation_not_allowlisted")
            _require(
                self.method == "GET" and self.path == read_paths[self.operation],
                "read_method_or_path_not_allowlisted",
            )
        else:
            _require(self.operation in MUTATION_PATHS, "mutation_operation_not_allowlisted")
            expected_method, expected_path = MUTATION_PATHS[self.operation]
            _require(self.method == expected_method, "mutation_method_not_allowlisted")
            if self.operation == "delete_ruleset":
                prefix = f"{API_PREFIX}/firewall-rulesets/"
                _require(self.path.startswith(prefix), "delete_ruleset_path_not_allowlisted")
                encoded_id = self.path[len(prefix) :]
                decoded_id = urllib.parse.unquote(encoded_id)
                _require(
                    bool(encoded_id)
                    and urllib.parse.quote(decoded_id, safe="") == encoded_id
                    and RESOURCE_ID_RE.fullmatch(decoded_id) is not None,
                    "delete_ruleset_id_not_allowlisted",
                )
            else:
                _require(self.path == expected_path, "mutation_path_not_allowlisted")
        _require(
            type(self.timeout_seconds) is int and 1 <= self.timeout_seconds <= 30,
            "request_timeout_rejected",
        )
        if self.body is None:
            _require(self.method in {"GET", "DELETE"}, "request_body_missing")
        else:
            _require(self.method in {"POST", "PATCH"}, "request_body_unexpected")
            _require(len(self.body) <= MAX_PROVIDER_RESPONSE_BYTES, "request_body_too_large")

    @property
    def body_sha256(self) -> str | None:
        return _sha256(self.body) if self.body is not None else None

    @property
    def request_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "operation": self.operation,
                    "method": self.method,
                    "path": self.path,
                    "body_sha256": self.body_sha256,
                    "timeout_seconds": self.timeout_seconds,
                }
            )
        )

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "method": self.method,
            "path": self.path,
            "mutating": self.mutating,
            "body_sha256": self.body_sha256,
            "body_redacted": self.body is not None,
            "timeout_seconds": self.timeout_seconds,
            "request_sha256": self.request_sha256,
        }

    def destroy_body(self) -> None:
        if self.body is None:
            return
        for index in range(len(self.body)):
            self.body[index] = 0
        self.body.clear()


@dataclass(frozen=True)
class ProviderResponse:
    request_sha256: str
    status_code: int | None
    content_type: str | None
    body: bytes = field(repr=False)
    timed_out: bool = False
    transport_error: bool = False


@dataclass(frozen=True)
class ResponseBinding:
    operation: str
    method: str
    path: str
    request_sha256: str
    request_body_sha256: str | None
    response_body_sha256: str
    status_code: int
    content_type: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "method": self.method,
            "path": self.path,
            "request_sha256": self.request_sha256,
            "request_body_sha256": self.request_body_sha256,
            "response_body_sha256": self.response_body_sha256,
            "status_code": self.status_code,
            "content_type": self.content_type,
        }


ProviderTransport = Callable[[ProviderRequest, SecretBuffer], ProviderResponse]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_: Any, **__: Any) -> None:
        return None


def _urllib_transport(request: ProviderRequest, api_key: SecretBuffer) -> ProviderResponse:
    """HTTPS transport.  It never follows redirects and never returns headers with secrets."""

    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
        _NoRedirect(),
    )
    url = urllib.parse.urljoin(f"{PRODUCTION_ORIGIN}/", request.path.lstrip("/"))
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "cloud.lambda.ai"
        or parsed.query
        or parsed.fragment
    ):
        raise TransportFailure("provider_origin_rejected")
    key_bytes = api_key.copy_bytes()
    try:
        key_text = key_bytes.decode("ascii")
    except UnicodeDecodeError:
        raise ContractError("lambda_api_key_not_ascii") from None
    data: bytes | None = bytes(request.body) if request.body is not None else None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {key_text}",
        "User-Agent": "explainiverse-release-controller/1",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    http_request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method=request.method,
    )
    try:
        with opener.open(http_request, timeout=request.timeout_seconds) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError:
                    return ProviderResponse(
                        request.request_sha256,
                        response.status,
                        response.headers.get("Content-Type"),
                        b"",
                        transport_error=True,
                    )
                if not 0 <= declared_size <= MAX_PROVIDER_RESPONSE_BYTES:
                    return ProviderResponse(
                        request.request_sha256,
                        response.status,
                        response.headers.get("Content-Type"),
                        b"",
                        transport_error=True,
                    )
            body = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
            return ProviderResponse(
                request_sha256=request.request_sha256,
                status_code=response.status,
                content_type=response.headers.get("Content-Type"),
                body=body,
            )
    except urllib.error.HTTPError as error:
        # A non-2xx mutation response is conservatively ambiguous.  Bound error
        # content is unnecessary and may contain provider-generated sensitive data.
        try:
            return ProviderResponse(
                request_sha256=request.request_sha256,
                status_code=error.code,
                content_type=error.headers.get("Content-Type") if error.headers else None,
                body=b"",
            )
        finally:
            error.close()
    except TimeoutError:
        return ProviderResponse(request.request_sha256, None, None, b"", timed_out=True)
    except (urllib.error.URLError, OSError, ssl.SSLError):
        return ProviderResponse(request.request_sha256, None, None, b"", transport_error=True)
    finally:
        # CPython cannot guarantee zeroization of immutable header/string copies;
        # drop all local references at the earliest possible point.
        del key_bytes
        del key_text
        del headers
        del http_request
        del data


class LambdaHttpClient:
    """One credential-bearing Lambda client; construct only from an FD."""

    __slots__ = (
        "_api_key",
        "_clock",
        "_last_request_started",
        "_pacing_lock",
        "_sleep",
        "_transport",
    )

    _api_key: SecretBuffer
    _clock: Callable[[], float]
    _sleep: Callable[[float], None]
    _last_request_started: float | None
    _pacing_lock: threading.Lock
    _transport: ProviderTransport

    def __init__(self, *_: Any, **__: Any) -> None:
        raise ContractError("lambda_client_requires_secret_fd_factory")

    @classmethod
    def from_secret_fd(
        cls,
        fd: int,
        *,
        transport: ProviderTransport | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> LambdaHttpClient:
        instance = object.__new__(cls)
        if transport is None:
            _require(
                monotonic_clock is time.monotonic and sleep is time.sleep,
                "live_provider_clock_injection_rejected",
            )
        _require(callable(monotonic_clock), "provider_pacing_clock_not_callable")
        _require(callable(sleep), "provider_pacing_sleep_not_callable")
        key = _read_secret_fd(fd, maximum=MAX_API_KEY_BYTES, label="lambda_api_key")
        raw = key.copy_bytes()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError:
            key.destroy()
            raise ContractError("lambda_api_key_not_ascii") from None
        if any(byte <= 0x20 or byte >= 0x7F for byte in raw):
            key.destroy()
            raise ContractError("lambda_api_key_whitespace_or_control")
        instance._api_key = key
        instance._transport = transport or _urllib_transport
        instance._clock = monotonic_clock
        instance._sleep = sleep
        instance._last_request_started = None
        instance._pacing_lock = threading.Lock()
        return instance

    def __enter__(self) -> LambdaHttpClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._api_key.destroy()

    def _pace_request_start(self) -> None:
        """Enforce Lambda's general one-request-per-second limit client-wide."""

        with self._pacing_lock:
            for _ in range(1_000):
                now = self._clock()
                _require(type(now) in {int, float}, "provider_pacing_clock_not_numeric")
                _require(math.isfinite(now), "provider_pacing_clock_not_finite")
                previous = self._last_request_started
                if previous is None:
                    self._last_request_started = float(now)
                    return
                _require(now >= previous, "provider_pacing_clock_reversed")
                remaining = PROVIDER_MIN_REQUEST_INTERVAL_SECONDS - (now - previous)
                if remaining <= 0:
                    self._last_request_started = float(now)
                    return
                self._sleep(remaining)
            _fail("provider_pacing_clock_stalled")

    def request(self, request: ProviderRequest) -> tuple[Mapping[str, Any], ResponseBinding]:
        """Execute one allowlisted read; mutations require the lifecycle adapter."""

        _require(not request.mutating, "direct_provider_mutation_rejected")
        return self._request_bound(request)

    def _request_mutation(
        self, request: ProviderRequest
    ) -> tuple[Mapping[str, Any], ResponseBinding]:
        _require(request.mutating, "internal_mutation_request_not_mutating")
        return self._request_bound(request)

    def _request_bound(self, request: ProviderRequest) -> tuple[Mapping[str, Any], ResponseBinding]:
        _require(not self._api_key.destroyed, "lambda_client_closed")
        original_request_sha = request.request_sha256
        original_body_sha = request.body_sha256
        self._pace_request_start()
        try:
            response = self._transport(request, self._api_key)
        except Exception:
            if request.mutating:
                raise AmbiguousMutation(
                    request.operation, original_request_sha, "transport_exception"
                ) from None
            raise TransportFailure(f"read_{request.operation}_transport_exception") from None
        finally:
            if request.sensitive_body:
                request.destroy_body()

        if response.request_sha256 != original_request_sha:
            if request.mutating:
                raise AmbiguousMutation(request.operation, original_request_sha, "response_unbound")
            raise TransportFailure(f"read_{request.operation}_response_unbound")
        if response.timed_out or response.transport_error or response.status_code is None:
            reason = "timeout" if response.timed_out else "transport_failure"
            if request.mutating:
                raise AmbiguousMutation(request.operation, original_request_sha, reason)
            raise TransportFailure(f"read_{request.operation}_{reason}")
        if len(response.body) > MAX_PROVIDER_RESPONSE_BYTES:
            if request.mutating:
                raise AmbiguousMutation(
                    request.operation, original_request_sha, "response_too_large"
                )
            raise TransportFailure(f"read_{request.operation}_response_too_large")
        media_type = (response.content_type or "").split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            if request.mutating:
                raise AmbiguousMutation(request.operation, original_request_sha, "content_type")
            raise TransportFailure(f"read_{request.operation}_content_type")
        if response.status_code != 200:
            if request.mutating:
                raise AmbiguousMutation(request.operation, original_request_sha, "http_status")
            raise TransportFailure(f"read_{request.operation}_http_status")
        try:
            payload = _strict_json_loads(response.body)
        except (UnicodeDecodeError, ValueError):
            if request.mutating:
                raise AmbiguousMutation(request.operation, original_request_sha, "json")
            raise TransportFailure(f"read_{request.operation}_json") from None
        if type(payload) is not dict:
            if request.mutating:
                raise AmbiguousMutation(
                    request.operation, original_request_sha, "response_not_object"
                )
            raise TransportFailure(f"read_{request.operation}_response_not_object")
        binding = ResponseBinding(
            operation=request.operation,
            method=request.method,
            path=request.path,
            request_sha256=original_request_sha,
            request_body_sha256=original_body_sha,
            response_body_sha256=_sha256(response.body),
            status_code=response.status_code,
            content_type=media_type,
        )
        return payload, binding


@dataclass(frozen=True)
class _Snapshot:
    payloads: Mapping[str, Any] = field(repr=False)
    bindings: tuple[ResponseBinding, ...]
    observed_started_monotonic_ns: int
    observed_finished_monotonic_ns: int

    @property
    def sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "payload_digests": {
                        name: _sha256(_canonical_json(value))
                        for name, value in sorted(self.payloads.items())
                    },
                    "bindings": [binding.to_mapping() for binding in self.bindings],
                }
            )
        )


def _capture_snapshot(client: LambdaHttpClient) -> _Snapshot:
    started = time.monotonic_ns()
    deadline = started + MAX_OBSERVATION_WINDOW_SECONDS * 1_000_000_000
    payloads: dict[str, Any] = {}
    bindings: list[ResponseBinding] = []
    for operation, path in READ_OPERATIONS:
        remaining_ns = deadline - time.monotonic_ns()
        _require(remaining_ns > 0, "observation_window_exhausted")
        remaining_seconds = max(1, (remaining_ns + 999_999_999) // 1_000_000_000)
        request = ProviderRequest(
            operation,
            "GET",
            path,
            False,
            timeout_seconds=min(PROVIDER_TIMEOUT_SECONDS, remaining_seconds),
        )
        payload, binding = client.request(request)
        payloads[operation] = payload
        bindings.append(binding)
    finished = time.monotonic_ns()
    _require(
        finished - started <= MAX_OBSERVATION_WINDOW_SECONDS * 1_000_000_000,
        "observation_window_too_long",
    )
    return _Snapshot(
        payloads=MappingProxyType(payloads),
        bindings=tuple(bindings),
        observed_started_monotonic_ns=started,
        observed_finished_monotonic_ns=finished,
    )


@dataclass(frozen=True)
class DiscoveryReceipt:
    """Read-only, zero-inventory source for one immutable live plan."""

    snapshot_sha256: str
    observed_monotonic_ns: int
    region_description: str
    instance_type_description: str
    gpu_description: str
    price_cents_per_hour: int
    vcpus: int
    memory_gib: int
    storage_gib: int
    images: tuple[Mapping[str, Any], ...]
    ssh_key_name: str
    ssh_public_key_sha256: str
    baseline_file_systems_sha256: str
    original_global_rules: tuple[FirewallRule, ...]
    binding_sha256: str
    _snapshot: _Snapshot = field(repr=False)

    def _binding_mapping(self) -> dict[str, Any]:
        return {
            "snapshot_sha256": self.snapshot_sha256,
            "region_description": self.region_description,
            "instance_type_description": self.instance_type_description,
            "gpu_description": self.gpu_description,
            "price_cents_per_hour": self.price_cents_per_hour,
            "vcpus": self.vcpus,
            "memory_gib": self.memory_gib,
            "storage_gib": self.storage_gib,
            "images": [dict(image) for image in self.images],
            "ssh_key_name": self.ssh_key_name,
            "ssh_public_key_sha256": self.ssh_public_key_sha256,
            "baseline_file_systems_sha256": self.baseline_file_systems_sha256,
            "original_global_rules": [rule.to_mapping() for rule in self.original_global_rules],
        }

    def validate_binding(self) -> None:
        _require(
            SHA256_RE.fullmatch(self.binding_sha256) is not None,
            "discovery_binding_sha256_rejected",
        )
        _require(
            _sha256(_canonical_json(self._binding_mapping())) == self.binding_sha256,
            "discovery_binding_digest_mismatch",
        )

    def to_public_mapping(self) -> dict[str, Any]:
        payload_digests = {
            name: _sha256(_canonical_json(value))
            for name, value in sorted(self._snapshot.payloads.items())
        }
        return {
            "snapshot_sha256": self.snapshot_sha256,
            "binding_sha256": self.binding_sha256,
            "payload_digests": payload_digests,
            "response_bindings": [binding.to_mapping() for binding in self._snapshot.bindings],
            "zero_instances": True,
            "zero_firewall_rulesets": True,
            "target": {
                "instance_type_name": TARGET_INSTANCE_TYPE,
                "instance_type_description": self.instance_type_description,
                "gpu_description": self.gpu_description,
                "price_cents_per_hour": self.price_cents_per_hour,
                "vcpus": self.vcpus,
                "memory_gib": self.memory_gib,
                "storage_gib": self.storage_gib,
                "gpus": TARGET_GPU_COUNT,
                "architecture": TARGET_ARCHITECTURE,
                "capacity_region": TARGET_REGION,
                "region_description": self.region_description,
            },
            "image_candidates": [dict(image) for image in self.images],
            "ssh_access": {
                "key_name": self.ssh_key_name,
                "public_key_sha256": self.ssh_public_key_sha256,
            },
            "baseline_file_systems_sha256": self.baseline_file_systems_sha256,
            "original_global_rules": [rule.to_mapping() for rule in self.original_global_rules],
        }


def capture_action_time_discovery(
    client: LambdaHttpClient, *, ssh_key_name: str
) -> DiscoveryReceipt:
    """Perform the credentialed read-only discovery required before planning."""

    _text(ssh_key_name, "ssh_key_name")
    snapshot = _capture_snapshot(client)
    payloads = snapshot.payloads

    instances = _instances(_data(payloads["instances"], "instances_response"))
    rulesets = _firewall_rulesets(
        _data(payloads["firewall_rulesets"], "firewall_rulesets_response")
    )
    _require(instances == [], "discovery_instances_not_zero")
    _require(rulesets == [], "discovery_rulesets_not_zero")

    regions = _data(payloads["regions"], "regions_response")
    _require(type(regions) is list, "regions_data_not_list")
    selected_regions = [
        value for value in regions if type(value) is dict and value.get("name") == TARGET_REGION
    ]
    _require(len(selected_regions) == 1, "target_region_not_unique")
    region_name, region_description = _region(selected_regions[0], "target_region")
    _require(region_name == TARGET_REGION, "target_region_name_mismatch")
    _require(
        region_description == TARGET_REGION_DESCRIPTION,
        "target_region_description_mismatch",
    )

    instance_types = _data(payloads["instance_types"], "instance_types_response")
    _require(type(instance_types) is dict, "instance_types_data_not_object")
    _require(TARGET_INSTANCE_TYPE in instance_types, "target_instance_type_missing")
    item = _exact_keys(
        instance_types[TARGET_INSTANCE_TYPE],
        {"instance_type", "regions_with_capacity_available"},
        "target_instance_type_item",
    )
    instance_type = _exact_keys(
        item["instance_type"],
        {
            "name",
            "description",
            "gpu_description",
            "price_cents_per_hour",
            "specs",
            "architecture",
        },
        "target_instance_type",
    )
    _require(instance_type["name"] == TARGET_INSTANCE_TYPE, "instance_type_name_drift")
    _require(instance_type["architecture"] == TARGET_ARCHITECTURE, "instance_type_arch_drift")
    gpu_description = _validate_a100_api_description(instance_type["gpu_description"])
    description = _text(instance_type["description"], "instance_type_description")
    price = _integer(instance_type["price_cents_per_hour"], "price_cents_per_hour", minimum=1)
    specs = _exact_keys(
        instance_type["specs"],
        {"vcpus", "memory_gib", "storage_gib", "gpus"},
        "instance_type_specs",
    )
    vcpus = _integer(specs["vcpus"], "instance_type_vcpus", minimum=1)
    memory_gib = _integer(specs["memory_gib"], "instance_type_memory_gib", minimum=1)
    storage_gib = _integer(specs["storage_gib"], "instance_type_storage_gib", minimum=1)
    _require(specs["gpus"] == TARGET_GPU_COUNT, "instance_type_gpu_count_drift")
    capacity = item["regions_with_capacity_available"]
    _require(type(capacity) is list, "capacity_regions_not_list")
    selected_capacity = [
        value for value in capacity if type(value) is dict and value.get("name") == TARGET_REGION
    ]
    _require(len(selected_capacity) == 1, "target_capacity_not_available")
    _require(
        _region(selected_capacity[0], "capacity_region") == (TARGET_REGION, region_description),
        "capacity_region_drift",
    )

    raw_images = _data(payloads["images"], "images_response")
    _require(type(raw_images) is list, "images_data_not_list")
    images: list[Mapping[str, Any]] = []
    for index, raw_image in enumerate(raw_images, 1):
        image = _exact_keys(
            raw_image,
            {
                "id",
                "created_time",
                "updated_time",
                "name",
                "description",
                "family",
                "version",
                "architecture",
                "region",
            },
            f"image_{index}",
        )
        image_created = _utc_timestamp(
            _text(image["created_time"], f"image_{index}_created"), f"image_{index}_created"
        )
        image_updated = _utc_timestamp(
            _text(image["updated_time"], f"image_{index}_updated"), f"image_{index}_updated"
        )
        _require(image_created <= image_updated, f"image_{index}_timestamp_order_rejected")
        _require(
            image_updated
            <= datetime.now(timezone.utc) + timedelta(seconds=MAX_PROVIDER_CLOCK_SKEW_SECONDS),
            f"image_{index}_timestamp_in_future",
        )
        image_region = _region(image["region"], f"image_{index}_region")
        if (
            image["architecture"] == TARGET_ARCHITECTURE
            and image_region == (TARGET_REGION, region_description)
            and image["family"] == TARGET_IMAGE_FAMILY
        ):
            images.append(
                {
                    "id": _text(image["id"], f"image_{index}_id"),
                    "created_time": image["created_time"],
                    "updated_time": image["updated_time"],
                    "name": _text(image["name"], f"image_{index}_name"),
                    "description": _text(
                        image["description"], f"image_{index}_description", allow_empty=True
                    ),
                    "family": _text(image["family"], f"image_{index}_family"),
                    "version": _text(image["version"], f"image_{index}_version"),
                    "architecture": TARGET_ARCHITECTURE,
                    "region": {"name": TARGET_REGION, "description": region_description},
                }
            )
    _require(bool(images), "eligible_lambda_stack_22_04_images_empty")
    _require(len({str(image["id"]) for image in images}) == len(images), "image_ids_duplicate")

    ssh_keys = _data(payloads["ssh_keys"], "ssh_keys_response")
    _require(type(ssh_keys) is list, "ssh_keys_data_not_list")
    selected_keys = [
        value for value in ssh_keys if type(value) is dict and value.get("name") == ssh_key_name
    ]
    _require(len(selected_keys) == 1, "ssh_access_key_not_unique")
    selected_key = _exact_keys(selected_keys[0], {"id", "name", "public_key"}, "ssh_access_key")
    _text(selected_key["id"], "ssh_access_key_id")
    public_key = _canonical_ed25519_public_key(selected_key["public_key"], "ssh_access_key")

    file_systems = _canonical_file_systems(_data(payloads["file_systems"], "file_systems_response"))
    global_rules = _global_ruleset(_data(payloads["global_firewall"], "global_firewall_response"))
    binding = {
        "snapshot_sha256": snapshot.sha256,
        "region_description": region_description,
        "instance_type_description": description,
        "gpu_description": gpu_description,
        "price_cents_per_hour": price,
        "vcpus": vcpus,
        "memory_gib": memory_gib,
        "storage_gib": storage_gib,
        "images": images,
        "ssh_key_name": ssh_key_name,
        "ssh_public_key_sha256": _sha256(public_key.encode("ascii")),
        "baseline_file_systems_sha256": _sha256(_canonical_json(file_systems)),
        "original_global_rules": [rule.to_mapping() for rule in global_rules],
    }
    return DiscoveryReceipt(
        snapshot_sha256=snapshot.sha256,
        observed_monotonic_ns=snapshot.observed_finished_monotonic_ns,
        region_description=region_description,
        instance_type_description=description,
        gpu_description=gpu_description,
        price_cents_per_hour=price,
        vcpus=vcpus,
        memory_gib=memory_gib,
        storage_gib=storage_gib,
        images=tuple(images),
        ssh_key_name=ssh_key_name,
        ssh_public_key_sha256=_sha256(public_key.encode("ascii")),
        baseline_file_systems_sha256=_sha256(_canonical_json(file_systems)),
        original_global_rules=global_rules,
        binding_sha256=_sha256(_canonical_json(binding)),
        _snapshot=snapshot,
    )


def build_plan_from_discovery(
    discovery: DiscoveryReceipt,
    *,
    head_sha: str,
    lifecycle_nonce: str,
    created_at_unix: int,
    expires_at_unix: int,
    current_public_ipv4_cidr: str,
    image_id: str,
    host_identity: HostIdentity,
    runtime_bundle: RuntimeBundle,
) -> ImmutablePlan:
    """Select one exact image from a fresh discovery and bind the host key."""

    _require(
        discovery.snapshot_sha256 == discovery._snapshot.sha256,
        "discovery_snapshot_digest_mismatch",
    )
    discovery.validate_binding()
    age = time.monotonic_ns() - discovery.observed_monotonic_ns
    _require(0 <= age <= PRESTATE_FRESHNESS_SECONDS * 1_000_000_000, "discovery_receipt_stale")
    matches = [image for image in discovery.images if image["id"] == image_id]
    _require(len(matches) == 1, "selected_image_not_in_discovery")
    image = matches[0]
    return build_immutable_plan(
        head_sha=head_sha,
        lifecycle_nonce=lifecycle_nonce,
        created_at_unix=created_at_unix,
        expires_at_unix=expires_at_unix,
        current_public_ipv4_cidr=current_public_ipv4_cidr,
        region_description=discovery.region_description,
        image_id=_text(image["id"], "selected_image_id"),
        image_created_time=_text(image["created_time"], "selected_image_created_time"),
        image_description=_text(
            image["description"], "selected_image_description", allow_empty=True
        ),
        image_name=_text(image["name"], "selected_image_name"),
        image_family=_text(image["family"], "selected_image_family"),
        image_version=_text(image["version"], "selected_image_version"),
        image_updated_time=_text(image["updated_time"], "selected_image_updated_time"),
        instance_type_description=discovery.instance_type_description,
        gpu_description=discovery.gpu_description,
        price_cents_per_hour=discovery.price_cents_per_hour,
        vcpus=discovery.vcpus,
        memory_gib=discovery.memory_gib,
        storage_gib=discovery.storage_gib,
        ssh_key_name=discovery.ssh_key_name,
        ssh_public_key_sha256=discovery.ssh_public_key_sha256,
        baseline_file_systems_sha256=discovery.baseline_file_systems_sha256,
        original_global_rules=[rule.to_mapping() for rule in discovery.original_global_rules],
        host_key_fingerprint=host_identity.fingerprint,
        runtime_bundle_sha256=runtime_bundle.sha256,
    )


@dataclass(frozen=True)
class SnapshotReceipt:
    plan_sha256: str
    phase: str
    snapshot_sha256: str
    receipt_nonce: str
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    ruleset_id: str | None
    instance_id: str | None
    instance_public_ipv4: str | None
    _snapshot: _Snapshot = field(repr=False)

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "phase": self.phase,
            "snapshot_sha256": self.snapshot_sha256,
            "receipt_nonce": self.receipt_nonce,
            "ruleset_id": self.ruleset_id,
            "instance_id": self.instance_id,
            "instance_public_ipv4": self.instance_public_ipv4,
            "response_bindings": [binding.to_mapping() for binding in self._snapshot.bindings],
        }


@dataclass(frozen=True)
class MutationIntent:
    """Durable, secret-free write-ahead record for one provider mutation."""

    plan_sha256: str
    operation: str
    prestate_phase: str
    prestate_snapshot_sha256: str
    prestate_receipt_nonce: str
    callback_binding_sha256: str
    method: str
    path: str
    request_sha256: str
    request_body_sha256: str | None
    sensitive_body: bool
    timeout_seconds: int

    @classmethod
    def from_public_mapping(cls, value: Any) -> MutationIntent:
        mapping = _exact_keys(
            value,
            {
                "plan_sha256",
                "operation",
                "prestate_phase",
                "prestate_snapshot_sha256",
                "prestate_receipt_nonce",
                "callback_binding_sha256",
                "method",
                "path",
                "request_sha256",
                "request_body_sha256",
                "request_body_redacted",
                "sensitive_body",
                "timeout_seconds",
            },
            "mutation_intent",
        )
        plan_sha256 = _text(mapping["plan_sha256"], "mutation_intent_plan_sha256")
        _require(
            SHA256_RE.fullmatch(plan_sha256) is not None,
            "mutation_intent_plan_sha256_rejected",
        )
        operation = _text(mapping["operation"], "mutation_intent_operation")
        _require(operation in MUTATION_PATHS, "mutation_intent_operation_rejected")
        prestate_phase = _text(mapping["prestate_phase"], "mutation_intent_prestate_phase")
        _require(
            prestate_phase == MUTATION_PRESTATE[operation],
            "mutation_intent_prestate_phase_mismatch",
        )
        prestate_snapshot_sha256 = _text(
            mapping["prestate_snapshot_sha256"],
            "mutation_intent_prestate_snapshot_sha256",
        )
        _require(
            SHA256_RE.fullmatch(prestate_snapshot_sha256) is not None,
            "mutation_intent_prestate_snapshot_sha256_rejected",
        )
        prestate_receipt_nonce = _text(
            mapping["prestate_receipt_nonce"], "mutation_intent_prestate_receipt_nonce"
        )
        _require(
            NONCE_RE.fullmatch(prestate_receipt_nonce) is not None,
            "mutation_intent_prestate_receipt_nonce_rejected",
        )
        callback_binding_sha256 = _text(
            mapping["callback_binding_sha256"],
            "mutation_intent_callback_binding_sha256",
        )
        _require(
            SHA256_RE.fullmatch(callback_binding_sha256) is not None,
            "mutation_intent_callback_binding_sha256_rejected",
        )
        method = _text(mapping["method"], "mutation_intent_method")
        path = _text(mapping["path"], "mutation_intent_path")
        expected_method, expected_path = MUTATION_PATHS[operation]
        _require(method == expected_method, "mutation_intent_method_mismatch")
        if operation == "delete_ruleset":
            prefix = f"{API_PREFIX}/firewall-rulesets/"
            _require(path.startswith(prefix), "mutation_intent_path_mismatch")
            encoded_id = path[len(prefix) :]
            decoded_id = urllib.parse.unquote(encoded_id)
            _require(
                bool(encoded_id)
                and urllib.parse.quote(decoded_id, safe="") == encoded_id
                and RESOURCE_ID_RE.fullmatch(decoded_id) is not None,
                "mutation_intent_path_mismatch",
            )
        else:
            _require(path == expected_path, "mutation_intent_path_mismatch")
        request_body_sha256 = mapping["request_body_sha256"]
        if method == "DELETE":
            _require(
                request_body_sha256 is None,
                "mutation_intent_request_body_digest_unexpected",
            )
        else:
            _require(
                type(request_body_sha256) is str
                and SHA256_RE.fullmatch(request_body_sha256) is not None,
                "mutation_intent_request_body_digest_rejected",
            )
        _require(
            mapping["request_body_redacted"] is True,
            "mutation_intent_request_body_redaction_rejected",
        )
        sensitive_body = mapping["sensitive_body"]
        _require(
            type(sensitive_body) is bool and sensitive_body is (operation == "launch"),
            "mutation_intent_sensitive_body_mismatch",
        )
        timeout_seconds = mapping["timeout_seconds"]
        _require(
            type(timeout_seconds) is int and timeout_seconds == PROVIDER_TIMEOUT_SECONDS,
            "mutation_intent_timeout_mismatch",
        )
        request_sha256 = _text(mapping["request_sha256"], "mutation_intent_request_sha256")
        expected_request_sha256 = _sha256(
            _canonical_json(
                {
                    "operation": operation,
                    "method": method,
                    "path": path,
                    "body_sha256": request_body_sha256,
                    "timeout_seconds": timeout_seconds,
                }
            )
        )
        _require(
            request_sha256 == expected_request_sha256,
            "mutation_intent_request_sha256_mismatch",
        )
        return cls(
            plan_sha256=plan_sha256,
            operation=operation,
            prestate_phase=prestate_phase,
            prestate_snapshot_sha256=prestate_snapshot_sha256,
            prestate_receipt_nonce=prestate_receipt_nonce,
            callback_binding_sha256=callback_binding_sha256,
            method=method,
            path=path,
            request_sha256=request_sha256,
            request_body_sha256=request_body_sha256,
            sensitive_body=sensitive_body,
            timeout_seconds=timeout_seconds,
        )

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "operation": self.operation,
            "prestate_phase": self.prestate_phase,
            "prestate_snapshot_sha256": self.prestate_snapshot_sha256,
            "prestate_receipt_nonce": self.prestate_receipt_nonce,
            "callback_binding_sha256": self.callback_binding_sha256,
            "method": self.method,
            "path": self.path,
            "request_sha256": self.request_sha256,
            "request_body_sha256": self.request_body_sha256,
            "request_body_redacted": True,
            "sensitive_body": self.sensitive_body,
            "timeout_seconds": self.timeout_seconds,
        }


MutationIntentCallback = Callable[[MutationIntent], None]


@dataclass(frozen=True)
class MutationReceipt:
    plan_sha256: str
    operation: str
    prestate_snapshot_sha256: str
    request_sha256: str
    request_body_sha256: str | None
    response_body_sha256: str
    ruleset_id: str | None = None
    instance_id: str | None = None
    host_fingerprint: str | None = None

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "operation": self.operation,
            "prestate_snapshot_sha256": self.prestate_snapshot_sha256,
            "request_sha256": self.request_sha256,
            "request_body_sha256": self.request_body_sha256,
            "response_body_sha256": self.response_body_sha256,
            "response_body_redacted": True,
            "ruleset_id": self.ruleset_id,
            "instance_id": self.instance_id,
            "host_fingerprint": self.host_fingerprint,
        }


@dataclass(frozen=True)
class RecoveryReceipt:
    plan_sha256: str
    operation: str
    ambiguous_request_sha256: str
    inventory_snapshot_sha256: str
    outcome: str
    ruleset_id: str | None
    instance_id: str | None


def _validate_fresh_bound_instance_receipt(
    receipt: SnapshotReceipt, plan: ImmutablePlan
) -> tuple[str, str]:
    """Revalidate a fresh post-command provider receipt down to its sealed payloads."""

    _require(receipt.phase == "instance_bound", "host_binding_provider_phase")
    _require(receipt.plan_sha256 == plan.sha256, "host_binding_provider_plan")
    _require(
        receipt.snapshot_sha256 == receipt._snapshot.sha256,
        "host_binding_provider_snapshot",
    )
    _require(NONCE_RE.fullmatch(receipt.receipt_nonce) is not None, "host_binding_receipt_nonce")
    monotonic_now = time.monotonic_ns()
    _require(
        receipt.issued_monotonic_ns <= monotonic_now <= receipt.expires_monotonic_ns,
        "host_binding_provider_receipt_stale",
    )
    payloads = receipt._snapshot.payloads
    instances = _instances(_data(payloads["instances"], "host_binding_instances_response"))
    rulesets = _firewall_rulesets(
        _data(payloads["firewall_rulesets"], "host_binding_rulesets_response")
    )
    global_rules = _global_ruleset(
        _data(payloads["global_firewall"], "host_binding_global_response")
    )
    _require(global_rules == plan.desired_firewall_rules, "host_binding_global_drift")
    _require(len(instances) == 1, "host_binding_instance_not_sole")
    _require(len(rulesets) == 1, "host_binding_ruleset_not_sole")
    ruleset_id = _validate_owned_ruleset(plan, rulesets[0])
    instance_id, instance_ip = _validate_owned_instance(plan, instances[0], ruleset_id)
    _require(
        rulesets[0]["instance_ids"] == [instance_id],
        "host_binding_ruleset_instance_mismatch",
    )
    _require(receipt.ruleset_id == ruleset_id, "host_binding_receipt_ruleset_mismatch")
    _require(receipt.instance_id == instance_id, "host_binding_receipt_instance_mismatch")
    _require(receipt.instance_public_ipv4 == instance_ip, "host_binding_receipt_ip_mismatch")
    return instance_id, instance_ip


def _validate_known_hosts_file_receipt(
    receipt: KnownHostsFileReceipt,
    *,
    plan: ImmutablePlan,
    instance_public_ipv4: str,
) -> str:
    """Re-read and cryptographically bind the dedicated public known_hosts file."""

    path = Path(receipt.absolute_path)
    _require(path.is_absolute(), "host_binding_known_hosts_path_not_absolute")
    _require(path == path.resolve(strict=False), "host_binding_known_hosts_path_not_canonical")
    _require(path.is_file() and not path.is_symlink(), "host_binding_known_hosts_file")
    content = path.read_bytes()
    _require(0 < len(content) <= 16_384, "host_binding_known_hosts_size")
    _require(
        content.endswith(b"\n") and content.count(b"\n") == 1, "host_binding_known_hosts_lines"
    )
    _require(_sha256(content) == receipt.content_sha256, "host_binding_known_hosts_digest")
    try:
        line = content[:-1].decode("ascii")
    except UnicodeDecodeError:
        _fail("host_binding_known_hosts_encoding")
    parts = line.split(" ")
    _require(len(parts) == 3, "host_binding_known_hosts_format")
    address, algorithm, encoded_key = parts
    _require(address == instance_public_ipv4, "host_binding_known_hosts_address")
    _require(algorithm == "ssh-ed25519", "host_binding_known_hosts_algorithm")
    key_blob = _ed25519_public_key_blob(
        f"{algorithm} {encoded_key}",
        "host_binding_known_hosts_key",
    )
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key_blob).digest()).decode(
        "ascii"
    ).rstrip("=")
    _require(fingerprint == plan.host_key_fingerprint, "host_binding_known_hosts_fingerprint")
    _require(receipt.host_fingerprint == fingerprint, "host_binding_receipt_fingerprint")
    _require(receipt.public_ipv4 == instance_public_ipv4, "host_binding_receipt_ip")
    _require(
        SHA256_RE.fullmatch(receipt.evidence_directory_acl_receipt_sha256) is not None,
        "host_binding_acl_receipt",
    )
    if os.name != "nt":
        _require(stat.S_IMODE(path.stat().st_mode) == 0o600, "host_binding_known_hosts_mode")
    return receipt.content_sha256


@dataclass(frozen=True)
class CloudInitWaitReceipt:
    """Public binding for the fixed readiness command which precedes probe-host."""

    plan_sha256: str
    provider_snapshot_sha256: str
    provider_receipt_nonce: str
    instance_id: str
    instance_public_ipv4: str
    host_fingerprint: str
    known_hosts_sha256: str
    observed_at: str
    stdout_sha256: str
    stderr_sha256: str
    binding_sha256: str

    def _binding_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "explainiverse-lambda-live-cloud-init-wait-binding",
            "plan_sha256": self.plan_sha256,
            "provider_snapshot_sha256": self.provider_snapshot_sha256,
            "provider_receipt_nonce": self.provider_receipt_nonce,
            "instance_id": self.instance_id,
            "instance_public_ipv4": self.instance_public_ipv4,
            "host_fingerprint": self.host_fingerprint,
            "known_hosts_sha256": self.known_hosts_sha256,
            "observed_at": self.observed_at,
            "exit_code": 0,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "fixed_command": list(FIXED_CLOUD_INIT_WAIT_COMMAND),
            "credential_received": False,
            "jit_config_received": False,
        }

    def validate_binding(self) -> None:
        _require(
            SHA256_RE.fullmatch(self.binding_sha256) is not None,
            "cloud_init_binding_sha256_rejected",
        )
        _require(
            _sha256(_canonical_json(self._binding_mapping())) == self.binding_sha256,
            "cloud_init_binding_digest_mismatch",
        )

    def to_public_mapping(self) -> dict[str, Any]:
        return {**self._binding_mapping(), "binding_sha256": self.binding_sha256}

    @property
    def sha256(self) -> str:
        return _sha256(_canonical_json(self.to_public_mapping()))


def validate_cloud_init_wait_receipt(
    stdout: bytes,
    stderr: bytes,
    exit_code: int,
    *,
    plan: ImmutablePlan,
    provider_instance: SnapshotReceipt,
    known_hosts: KnownHostsFileReceipt,
    now: datetime | None = None,
) -> CloudInitWaitReceipt:
    """Validate the first fixed SSH command after a fresh post-command inventory."""

    _require(type(stdout) is bytes and 0 < len(stdout) <= 32_768, "cloud_init_stdout_size")
    _require(type(stderr) is bytes and len(stderr) <= 32_768, "cloud_init_stderr_size")
    _require(type(exit_code) is int and exit_code == 0, "cloud_init_exit_code")
    _require(b"\x00" not in stdout and b"\x00" not in stderr, "cloud_init_output_nul")
    try:
        stdout_text = stdout.decode("ascii")
        stderr_text = stderr.decode("ascii")
    except UnicodeDecodeError:
        _fail("cloud_init_output_encoding")
    lines = stdout_text.splitlines()
    _require(bool(lines) and lines[0] == "status: done", "cloud_init_status_not_done")
    combined_lower = f"{stdout_text}\n{stderr_text}".lower()
    _require(
        "degraded" not in combined_lower and "error" not in combined_lower,
        "cloud_init_output_failure_marker",
    )
    instance_id, instance_ip = _validate_fresh_bound_instance_receipt(provider_instance, plan)
    known_hosts_sha256 = _validate_known_hosts_file_receipt(
        known_hosts,
        plan=plan,
        instance_public_ipv4=instance_ip,
    )
    observed_now = now or datetime.now(timezone.utc)
    _require(observed_now.tzinfo is not None, "cloud_init_now_timezone_missing")
    observed_at = observed_now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    receipt = CloudInitWaitReceipt(
        plan_sha256=plan.sha256,
        provider_snapshot_sha256=provider_instance.snapshot_sha256,
        provider_receipt_nonce=provider_instance.receipt_nonce,
        instance_id=instance_id,
        instance_public_ipv4=instance_ip,
        host_fingerprint=plan.host_key_fingerprint,
        known_hosts_sha256=known_hosts_sha256,
        observed_at=observed_at,
        stdout_sha256=_sha256(stdout),
        stderr_sha256=_sha256(stderr),
        binding_sha256="0" * 64,
    )
    sealed = replace(
        receipt,
        binding_sha256=_sha256(_canonical_json(receipt._binding_mapping())),
    )
    sealed.validate_binding()
    return sealed


@dataclass(frozen=True)
class HostPreflightReceipt:
    """Local binding of one canonical, credential-free remote host probe."""

    plan_sha256: str
    provider_snapshot_sha256: str
    provider_receipt_nonce: str
    instance_id: str
    instance_public_ipv4: str
    host_fingerprint: str
    known_hosts_sha256: str
    cloud_init_wait_binding_sha256: str
    remote_response_sha256: str
    observed_at: str
    runtime_bundle_sha256: str
    host_physical_gpu_uuids: tuple[str, ...]
    host_physical_gpu_products: tuple[str, ...]
    image_probe_sha256: str
    gpu_injection_output_sha256: str
    gpu_injection_device_request_sha256: str

    def to_public_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "explainiverse-lambda-live-host-preflight-binding",
            "plan_sha256": self.plan_sha256,
            "provider_snapshot_sha256": self.provider_snapshot_sha256,
            "provider_receipt_nonce": self.provider_receipt_nonce,
            "instance_id": self.instance_id,
            "instance_public_ipv4": self.instance_public_ipv4,
            "host_fingerprint": self.host_fingerprint,
            "known_hosts_sha256": self.known_hosts_sha256,
            "cloud_init_wait_binding_sha256": self.cloud_init_wait_binding_sha256,
            "remote_response_sha256": self.remote_response_sha256,
            "observed_at": self.observed_at,
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "host_physical_gpu_count": len(self.host_physical_gpu_uuids),
            "host_physical_gpu_uuids": list(self.host_physical_gpu_uuids),
            "host_physical_gpu_products": list(self.host_physical_gpu_products),
            "image_probe_sha256": self.image_probe_sha256,
            "gpu_injection": {
                "verified": True,
                "gpu_count": TARGET_GPU_COUNT,
                "gpu_product": EXPECTED_HOST_GPU_PRODUCT,
                "output_sha256": self.gpu_injection_output_sha256,
                "physical_gpu_uuids": list(self.host_physical_gpu_uuids),
                "device_request_sha256": self.gpu_injection_device_request_sha256,
                "network_mode": "none",
                "published_ports": False,
            },
            "fixed_preflight_command": list(FIXED_PREFLIGHT_COMMAND),
            "jit_config_received": False,
            "github_api_credential_received": False,
            "accepted_actions_evidence": False,
        }


def validate_host_preflight_receipt(
    response_body: bytes,
    *,
    plan: ImmutablePlan,
    provider_instance: SnapshotReceipt,
    known_hosts: KnownHostsFileReceipt,
    cloud_init_wait: CloudInitWaitReceipt,
    now: datetime | None = None,
) -> HostPreflightReceipt:
    """Validate the final fixed `probe-host` receipt before any JIT request."""

    _require(
        type(response_body) is bytes and 0 < len(response_body) <= 131_072,
        "host_preflight_response_size_rejected",
    )
    try:
        payload = _strict_json_loads(response_body)
    except (UnicodeDecodeError, ValueError):
        _fail("host_preflight_json_rejected")
    _require(type(payload) is dict, "host_preflight_not_object")
    _require(_canonical_json(payload) == response_body, "host_preflight_not_canonical")
    expected_keys = {
        "schema_version",
        "kind",
        "observed_at",
        "cloud_init_status",
        "cloud_init_output_sha256",
        "effective_uid",
        "root_owned_nonwritable_runtime_bundle",
        "runtime_bundle_sha256",
        "host_physical_gpu_count",
        "host_physical_gpu_uuids",
        "host_physical_gpu_products",
        "gpu_inventory_output_sha256",
        "image",
        "gpu_injection",
        "local_runtime_residue_absent",
        "jit_config_received",
        "github_api_credential_received",
        "github_api_contacted",
        "accepted_actions_evidence",
    }
    _exact_keys(payload, expected_keys, "host_preflight")
    _require(
        type(payload["schema_version"]) is int and payload["schema_version"] == 1,
        "host_preflight_schema_version_rejected",
    )
    _require(
        payload["kind"] == "explainiverse-lambda-jit-host-preflight",
        "host_preflight_kind_rejected",
    )
    observed_text = _text(payload["observed_at"], "host_preflight_observed_at")
    observed = _utc_timestamp(observed_text, "host_preflight_observed_at")
    reference_now = now or datetime.now(timezone.utc)
    _require(reference_now.tzinfo is not None, "host_preflight_now_timezone_missing")
    age = (reference_now.astimezone(timezone.utc) - observed).total_seconds()
    _require(
        -30 <= age <= HOST_PREFLIGHT_FRESHNESS_SECONDS,
        "host_preflight_not_fresh",
    )
    _require(payload["cloud_init_status"] == "done", "host_preflight_cloud_init_not_done")
    _require(
        SHA256_RE.fullmatch(payload["cloud_init_output_sha256"] or "") is not None,
        "host_preflight_cloud_init_digest_rejected",
    )
    _require(
        type(payload["effective_uid"]) is int and payload["effective_uid"] == 0,
        "host_preflight_not_root",
    )
    _require(
        payload["root_owned_nonwritable_runtime_bundle"] is True,
        "host_preflight_runtime_posture_rejected",
    )
    _require(
        payload["runtime_bundle_sha256"] == plan.runtime_bundle_sha256,
        "host_preflight_runtime_bundle_mismatch",
    )
    _require(
        type(payload["host_physical_gpu_count"]) is int and payload["host_physical_gpu_count"] == 8,
        "host_preflight_gpu_count_rejected",
    )
    uuids = payload["host_physical_gpu_uuids"]
    products = payload["host_physical_gpu_products"]
    _require(type(uuids) is list and len(uuids) == 8, "host_preflight_gpu_uuids_rejected")
    _require(
        all(type(value) is str and GPU_UUID_RE.fullmatch(value) is not None for value in uuids),
        "host_preflight_gpu_uuid_rejected",
    )
    _require(len(set(uuids)) == 8, "host_preflight_gpu_uuids_not_distinct")
    _require(
        products == [EXPECTED_HOST_GPU_PRODUCT] * 8,
        "host_preflight_gpu_products_rejected",
    )
    _require(
        SHA256_RE.fullmatch(payload["gpu_inventory_output_sha256"] or "") is not None,
        "host_preflight_gpu_inventory_digest_rejected",
    )

    gpu_injection = _exact_keys(
        payload["gpu_injection"],
        {
            "gpu_injection_verified",
            "gpu_count",
            "gpu_product",
            "output_sha256",
        },
        "host_preflight_gpu_injection",
    )
    _require(
        gpu_injection["gpu_injection_verified"] is True,
        "host_preflight_gpu_injection_not_verified",
    )
    _require(
        type(gpu_injection["gpu_count"]) is int and gpu_injection["gpu_count"] == TARGET_GPU_COUNT,
        "host_preflight_gpu_injection_count",
    )
    _require(
        gpu_injection["gpu_product"] == EXPECTED_HOST_GPU_PRODUCT,
        "host_preflight_gpu_injection_product",
    )
    expected_injection_output = (
        b"gpu_injection=verified\n" b"gpu_count=8\n" b"gpu_product=NVIDIA A100-SXM4-80GB\n"
    )
    expected_injection_output_sha256 = _sha256(expected_injection_output)
    _require(
        gpu_injection["output_sha256"] == expected_injection_output_sha256,
        "host_preflight_gpu_injection_output_digest",
    )
    gpu_device_request_sha256 = _sha256(",".join(uuids).encode("ascii"))

    image = _exact_keys(
        payload["image"],
        {
            "schema_version",
            "kind",
            "observed_at",
            "image_reference",
            "manifest_digest",
            "config_digest",
            "platform",
            "pull_output_sha256",
            "inspect_response_sha256",
            "probe",
            "network_contact_during_probe_container",
            "registry_contacted_for_digest_pull",
            "github_api_contacted",
        },
        "host_preflight_image",
    )
    _require(
        type(image["schema_version"]) is int and image["schema_version"] == 1,
        "host_preflight_image_schema_rejected",
    )
    _require(image["kind"] == "explainiverse-runner-image-probe", "host_preflight_image_kind")
    image_observed = _utc_timestamp(
        _text(image["observed_at"], "image_probe_observed_at"),
        "image_probe_observed_at",
    )
    image_age = (reference_now.astimezone(timezone.utc) - image_observed).total_seconds()
    _require(
        -30 <= image_age <= HOST_PREFLIGHT_FRESHNESS_SECONDS,
        "host_preflight_image_not_fresh",
    )
    _require(
        image["image_reference"] == EXPECTED_RUNNER_IMAGE_REFERENCE,
        "host_preflight_image_reference_mismatch",
    )
    _require(
        image["manifest_digest"] == EXPECTED_RUNNER_IMAGE_MANIFEST,
        "host_preflight_manifest_mismatch",
    )
    _require(
        image["config_digest"] == EXPECTED_RUNNER_IMAGE_CONFIG,
        "host_preflight_config_mismatch",
    )
    _require(image["platform"] == EXPECTED_RUNNER_IMAGE_PLATFORM, "host_preflight_platform")
    for field_name in ("pull_output_sha256", "inspect_response_sha256"):
        _require(
            SHA256_RE.fullmatch(image[field_name] or "") is not None,
            f"host_preflight_{field_name}_rejected",
        )
    probe = _exact_keys(
        image["probe"],
        {
            "container_uid",
            "container_gid",
            "runner_listener_present",
            "runner_listener_version",
            "runner_commit",
            "node20_present",
            "node20_version",
            "node20_sha256",
            "output_sha256",
        },
        "host_preflight_image_probe",
    )
    _require(
        type(probe["container_uid"]) is int and probe["container_uid"] == 1001,
        "host_preflight_container_uid",
    )
    _require(
        type(probe["container_gid"]) is int and probe["container_gid"] == 1001,
        "host_preflight_container_gid",
    )
    _require(probe["runner_listener_present"] is True, "host_preflight_runner_missing")
    _require(
        probe["runner_listener_version"] == EXPECTED_RUNNER_VERSION,
        "host_preflight_runner_version",
    )
    _require(probe["runner_commit"] == EXPECTED_RUNNER_COMMIT, "host_preflight_runner_commit")
    _require(probe["node20_present"] is True, "host_preflight_node_missing")
    _require(probe["node20_version"] == EXPECTED_NODE20_VERSION, "host_preflight_node_version")
    _require(probe["node20_sha256"] == EXPECTED_NODE20_SHA256, "host_preflight_node_digest")
    _require(
        SHA256_RE.fullmatch(probe["output_sha256"] or "") is not None,
        "host_preflight_probe_output_digest",
    )
    _require(
        image["network_contact_during_probe_container"] is False,
        "host_preflight_probe_network_contact",
    )
    _require(
        image["registry_contacted_for_digest_pull"] is True,
        "host_preflight_registry_pull_missing",
    )
    _require(image["github_api_contacted"] is False, "host_preflight_image_github_contact")
    for field_name in ("local_runtime_residue_absent",):
        _require(payload[field_name] is True, f"host_preflight_{field_name}_rejected")
    for field_name in (
        "jit_config_received",
        "github_api_credential_received",
        "github_api_contacted",
        "accepted_actions_evidence",
    ):
        _require(payload[field_name] is False, f"host_preflight_{field_name}_rejected")

    instance_id, instance_ip = _validate_fresh_bound_instance_receipt(provider_instance, plan)
    known_hosts_sha256 = _validate_known_hosts_file_receipt(
        known_hosts,
        plan=plan,
        instance_public_ipv4=instance_ip,
    )
    _require(cloud_init_wait.plan_sha256 == plan.sha256, "host_preflight_cloud_init_plan")
    cloud_init_wait.validate_binding()
    _require(
        SHA256_RE.fullmatch(cloud_init_wait.provider_snapshot_sha256) is not None,
        "host_preflight_cloud_init_provider_snapshot",
    )
    _require(
        NONCE_RE.fullmatch(cloud_init_wait.provider_receipt_nonce) is not None,
        "host_preflight_cloud_init_provider_nonce",
    )
    _require(
        provider_instance.receipt_nonce != cloud_init_wait.provider_receipt_nonce,
        "host_preflight_provider_inventory_not_refreshed",
    )
    for field_name, digest in (
        ("stdout", cloud_init_wait.stdout_sha256),
        ("stderr", cloud_init_wait.stderr_sha256),
    ):
        _require(
            SHA256_RE.fullmatch(digest) is not None,
            f"host_preflight_cloud_init_{field_name}_digest",
        )
    cloud_init_observed = _utc_timestamp(
        cloud_init_wait.observed_at,
        "host_preflight_cloud_init_observed_at",
    )
    readiness_delta = (observed - cloud_init_observed).total_seconds()
    _require(
        -30 <= readiness_delta <= 1_200,
        "host_preflight_cloud_init_order_or_age",
    )
    _require(cloud_init_wait.instance_id == instance_id, "host_preflight_cloud_init_instance")
    _require(
        cloud_init_wait.instance_public_ipv4 == instance_ip,
        "host_preflight_cloud_init_ip",
    )
    _require(
        cloud_init_wait.host_fingerprint == plan.host_key_fingerprint,
        "host_preflight_cloud_init_host_key",
    )
    _require(
        cloud_init_wait.known_hosts_sha256 == known_hosts_sha256,
        "host_preflight_cloud_init_known_hosts",
    )
    _require(
        SHA256_RE.fullmatch(cloud_init_wait.sha256) is not None,
        "host_preflight_cloud_init_binding",
    )
    return HostPreflightReceipt(
        plan_sha256=plan.sha256,
        provider_snapshot_sha256=provider_instance.snapshot_sha256,
        provider_receipt_nonce=provider_instance.receipt_nonce,
        instance_id=instance_id,
        instance_public_ipv4=instance_ip,
        host_fingerprint=plan.host_key_fingerprint,
        known_hosts_sha256=known_hosts_sha256,
        cloud_init_wait_binding_sha256=cloud_init_wait.sha256,
        remote_response_sha256=_sha256(response_body),
        observed_at=observed_text,
        runtime_bundle_sha256=plan.runtime_bundle_sha256,
        host_physical_gpu_uuids=tuple(uuids),
        host_physical_gpu_products=tuple(products),
        image_probe_sha256=_sha256(_canonical_json(image)),
        gpu_injection_output_sha256=expected_injection_output_sha256,
        gpu_injection_device_request_sha256=gpu_device_request_sha256,
    )


def _data(payload: Any, context: str) -> Any:
    return _exact_keys(payload, {"data"}, context)["data"]


def _canonical_file_systems(value: Any) -> list[Mapping[str, Any]]:
    _require(type(value) is list, "file_systems_not_list")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value, 1):
        mapping = _required_keys(
            item,
            {
                "id",
                "name",
                "mount_point",
                "created",
                "created_by",
                "is_in_use",
                "region",
            },
            {
                "id",
                "name",
                "mount_point",
                "created",
                "created_by",
                "is_in_use",
                "region",
                "bytes_used",
            },
            f"file_system_{index}",
        )
        _require(
            RESOURCE_ID_RE.fullmatch(_text(mapping["id"], f"file_system_{index}_id")) is not None,
            f"file_system_{index}_id_rejected",
        )
        _text(mapping["name"], f"file_system_{index}_name")
        mount_point = _text(mapping["mount_point"], f"file_system_{index}_mount_point")
        _require(mount_point.startswith("/"), f"file_system_{index}_mount_point_not_absolute")
        _utc_timestamp(
            _text(mapping["created"], f"file_system_{index}_created"),
            f"file_system_{index}_created",
        )
        created_by = _exact_keys(
            mapping["created_by"],
            {"id", "email", "status"},
            f"file_system_{index}_created_by",
        )
        _text(created_by["id"], f"file_system_{index}_created_by_id")
        _text(created_by["email"], f"file_system_{index}_created_by_email")
        created_by_status = _text(
            created_by["status"],
            f"file_system_{index}_created_by_status",
        )
        _require(
            created_by_status in {"active", "deactivated"},
            f"file_system_{index}_created_by_status",
        )
        _require(type(mapping["is_in_use"]) is bool, f"file_system_{index}_is_in_use")
        _region(mapping["region"], f"file_system_{index}_region")
        if "bytes_used" in mapping:
            _integer(mapping["bytes_used"], f"file_system_{index}_bytes_used", minimum=0)
        result.append(mapping)
    return sorted(result, key=lambda item: str(item["id"]))


def _parse_rules(value: Any, context: str) -> tuple[FirewallRule, ...]:
    _require(type(value) is list, f"{context}_not_list")
    return tuple(
        FirewallRule.from_mapping(rule, f"{context}_{index}") for index, rule in enumerate(value, 1)
    )


def _global_ruleset(value: Any) -> tuple[FirewallRule, ...]:
    mapping = _exact_keys(value, {"id", "name", "rules"}, "global_firewall")
    _require(mapping["id"] == "global", "global_firewall_id_rejected")
    _text(mapping["name"], "global_firewall_name")
    return _parse_rules(mapping["rules"], "global_firewall_rules")


def _region(value: Any, context: str) -> tuple[str, str]:
    mapping = _exact_keys(value, {"name", "description"}, context)
    return _text(mapping["name"], f"{context}_name"), _text(
        mapping["description"], f"{context}_description"
    )


def _validate_action_time_target(
    plan: ImmutablePlan, payloads: Mapping[str, Any], *, require_launch_catalog: bool
) -> None:
    if not require_launch_catalog:
        return
    regions = _data(payloads["regions"], "regions_response")
    _require(type(regions) is list, "regions_data_not_list")
    matching_regions = [
        item for item in regions if type(item) is dict and item.get("name") == TARGET_REGION
    ]
    _require(len(matching_regions) == 1, "target_region_not_unique")
    region_name, description = _region(matching_regions[0], "target_region")
    _require(region_name == TARGET_REGION, "target_region_name_mismatch")
    _require(
        description == TARGET_REGION_DESCRIPTION == plan.region_description,
        "target_region_description_drift",
    )

    instance_types = _data(payloads["instance_types"], "instance_types_response")
    _require(type(instance_types) is dict, "instance_types_data_not_object")
    _require(TARGET_INSTANCE_TYPE in instance_types, "target_instance_type_missing")
    item = _exact_keys(
        instance_types[TARGET_INSTANCE_TYPE],
        {"instance_type", "regions_with_capacity_available"},
        "target_instance_type_item",
    )
    instance_type = _exact_keys(
        item["instance_type"],
        {
            "name",
            "description",
            "gpu_description",
            "price_cents_per_hour",
            "specs",
            "architecture",
        },
        "target_instance_type",
    )
    _require(instance_type["name"] == TARGET_INSTANCE_TYPE, "instance_type_name_drift")
    _require(
        instance_type["description"] == plan.instance_type_description,
        "instance_type_description_drift",
    )
    _validate_a100_api_description(instance_type["gpu_description"])
    _require(
        instance_type["gpu_description"] == plan.gpu_description,
        "instance_type_gpu_product_drift",
    )
    _require(instance_type["architecture"] == TARGET_ARCHITECTURE, "instance_type_arch_drift")
    _require(
        instance_type["price_cents_per_hour"] == plan.price_cents_per_hour,
        "instance_type_price_drift",
    )
    specs = _exact_keys(
        instance_type["specs"],
        {"vcpus", "memory_gib", "storage_gib", "gpus"},
        "instance_type_specs",
    )
    for key in ("vcpus", "memory_gib", "storage_gib", "gpus"):
        _integer(specs[key], f"instance_type_specs_{key}", minimum=1)
    _require(specs["gpus"] == TARGET_GPU_COUNT, "instance_type_gpu_count_drift")
    _require(specs["vcpus"] == plan.vcpus, "instance_type_vcpus_drift")
    _require(specs["memory_gib"] == plan.memory_gib, "instance_type_memory_drift")
    _require(specs["storage_gib"] == plan.storage_gib, "instance_type_storage_drift")
    available = item["regions_with_capacity_available"]
    _require(type(available) is list, "capacity_regions_not_list")
    matching_capacity = [
        entry for entry in available if type(entry) is dict and entry.get("name") == TARGET_REGION
    ]
    _require(len(matching_capacity) == 1, "target_capacity_not_available")
    _require(
        _region(matching_capacity[0], "capacity_region")
        == (TARGET_REGION, plan.region_description),
        "capacity_region_drift",
    )

    images = _data(payloads["images"], "images_response")
    _require(type(images) is list, "images_data_not_list")
    matches = [item for item in images if type(item) is dict and item.get("id") == plan.image_id]
    _require(len(matches) == 1, "target_image_not_unique")
    image = _exact_keys(
        matches[0],
        {
            "id",
            "created_time",
            "updated_time",
            "name",
            "description",
            "family",
            "version",
            "architecture",
            "region",
        },
        "target_image",
    )
    _require(image["created_time"] == plan.image_created_time, "image_created_time_drift")
    _utc_timestamp(_text(image["created_time"], "image_created_time"), "image_created_time")
    _require(image["updated_time"] == plan.image_updated_time, "image_updated_time_drift")
    _require(image["description"] == plan.image_description, "image_description_drift")
    _require(image["name"] == plan.image_name, "image_name_drift")
    _require(image["family"] == plan.image_family, "image_family_drift")
    _require(image["version"] == plan.image_version, "image_version_drift")
    _require(image["architecture"] == TARGET_ARCHITECTURE, "image_architecture_drift")
    _require(
        _region(image["region"], "image_region") == (TARGET_REGION, plan.region_description),
        "image_region_drift",
    )

    ssh_keys = _data(payloads["ssh_keys"], "ssh_keys_response")
    _require(type(ssh_keys) is list, "ssh_keys_data_not_list")
    matches = [
        item for item in ssh_keys if type(item) is dict and item.get("name") == plan.ssh_key_name
    ]
    _require(len(matches) == 1, "ssh_access_key_not_unique")
    ssh_key = _exact_keys(matches[0], {"id", "name", "public_key"}, "ssh_access_key")
    _text(ssh_key["id"], "ssh_access_key_id")
    public_key = _canonical_ed25519_public_key(ssh_key["public_key"], "ssh_access_key")
    _require(
        _sha256(public_key.encode("ascii")) == plan.ssh_public_key_sha256,
        "ssh_access_key_drift",
    )

    file_systems = _canonical_file_systems(_data(payloads["file_systems"], "file_systems_response"))
    _require(
        _sha256(_canonical_json(file_systems)) == plan.baseline_file_systems_sha256,
        "unrelated_file_system_inventory_drift",
    )


def _firewall_rulesets(value: Any) -> list[Mapping[str, Any]]:
    _require(type(value) is list, "firewall_rulesets_not_list")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value, 1):
        mapping = _exact_keys(
            item,
            {"id", "name", "region", "rules", "created", "instance_ids"},
            f"firewall_ruleset_{index}",
        )
        _require(
            RESOURCE_ID_RE.fullmatch(_text(mapping["id"], "ruleset_id")) is not None,
            "ruleset_id_rejected",
        )
        _text(mapping["name"], "ruleset_name")
        _region(mapping["region"], "ruleset_region")
        _parse_rules(mapping["rules"], "ruleset_rules")
        _utc_timestamp(_text(mapping["created"], "ruleset_created"), "ruleset_created")
        _require(type(mapping["instance_ids"]) is list, "ruleset_instance_ids_not_list")
        result.append(mapping)
    return result


INSTANCE_REQUIRED = {
    "id",
    "status",
    "ssh_key_names",
    "file_system_names",
    "region",
    "instance_type",
    "actions",
}
INSTANCE_ALLOWED = INSTANCE_REQUIRED | {
    "name",
    "ip",
    "private_ip",
    "file_system_mounts",
    "image",
    "hostname",
    "jupyter_token",
    "jupyter_url",
    "tags",
    "firewall_rulesets",
}


def _instances(value: Any) -> list[Mapping[str, Any]]:
    _require(type(value) is list, "instances_not_list")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value, 1):
        mapping = _required_keys(item, INSTANCE_REQUIRED, INSTANCE_ALLOWED, f"instance_{index}")
        _require(
            RESOURCE_ID_RE.fullmatch(_text(mapping["id"], "instance_id")) is not None,
            "instance_id_rejected",
        )
        _text(mapping["status"], "instance_status")
        _require(type(mapping["ssh_key_names"]) is list, "instance_ssh_keys_not_list")
        _require(type(mapping["file_system_names"]) is list, "instance_file_systems_not_list")
        _region(mapping["region"], "instance_region")
        result.append(mapping)
    return result


def _tags(value: Any) -> tuple[tuple[str, str], ...]:
    _require(type(value) is list, "instance_tags_not_list")
    parsed: list[tuple[str, str]] = []
    for index, tag in enumerate(value, 1):
        mapping = _exact_keys(tag, {"key", "value"}, f"instance_tag_{index}")
        parsed.append(
            (
                _text(mapping["key"], f"instance_tag_{index}_key"),
                _text(mapping["value"], f"instance_tag_{index}_value"),
            )
        )
    _require(len(set(parsed)) == len(parsed), "instance_tags_duplicate")
    return tuple(sorted(parsed))


def _validate_owned_ruleset(plan: ImmutablePlan, ruleset: Mapping[str, Any]) -> str:
    ruleset_id = _text(ruleset["id"], "owned_ruleset_id")
    _require(ruleset["name"] == plan.ruleset_name, "owned_ruleset_name_mismatch")
    _require(
        _region(ruleset["region"], "owned_ruleset_region")
        == (TARGET_REGION, plan.region_description),
        "owned_ruleset_region_mismatch",
    )
    _require(
        _parse_rules(ruleset["rules"], "owned_ruleset_rules") == plan.desired_firewall_rules,
        "owned_ruleset_rules_mismatch",
    )
    return ruleset_id


def _validate_owned_instance(
    plan: ImmutablePlan,
    instance: Mapping[str, Any],
    ruleset_id: str,
    *,
    require_active: bool = True,
) -> tuple[str, str]:
    instance_id = _text(instance["id"], "owned_instance_id")
    if require_active:
        _require(instance["status"] == "active", "owned_instance_not_active")
    _require(instance.get("name") == plan.instance_name, "owned_instance_name_mismatch")
    _require(instance["ssh_key_names"] == [plan.ssh_key_name], "owned_instance_ssh_key_mismatch")
    _require(instance["file_system_names"] == [], "owned_instance_filesystem_rejected")
    _require(
        _region(instance["region"], "owned_instance_region")
        == (TARGET_REGION, plan.region_description),
        "owned_instance_region_mismatch",
    )
    instance_type = _exact_keys(
        instance["instance_type"],
        {
            "name",
            "description",
            "gpu_description",
            "price_cents_per_hour",
            "specs",
            "architecture",
        },
        "owned_instance_type",
    )
    _require(instance_type["name"] == TARGET_INSTANCE_TYPE, "owned_instance_type_mismatch")
    _require(
        instance_type["description"] == plan.instance_type_description,
        "owned_instance_type_description_mismatch",
    )
    _require(
        instance_type["gpu_description"] == plan.gpu_description, "owned_instance_gpu_mismatch"
    )
    _require(instance_type["architecture"] == TARGET_ARCHITECTURE, "owned_instance_arch_mismatch")
    _require(
        instance_type["price_cents_per_hour"] == plan.price_cents_per_hour,
        "owned_instance_price_mismatch",
    )
    instance_specs = _exact_keys(
        instance_type["specs"],
        {"vcpus", "memory_gib", "storage_gib", "gpus"},
        "owned_instance_specs",
    )
    _require(instance_specs["gpus"] == TARGET_GPU_COUNT, "owned_instance_gpu_count_mismatch")
    _require(instance_specs["vcpus"] == plan.vcpus, "owned_instance_vcpus_mismatch")
    _require(instance_specs["memory_gib"] == plan.memory_gib, "owned_instance_memory_mismatch")
    _require(instance_specs["storage_gib"] == plan.storage_gib, "owned_instance_storage_mismatch")
    image = instance.get("image")
    image_mapping = _exact_keys(image, {"id", "family"}, "owned_instance_image")
    _require(image_mapping["id"] == plan.image_id, "owned_instance_image_mismatch")
    _require(
        image_mapping["family"] == plan.image_family,
        "owned_instance_image_family_mismatch",
    )
    _require(
        _tags(instance.get("tags")) == tuple(sorted(plan.ownership_tags)),
        "owned_instance_tags_mismatch",
    )
    attached = instance.get("firewall_rulesets")
    if type(attached) is not list:
        _fail("owned_instance_rulesets_missing")
    _require(
        len(attached) == 1 and type(attached[0]) is dict and attached[0].get("id") == ruleset_id,
        "owned_instance_ruleset_mismatch",
    )
    public_ip = _text(instance.get("ip"), "owned_instance_public_ip")
    address = ipaddress.ip_address(public_ip)
    _require(
        type(address) is ipaddress.IPv4Address and address.is_global, "owned_instance_ip_not_public"
    )
    return instance_id, address.compressed


class LambdaLiveAdapter:
    """State-verifying one-shot lifecycle adapter.

    The object tracks only receipt nonces already consumed in this process.  It
    does not infer success from an exception and never retries a mutation.
    """

    def __init__(
        self,
        client: LambdaHttpClient,
        plan: ImmutablePlan,
        gates: LiveGates,
        *,
        mutation_intent_callback: MutationIntentCallback | None = None,
    ) -> None:
        gates.validate(plan, require_current=False)
        _require(
            mutation_intent_callback is None or callable(mutation_intent_callback),
            "mutation_intent_callback_not_callable",
        )
        self._client = client
        self._plan = plan
        self._gates = gates
        self._mutation_intent_callback = mutation_intent_callback
        self._mutation_intent_binding_sha256 = (
            self._new_mutation_intent_binding() if mutation_intent_callback is not None else None
        )
        self._observation_receipts_issued = 0
        self._consumed_receipts: set[str] = set()

    @property
    def plan_sha256(self) -> str:
        return self._plan.sha256

    @property
    def mutation_intent_callback_bound(self) -> bool:
        return self._mutation_intent_callback is not None

    @property
    def mutation_intent_binding_sha256(self) -> str | None:
        return self._mutation_intent_binding_sha256

    def _new_mutation_intent_binding(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "kind": "lambda-provider-mutation-intent-callback-binding",
                    "plan_sha256": self._plan.sha256,
                    "binding_nonce": secrets.token_hex(16),
                }
            )
        )

    def bind_mutation_intent_callback(self, callback: MutationIntentCallback) -> str:
        """Bind the sole write-ahead sink before any observation receipt exists."""

        _require(callable(callback), "mutation_intent_callback_not_callable")
        _require(
            self._mutation_intent_callback is None,
            "mutation_intent_callback_already_bound",
        )
        _require(
            self._observation_receipts_issued == 0 and not self._consumed_receipts,
            "mutation_intent_callback_binding_too_late",
        )
        binding_sha256 = self._new_mutation_intent_binding()
        self._mutation_intent_callback = callback
        self._mutation_intent_binding_sha256 = binding_sha256
        return binding_sha256

    def mutation_intent_callback_matches(self, callback: MutationIntentCallback) -> bool:
        """Return whether ``callback`` is the exact sink bound to this adapter."""

        current = self._mutation_intent_callback
        if current is None:
            return False
        current_self = getattr(current, "__self__", None)
        callback_self = getattr(callback, "__self__", None)
        current_function = getattr(current, "__func__", None)
        callback_function = getattr(callback, "__func__", None)
        if current_function is not None or callback_function is not None:
            return current_self is callback_self and current_function is callback_function
        if current_self is not None or callback_self is not None:
            return current_self is callback_self and getattr(current, "__name__", None) == getattr(
                callback, "__name__", None
            )
        return current is callback

    def ambiguity_from_persisted_intent(self, value: Any) -> AmbiguousMutation:
        """Validate a prior write-ahead record without replaying its mutation."""

        intent = MutationIntent.from_public_mapping(value)
        _require(
            intent.plan_sha256 == self._plan.sha256,
            "mutation_intent_plan_mismatch",
        )
        return AmbiguousMutation(
            intent.operation,
            intent.request_sha256,
            "process_exit_after_persisted_intent",
        )

    def _observe_raw(self, phase: str) -> _Snapshot:
        snapshot = _capture_snapshot(self._client)
        _validate_action_time_target(
            self._plan,
            snapshot.payloads,
            require_launch_catalog=phase in {"baseline", "global_restricted", "ruleset_ready"},
        )
        return snapshot

    def observe(self, phase: str) -> SnapshotReceipt:
        _require(
            phase
            in {
                "baseline",
                "global_restricted",
                "ruleset_ready",
                "instance_bound",
                "instance_absent",
                "ruleset_absent",
                "restored",
                "recovery",
            },
            "observation_phase_rejected",
        )
        snapshot = self._observe_raw(phase)
        ruleset_id, instance_id, instance_ip = self._validate_phase(snapshot, phase)
        issued = time.monotonic_ns()
        receipt = SnapshotReceipt(
            plan_sha256=self._plan.sha256,
            phase=phase,
            snapshot_sha256=snapshot.sha256,
            receipt_nonce=secrets.token_hex(16),
            issued_monotonic_ns=issued,
            expires_monotonic_ns=issued + PRESTATE_FRESHNESS_SECONDS * 1_000_000_000,
            ruleset_id=ruleset_id,
            instance_id=instance_id,
            instance_public_ipv4=instance_ip,
            _snapshot=snapshot,
        )
        self._observation_receipts_issued += 1
        return receipt

    def _validate_phase(
        self, snapshot: _Snapshot, phase: str
    ) -> tuple[str | None, str | None, str | None]:
        payloads = snapshot.payloads
        instances = _instances(_data(payloads["instances"], "instances_response"))
        rulesets = _firewall_rulesets(
            _data(payloads["firewall_rulesets"], "firewall_rulesets_response")
        )
        global_rules = _global_ruleset(
            _data(payloads["global_firewall"], "global_firewall_response")
        )
        original = self._plan.original_global_rules
        desired = self._plan.desired_firewall_rules
        owned_rulesets = [item for item in rulesets if item["name"] == self._plan.ruleset_name]

        if phase == "baseline":
            _require(instances == [], "baseline_instances_not_zero")
            _require(rulesets == [], "baseline_rulesets_not_zero")
            _require(global_rules == original, "baseline_global_firewall_drift")
            return None, None, None
        if phase == "global_restricted":
            _require(instances == [], "restricted_instances_not_zero")
            _require(rulesets == [], "restricted_rulesets_not_zero")
            _require(global_rules == desired, "global_firewall_not_restricted")
            return None, None, None
        if phase == "ruleset_ready":
            _require(instances == [], "ruleset_ready_instances_not_zero")
            _require(len(rulesets) == 1 and len(owned_rulesets) == 1, "owned_ruleset_not_sole")
            _require(global_rules == desired, "ruleset_ready_global_not_restricted")
            ruleset_id = _validate_owned_ruleset(self._plan, owned_rulesets[0])
            _require(owned_rulesets[0]["instance_ids"] == [], "owned_ruleset_already_attached")
            return ruleset_id, None, None
        if phase == "instance_bound":
            _require(len(rulesets) == 1 and len(owned_rulesets) == 1, "owned_ruleset_not_sole")
            _require(len(instances) == 1, "owned_instance_not_sole")
            _require(global_rules == desired, "instance_global_not_restricted")
            ruleset_id = _validate_owned_ruleset(self._plan, owned_rulesets[0])
            instance_id, instance_ip = _validate_owned_instance(
                self._plan, instances[0], ruleset_id
            )
            _require(
                owned_rulesets[0]["instance_ids"] == [instance_id],
                "owned_ruleset_instance_binding_mismatch",
            )
            return ruleset_id, instance_id, instance_ip
        if phase == "instance_absent":
            _require(instances == [], "terminated_instance_inventory_not_zero")
            _require(len(rulesets) == 1 and len(owned_rulesets) == 1, "owned_ruleset_not_sole")
            _require(global_rules == desired, "post_terminate_global_not_restricted")
            ruleset_id = _validate_owned_ruleset(self._plan, owned_rulesets[0])
            _require(owned_rulesets[0]["instance_ids"] == [], "terminated_ruleset_still_attached")
            return ruleset_id, None, None
        if phase == "ruleset_absent":
            _require(instances == [], "post_delete_instances_not_zero")
            _require(rulesets == [], "post_delete_rulesets_not_zero")
            _require(global_rules == desired, "post_delete_global_not_restricted")
            return None, None, None
        if phase == "restored":
            _require(instances == [], "restored_instances_not_zero")
            _require(rulesets == [], "restored_rulesets_not_zero")
            _require(global_rules == original, "global_firewall_not_restored")
            return None, None, None
        # Recovery deliberately does not assert one state; classification does.
        return None, None, None

    def _validate_owned_instance_in_progress(self, snapshot: _Snapshot) -> tuple[str, str]:
        """Accept only the exact owned instance in a non-active provider state."""

        payloads = snapshot.payloads
        instances = _instances(_data(payloads["instances"], "instances_response"))
        rulesets = _firewall_rulesets(
            _data(payloads["firewall_rulesets"], "firewall_rulesets_response")
        )
        global_rules = _global_ruleset(
            _data(payloads["global_firewall"], "global_firewall_response")
        )
        _require(global_rules == self._plan.desired_firewall_rules, "pending_global_drift")
        _require(len(instances) == 1, "pending_instance_not_sole")
        _require(len(rulesets) == 1, "pending_ruleset_not_sole")
        ruleset_id = _validate_owned_ruleset(self._plan, rulesets[0])
        instance_id, _ = _validate_owned_instance(
            self._plan,
            instances[0],
            ruleset_id,
            require_active=False,
        )
        _require(instances[0]["status"] != "active", "pending_instance_is_active")
        _require(
            rulesets[0]["instance_ids"] == [instance_id],
            "pending_ruleset_instance_binding_mismatch",
        )
        return ruleset_id, instance_id

    def _consume(self, receipt: SnapshotReceipt, operation: str) -> None:
        self._gates.validate(
            self._plan,
            require_current=operation in {"restrict_global", "create_ruleset", "launch"},
        )
        _require(receipt.plan_sha256 == self._plan.sha256, "prestate_plan_mismatch")
        _require(receipt.phase == MUTATION_PRESTATE[operation], "prestate_phase_mismatch")
        _require(receipt.snapshot_sha256 == receipt._snapshot.sha256, "prestate_snapshot_mismatch")
        _require(receipt.receipt_nonce not in self._consumed_receipts, "prestate_receipt_reused")
        now = time.monotonic_ns()
        _require(
            receipt.issued_monotonic_ns <= now <= receipt.expires_monotonic_ns,
            "prestate_receipt_stale",
        )
        self._consumed_receipts.add(receipt.receipt_nonce)

    def _mutate(
        self,
        operation: str,
        receipt: SnapshotReceipt,
        body: Mapping[str, Any] | None,
        *,
        path: str | None = None,
        sensitive_body: bool = False,
    ) -> tuple[Mapping[str, Any], ResponseBinding]:
        self._consume(receipt, operation)
        method, expected_path = MUTATION_PATHS[operation]
        request_path = path or expected_path
        if operation != "delete_ruleset":
            _require(request_path == expected_path, "mutation_path_drift")
        body_bytes = bytearray(_canonical_json(body)) if body is not None else None
        request = ProviderRequest(
            operation=operation,
            method=method,
            path=request_path,
            mutating=True,
            body=body_bytes,
            sensitive_body=sensitive_body,
        )
        intent = MutationIntent(
            plan_sha256=self._plan.sha256,
            operation=operation,
            prestate_phase=receipt.phase,
            prestate_snapshot_sha256=receipt.snapshot_sha256,
            prestate_receipt_nonce=receipt.receipt_nonce,
            callback_binding_sha256=self._mutation_intent_binding_sha256 or "",
            method=request.method,
            path=request.path,
            request_sha256=request.request_sha256,
            request_body_sha256=request.body_sha256,
            sensitive_body=request.sensitive_body,
            timeout_seconds=request.timeout_seconds,
        )
        try:
            callback = self._mutation_intent_callback
            if callback is None:
                _fail("mutation_intent_callback_not_bound")
            _require(
                SHA256_RE.fullmatch(intent.callback_binding_sha256) is not None,
                "mutation_intent_callback_binding_missing",
            )
            callback(intent)
            return self._client._request_mutation(request)
        finally:
            if request.sensitive_body:
                request.destroy_body()

    def restrict_global(self, receipt: SnapshotReceipt) -> MutationReceipt:
        payload, binding = self._mutate(
            "restrict_global",
            receipt,
            {"rules": [rule.to_mapping() for rule in self._plan.desired_firewall_rules]},
        )
        try:
            rules = _global_ruleset(_data(payload, "restrict_global_response"))
            _require(
                rules == self._plan.desired_firewall_rules,
                "restrict_global_response_drift",
            )
        except ContractError:
            raise AmbiguousMutation(
                "restrict_global", binding.request_sha256, "response_schema"
            ) from None
        return self._mutation_receipt(receipt, binding)

    def create_ruleset(self, receipt: SnapshotReceipt) -> MutationReceipt:
        payload, binding = self._mutate(
            "create_ruleset",
            receipt,
            {
                "name": self._plan.ruleset_name,
                "region": TARGET_REGION,
                "rules": [rule.to_mapping() for rule in self._plan.desired_firewall_rules],
            },
        )
        try:
            ruleset = _data(payload, "create_ruleset_response")
            parsed = _firewall_rulesets([ruleset])
            ruleset_id = _validate_owned_ruleset(self._plan, parsed[0])
            _require(
                parsed[0]["instance_ids"] == [],
                "created_ruleset_already_attached",
            )
        except ContractError:
            raise AmbiguousMutation(
                "create_ruleset", binding.request_sha256, "response_schema"
            ) from None
        return self._mutation_receipt(receipt, binding, ruleset_id=ruleset_id)

    def launch(
        self,
        receipt: SnapshotReceipt,
        identity: HostIdentity,
        runtime_bundle: RuntimeBundle,
    ) -> MutationReceipt:
        _require(receipt.ruleset_id is not None, "launch_ruleset_id_missing")
        _require(
            identity.fingerprint == self._plan.host_key_fingerprint, "launch_host_key_mismatch"
        )
        _require(
            runtime_bundle.sha256 == self._plan.runtime_bundle_sha256,
            "launch_runtime_bundle_mismatch",
        )
        cloud_init = identity.cloud_init(runtime_bundle)
        try:
            user_data = cloud_init.copy_bytes().decode("ascii")
            body = {
                "region_name": TARGET_REGION,
                "instance_type_name": TARGET_INSTANCE_TYPE,
                "ssh_key_names": [self._plan.ssh_key_name],
                "file_system_names": [],
                "name": self._plan.instance_name,
                "image": {"id": self._plan.image_id},
                "user_data": user_data,
                "tags": [{"key": key, "value": value} for key, value in self._plan.ownership_tags],
                "firewall_rulesets": [{"id": receipt.ruleset_id}],
            }
            payload, binding = self._mutate("launch", receipt, body, sensitive_body=True)
        finally:
            cloud_init.destroy()
            identity.destroy()
        try:
            response = _exact_keys(
                _data(payload, "launch_response"), {"instance_ids"}, "launch_data"
            )
            ids = response["instance_ids"]
            _require(
                type(ids) is list and len(ids) == 1,
                "launch_instance_id_count_rejected",
            )
            instance_id = _text(ids[0], "launch_instance_id")
            _require(
                RESOURCE_ID_RE.fullmatch(instance_id) is not None,
                "launch_instance_id_rejected",
            )
        except ContractError:
            raise AmbiguousMutation("launch", binding.request_sha256, "response_schema") from None
        return self._mutation_receipt(
            receipt,
            binding,
            ruleset_id=receipt.ruleset_id,
            instance_id=instance_id,
            host_fingerprint=identity.fingerprint,
        )

    def terminate(self, receipt: SnapshotReceipt) -> MutationReceipt:
        _require(receipt.instance_id is not None, "terminate_instance_id_missing")
        payload, binding = self._mutate(
            "terminate", receipt, {"instance_ids": [receipt.instance_id]}
        )
        try:
            response = _exact_keys(
                _data(payload, "terminate_response"),
                {"terminated_instances"},
                "terminate_data",
            )
            terminated = _instances(response["terminated_instances"])
            _require(len(terminated) == 1, "terminate_instance_count_rejected")
            _require(
                terminated[0]["id"] == receipt.instance_id,
                "terminate_instance_id_mismatch",
            )
            _require(
                _tags(terminated[0].get("tags")) == tuple(sorted(self._plan.ownership_tags)),
                "terminate_instance_ownership_mismatch",
            )
        except ContractError:
            raise AmbiguousMutation(
                "terminate", binding.request_sha256, "response_schema"
            ) from None
        return self._mutation_receipt(
            receipt,
            binding,
            ruleset_id=receipt.ruleset_id,
            instance_id=receipt.instance_id,
        )

    def delete_ruleset(self, receipt: SnapshotReceipt) -> MutationReceipt:
        ruleset_id = receipt.ruleset_id
        if ruleset_id is None:
            _fail("delete_ruleset_id_missing")
        quoted_id = urllib.parse.quote(ruleset_id, safe="")
        path = f"{API_PREFIX}/firewall-rulesets/{quoted_id}"
        payload, binding = self._mutate("delete_ruleset", receipt, None, path=path)
        try:
            _require(
                _data(payload, "delete_ruleset_response") == {},
                "delete_ruleset_data_not_empty",
            )
        except ContractError:
            raise AmbiguousMutation(
                "delete_ruleset", binding.request_sha256, "response_schema"
            ) from None
        return self._mutation_receipt(receipt, binding, ruleset_id=receipt.ruleset_id)

    def restore_global(self, receipt: SnapshotReceipt) -> MutationReceipt:
        payload, binding = self._mutate(
            "restore_global",
            receipt,
            {"rules": [rule.to_mapping() for rule in self._plan.original_global_rules]},
        )
        try:
            rules = _global_ruleset(_data(payload, "restore_global_response"))
            _require(
                rules == self._plan.original_global_rules,
                "restore_global_response_drift",
            )
        except ContractError:
            raise AmbiguousMutation(
                "restore_global", binding.request_sha256, "response_schema"
            ) from None
        return self._mutation_receipt(receipt, binding)

    def _mutation_receipt(
        self,
        prestate: SnapshotReceipt,
        binding: ResponseBinding,
        *,
        ruleset_id: str | None = None,
        instance_id: str | None = None,
        host_fingerprint: str | None = None,
    ) -> MutationReceipt:
        return MutationReceipt(
            plan_sha256=self._plan.sha256,
            operation=binding.operation,
            prestate_snapshot_sha256=prestate.snapshot_sha256,
            request_sha256=binding.request_sha256,
            request_body_sha256=binding.request_body_sha256,
            response_body_sha256=binding.response_body_sha256,
            ruleset_id=ruleset_id,
            instance_id=instance_id,
            host_fingerprint=host_fingerprint,
        )

    def recover_ambiguous(
        self, ambiguity: AmbiguousMutation, receipt: SnapshotReceipt
    ) -> RecoveryReceipt:
        """Classify one ambiguous mutation from a new full inventory.

        This method never repeats the mutation.  Only exact before/after states
        are accepted; partial or foreign state is a hard stop.
        """

        self._gates.validate(self._plan, require_current=False)
        _require(receipt.phase == "recovery", "recovery_receipt_phase_rejected")
        _require(receipt.plan_sha256 == self._plan.sha256, "recovery_plan_mismatch")
        _require(receipt.snapshot_sha256 == receipt._snapshot.sha256, "recovery_snapshot_mismatch")
        _require(
            receipt.receipt_nonce not in self._consumed_receipts,
            "recovery_receipt_reused",
        )
        now = time.monotonic_ns()
        _require(
            receipt.issued_monotonic_ns <= now <= receipt.expires_monotonic_ns,
            "recovery_receipt_stale",
        )
        self._consumed_receipts.add(receipt.receipt_nonce)
        _require(
            ambiguity.operation in MUTATION_PATHS,
            "recovery_operation_rejected",
        )
        before_phase = MUTATION_PRESTATE[ambiguity.operation]
        after_phase = {
            "restrict_global": "global_restricted",
            "create_ruleset": "ruleset_ready",
            "launch": "instance_bound",
            "terminate": "instance_absent",
            "delete_ruleset": "ruleset_absent",
            "restore_global": "restored",
        }[ambiguity.operation]
        outcome: str
        ruleset_id: str | None
        instance_id: str | None
        try:
            ruleset_id, instance_id, _ = self._validate_phase(receipt._snapshot, after_phase)
            outcome = "applied_exactly_once"
        except ContractError:
            try:
                ruleset_id, instance_id, _ = self._validate_phase(receipt._snapshot, before_phase)
                outcome = "not_applied"
            except ContractError:
                if ambiguity.operation in {"launch", "terminate"}:
                    try:
                        ruleset_id, instance_id = self._validate_owned_instance_in_progress(
                            receipt._snapshot
                        )
                        outcome = "applied_in_progress"
                    except ContractError:
                        raise ContractError(
                            "ambiguous_mutation_has_partial_or_foreign_state"
                        ) from None
                else:
                    raise ContractError("ambiguous_mutation_has_partial_or_foreign_state") from None
        return RecoveryReceipt(
            plan_sha256=self._plan.sha256,
            operation=ambiguity.operation,
            ambiguous_request_sha256=ambiguity.request_sha256,
            inventory_snapshot_sha256=receipt.snapshot_sha256,
            outcome=outcome,
            ruleset_id=ruleset_id,
            instance_id=instance_id,
        )


def write_public_evidence(
    path: str | os.PathLike[str],
    value: Mapping[str, Any],
    *,
    evidence_directory_receipt: EvidenceDirectoryReceipt,
) -> str:
    """Atomically write sanitized public evidence in the held exact directory."""

    _require(
        type(evidence_directory_receipt) is EvidenceDirectoryReceipt,
        "evidence_directory_receipt_type_rejected",
    )
    evidence_directory_receipt.validate()
    destination = Path(path)
    _require(destination.is_absolute(), "public_evidence_path_not_absolute")
    resolved_destination = destination.resolve(strict=False)
    _require(destination == resolved_destination, "public_evidence_path_not_canonical")
    destination = resolved_destination
    _require(
        destination.parent == Path(evidence_directory_receipt.absolute_path),
        "public_evidence_outside_evidence_directory",
    )
    payload = _canonical_json(value) + b"\n"
    _write_exclusive_bytes(destination, payload)
    evidence_directory_receipt.validate()
    return _sha256(payload)


def dry_run_contract() -> dict[str, Any]:
    """Credential-free declaration; it never performs provider contact."""

    return {
        "schema_version": 1,
        "kind": "explainiverse-lambda-live-adapter-dry-run",
        "openapi": {
            "openapi_version": OPENAPI_VERSION,
            "api_version": LAMBDA_API_VERSION,
            "document_sha256": OPENAPI_SHA256,
            "production_origin": PRODUCTION_ORIGIN,
        },
        "allowed_target": {
            "instance_type_name": TARGET_INSTANCE_TYPE,
            "region_name": TARGET_REGION,
            "architecture": TARGET_ARCHITECTURE,
            "image_family": TARGET_IMAGE_FAMILY,
            "gpu_description": "action-time API value constrained to A100 80 GB SXM4",
            "physical_gpu_count": TARGET_GPU_COUNT,
        },
        "exact_read_operations": [{"method": "GET", "path": path} for _, path in READ_OPERATIONS],
        "exact_mutation_operations": [
            {"operation": name, "method": method, "path": path}
            for name, (method, path) in MUTATION_PATHS.items()
        ],
        "provider_request_pacing": {
            "minimum_seconds_between_starts": PROVIDER_MIN_REQUEST_INTERVAL_SECONDS,
            "full_snapshot_minimum_pacing_seconds": (len(READ_OPERATIONS) - 1)
            * PROVIDER_MIN_REQUEST_INTERVAL_SECONDS,
            "observation_window_seconds": MAX_OBSERVATION_WINDOW_SECONDS,
            "prestate_freshness_seconds": PRESTATE_FRESHNESS_SECONDS,
            "automatic_retry": False,
        },
        "production_authorized": False,
        "provider_mutation_authorized": False,
        "live_go": False,
        "provider_contacted": False,
        "mutations_performed": False,
        "credentials_accepted": False,
        "api_key_lifecycle_supported": False,
        "mutation_retry_supported": False,
        "provider_mutation_write_ahead_intent_required": True,
        "persisted_mutation_intent_exact_schema_revalidated": True,
        "ambiguous_outcome_requires_inventory": True,
        "fixed_cloud_init_wait_command": list(FIXED_CLOUD_INIT_WAIT_COMMAND),
        "fixed_host_preflight_command": list(FIXED_PREFLIGHT_COMMAND),
        "fixed_runtime_command": list(FIXED_REMOTE_COMMAND),
        "host_private_key_persisted": False,
        "access_identity_private_metadata_persisted": False,
        "access_identity_provider_public_key_binding_required": True,
        "evidence_directory_exact_acl_receipt_required": True,
        "evidence_file_no_replace_atomic_publish_required": True,
        "jit_config_persisted": False,
    }
