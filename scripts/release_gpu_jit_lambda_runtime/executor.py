"""Production runtime adapter for one disposable GitHub Actions JIT runner.

The executable is intentionally narrow:

* one canonical per-job plan arrives on inherited anonymous FD 4;
* one encoded JIT configuration arrives on inherited anonymous FD 3;
* Docker receives the JIT configuration through stdin, never argv, an
  environment option, a bind mount, or a file;
* stdout contains only a canonical sanitized receipt.

The module has no GitHub credential or HTTP client and performs no provider or
GitHub operation.  The trusted local control plane must create the JIT
configuration immediately before invoking it, observe the job, and
independently verify the final zero-runner inventory after it returns.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

try:
    from . import runtime_contract as contract
except ImportError:  # pragma: no cover - direct remote-host invocation
    import runtime_contract as contract  # type: ignore[import-not-found,no-redef]

MAX_PLAN_BYTES = 1_048_576
MAX_JIT_BYTES = 1_048_576
MAX_COMMAND_OUTPUT_BYTES = 1_048_576
ANONYMOUS_FD_RE = re.compile(r"pipe:\[[0-9]+\]\Z|/memfd:[^/]+ \(deleted\)\Z")
JIT_CONFIG_RE = re.compile(rb"[A-Za-z0-9_+/=-]+\Z")
ABSTRACT_LOCK_NAME = "\0explainiverse-lambda-jit-runtime-v1"
SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:^GH_|^GITHUB_|^ACTIONS_|^LAMBDA_|^AWS_|^AZURE_|^GOOGLE_|"
    r"TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIAL|PRIVATE_KEY|API_KEY|JITCONFIG|"
    r"DOCKER_AUTH_CONFIG|SSH_AUTH_SOCK)",
    re.IGNORECASE,
)


class RuntimeErrorClosed(RuntimeError):
    """Stable fail-closed runtime error whose text never contains secrets."""


def _fail(code: str) -> NoReturn:
    raise RuntimeErrorClosed(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _zeroize(value: bytearray | None) -> None:
    if value is not None:
        value[:] = b"\x00" * len(value)


def _read_anonymous_fd(
    fd: int,
    *,
    maximum: int,
    context: str,
    absolute_deadline: float | None = None,
) -> bytearray:
    """Read and close one inherited pipe/socket/deleted-memfd with a deadline."""

    _require(sys.platform.startswith("linux"), f"{context}_linux_required")
    _require(type(fd) is int and 3 <= fd <= 1024, f"{context}_fd_rejected")
    try:
        target = os.readlink(f"/proc/self/fd/{fd}")
        status = os.fstat(fd)
    except OSError:
        _fail(f"{context}_fd_unavailable")
    _require(ANONYMOUS_FD_RE.fullmatch(target) is not None, f"{context}_fd_not_anonymous")
    _require(
        stat.S_ISFIFO(status.st_mode) or target.startswith("/memfd:"),
        f"{context}_fd_type_rejected",
    )
    deadline = time.monotonic() + contract.FD_READ_SECONDS
    if absolute_deadline is not None:
        deadline = min(deadline, absolute_deadline)
    result = bytearray(maximum + 1)
    length = 0
    try:
        os.set_blocking(fd, False)
        while True:
            try:
                view = memoryview(result)[length : min(length + 65_536, maximum + 1)]
                count = getattr(os, "readv")(fd, [view])
                view.release()
            except BlockingIOError:
                view.release()
                if time.monotonic() >= deadline:
                    _fail(f"{context}_fd_timeout")
                time.sleep(0.01)
                continue
            if count == 0:
                break
            length += count
            _require(length <= maximum, f"{context}_too_large")
            if time.monotonic() >= deadline:
                _fail(f"{context}_fd_timeout")
    except BaseException:
        _zeroize(result)
        raise
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    del result[length:]
    _require(len(result) > 0, f"{context}_empty")
    return result


def validate_jit_config(value: bytearray, expected_sha256: str) -> None:
    _require(100 <= len(value) <= MAX_JIT_BYTES, "jit_config_length_rejected")
    _require(JIT_CONFIG_RE.fullmatch(value) is not None, "jit_config_encoding_rejected")
    observed = hashlib.sha256(value).hexdigest()
    _require(observed == expected_sha256, "jit_config_digest_mismatch")


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    stdout: bytes
    stderr: bytes


class HostCommands:
    """Shell-free bounded subprocess runner."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        expected: Sequence[int] = (0,),
        maximum_output: int = MAX_COMMAND_OUTPUT_BYTES,
    ) -> CommandResult:
        _require(type(argv) in {tuple, list} and len(argv) > 0, "host_argv_rejected")
        _require(all(type(item) is str and "\x00" not in item for item in argv), "host_argv_unsafe")
        try:
            process = subprocess.run(
                tuple(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                timeout=timeout,
                check=False,
                env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
        except (OSError, subprocess.TimeoutExpired):
            _fail("host_command_transport_failure")
        _require(
            len(process.stdout) <= maximum_output and len(process.stderr) <= maximum_output,
            "host_command_output_too_large",
        )
        _require(process.returncode in expected, "host_command_exit_rejected")
        return CommandResult(tuple(argv), process.returncode, process.stdout, process.stderr)


class DeadlineCommands(HostCommands):
    """Cap every host command to one shared monotonic deadline."""

    def __init__(self, inner: HostCommands, deadline: float, error_code: str) -> None:
        self._inner = inner
        self._deadline = deadline
        self._error_code = error_code

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: int = 30,
        expected: Sequence[int] = (0,),
        maximum_output: int = MAX_COMMAND_OUTPUT_BYTES,
    ) -> CommandResult:
        remaining = self._deadline - time.monotonic()
        _require(remaining > 0, self._error_code)
        bounded_timeout = max(1, min(timeout, math.ceil(remaining)))
        try:
            result = self._inner.run(
                argv,
                timeout=bounded_timeout,
                expected=expected,
                maximum_output=maximum_output,
            )
        except RuntimeErrorClosed:
            if time.monotonic() >= self._deadline:
                _fail(self._error_code)
            raise
        _require(time.monotonic() <= self._deadline, self._error_code)
        return result


def _authority_deadline(plan: Mapping[str, Any]) -> float:
    expires_text = str(plan["authority_window"]["expires_at"])
    try:
        expires = datetime.fromisoformat(expires_text.removesuffix("Z") + "+00:00")
    except ValueError:
        _fail("authority_expiry_rejected")
    seconds = (expires - _utc_now()).total_seconds()
    _require(seconds > 0, "authority_window_expired")
    return time.monotonic() + seconds


def _verify_host_binary(path: str) -> None:
    candidate = Path(path)
    try:
        link_status = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        status = resolved.stat()
    except OSError:
        _fail("host_binary_missing")
    _require(
        (stat.S_ISREG(link_status.st_mode) or stat.S_ISLNK(link_status.st_mode))
        and stat.S_ISREG(status.st_mode)
        and status.st_uid == 0
        and status.st_mode & 0o111 != 0
        and status.st_mode & 0o022 == 0,
        "host_binary_posture_rejected",
    )


def runtime_bundle_sha256() -> str:
    """Hash the exact executable source bundle with path framing."""

    bundle_root = Path(__file__).absolute().parent
    try:
        root_link_status = bundle_root.lstat()
        root_status = bundle_root.stat()
        present_names = sorted(path.name for path in bundle_root.iterdir())
    except OSError:
        _fail("runtime_bundle_directory_unreadable")
    expected_names = ["__init__.py", "bootstrap.py", "executor.py", "runtime_contract.py"]
    _require(
        stat.S_ISDIR(root_link_status.st_mode)
        and stat.S_ISDIR(root_status.st_mode)
        and root_link_status.st_uid == 0
        and root_status.st_uid == 0
        and root_status.st_mode & 0o022 == 0
        and present_names == expected_names,
        "runtime_bundle_directory_posture_rejected",
    )
    digest = hashlib.sha256()
    for name in expected_names:
        path = bundle_root / name
        try:
            link_status = path.lstat()
            status = path.stat()
            payload = path.read_bytes()
        except OSError:
            _fail("runtime_bundle_unreadable")
        _require(
            stat.S_ISREG(link_status.st_mode)
            and stat.S_ISREG(status.st_mode)
            and status.st_uid == 0
            and status.st_mode & 0o022 == 0,
            "runtime_bundle_posture_rejected",
        )
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(2, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def verify_host_posture() -> None:
    _require(sys.platform.startswith("linux"), "linux_host_required")
    _require(getattr(os, "geteuid")() == 0, "root_control_process_required")
    for path in (
        contract.DOCKER_PATH,
        contract.IPTABLES_PATH,
        contract.NVIDIA_SMI_PATH,
        contract.PYTHON_PATH,
    ):
        _verify_host_binary(path)
    try:
        swaps = Path("/proc/swaps").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError):
        _fail("host_swap_inventory_unavailable")
    _require(len(swaps) == 1 and swaps[0].split()[:2] == ["Filename", "Type"], "host_swap_enabled")


def harden_secret_process() -> None:
    """Disable core dumps and process dumping before secret bytes are read."""

    _require(sys.platform.startswith("linux"), "linux_host_required")
    try:
        import resource

        setrlimit = getattr(resource, "setrlimit")
        rlimit_core = getattr(resource, "RLIMIT_CORE")
        setrlimit(rlimit_core, (0, 0))
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE
    except (ImportError, OSError, ValueError):
        _fail("secret_process_hardening_failed")
    _require(result == 0, "secret_process_hardening_failed")


def verify_no_sensitive_environment() -> None:
    """Reject inherited credential-shaped environment variables without reading values."""

    _require(
        not any(SENSITIVE_ENV_NAME_RE.search(name) for name in os.environ),
        "sensitive_environment_rejected",
    )


def require_probe_stdin_eof() -> None:
    """Require the fixed public probe command to receive no stdin payload."""

    _require(sys.platform.startswith("linux"), "linux_host_required")
    try:
        status = os.fstat(0)
        _require(
            stat.S_ISFIFO(status.st_mode)
            or stat.S_ISSOCK(status.st_mode)
            or stat.S_ISCHR(status.st_mode),
            "probe_stdin_type_rejected",
        )
        os.set_blocking(0, False)
    except OSError:
        _fail("probe_stdin_unavailable")
    deadline = time.monotonic() + contract.FD_READ_SECONDS
    probe = bytearray(1)
    try:
        while True:
            view = memoryview(probe)
            try:
                count = getattr(os, "readv")(0, [view])
            except BlockingIOError:
                view.release()
                _require(time.monotonic() < deadline, "probe_stdin_eof_timeout")
                time.sleep(0.01)
                continue
            view.release()
            _require(count == 0, "probe_stdin_payload_rejected")
            return
    finally:
        _zeroize(probe)


class ExclusiveRuntimeLock:
    """A non-filesystem Linux abstract socket lock enforcing host sequencing."""

    def __init__(self) -> None:
        self._socket: socket.socket | None = None

    def __enter__(self) -> ExclusiveRuntimeLock:
        _require(hasattr(socket, "AF_UNIX"), "unix_socket_required")
        holder = socket.socket(getattr(socket, "AF_UNIX"), socket.SOCK_STREAM)
        try:
            holder.bind(ABSTRACT_LOCK_NAME)
            holder.listen(1)
        except OSError:
            holder.close()
            _fail("another_runtime_is_active")
        self._socket = holder
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class TerminationGuard:
    """Turn catchable termination signals into exceptions so cleanup runs."""

    SIGNAL_NAMES = ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT")

    def __init__(self) -> None:
        self._previous: dict[int, Any] = {}

    @staticmethod
    def _handle(signum: int, frame: object) -> NoReturn:
        del signum, frame
        _fail("termination_signal_received")

    def __enter__(self) -> TerminationGuard:
        for name in self.SIGNAL_NAMES:
            value = getattr(signal, name, None)
            if value is None:
                continue
            self._previous[value] = signal.getsignal(value)
            signal.signal(value, self._handle)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        for value, previous in self._previous.items():
            signal.signal(value, previous)
        self._previous.clear()


def _ensure_no_runtime_residue(commands: HostCommands, plan: Mapping[str, Any]) -> None:
    names = contract.runtime_names(plan)
    for argv in (
        (contract.DOCKER_PATH, "container", "inspect", names["container"]),
        (contract.DOCKER_PATH, "network", "inspect", names["network"]),
        (contract.IPTABLES_PATH, "-S", names["chain"]),
        (
            contract.IPTABLES_PATH,
            "-C",
            "DOCKER-USER",
            "-i",
            names["bridge"],
            "-j",
            names["chain"],
        ),
    ):
        result = commands.run(argv, expected=(0, 1), maximum_output=32_768)
        _require(result.exit_code == 1, "runtime_named_residue_present")
    _ensure_no_global_runtime_residue(commands)


def _ensure_no_global_runtime_residue(commands: HostCommands) -> None:
    probe = commands.run(
        (contract.DOCKER_PATH, "container", "inspect", contract.PROBE_CONTAINER_NAME),
        expected=(0, 1),
        maximum_output=32_768,
    )
    _require(probe.exit_code == 1, "probe_container_residue")
    containers = commands.run(
        (
            contract.DOCKER_PATH,
            "container",
            "ls",
            "--all",
            "--filter",
            "label=org.opencontainers.image.vendor=explainiverse-release-control",
            "--format",
            "{{.ID}}",
        )
    )
    networks = commands.run(
        (
            contract.DOCKER_PATH,
            "network",
            "ls",
            "--filter",
            "label=org.opencontainers.image.vendor=explainiverse-release-control",
            "--format",
            "{{.ID}}",
        )
    )
    _require(containers.stdout.strip() == b"" and networks.stdout.strip() == b"", "runtime_residue")
    rules = commands.run((contract.IPTABLES_PATH, "-S"))
    _require(b"EXJIT_" not in rules.stdout, "runtime_firewall_residue")


def _verify_network(commands: HostCommands, plan: Mapping[str, Any]) -> None:
    names = contract.runtime_names(plan)
    inspected = commands.run((contract.DOCKER_PATH, "network", "inspect", names["network"]))
    try:
        values = json.loads(inspected.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("network_inspect_json_rejected")
    _require(type(values) is list and len(values) == 1, "network_inspect_cardinality")
    network = values[0]
    _require(
        type(network) is dict
        and network.get("Name") == names["network"]
        and network.get("Driver") == "bridge"
        and network.get("Internal") is False
        and network.get("EnableIPv6") is False
        and network.get("Containers") == {},
        "network_inspect_rejected",
    )
    ipam = network.get("IPAM")
    _require(type(ipam) is dict and type(ipam.get("Config")) is list, "network_ipam_rejected")
    _require(
        len(ipam["Config"]) == 1 and ipam["Config"][0].get("Subnet") == contract.NETWORK_SUBNET,
        "network_subnet_rejected",
    )
    options = network.get("Options")
    _require(
        type(options) is dict
        and options.get("com.docker.network.bridge.name") == names["bridge"]
        and options.get("com.docker.network.bridge.enable_icc") == "false"
        and options.get("com.docker.network.bridge.enable_ip_masquerade") == "true",
        "network_options_rejected",
    )
    for argv in contract.expected_network_setup_argv(plan)[2:]:
        _require(argv[1] == "-A" or argv[1] == "-I", "firewall_model_rejected")
        if argv[1] == "-A":
            check = (argv[0], "-C", *argv[2:])
        else:
            check = (argv[0], "-C", argv[2], *argv[4:])
        commands.run(check)


def _setup_network(commands: HostCommands, plan: Mapping[str, Any]) -> None:
    for argv in contract.expected_network_setup_argv(plan):
        commands.run(argv)
    _verify_network(commands, plan)


def _remove_container(commands: HostCommands, plan: Mapping[str, Any]) -> None:
    name = contract.runtime_names(plan)["container"]
    commands.run(
        (contract.DOCKER_PATH, "container", "rm", "--force", name),
        expected=(0, 1),
    )


def _cleanup_runtime(commands: HostCommands, plan: Mapping[str, Any]) -> None:
    """Try every exact cleanup target, then prove the complete residue set is empty."""

    cleanup_commands = (
        (
            contract.DOCKER_PATH,
            "container",
            "rm",
            "--force",
            contract.runtime_names(plan)["container"],
        ),
        *contract.expected_network_cleanup_argv(plan),
    )
    for argv in cleanup_commands:
        try:
            commands.run(argv, expected=(0, 1))
        except RuntimeErrorClosed:
            # A later read-only inventory is authoritative.  Continuing here
            # avoids leaving later resources behind after one cleanup error.
            pass
    try:
        _ensure_no_runtime_residue(commands, plan)
    except RuntimeErrorClosed:
        _fail("runtime_cleanup_not_verified")


def _verify_image(commands: HostCommands, plan: Mapping[str, Any]) -> None:
    result = commands.run((contract.DOCKER_PATH, "image", "inspect", contract.IMAGE_REFERENCE))
    try:
        value = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("image_inspect_json_rejected")
    contract.validate_image_inspect(plan, value)


def _host_gpu_inventory(commands: HostCommands, plan: Mapping[str, Any]) -> list[str]:
    result = commands.run(
        (
            contract.NVIDIA_SMI_PATH,
            "--query-gpu=uuid,name",
            "--format=csv,noheader",
        )
    )
    return contract.validate_host_gpu_inventory(plan, result.stdout)


def _write_secret_to_runner(
    process: subprocess.Popen[bytes], secret: bytearray, authority_deadline: float
) -> None:
    _require(process.stdin is not None, "runner_stdin_unavailable")
    stdin = process.stdin
    assert stdin is not None
    descriptor = stdin.fileno()
    os.set_blocking(descriptor, False)
    view = memoryview(secret)
    written = 0
    deadline = min(
        time.monotonic() + contract.FD_READ_SECONDS,
        authority_deadline,
    )
    try:
        while written < len(view):
            try:
                count = os.write(descriptor, view[written:])
            except BlockingIOError:
                _require(time.monotonic() < deadline, "runner_stdin_write_timeout")
                _require(process.poll() is None, "runner_exited_before_jit_transport")
                time.sleep(0.01)
                continue
            _require(count > 0, "runner_stdin_write_failed")
            written += count
            _require(time.monotonic() < deadline, "runner_stdin_write_timeout")
        while True:
            try:
                count = os.write(descriptor, b"\n")
                break
            except BlockingIOError:
                _require(time.monotonic() < deadline, "runner_stdin_write_timeout")
                _require(process.poll() is None, "runner_exited_before_jit_transport")
                time.sleep(0.01)
        _require(count == 1, "runner_stdin_terminator_failed")
        _require(time.monotonic() < deadline, "runner_stdin_write_timeout")
    except (BrokenPipeError, OSError):
        _fail("runner_stdin_transport_failure")
    finally:
        view.release()
        stdin.close()


def _run_container(
    cleanup_commands: HostCommands,
    plan: Mapping[str, Any],
    jit_config: bytearray,
    authority_deadline: float,
) -> tuple[int, str, str, str]:
    argv = contract.render_docker_run_argv(plan)
    cleanup_runner = DeadlineCommands(
        cleanup_commands,
        authority_deadline + contract.CLEANUP_GRACE_SECONDS,
        "cleanup_grace_exceeded",
    )
    started_at = _timestamp()
    hard_deadline = min(
        time.monotonic() + contract.HARD_WALL_SECONDS,
        authority_deadline,
    )
    _require(time.monotonic() < authority_deadline, "authority_expired_before_launch")
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=True,
            close_fds=True,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except OSError:
        _fail("runner_process_start_failed")
    try:
        _write_secret_to_runner(process, jit_config, authority_deadline)
        jit_config_sent_at = _timestamp()
        _zeroize(jit_config)
        try:
            remaining = max(0.001, hard_deadline - time.monotonic())
            exit_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            authority_expired = time.monotonic() >= authority_deadline
            if authority_expired:
                cleanup_runner.run(
                    (
                        contract.DOCKER_PATH,
                        "container",
                        "kill",
                        contract.runtime_names(plan)["container"],
                    ),
                    expected=(0, 1),
                    timeout=10,
                )
            else:
                cleanup_runner.run(
                    (
                        contract.DOCKER_PATH,
                        "container",
                        "stop",
                        "--time=10",
                        contract.runtime_names(plan)["container"],
                    ),
                    expected=(0, 1),
                    timeout=20,
                )
            _remove_container(cleanup_runner, plan)
            try:
                getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            if authority_expired:
                _fail("authority_expired_during_runner")
            _fail("runner_hard_wall_exceeded")
    except BaseException:
        try:
            _remove_container(cleanup_runner, plan)
        except RuntimeErrorClosed:
            pass
        if process.poll() is None:
            try:
                getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        raise
    finally:
        _zeroize(jit_config)
    stopped_at = _timestamp()
    _require(time.monotonic() <= authority_deadline, "authority_expired_during_runner")
    _require(exit_code == 0, "runner_container_exit_nonzero")
    return exit_code, started_at, jit_config_sent_at, stopped_at


def _parse_cli(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe-host", allow_abbrev=False)
    subparsers.add_parser("run", allow_abbrev=False)
    return parser.parse_args(list(argv))


def probe_image(commands: HostCommands | None = None) -> dict[str, Any]:
    runner = commands or HostCommands()
    _verify_host_binary(contract.DOCKER_PATH)
    pull = runner.run(contract.render_image_pull_argv(), timeout=600)
    inspect = runner.run((contract.DOCKER_PATH, "image", "inspect", contract.IMAGE_REFERENCE))
    try:
        inspected = json.loads(inspect.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("image_inspect_json_rejected")
    contract.validate_image_inspect({}, inspected)
    try:
        probe = runner.run(contract.render_image_probe_argv(), timeout=60, maximum_output=512)
    finally:
        runner.run(
            (contract.DOCKER_PATH, "container", "rm", "--force", contract.PROBE_CONTAINER_NAME),
            expected=(0, 1),
        )
    result = contract.validate_image_probe_output(probe.stdout)
    return {
        "schema_version": 1,
        "kind": "explainiverse-runner-image-probe",
        "observed_at": _timestamp(),
        "image_reference": contract.IMAGE_REFERENCE,
        "manifest_digest": contract.IMAGE_MANIFEST_DIGEST,
        "config_digest": contract.IMAGE_CONFIG_DIGEST,
        "platform": contract.IMAGE_PLATFORM,
        "pull_output_sha256": hashlib.sha256(pull.stdout + pull.stderr).hexdigest(),
        "inspect_response_sha256": hashlib.sha256(inspect.stdout).hexdigest(),
        "probe": result,
        "network_contact_during_probe_container": False,
        "registry_contacted_for_digest_pull": True,
        "github_api_contacted": False,
    }


def probe_gpu_injection(
    gpu_uuids: Sequence[str], commands: HostCommands | None = None
) -> dict[str, Any]:
    runner = commands or HostCommands()
    try:
        result = runner.run(
            contract.render_gpu_injection_probe_argv(gpu_uuids),
            timeout=60,
            maximum_output=512,
        )
    finally:
        runner.run(
            (contract.DOCKER_PATH, "container", "rm", "--force", contract.PROBE_CONTAINER_NAME),
            expected=(0, 1),
        )
    return contract.validate_gpu_injection_probe_output(result.stdout)


def probe_host(commands: HostCommands | None = None) -> dict[str, Any]:
    """Perform the fixed, credential-free host and image pre-JIT preflight."""

    runner = commands or HostCommands()
    require_probe_stdin_eof()
    verify_no_sensitive_environment()
    harden_secret_process()
    _require(getattr(os, "geteuid")() == 0, "root_control_process_required")
    _verify_host_binary(contract.PYTHON_PATH)
    _verify_host_binary(contract.CLOUD_INIT_PATH)
    cloud_init = runner.run(
        (contract.CLOUD_INIT_PATH, "status", "--wait"),
        timeout=600,
        maximum_output=32_768,
    )
    try:
        cloud_lines = cloud_init.stdout.decode("ascii").splitlines()
    except UnicodeDecodeError:
        _fail("cloud_init_status_encoding_rejected")
    _require(
        len(cloud_lines) >= 1
        and cloud_lines[0] == "status: done"
        and not any("degraded" in line.lower() or "error" in line.lower() for line in cloud_lines),
        "cloud_init_not_ready",
    )
    verify_host_posture()
    with ExclusiveRuntimeLock():
        _ensure_no_global_runtime_residue(runner)
        inventory_result = runner.run(
            (
                contract.NVIDIA_SMI_PATH,
                "--query-gpu=uuid,name",
                "--format=csv,noheader",
            )
        )
        host_gpu_uuids, host_gpu_products = contract.parse_host_gpu_inventory(
            inventory_result.stdout
        )
        image = probe_image(runner)
        gpu_injection = probe_gpu_injection(host_gpu_uuids, runner)
        _ensure_no_global_runtime_residue(runner)
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "kind": contract.HOST_PREFLIGHT_KIND,
        "observed_at": _timestamp(),
        "cloud_init_status": "done",
        "cloud_init_output_sha256": hashlib.sha256(
            cloud_init.stdout + cloud_init.stderr
        ).hexdigest(),
        "effective_uid": 0,
        "root_owned_nonwritable_runtime_bundle": True,
        "runtime_bundle_sha256": runtime_bundle_sha256(),
        "host_physical_gpu_count": len(host_gpu_uuids),
        "host_physical_gpu_uuids": host_gpu_uuids,
        "host_physical_gpu_products": host_gpu_products,
        "gpu_inventory_output_sha256": hashlib.sha256(inventory_result.stdout).hexdigest(),
        "image": image,
        "gpu_injection": gpu_injection,
        "local_runtime_residue_absent": True,
        "jit_config_received": False,
        "github_api_credential_received": False,
        "github_api_contacted": False,
        "accepted_actions_evidence": False,
    }


def execute(args: argparse.Namespace, commands: HostCommands | None = None) -> dict[str, Any]:
    runner = commands or HostCommands()
    execution_runner: HostCommands = runner
    plan_bytes: bytearray | None = None
    jit_config: bytearray | None = None
    plan: dict[str, Any] | None = None
    network_attempted = False
    authority_deadline: float | None = None
    termination_guard: TerminationGuard | None = None
    try:
        verify_no_sensitive_environment()
        plan_bytes = _read_anonymous_fd(4, maximum=MAX_PLAN_BYTES, context="runner_plan")
        plan = contract.parse_plan_document(bytes(plan_bytes))
        authority_deadline = _authority_deadline(plan)
        execution_runner = DeadlineCommands(
            runner, authority_deadline, "authority_expired_during_setup"
        )
        _zeroize(plan_bytes)
        harden_secret_process()
        verify_host_posture()
        _require(
            runtime_bundle_sha256() == plan["runtime_bundle_sha256"],
            "runtime_bundle_digest_mismatch",
        )
        _require(
            time.monotonic() < authority_deadline,
            "authority_expired_during_setup",
        )
        termination_guard = TerminationGuard()
        termination_guard.__enter__()
        jit_config = _read_anonymous_fd(
            3,
            maximum=MAX_JIT_BYTES,
            context="jit_config",
            absolute_deadline=authority_deadline,
        )
        validate_jit_config(jit_config, plan["job"]["jit_config_sha256"])
        _require(
            time.monotonic() < authority_deadline,
            "authority_expired_during_setup",
        )
        with ExclusiveRuntimeLock():
            _ensure_no_runtime_residue(execution_runner, plan)
            _verify_image(execution_runner, plan)
            host_gpu_uuids = _host_gpu_inventory(execution_runner, plan)
            network_attempted = True
            _setup_network(execution_runner, plan)
            try:
                _require(
                    time.monotonic() < authority_deadline,
                    "authority_expired_before_launch",
                )
                exit_code, started_at, sent_at, stopped_at = _run_container(
                    runner, plan, jit_config, authority_deadline
                )
            finally:
                cleanup_runner = DeadlineCommands(
                    runner,
                    authority_deadline + contract.CLEANUP_GRACE_SECONDS,
                    "cleanup_grace_exceeded",
                )
                _cleanup_runtime(cleanup_runner, plan)
                network_attempted = False
            cleanup_at = _timestamp()
            return contract.build_runtime_receipt(
                plan,
                host_gpu_uuids=host_gpu_uuids,
                started_at=started_at,
                jit_config_sent_at=sent_at,
                stopped_at=stopped_at,
                cleanup_verified_at=cleanup_at,
                runner_exit_code=exit_code,
            )
    finally:
        try:
            if network_attempted and plan is not None:
                cleanup_base = authority_deadline or time.monotonic()
                cleanup_runner = DeadlineCommands(
                    runner,
                    cleanup_base + contract.CLEANUP_GRACE_SECONDS,
                    "cleanup_grace_exceeded",
                )
                _cleanup_runtime(cleanup_runner, plan)
        finally:
            _zeroize(plan_bytes)
            _zeroize(jit_config)
            if termination_guard is not None:
                termination_guard.__exit__(None, None, None)


def _write_canonical_fd(fd: int, value: Mapping[str, Any]) -> None:
    payload = contract.canonical_json(value)
    written = 0
    while written < len(payload):
        count = os.write(fd, payload[written:])
        _require(count > 0, "output_write_failed")
        written += count


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_cli(sys.argv[1:] if argv is None else argv)
        if args.command == "probe-host":
            receipt = probe_host()
            _write_canonical_fd(1, receipt)
        else:
            receipt = execute(args)
            _write_canonical_fd(1, receipt)
        return 0
    except (RuntimeErrorClosed, contract.ContractError) as error:
        code = str(error)
        if re.fullmatch(r"[a-z0-9_]+", code) is None:
            code = "runtime_contract_failure"
        os.write(2, f"release_gpu_jit_runtime:{code}\n".encode("ascii"))
        return 1
    except BaseException:
        os.write(2, b"release_gpu_jit_runtime:unexpected_failure\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
