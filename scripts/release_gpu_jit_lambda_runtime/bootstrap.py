"""Split one framed SSH stdin stream into the executor's anonymous FDs 3 and 4.

The remote command is fixed and carries no plan or secret-derived argv value.
This bootstrap validates the complete frame before forking the executor, then
writes the JIT configuration to child FD 3 and the public canonical plan to
child FD 4 through anonymous pipes.  No payload is written to disk.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import signal
import socket
import stat
import struct
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

try:
    fcntl_module: Any = importlib.import_module("fcntl")
except ImportError:  # pragma: no cover - Linux-only production dependency
    fcntl_module = None

try:
    from . import executor
    from . import runtime_contract as contract
except ImportError:  # pragma: no cover - fixed direct remote invocation
    import executor  # type: ignore[import-not-found,no-redef]
    import runtime_contract as contract  # type: ignore[import-not-found,no-redef]

MAGIC = b"EXJIT01\n"
FRAME_VERSION = 1
FRAME_FLAGS = 0
HEADER = struct.Struct(">8sHHII32s32s")
MAX_FRAME_PART_BYTES = 1_048_576
PIPE_TARGET_RE = re.compile(r"pipe:\[[0-9]+\]\Z")


class BootstrapError(RuntimeError):
    """Stable framing/transport error without payload values."""


def _fail(code: str) -> NoReturn:
    raise BootstrapError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _zeroize(value: bytearray | None) -> None:
    if value is not None:
        value[:] = b"\x00" * len(value)


def _stdin_is_anonymous() -> bool:
    try:
        target = os.readlink("/proc/self/fd/0")
        status = os.fstat(0)
    except OSError:
        return False
    if stat.S_ISFIFO(status.st_mode):
        return PIPE_TARGET_RE.fullmatch(target) is not None
    if not stat.S_ISSOCK(status.st_mode):
        return False
    duplicate = os.dup(0)
    candidate: socket.socket | None = None
    try:
        candidate = socket.socket(fileno=duplicate)
        so_domain = getattr(socket, "SO_DOMAIN", 39)
        return candidate.getsockopt(socket.SOL_SOCKET, so_domain) == getattr(
            socket, "AF_UNIX"
        ) and candidate.getsockname() in {"", b""}
    except OSError:
        return False
    finally:
        if candidate is not None:
            candidate.close()
        else:
            os.close(duplicate)


def _read_exact(length: int, *, context: str, deadline: float) -> bytearray:
    _require(0 <= length <= MAX_FRAME_PART_BYTES, f"{context}_length_rejected")
    result = bytearray(length)
    cursor = 0
    try:
        while cursor < length:
            view = memoryview(result)[cursor:]
            try:
                count = getattr(os, "readv")(0, [view])
            except BlockingIOError:
                view.release()
                _require(time.monotonic() < deadline, f"{context}_read_timeout")
                time.sleep(0.01)
                continue
            view.release()
            _require(count > 0, f"{context}_truncated")
            cursor += count
            _require(time.monotonic() < deadline, f"{context}_read_timeout")
        return result
    except BaseException:
        _zeroize(result)
        raise


def _require_eof(*, deadline: float) -> None:
    probe = bytearray(1)
    try:
        while True:
            view = memoryview(probe)
            try:
                count = getattr(os, "readv")(0, [view])
            except BlockingIOError:
                view.release()
                _require(time.monotonic() < deadline, "frame_eof_timeout")
                time.sleep(0.01)
                continue
            view.release()
            _require(count == 0, "frame_trailing_bytes_rejected")
            return
    finally:
        _zeroize(probe)


def read_frame() -> tuple[bytearray, bytearray]:
    """Read exact header/parts/EOF from anonymous stdin and verify both hashes."""

    _require(sys.platform.startswith("linux"), "linux_host_required")
    _require(_stdin_is_anonymous(), "stdin_not_anonymous")
    os.set_blocking(0, False)
    deadline = time.monotonic() + contract.FD_READ_SECONDS
    header = _read_exact(HEADER.size, context="frame_header", deadline=deadline)
    try:
        magic, version, flags, plan_length, jit_length, plan_sha, jit_sha = HEADER.unpack(header)
    except struct.error:
        _zeroize(header)
        _fail("frame_header_rejected")
    _zeroize(header)
    _require(magic == MAGIC, "frame_magic_rejected")
    _require(version == FRAME_VERSION, "frame_version_rejected")
    _require(flags == FRAME_FLAGS, "frame_flags_rejected")
    _require(0 < plan_length <= MAX_FRAME_PART_BYTES, "frame_plan_length_rejected")
    _require(100 <= jit_length <= MAX_FRAME_PART_BYTES, "frame_jit_length_rejected")
    plan: bytearray | None = None
    jit: bytearray | None = None
    try:
        plan = _read_exact(plan_length, context="frame_plan", deadline=deadline)
        jit = _read_exact(jit_length, context="frame_jit", deadline=deadline)
        _require_eof(deadline=deadline)
        _require(hashlib.sha256(plan).digest() == plan_sha, "frame_plan_digest_mismatch")
        _require(hashlib.sha256(jit).digest() == jit_sha, "frame_jit_digest_mismatch")
        return plan, jit
    except BaseException:
        _zeroize(plan)
        _zeroize(jit)
        raise


def frame_header(plan: bytes, jit_config: bytes | bytearray | memoryview) -> bytes:
    """Return the public fixed header; callers stream header, plan, then JIT."""

    _require(type(plan) is bytes and 0 < len(plan) <= MAX_FRAME_PART_BYTES, "plan_bytes_rejected")
    _require(
        type(jit_config) in {bytes, bytearray, memoryview}
        and 100 <= len(jit_config) <= MAX_FRAME_PART_BYTES,
        "jit_bytes_rejected",
    )
    return HEADER.pack(
        MAGIC,
        FRAME_VERSION,
        FRAME_FLAGS,
        len(plan),
        len(jit_config),
        hashlib.sha256(plan).digest(),
        hashlib.sha256(jit_config).digest(),
    )


def _move_fd_high(fd: int) -> int:
    _require(fcntl_module is not None, "fcntl_unavailable")
    moved = fcntl_module.fcntl(fd, fcntl_module.F_DUPFD_CLOEXEC, 10)
    os.close(fd)
    return moved


def _pipe_high() -> tuple[int, int]:
    read_fd, write_fd = getattr(os, "pipe2")(getattr(os, "O_CLOEXEC"))
    return _move_fd_high(read_fd), _move_fd_high(write_fd)


def _write_pipe(
    fd: int,
    value: bytearray,
    child_pid: int,
    *,
    context: str,
    deadline: float,
) -> None:
    os.set_blocking(fd, False)
    view = memoryview(value)
    cursor = 0
    try:
        while cursor < len(view):
            try:
                count = os.write(fd, view[cursor:])
            except BlockingIOError:
                waited_pid, _ = os.waitpid(child_pid, getattr(os, "WNOHANG"))
                _require(waited_pid == 0, f"{context}_child_exited")
                _require(time.monotonic() < deadline, f"{context}_write_timeout")
                time.sleep(0.01)
                continue
            _require(count > 0, f"{context}_write_failed")
            cursor += count
            _require(time.monotonic() < deadline, f"{context}_write_timeout")
    except (BrokenPipeError, OSError):
        _fail(f"{context}_transport_failed")
    finally:
        view.release()
        os.close(fd)


def _child_exec(jit_read: int, plan_read: int, jit_write: int, plan_write: int) -> NoReturn:
    os.close(jit_write)
    os.close(plan_write)
    os.dup2(jit_read, 3, inheritable=True)
    os.dup2(plan_read, 4, inheritable=True)
    os.close(jit_read)
    os.close(plan_read)
    getattr(os, "setsid")()
    executor_path = Path(__file__).resolve().with_name("executor.py")
    environment = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}
    try:
        os.execve(
            contract.PYTHON_PATH,
            (contract.PYTHON_PATH, "-B", str(executor_path), "run"),
            {**environment, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except OSError:
        os._exit(126)


def _signal_child(child_pid: int, signum: int) -> None:
    try:
        getattr(os, "killpg")(child_pid, signum)
        return
    except ProcessLookupError:
        pass
    try:
        os.kill(child_pid, signum)
    except ProcessLookupError:
        pass


def _terminate_child(child_pid: int) -> None:
    _signal_child(child_pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            waited_pid, _ = os.waitpid(child_pid, getattr(os, "WNOHANG"))
        except ChildProcessError:
            return
        if waited_pid == child_pid:
            return
        time.sleep(0.1)
    _signal_child(child_pid, getattr(signal, "SIGKILL"))
    try:
        os.waitpid(child_pid, 0)
    except ChildProcessError:
        pass


def _kill_child_now(child_pid: int) -> None:
    _signal_child(child_pid, getattr(signal, "SIGKILL"))
    try:
        os.waitpid(child_pid, 0)
    except ChildProcessError:
        pass


def run_child(plan: bytearray, jit_config: bytearray, authority_deadline: float) -> int:
    jit_read, jit_write = _pipe_high()
    plan_read, plan_write = _pipe_high()
    try:
        child_pid = getattr(os, "fork")()
    except OSError:
        for fd in (jit_read, jit_write, plan_read, plan_write):
            os.close(fd)
        _fail("executor_fork_failed")
    if child_pid == 0:
        _child_exec(jit_read, plan_read, jit_write, plan_write)
    os.close(jit_read)
    os.close(plan_read)
    child_deadline = min(
        time.monotonic() + contract.HARD_WALL_SECONDS + contract.CLEANUP_GRACE_SECONDS,
        authority_deadline + contract.CLEANUP_GRACE_SECONDS,
    )
    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, frame: object) -> NoReturn:
        del frame
        _signal_child(child_pid, signum)
        _fail("bootstrap_termination_signal")

    for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
        value = getattr(signal, name, None)
        if value is not None:
            previous_handlers[value] = signal.getsignal(value)
            signal.signal(value, forward_signal)
    try:
        _write_pipe(
            plan_write,
            plan,
            child_pid,
            context="plan_pipe",
            deadline=authority_deadline,
        )
        _write_pipe(
            jit_write,
            jit_config,
            child_pid,
            context="jit_pipe",
            deadline=authority_deadline,
        )
        _zeroize(jit_config)
        while True:
            waited_pid, status = os.waitpid(child_pid, getattr(os, "WNOHANG"))
            if waited_pid == child_pid:
                if getattr(os, "WIFEXITED")(status):
                    return getattr(os, "WEXITSTATUS")(status)
                return 128 + getattr(os, "WTERMSIG")(status)
            if time.monotonic() >= child_deadline:
                _kill_child_now(child_pid)
                _fail("executor_external_watchdog_expired")
            time.sleep(0.1)
    except BaseException:
        _terminate_child(child_pid)
        raise
    finally:
        _zeroize(jit_config)
        for fd in (jit_write, plan_write):
            try:
                os.close(fd)
            except OSError:
                pass
        for value, previous in previous_handlers.items():
            signal.signal(value, previous)


def main() -> int:
    plan: bytearray | None = None
    jit_config: bytearray | None = None
    try:
        _require(len(sys.argv) == 1, "bootstrap_argv_rejected")
        executor.verify_no_sensitive_environment()
        executor.harden_secret_process()
        executor.verify_host_posture()
        plan, jit_config = read_frame()
        normalized = contract.parse_plan_document(bytes(plan))
        authority_expires = contract._timestamp(
            normalized["authority_window"]["expires_at"], "authority_expires_at"
        )
        remaining = (authority_expires - executor._utc_now()).total_seconds()
        _require(remaining > 0, "authority_window_expired")
        authority_deadline = time.monotonic() + remaining
        _require(
            executor.runtime_bundle_sha256() == normalized["runtime_bundle_sha256"],
            "runtime_bundle_digest_mismatch",
        )
        executor.validate_jit_config(jit_config, normalized["job"]["jit_config_sha256"])
        _require(time.monotonic() < authority_deadline, "authority_expired_before_executor")
        return run_child(plan, jit_config, authority_deadline)
    except (BootstrapError, executor.RuntimeErrorClosed, contract.ContractError) as error:
        code = str(error)
        if re.fullmatch(r"[a-z0-9_]+", code) is None:
            code = "bootstrap_contract_failure"
        os.write(2, f"release_gpu_jit_bootstrap:{code}\n".encode("ascii"))
        return 1
    except BaseException:
        os.write(2, b"release_gpu_jit_bootstrap:unexpected_failure\n")
        return 1
    finally:
        _zeroize(plan)
        _zeroize(jit_config)


if __name__ == "__main__":
    raise SystemExit(main())
