"""Native Windows raw-HANDLE launcher for the secure operator entrypoint.

The launcher is the only supported Windows bridge for Lambda credentials.  It
passes no secret in argv, environment, or a file: two anonymous-pipe read
HANDLEs are allowlisted into the child, which converts them to CRT descriptors
before importing the controller.
"""

from __future__ import annotations

import argparse
import builtins
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence, cast

MAX_SECRET_BYTES = 4096
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PARENT_RECEIPT_KIND = "explainiverse-windows-launcher-parent-boundary"
PRELOADER_MARKER_NAME = "_EXPLAINIVERSE_OPERATOR_PRELOADER_RECEIPT"
PRELOADER_SHIM = (
    r"""import builtins,hashlib,os,stat,sys
s=sys.argv[1]; p=os.path.abspath(sys.argv[2]); e=sys.argv[3]
if not (len(s)==64 and all(c in "0123456789abcdef" for c in s) and os.path.isabs(sys.argv[2]) and p==sys.argv[2] and len(e)==64 and all(c in "0123456789abcdef" for c in e)): raise SystemExit("operator_preloader_shim_arguments_rejected")
f=os.open(p,os.O_RDONLY|getattr(os,"O_BINARY",0)|getattr(os,"O_NOFOLLOW",0))
try:
 b=os.fstat(f); q=os.lstat(p)
 if not (stat.S_ISREG(b.st_mode) and b.st_nlink==q.st_nlink==1 and (b.st_dev,b.st_ino)==(q.st_dev,q.st_ino) and 0<b.st_size<=4194304): raise SystemExit("operator_preloader_shim_identity_rejected")
 r=b""; n=b.st_size
 while n:
  x=os.read(f,min(65536,n))
  if not x: raise SystemExit("operator_preloader_shim_short_read")
  r+=x; n-=len(x)
 if os.read(f,1): raise SystemExit("operator_preloader_shim_grew")
 a=os.fstat(f)
 if (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns)!=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns): raise SystemExit("operator_preloader_shim_changed")
finally: os.close(f)
if hashlib.sha256(r).hexdigest()!=e: raise SystemExit("operator_preloader_shim_digest_rejected")
builtins._EXPLAINIVERSE_OPERATOR_SHIM_RECEIPT={"schema_version":1,"kind":"explainiverse-operator-preloader-shim","shim_sha256":s,"preloader_path":p,"preloader_bytes":len(r),"preloader_sha256":e,"stable_descriptor_read":True,"compiled_verified_bytes_without_reopen":True}
sys.argv=[p,*sys.argv[4:]]
g={"__name__":"__main__","__file__":p,"__builtins__":builtins.__dict__}
exec(compile(r,p,"exec",dont_inherit=True),g,g)"""
    + "\n"
)


class LauncherError(RuntimeError):
    """Stable launcher rejection code."""


def _fail(code: str) -> NoReturn:
    raise LauncherError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _forbidden_environment_name(name: str) -> bool:
    normalized = name.upper()
    exact = {
        "ALL_PROXY",
        "GH_ENTERPRISE_TOKEN",
        "GH_HOST",
        "GH_REPO",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LAMBDA_API_KEY",
        "NO_PROXY",
        "PYTHONHOME",
        "PYTHONPATH",
        "REQUESTS_CA_BUNDLE",
        "SSH_AGENT_PID",
        "SSH_AUTH_SOCK",
        "SSL_CERT_FILE",
    }
    prefixes = ("ANTHROPIC_", "AWS_", "AZURE_", "GCP_", "GOOGLE_", "OPENAI_")
    fragments = (
        "API_KEY",
        "CREDENTIAL",
        "JITCONFIG",
        "PASSWORD",
        "PASSWD",
        "PRIVATE_KEY",
        "SECRET",
        "TOKEN",
    )
    return (
        normalized in exact
        or normalized.startswith(prefixes)
        or any(fragment in normalized for fragment in fragments)
    )


def _child_environment() -> dict[str, str]:
    allowed = {
        "APPDATA",
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "NO_COLOR",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
    return {
        name: value
        for name, value in os.environ.items()
        if name.upper() in allowed and not _forbidden_environment_name(name)
    }


def _consume_preloader_receipt(repository_root: Path) -> dict[str, Any]:
    receipt_value = getattr(builtins, PRELOADER_MARKER_NAME, None)
    _require(type(receipt_value) is dict, "launcher_preloader_receipt_missing")
    value = cast(dict[str, Any], receipt_value)
    material = dict(value)
    evidence_sha256 = material.pop("evidence_sha256", None)
    _require(
        type(evidence_sha256) is str
        and hashlib.sha256(_canonical(material)).hexdigest() == evidence_sha256
        and value.get("schema_version") == 1
        and value.get("kind") == "explainiverse-operator-isolated-preloader"
        and value.get("isolated") is True
        and value.get("safe_path") is True
        and value.get("site_disabled") is True
        and value.get("bytecode_disabled") is True
        and value.get("repository_absent_from_sys_path") is True
        and value.get("project_imports_from_captured_bytes") is True
        and type(value.get("source")) is dict
        and value["source"].get("repository_root") == str(repository_root)
        and value["source"].get("tracked_and_untracked_clean") is True
        and type(value.get("bootstrap")) is dict
        and value["bootstrap"].get("python_archive_sha256")
        == "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf",
        "launcher_preloader_receipt_rejected",
    )
    delattr(builtins, PRELOADER_MARKER_NAME)
    return dict(value)


def _secure_parent_launch(
    repository_root: Path, preloader_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and record the parent interpreter before reading any secret."""

    _require(
        sys.flags.isolated == 1
        and sys.flags.safe_path
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and sys.flags.no_site == 1
        and sys.dont_write_bytecode,
        "launcher_requires_pinned_python_I_S_B",
    )
    try:
        working = Path.cwd().resolve(strict=True)
    except OSError:
        _fail("launcher_import_root_unavailable")
    _require(
        preloader_receipt.get("working_directory") == str(working)
        and preloader_receipt.get("working_directory_is_python_install_receipt_directory") is True
        and working != repository_root
        and repository_root not in working.parents,
        "launcher_import_root_mismatch",
    )
    _require(
        not any(_forbidden_environment_name(name) for name in os.environ),
        "launcher_environment_not_scrubbed",
    )
    normalized_path: list[str] = []
    for item in sys.path:
        try:
            resolved = Path(item or os.curdir).resolve(strict=False)
        except OSError:
            _fail("launcher_sys_path_rejected")
        normalized_path.append(str(resolved))
    _require(
        all(
            Path(item) != repository_root and repository_root not in Path(item).parents
            for item in normalized_path
        ),
        "launcher_repository_present_in_sys_path",
    )
    source = preloader_receipt["source"]
    bootstrap = preloader_receipt["bootstrap"]
    preloader_binding = {
        "evidence_sha256": preloader_receipt["evidence_sha256"],
        "head_sha": source["head_sha"],
        "preloader_path": source["preloader_path"],
        "preloader_sha256": source["preloader_sha256"],
        "manifest_sha256": bootstrap["manifest_sha256"],
        "python_manifest_sha256": bootstrap["python_manifest_sha256"],
        "python_archive_sha256": bootstrap["python_archive_sha256"],
    }
    material = {
        "schema_version": 1,
        "kind": PARENT_RECEIPT_KIND,
        "isolated": True,
        "safe_path": True,
        "ignore_environment": True,
        "no_user_site": True,
        "no_site": True,
        "dont_write_bytecode": True,
        "invocation": "pinned-python -I -S -B -c <byte-sealing-shim>",
        "working_directory": str(working),
        "repository_absent_from_sys_path": True,
        "sys_path_sha256": hashlib.sha256("\n".join(normalized_path).encode("utf-8")).hexdigest(),
        "site_processing_disabled": True,
        "preloader_binding": preloader_binding,
        "environment": preloader_receipt["environment"],
        "parent_declares_secret_read_after_boundary_validation": True,
        "parent_provenance_authenticated": False,
        "child_must_revalidate_all_security_boundaries": True,
    }
    return {**material, "evidence_sha256": hashlib.sha256(_canonical(material)).hexdigest()}


class _PipeInput:
    def __init__(self, descriptor: int) -> None:
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_file_type = kernel32.GetFileType
        get_file_type.argtypes = [wintypes.HANDLE]
        get_file_type.restype = wintypes.DWORD
        handle = msvcrt.get_osfhandle(descriptor)
        _require(get_file_type(handle) == 3, "launcher_stdin_not_pipe")
        self._descriptor = descriptor

    def read_line(self, *, maximum: int, context: str) -> bytearray:
        result = bytearray()
        while len(result) <= maximum:
            chunk = os.read(self._descriptor, 1)
            _require(bool(chunk), f"{context}_unexpected_eof")
            if chunk == b"\n":
                return result
            _require(chunk != b"\r", f"{context}_carriage_return_rejected")
            result.extend(chunk)
        for index in range(len(result)):
            result[index] = 0
        _fail(f"{context}_too_large")


class _WindowsConsoleApi:
    """Typed WinAPI surface used by the no-echo console reader."""

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.get_std_handle = kernel32.GetStdHandle
        self.get_std_handle.argtypes = [wintypes.DWORD]
        self.get_std_handle.restype = wintypes.HANDLE
        self.get_console_mode = kernel32.GetConsoleMode
        self.get_console_mode.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
        self.get_console_mode.restype = wintypes.BOOL
        self.set_console_mode = kernel32.SetConsoleMode
        self.set_console_mode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.set_console_mode.restype = wintypes.BOOL
        self.read_console_w = kernel32.ReadConsoleW
        self.read_console_w.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPDWORD,
            wintypes.LPVOID,
        ]
        self.read_console_w.restype = wintypes.BOOL


def _console_secret(*, api: Any | None = None, prompt_stream: Any = None) -> bytearray:
    console = api if api is not None else _WindowsConsoleApi()
    prompt = prompt_stream if prompt_stream is not None else sys.stderr
    handle = console.get_std_handle(wintypes.DWORD(-10 & 0xFFFFFFFF))
    invalid_handle = ctypes.c_void_p(-1).value
    _require(handle not in {None, 0, invalid_handle}, "launcher_console_unavailable")
    mode = wintypes.DWORD()
    _require(console.get_console_mode(handle, ctypes.byref(mode)), "launcher_console_unavailable")
    original_mode = mode.value
    _require(
        console.set_console_mode(handle, wintypes.DWORD(original_mode & ~0x0004)),
        "launcher_console_noecho_failed",
    )
    prompt.write("Lambda API key (not echoed): ")
    prompt.flush()
    buffer = (ctypes.c_wchar * (MAX_SECRET_BYTES + 3))()
    read = wintypes.DWORD()
    value: bytearray | None = None
    try:
        ok = console.read_console_w(
            handle,
            ctypes.byref(buffer),
            wintypes.DWORD(MAX_SECRET_BYTES + 2),
            ctypes.byref(read),
            None,
        )
        _require(bool(ok), "launcher_console_read_failed")
        count = int(read.value)
        _require(count <= MAX_SECRET_BYTES + 2, "launcher_console_read_count_rejected")
        while count and ord(buffer[count - 1]) in {10, 13}:
            count -= 1
        _require(
            all(0x20 < ord(buffer[index]) < 0x7F for index in range(count)),
            "launcher_secret_character_rejected",
        )
        value = bytearray(ord(buffer[index]) for index in range(count))
    finally:
        ctypes.memset(ctypes.byref(buffer), 0, ctypes.sizeof(buffer))
        restored = console.set_console_mode(handle, wintypes.DWORD(original_mode))
        prompt.write("\n")
        prompt.flush()
        _require(bool(restored), "launcher_console_mode_restore_failed")
    assert value is not None
    return value


def _console_confirmation(expected: str) -> bytearray:
    try:
        with open("CONIN$", "r", encoding="ascii", errors="strict") as console:
            sys.stderr.write(f"Retype plan SHA {expected}: ")
            sys.stderr.flush()
            value = console.readline(66)
    except (OSError, UnicodeError):
        _fail("launcher_confirmation_console_failed")
    return bytearray(value.rstrip("\n").encode("ascii"))


def _validate_secret(value: bytearray) -> None:
    _require(0 < len(value) <= MAX_SECRET_BYTES, "launcher_secret_size_rejected")
    _require(
        all(0x20 < byte < 0x7F for byte in value),
        "launcher_secret_character_rejected",
    )


def _write_all(descriptor: int, value: bytearray | bytes) -> None:
    view = memoryview(value)
    try:
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            _require(written > 0, "launcher_pipe_short_write")
            offset += written
    finally:
        view.release()


def _zero(value: bytearray | None) -> None:
    if value is None:
        return
    for index in range(len(value)):
        value[index] = 0
    value.clear()


def _argument_value(arguments: Sequence[str], name: str) -> str:
    values: list[str] = []
    for index, item in enumerate(arguments):
        if item == name:
            _require(index + 1 < len(arguments), "launcher_argument_value_missing")
            values.append(arguments[index + 1])
        elif item.startswith(name + "="):
            values.append(item.split("=", 1)[1])
    _require(len(values) == 1, f"launcher_{name.removeprefix('--').replace('-', '_')}_cardinality")
    return values[0]


def _stderr_pump(stream: Any) -> None:
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        sys.stderr.buffer.write(chunk)
        sys.stderr.buffer.flush()


def _read_plan_line(stream: Any) -> tuple[bytes, str]:
    raw = stream.readline(4 * 1024 * 1024 + 1)
    _require(0 < len(raw) <= 4 * 1024 * 1024, "launcher_plan_line_rejected")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("launcher_plan_json_rejected")
    _require(
        type(value) is dict
        and value.get("kind") == "explainiverse-lambda-plan-awaiting-confirmation"
        and type(value.get("plan_sha256")) is str
        and SHA256_RE.fullmatch(value["plan_sha256"]) is not None
        and raw == _canonical(value),
        "launcher_plan_binding_rejected",
    )
    return raw, value["plan_sha256"]


def _relay_remaining(stream: Any) -> None:
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()


def _parse_launcher(arguments: Sequence[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--launcher-secret-source",
        choices=("console", "stdin-pipe"),
        default="console",
    )
    known, child = parser.parse_known_args(arguments)
    forbidden = {
        "--lambda-api-key-fd",
        "--plan-confirmation-fd",
        "--lambda-api-key-handle",
        "--plan-confirmation-handle",
        "--windows-launcher-parent-receipt",
    }
    _require(
        not any(
            item in forbidden or any(item.startswith(option + "=") for option in forbidden)
            for item in child
        ),
        "launcher_child_transport_override_rejected",
    )
    return known.launcher_secret_source, child


def run(arguments: Sequence[str]) -> int:
    _require(os.name == "nt", "windows_launcher_windows_only")
    secret_source, child_arguments = _parse_launcher(arguments)
    action = _argument_value(child_arguments, "--action")
    _require(
        action in {"execute", "resume-abort", "transport-self-test"},
        "launcher_action_rejected",
    )
    repository_root = Path(_argument_value(child_arguments, "--repository-root"))
    _require(repository_root.is_absolute(), "launcher_repository_root_not_absolute")
    repository_root = repository_root.resolve(strict=True)
    _require(
        repository_root.is_dir() and not repository_root.is_symlink(),
        "launcher_repository_root_rejected",
    )
    preloader_receipt = _consume_preloader_receipt(repository_root)
    parent_receipt = _secure_parent_launch(repository_root, preloader_receipt)

    pipe_source = _PipeInput(0) if secret_source == "stdin-pipe" else None
    secret = (
        pipe_source.read_line(maximum=MAX_SECRET_BYTES, context="launcher_secret")
        if pipe_source is not None
        else _console_secret()
    )
    _validate_secret(secret)
    lambda_read: int | None = None
    lambda_write: int | None = None
    confirmation_read: int | None = None
    confirmation_write: int | None = None
    process: subprocess.Popen[bytes] | None = None
    stderr_thread: threading.Thread | None = None
    try:
        lambda_read, lambda_write = os.pipe()
        if action in {"execute", "transport-self-test"}:
            confirmation_read, confirmation_write = os.pipe()
        import msvcrt

        lambda_handle = msvcrt.get_osfhandle(lambda_read)
        os.set_handle_inheritable(lambda_handle, True)
        handles = [lambda_handle]
        if confirmation_read is not None:
            confirmation_handle = msvcrt.get_osfhandle(confirmation_read)
            os.set_handle_inheritable(confirmation_handle, True)
            handles.append(confirmation_handle)
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {"handle_list": handles}
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            PRELOADER_SHIM,
            hashlib.sha256(PRELOADER_SHIM.encode("utf-8")).hexdigest(),
            str(preloader_receipt["source"]["preloader_path"]),
            str(preloader_receipt["source"]["preloader_sha256"]),
            "--operator-target",
            "operator",
            *child_arguments,
            "--windows-launcher-parent-receipt",
            _canonical(parent_receipt).decode("ascii").rstrip("\n"),
            "--lambda-api-key-handle",
            str(lambda_handle),
        ]
        if confirmation_read is not None:
            command.extend(("--plan-confirmation-handle", str(handles[1])))
        process = subprocess.Popen(
            command,
            cwd=str(Path.cwd().resolve(strict=True)),
            env=_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            startupinfo=startup,
            shell=False,
        )
        os.close(lambda_read)
        lambda_read = None
        if confirmation_read is not None:
            os.close(confirmation_read)
            confirmation_read = None
        assert process.stderr is not None
        stderr_thread = threading.Thread(target=_stderr_pump, args=(process.stderr,), daemon=True)
        stderr_thread.start()
        _write_all(lambda_write, secret)
        os.close(lambda_write)
        lambda_write = None
        _zero(secret)
        if confirmation_write is not None:
            assert process.stdout is not None
            raw_plan, plan_sha256 = _read_plan_line(process.stdout)
            sys.stdout.buffer.write(raw_plan)
            sys.stdout.buffer.flush()
            confirmation = (
                pipe_source.read_line(maximum=64, context="launcher_confirmation")
                if pipe_source is not None
                else _console_confirmation(plan_sha256)
            )
            try:
                try:
                    supplied = bytes(confirmation).decode("ascii", errors="strict")
                except UnicodeDecodeError:
                    _fail("launcher_confirmation_encoding_rejected")
                _require(supplied == plan_sha256, "launcher_confirmation_digest_mismatch")
                _write_all(confirmation_write, confirmation)
                _write_all(confirmation_write, b"\n")
            finally:
                _zero(confirmation)
                os.close(confirmation_write)
                confirmation_write = None
            _relay_remaining(process.stdout)
        else:
            assert process.stdout is not None
            _relay_remaining(process.stdout)
        return_code = process.wait()
        if stderr_thread is not None:
            stderr_thread.join(timeout=5)
        return return_code
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_error: BaseException | None = None
        _zero(secret)
        for descriptor in (
            lambda_read,
            lambda_write,
            confirmation_read,
            confirmation_write,
        ):
            if descriptor is not None and descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None and not active_exception:
            raise LauncherError("launcher_local_cleanup_failed") from cleanup_error


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        return run(sys.argv[1:] if arguments is None else arguments)
    except LauncherError as exc:
        sys.stderr.buffer.write(
            _canonical(
                {
                    "schema_version": 1,
                    "kind": "explainiverse-windows-launcher-error",
                    "stable_code": str(exc),
                    "secret_values_logged": False,
                }
            )
        )
        sys.stderr.buffer.flush()
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
