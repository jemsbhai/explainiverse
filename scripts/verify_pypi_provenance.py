"""Capture and verify PyPI Integrity provenance for exact release files.

PyPI's Integrity API exposes the attestations accepted with each distribution.
This verifier binds every response to the reviewed release hashes and to the
expected Trusted Publisher identity.  It deliberately uses only the Python
standard library; it does not install or invoke an unpinned verification tool.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,254}")
_DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz")
_IN_TOTO_STATEMENT_V1 = "https://in-toto.io/Statement/v1"
_PYPI_PUBLISH_ATTESTATION_V1 = "https://docs.pypi.org/attestations/publish/v1"
_INTEGRITY_ACCEPT = "application/vnd.pypi.integrity.v1+json"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _safe_distribution_filename(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_FILENAME.fullmatch(value) is None
        or not value.endswith(_DISTRIBUTION_SUFFIXES)
    ):
        raise ValueError(f"{name} is not a safe wheel or source-distribution filename")
    return value


def _normalize_project(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def parse_sha256sums(path: Path) -> dict[str, str]:
    """Parse the exact GNU sha256sum subset produced by the release build."""
    hashes: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise ValueError(f"SHA256SUMS line {line_number} is malformed")
        digest = line[:64].lower()
        filename = _safe_distribution_filename(line[66:], f"SHA256SUMS line {line_number} filename")
        if _SHA256.fullmatch(digest) is None:
            raise ValueError(f"SHA256SUMS line {line_number} has an invalid digest")
        if filename in hashes:
            raise ValueError(f"SHA256SUMS contains duplicate filename {filename!r}")
        hashes[filename] = digest
    if not hashes:
        raise ValueError("SHA256SUMS must contain at least one distribution")
    if not any(name.endswith(".whl") for name in hashes):
        raise ValueError("SHA256SUMS must contain a wheel")
    if not any(name.endswith(".tar.gz") for name in hashes):
        raise ValueError("SHA256SUMS must contain a source distribution")
    return hashes


def verify_pypi_release_json(
    metadata: Mapping[str, Any],
    *,
    project: str,
    version: str,
    expected: Mapping[str, str],
) -> None:
    """Require PyPI's release inventory to equal the reviewed hashes."""
    info = _mapping(metadata.get("info"), "PyPI info")
    name = info.get("name")
    if not isinstance(name, str) or _normalize_project(name) != _normalize_project(project):
        raise ValueError(f"PyPI project mismatch: expected {project!r}, got {name!r}")
    if info.get("version") != version:
        raise ValueError(
            f"PyPI version mismatch: expected {version!r}, got {info.get('version')!r}"
        )

    published: dict[str, str] = {}
    for raw_file in _sequence(metadata.get("urls"), "PyPI urls"):
        file = _mapping(raw_file, "PyPI release file")
        filename = _safe_distribution_filename(file.get("filename"), "PyPI filename")
        digest = _mapping(file.get("digests"), "PyPI file digests").get("sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest.lower()) is None:
            raise ValueError(f"PyPI returned an invalid SHA-256 for {filename!r}")
        if filename in published:
            raise ValueError(f"PyPI returned duplicate filename {filename!r}")
        published[filename] = digest.lower()
    if published != dict(expected):
        raise ValueError(
            f"PyPI artifact inventory/hash mismatch: expected {dict(expected)!r}, "
            f"got {published!r}"
        )


def _decode_statement(value: Any, *, filename: str) -> Mapping[str, Any]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"PyPI provenance for {filename!r} has no encoded statement")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
        return _mapping(json.loads(decoded), f"attestation statement for {filename!r}")
    except (UnicodeEncodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"PyPI provenance for {filename!r} has an invalid base64 JSON statement"
        ) from exc


def verify_provenance(
    payload: Mapping[str, Any],
    *,
    filename: str,
    sha256: str,
    repository: str,
    workflow: str,
    environment: str,
) -> dict[str, int]:
    """Verify one Integrity response's publisher and attestation subjects."""
    filename = _safe_distribution_filename(filename, "provenance filename")
    if _SHA256.fullmatch(sha256) is None:
        raise ValueError(f"expected SHA-256 for {filename!r} is invalid")
    if payload.get("version") != 1:
        raise ValueError(f"PyPI provenance for {filename!r} must use schema version 1")
    bundles = [
        _mapping(value, f"attestation bundle for {filename!r}")
        for value in _sequence(payload.get("attestation_bundles"), "attestation_bundles")
    ]
    if not bundles:
        raise ValueError(f"PyPI provenance for {filename!r} contains no attestation bundles")

    expected_publisher = {
        "kind": "GitHub",
        "repository": repository,
        "workflow": workflow,
        "environment": environment,
    }
    attestation_count = 0
    publish_attestation_count = 0
    for bundle in bundles:
        publisher = _mapping(bundle.get("publisher"), "attestation publisher")
        actual_publisher = {key: publisher.get(key) for key in expected_publisher}
        if actual_publisher != expected_publisher:
            raise ValueError(
                f"PyPI provenance publisher mismatch for {filename!r}: "
                f"expected {expected_publisher!r}, got {actual_publisher!r}"
            )
        attestations = [
            _mapping(value, f"attestation for {filename!r}")
            for value in _sequence(bundle.get("attestations"), "attestations")
        ]
        if not attestations:
            raise ValueError(f"PyPI provenance bundle for {filename!r} is empty")
        for attestation in attestations:
            attestation_count += 1
            if attestation.get("version") != 1:
                raise ValueError(f"PyPI attestation for {filename!r} must use version 1")
            envelope = _mapping(attestation.get("envelope"), "attestation envelope")
            if not isinstance(envelope.get("signature"), str) or not envelope.get("signature"):
                raise ValueError(f"PyPI attestation for {filename!r} has no signature")
            statement = _decode_statement(envelope.get("statement"), filename=filename)
            if statement.get("_type") != _IN_TOTO_STATEMENT_V1:
                raise ValueError(f"PyPI attestation for {filename!r} has the wrong statement type")
            subjects = [
                _mapping(value, f"attestation subject for {filename!r}")
                for value in _sequence(statement.get("subject"), "attestation subject")
            ]
            if len(subjects) != 1:
                raise ValueError(
                    f"PyPI attestation for {filename!r} must contain exactly one subject"
                )
            subject = subjects[0]
            digest = _mapping(subject.get("digest"), "attestation subject digest")
            if subject.get("name") != filename or digest.get("sha256") != sha256:
                raise ValueError(
                    f"PyPI attestation subject mismatch for {filename!r}: got "
                    f"{subject.get('name')!r}/{digest.get('sha256')!r}"
                )
            verification = _mapping(
                attestation.get("verification_material"), "attestation verification material"
            )
            if not isinstance(verification.get("certificate"), str) or not verification.get(
                "certificate"
            ):
                raise ValueError(f"PyPI attestation for {filename!r} has no certificate")
            transparency = _sequence(
                verification.get("transparency_entries"), "attestation transparency entries"
            )
            if not transparency:
                raise ValueError(f"PyPI attestation for {filename!r} has no transparency entry")
            if statement.get("predicateType") == _PYPI_PUBLISH_ATTESTATION_V1:
                if "predicate" not in statement:
                    raise ValueError(
                        f"PyPI publish attestation for {filename!r} must contain a predicate"
                    )
                predicate = statement["predicate"]
                if predicate is not None and (
                    not isinstance(predicate, Mapping) or bool(predicate)
                ):
                    raise ValueError(
                        f"PyPI publish attestation for {filename!r} predicate must be "
                        "JSON null or an empty object"
                    )
                publish_attestation_count += 1

    if publish_attestation_count == 0:
        raise ValueError(f"PyPI provenance for {filename!r} has no publish attestation")
    return {
        "bundle_count": len(bundles),
        "attestation_count": attestation_count,
        "publish_attestation_count": publish_attestation_count,
    }


def _integrity_url(project: str, version: str, filename: str) -> str:
    return "https://pypi.org/integrity/{}/{}/{}/provenance".format(
        urllib.parse.quote(project, safe=""),
        urllib.parse.quote(version, safe=""),
        urllib.parse.quote(filename, safe=""),
    )


def _fetch_json(
    url: str,
    *,
    attempts: int,
    retry_delay: float,
    opener: Callable[..., Any] | None = None,
) -> tuple[bytes, Mapping[str, Any]]:
    if attempts < 1:
        raise ValueError("fetch attempts must be at least one")
    if retry_delay < 0:
        raise ValueError("retry delay cannot be negative")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": _INTEGRITY_ACCEPT,
            "User-Agent": "explainiverse-release-provenance/1",
        },
    )
    open_url = opener or urllib.request.urlopen
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with open_url(request, timeout=30) as response:
                status = response.getcode()
                if status != 200:
                    raise ValueError(f"PyPI Integrity API returned HTTP {status} for {url}")
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    raise ValueError(f"PyPI Integrity response is too large for {url}")
                return raw, _mapping(json.loads(raw), f"PyPI Integrity response for {url}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(retry_delay)
    assert last_error is not None
    raise ValueError(
        f"could not fetch valid PyPI Integrity provenance after {attempts} attempt(s): "
        f"{last_error}"
    ) from last_error


def capture_provenance(
    *,
    metadata: Mapping[str, Any],
    expected: Mapping[str, str],
    project: str,
    version: str,
    repository: str,
    workflow: str,
    environment: str,
    output_dir: Path,
    attempts: int = 12,
    retry_delay: float = 5,
    opener: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Fetch, verify, and retain every release file's raw provenance response."""
    verify_pypi_release_json(metadata, project=project, version=version, expected=expected)
    output_dir.mkdir(parents=True, exist_ok=False)
    files = []
    for filename, digest in sorted(expected.items()):
        url = _integrity_url(project, version, filename)
        raw, payload = _fetch_json(
            url,
            attempts=attempts,
            retry_delay=retry_delay,
            opener=opener,
        )
        counts = verify_provenance(
            payload,
            filename=filename,
            sha256=digest,
            repository=repository,
            workflow=workflow,
            environment=environment,
        )
        evidence_name = f"{filename}.provenance.json"
        (output_dir / evidence_name).write_bytes(raw)
        files.append(
            {
                "filename": filename,
                "sha256": digest,
                "integrity_url": url,
                "evidence_file": evidence_name,
                "evidence_sha256": hashlib.sha256(raw).hexdigest(),
                **counts,
            }
        )
    manifest = {
        "schema_version": 1,
        "project": project,
        "version": version,
        "publisher": {
            "kind": "GitHub",
            "repository": repository,
            "workflow": workflow,
            "environment": environment,
        },
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _version_from_tag(tag: str) -> str:
    match = _TAG.fullmatch(tag)
    if match is None:
        raise ValueError("release tag must have the form vMAJOR.MINOR.PATCH")
    return ".".join(match.groups())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypi-json", type=Path, required=True)
    parser.add_argument("--sums", type=Path, required=True)
    parser.add_argument("--project", default="explainiverse")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--retry-delay", type=float, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        metadata = _mapping(
            json.loads(args.pypi_json.read_text(encoding="utf-8")), "PyPI release JSON"
        )
        capture_provenance(
            metadata=metadata,
            expected=parse_sha256sums(args.sums),
            project=args.project,
            version=_version_from_tag(args.tag),
            repository=args.repository,
            workflow=args.workflow,
            environment=args.environment,
            output_dir=args.output_dir,
            attempts=args.attempts,
            retry_delay=args.retry_delay,
        )
        print(f"verified PyPI Integrity provenance in {args.output_dir}")
        return 0
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
