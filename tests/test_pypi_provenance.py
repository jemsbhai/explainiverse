"""Adversarial contracts for PyPI Integrity provenance capture."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_pypi_provenance.py"
SPEC = importlib.util.spec_from_file_location("verify_pypi_provenance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
provenance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provenance
SPEC.loader.exec_module(provenance)

FILES = {
    "explainiverse-0.15.0-py3-none-any.whl": hashlib.sha256(b"wheel").hexdigest(),
    "explainiverse-0.15.0.tar.gz": hashlib.sha256(b"sdist").hexdigest(),
}


def _statement(
    filename,
    digest,
    *,
    predicate_type=provenance._PYPI_PUBLISH_ATTESTATION_V1,
    predicate=None,
):
    statement = {
        "_type": provenance._IN_TOTO_STATEMENT_V1,
        "subject": [{"name": filename, "digest": {"sha256": digest}}],
        "predicateType": predicate_type,
        "predicate": predicate,
    }
    return base64.b64encode(json.dumps(statement).encode()).decode()


def _payload(filename, digest, *, predicate=None):
    return {
        "version": 1,
        "attestation_bundles": [
            {
                "publisher": {
                    "kind": "GitHub",
                    "repository": "jemsbhai/explainiverse",
                    "workflow": "publish-pypi.yml",
                    "environment": "pypi",
                    "claims": None,
                },
                "attestations": [
                    {
                        "version": 1,
                        "envelope": {
                            "signature": "signed",
                            "statement": _statement(filename, digest, predicate=predicate),
                        },
                        "verification_material": {
                            "certificate": "certificate",
                            "transparency_entries": [{"logIndex": "1"}],
                        },
                    }
                ],
            }
        ],
    }


def _metadata():
    return {
        "info": {"name": "Explainiverse", "version": "0.15.0"},
        "urls": [
            {"filename": filename, "digests": {"sha256": digest}}
            for filename, digest in FILES.items()
        ],
    }


def _verify(payload, filename=None, digest=None):
    filename = filename or next(iter(FILES))
    return provenance.verify_provenance(
        payload,
        filename=filename,
        sha256=digest or FILES[filename],
        repository="jemsbhai/explainiverse",
        workflow="publish-pypi.yml",
        environment="pypi",
    )


@pytest.mark.parametrize("predicate", [None, {}], ids=["json-null", "empty-object"])
def test_matching_pypi_publisher_publish_subject_and_predicate_are_required(predicate):
    filename = next(iter(FILES))
    assert _verify(_payload(filename, FILES[filename], predicate=predicate)) == {
        "bundle_count": 1,
        "attestation_count": 1,
        "publish_attestation_count": 1,
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("kind", "GitLab"),
        ("repository", "attacker/explainiverse"),
        ("workflow", "attacker.yml"),
        ("environment", "unreviewed"),
    ],
)
def test_publisher_identity_mismatch_fails_closed(field, replacement):
    filename = next(iter(FILES))
    payload = _payload(filename, FILES[filename])
    payload["attestation_bundles"][0]["publisher"][field] = replacement
    with pytest.raises(ValueError, match="publisher mismatch"):
        _verify(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda statement: statement["subject"][0].update(name="other.whl"),
            "subject mismatch",
        ),
        (
            lambda statement: statement["subject"][0]["digest"].update(sha256="0" * 64),
            "subject mismatch",
        ),
        (lambda statement: statement.update(_type="attacker"), "wrong statement type"),
        (lambda statement: statement.update(predicateType="attacker"), "no publish attestation"),
        (lambda statement: statement.update(subject=[]), "exactly one subject"),
    ],
)
def test_statement_substitution_or_malformed_publish_claim_fails_closed(mutation, match):
    filename = next(iter(FILES))
    payload = _payload(filename, FILES[filename])
    envelope = payload["attestation_bundles"][0]["attestations"][0]["envelope"]
    statement = json.loads(base64.b64decode(envelope["statement"]))
    mutation(statement)
    envelope["statement"] = base64.b64encode(json.dumps(statement).encode()).decode()
    with pytest.raises(ValueError, match=match):
        _verify(payload)


@pytest.mark.parametrize(
    "predicate",
    [
        {"unexpected": None},
        [],
        [1],
        "",
        "claim",
        0,
        1,
        1.5,
        False,
        True,
    ],
    ids=[
        "nonempty-object",
        "empty-array",
        "nonempty-array",
        "empty-string",
        "string",
        "zero",
        "integer",
        "float",
        "false",
        "true",
    ],
)
def test_publish_predicate_rejects_nonempty_mappings_arrays_and_scalars(predicate):
    filename = next(iter(FILES))
    payload = _payload(filename, FILES[filename], predicate=predicate)
    with pytest.raises(ValueError, match="JSON null or an empty object"):
        _verify(payload)


def test_publish_predicate_rejects_a_missing_field():
    filename = next(iter(FILES))
    payload = _payload(filename, FILES[filename])
    envelope = payload["attestation_bundles"][0]["attestations"][0]["envelope"]
    statement = json.loads(base64.b64decode(envelope["statement"]))
    statement.pop("predicate")
    envelope["statement"] = base64.b64encode(json.dumps(statement).encode()).decode()
    with pytest.raises(ValueError, match="must contain a predicate"):
        _verify(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(version=2), "schema version 1"),
        (lambda value: value.update(attestation_bundles=[]), "no attestation bundles"),
        (
            lambda value: value["attestation_bundles"][0].update(attestations=[]),
            "bundle.*empty",
        ),
        (
            lambda value: value["attestation_bundles"][0]["attestations"][0]["envelope"].update(
                signature=""
            ),
            "no signature",
        ),
        (
            lambda value: value["attestation_bundles"][0]["attestations"][0].update(
                verification_material={
                    "certificate": "certificate",
                    "transparency_entries": [],
                }
            ),
            "no transparency entry",
        ),
    ],
)
def test_empty_or_unverifiable_integrity_evidence_fails_closed(mutation, match):
    filename = next(iter(FILES))
    payload = _payload(filename, FILES[filename])
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        _verify(payload)


def test_invalid_base64_statement_fails_closed():
    filename = next(iter(FILES))
    payload = _payload(filename, FILES[filename])
    payload["attestation_bundles"][0]["attestations"][0]["envelope"]["statement"] = "!"
    with pytest.raises(ValueError, match="invalid base64 JSON statement"):
        _verify(payload)


def test_pypi_release_inventory_must_exactly_match_reviewed_hashes():
    provenance.verify_pypi_release_json(
        _metadata(), project="explainiverse", version="0.15.0", expected=FILES
    )
    for mutation in (
        lambda value: value["urls"].pop(),
        lambda value: value["urls"][0]["digests"].update(sha256="0" * 64),
        lambda value: value["info"].update(version="0.14.0"),
    ):
        metadata = _metadata()
        mutation(metadata)
        with pytest.raises(ValueError):
            provenance.verify_pypi_release_json(
                metadata, project="explainiverse", version="0.15.0", expected=FILES
            )


class _Response:
    def __init__(self, raw, status=200):
        self.raw = raw
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def getcode(self):
        return self.status

    def read(self, limit):
        return self.raw[:limit]


def test_capture_retains_each_raw_response_and_a_hash_bound_manifest(tmp_path):
    requested = []

    def opener(request, *, timeout):
        assert timeout == 30
        assert request.headers["Accept"] == provenance._INTEGRITY_ACCEPT
        requested.append(request.full_url)
        filename = urllib_filename = request.full_url.rsplit("/", 2)[-2]
        filename = provenance.urllib.parse.unquote(urllib_filename)
        return _Response(json.dumps(_payload(filename, FILES[filename])).encode())

    output = tmp_path / "pypi-provenance"
    manifest = provenance.capture_provenance(
        metadata=_metadata(),
        expected=FILES,
        project="explainiverse",
        version="0.15.0",
        repository="jemsbhai/explainiverse",
        workflow="publish-pypi.yml",
        environment="pypi",
        output_dir=output,
        attempts=1,
        retry_delay=0,
        opener=opener,
    )
    assert len(requested) == len(FILES)
    assert {value["filename"] for value in manifest["files"]} == set(FILES)
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for record in manifest["files"]:
        raw = (output / record["evidence_file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["evidence_sha256"]


def test_capture_rejects_stale_output_directory_before_fetching(tmp_path):
    output = tmp_path / "pypi-provenance"
    output.mkdir()
    with pytest.raises(FileExistsError):
        provenance.capture_provenance(
            metadata=_metadata(),
            expected=FILES,
            project="explainiverse",
            version="0.15.0",
            repository="jemsbhai/explainiverse",
            workflow="publish-pypi.yml",
            environment="pypi",
            output_dir=output,
            attempts=1,
            retry_delay=0,
            opener=lambda *_args, **_kwargs: pytest.fail("must not fetch"),
        )


def test_fetch_retries_transient_failure_but_never_accepts_invalid_json():
    calls = 0

    def opener(_request, *, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient")
        return _Response(b"not JSON")

    with pytest.raises(ValueError, match="after 2 attempt"):
        provenance._fetch_json(
            "https://pypi.org/integrity/example",
            attempts=2,
            retry_delay=0,
            opener=opener,
        )
    assert calls == 2
