"""Regression tests for the intentionally blocked Python typing claim."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_typing_readiness.py"
SPEC = importlib.util.spec_from_file_location("audit_typing_readiness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
typing_audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = typing_audit
SPEC.loader.exec_module(typing_audit)
POLICY = ROOT / ".github" / "typing-readiness-policy.json"


def _audit(root: Path, policy: Path = POLICY, distributions=()):
    return typing_audit.audit_blocked_distribution(
        policy_path=policy,
        project_file=root / "pyproject.toml",
        repository_root=root,
        distributions=distributions,
    )


def _untyped_project(root: Path) -> None:
    (root / "src" / "explainiverse").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "explainiverse"\nclassifiers = ["Programming Language :: Python :: 3"]\n',
        encoding="utf-8",
    )


def test_repository_remains_truthfully_untyped():
    report = _audit(ROOT)
    assert report["claim_status"] == "blocked"
    assert report["source_marker_absent"] is True


def test_source_marker_and_typed_classifier_are_both_rejected(tmp_path):
    _untyped_project(tmp_path)
    marker = tmp_path / "src" / "explainiverse" / "py.typed"
    marker.touch()
    with pytest.raises(typing_audit.TypingReadinessError, match="marker exists"):
        _audit(tmp_path)

    marker.unlink()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nclassifiers = ["Typing :: Typed"]\n', encoding="utf-8"
    )
    with pytest.raises(typing_audit.TypingReadinessError, match="classifier exists"):
        _audit(tmp_path)


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_built_archives_cannot_smuggle_a_typing_marker(tmp_path, archive_kind):
    _untyped_project(tmp_path)
    if archive_kind == "wheel":
        archive = tmp_path / "explainiverse-0-py3-none-any.whl"
        with zipfile.ZipFile(archive, "w") as wheel:
            wheel.writestr("explainiverse/py.typed", "")
    else:
        archive = tmp_path / "explainiverse-0.tar.gz"
        marker = tmp_path / "py.typed"
        marker.touch()
        with tarfile.open(archive, "w:gz") as sdist:
            sdist.add(marker, arcname="explainiverse-0/src/explainiverse/py.typed")

    with pytest.raises(typing_audit.TypingReadinessError, match="shipped"):
        _audit(tmp_path, distributions=[archive])


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
def test_built_archives_cannot_smuggle_a_typed_classifier(tmp_path, archive_kind):
    _untyped_project(tmp_path)
    payload = b"Metadata-Version: 2.4\nName: explainiverse\nClassifier: Typing :: Typed\n"
    if archive_kind == "wheel":
        archive = tmp_path / "explainiverse-0-py3-none-any.whl"
        with zipfile.ZipFile(archive, "w") as wheel:
            wheel.writestr("explainiverse-0.dist-info/METADATA", payload)
    else:
        archive = tmp_path / "explainiverse-0.tar.gz"
        with tarfile.open(archive, "w:gz") as sdist:
            member = tarfile.TarInfo("explainiverse-0/PKG-INFO")
            member.size = len(payload)
            sdist.addfile(member, io.BytesIO(payload))

    with pytest.raises(typing_audit.TypingReadinessError, match="classifier shipped"):
        _audit(tmp_path, distributions=[archive])


def test_policy_cannot_flip_to_a_typed_claim_without_replacing_the_guard(tmp_path):
    _untyped_project(tmp_path)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    policy["claim_status"] = "ready"
    policy_path = tmp_path / "typing-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(typing_audit.TypingReadinessError, match="strict consumer certification"):
        _audit(tmp_path, policy=policy_path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("forbidden_marker", "src/explainiverse/not-the-typing-marker"),
        ("forbidden_classifier", "Typing :: Definitely Untyped"),
    ],
)
def test_policy_cannot_redirect_the_canonical_typing_boundaries(tmp_path, field, replacement):
    _untyped_project(tmp_path)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    policy[field] = replacement
    policy_path = tmp_path / "typing-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    if field == "forbidden_marker":
        (tmp_path / "src" / "explainiverse" / "py.typed").touch()
    else:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nclassifiers = ["Typing :: Typed"]\n', encoding="utf-8"
        )

    with pytest.raises(typing_audit.TypingReadinessError, match="canonical boundary"):
        _audit(tmp_path, policy=policy_path)
