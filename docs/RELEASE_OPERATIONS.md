# Stable release control snapshot and recovery drill

This runbook is executable automation, not evidence that mutable service settings, Trusted
Publishing, or CUDA hardware are already configured. The default stable path remains blocked
until every preflight and hardware job is green. Release `0.15.2` alone may instead use the
explicitly authorized CPU-only exception described below; that exception is a disclosed absence
of CUDA evidence, never a representation that a GPU gate passed.

The signed immutable `v0.15.0` tag is retained as failed release-automation history. Workflow run
`33891048942` stopped during SBOM generation before artifact upload, attestation, PyPI publication,
or GitHub Release creation. The signed immutable `v0.15.1` tag is also retained as failed
release-automation history. Workflow run `33901507340` built successfully and retained workflow
artifacts, including the repaired SBOM, but GitHub skipped distribution attestation, PyPI
publication, and GitHub Release creation because a skipped ancestor condition propagated to those
jobs. Neither version is on PyPI or has a GitHub Release. Do not move, delete, republish, or create
a retroactive Release for either tag; `0.15.2` is the governed roll-forward.

## One-time 0.15.2 CPU-only roll-forward exception

The reviewed policy contains exactly one exception:
`EXPLAINIVERSE-v0.15.2-CPU-ONLY`. It is bound to tag/package version `v0.15.2`/`0.15.2`, merge
PR #7, maintainer `jemsbhai`, and the 2026-09-04 authorization. It records
`hardware_evidence_collected=false` and `cuda_release_verified=false`. The exception omits only
successful exact-commit evidence for these checks:

- `CUDA single-GPU (Torch latest)`
- `CUDA single-GPU (Torch minimum)`

It also records these four release jobs as omitted, not passed:

- `CUDA single-GPU (Torch latest)`
- `CUDA single-GPU (Torch minimum)`
- `CUDA two-GPU scheduled (Torch latest)`
- `CUDA two-GPU scheduled (Torch minimum)`

The authorization exists because isolated one- and two-GPU release runners remain unavailable and
the immutable `v0.15.0` and `v0.15.1` release attempts both stopped before publication.

All 23 provider-bound branch-protection contexts must already be restored when the administrator
capture is made; the exception applies only to the two CUDA check-run results on the exact release
commit.
The other 21 successful check runs, complete CPU test suite, Python/JavaScript/tutorial gates,
distribution reproducibility, signed tag, artifact attestations, environment approval, Trusted
Publishing, and immutable GitHub Release remain mandatory.

Use a narrow, reversible branch-protection window. Export the full current protection document
first as rollback input. Change only `required_status_checks`, preserving `strict=true` and
GitHub Actions app ID `15368` for each remaining check; do not disable administrator enforcement,
conversation resolution, review rules, force-push protection, or deletion protection. Remove only the two
names above, verify that the effective set is exactly the 21 names derived by the exception
policy, and merge PR #7 through GitHub with its expected head SHA. Do not push directly to
`main` and do not use a broad administrator bypass.

Immediately after GitHub merges PR #7, restore both CUDA checks with app ID `15368`, leaving all
23 baseline checks strict and administrator-enforced. Re-read protection and compare the complete
check/app bindings with `.github/release-control-policy.json`. Then wait for all 21 non-CUDA
`push`/`main` checks on the resulting merge commit; pull-request results do not qualify. Make the
fresh administrator capture and complete the successful preflight only while the restored
23-context protection is live. The exception waives the two missing CUDA check-run results, not
their branch-protection contexts. If the merge or restoration fails, restore or retain the
23-check baseline before investigating or retrying.

The preflight and publish dispatches must both name the exact same exception ID. A missing,
different, mixed CUDA-run/exception, future-tag, altered-policy, or stale-snapshot request fails
closed. The attested snapshot and immutable `RELEASE_GOVERNANCE.json`/`.md` record the exception,
every omitted check/job, its authorization, the full restored 23-context protection, and the
absence of CUDA verification. The capture also queries GitHub PR #7's fully paginated issue
timeline, requires exactly one authoritative merged event, and requires the release commit to
equal that event's commit with the reviewed repository, branch, merger identity, and merge time.
After the release, the normal path still requires all 23 branch contexts and the
complete one-/two-GPU evidence run; this exception cannot authorize any other tag.

## Administrator-authenticated pre-tag snapshot

GitHub's workflow-scoped token cannot read branch-protection or secret metadata. The release
administrator therefore captures those settings locally with the authenticated `gh` principal.
The same principal must immediately dispatch the preflight on the exact `main` commit. The
workflow accepts captures no more than 30 minutes old, binds the capture to its run, and attests
both the original and bound JSON files. The publish workflow recomputes that age against its own
clock and rejects a once-valid snapshot after 30 minutes; an immutable attestation proves what was
observed, not that mutable settings remained unchanged indefinitely.

The reviewed policy currently contains 23 required contexts. Each branch-protection entry and
exact-SHA check run must be bound to GitHub Actions app ID `15368` (slug `github-actions`); a
same-named context from another provider is not release evidence. Six of those contexts are the
automatic dependency-constraint matrix edges. Repository-level immutable Releases must also be
enabled so that finalized assets and their tag cannot be altered through the Release API. Update
live protection or immutable-Release settings only through separately authorized administrator
action, then recapture rather than editing the observation by hand.

For the normal hardware path, first dispatch `cuda-ci.yml` from the exact `main` commit and record
its run ID. All four all-attempt job records (single- and two-GPU, Torch minimum and latest) must
complete successfully exactly once. The preflight queries every jobs page with `filter=all`,
binds normalized runner and job evidence into the attested snapshot, and the publish workflow
queries and compares it again.

PowerShell example (do not print or persist the token):

```powershell
$releaseTag = "v0.15.2"
$releaseCommit = git rev-parse origin/main
$cudaRunId = "<successful-cuda-ci-run-id>"
$snapshot = Join-Path $env:TEMP "explainiverse-$releaseTag-controls.json"
$env:GH_TOKEN = gh auth token
python scripts/release_external_controls.py capture `
  --policy .github/release-control-policy.json `
  --output $snapshot `
  --repository jemsbhai/explainiverse `
  --tag $releaseTag `
  --commit $releaseCommit
if ($LASTEXITCODE -ne 0) { throw "release controls differ from reviewed policy" }
$snapshotBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($snapshot))
gh workflow run release-preflight.yml --ref main `
  -f tag=$releaseTag `
  -f release_commit=$releaseCommit `
  -f cuda_run_id=$cudaRunId `
  -f admin_snapshot_base64=$snapshotBase64
```

For the authorized `0.15.2` CPU-only path, omit `cuda_run_id` and use:

```powershell
$releaseTag = "v0.15.2"
$releaseCommit = git rev-parse origin/main
$cudaExceptionId = "EXPLAINIVERSE-v0.15.2-CPU-ONLY"
$snapshot = Join-Path $env:TEMP "explainiverse-$releaseTag-controls.json"
$env:GH_TOKEN = gh auth token
python scripts/release_external_controls.py capture `
  --policy .github/release-control-policy.json `
  --output $snapshot `
  --repository jemsbhai/explainiverse `
  --tag $releaseTag `
  --commit $releaseCommit `
  --cuda-exception-id $cudaExceptionId
if ($LASTEXITCODE -ne 0) { throw "release controls differ from exception policy" }
$snapshotBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($snapshot))
gh workflow run release-preflight.yml --ref main `
  -f tag=$releaseTag `
  -f release_commit=$releaseCommit `
  -f cuda_exception_id=$cudaExceptionId `
  -f admin_snapshot_base64=$snapshotBase64
```

The dispatch actor and the actor who triggers the current attempt must both be the authenticated
capture principal. A rerun records its actual `GITHUB_RUN_ATTEMPT` and
`GITHUB_TRIGGERING_ACTOR`; publication re-queries the Actions source run and rejects any attempt
or triggering-actor mismatch before building or publishing.

## Signed immutable tag

Configure and test a GPG, SSH, or S/MIME signing key that GitHub recognizes before beginning the
30-minute snapshot window. The stable tag must be an annotated tag object, point to the exact
preflighted `main` commit, and carry a signature GitHub reports as verified. The active `v*`
ruleset has no bypass actor and prevents tag update or deletion, so a bad pushed tag cannot be
repaired in place.

After preflight succeeds from an attested administrator capture that already proves all 23
baseline contexts are restored, create and inspect the tag locally before its first push:

```powershell
git tag -s v0.15.2 $releaseCommit -m "Release v0.15.2"
if ((git cat-file -t v0.15.2) -ne "tag") { throw "release tag is not annotated" }
if ((git rev-list -n 1 v0.15.2) -ne $releaseCommit) { throw "release tag moved" }
git push origin refs/tags/v0.15.2
$tagObject = git rev-parse "v0.15.2^{tag}"
$verified = gh api "repos/jemsbhai/explainiverse/git/tags/$tagObject" `
  --jq '.verification.verified'
if ($verified -ne "true") { throw "GitHub did not verify the release-tag signature" }
```

Do not dispatch publication unless every check above succeeds and the PyPI project owner has
confirmed the Trusted Publisher fields are owner `jemsbhai`, repository `explainiverse`, workflow
`publish-pypi.yml`, and environment `pypi`.

Record the successful preflight run ID and confirm its attested capture contains all 23 baseline
contexts and exact app bindings. Only after separately authorized signed-tag creation, dispatch
`publish-pypi.yml` from that tag and pass the same preflight run ID. The publish workflow
re-verifies the attestation, source workflow, run ID, repository, commit, tag, policy digest, PR #7
merge binding, and all observed controls. It also re-verifies either the exact CUDA run or the
exact attested `0.15.2` exception before any build. It accepts only PyPI's
HTTP 404 for the version before build and repeats that fail-closed check immediately before the
sole OIDC publisher action. An existing version or an ambiguous API/network result never reaches
the publisher; there is no `skip-existing` path. Normal hardware example:

```powershell
gh workflow run publish-pypi.yml --ref $releaseTag `
  -f tag=$releaseTag `
  -f preflight_run_id="<successful-release-preflight-run-id>" `
  -f cuda_run_id=$cudaRunId
```

Authorized `0.15.2` CPU-only example:

```powershell
gh workflow run publish-pypi.yml --ref $releaseTag `
  -f tag=$releaseTag `
  -f preflight_run_id="<successful-release-preflight-run-id>" `
  -f cuda_exception_id=$cudaExceptionId
```

The intentionally skipped `cuda-release` job remains an ancestor of the publication chain. To
prevent that valid CPU-only skip from silently suppressing later work, while still failing closed
on any real error, the three downstream jobs use these explicit conditions:

- `attest`: `always() && !cancelled() && needs.build.result == 'success'`
- `publish`: `always() && !cancelled() && needs.attest.result == 'success'`
- `github-release`: `always() && !cancelled() && needs.publish.result == 'success'`

`always()` makes each condition evaluate even when an earlier ancestor was intentionally skipped;
it is not an authorization to continue by itself. `!cancelled()` blocks cancellation, and the
exact direct-upstream success check blocks every upstream failure or non-successful direct result.
Do not weaken or remove any of these three parts. Their absence caused the `v0.15.1` downstream
jobs to be skipped even though its build and repaired SBOM succeeded.

Candidate-authored Python, tutorial, and JavaScript gates run before any release input is created.
After those gates finish, the workflow refuses reserved release paths, checks the signed tag out a
second time into a clean directory, and revalidates the tag object, exact commit, and `main`
ancestry. It then obtains the exact `Artifact byte reproducibility` run ID from the accepted
external-control snapshot, downloads both clean-build artifact sets and their report from that
run, and rechecks their environment and byte manifests. The distribution is built once from the
second checkout and must be byte-identical to both accepted builds before it can be attested or
uploaded. Missing or expired proof artifacts require a fresh exact-commit reproducibility run;
they are never bypassed.

For SBOM generation, the workflow derives and retains a PEP-621-only view of the reviewed
`pyproject.toml`. This removes only the `tool.poetry` namespace before invoking CycloneDX, because
CycloneDX otherwise treats Poetry 2's partial tool configuration as legacy package metadata and
fails before publication. The helper parses both documents, requires the complete `project` table
to remain equal, and fails if any `tool.poetry` metadata survives.

GitHub deploys hosted-runner image releases gradually, so two parallel jobs that both request the
versioned `ubuntu-24.04` label can receive different exact `ImageVersion` values. The comparison
does not erase or normalize that difference: each value is mandatory, both are retained in the
complete manifests, and the comparison report records both values and whether they match. The
requested label, actual image family, operating system, architecture, Python and pip versions,
hash-locked tool graph, source, workflow run, and attempt remain exact requirements. A rollout
version difference is acceptable only when those stable build inputs match and the separate
mandatory comparison proves the wheel and source distribution byte-identical. Missing image
provenance or any stable-input difference fails closed, and publication replays both comparisons
before matching its clean-checkout build against each accepted artifact set.

Before the draft GitHub Release is finalized, the workflow downloads and attestation-verifies the
full preflight evidence, then generates `RELEASE_GOVERNANCE.json` and
`RELEASE_GOVERNANCE.md`. The record binds the release actor, environment reviewer and self-review
setting, tag/commit, preflight, CUDA gate mode, and external-control policy/snapshot digests. The
hardware path includes its CUDA run ID; the `0.15.2` exception instead includes its exact ID,
authorization, reason, PR #7 merge commit, omitted checks/jobs, and explicit false
hardware/verification fields. If the live project still uses one operator, the release notes also
disclose the absence of segregation of duties. The draft and its assets must verify before it is
published; recovery rebuilds the record and canonical Markdown from the retained policy and
attested external-control snapshot before preserving the identical governance disclosure. The
normal and recovery paths both fetch PyPI's
release JSON and per-file Integrity provenance, constrain the DSSE subjects and Trusted Publisher
identity to the exact repository/workflow/environment, and cryptographically verify every file
with the hash-locked `pypi-attestations` tool before finalizing. Both paths then re-read the final
GitHub Release and require the service's immutable flag.

GitHub's immutable-Release setting protects the tag and attached assets, but the title and
release notes remain mutable. The attached `RELEASE_GOVERNANCE.json` and
`RELEASE_GOVERNANCE.md` governance assets are authoritative. The final REST re-read proves that
the release notes contained the exact Markdown disclosure at finalization; it does not make that
body permanent, so any later notes drift must not supersede the retained governance assets.

PyPI publisher configuration cannot be read through a
public API; the project owner must separately archive direct settings-page evidence for owner
`jemsbhai`, repository `explainiverse`, workflow `publish-pypi.yml`, environment `pypi`. The
token-free OIDC publication job is the final acceptance test. No repository or environment PyPI
API token is permitted by the reviewed snapshot.

## Post-PyPI recovery drill

For a separately authorized release intended to exercise recovery, set
`stage_recovery_drill=true` on `publish-pypi.yml`. The workflow intentionally exits only after the
single OIDC PyPI upload succeeds. Do not rerun all jobs and do not add `skip-existing`.

Dispatch `recover-github-release.yml` with the tag and failed source run ID. For the formal drill,
set `require_staged_drill=true`; the verifier distinguishes that explicit failed staging step from
an unplanned downstream failure and refuses a successful or ambiguous source run. It will:

```powershell
gh workflow run recover-github-release.yml --ref $releaseTag `
  -f tag=$releaseTag `
  -f source_run_id="<failed-publish-workflow-run-id>" `
  -f require_staged_drill=true
```

The `--ref $releaseTag` argument is mandatory: recovery fails unless the workflow execution ref
is `refs/tags/$releaseTag` and its `GITHUB_SHA` equals the checked-out tag commit.

1. query every jobs page with `filter=all` and require the original build, attestation, and PyPI
   jobs to have completed successfully exactly once in the named `publish-pypi.yml` run on the
   same tag commit;
2. download only that run's retained `release-distributions` and `release-provenance` artifacts;
3. rebuild the governance record and canonical Markdown from the exact tag policy and retained
   external-control snapshot, including the reviewed CPU-only exception and PR merge binding;
4. verify `SHA256SUMS`, exact-workflow GitHub artifact attestations, the exact filename/SHA-256
   inventory returned by PyPI, and every file's PyPI Integrity provenance and publisher identity;
5. create or reuse a draft GitHub Release, reuse only byte-identical existing assets, and upload
   only missing downstream assets;
6. download the draft assets and prove the GitHub distribution hashes equal PyPI before finalizing.

The recovery workflow contains no PyPI publisher action, credential, `twine upload`, or
`skip-existing` path. Its always-running evidence upload retains complete or partial source-run
and all-attempt jobs JSON, PyPI JSON, attestation verification output, final GitHub metadata, and
the final asset hash inventory for 90 days. Archive that artifact, the source and recovery run
URLs, and both attestations as the drill record.

## Legacy 0.14.0 incident

The public `0.14.0` files were uploaded to PyPI on 2026-08-10 through Twine, not Trusted
Publishing. PyPI reports wheel SHA-256
`b1b98dfdfc0acbc8dc2113d8db87d40ae9cec2f958ed25b00bc6e30d43db41d4` and source SHA-256
`e2ab525f720d9970f25c307be84b9a5a6bb5feb612a4457ba9d72925cf2af68b`. Its `v0.14.0` tag is
unsigned, no GitHub Release exists, and no original attested build run/artifact set is available.
The recovery workflow must and does reject that incident: downloading the public PyPI files now
cannot recreate original provenance. Recovery requires owner authority and a separately reviewed
legacy disclosure decision; it cannot be represented as a successful recovery drill.

## CUDA capacity acceptance

Set `CUDA_SINGLE_RUNNER` and `CUDA_TWO_RUNNER` only to GitHub-managed GPU runners or isolated,
ephemeral runners approved for public-repository code. With variables absent or incorrect, a
hosted reporter fails the routing contract before checkout, dependency installation, or hardware
execution; it cannot produce CUDA evidence. Require both single-GPU Torch
minimum/latest contexts on `main`; run both two-GPU edges on the scheduled workflow. A green gate
means the checked-in exact node manifest matched collection and all 15 CUDA tests ran with zero
skips, including adapter prediction/gradients, every vector and CAM gradient family,
randomisation success/failure, initialized-device RNG byte restoration, dtype/device placement,
and hook cleanup.

The reviewed policy requires the single-GPU jobs to carry the custom runner label
`explainiverse-cuda-single` and the two-GPU jobs to carry
`explainiverse-cuda-two`. Configure the variables to those exact labels and assign each label only
to capacity whose visible-device topology matches its name. The preflight checks the labels and
exact device count; the repository administrator must still retain the infrastructure-owner
record establishing that the selected self-hosted runners are isolated and approved.

As of the 2026-08-11 audit, the repository API reported zero Actions variables and zero registered
runners. Both variables are therefore unset and no live GPU acceptance exists. A repository
administrator must provision approved capacity, set both variables, dispatch `cuda-ci.yml` on the
exact release commit, and retain a green four-job run before the preflight can succeed. The
`tests_cuda` session hook converts every runtime or collection skip into a failed job and rejects
missing, extra, reordered, or duplicate release nodes when manifest enforcement is enabled. A
local one-GPU diagnostic, even if green, does not substitute for this hosted four-job evidence.
The exact `0.15.2` CPU-only exception is the sole departure: it truthfully records that this
evidence was not collected and makes no CUDA release-verification claim. It does not relax the
runner or evidence requirements for a future tag.
