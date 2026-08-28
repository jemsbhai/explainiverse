# Stable release control snapshot and recovery drill

This runbook is executable automation, not evidence that mutable service settings or CUDA
hardware are already configured. A stable release remains blocked until every preflight and
hardware job is green and the PyPI project owner directly verifies the Trusted Publisher.

## Administrator-authenticated pre-tag snapshot

GitHub's workflow-scoped token cannot read branch-protection or secret metadata. The release
administrator therefore captures those settings locally with the authenticated `gh` principal.
The same principal must immediately dispatch the preflight on the exact `main` commit. The
workflow accepts captures no more than 30 minutes old, binds the capture to its run, and attests
both the original and bound JSON files. The complete owner-authenticated installed-App record must
be captured no more than 10 minutes before the JSON snapshot and cannot postdate it beyond the
one-minute clock-skew allowance. The publish workflow recomputes both ages against its own clock and
rejects a once-valid snapshot after 30 minutes; an immutable attestation proves what was observed,
not that mutable settings remained unchanged indefinitely.

The reviewed policy currently contains 23 required contexts. Each branch-protection entry and
exact-SHA check run must be bound to GitHub Actions app ID `15368` (slug `github-actions`); a
same-named context from another provider is not release evidence. Six of those contexts are the
automatic dependency-constraint matrix edges. Repository-level immutable Releases must also be
enabled so that finalized assets and their tag cannot be altered through the Release API. Update
live protection or immutable-Release settings only through separately authorized administrator
action, then recapture rather than editing the observation by hand.
The capture also queries
`/repos/jemsbhai/explainiverse/actions/permissions/fork-pr-contributor-approval` directly and
requires the exact `all_external_contributors` value; the browser setting or an earlier manual
audit alone is not accepted as pre-tag evidence.

First dispatch `cuda-ci.yml` from the exact `main` commit and record its run ID. All four
all-attempt job records (single- and two-GPU, Torch minimum and latest) must complete successfully
exactly once. The preflight queries every jobs page with `filter=all`, binds normalized runner and
job evidence into the attested snapshot, and the publish workflow queries and compares it again.

PowerShell example (do not print or persist the token):

```powershell
$releaseTag = "v0.15.0"
$releaseCommit = git rev-parse origin/main
$cudaRunId = "<successful-cuda-ci-run-id>"
$snapshot = Join-Path $env:TEMP "explainiverse-$releaseTag-controls.json"
$installedAppAuthority = Join-Path $env:TEMP "explainiverse-$releaseTag-installed-apps.json"
$env:GH_TOKEN = gh auth token
python scripts/release_external_controls.py capture `
  --policy .github/release-control-policy.json `
  --output $snapshot `
  --repository jemsbhai/explainiverse `
  --tag $releaseTag `
  --commit $releaseCommit `
  --installed-app-authority $installedAppAuthority
if ($LASTEXITCODE -ne 0) { throw "release controls differ from reviewed policy" }
$snapshotBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($snapshot))
gh workflow run release-preflight.yml --ref main `
  -f tag=$releaseTag `
  -f release_commit=$releaseCommit `
  -f cuda_run_id=$cudaRunId `
  -f admin_snapshot_base64=$snapshotBase64
```

Create `$installedAppAuthority` immediately beforehand from the owner-authenticated
`https://github.com/settings/installations` list and each installation's Configure and pending
permission-update views. It must use schema version 1; identify capture principal `jemsbhai` and
repository `jemsbhai/explainiverse`; set `coverage_complete` true only after every installed App
has been inspected; and include every installation ID, exact displayed name, repository selection
and access, suspension state, current read/write permissions, and requested additional permissions.
For suspension semantics, record `Suspend` only when the App is currently active and the danger
zone offers to suspend it; record `Unsuspend` only when the App is currently suspended and the
page offers to restore it. Never infer the inverse state from a stale capture.

The JSON must also name a same-directory, full-page, inert text capture for the complete
installation list, exactly one Configure page for each installation ID, and exactly one expanded
permission-update page for each App with an outstanding request. Each manifest entry binds its safe
basename, canonical query-free GitHub settings URL, aware capture time, exact media type and byte
count, `full_page=true`, and lowercase SHA-256. The capture command reopens each single-link regular
file, recomputes its byte count and digest, and embeds the normalized manifest. All page captures
must fall within the same ten-minute session; the aggregate `captured_at` is the latest page time.
The verifier requires the aggregate time to equal the maximum page time exactly. Prefix each
retained strict-UTF-8 text file with this exact tab-separated first line before non-empty complete
page content (use literal field values and `null` for the list page's installation ID):

```text
source_url=<canonical-url>\tcaptured_at=<canonical-aware-time>\tkind=<kind>\tinstallation_id=<id-or-null>
```

Each displayed `\t` denotes one literal U+0009 tab byte, and the aware time must use the same
canonical UTC `+00:00` form embedded by the manifest normalizer. This binds the retained bytes to
their manifest role and ensures a later genuine recapture cannot
be byte-identical to old evidence. A typed byte count, hash, media type, or header that does not
match the retained file is rejected. The snapshot/report output and its digest sidecar must not
alias the authority JSON or any retained evidence file.

The capture fails unless the complete normalized set exactly equals the reviewed policy. It also
fails while ChatGPT Codex Connector, Claude, GitGuardian, lovable.dev, or Vercel is unsuspended;
GitGuardian is included because its pending-update page displays no repository-permission delta,
which cannot safely prove that every requested change is representable. Never accept a pending
permission request during the authority window. Retain the raw page captures, exact JSON, and
digests in a durable owner-private release-evidence directory outside the public repository; do not
leave the only copies in `%TEMP%` and do not edit a prior record to update its time.

The dispatch actor and the actor who triggers the current attempt must both be the authenticated
capture principal. A rerun records its actual `GITHUB_RUN_ATTEMPT` and
`GITHUB_TRIGGERING_ACTOR`; publication re-queries the Actions source run and rejects any attempt
or triggering-actor mismatch before building or publishing.

Record the successful preflight run ID. Only after separately authorized signed-tag creation,
dispatch `publish-pypi.yml` from that tag and pass the same preflight run ID. The publish workflow
re-verifies the attestation, source workflow, run ID, repository, commit, tag, policy digest, and
all observed controls and exact CUDA run before any build. It also accepts only PyPI's HTTP 404
for the version before build and repeats that fail-closed check immediately before the sole OIDC
publisher action. An existing version or an ambiguous API/network result never reaches the
publisher; there is no `skip-existing` path. For example:

```powershell
gh workflow run publish-pypi.yml --ref $releaseTag `
  -f tag=$releaseTag `
  -f preflight_run_id="<successful-release-preflight-run-id>" `
  -f cuda_run_id=$cudaRunId
```

Candidate-authored Python, tutorial, and JavaScript gates run before any release input is created.
After those gates finish, the workflow refuses reserved release paths, checks the signed tag out a
second time into a clean directory, and revalidates the tag object, exact commit, and `main`
ancestry. It then obtains the exact `Artifact byte reproducibility` run ID from the accepted
external-control snapshot, downloads both clean-build artifact sets and their report from that
run, and rechecks their environment and byte manifests. The distribution is built once from the
second checkout and must be byte-identical to both accepted builds before it can be attested or
uploaded. Missing or expired proof artifacts require a fresh exact-commit reproducibility run;
they are never bypassed.

Before the draft GitHub Release is finalized, the workflow downloads and attestation-verifies the
full preflight evidence, then generates `RELEASE_GOVERNANCE.json` and
`RELEASE_GOVERNANCE.md`. The record binds the release actor, environment reviewer and self-review
setting, tag/commit, preflight and CUDA run IDs, and external-control policy/snapshot digests. If
the live project still uses one operator, the release notes explicitly disclose the absence of
segregation of duties. The draft and its assets must verify before it is published; recovery must
preserve the identical governance disclosure. The normal and recovery paths both fetch PyPI's
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
If the recovery workflow itself fails, never use **Re-run failed jobs**: its fixed writer binds the
handoff artifact to the current whole-workflow attempt and deliberately rejects a successful
verifier reused from an older attempt. Re-audit the partial draft, then either re-run **all jobs**
or issue a fresh dispatch with the same tag and source run ID. Both supported paths re-download
the original source-run artifacts and remain downstream-only; neither can invoke PyPI publication.

1. query every jobs page with `filter=all` and require the original build, attestation, and PyPI
   jobs to have completed successfully exactly once in the named `publish-pypi.yml` run on the
   same tag commit;
2. download only that run's retained `release-distributions` and `release-provenance` artifacts;
3. verify `SHA256SUMS`, exact-workflow GitHub artifact attestations, the exact filename/SHA-256
   inventory returned by PyPI, and every file's PyPI Integrity provenance and publisher identity;
4. create or reuse a draft GitHub Release, reuse only byte-identical existing assets, and upload
   only missing downstream assets;
5. download the draft assets and prove the GitHub distribution hashes equal PyPI before finalizing.

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

The checked-in event/actor guard is defense in depth, not the authority boundary for a
repository-scoped self-hosted runner. A user with repository write access can create a branch
whose workflow targets a revealed runner label and dispatch that branch. Before any nonce is
disclosed through a dispatch or any runner is registered, the repository administrator must
therefore:

1. Capture all effective human collaborators and pending invitations through the authenticated
   repository API. Temporarily remove every non-owner collaborator and require zero pending
   invitations. A retained read collaborator may still be treated as a repository member and
   bypass the `all_external_contributors` fork-workflow approval policy, so a downgrade is not
   sufficient. For the audited personal repository, this means recording and temporarily removing
   `b-urge` at `write`. Coordinate restoration before starting: re-adding a personal-repository
   collaborator can require that user to accept a new invitation. The release-control policy
   rejects a snapshot unless `jemsbhai` is the sole collaborator, retains effective write
   authority, and the invitation list is empty.
2. Keep fork approval at `all_external_contributors`; require no untrusted open pull request and no
   queued or in-progress job targeting any planned nonce-bearing label. Under an owner-authenticated browser
   session, export and review the repository's installed GitHub Apps and automation grants; require
   no non-owner-equivalent principal able to modify workflows and dispatch Actions, or temporarily
   suspend/restrict it. Retain that export and its SHA-256. The collaborator API cannot certify
   this App boundary, so an absent authenticated App-permission record remains a blocker. Cancel
   and archive any unexpected queued job before capacity appears.
3. Generate four distinct, unpredictable 16-lowercase-hex nonces outside GitHub and bind one
   reviewed runner bundle to each topology/edge. Have the owner dispatch the reviewed ref with
   those four exact nonce inputs while no custom runner is registered. Verify the resulting run
   ID, ref, SHA, actor, triggering actor, first attempt, and the complete set of expected queued
   nonce-bearing jobs. Only then directly generate each one-use JIT configuration and register the matching
   clean, isolated runner. Each runner has no default labels, carries only its full nonce-bearing
   runner name as a label, and may accept at most one job. Record the exact runner ID, name/label,
   nonce binding, VM identity, device inventory, and registration time.
   These are administrator/provider controls, not facts proved merely by the workflow's nonce
   syntax. Archive every direct `generate-jitconfig` receipt and bind its runner ID, exact name,
   sole label, creation time, and VM identity to the eventual job record. Search complete historical
   job records before registration to prove that none of the four nonces was used previously; after
   the run, retain zero-runner and zero-resource inventories. Publication uses two additional fresh
   nonces that must be mutually distinct and disjoint from all four accepted CUDA-evidence nonces,
   with the same receipt, historical-search, job-binding, and teardown evidence.
4. Retain sole-writer authority through the accepted PR and final-main CUDA dispatches and the two
   fresh single-GPU publication jobs. On success, failure, or cancellation, delete every
   runner/VM/disk and prove the relevant queues and resource inventories returned to
   zero. After publication/recovery succeeds—or immediately after a failed window is cleaned up—
   re-invite each collaborator at the exact prior permission with a before/after record and retain
   the later acceptance record. Restoration is not complete merely because an invitation exists.
   Only after every runner, VM, disk, queue, and publication/recovery action is complete, restore
   the same suspended App installation IDs to their exact pre-window active state without accepting
   pending updates or changing repository selection or permissions. Capture a separate complete
   all-six restored-state record and raw evidence manifest, and compare it to the pre-window record;
   restored active Apps are not expected to satisfy the intentionally suspended pre-tag policy.

Verify that separate restoration record immediately after the restored capture:

```powershell
python scripts/release_external_controls.py verify-app-restoration `
  --before $preWindowInstalledApps `
  --restored $restoredInstalledApps `
  --output $appRestorationReport `
  --repository jemsbhai/explainiverse `
  --capture-principal jemsbhai
if ($LASTEXITCODE -ne 0) { throw "installed App restoration differs from pre-window state" }
```

The verifier recomputes both manifests from their separately retained raw files, requires every
restored page to postdate its matching pre-window role and remain fresh against verifier time,
rejects reuse of any pre-window page digest, and requires the complete normalized installation
records to be byte-equivalent apart from capture evidence. Neither the report nor its digest
sidecar may alias either authority JSON or any retained page capture.

Do not remove a collaborator merely while capacity is unavailable: without a registered runner
there is no self-hosted execution surface. Removal is a short, coordinated, evidenced release
window, and accepted restoration is part of cleanup.

The workflow's enforceable routing check accepts custom-runner routing only on an owner-triggered first-attempt
`workflow_dispatch` with four distinct exact nonce inputs. Push, pull-request, scheduled, malformed,
reused-nonce, and rerun contexts select a hosted failure reporter before checkout, dependency
installation, or hardware execution; they cannot produce accepted CUDA evidence. Require a fresh
final-main dispatch containing both single-GPU and both two-GPU Torch edges. A green gate means the
checked-in exact node manifest matched collection and all 15 CUDA tests ran with zero skips,
including adapter prediction/gradients, every vector and CAM gradient family, randomisation
success/failure, initialized-device RNG byte restoration, dtype/device placement, and hook cleanup.

The reviewed policy requires each single-GPU job name/label to match
`explainiverse-cuda-single-jit-<nonce>` and each two-GPU job name/label to match
`explainiverse-cuda-two-jit-<nonce>`, where `<nonce>` is the corresponding dispatch input. Do not
apply `self-hosted`, a shared topology label, or any other default/custom label. The preflight
requires four distinct positive runner IDs, four distinct exact JIT names, exact one-label job
records, and the exact device count. It does not itself prove JIT freshness, nonce history, or
ephemeral registration; the repository administrator must retain the administrator/provider
records above establishing that each selected runner is fresh, isolated, one-job, and approved.

As of the 2026-08-28 audit, the repository API reported zero Actions variables and zero registered
runners, zero pending invitations, owner `jemsbhai` at `admin`, and `b-urge` at `write`. There is
therefore no current self-hosted execution surface, but the authority policy is deliberately not
release-ready and `b-urge` must be removed only when an approved, restoration-coordinated runner
window can begin. A repository administrator must provision approved capacity, complete the
authority sequence above, dispatch `cuda-ci.yml` with the four bound nonces on the exact release
commit, and retain a green four-job attempt-1 run before the preflight can succeed. The `tests_cuda`
session hook converts every runtime or collection skip into a failed job and rejects missing,
extra, reordered, or duplicate release nodes when manifest enforcement is enabled. A local
one-GPU diagnostic, even if green, does not substitute for this hosted four-job evidence.
