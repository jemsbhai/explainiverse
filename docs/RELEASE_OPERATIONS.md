# Stable release control snapshot and recovery drill

This runbook describes the reviewed automation and operator boundaries; it is not evidence that
mutable service settings or CUDA hardware are already configured. A stable release remains blocked
until every preflight and hardware job is green and the PyPI project owner directly verifies the
Trusted Publisher.

## Release-window entry conditions

Finish and settle the final-main automatic checks, the four-job final-main CUDA dispatch, and the
accepted reproducibility artifacts first. Only then may the PyPI project owner register and archive
the exact Trusted Publisher for project `explainiverse`, owner `jemsbhai`, repository
`explainiverse`, workflow `publish-pypi.yml`, and environment `pypi`. In the same preparation phase,
register the already-proved SSH public key on GitHub as a **signing** key, archive its fingerprint
and account setting, and prove the corresponding private key can create and locally verify an
annotated test tag without pushing it. Keep both repository and `pypi` environment PyPI secret
inventories empty. Complete these B01 prerequisites before beginning the final installed-App
capture and B02's 30-minute pre-tag clock.

The assembled, tested production entrypoint is
`scripts.release_gpu_jit_lambda_operator`; its exact invocation and evidence contracts are in that
package's `README.md`. It reloads the final-main journal by terminal SHA-256, obtains one fresh raw
installed-App capture per JIT through owner-private receipts, repeats action-time authority and
run/artifact checks, reconciles ambiguous mutations without replay, and drives cleanup. Its
default action is non-mutating inspection, and publication accepts only a final-main object loaded
from the closed journal. This code boundary does not make the current release a GO: every live
preflight, hardware, signing, Trusted Publisher, and restoration record below must still exist and
pass. Never substitute a direct GitHub CLI publication-workflow dispatch; after an immutable tag
exists, a duplicate or unbound publication dispatch cannot be repaired by moving or deleting the
tag.

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
$expectedRepository = "jemsbhai/explainiverse"
git fetch --no-tags origin '+refs/heads/main:refs/remotes/origin/main'
if ($LASTEXITCODE -ne 0) { throw "could not refresh origin/main" }
$releaseCommit = git rev-parse refs/remotes/origin/main
$liveMainCommit = gh api "repos/$expectedRepository/commits/main" --jq .sha
if ($releaseCommit -notmatch '^[0-9a-f]{40}$' -or $liveMainCommit -ne $releaseCommit) {
  throw "frozen local origin/main differs from the live GitHub main commit"
}
$cudaRunId = "<successful-cuda-ci-run-id>"
$snapshot = Join-Path $env:TEMP "explainiverse-$releaseTag-controls.json"
$installedAppAuthority = Join-Path $env:TEMP "explainiverse-$releaseTag-installed-apps.json"
$previousGhTokenWasSet = Test-Path Env:GH_TOKEN
$previousGhToken = $env:GH_TOKEN
try {
  $env:GH_TOKEN = gh auth token
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($env:GH_TOKEN)) {
    throw "could not obtain the authenticated gh token"
  }
  python scripts/release_external_controls.py capture `
    --policy .github/release-control-policy.json `
    --output $snapshot `
    --repository $expectedRepository `
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
  if ($LASTEXITCODE -ne 0) { throw "release-preflight dispatch failed or was ambiguous" }
}
finally {
  if ($previousGhTokenWasSet) {
    $env:GH_TOKEN = $previousGhToken
  }
  else {
    Remove-Item Env:GH_TOKEN -ErrorAction SilentlyContinue
  }
}
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

## Immutable signed-tag boundary

Record the successful preflight run and artifact IDs/digests, then query the Actions API again and
require both preflight jobs to be successful on the frozen commit. Before creating a tag, require
the accepted snapshot and installed-App evidence to remain within the 30-minute publication
freshness limit. Also query and download all three artifacts from the exact accepted
`Artifact reproducibility` run: its two independent-build artifacts are retained for only 14 days.
An expired artifact, a changed run attempt, or a different reproducibility run requires a fresh
exact-main push run and a fresh pre-tag capture **before** any tag exists. It is not recoverable by
rerunning or rebinding evidence after the immutable tag is pushed.

At the irreversible boundary, re-read live `main` and require the same 40-character commit; require
exact HTTP 404 for PyPI `explainiverse/0.15.0`, the Git tag ref, and the GitHub Release; require the
Trusted Publisher and GitHub signing-key records to remain exact; and require the reviewed
publication controller to be ready to dispatch immediately. Do not push the tag merely to test
these conditions. In a clean release checkout, create, inspect, verify, and push exactly one signed
annotated tag:

```powershell
$releaseTag = "v0.15.0"
$releaseCommit = "<frozen-40-character-final-main-sha>"
$signingKey = "<absolute-owner-private-ssh-signing-key-path>"
$allowedSigners = "<absolute-owner-private-allowed-signers-file>"

git fetch --no-tags origin '+refs/heads/main:refs/remotes/origin/main'
if ($LASTEXITCODE -ne 0) { throw "could not refresh origin/main at the tag boundary" }
$fetchedMainCommit = git rev-parse refs/remotes/origin/main
$liveMainCommit = gh api "repos/jemsbhai/explainiverse/commits/main" --jq .sha
if ($fetchedMainCommit -ne $releaseCommit -or $liveMainCommit -ne $releaseCommit) {
  throw "final main changed after the accepted pre-tag capture"
}

$worktreeState = git status --porcelain --untracked-files=all
if ($LASTEXITCODE -ne 0) { throw "release checkout could not be inspected" }
if ($worktreeState) { throw "release checkout is not clean" }
git show-ref --verify --quiet "refs/tags/$releaseTag"
if ($LASTEXITCODE -eq 0) { throw "release tag already exists locally" }
if ($LASTEXITCODE -ne 1) { throw "local tag absence check was ambiguous" }

function Assert-ExactGitHub404([string]$Endpoint, [string]$Label) {
  $response = @(gh api --include $Endpoint 2>&1 | ForEach-Object { "$_" })
  $apiExit = $LASTEXITCODE
  $statusLines = @($response | Where-Object { $_ -match '^HTTP/\S+\s+\d{3}(?:\s|$)' })
  if ($apiExit -eq 0 -or $statusLines.Count -ne 1 -or
      $statusLines[0] -notmatch '^HTTP/\S+\s+404(?:\s|$)') {
    throw "$Label exists or its absence could not be proved by one exact HTTP 404"
  }
}
Assert-ExactGitHub404 `
  "repos/jemsbhai/explainiverse/git/ref/tags/$releaseTag" "remote release tag"
Assert-ExactGitHub404 `
  "repos/jemsbhai/explainiverse/releases/tags/$releaseTag" "GitHub Release"
python scripts/check_pypi_version_absent.py --project explainiverse --tag $releaseTag
if ($LASTEXITCODE -ne 0) { throw "PyPI version is present or could not be checked" }

git -c gpg.format=ssh -c user.signingkey="$signingKey" tag --sign `
  --message "Explainiverse $releaseTag" $releaseTag $releaseCommit
if ($LASTEXITCODE -ne 0) { throw "signed annotated tag creation failed" }
if ((git cat-file -t $releaseTag) -ne "tag") { throw "release ref is not an annotated tag" }
if ((git rev-parse "$($releaseTag)^{commit}") -ne $releaseCommit) {
  throw "release tag does not peel to the frozen main commit"
}
git -c gpg.format=ssh -c gpg.ssh.allowedSignersFile="$allowedSigners" verify-tag $releaseTag
if ($LASTEXITCODE -ne 0) { throw "local SSH tag verification failed" }
git push origin "refs/tags/$releaseTag:refs/tags/$releaseTag"
if ($LASTEXITCODE -ne 0) { throw "release tag push failed or was ambiguous; re-read, do not retry" }

$tagRef = gh api "repos/jemsbhai/explainiverse/git/ref/tags/$releaseTag" | ConvertFrom-Json
if ($tagRef.object.type -ne "tag" -or $tagRef.object.sha -notmatch '^[0-9a-f]{40}$') {
  throw "GitHub release ref is not an annotated tag object"
}
$tagObject = gh api "repos/jemsbhai/explainiverse/git/tags/$($tagRef.object.sha)" |
  ConvertFrom-Json
if ($tagObject.verification.verified -ne $true -or
    $tagObject.object.type -ne "commit" -or
    $tagObject.object.sha -ne $releaseCommit) {
  throw "GitHub did not verify the exact signed tag; the immutable tag is an external blocker"
}
```

The local allowed-signers file must bind the intended maintainer identity to the same public key
registered on GitHub; it is verification input, not a substitute for GitHub's tag-object
`verification.verified` record. The `v*` ruleset permits no update, deletion, or bypass. If the tag
push response is lost, re-read the exact ref and tag object; never replay the push blindly. If the
remote object is anything other than the exact verified tag above, stop.

Only after this separately authorized signed-tag creation may the publication controller dispatch
`publish-pypi.yml` from that tag. It must pass the same preflight and CUDA run IDs. The workflow
re-verifies the attestation, source workflow, run ID, repository, commit, tag, policy digest, and
all observed controls and exact CUDA run before any build. It also accepts only PyPI's HTTP 404
for the version before build and repeats that fail-closed check immediately before the sole OIDC
publisher action. An existing version or an ambiguous API/network result never reaches the
publisher; there is no `skip-existing` path.

The reviewed Lambda operator constructs exactly this request after its immutable plan is confirmed;
this is an input contract, **not** an instruction to dispatch it directly:

```text
workflow: publish-pypi.yml
ref: v0.15.0
tag: v0.15.0
preflight_run_id: <successful-release-preflight-run-id>
cuda_run_id: <accepted-final-main-cuda-run-id>
single_minimum_runner_nonce: <fresh-16-lowercase-hex-nonce>
single_latest_runner_nonce: <different-fresh-16-lowercase-hex-nonce>
stage_recovery_drill: true
```

Both publication nonces must be distinct, previously unused, and disjoint from all four nonces in
the accepted final-main CUDA run. For `v0.15.0`, omitting `stage_recovery_drill=true` is a rejected
front-door request rather than a normal publication path. The controller must journal the request
before mutation, reconcile a missing/ambiguous dispatch response by read-only discovery, prove the
exact two release-CUDA jobs are queued before creating one-use runners, and never replay the POST
unless it has proved that the first request did not take effect.

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

If PyPI exposes either file but the source publish job did not finish with one unambiguous
`completed/success` execution, stop. Recovery deliberately rejects that state because it cannot
prove the upload completed exactly once. Treat a lost success response or a publish job that failed
after bytes reached PyPI as an external publication incident: archive PyPI JSON and Integrity
records, do not rerun OIDC, and do not represent a manual decision as the staged recovery drill.

Never dispatch `recover-github-release.yml` directly through GitHub CLI, the web UI, an API call,
or a rerun button. Use the frozen Lambda operator's explicit
`--action dispatch-release-recovery` contract documented in
`scripts/release_gpu_jit_lambda_operator/README.md`, bound to the exact immutable publication plan,
tag commit, failed source run ID, and caller-supplied `lifecycle-restored` publication-journal
SHA-256. A sealed loader first proves the exact two publication CUDA jobs, phase settlement, and
zero provider resources/runners at that anchor and journals its receipt idempotently. Only then
does the operator check the journal for a durable pending intent
before making any decision. A pending intent permits only observation-only run-history
reconciliation; with no pending intent, the controller records the exact
`tag`/`source_run_id`/`require_staged_drill=true` inputs, request nonce, and pre-dispatch run IDs
before its single POST. Response loss is reconciled from that record and is never blindly retried.
The workflow itself rejects any execution whose ref is not the exact release tag or whose
`GITHUB_SHA` differs from the checked-out tag commit.

Both workflow jobs hard-fail unless `GITHUB_RUN_ATTEMPT` is exactly `1`; the write-authorized job
uses an `always()` entry guard and also requires the verification job to have succeeded, so a
failed verification cannot turn the mutation job into a harmless-looking dependency skip. Before
checkout, verification paginates the complete retained workflow API history without GitHub's
1,000-result filtered-search cap, locally validates the `workflow_dispatch` history, archives a canonical
snapshot, and accepts exactly the current owner-triggered tag/source/nonce run plus unique prior
first-attempt terminal failures for that tag/source. It rejects a prior success, reused nonce or run
ID, foreign matching title/context, and every other active recovery run. The write-authorized job
repeats the complete paginated observation and requires the normalized bytes to match as the last
gate before every Release create, asset upload, or finalization mutation.

This history check cannot cryptographically distinguish a direct call by the same trusted repository owner
from the controller-mediated call with identical declared inputs. It enforces the
declared first-attempt, actor, nonce, and history contract as defense in depth; the durable operator
intent and journal remain the procedural provenance required by this runbook.

If the recovery workflow itself fails, never use **Re-run failed jobs** or **Re-run all jobs**.
Reopen the exact evidence journal, reconcile any pending dispatch intent first, and capture the
prior run's terminal failure and partial draft state. Only after that exact run is unambiguously
terminal may a separately authorized operator invocation record a fresh nonce and durable intent;
there is no raw dispatch fallback. Each accepted run re-downloads the original source-run
artifacts and remains downstream-only; it cannot invoke PyPI publication.
If the prior attempt already finalized the Release before losing its response or evidence upload,
the retry accepts it only when the service reports an immutable, non-draft Release containing the
complete exact approved asset bundle and governance disclosure. In that state it performs no
create, upload, or finalize mutation and completes the retained verification evidence. Any
different final Release is an external blocker, not a reason to replace an asset or republish.

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
URLs, and both attestations as the drill record. The operator's dispatch-settled receipt is not a
recovery-success record: it explicitly leaves workflow completion and no-republish proof false.
Before closing B03, use read-only API observations to bind the terminal recovery run and every job,
the retained recovery artifact and digest, the immutable GitHub Release and exact assets, and the
PyPI file inventory proving no second upload into the evidence journal and both release ledgers.

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

The production Lambda operator has one exact provider target:
`gpu_8x_a100_80gb_sxm4` in `us-midwest-1` (`Illinois, USA`) with the
`lambda-stack-22-04` image family. Fresh discovery and action-time revalidation
must both prove that exact target; another region, shape, or image family is a
hard stop rather than a fallback.

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
   clean, isolated runner. Request only its full nonce-bearing runner name as the custom label; the
   repository JIT endpoint has no `no_default_labels` request field. Inspect the exact `201`
   response before starting the runner and require its returned label set to contain only that
   requested nonce label. If GitHub returns a default or other extra label, delete the unused
   registration, capture the zero-runner inventory, and stop rather than claiming suppression.
   Each accepted runner may execute at most one job. Record the exact runner ID, name, returned
   label set, nonce binding, VM identity, device inventory, and registration time.
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
request `self-hosted`, a shared topology label, or any other default/custom label, and do not infer
default-label suppression from the request: the exact JIT response must prove the sole returned
label before execution. The preflight
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
