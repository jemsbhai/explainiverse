# Lambda CUDA release operator

This package is the only production entrypoint for the disposable Lambda GPU
controller. Its default action, `inspect`, is read-only. There is no raw
publication, recovery-workflow, JIT-secret, PyPI-upload, provider, or GitHub
mutation bypass. The entrypoint's existence does not clear a live release
blocker or authorize a tag.

Immutable planning and action-time rediscovery accept only
`gpu_8x_a100_80gb_sxm4` in `us-midwest-1` (`Illinois, USA`) with the exact
`lambda-stack-22-04` image family. There is no regional, GPU-shape, or image-family
fallback.

Run every production action from a fresh detached worktree at the frozen
candidate commit. Never clean, reuse, or whitelist the shared development
checkout: it may contain unrelated user work. All paths below are absolute.

## 1. Build the exact credential-free runtime

The runtime is the official CPython 3.13.15 Windows AMD64 embeddable archive:

- URL: `https://www.python.org/ftp/python/3.13.15/python-3.13.15-embed-amd64.zip`
- bytes: `11009825`
- SHA-256: `d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf`

The tracked Python manifest binds all 34 archive files. The separate site tree
is a custom extraction of exactly four hash-locked wheels: cryptography 50.0.0,
cffi 2.1.1, pycparser 3.0, and pywin32 311. It contains neither pip nor wheel
scripts, RECORD files, bytecode, `.pth` execution, nor an unowned file. The
tracked site manifest binds every file and directory to the exact wheel bytes.

Use a disposable installer environment only to download the locked wheels. It
is not a production runtime:

```powershell
$CleanWorktree = '<absolute-clean-detached-worktree>'
$InstallerPython = '<absolute-disposable-cpython-3.13-python.exe>'
$InstallerVenv = '<absolute-new-disposable-installer-venv>'
$BootstrapWheelhouse = '<absolute-new-bootstrap-pip-wheelhouse>'
$RuntimeWheelhouse = '<absolute-new-four-wheel-runtime-wheelhouse>'
$PythonArchive = '<absolute-python-3.13.15-embed-amd64.zip>'
$OperatorPythonRoot = '<absolute-new-owner-private-python-root>'
$OperatorSiteRoot = '<absolute-new-owner-private-site-root>'
$PythonReceiptDirectory = '<absolute-new-owner-private-python-receipt-directory>'
$SiteReceiptDirectory = '<absolute-new-owner-private-site-receipt-directory>'

& $InstallerPython -I -B -m venv $InstallerVenv
if ($LASTEXITCODE -ne 0) { throw 'installer venv creation failed' }
$Installer = Join-Path $InstallerVenv 'Scripts\python.exe'
New-Item -ItemType Directory -Path $BootstrapWheelhouse -ErrorAction Stop | Out-Null
New-Item -ItemType Directory -Path $RuntimeWheelhouse -ErrorAction Stop | Out-Null
& $Installer -I -B -m pip download --require-hashes --only-binary=:all: `
  --dest $BootstrapWheelhouse `
  -r (Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_operator\requirements-windows-cp313-bootstrap.txt')
if ($LASTEXITCODE -ne 0) { throw 'locked pip wheel download failed' }
& $Installer -I -B -m pip install --no-index --no-compile --require-hashes `
  --find-links $BootstrapWheelhouse `
  -r (Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_operator\requirements-windows-cp313-bootstrap.txt')
if ($LASTEXITCODE -ne 0) { throw 'locked installer pip activation failed' }
& $Installer -I -B -m pip download --require-hashes --only-binary=:all: `
  --dest $RuntimeWheelhouse `
  -r (Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_operator\requirements-windows-cp313.txt')
if ($LASTEXITCODE -ne 0) { throw 'locked runtime wheel download failed' }
```

The two installers are standalone stdlib scripts. They accept only the exact
tracked manifests/archive set, create new directories, harden their DACLs
before writing children, verify every extracted byte, and publish canonical
no-replace receipts into separate owner-private directories. A partial output
is never resumed or deleted by the operator; retain it as failure evidence and
choose wholly new paths.

```powershell
$SetupPython = $Installer
$PythonInstaller = Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_operator\install_windows_python.py'
$SiteInstaller = Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_operator\install_windows_runtime.py'
$PythonManifest = Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_operator\python-runtime-windows-cp313.json'
$SiteManifest = Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_operator\site-packages-windows-cp313.json'

$PythonInstallPublished = (& $SetupPython -I -S -B $PythonInstaller `
  --archive $PythonArchive `
  --manifest $PythonManifest `
  --output $OperatorPythonRoot `
  --receipt-directory $PythonReceiptDirectory) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'pinned Python extraction failed' }
$SiteInstallPublished = (& $SetupPython -I -S -B $SiteInstaller `
  --wheelhouse $RuntimeWheelhouse `
  --manifest $SiteManifest `
  --output $OperatorSiteRoot `
  --receipt-directory $SiteReceiptDirectory) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'pinned site extraction failed' }

$PythonInstallReceipt = [string]$PythonInstallPublished.receipt_path
$PythonInstallReceiptSha = [string]$PythonInstallPublished.receipt_sha256
$SiteInstallReceipt = [string]$SiteInstallPublished.receipt_path
$SiteInstallReceiptSha = [string]$SiteInstallPublished.receipt_sha256
```

The launcher uses the embeddable `python.exe`, never the installer venv's
`Scripts\python.exe`. It starts with `-I -S -B`: environment/user-site/current
directory paths are absent before any repository code is read. The complete
Python and site trees, install receipts, owner-private ACLs, and held path
identities are validated before any third-party site import or third-party
native-module import. This claim
is intentionally scoped: pinned CPython stdlib native modules such as
`_hashlib.pyd` and `_ctypes.pyd` load before the held-tree boundary, and their
bytes belong to the exact official Python manifest.

## 2. Seal the candidate source, then create a clean worktree

This is a maintainer preparation step, not a live release action. First stage
only the reviewed positive candidate allowlist. Do not stage or remove unrelated
user files. The stdlib-only builder reads exact stage-0 index blobs through the
pinned Git executable; dirty worktree bytes and ignored/untracked residue cannot
enter the manifest. It excludes only its own manifest and `preloader.py`, emits
the manifest no-replace, and emits a preloader whose hard-coded manifest digest
is replaced exactly once.

```powershell
$CandidateRepository = '<absolute-development-worktree-with-exact-allowlist-staged>'
$GeneratedManifest = '<absolute-new-generated-source-manifest.json>'
$GeneratedPreloader = '<absolute-new-generated-preloader.py>'
$SourceBuilder = Join-Path $CandidateRepository 'scripts\release_gpu_jit_lambda_operator\build_source_worktree_manifest.py'
& $SetupPython -I -S -B $SourceBuilder `
  --repository-root $CandidateRepository `
  --output $GeneratedManifest `
  --sealed-preloader-output $GeneratedPreloader
if ($LASTEXITCODE -ne 0) { throw 'staged-index source sealing failed' }
```

Review the builder receipt, copy those two exact generated files into their
tracked paths, stage only them, and repeat the builder into two new paths. Both
second-pass files must be byte-identical to the first pass. Commit the exact
candidate, then create a new detached worktree from that commit and require an
empty status. The production preloader rejects every entry outside the manifest
apart from the worktree's `.git` metadata entry, including ignored residue.

```powershell
$CandidateSha = '<reviewed-40-character-candidate-commit>'
$CleanWorktree = '<absolute-new-detached-worktree>'
& 'C:\Program Files\Git\cmd\git.exe' worktree add --detach $CleanWorktree $CandidateSha
if ($LASTEXITCODE -ne 0) { throw 'clean detached worktree creation failed' }
$State = & 'C:\Program Files\Git\cmd\git.exe' -C $CleanWorktree status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $State) { throw 'detached worktree is not clean' }
```

## 3. Use the byte-sealed preloader for every action

Set the exact reviewed preloader SHA from the second builder receipt. The tiny
shim is a reviewed external trust input: verify its fixed digest before decoding
it, and pass that fixed digest rather than a freshly self-derived assertion.
Every child created by the Windows launcher repeats the same shim and preloader
seal. The working directory is the Python install-receipt directory, which must
be pairwise disjoint from the repository, Python tree, site tree, and every
evidence/App directory.

```powershell
$OperatorPython = Join-Path $OperatorPythonRoot 'python.exe'
$Preloader = Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_operator\preloader.py'
$PreloaderSha = '<exact-second-builder-sealed-preloader-sha256>'
$ShimPath = Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_operator\preloader_shim.py'
$ShimSha = '22bb14f6e5fed4e7c5456f62e11569c9e5a0846ad1428854e545b2e4c1c979aa'
$ShimBytes = [IO.File]::ReadAllBytes($ShimPath)
$ObservedShimSha = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($ShimBytes)).ToLowerInvariant()
if ($ObservedShimSha -cne $ShimSha) { throw 'reviewed preloader shim digest mismatch' }
$Shim = [Text.UTF8Encoding]::new($false, $true).GetString($ShimBytes)

function Invoke-ExplainiverseOperator {
  param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('operator','windows-launcher')]
    [string]$Target,
    [Parameter(Mandatory=$true)]
    [string[]]$Arguments
  )
  Push-Location $PythonReceiptDirectory
  try {
    & $OperatorPython -I -S -B -c $Shim $ShimSha $Preloader $PreloaderSha `
      --operator-target $Target `
      --repository-root $CleanWorktree `
      --operator-python-root $OperatorPythonRoot `
      --operator-site-root $OperatorSiteRoot `
      --operator-python-install-receipt $PythonInstallReceipt `
      --operator-python-install-receipt-sha256 $PythonInstallReceiptSha `
      --operator-site-install-receipt $SiteInstallReceipt `
      --operator-site-install-receipt-sha256 $SiteInstallReceiptSha `
      @Arguments
    if ($LASTEXITCODE -ne 0) { throw "operator target $Target failed closed" }
  }
  finally {
    Pop-Location
  }
}

$GitExe = 'C:\Program Files\Git\cmd\git.exe'
$GhExe = 'C:\Program Files\HP\AIStudio\bin\gh.exe'
$SshExe = 'C:\Windows\System32\OpenSSH\ssh.exe'
```

Inspection requires exactly those reviewed paths, bytes, versions, owners, and
Authenticode signer identities. The operator revalidates them at action time.
It also binds the exact Python/site/source manifests, complete clean checkout,
origin URL, local HEAD, and live GitHub ref. Its canonical receipt is written
crash-safely and no-replace into a new owner-private directory; never use shell
redirection or copy that receipt. The action preflight archives the complete
canonical inventory and the complete canonical source-manifest object, not only
their digests, so an accepted-evidence reader can reconstruct every inventory
digest and cross-bind the two deliberately excluded source files.

```powershell
$Phase = 'pull-request'
$PhaseSha = '<exact-pull-request-head-sha>'
$PhaseRef = 'refs/heads/codex/harden-cuda-runner-routing'
$InspectionEvidenceDirectory = '<absolute-new-inspection-evidence-directory>'
$InspectionPublished = Invoke-ExplainiverseOperator -Target operator -Arguments @(
  '--action','inspect',
  '--phase',$Phase,
  '--expected-head-sha',$PhaseSha,
  '--supplied-ref',$PhaseRef,
  '--git-executable',$GitExe,
  '--gh-executable',$GhExe,
  '--ssh-executable',$SshExe,
  '--inspection-evidence-directory',$InspectionEvidenceDirectory
) | ConvertFrom-Json
$InspectionReceipt = [string]$InspectionPublished.inspection_receipt
$InspectionReceiptSha = [string]$InspectionPublished.inspection_receipt_sha256
$InspectionDirectoryReceiptSha = [string]$InspectionPublished.inspection_evidence_directory_receipt.receipt_sha256
```

Repeat inspection for final-main and publication using respectively
`refs/heads/main` and `refs/tags/v0.15.0`; do not reuse a receipt across phases
or candidate SHAs.

## 4. Publish fresh installed-App captures

Create one new inbox per driver phase and one new staging directory per JIT job.
All must be outside the repository and pairwise disjoint from driver/final-main
evidence. These creation actions are non-mutating outside the named local paths:

```powershell
$AppInbox = '<absolute-new-phase-app-inbox>'
$InboxPublished = Invoke-ExplainiverseOperator -Target operator -Arguments @(
  '--action','create-app-inbox',
  '--expected-head-sha',$PhaseSha,
  '--app-capture-inbox',$AppInbox
) | ConvertFrom-Json
$AppInboxReceiptSha = [string]$InboxPublished.receipt.receipt_sha256

$AppStaging = '<absolute-new-one-job-app-staging-directory>'
$StagingPublished = Invoke-ExplainiverseOperator -Target operator -Arguments @(
  '--action','create-app-staging',
  '--expected-head-sha',$PhaseSha,
  '--app-capture-staging',$AppStaging
) | ConvertFrom-Json
$AppStagingReceiptSha = [string]$StagingPublished.receipt.receipt_sha256
```

Using the owner-authenticated browser, write canonical `capture.json` and the
complete raw `pages` child into that held staging directory. Do not stage page
evidence in `%TEMP%` or another loose directory. Then publish it:

```powershell
$FreshCapturePublicationNonce = '<fresh-32-lowercase-hex-nonce>'
$CapturePublished = Invoke-ExplainiverseOperator -Target operator -Arguments @(
  '--action','publish-app-capture',
  '--phase',$Phase,
  '--expected-head-sha',$PhaseSha,
  '--app-capture-inbox',$AppInbox,
  '--app-capture-inbox-receipt-sha256',$AppInboxReceiptSha,
  '--app-capture-staging',$AppStaging,
  '--app-capture-staging-receipt-sha256',$AppStagingReceiptSha,
  '--capture-ordinal','1',
  '--capture-generation','1',
  '--capture-publication-nonce',$FreshCapturePublicationNonce
) | ConvertFrom-Json
```

The publisher validates all raw bytes, authority JSON, freshness, names, and
policy before copying. It writes the immutable bundle first and a canonical
no-replace ready marker last, then rereads the marker. A crash before the marker
is abort/restart-only: retain the orphaned inbox as evidence and begin the whole
phase with a new inbox. Never skip that missing generation. The consumer blocks
on demand once per JIT job, rejects stale/replayed capture or page digests, and
archives accepted pages content-addressably. For fresh captures, use exact
`(ordinal, generation)` pairs `(1, 1), (2, 1)` for pull-request/publication and
`(1, 1), (2, 1), (3, 1), (4, 1)` for final-main. Increment the generation only
to replace a stale capture for the same ordinal; every accepted ordinal resets
the next generation to 1. Successful phase settlement does not silently delete
the source bundles: it archives a canonical final
inventory proving that every consumed stale and accepted source generation is
still present under the held owner-private inbox and that no unobserved residue
exists. Retain that inbox until the release ledger binds the driver archive and
the final inbox-inventory digest; teardown the whole named inbox only under the
separate recorded teardown procedure.

## 5. Execute pull-request, final-main, and publication phases

The native launcher prompts for the Lambda API key without echo and transports
it only through an anonymous inherited HANDLE. It prints the immutable plan,
then requires the operator to retype the exact plan SHA. No live mutation gate
exists before confirmation. The key/confirmation never enters argv,
environment, disk, evidence, or output.

Complete pull-request invocation:

```powershell
$EvidenceDirectory = '<absolute-new-pull-request-evidence-directory>'
Invoke-ExplainiverseOperator -Target windows-launcher -Arguments @(
  '--action','execute',
  '--phase','pull-request',
  '--expected-head-sha',$PhaseSha,
  '--supplied-ref','refs/heads/codex/harden-cuda-runner-routing',
  '--git-executable',$GitExe,
  '--gh-executable',$GhExe,
  '--ssh-executable',$SshExe,
  '--inspection-receipt',$InspectionReceipt,
  '--inspection-receipt-sha256',$InspectionReceiptSha,
  '--inspection-evidence-directory-receipt-sha256',$InspectionDirectoryReceiptSha,
  '--runtime-root',(Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_runtime'),
  '--evidence-directory',$EvidenceDirectory,
  '--app-capture-inbox',$AppInbox,
  '--app-capture-inbox-receipt-sha256',$AppInboxReceiptSha,
  '--ssh-access-key','<absolute-existing-owner-private-lambda-ssh-key>',
  '--ssh-key-name','<exact-existing-lambda-key-name>',
  '--image-id','<fresh-action-time-image-id>',
  '--controller-public-ipv4-cidr','<fresh-controller-public-ipv4/32>',
  '--lifecycle-nonce','<fresh-32-lowercase-hex-lifecycle-nonce>',
  '--plan-lifetime-seconds','3600'
)
```

For final-main, use the same complete shape with `--phase final-main`, the exact
final main SHA, `--supplied-ref refs/heads/main`, and wholly new inspection,
inbox, evidence, key/capacity discovery, and lifecycle values. Publish four
fresh App generations while it waits.

Publication is permitted only after the closed final-main journal loader proves
the accepted four-job run. Complete publication invocation:

```powershell
Invoke-ExplainiverseOperator -Target windows-launcher -Arguments @(
  '--action','execute',
  '--phase','publication',
  '--expected-head-sha',$FinalMainSha,
  '--supplied-ref','refs/tags/v0.15.0',
  '--git-executable',$GitExe,
  '--gh-executable',$GhExe,
  '--ssh-executable',$SshExe,
  '--inspection-receipt',$PublicationInspectionReceipt,
  '--inspection-receipt-sha256',$PublicationInspectionReceiptSha,
  '--inspection-evidence-directory-receipt-sha256',$PublicationInspectionDirectoryReceiptSha,
  '--runtime-root',(Join-Path $CleanWorktree 'scripts\release_gpu_jit_lambda_runtime'),
  '--evidence-directory',$NewPublicationEvidenceDirectory,
  '--app-capture-inbox',$PublicationAppInbox,
  '--app-capture-inbox-receipt-sha256',$PublicationAppInboxReceiptSha,
  '--ssh-access-key',$ExistingLambdaSshPrivateKey,
  '--ssh-key-name',$ExistingLambdaSshKeyName,
  '--image-id',$ActionTimeImageId,
  '--controller-public-ipv4-cidr',$ControllerPublicIp32,
  '--lifecycle-nonce',$FreshPublicationLifecycleNonce,
  '--plan-lifetime-seconds','3600',
  '--prior-accepted-cuda-runner-nonce',$FinalSingleMinimumNonce,
  '--prior-accepted-cuda-runner-nonce',$FinalSingleLatestNonce,
  '--prior-accepted-cuda-runner-nonce',$FinalTwoMinimumNonce,
  '--prior-accepted-cuda-runner-nonce',$FinalTwoLatestNonce,
  '--preflight-run-id',$AcceptedPreflightRunId,
  '--cuda-run-id',$AcceptedFinalMainCudaRunId,
  '--final-main-evidence-directory',$FinalMainEvidenceDirectory,
  '--final-main-evidence-receipt-sha256',$FinalMainEvidenceReceiptSha,
  '--final-main-plan-sha256',$FinalMainPlanSha,
  '--final-main-journal-sha256',$FinalMainTerminalJournalSha
)
```

The only accepted `v0.15.0` publication request has
`stage_recovery_drill=true`; that exact contract lives in the sealed controller.
There is no raw `gh workflow run`, web button, direct API, or rerun fallback.

## 6. Abort/recover a crashed live phase

`execute` catches every `BaseException`, calls the exact abort path, and then
unconditionally closes the journal/receipts and zeroizes/cleans all owned
inputs. After process loss, invoke cleanup-only recovery through the launcher;
it cannot execute SSH jobs:

```powershell
Invoke-ExplainiverseOperator -Target windows-launcher -Arguments @(
  '--action','resume-abort',
  '--phase',$InterruptedPhase,
  '--expected-head-sha',$InterruptedHeadSha,
  '--supplied-ref',$ExactInterruptedRef,
  '--git-executable',$GitExe,
  '--gh-executable',$GhExe,
  '--ssh-executable',$SshExe,
  '--inspection-receipt',$InterruptedInspectionReceipt,
  '--inspection-receipt-sha256',$InterruptedInspectionReceiptSha,
  '--inspection-evidence-directory-receipt-sha256',$InterruptedInspectionDirectoryReceiptSha,
  '--evidence-directory',$InterruptedEvidenceDirectory,
  '--evidence-directory-receipt-sha256',$InterruptedEvidenceReceiptSha,
  '--confirm-plan-sha256',$InterruptedPlanSha
)
```

The phase/ref pair must be the original exact pair. The loader reconstructs the
immutable plan from the journal, revalidates source/live/executable posture, and
permits cleanup only.

## 7. Dispatch staged GitHub Release recovery

After the sole staged publish run has uploaded once to PyPI and intentionally
failed before GitHub Release finalization, use only the same restored
publication journal:

```powershell
Invoke-ExplainiverseOperator -Target operator -Arguments @(
  '--action','dispatch-release-recovery',
  '--phase','publication',
  '--expected-head-sha',$FinalMainSha,
  '--supplied-ref','refs/tags/v0.15.0',
  '--git-executable',$GitExe,
  '--gh-executable',$GhExe,
  '--ssh-executable',$SshExe,
  '--inspection-receipt',$PublicationInspectionReceipt,
  '--inspection-receipt-sha256',$PublicationInspectionReceiptSha,
  '--inspection-evidence-directory-receipt-sha256',$PublicationInspectionDirectoryReceiptSha,
  '--evidence-directory',$PublicationEvidenceDirectory,
  '--evidence-directory-receipt-sha256',$PublicationEvidenceReceiptSha,
  '--confirm-plan-sha256',$PublicationPlanSha,
  '--publication-journal-sha256',$PublicationLifecycleRestoredJournalSha,
  '--source-run-id',$StagedPublishRunId
)
```

The sealed publication-source loader proves exact plan/source-run/two-job/phase
settlement and `lifecycle-restored` continuity before any recovery decision. A
durable pending intent permits observation-only reconciliation; a controller
receipt awaiting its operator settlement is completed locally with zero GitHub
call. A fresh POST is possible only from the exact loader-sealed complete tail,
and prior retries require exact archived terminal-failure continuity. Never use
`gh workflow run`, the web UI, direct API dispatch, or any rerun button.

The recovery workflow itself hard-fails unless the run attempt is exactly one,
archives the complete retained workflow API history without the filtered-search
1,000-result cap before checkout, locally validates its dispatch rows, and
requires an identical fresh observation as the last gate before every Release
create, asset upload, or finalization mutation.
That history gate cannot cryptographically distinguish a direct call by the same
trusted repository owner with identical inputs; it enforces only the declared
attempt/actor/nonce/history contract. The durable operator intent remains the
required procedural provenance.

The returned dispatch-settled receipt is not recovery success. It deliberately
leaves `workflow_completion_verified=false` and `no_republish_verified=false`.
Before B03 closes, independently archive read-only terminal run/job/artifact
evidence, immutable GitHub Release/assets, and the PyPI file inventory proving
there was no second upload.

## Windows launcher declaration scope

The parent launcher places a canonical self-digested declaration on the child
argv. It is deliberately unauthenticated and non-authoritative: a direct caller
can forge it. Evidence records
`parent_provenance_authenticated=false` and
`security_authority_derived_from_declaration=false`. The declaration is useful
only for matching public parent metadata. All authority comes from the child's
own repeated byte-sealed preloader, exact runtime/source/resource validation,
and direct anonymous-pipe HANDLE cardinality/type/noninheritability checks.
