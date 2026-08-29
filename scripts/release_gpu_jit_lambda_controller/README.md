# Trusted-local Lambda GitHub controller

This package is the only component in the disposable GPU path allowed to call
GitHub. It is production-callable through `ReleaseGpuController` and the
shell-free `GhCliTransport`; all tests inject inert transports. It never runs
`gh auth token`, never sends a GitHub API credential or Lambda API key to the
remote host, and never places a nonce, GPU UUID, runtime plan, or encoded JIT
configuration in SSH argv, environment variables, or a shell command.

The supported sequence is deliberately narrow:

1. Dispatch `cuda-ci.yml` at the exact PR branch with four fresh 16-hex
   nonces, service only its two protected single-GPU jobs, prove both as
   first-attempt 15/15 zero-skip Actions successes, then cancel and settle the
   still-unserviced two-GPU work without a rerun.
2. Re-observe `main`, dispatch a fresh first-attempt run, and service all four
   single/two, minimum/latest jobs sequentially on the same eight-A100 host.
3. After the signed annotated `v0.15.0` tag is GitHub-verified and peeled to
   that final main SHA, validate positive preflight/final-CUDA runs and
   dispatch `publish-pypi.yml` with `stage_recovery_drill=true`; service only
   its two release CUDA jobs.

Before each JIT request, the controller revalidates sole-owner authority, zero
invitations/runners/repository variables, a fresh (never replayed) trusted
suspended-App capture and its raw page digests,
all attempts/pages of nonce history, exact queued job/name/sole label, the
repository runner download metadata, and the fixed remote readiness receipts.
It posts `generate-jitconfig` with `runner_group_id=1` and does not invent a
runner-group read. The exact 201 body must contain only `runner` and
`encoded_jit_config`; the new runner must be offline, not busy, have pre-start
OS `unknown`, and return only the requested custom label. Any generated but
rejected runner is explicitly deleted and zero inventory is proven.

Remote readiness is always two-stage: fixed no-PTY cloud-init readiness first,
then the installed runtime's fixed `probe-host`. Per job, the live adapter's
fixed run binding carries an `EXJIT01` framed public canonical plan plus the
JIT secret on SSH stdin. The secret buffer is destroyed locally. The remote
receipt is accepted only for container/GPU/cleanup facts; direct Actions
job/check/step/log records must independently prove exact runner binding,
15 passed, zero skipped, and zero registered repository runners before the
next identity.

`LiveReleaseDriver` is the supported high-level lifecycle boundary. It binds
the live provider's mandatory write-ahead mutation callback to one held,
owner-private `EvidenceJournal` before the first observation; provisions the
ruleset and host; establishes both SSH readiness receipts; runs exactly one
phase; and terminates the host, retires only its exact runners, deletes the
ruleset, restores the global provider state, and proves stable zero inventory.
`resume_for_abort` reopens an interrupted hash chain and reconciles recorded
provider/GitHub intents without replaying a mutation. Cleanup continues even
when ordinary evidence archival fails, using the preallocated provider-intent
reserve for the mutation boundary.

The App-capture supplier is called on demand for every JIT job (two, four, or
two times for PR, final-main, and publication). A capture loaded once at
startup is not supported. After final-main cleanup, publication must reopen
the held evidence directory and call
`EvidenceJournal.load_final_main_acceptance` with the exact final plan and
journal-anchor digests. The in-memory acceptance returned by the final-main
phase is archival material only and is deliberately rejected by publication.

The separate `release_gpu_jit_lambda_operator` entrypoint owns local secret
FDs, environment scrubbing, executable/source inventory, secure evidence
directory creation/reopen, fresh App-capture intake, and normal or crash-abort
driver construction. Browser-confirmed App restriction/restoration, Lambda
credential setup/removal, merge, non-GPU gates, tag/release publication, and
ledger updates remain explicit release-operator actions outside this package.

After the staged OIDC run has failed at its intentional post-PyPI drill step,
`dispatch_release_recovery` is the only supported recovery-workflow dispatch.
It revalidates the signed tag and the exact completed first-attempt source run,
including the successful build/attest/PyPI jobs and failed staged-drill step;
requires a fresh 16-hex request nonce; archives the complete request before
POST; and identifies the resulting run by the workflow's deterministic
tag/source-run/nonce `run-name`. A lost response is reconciled only by fresh,
complete run history. Absence is never treated as proof that the POST did not
apply, and `reconcile_release_recovery_dispatch` never sends a mutation.
Earlier matching recovery runs must be terminal failures, a successful prior
recovery stops the flow, and a request nonce can never be reused.
