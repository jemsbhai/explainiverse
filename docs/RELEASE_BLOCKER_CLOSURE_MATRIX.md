# Stable-release blocker closure matrix

This is the current closure ledger for B01-B11. Its baseline was re-audited from clean commit
`dd76815c79076c43d88568ae10f43be7bb546d9c` on 2026-08-11; the predecessor rehearsal and both
immutable unpublished release attempts remain as dated history below. The current decision is the
authorized `v0.15.2`/PR #7 roll-forward. A repository guard can make future evidence falsifiable,
but it cannot close a live-service, hosted-runner, hardware, or
human-review blocker. `BLOCKED` therefore remains the only honest state until the acceptance
column is satisfied with direct evidence from the exact candidate commit. B10 governs a future
bound widening and B11 is an intentionally retained P2 quarantine, so their blocked states are
not stable-release failures while those surfaces remain unclaimed.

The default stable-release recommendation is **NO** while every P0 external/live-main acceptance
row remains open. There is one current exception: the maintainer authorized
`EXPLAINIVERSE-v0.15.2-CPU-ONLY` on 2026-09-04 for PR #7 and release `v0.15.2`. That release may
proceed only with every non-CUDA gate and publication control green and with immutable disclosure
that CUDA hardware evidence was not collected and CUDA release verification is false. The
exception cannot close B04, support a CUDA claim, or authorize another tag.

The 2026-08-11 evidence below predates this exception and remains a historical record of that
audit. Its statements that no merge, tag, publication, release creation, settings mutation, or
GPU representation had been authorized describe that audit, not the later one-release decision.

## 2026-09-03 historical v0.15.0 exception authorization

- The versioned policy keeps the 23-context baseline. For `v0.15.0` plus exception ID
  `EXPLAINIVERSE-v0.15.0-CPU-ONLY`, it omits only the two single-GPU exact-commit check-run
  results and lists all four release CUDA jobs as not run; it never omits their branch-protection
  contexts or GitHub Actions app bindings from the administrator capture.
- PR #5 predecessor head `439fa420601fd386b2093c2077707a1716745b98` demonstrated all 21
  non-CUDA contexts successfully. The two CUDA contexts failed closed because the repository has
  no approved runner routing; those failures are not GPU evidence. The exception implementation
  still requires a fresh green run on its final PR head and, after merge, on the exact `main` SHA.
- Merge requires a narrow required-check change rather than a broad administrator bypass. The
  21-context state exists only long enough to merge PR #5. Both app-bound CUDA contexts are
  restored immediately after merge; then the operator waits for the 21 non-CUDA `push`/`main`
  results and performs administrator capture/preflight while all 23 protections are live.
- The preflight snapshot, publish verification, and `RELEASE_GOVERNANCE.json`/`.md` bind the
  exception ID, tag/version, GitHub PR #5's actual merge commit, authorizer/date/reason, every
  omission, the restored 23-context protection, and both false CUDA evidence fields. Its scope
  ended with `v0.15.0`; any later exception requires a separate, explicit authorization.

The exception was exercised through signed tag creation, but publication run `33891048942`
stopped during SBOM generation before artifact upload, distribution attestation, PyPI publication,
or GitHub Release creation. The signed immutable `v0.15.0` tag remains unchanged and is not a
published release.

## 2026-09-04 historical v0.15.1 roll-forward authorization

- `EXPLAINIVERSE-v0.15.1-CPU-ONLY` is bound to tag/package version `v0.15.1`/`0.15.1`, PR #6,
  maintainer `jemsbhai`, and the 2026-09-04 authorization. It retains the same exact two omitted
  single-GPU contexts, four omitted release CUDA jobs, and false hardware/verification fields.
- The roll-forward fixes the deterministic SBOM metadata failure and GitHub's removal of
  `merge_commit_sha` from current pull-request payloads. It does not broaden any CUDA claim or
  waive any non-CUDA check, reproducibility proof, signing, attestation, OIDC, or immutable-Release
  requirement.
- The 21-context protection state may exist only long enough to merge PR #6. Both CUDA contexts
  must then be restored with Actions app ID `15368` before exact-main checks, a fresh administrator
  capture, and preflight for `v0.15.1`.

That path produced signed immutable tag `v0.15.1`. Publication run `33901507340` built
successfully and retained workflow artifacts, including the repaired SBOM, but GitHub skipped
distribution attestation, PyPI publication, and GitHub Release creation because a skipped ancestor
condition propagated to those jobs. The tag remains unchanged; `0.15.1` is not on PyPI and has no
GitHub Release.

## 2026-09-04 v0.15.2 roll-forward authorization

- `EXPLAINIVERSE-v0.15.2-CPU-ONLY` is bound to tag/package version `v0.15.2`/`0.15.2`, PR #7,
  maintainer `jemsbhai`, and the 2026-09-04 authorization. It retains the same exact two omitted
  single-GPU contexts, four omitted release CUDA jobs, and false hardware/verification fields.
- The roll-forward adds explicit skipped-ancestor bridges to the distribution-attestation, PyPI,
  and GitHub Release jobs. Each evaluates after an intentional skip but still requires
  `!cancelled()` and exact success from its direct upstream job; upstream failure or cancellation
  cannot proceed.
- The 21-context protection state may exist only long enough to merge PR #7. Both CUDA contexts
  must then be restored with Actions app ID `15368` before exact-main checks, a fresh administrator
  capture, and preflight for `v0.15.2`.

## Release-attempt observations

- On 2026-09-04, the owner authenticated to PyPI and registered the exact Trusted Publisher for
  owner `jemsbhai`, repository `explainiverse`, workflow `publish-pypi.yml`, and environment
  `pypi`. GitHub preflight run `33890660777` accepted all 23 restored branch contexts and the
  original exception on PR #5's merge. Signed-tag publish run `33891048942` then passed the full
  Python, JavaScript, and reproducibility gates but failed deterministically while CycloneDX read
  hybrid Poetry-2/PEP-621 metadata. It produced no artifacts and never reached attestation, the
  protected PyPI environment, publication, or GitHub Release creation. Later run `33901507340`
  for signed tag `v0.15.1` built successfully and retained workflow artifacts, including the
  repaired SBOM, but skipped distribution attestation, PyPI publication, and GitHub Release
  creation through GitHub's skipped-ancestor condition propagation. Signed tags `v0.15.0` and
  `v0.15.1` remain unchanged; neither version is on PyPI or has a GitHub Release.

### Historical 2026-08-11 control observations

- The authenticated GitHub capture was made by repository administrator `jemsbhai` at
  `2026-08-11T17:16:13.612560+00:00` against `origin/main`
  `49e962c090e90e62f315837067e5adc3e3f04d1c` and planned tag `v0.15.0`. Snapshot SHA-256:
  `e2f704a4996056de8dda2c7977f55c4dd55135c7054b6ecbf0ac721c9121fc1e`; reviewed-policy
  SHA-256: `e6d8e4de111f5efdcecfef726e9c4f4a526eebd0d16520c426d9405aca03a443`.
- The capture is rejected with 19 violations. Live `main` protection has 10 required contexts,
  while the reviewed policy has 23. All 10 live contexts are bound to GitHub Actions app ID
  `15368`; 16 policy checks have no successful exact-SHA check run from that provider. Only the
  old Deploy Demo, JS CI, and Python CI workflow files are present in the `main` branch tree.
  Draft PR #2 contains the audited branch, but it is not landed or live on `main`. Repository and
  `pypi` environment secret-name inventories each have count zero. The repository-level
  immutable Releases control is disabled and is not enforced by the owner.
- Separate authenticated read-only API observations at `2026-08-11T17:17:51.9860715Z` report
  zero Actions variables and zero registered runners. Six workflows are now registered at the
  repository level because the PR executed three new workflow files, but those files are still
  absent from the `main` tree.
- Draft PR #2 head `9408f5c63c8accab2611658422b768990ead42d5` registered all 23 policy check
  names under GitHub Actions app ID `15368`; Actions executed merge checkout
  `0d5ca6996e548e6ffb4d89e83aac7fc524ba5dbd`. Python run `31513521028`, dependency run
  `31513520975`, JavaScript run `31513521103`, and artifact run `31513521096` attempt 2
  succeeded. All 21 non-CUDA policy contexts succeeded, but their
  reviewed acceptance event is `push` on `main`. Both required CUDA contexts terminated in
  failure at the routing reporter; their acceptance requires `workflow_dispatch` on `main` and
  real approved capacity. The PR results diagnose the repository but do not satisfy live-main
  release acceptance.
- PyPI still serves `explainiverse` 0.14.0. Its wheel and sdist SHA-256 values are
  `b1b98dfdfc0acbc8dc2113d8db87d40ae9cec2f958ed25b00bc6e30d43db41d4` and
  `e2ab525f720d9970f25c307be84b9a5a6bb5feb612a4457ba9d72925cf2af68b`; both Integrity API
  provenance requests return HTTP 404. Version 0.15.0 is absent. The authenticated owner
  publishing-settings page was unavailable without a fresh login, so no settings claim was
  made. GitHub has no 0.14.0 Release or eligible publish/recovery run; its annotated tag is
  unsigned.

## Closure matrix

| Blocker | State | Owner | Required authority and action | Falsifiable acceptance | Fresh evidence |
|---|---|---|---|---|---|
| B01 | **TRUSTED PUBLISHER CONFIGURED; OIDC RELEASE PENDING** | PyPI project owner `jemsbhai` | Preserve the exact Trusted Publisher for owner `jemsbhai`, repository `explainiverse`, workflow `publish-pypi.yml`, environment `pypi`; complete the authorized roll-forward without adding a token fallback. | On the exact candidate, the sole publisher uses OIDC with no token/user/password/secret fallback; each exact PyPI file has Integrity provenance whose DSSE subject/digest and GitHub publisher repository/workflow/environment pass the hash-locked cryptographic verifier; repository and environment secret-name inventories remain empty. | The owner authenticated on 2026-09-04 and registered the exact publisher. Run `33891048942` stopped before requesting the environment or OIDC token. Run `33901507340` never reached the environment or OIDC publisher because its downstream jobs were skipped. Successful publication remains pending. |
| B02 | **V0.15.2 EXCEPTION PATH AUTHORIZED; EXECUTION EVIDENCE PENDING** | GitHub repository administrator and CI/merge authority | For `v0.15.2`, narrow only the app-bound required-check set to the policy-derived 21, merge PR #7, immediately restore all 23 contexts, and run every non-CUDA gate on the exact `main` commit. Capture/preflight only while the restored 23-context protection is live. | A capture no older than 30 minutes has `repository_controls_accepted=true`, zero violations under the exact exception, immutable Releases enabled, exact policy/snapshot digests, all 23 exact context/app bindings, and one successful provider-bound check for each of the 21 non-CUDA names on the candidate SHA. The capture proves that SHA is GitHub PR #7's single authoritative merged-event commit from the fully paginated issue timeline; actor and triggering actor match the capture principal. | PR #5 restored and proved all 23 protections before `v0.15.0` failed safely at SBOM generation. PR #6 repeated the narrow merge window, restoration, exact-main checks, capture, and preflight before the `v0.15.1` build succeeded but downstream jobs were skipped. PR #7 must repeat those controls with the explicit downstream conditions. |
| B03 | **BLOCKED — authorized recovery drill** | Separately authorized release operator | On a future build/attest/OIDC run, request the deliberate post-PyPI failure, then dispatch recovery from the immutable release tag and recover only the downstream GitHub Release from that original run. Do not reuse 0.14.0. | Source run concludes failure; build, attest, and publish each have exactly one successful attempt; the release job fails at the explicit drill step with later release steps skipped. Recovery executes from `refs/tags/<tag>` at that tag's exact commit, verifies attestations, and proves original, PyPI, and GitHub files byte-identical without a second upload. Retain source/all-attempt jobs, inventories, hashes, service JSON, and recovery evidence. | Verifier now distinguishes a staged drill from an unplanned downstream failure and rejects successful, missing, skipped, duplicate, branch-dispatched, or wrong-SHA source evidence. No authorized live drill exists; legacy 0.14.0 cannot satisfy it. |
| B04 | **WAIVED FOR `v0.15.2` ONLY; OTHERWISE BLOCKED** | Repository administrator and GPU-infrastructure owner | The exceptional release uses only `EXPLAINIVERSE-v0.15.2-CPU-ONLY` and makes no CUDA claim. Provision approved isolated Linux runners carrying `explainiverse-cuda-single` and `explainiverse-cuda-two` before any future stable release or CUDA verification claim. | The `v0.15.2` immutable record says all four CUDA jobs were not run, hardware evidence was not collected, and CUDA release verification is false. Closing B04 still requires all four minimum/latest one-/two-GPU jobs to succeed once on the exact candidate, with expected custom labels, visible-device counts, exact 15-node manifest, zero skips, retained evidence, and infrastructure approval. | Live variables/runners remain zero, so no authorized hosted hardware record exists. PR routing failures prove the fail-closed guard only. The explicit CPU-only record preserves that fact rather than representing manual/local testing as release evidence. |
| B05 | **BLOCKED — accepted push/live-main dependency evidence; hosted PR rehearsal green** | CI/merge authority | Land and run Python 3.10-3.13 plus direct-floor, Captum, SHAP/XGBoost, and current/floor dependency jobs; require their contexts. | Every declared edge is green on the exact candidate, with the correct substantive test surface, resolved graph, `pip check`, and no unowned skip. | PR dependency run `31513520975`, associated with head `9408f5c…` and merge checkout `0d5ca699…`, completed successfully across all six direct-floor/latest, SHAP/XGBoost, and Captum jobs; Python run `31513521028` also passed full-quality, minimum-direct, Quantus 9/9, and every supported-platform job. The rehearsals used event `pull_request`; the workflows are not landed or required on live `main`, so accepted `push` evidence remains open. |
| B06 | **BLOCKED — accepted push-event/publish binding; hosted PR rehearsal green** | Release-CI owner | Execute the artifact reproducibility workflow on two clean hosted Linux/Python-3.12 jobs. | Both builds use the same source SHA, Python and pip versions, platform family/architecture, hash-locked tools, requested `ubuntu-24.04` label, actual Ubuntu image family/OS/architecture, GitHub run, and attempt; matrix slots, job indexes, and build identities are distinct; wheel and sdist bytes match. Each exact hosted `ImageVersion` is required and retained as observed provenance but may differ during a fleet rollout; the report must expose both values and whether they match rather than normalize them. Full `platform.platform()` strings and runner display names are also retained for diagnosis but need not be equal. The exact accepted run and both complete manifests/hashes are retained, and the later publish distribution is byte-identical to each accepted build before attestation or upload. | PR run `31513521096` attempt 2 built merge checkout `0d5ca6996e548e6ffb4d89e83aac7fc524ba5dbd` on two distinct hosted jobs using Ubuntu image `20260720.247.2`; comparator job `93855161504` retained report artifact `9110345374`, digest `sha256:ae02222d908d820ca12e5aac0c765fe51a0e267cca70fca4befae01e81ab183e`. The 377,159-byte wheel SHA-256 is `56bd9d021b19ddc0ec4a49cdea67c142ace4b2faaa9df7213d9e94009a7b8746`; the 335,201-byte sdist SHA-256 is `5e51f2f1bf59bfea28a3f3b84a910709e673dcef34e66a42bb8fffdc1ca850a3`. This was a `pull_request` rehearsal associated with head `9408f5c…`, not the required landed `push` run, and no later publish distribution has been byte-bound to it. Later PR run `33795426655` recorded `20260831.293.1` and `20260823.283.1` on the same requested Ubuntu 24.04 profile and passed the byte comparison, but the former exact-version equality rule failed the environment step; that failed run is diagnostic evidence, not accepted release evidence. |
| B07 | **BLOCKED — accepted push/live-main binding; hosted PR rehearsal green** | Captum/CI owner | Run the mandatory five-file surface under exact Captum 0.8 and current after graph-integrity changes, locally and in required hosted jobs. | Both versions explicitly import Captum and pass all analytical, reference-parity, restoration, and fail-closed graph tests with zero skips before any surface is widened. | Dependency run `31513520975` explicitly imported Captum 0.8.0 and 0.9.0 and passed the exact five-file 306/306 surface with zero skips in both jobs. The Quantus/fixture partition guard also passed the exact 9/9 hosted reference lane. This demonstrates the repository contract on the PR merge checkout, but not the policy-required landed `push` contexts on live `main`. |
| B08 | **BLOCKED — governance evidence** | Project governance/release manager | Add a second trusted principal and prevent self-review, or retain the approved single-operator route and disclose it in the actual release record. | Either approver and executor are distinct, or the release body states single-operator approval at finalization and the immutable `RELEASE_GOVERNANCE.json`/`.md` assets bind the attested accepted external-control snapshot and preflight identity. The generated governance record validates actor, reviewer, self-review setting, commit, tag, run, and policy/snapshot digests. | Live `pypi` reviewer is only `jemsbhai` and self-review is permitted; the only other direct collaborator has write rather than release-admin authority. A fail-closed governance record, final-body recheck, and authoritative immutable governance assets now exist, but no future release record exists. GitHub release notes remain editable and are not treated as the durable record. |
| B09 | **MIXED — hosted PR rehearsal green; accepted push contexts and claim evidence open** | Platform, typing, JS, and independent accessibility owners | Run the policy-required exact-candidate macOS ARM and Node/React/browser gates. Separately finish typing before adding `py.typed`, and obtain physical NVDA/VoiceOver review before any AT-support claim. | For this stable release, the macOS ARM and JS/browser policy contexts must be green. A future typed claim additionally requires strict mypy zero, Pyright 100%, and clean installed-wheel consumers; a future AT claim requires both physical profiles, exact deployment/build binding, reviewed bytes/hashes, and evidence at most 180 days old. | PR Python run `31513521028` passed macOS 15 ARM64, its OpenMP coexistence proof, 3,630 tests with four owned skips, and `pip check`. PR JS run `31513521103` passed Node 20/22, React 18, and zero-retry Playwright 9/9 across Chromium, Firefox, and WebKit. These are `pull_request` rehearsals, not the required landed `push` contexts. Strict mypy still reports 1,375 errors in 44 files; Pyright remains 0% with no marker; physical AT evidence is absent. Stable Python remains explicitly untyped and the private demo AT-uncertified. |
| B10 | **BLOCKED FOR BOUND WIDENING — not a claim of 1.x support** | Dependency maintainer; upstream supplies candidate | Keep `scikit-image<1.0`. Let the scheduled monitor capture PyPI metadata and select only a real non-yanked 1.x prerelease; propose any bound change separately. | The discovery record identifies a real candidate and preserves metadata hash/serial. A distinct source-only probe requires the capped Explainiverse distribution to be absent, records that post-candidate `pip check` covers dependencies only, and passes localisation, LIME, package/twine, and all tutorial gates with retained freeze, JUnit/log, boundary records, and distribution hashes. Its built wheel must still disclose `<1.0` and `full_distribution_graph_verified=false`; only a separate reviewed bound-change wheel installed with the candidate and a full `pip check` may support widening. No candidate produces a blocked/failing monitor and no green probe context. | PyPI current is 0.26.0 and exposes no qualifying 1.x prerelease (metadata SHA-256 `2a986d45ee278f7820a060319d53f6e4c9b68d9841907829aeb68b09c54ff7bc`, serial `33119953`). The `<1.0` cap remains. |
| B11 | **QUARANTINED — P2 publication capability** | JS, security, release, accessibility, and scientific-parity owners | Preserve `private=true`, CommonJS/experimental metadata, non-parity disclosure, and no npm publication until every separately reviewed publication and scientific prerequisite exists. | Direct ESM/browser library consumers, exact export/tarball contract, publication threat model, provenance/recovery, physical AT evidence, and an explicit algorithm-parity decision are all exact-candidate green before changing private/module/publication/parity metadata. | Local package tests, audit, browser checks, and 16-file dry-run tarball are diagnostic controls only. The npm name is unpublished. No publication, parity, AT, provenance, or recovery acceptance exists, so no promotion is permitted. |

## Interpretation

B10 is a blocker to widening the existing dependency bound, and B11 is a deliberately retained
P2 quarantine; neither may be reworded as supported capability. They do not offset the open P0
rows. The reviewed executable policy also promotes B06, B07, and the macOS/JS portions of B09 to
this stable release's gates even though their general roadmap priority is P1. Typing and physical
AT remain claim gates: stable Python can remain explicitly untyped and the private demo can remain
explicitly AT-uncertified. Green pull-request rehearsals do not satisfy the reviewed event/ref
binding. A normal stable release still cannot be recommended while B01-B08 and B09's selected
macOS/JS contexts lack accepted `push` evidence on the landed candidate. The exact `v0.15.2`
exception may proceed after every non-CUDA and publication acceptance is satisfied, but B04
remains open and the resulting release scope is CPU-verified only.
