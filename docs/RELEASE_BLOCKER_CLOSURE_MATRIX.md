# Stable-release blocker closure matrix

This is the current closure ledger for B01-B11. It was initially re-audited from clean commit
`dd76815c79076c43d88568ae10f43be7bb546d9c` on 2026-08-11, refreshed through predecessor head
`9408f5c63c8accab2611658422b768990ead42d5`, and updated after the authorized merge and live-control
continuation, the 2026-08-12 private Kaggle diagnostic capacity probe, and release-verifier
hardening described below. A repository guard can make future evidence falsifiable, but it
cannot close a live-service, hosted-runner, hardware, or human-review blocker. `BLOCKED` therefore
remains the only honest state until the acceptance column is satisfied with direct evidence from
the final candidate commit. B10 governs a future bound widening and B11 is an intentionally
retained P2 quarantine, so their blocked states are not stable-release failures while those
surfaces remain unclaimed.

The stable-release recommendation remains **NO**. The reviewed PR chain and B02 settings have
landed, and every selected non-CUDA `push` context succeeded on current `main`. A private Kaggle
probe found two T4 devices, but no approved owner-locked JIT Actions execution has occurred. That
B04 evidence gap prevents the trusted CUDA dispatch, security PR #4's merge, a final-candidate
zero-violation B02 capture, and every downstream publication, governance, and recovery action. No
tag, PyPI publication, GitHub Release, release governance record, or staged recovery drill has
been created, and no manual accessibility or accepted Actions GPU evidence is represented.

## Live continuation observations

- PR #2 merged as `e0e42cfbf99fdb80bcb0d4f9a4e281b736ee7506`, with parents
  `49e962c090e90e62f315837067e5adc3e3f04d1c` and
  `9408f5c63c8accab2611658422b768990ead42d5`. The continuation reconciliation commit
  `64e9255409b450e407aa9f77a75092cadbe1d9e9` has parents
  `8d96a5b8fab6beac3eaf70876dc16f9aebddb3a3` and `e0e42cfbf99fdb80bcb0d4f9a4e281b736ee7506`.
  PR #3 then merged as current `main` `a9789d009f6ec5134bc53b9d2f6a8b59726e75c7`, with parents
  `e0e42cfbf99fdb80bcb0d4f9a4e281b736ee7506` and
  `64e9255409b450e407aa9f77a75092cadbe1d9e9`. The reconciliation and final PR merge share tree
  `707a10435c7f257663b50729009c9639d4a082e0`, so the audit and hardening trees were not lost.
- Live `main` protection is now strict and requires all 23 reviewed contexts, each provider-bound
  to GitHub Actions app ID `15368`. Administrator enforcement and conversation resolution remain
  enabled; force pushes and deletions remain disabled. Repository-level immutable Releases is
  enabled. The authenticated capture at `2026-08-11T18:39:00.267775+00:00` has policy SHA-256
  `e6d8e4de111f5efdcecfef726e9c4f4a526eebd0d16520c426d9405aca03a443` and snapshot SHA-256
  `4833e9f245c4d30f3753fd9976e8b540057e8ffbb187cbdccae9b9b7611b34e5`. It is still rejected,
  with exactly two violations: CUDA minimum and latest were completed with `failure` rather than
  `success`. Strengthened capture `ab182478…693da` at
  `2026-08-11T19:25:39.958091+00:00`, policy SHA-256 `03478995…2441d`, is rejected with exactly
  three violations: the same CUDA failures and collaborators `b-urge` plus `jemsbhai` rather than
  sole collaborator `jemsbhai`. It directly records zero pending invitations. Thus 21/23
  provider-bound exact-`main` checks are green and B02 settings are configured, but B02 acceptance
  is not. `b-urge` remains at `write` while variables/runners are absent and must be temporarily
  removed only when approved capacity and restoration coordination exist.
- On exact `main` `a9789d0…`, `push` runs Python `31520989339`, dependency `31520989552`,
  JavaScript `31520989433`, and artifact reproducibility `31520989403` all succeeded. The JS
  Playwright artifact is `9113076913`, digest
  `sha256:0aa14ffc98ad0a3cff6c3bd261d6f822b757c3a2cb4712797230564e4290ae9f`.
  Artifact comparator job `93877997328` retained report artifact `9113035037`, digest
  `sha256:f0cf538830cde7bab7504ca7bec0c1eff218fdb806b98cbe142826d65913c0f2`.
  Its 377,159-byte `explainiverse-0.15.0-py3-none-any.whl` has SHA-256
  `972080a133134f4497c83b9069fde95379b171aefa9c756036ef505d30474011`; its 335,212-byte
  `explainiverse-0.15.0.tar.gz` has SHA-256
  `aa234f7b20d53a4ace957df08f164c34b3729af4f3a6840768c58dce1a2f1105`. CUDA `push` run
  `31520989476` failed closed before genuine hardware evidence; those failures are not represented
  as test results from a GPU.
- Security commit `9ad738a2dc62777ae199b03a3348b24b45006da2` remains in open, unmerged PR #4; its parent
  is `a9789d009f6ec5134bc53b9d2f6a8b59726e75c7`. Within the reviewed workflow, `pull_request` and
  `push` events cannot route code onto the custom labels. Because a write collaborator can instead
  alter and dispatch a branch workflow, that guard is defense in depth; the authenticated control
  snapshot enforces the human-collaborator component (sole human writer and zero invitations).
  An owner-authenticated installed-App/automation permission export is separately required because
  the collaborator API cannot certify non-human principals. On exact PR head
  `52872aa737635535d8fe6d22dd0ba8dbc98956b1`,
  Python `31524703765` attempt 2, dependency `31524703764`, JavaScript `31524703778`, and artifact
  `31524703797` succeeded; the artifact comparator was `93890433539`. Exactly the two required
  CUDA reporters remained red. Merging still requires a genuine owner dispatch through approved,
  fresh one-job JIT runners at that exact future head, not a gate waiver.
- Dedicated GCP project `explainiverse-release-ci-26` (project number `305968033598`) is linked to
  enabled billing account `019553-F5B725-3C44B5`. It has zero VMs. The default network was removed;
  custom-mode VPC `explainiverse-runner` contains only `runner-us-central1` (`10.61.0.0/28`, private
  Google access), Cloud Router/NAT `explainiverse-runner-router`/`explainiverse-runner-nat`, and an
  IAP-only TCP/22 ingress rule from `35.235.240.0/20`. The Compute Engine default service account
  `305968033598-compute@developer.gserviceaccount.com` is disabled. This is prepared isolation,
  not GPU evidence or infrastructure approval.
- Every requested GPU quota path was denied with state detail `Quota request denied`. In the
  dedicated project: global `GPUS-ALL-REGIONS` 3 requested/0 granted, trace
  `f489e693-6d95-456d-b4e5-ca81645734df`; standard T4 `us-central1` 3/1,
  `56f86983-a7b9-4025-8c8d-8939dc20505c`; minimum global retry 2/0,
  `0209f3c3-4940-4adf-bc5f-b9e867766a5e`; minimum standard T4 retry 2/1,
  `e7dfb49c-7be9-4654-a16b-a022bf5ce68b`; preemptible T4 2/1,
  `781fe031-a0c1-4b6d-8c87-ec45915a59a7`; and preemptible L4 2/1,
  `1e125e5c-0f3a-467d-8bce-b9b457644a10`. Owner-only fallback project
  `deepstation-freshfridge` (number `605561044196`) was audited but not provisioned because it has
  unrelated enabled services: global 2/0, `35d45aee-27a4-4097-8a8f-5e925cae725a`; standard T4
  2/1, `5474ff04-daac-430a-9eed-b74c715e2903`; preemptible T4 2/1,
  `4d81c462-10ea-427e-b45f-718e10dd6d6f`; preemptible L4 2/1,
  `bf7d71d5-c999-4c5d-8963-5720c205caa8`.
- The 2026-08-12 private Kaggle diagnostic proved two-T4 capacity exists. The retained
  [raw probe](evidence/kaggle-gpu-capacity-probe-20260812.json), SHA-256
  `8e14b73c8c22a5a09e87b7e916a793264ab7a9566af614031fc23df4fab9e944`, records two distinct
  Tesla T4 UUIDs, `GPU-665b7910-7d3c-2ea8-96fb-4df7e940028f` and
  `GPU-e6fb927c-8f6f-a149-b798-cd5bcc8c2247`, with driver `580.159.04`, Torch
  `2.10.0+cu128`, and CUDA runtime `12.8`. The retained
  [kernel metadata](evidence/kaggle-gpu-capacity-probe-20260812-metadata.json), SHA-256
  `90007b376321264e1c42c3e91f0b5856882b8bd3223dd214c47bc4338ef8124d`, identifies private kernel
  `muntasersyed/explainiverse-gpu-capacity-probe-20260812` (`id_no: 130496083`) and image
  `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`.
  A separate live kernel-status query observed version `1`. The separate live CLI observation
  `uvx --from kaggle==2.2.2 kaggle quota` reported
  `GPU 0.01 hours used / 29.99 hours remaining`; that value is not part of the retained JSON. The
  kernel received no GitHub credential, did not fetch the repository, and was never registered as
  an Actions runner; its only GitHub request was unauthenticated `/meta`. This was capacity
  discovery only: it ran none of the 15 CUDA nodes and supplies no accepted B04 evidence.
- Release-verifier hardening now rejects CUDA run/job attempts other than exact integer `1`, null
  or mismatched job SHAs, boolean/non-positive job and runner IDs, empty runner names, and runner-ID
  reuse across required one-job JIT assignments. The staged-recovery verifier separately requires
  exact integer attempt `1` for the source run and every trusted build, attest, publish, and release
  job. The combined command
  `poetry run pytest -q tests/test_release_external_controls.py tests/test_release_recovery.py tests/test_release_governance.py tests/test_p0_release_workflows.py tests/test_cuda_skip_policy.py`
  passed `198` tests in `2.36s`; Black, isort, mypy on both verifier scripts, and
  `git diff --check` were clean. These are repository-logic results, not GPU execution.
- PyPI `explainiverse` 0.15.0 and GitHub tag `v0.15.0` remain absent. The exact PyPI Trusted
  Publisher has not been authenticated or captured. A local-only SSH annotated-tag preflight
  succeeded with Ed25519 fingerprint `SHA256:84G7/ewIxErnHPmIrtaW52+1qy+2MlQqzbmCOW6tGc0` and
  the local test tag was deleted; GitHub registration still requires live sudo/2FA, so no
  GitHub-verified signing is claimed. No upload, tag, release, governance record, or recovery drill
  has occurred. Strict typing and physical NVDA/VoiceOver evidence remain explicitly unclaimed.
  `scikit-image<1.0` remains unchanged, and the experimental JavaScript package remains
  `private=true`, CommonJS-only, unpublished, AT-uncertified, and outside any ESM/browser-library
  or Python-algorithm-parity claim.

## Historical predecessor control observations

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
| B01 | **BLOCKED — authenticated PyPI publisher/signing setup** | PyPI project owner `jemsbhai` | Directly verify or register the exact Trusted Publisher for owner `jemsbhai`, repository `explainiverse`, workflow `publish-pypi.yml`, environment `pypi`; register a GitHub-verifiable signing key; separately authorize one release. | Archive the authenticated publisher and signing-key records. On the exact candidate, the sole publisher uses OIDC with no token/user/password/secret fallback; each exact PyPI file has Integrity provenance whose DSSE subject/digest and GitHub publisher repository/workflow/environment pass the hash-locked cryptographic verifier; repository and environment secret-name inventories remain empty. | PyPI 0.15.0 and tag `v0.15.0` remain absent. The PyPI owner settings still require an authenticated owner capture. A local-only SSH annotated-tag preflight succeeded with Ed25519 public-key fingerprint `SHA256:84G7/ewIxErnHPmIrtaW52+1qy+2MlQqzbmCOW6tGc0` and a test tag object containing an SSH signature; the local test tag was deleted. GitHub key registration remains blocked at live sudo/2FA confirmation, so no GitHub-verified signing or authorized OIDC upload is claimed. |
| B02 | **BLOCKED — controls configured; CUDA and runner authority open** | GitHub repository administrator and CI/merge authority | Keep all 23 policy contexts provider-bound to GitHub Actions app ID 15368 and immutable Releases enabled; obtain genuine CUDA success on the final candidate, enforce sole-collaborator/zero-invitation authority, retain an authenticated installed-App permission audit, and repeat the authenticated capture. | A capture no older than 30 minutes has `repository_controls_accepted=true`, zero violations, immutable Releases enabled, exact policy/snapshot digests, exact context/app bindings, sole collaborator/effective writer `jemsbhai`, zero pending invitations, and one successful provider-bound check per required name on the candidate SHA. A separate owner-authenticated export proves no non-owner-equivalent App/automation can modify workflows and dispatch Actions. Actor and triggering actor both equal the capture principal, and the retained run attempt/triggering actor matches the Actions API source run. | Strict `main` protection now has all 23 exact contexts with app ID `15368`, and immutable Releases is enabled. Strengthened capture `ab182478…693da` at `2026-08-11T19:25:39.958091+00:00` has policy SHA `03478995…2441d` and exactly three violations: `b-urge` plus `jemsbhai` are collaborators instead of sole collaborator `jemsbhai`, and CUDA minimum/latest failed. It records zero invitations and all other 21 exact-`main` checks successful. Installed-App authority is not yet authenticated; no gate is waived. |
| B03 | **BLOCKED — recovery drill not executed** | Separately authorized release operator | On the eventual build/attest/OIDC run, request the deliberate post-PyPI failure, then dispatch recovery from the immutable release tag and recover only the downstream GitHub Release from that original run. Do not reuse 0.14.0. | Source run concludes failure; build, attest, and publish each have exactly one successful attempt; the release job fails at the explicit drill step with later release steps skipped. Recovery executes from `refs/tags/<tag>` at that tag's exact commit, verifies attestations, and proves original, PyPI, and GitHub files byte-identical without a second upload. Retain source/all-attempt jobs, inventories, hashes, service JSON, and recovery evidence. | No tag, PyPI upload, source publication run, or recovery run exists. The drill is downstream of B04, authenticated B01 configuration, and the final B02 capture; legacy 0.14.0 cannot satisfy it. |
| B04 | **BLOCKED — diagnostic two-GPU capacity found; accepted Actions evidence absent** | Repository administrator, GPU-capacity account owner, and GPU-infrastructure owner | Convert suitable two-GPU Linux capacity into an approved owner-locked Actions window; coordinate restoration and temporarily remove every non-owner collaborator, require zero invitations, audit installed App/automation authority, and clear custom-label queues. Keep repository variables empty and owner-dispatch the reviewed ref with four fresh exact nonce inputs and no runner online; verify the exact queued jobs, then register fresh one-job JIT runners without exposing a reusable credential. | All four minimum/latest one-/two-GPU jobs complete successfully exactly once on the candidate, with the expected custom label, exact declared visible-device count, exact expected 15-node manifest, 15 executed, zero skips, positive/distinct runner IDs, and retained human/App authority, queue, job, runner, and infrastructure-isolation evidence. The authority lock remains through publication. Success/failure cleanup proves variables, queues, runners, VMs, disks, and capacity-side credentials are zero before the exact prior permission is re-invited and its acceptance recorded. | GCP quota requests remain denied and the dedicated project has zero VMs/runners/variables. The linked private Kaggle kernel version `1` found two distinct T4s and the live quota query reported `29.99` GPU hours remaining, but it had no GitHub credential/repository/runner and ran no release test. The hardened verifier is green across the `198`-test release suite, but that is logic evidence only. `b-urge` remains `write`, installed-App authority is not yet locked, and no owner-dispatched JIT Actions job exists; B04 remains open. |
| B05 | **BLOCKED — final-candidate rerun; current-main push green** | CI/merge authority | After PR #4 can merge, rerun Python 3.10-3.13 plus direct-floor, Captum, SHAP/XGBoost, and current/floor dependency jobs on final `main`. | Every declared edge is green on the exact final candidate, with the correct substantive test surface, resolved graph, `pip check`, and no unowned skip. | Exact current `main` `a9789d0…` has successful `push` runs Python `31520989339` and dependency `31520989552`, including supported platforms, Quantus 9/9, Captum minimum/current, direct floors, and SHAP/XGBoost edges. PR #4 security commit `9ad738a…` also settled 21/23 required contexts green. Because that security commit is required but unmerged, these runs cannot yet be designated final-candidate evidence. |
| B06 | **BLOCKED — final-candidate/publish binding; current-main reproducibility green** | Release-CI owner | After PR #4 can merge, rerun artifact reproducibility and bind the later clean-tag-checkout publish bytes to that exact accepted run. | Both builds use the same source SHA, Python and pip versions, platform family/architecture, locked tools, runner image/OS/architecture, GitHub run, and attempt; matrix slots, job indexes, and build identities are distinct; wheel and sdist bytes match. Full `platform.platform()` strings and runner display names are retained for diagnosis but need not be equal. The exact accepted run and both complete manifests/hashes are retained, and the later publish distribution is byte-identical to each accepted build before attestation or upload. | Exact current-main `push` run `31520989403` succeeded; comparator `93877997328`, report artifact `9113035037`, digest `sha256:f0cf538830cde7bab7504ca7bec0c1eff218fdb806b98cbe142826d65913c0f2`, wheel SHA-256 `972080a133134f4497c83b9069fde95379b171aefa9c756036ef505d30474011`, and sdist SHA-256 `aa234f7b20d53a4ace957df08f164c34b3729af4f3a6840768c58dce1a2f1105` are archived. PR #4 artifact run `31522899822` required whole-run attempt 2 after hosted-image drift and then succeeded (comparator `93884787455`). A final-main rerun and publication byte binding remain absent. |
| B07 | **BLOCKED — final-candidate rerun; current-main Captum evidence green** | Captum/CI owner | Run the mandatory five-file surface under exact Captum 0.8 and current after PR #4 merges and after any further graph-integrity change. | Both versions explicitly import Captum and pass all analytical, reference-parity, restoration, and fail-closed graph tests with zero skips before any surface is widened. | Dependency `push` run `31520989552` on `a9789d0…` succeeded in Captum minimum/current jobs; predecessor hosted evidence recorded exact 306/306 with zero skips under 0.8.0 and 0.9.0. PR #4's dependency contexts are also green. Final-candidate identity remains pending B04 and PR #4; no Captum surface is widened. |
| B08 | **BLOCKED — actual governance/release evidence absent** | Project governance/release manager | Add a second trusted principal and prevent self-review, or retain the approved single-operator route and disclose it in the actual release record. | Either approver and executor are distinct, or the release body states single-operator approval at finalization and the immutable `RELEASE_GOVERNANCE.json`/`.md` assets bind the attested accepted external-control snapshot and preflight identity. The generated governance record validates actor, reviewer, self-review setting, commit, tag, run, and policy/snapshot digests. | Live `pypi` reviewer is only `jemsbhai` and self-review is permitted. Repository generators/validators exist, but there is no tag, finalized Release, or actual `RELEASE_GOVERNANCE.json`/`.md` evidence asset for 0.15.0. No B08 closure is claimed. |
| B09 | **MIXED — selected current-main contexts green; final-candidate and claim evidence open** | Platform, typing, JS, and independent accessibility owners | Rerun the policy-required macOS ARM and Node/React/browser gates after PR #4 merges. Separately finish typing before adding `py.typed`, and obtain physical NVDA/VoiceOver review before any AT-support claim. | For this stable release, the macOS ARM and JS/browser policy contexts must be green on the final candidate. A future typed claim additionally requires strict mypy zero, Pyright 100%, and clean installed-wheel consumers; a future AT claim requires both physical profiles, exact deployment/build binding, reviewed bytes/hashes, and evidence at most 180 days old. | Python `push` run `31520989339` and JS `push` run `31520989433` succeeded on `a9789d0…`, including macOS 15 ARM64/OpenMP, Node 20/22, React 18, and Playwright 9/9. Playwright artifact `9113076913` has digest `sha256:0aa14ffc98ad0a3cff6c3bd261d6f822b757c3a2cb4712797230564e4290ae9f`. PR #4's selected contexts are green but it is not merged. Strict mypy/Pyright readiness and physical NVDA/VoiceOver evidence remain absent and explicitly unclaimed; stable Python remains untyped and the private demo AT-uncertified. |
| B10 | **BLOCKED FOR BOUND WIDENING — not a claim of 1.x support** | Dependency maintainer; upstream supplies candidate | Keep `scikit-image<1.0`. Let the scheduled monitor capture PyPI metadata and select only a real non-yanked 1.x prerelease; propose any bound change separately. | The discovery record identifies a real candidate and preserves metadata hash/serial. A distinct source-only probe requires the capped Explainiverse distribution to be absent, records that post-candidate `pip check` covers dependencies only, and passes localisation, LIME, package/twine, and all tutorial gates with retained freeze, JUnit/log, boundary records, and distribution hashes. Its built wheel must still disclose `<1.0` and `full_distribution_graph_verified=false`; only a separate reviewed bound-change wheel installed with the candidate and a full `pip check` may support widening. No candidate produces a blocked/failing monitor and no green probe context. | PyPI current is 0.26.0 and exposes no qualifying 1.x prerelease (metadata SHA-256 `2a986d45ee278f7820a060319d53f6e4c9b68d9841907829aeb68b09c54ff7bc`, serial `33119953`). The `<1.0` cap remains. |
| B11 | **QUARANTINED — P2 publication capability** | JS, security, release, accessibility, and scientific-parity owners | Preserve `private=true`, CommonJS/experimental metadata, non-parity disclosure, and no npm publication until every separately reviewed publication and scientific prerequisite exists. | Direct ESM/browser library consumers, exact export/tarball contract, publication threat model, provenance/recovery, physical AT evidence, and an explicit algorithm-parity decision are all exact-candidate green before changing private/module/publication/parity metadata. | Local package tests, audit, browser checks, and 16-file dry-run tarball are diagnostic controls only. The npm name is unpublished. No publication, parity, AT, provenance, or recovery acceptance exists, so no promotion is permitted. |

## Interpretation

B10 is a blocker to widening the existing dependency bound, and B11 is a deliberately retained
P2 quarantine; neither may be reworded as supported capability. They remain unchanged and do not
offset the open P0 rows. The reviewed executable policy also promotes B06, B07, and the macOS/JS
portions of B09 to this stable release's gates even though their general roadmap priority is P1.
Typing and physical AT remain claim gates: stable Python can remain explicitly untyped and the
private demo can remain explicitly AT-uncertified.

The operative cascade is B04 owner-locked JIT Actions evidence → trusted CUDA evidence on PR #4 →
merge of the self-hosted-runner security boundary → final-main reruns → authenticated B01
publisher/signing configuration and direct owner capture → zero-violation pre-tag B02 capture →
signed immutable tag and sole OIDC publication → B08 release record and B03 downstream-only
recovery. Complete B01 setup before B02's 30-minute freshness window. Current-main and PR #4
non-CUDA successes are valuable direct evidence, but they cannot bypass that order or turn
`a9789d0…` into the final candidate.
The stable release recommendation therefore remains **NO**.
