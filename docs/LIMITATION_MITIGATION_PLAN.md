# Residual-limitations mitigation plan

This plan starts from the remediation audit that prepared the `0.15.0` candidate. A limitation is not called fixed
until its retirement test is falsifiably green. Some scientific limits cannot be removed by
software; their mitigation is disclosure, sensitivity analysis, and refusal to make a broader
claim.

The B01-B11 audit began from clean commit
`dd76815c79076c43d88568ae10f43be7bb546d9c`, was refreshed through predecessor head
`9408f5c63c8accab2611658422b768990ead42d5`, and now includes the authorized 2026-08-11 PR merges,
live B02 control mutations, hosted `push` evidence, B04 provisioning attempts, the 2026-08-12
private Kaggle diagnostic capacity probe, and release-verifier hardening. It is tracked in
[`RELEASE_BLOCKER_CLOSURE_MATRIX.md`](RELEASE_BLOCKER_CLOSURE_MATRIX.md). That ledger supersedes
older observation timestamps and counts below when deciding whether a stable release is
supportable; the answer remains **no** until its applicable executable stable-gate rows are green.

## Current continuation snapshot

- PR #2 merge `e0e42cfbf99fdb80bcb0d4f9a4e281b736ee7506` has parents
  `49e962c090e90e62f315837067e5adc3e3f04d1c` and
  `9408f5c63c8accab2611658422b768990ead42d5`. Reconciliation commit
  `64e9255409b450e407aa9f77a75092cadbe1d9e9` has parents
  `8d96a5b8fab6beac3eaf70876dc16f9aebddb3a3` and `e0e42cfbf99fdb80bcb0d4f9a4e281b736ee7506`.
  PR #3 merge/current `main` `a9789d009f6ec5134bc53b9d2f6a8b59726e75c7` has parents
  `e0e42cfbf99fdb80bcb0d4f9a4e281b736ee7506` and
  `64e9255409b450e407aa9f77a75092cadbe1d9e9`; both latter commits share tree
  `707a10435c7f257663b50729009c9639d4a082e0`.
- Strict `main` protection now contains all 23 exact required contexts, each bound to GitHub
  Actions app ID `15368`, and immutable Releases is enabled. Capture
  `4833e9f245c4d30f3753fd9976e8b540057e8ffbb187cbdccae9b9b7611b34e5` at
  `2026-08-11T18:39:00.267775+00:00`, against policy
  `e6d8e4de111f5efdcecfef726e9c4f4a526eebd0d16520c426d9405aca03a443`, is rejected with exactly
  two violations: the genuine CUDA minimum/latest contexts failed. The other 21 exact-`main`
  contexts succeeded. Strengthened capture
  `ab182478d423dbdd8ba2751281d36e28371444d650cef5e069b15fa0f29693da` at
  `2026-08-11T19:25:39.958091+00:00`, against policy
  `0347899501f8fe97197974197e3acccc45c088caa2a844ed70d48c743062441d`, is rejected with exactly
  three violations: the same two CUDA failures and collaborators `b-urge` plus `jemsbhai` instead
  of sole collaborator `jemsbhai`. It directly recorded zero pending invitations; no permission
  was changed before an owner-locked Actions runner window existed.
- Exact-`main` `push` runs Python `31520989339`, dependency `31520989552`, JavaScript
  `31520989433`, and artifact reproducibility `31520989403` succeeded. Playwright artifact
  `9113076913` has digest
  `sha256:0aa14ffc98ad0a3cff6c3bd261d6f822b757c3a2cb4712797230564e4290ae9f`.
  Artifact comparator `93877997328` retained report artifact `9113035037`, digest
  `sha256:f0cf538830cde7bab7504ca7bec0c1eff218fdb806b98cbe142826d65913c0f2`; wheel SHA-256 is
  `972080a133134f4497c83b9069fde95379b171aefa9c756036ef505d30474011` and sdist SHA-256 is
  `aa234f7b20d53a4ace957df08f164c34b3729af4f3a6840768c58dce1a2f1105`.
- Security commit `9ad738a2dc62777ae199b03a3348b24b45006da2` remains in unmerged PR #4 (parent
  `a9789d009f6ec5134bc53b9d2f6a8b59726e75c7`). Within the reviewed workflow it prevents
  `pull_request`/`push` code from routing onto self-hosted GPU runners, but a mutable branch
  workflow means this guard is defense in depth rather than the authority boundary. The control
  policy now requires `jemsbhai` to be the sole collaborator with zero invitations before any
  runner window; an owner-authenticated installed-App/automation audit is separately required.
  The exact prior `b-urge` write permission is re-invited and accepted only after variables,
  runners, VMs, disks, and relevant queues return to zero. On PR head
  `52872aa737635535d8fe6d22dd0ba8dbc98956b1`, 21/23 required contexts were green after Python run
  `31524703765` attempt 2 recovered a hosted-runner CA failure; only the two intentional CUDA
  reporters remained red. Artifact run `31524703797` succeeded on attempt 1, comparator
  `93890433539`. PR #4 must not be admin-bypassed or merged by weakening either required CUDA gate.
- Dedicated billed GCP project `explainiverse-release-ci-26` (number `305968033598`) has zero VMs,
  no registered GitHub runner, and no runner variables. Its default network was removed; custom
  VPC `explainiverse-runner` uses subnet `runner-us-central1` (`10.61.0.0/28`, private Google
  access), Cloud Router/NAT, and only IAP-source TCP/22 ingress. Its default Compute Engine service
  account is disabled. These isolation preparations are not GPU evidence.
- Every requested quota change was denied. Dedicated-project traces are: global GPU 3/0
  `f489e693-6d95-456d-b4e5-ca81645734df`; standard T4 3/1
  `56f86983-a7b9-4025-8c8d-8939dc20505c`; minimum global retry 2/0
  `0209f3c3-4940-4adf-bc5f-b9e867766a5e`; minimum standard T4 retry 2/1
  `e7dfb49c-7be9-4654-a16b-a022bf5ce68b`; preemptible T4 2/1
  `781fe031-a0c1-4b6d-8c87-ec45915a59a7`; preemptible L4 2/1
  `1e125e5c-0f3a-467d-8bce-b9b457644a10`. Owner-only fallback
  `deepstation-freshfridge` (`605561044196`) was audited but not provisioned because it has
  unrelated enabled services: global 2/0 `35d45aee-27a4-4097-8a8f-5e925cae725a`; standard T4 2/1
  `5474ff04-daac-430a-9eed-b74c715e2903`; preemptible T4 2/1
  `4d81c462-10ea-427e-b45f-718e10dd6d6f`; preemptible L4 2/1
  `bf7d71d5-c999-4c5d-8963-5720c205caa8`. All reported state detail `Quota request denied`.
- A private Kaggle capacity probe on 2026-08-12 found two distinct Tesla T4 devices. The retained
  [raw probe](evidence/kaggle-gpu-capacity-probe-20260812.json) has SHA-256
  `8e14b73c8c22a5a09e87b7e916a793264ab7a9566af614031fc23df4fab9e944` and records UUIDs
  `GPU-665b7910-7d3c-2ea8-96fb-4df7e940028f` and
  `GPU-e6fb927c-8f6f-a149-b798-cd5bcc8c2247`, driver `580.159.04`, platform
  `Linux-6.12.90+-x86_64-with-glibc2.35`, Python `3.12.13`, Torch `2.10.0+cu128`, and CUDA runtime
  `12.8`. The retained
  [kernel metadata](evidence/kaggle-gpu-capacity-probe-20260812-metadata.json), SHA-256
  `90007b376321264e1c42c3e91f0b5856882b8bd3223dd214c47bc4338ef8124d`, identifies private kernel
  `muntasersyed/explainiverse-gpu-capacity-probe-20260812` (`id_no: 130496083`) and image
  `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`.
  A separate live kernel-status query observed version `1`. The post-probe live command
  `uvx --from kaggle==2.2.2 kaggle quota` reported
  `GPU 0.01 hours used / 29.99 hours remaining`; that quota observation is not embedded in either
  JSON file. The kernel received no GitHub credential, did not fetch the repository, and was not
  registered as a GitHub Actions runner; its only GitHub request was unauthenticated `/meta`.
  This is a diagnostic capacity inventory only, not an accepted CUDA run or B04 closure.
- The CUDA evidence verifier now requires the top-level run and all four required jobs to be exact
  integer attempt `1`, binds every job to the non-null release SHA, rejects boolean/non-positive
  job and runner IDs, requires a nonempty runner name, and requires a distinct runner ID for every
  one-job JIT assignment. The staged-recovery verifier likewise requires exact integer attempt
  `1` for the source run and each trusted build, attest, publish, and release job. The combined
  command `poetry run pytest -q tests/test_release_external_controls.py tests/test_release_recovery.py tests/test_release_governance.py tests/test_p0_release_workflows.py tests/test_cuda_skip_policy.py`
  passed `198` tests in `2.36s`; Black, isort, mypy on both verifier scripts, and
  `git diff --check` were also clean. This validates fail-closed repository logic, not GPU
  execution.
- PyPI 0.15.0 and tag `v0.15.0` remain absent. The exact Trusted Publisher still lacks an
  authenticated owner capture. Local-only SSH tag signing was proved with Ed25519 fingerprint
  `SHA256:84G7/ewIxErnHPmIrtaW52+1qy+2MlQqzbmCOW6tGc0`, but GitHub registration remains blocked at
  live sudo/2FA and no GitHub-verified signature is claimed. No tag, publication, GitHub Release,
  actual governance record, or recovery drill exists. Strict typing and physical NVDA/VoiceOver
  evidence remain explicitly unclaimed; `scikit-image<1.0` and B11's private/CommonJS/unpublished
  quarantine remain unchanged.

The dependency cascade is B04 owner-locked JIT Actions evidence → genuine PR #4 CUDA dispatch →
PR #4 merge → final-main reruns → zero-violation B02 capture → authenticated B01
setup/publication → B08 release record and B03 downstream-only recovery. Green non-CUDA runs do
not bypass that order.

## Priority and release policy

- **P0 — before a stable release:** a missing gate or control that could invalidate released
  artifacts or an advertised supported configuration.
- **P1 — next evidence milestone:** a material supported-domain restriction with a feasible
  implementation or dedicated validation gate.
- **P2 — research/expansion:** quarantined or absent capability that needs new algorithmic
  evidence, not a compatibility patch.
- **Permanent boundary:** no honest implementation can eliminate the underlying uncertainty;
  retain explicit scope and require task-specific evidence.

No row is retired merely because an example passes. Retirement requires the stated oracle on
every declared platform/version and corresponding public metadata/documentation.

Priority and executable release selection are related but distinct. The reviewed control policy
has promoted B06, B07, and the macOS/Node/browser portions of B09 into this candidate's required
contexts. Conversely, typing and physical assistive-technology evidence remain gates on those
specific support claims: a stable Python release may remain explicitly untyped, and the private
demo may remain explicitly AT-uncertified.

## Execution evidence matrix

This matrix gives every row below a stable identifier so that release review cannot silently
omit one. It records the repository and live-service audit performed on 2026-08-11 from task
base `dd76815c79076c43d88568ae10f43be7bb546d9c`, including the later PR rehearsals and live
continuation.
`RETIRED` means the repository-side falsifiable gate is green; it does not stand
in for a required hosted-platform run. `ACCEPTED — permanent` means the mitigation gate is green
but the underlying scientific boundary cannot honestly be eliminated. `BLOCKED` names evidence
that this repository and local machine cannot create without the owner/action listed below. The
detailed row remains the source of the acceptance criterion.

| ID | Priority | Tracked limitation | Status / evidence |
|---|---|---|---|
| ENG-P0-01 | P0 | PyPI Trusted Publisher registration | **BLOCKED — authenticated external setup.** The workflow is OIDC-only, token-free, version-absence guarded, and policy-bound, but PyPI has no public settings read API and no authenticated owner capture exists. PyPI 0.15.0/tag `v0.15.0` remain absent. Local SSH signing preflight succeeded, but GitHub key registration still requires live sudo/2FA; no GitHub-verified tag or OIDC publication exists. See B01. |
| ENG-P0-02 | P0 | Live GitHub branch/tag/environment controls | **BLOCKED — settings configured; CUDA acceptance missing.** Live strict protection now has all 23 exact contexts provider-bound to Actions app ID `15368`, and immutable Releases is enabled. Capture `4833e9f2…1b34e5` against current `main` `a9789d0…` has 21 successes and exactly two violations: CUDA minimum/latest failed. `repository_controls_accepted=false` remains correct until genuine hardware evidence succeeds on the final candidate. See B02. |
| ENG-P0-03 | P0 | Non-atomic PyPI/GitHub release recovery | **BLOCKED — authorized drill.** Downstream-only recovery, all-attempt exactly-once inspection, immutable-tag execution binding, cross-service hash equality, retained evidence, and no-republish guards are implemented and tested. No real staged post-PyPI failure/recovery exists; legacy `0.14.0` lacks the required attested source run. See B03. |
| ENG-P0-04 | P0 | Real CUDA and multi-GPU coverage | **BLOCKED — accepted owner-locked Actions evidence.** Dedicated billed/isolation-prepared GCP project `explainiverse-release-ci-26` still has zero VMs/runners and all requested GPU quota paths were denied. The private Kaggle probe linked above proves diagnostic two-T4 capacity exists, but it had no GitHub credential, repository checkout, or Actions runner and ran none of the release test nodes. PR #4 prevents routing from PR/push in the reviewed workflow, while the stricter runner-window procedure requires sole-collaborator authority, an installed-App audit, queue-before-runner registration, and fresh one-job JIT runners. B04 remains open until the four exact attempt-1 Actions jobs supply accepted evidence. See B04 and the current snapshot. |
| ENG-P0-05 | P0 | Dependency resolver/version matrix | **BLOCKED — final-candidate rerun; current-main push green.** Python `31520989339` and dependency `31520989552` succeeded on exact `main` `a9789d0…` across every selected platform and dependency edge. PR #4 security commit `9ad738a…` also has those contexts green, but cannot merge until B04 supplies genuine CUDA evidence; the eventual final-main SHA must be rerun. See B05. |
| ENG-P0-06 | P0 | Quantus versus pandas-floor separation | **BLOCKED — final-candidate rerun; current-main push green.** The exact marker manifest, minimum-lane exclusion audit, explicit Quantus import, and zero-skip reference job are implemented. Python `push` run `31520989339` passed Quantus 9/9 on `a9789d0…`; the security commit awaiting B04 means this is not yet final-candidate evidence. See B05. |
| ENG-P1-01 | P1 | Model-state ownership and extra RNG/state | **RETIRED — declared repository contract; residual boundary permanent.** Adversarial gates cover registered-state traversal, default/injected RNG, callbacks, protocol/fingerprint state, success/error restoration, and serialized adapter operations. Python/NumPy RNG, processes, distributed workers, external mutation, and nondeterministic kernels remain explicitly outside the claim. |
| ENG-P1-02 | P1 | Shared explainer-instance mutation | **RETIRED — repository gate.** All built-in public explainer operations use per-instance re-entrant synchronization; barrier tests cover IG atomic shape commit, DeepSHAP background mutation, TCAV concept mutation, success, and failure. |
| ENG-P1-03 | P1 | Hashed release tools and artifact reproducibility | **BLOCKED — final-candidate/publish binding; current-main push green.** Exact-main `push` run `31520989403` retained report `9113035037` (`sha256:f0cf538830cde7bab7504ca7bec0c1eff218fdb806b98cbe142826d65913c0f2`) and byte-identical wheel/sdist SHA-256 values `972080a1…4011`/`aa234f7b…1105`. PR #4 attempt 2 also passed after attempt 1 exposed hosted-image drift. The final-main rerun and clean-tag publish-byte binding remain required. See B06 and the gate ledger. |
| ENG-P1-04 | P1 | bfloat16 tensor result path | **RETIRED — repository gate.** Owned tensor and DLPack endpoints preserve bfloat16 dtype/values and lifetime; NumPy widening remains explicit and unchanged. |
| ENG-P1-05 | P1 | Failed custom `nn.Module.to()` rollback | **RETIRED — fail-closed repository gate.** Meta moves/models reject before mutation. Standard inherited rollback is tested; failed custom `to`, `_apply`, traversal, or registered-state semantics poison/fail before every later public adapter operation even when rollback returns. No universal atomicity claim remains. |
| ENG-P1-06 | P1 | Explicit image layout/channel axis | **RETIRED — repository gate.** `hw/chw/hwc/nchw/nhwc`, NHW batching, custom channels, metadata, and pre-model rank errors are covered without size heuristics. |
| ENG-P1-07 | P1 | Reused target-layer occurrence selection | **RETIRED — repository gate.** Immutable per-call traces, first/middle/last analytical oracles, dynamic/out-of-range rejection, cleanup, TCAV occurrence identity, and shared-adapter CAM races are covered. Legacy `last_layer_*` fields are compatibility state, not result evidence. |
| ENG-P1-08 | P1 | Captum operator/graph/version support | **RETIRED — declared surface; final-candidate binding blocked.** The support matrix, graph-integrity checks, rejection boundaries, private-rule dependency, and five-file parity suite are explicit. Dependency `push` run `31520989552` passed Captum minimum/current on `a9789d0…`; predecessor exact surfaces were 306/306 with zero skips. PR #4 is green here, but its unmerged security commit requires a later final-main rerun. See B07. |
| ENG-P1-09 | P1 | Versioned Python/JavaScript wire schema | **RETIRED — repository gate; final-candidate binding pending.** The v1 finite-JSON API, fixtures, Python→Node→Python bridge, target/safe-number parity, and JS suite are green. JS `push` run `31520989433` passed Node 20/22, React 18, and browser jobs on `a9789d0…`; Playwright artifact `9113076913` has digest `sha256:0aa14ffc98ad0a3cff6c3bd261d6f822b757c3a2cb4712797230564e4290ae9f`. PR #4 must merge and final `main` rerun. |
| ENG-P1-10 | P1 | Single-operator release approval | **BLOCKED — external governance.** The live environment has only reviewer `jemsbhai` and permits self-review. The publish/recovery paths now generate and retain a validated single-operator governance disclosure with the attested preflight, but neither two-principal approval nor an actual future release record exists. See B08. |
| ENG-P1-11 | P1 | macOS ARM, typing, browser, and accessibility certification | **MIXED — selected current-main contexts green; final-candidate/claim evidence open.** Python `31520989339` and JS `31520989433` passed macOS ARM/OpenMP, Node/React, and zero-retry Playwright 9/9 on `a9789d0…`; PR #4 is green on those contexts but unmerged. Final-main reruns remain required. Strict typing remains unready and explicitly unclaimed; physical NVDA/VoiceOver evidence is absent and explicitly unclaimed. See B09. |
| ENG-P1-12 | P1 | scikit-image next-major compatibility | **BLOCKED FOR WIDENING — upstream/hosted.** The `<1.0` cap remains. Discovery archives exact PyPI metadata and fails while no real non-yanked 1.x prerelease exists; only a separate, explicitly source-only candidate job can produce localisation/LIME/package/tutorial evidence without misrepresenting the capped wheel or a dependency-only graph as 1.x distribution support. See B10. |
| NUM-PERM-01 | Permanent | Finite-real and representability boundary | **ACCEPTED — permanent.** Exact/high-precision near-max, subnormal, cancellation, safe-conversion, and true-out-of-range gates preserve every owned representable case and reject unrepresentable values; the finite-real boundary is not retired. |
| NUM-P1-01 | P1 | Extreme-value CAM variant numerics | **RETIRED — numerical gate only.** LayerCAM, GradCAMElementWise, and EigenGradCAM ordinary/extreme paths match independent high-precision amplitude/projection oracles and reject true overflow without NaN/infinity. CAP-P2-03 remains quarantined. |
| NUM-P1-02 | P1 | Representable scalar versus unrepresentable details | **RETIRED — repository gate.** Opt-in scaled-detail v1 carries exact Decimal/Fraction values, source dtype, and wire-safe integers across every owned counterexample; legacy details still raise `DetailRepresentationError`, and ordinary payloads are unchanged. |
| NUM-P2-01 | P2 | Efficient-MPRT exact-zero entropy | **RETIRED — repository gate.** The guard is exact zero; a symbolic count-domain oracle distinguishes positive entropy below epsilon without impossible allocation. |
| NUM-PERM-02 | Permanent | Finite-estimator uncertainty | **ACCEPTED — permanent/scoped.** Seeded reports disclose streams, counts, estimates, scale-safe Student-t intervals, convergence diagnostics, and `finite_estimate_is_global_proof=false`; lossless binary64 conversion rejects fabricated zero/roundoff. Tutorial 04 demonstrates one estimand, not a universal proof. |
| NUM-PERM-03 | Permanent | Baseline/background/intervention estimands | **ACCEPTED — permanent/scoped.** The dedicated sensitivity comparator requires matching named contracts and exact ordered reference fingerprints, rejects lossy scores/references, and tutorial 04 shows a sign-changing three-reference result. Generic `ExplanationSuite.compare()` remains a caller-asserted display contract and is not claimed as an intervention-sensitivity proof. |
| NUM-P1-03 | P1 | Small-map SSIM boundary | **RETIRED — acceptance gate.** `<3` fails with Pearson/cosine/upstream-aggregation guidance; unequal 3–7 and larger maps use the owned window and match scikit-image where defined. No one-pixel SSIM was invented. |
| NUM-P1-04 | P1 | Consistency cutoff tie policy | **RETIRED — repository gate.** Stable/reject/include-all policies, incidence metadata, deterministic adversarial ties, and mixed-policy rejection are covered. |
| NUM-PERM-04 | Permanent | Fairness/causality/deployment claim boundary | **ACCEPTED — permanent.** Registry/docs/results explicitly define no fairness conclusion, certificate, causal label, best explainer, or deployment recommendation; external domain review remains required. |
| NUM-PERM-05 | Permanent | Fairness extended/undefined statistics | **ACCEPTED — permanent.** Equal constants, unequal constants, insufficient samples, tied Mann–Whitney, and finite variance expose distinct machine-readable defined/reason states without coercing infinity or `None`. |
| AMB-PERM-01 | Permanent | Ambiguous unmarked one-dimensional `{0,1}` output | **ACCEPTED — permanent.** Declared probabilities and labels take distinct correct paths; undeclared one-dimensional and one-column endpoint outputs fail closed in probability-only consumers with declaration guidance. |
| AMB-P1-01 | P1 | Explicit PyTorch score/probability declaration | **RETIRED — repository gate.** Same-shaped raw-score/probability counterexamples honor the declaration, declared probabilities receive range/simplex validation, and undeclared ambiguous matrices fail closed. |
| AMB-PERM-02 | Permanent | ProtoDash near-zero objective mass | **ACCEPTED — permanent.** Exact below/equal/above-threshold metadata works across scales; undefined paths produce neither uniform mass nor `mmd_score`. |
| AMB-PERM-03 | Permanent | Dynamic output-width target mapping | **ACCEPTED — permanent/current boundary.** PDP/ALE pin width and reject changes. No automatic or explicit mapping is fabricated; a future mapping API still requires stable caller semantics. |
| CAP-P2-01 | P2 | Historical fixed-sample `anchors` | **BLOCKED — research; quarantine accepted.** Historical `anchors` remains fixed-sample and uncertified. Separate `anchor_tabular` is continuous-numeric and sequentially certified; categorical inputs fail explicitly. Official parity and mixed categorical/numeric scope remain absent. |
| CAP-P2-02 | P2 | Historical constrained `counterfactual` | **BLOCKED — research; quarantine accepted.** The key is explicitly constrained multistart search, not DiCE. Joint proximity/diversity optimization, official parity, and a supported-model/actionability contract remain absent. |
| CAP-P2-03 | P2 | EigenGradCAM/GradCAMElementWise variants | **BLOCKED — research; quarantine accepted.** Numerical paths are repaired, but primary-formula, score-space, and independent scientific-promotion evidence is absent. |
| CAP-P2-04 | P2 | Score-CAM score-space variants | **BLOCKED — research; distinction accepted.** The current key remains Algorithm-1 raw-score/channel-softmax. A separately named probability-weighted official-code variant pinned to an exact commit, distinct parity, and a deprecation cycle are absent. |
| CAP-P2-05 | P2 | DeepSHAP/Captum quarantined surfaces | **BLOCKED — graph/helper-specific research; quarantine accepted.** Background-distribution support is scoped; inherited single-baseline/comparison helpers and unsupported/dynamic/custom graphs fail closed. DeepLIFT permits shared linear modules but rejects reused nonlinear modules; LRP rejects every shared module. Each new graph/helper still needs its own estimand, analytical evidence, and min/current parity. |
| CAP-P2-06 | P2 | Effective-complexity compatibility aliases | **BLOCKED — research; quarantine accepted.** Aliases warn and delegate to the accurately named threshold count. A genuine Nguyen–Martínez endpoint, formula, perturbation contract, and non-alias evidence are absent. |
| CAP-P2-07 | P2 | Grad-CAM++ | **BLOCKED — research; absence accepted.** No key/alias exists. Higher-derivative adapter support and primary/reference formula evidence remain absent. |
| CAP-P2-08 | P2 | One-logit LRP and unsupported graphs | **BLOCKED — graph/rule-specific research; restriction accepted.** One-logit asymmetric-rule and exact sequential/operator restrictions remain; reused/unsupported graphs reject. Promotion requires selected-score propagation and method-specific conservation/reference parity. |
| CAP-P2-09 | P2 | Experimental JavaScript package | **BLOCKED — publication.** Wire/Node/React/CJS/demo/browser/bundle/private-scope security gates are implemented, but library ESM/browser support, manual AT evidence, npm publication threat/provenance/recovery, and algorithm parity are absent. `private=true`, CommonJS, experimental, and non-parity metadata remain. See B11. |
| CAP-P2-10 | P2 | Tutorial curriculum | **RETIRED — prior three-notebook limitation; ongoing P2 boundary accepted.** Four notebooks now have deterministic offline execution and source/runner/lock provenance; tutorial 04 covers finite-estimator uncertainty and intervention sensitivity. Planned topics remain non-capabilities. |

### Final local, PR, and live-main gate ledger

The predecessor repository snapshot and PR rehearsals below remain historical evidence. Current
`main` `a9789d009f6ec5134bc53b9d2f6a8b59726e75c7` additionally produced the following `push`
evidence. Expected skips are owned by the repository's allowlist. The dedicated CUDA workflow is
guarded by an exact node manifest and zero-skip policy, but no accepted Actions hardware result is
represented.

- Python run `31520989339`, dependency run `31520989552`, JavaScript run `31520989433`, and
  artifact run `31520989403` all completed successfully on exact `main`. Together they supplied
  21/23 provider-bound required successes, including supported Python platforms, dependency
  edges, Quantus 9/9, Captum minimum/current, Node 20/22, React 18, and Playwright 9/9.
- JavaScript run `31520989433` retained Playwright artifact `9113076913`, digest
  `sha256:0aa14ffc98ad0a3cff6c3bd261d6f822b757c3a2cb4712797230564e4290ae9f`.
  Artifact run `31520989403` comparator job `93877997328` retained report `9113035037`, digest
  `sha256:f0cf538830cde7bab7504ca7bec0c1eff218fdb806b98cbe142826d65913c0f2`.
  Its 377,159-byte wheel SHA-256 is
  `972080a133134f4497c83b9069fde95379b171aefa9c756036ef505d30474011`; its 335,212-byte sdist
  SHA-256 is `aa234f7b20d53a4ace957df08f164c34b3729af4f3a6840768c58dce1a2f1105`.
- CUDA run `31520989476` failed closed and produced no GPU evidence. PR #4 security commit `9ad738a…` later
  settled with the same 21 non-CUDA required contexts green; only CUDA minimum/latest failed.
  Its artifact run `31522899822` succeeded on whole-run attempt 2 after attempt 1 detected hosted
  image drift; comparator job `93884787455`.

- Python 3.12.2/Torch 2.10/Captum 0.9: 3,630 passed, four owned CPU CUDA skips, and
  82.16% branch coverage against the 81% threshold. The configured `mypy src scripts` gate,
  Black, isort, compileall, Poetry lock, and installed-graph checks passed. The corresponding
  PR full-quality job passed
  the same 3,630/four-skip surface at 82.18% coverage and verified all four tutorials.
- Exact Captum 0.8/Torch 2.0 and Captum 0.9/Torch 2.10 five-file lanes each passed 306 tests
  with zero skips. The explicit Quantus 0.6 partition passed all nine comparisons with zero
  skips, and the strengthened partition validator passed its adversarial fixture/import tests.
- Fresh wheel/all and sdist/all consumers passed `pip check`, imported every module, and
  enumerated 27 explainers and 131 metrics. The base-only consumer excluded Torch/Captum,
  retained scikit-image, and passed the same import/registry surface.
- JavaScript passed 88 tests, typecheck, lint, build, demo, exact 16-file package-boundary, and
  `npm audit --audit-level=high` with zero vulnerabilities on Node 20.11.1. Chromium, Firefox,
  and WebKit passed all nine browser/axe tests with zero retries. Bundle size was 213,136 bytes
  total and 66,362 gzip bytes of JavaScript. PR run `31513521103` repeated the Node 20/22,
  React 18, and three-engine browser gates successfully.
- All four notebooks were regenerated after source freeze, replayed from fresh kernels, and
  passed the 22-test provenance/output contract. Actionlint and pinned Prettier passed every
  workflow/policy file. Focused release/security/adversarial suites passed. Structural CUDA
  tests proved manifest, topology, class-agnostic EigenCAM, and
  fail-closed collection contracts without claiming hardware acceptance.
- Predecessor draft-PR artifact run `31513521096` attempt 2 used two distinct clean Ubuntu jobs with matching image
  identity to build merge checkout `0d5ca6996e548e6ffb4d89e83aac7fc524ba5dbd`, producing
  byte-identical distributions. The wheel is 377,159 bytes, SHA-256
  `56bd9d021b19ddc0ec4a49cdea67c142ace4b2faaa9df7213d9e94009a7b8746`; the sdist is
  335,201 bytes, SHA-256
  `5e51f2f1bf59bfea28a3f3b84a910709e673dcef34e66a42bb8fffdc1ca850a3`.
  Current-main run `31520989403` now supplies landed `push` evidence for `a9789d0…`; a rerun on
  the eventual final candidate and the later publish-byte binding remain B06.

### Live evidence and blocker ledger

The latest authenticated capture was made as repository administrator `jemsbhai` at
`2026-08-11T18:39:00.267775+00:00` against `origin/main`
`a9789d009f6ec5134bc53b9d2f6a8b59726e75c7`. Its JSON SHA-256 is
`4833e9f245c4d30f3753fd9976e8b540057e8ffbb187cbdccae9b9b7611b34e5`; its reviewed-policy
SHA-256 is `e6d8e4de111f5efdcecfef726e9c4f4a526eebd0d16520c426d9405aca03a443`.
Strict protection has all 23 contexts bound to GitHub Actions app ID `15368`, and immutable
Releases is enabled. The capture reports exactly two violations because CUDA minimum/latest were
completed failures rather than successes; `repository_controls_accepted=false` remains correct.

For history, the predecessor capture at `2026-08-11T17:16:13.612560+00:00` against
`49e962c090e90e62f315837067e5adc3e3f04d1c` had JSON SHA-256
`e2f704a4996056de8dda2c7977f55c4dd55135c7054b6ecbf0ac721c9121fc1e` and reported 19
violations: 10/23 contexts, 16 missing provider-bound exact-commit checks, and immutable Releases
disabled. The B02 mutations reduced that exact gap to the two genuine CUDA results without
weakening the policy. Repository and `pypi`-environment secret-name inventories remain empty.

Historical authenticated read-only API observations at `2026-08-11T17:17:51.9860715Z` reported
Actions variables `total_count: 0`, registered runners `total_count: 0`, and six repository-level
workflow registrations while only the original three workflow files were present on `main`.
After the reviewed PR chain landed, ten workflows are active and the release workflows are in the
`main` tree; variables and runners remain zero, and external-contributor workflow approval is now
`all_external_contributors`.

The reviewed PR chain is now landed with the ancestry recorded in the current snapshot. PR #4
contains security commit `9ad738a2dc62777ae199b03a3348b24b45006da2` and remains open because its public-repository security
boundary deliberately prevents PR/push execution on self-hosted GPU runners and no owner-locked
JIT Actions dispatch has run. Live Actions variables and registered runners remain zero.
Dedicated GCP project `explainiverse-release-ci-26`/`305968033598` is billed and
isolation-prepared but has zero VMs; all quota requests listed in the current snapshot were
denied. The private Kaggle probe proves two-T4 diagnostic capacity, but it had no credential,
repository checkout, runner registration, or release test execution. The missing owner-locked
JIT Actions evidence keeps B04 open and blocks PR #4, final-main reruns, accepted B02 capture, and
all publication-dependent steps. No tag, publication, release, recovery drill, accepted Actions
CUDA evidence, or physical accessibility evidence exists.

PyPI `0.14.0` is legacy incident evidence, not a recovery proof: its wheel SHA-256 is
`b1b98dfdfc0acbc8dc2113d8db87d40ae9cec2f958ed25b00bc6e30d43db41d4`, its sdist SHA-256 is
`e2ab525f720d9970f25c307be84b9a5a6bb5feb612a4457ba9d72925cf2af68b`, annotated tag object
`ffc3a75c0bdbf8feccbc60ffa451f5cc919dbaaa` targets
`49e962c090e90e62f315837067e5adc3e3f04d1c`, the tag is unsigned/unverified, and the GitHub
Release endpoint returns 404. There is no retained build/attestation/OIDC source run from which
the new recovery workflow could truthfully reconstruct provenance.

Fresh absence checks at `2026-08-11T18:42:57.9011768Z` found PyPI 0.15.0 absent and the GitHub
`refs/tags/v0.15.0` endpoint returned 404. The exact PyPI Trusted Publisher still lacks an
authenticated owner capture. A local SSH annotated-tag preflight succeeded with Ed25519 public
key fingerprint `SHA256:84G7/ewIxErnHPmIrtaW52+1qy+2MlQqzbmCOW6tGc0`; the test tag was deleted,
and GitHub key registration remains blocked at live sudo/2FA, so no verified signature is claimed.
No actual governance record or recovery evidence exists.

| Blocker | Rows | Owner | Required authority/action | Reproducible acceptance procedure |
|---|---|---|---|---|
| B01 | ENG-P0-01 | PyPI project owner `jemsbhai` | Authentically register/archive the exact owner/repository/workflow/environment Trusted Publisher and GitHub-verifiable signing key; then use the already-authorized sole OIDC release only after upstream gates pass. | Capture the authenticated PyPI Publishing settings for `jemsbhai/explainiverse`, `publish-pypi.yml`, environment `pypi`; then require a token-free OIDC upload whose per-file Integrity records and cryptographic verification prove the exact DSSE subjects/digests and Trusted Publisher identity. Repository/environment secret inventories must remain empty. Local-only signing preflight is insufficient. |
| B02 | ENG-P0-02 | GitHub repository administrator | Preserve immutable Releases and all 23 exact provider-bound contexts; obtain genuine CUDA success on final `main`, enforce sole-collaborator/zero-invitation human authority, retain an installed-App permission audit, then repeat the capture. | Run the documented admin capture, dispatch `release-preflight.yml` on exact `origin/main`, bind within 30 minutes with actor and triggering actor both equal to the authenticated capture principal, and verify the retained run attempt/triggering actor against the Actions API immediately before tag creation. Acceptance is `repository_controls_accepted=true`, immutable Releases enabled, sole collaborator/effective writer `jemsbhai`, zero invitations, no violations, exact policy/snapshot digests, every required provider-bound check successful, and a separate authenticated export showing no non-owner-equivalent App/automation authority. Strengthened capture `ab182478…693da` has exactly three violations: current collaborator `b-urge` and the two CUDA failures; it records zero invitations. |
| B03 | ENG-P0-03 | Separately authorized release operator | Exercise a staged failure only on a future build-attest-OIDC source run; do not reuse `0.14.0`. | Dispatch with `stage_recovery_drill=true`, then run `recover-github-release.yml` from the immutable release tag and original run ID with `require_staged_drill=true`. Archive source-run/all-attempt job JSON, PyPI JSON, inventories, and hashes; require exact tag/SHA execution, the failed stage step, exactly one successful upload execution, and byte-identical original/PyPI/GitHub artifacts. |
| B04 | ENG-P0-04 | Repository administrator, GPU-capacity account owner, and GPU-infrastructure owner | Convert the discovered two-GPU Linux capacity into an approved owner-locked Actions window. Coordinate restoration, temporarily remove non-owner collaborators, require zero invitations, audit installed Apps/automation, and clear custom-label queues. Set variables and owner-dispatch each reviewed ref while no runner is online; verify only the exact expected jobs are queued, then register fresh one-job JIT runners without exposing a reusable credential. | Retain authority/queue evidence plus all four named min/latest one-/two-GPU jobs. Each must carry its topology label, match and execute the exact 15-node manifest with the exact declared visible-device count, zero skips, exact-commit identity, and exactly one successful attempt on its own positive, distinct runner ID. Keep the authority lock through publication. On success or failure, prove variables, queues, runners, VMs, disks, and capacity-side credentials are zero before re-inviting the exact prior permission and recording acceptance. The private Kaggle inventory is diagnostic only and does not satisfy this. |
| B05 | ENG-P0-05, ENG-P0-06 | CI/merge authority | After B04 permits PR #4 to merge, repeat the currently green Python/dependency `push` workflows on final `main`. | Require exact-final-candidate green Python 3.10–3.13, direct-floor, SHAP/XGBoost edge, Captum 0.8/current, complete non-Quantus minimum partition, and explicit-import Quantus reference partition with zero skips. |
| B06 | ENG-P1-03 | Release-CI owner | Repeat byte reproducibility on final `main`, then bind clean-tag publish bytes to that exact accepted run. | Both jobs install the same hash-locked tool graph and record matching source, Python and pip versions, platform family/architecture, tool, lock, run, attempt, image, OS, and architecture fields with distinct matrix slots, job indexes, and build identities. The comparator must prove byte-identical wheel/sdist and retain both manifests; publication must prove its distribution byte-identical to both builds before upload. |
| B07 | ENG-P1-08, CAP-P2-05, CAP-P2-08 | Captum/CI owner | Repeat the currently green exact Captum 0.8/current five-file surface on final `main` after PR #4 merges and after every graph-integrity change. | Explicitly import Captum, execute the command in `docs/CAPTUM_SUPPORT_MATRIX.md`, and require zero skips plus analytical/parity/fail-closed graph gates before changing any supported surface. |
| B08 | ENG-P1-10 | Project governance/release manager | Add a second trusted principal or retain explicit single-operator disclosure. | Either record two distinct approver/executor identities, or publish a release record whose validated governance section states single-operator approval and binds the actor, reviewer/self-review setting, tag/commit, preflight run, and archived external-control policy/snapshot digests. |
| B09 | ENG-P1-11 | Platform, typing, JS, and independent accessibility owners | Run the policy-required macOS ARM/Node/browser contexts. Complete typing before `py.typed`; collect physical AT evidence before an AT-support claim. | Stable-release contexts require exact-candidate macOS ARM and Node/React/browser success. The separate typed claim requires strict mypy zero, Pyright 100%, and strict clean installed-wheel consumers. The separate AT claim requires both policy profiles (Windows 11/Edge/NVDA and Apple-Silicon macOS/Safari/VoiceOver), every scenario passing, exact commit/deployment/build hash, HTTPS transcript/recording URIs with SHA-256, age at most 180 days, validator/workflow success, and human hash review. |
| B10 | ENG-P1-12 | Dependency maintainer; upstream supplies the candidate | Keep `<1.0` until a non-yanked scikit-image 1.x prerelease exists. | Let the scheduled discovery job archive PyPI metadata/hash/serial and choose the candidate. Absence fails the monitor and leaves the source probe non-green. A real candidate must pass the dependencies-only graph check, localisation, LIME, capped-wheel boundary audit, and all four source-only tutorials with retained environment/test/artifact evidence. That probe explicitly records `full_distribution_graph_verified=false`; a separate reviewed bound-change wheel must then install with the candidate and pass full `pip check` before widening. |
| B11 | CAP-P2-09 | JS, security, release, and accessibility owners | Add direct-library ESM/browser consumers, publication threat model/release/provenance/recovery, physical AT evidence, and an explicit parity decision. | Every criterion in the detailed CAP-P2-09 row must be exact-commit green before changing `private`, module format, `publicationReady`, or algorithm-parity metadata. |

## Engineering and release limitations

| Priority | Residual limitation and current control | Mitigation work | Falsifiable retirement/acceptance gate |
|---|---|---|---|
| P0 | PyPI Trusted Publisher registration is external to the repository. The `pypi` GitHub environment is protected and the workflow has no token fallback. | Register owner `jemsbhai`, repository `explainiverse`, workflow `publish-pypi.yml`, environment `pypi` in PyPI; perform a non-production OIDC preflight where PyPI supports it. | PyPI project publishing settings list that exact publisher; the next separately authorized release reports Trusted Publishing and accepts no stored API token. |
| P0 | GitHub branch protection, immutable-tag rules, required checks, and environment reviewers are external mutable settings rather than versioned repository state. | Before every stable release, query and archive the `main` protection, `v*` tag ruleset, `pypi` environment, and required-check contexts; fail the release checklist if they differ from the reviewed policy. | A machine-readable preflight records admin enforcement, strict required checks, resolved-conversation requirement, no force-push/delete, immutable `v*` tags, the `pypi` environment's tag-only `v*` deployment rule, disabled administrator bypass, and the expected reviewer immediately before tagging. |
| P0 | PyPI publication and the later GitHub Release creation are not one atomic transaction. A fresh full rerun after PyPI succeeds must not attempt to republish. | Retain immutable build artifacts long enough to rerun only failed downstream jobs; document an operator recovery drill. Evaluate a draft/finalize release flow only if it verifies the exact PyPI artifact hashes before reuse and never relies on unchecked `skip-existing`. | A staged failure after PyPI publication is recovered in a drill using the original attested artifacts; GitHub Release hashes equal PyPI hashes and no upload job runs twice. |
| P0 | CUDA is not release-accepted on GitHub Actions. A private diagnostic inventory found two T4 devices, but it received no repository credential or code, registered no runner, and ran no release test. CPU and structural all-device RNG tests likewise cannot prove real kernels or multi-GPU restoration. | Run the required single-GPU and scheduled two-GPU jobs through owner-locked, fresh one-job JIT runners, covering adapter prediction/gradients, every gradient family, randomisation success/failure, initialized-device RNG byte identity, dtype/device placement, and hook cleanup. | All CUDA tests pass on supported Torch minimum/latest in four exact-commit, attempt-1 Actions jobs with zero unexpected CUDA skips and distinct runner IDs. Until then registry scopes remain CPU-verified. |
| P0 | Minimum direct dependencies and current lock are tested, but one environment cannot cover every resolver combination. | Keep Python 3.10–3.13 latest jobs, the exact Python 3.10 direct-floor job, Captum 0.8/current probes, old/new SHAP output forms, and XGBoost floor/current cases. Add a scheduled constraints matrix before widening any bound. | Every declared lower bound has a substantive public-surface test and each upper-bound change lands with a green compatibility job. |
| P0 | Quantus 0.6 cannot coexist with the exact pandas 1.5.0 floor. Minimum CI deselects only tests carrying the registered `quantus_reference` marker; the all-extras/reference jobs own parity. | Keep pure metric contract tests importable without Quantus and make every official Quantus comparison mandatory in the reference environment. | Reference job imports Quantus explicitly and all official comparisons run with zero skips; minimum direct-floor job runs all tests not marked `quantus_reference`. |
| P1 | Same-model state contexts serialize participating calls and restore each module's training flag, the contents/storage layout of registered buffer objects that existed on entry, existing parameter `.grad` objects/values, and Torch's default CPU/CUDA RNG. They do not snapshot arbitrary Python attributes, parameter values or rebinding, buffer rebinding, caller-owned/custom `torch.Generator` objects, Python/NumPy RNG, external libraries, subprocesses, distributed workers, or nondeterministic kernels. | **Owner: model-state workstream.** Keep a pure-forward/state-ownership contract as the default. Add explicit generator injection and an opt-in state protocol or pre/post fingerprint for models that declare additional owned state; expand value/binding restoration only where ownership and copy cost are explicit. | Adversarial modules that mutate every named exclusion either restore it exactly after success and exception or fail before returning an explanation with the unsupported mutation identified. Process/distributed probes precede any expanded claim; no gate asserts universal model-call atomicity. |
| P1 | Per-model serialization protects shared model state, not mutable fields on a shared explainer instance. In particular, Integrated Gradients infers and commits `input_shape` after a successful call, so concurrent first calls with different shapes can race; background/concept stores and other mutable explainer configuration also require ownership discipline. | **Owner: explainer-state workstream.** Document one-instance-per-worker or caller locking now. Audit all persistent explainer fields, then use immutable configuration, explicit mutation APIs, or per-instance reentrant locks with atomic validate-and-commit semantics. | Barrier-controlled same-instance tests cover concurrent success/failure and same/different shapes. Exactly one compatible IG shape is committed, no failed call leaves partial state, mutable background/concept operations cannot interleave with reads, and repeated schedules match an allowed serial execution. |
| P1 | The release workflow pins top-level Poetry/Twine/CycloneDX versions and records the resolved build environment, but their transitive installer graph and hosted runner image are not hash-locked. Build-once artifacts are stable within a run; a later rebuild is not claimed bit-reproducible. | Check in a Python-3.12/Linux release-tool requirements file with every transitive artifact hash; install with `pip --require-hashes`; record bootstrap pip and runner-image identities; add a two-run artifact reproducibility comparison. | Two clean runners install the identical hashed tool graph and either produce byte-identical wheel/sdist files or publish documented, explained reproducibility differences. |
| P1 | bfloat16 has no NumPy dtype, so public NumPy results are widened to float32. | Add an opt-in tensor-return or DLPack result path with an ownership/lifetime contract. | Round-trip bfloat16 endpoint tests preserve dtype and values without NumPy; existing NumPy endpoints continue to disclose widening. |
| P1 | A failed arbitrary custom `nn.Module.to()` can mutate before raising and may be impossible to roll back (notably meta tensors). | Fail loudly when rollback fails, document standard-module/custom-`to` requirements, and investigate a preflight/copy strategy only for models that explicitly support it. | Standard modules pass transactional success/failure tests. Custom rollback failure is detected before another prediction and instructs the caller to reconstruct; no claim of universal atomicity remains. |
| P1 | Two-dimensional IG assumes one implicit NCHW grayscale channel; CAM auto-layout supports only unambiguous edge-channel shapes. | Add explicit `channel_axis`/layout configuration shared by image explainers and preserve it in metadata. | CHW, HWC, HW, NHW and custom-channel tests either map exactly to one declared model input or fail before model work; no size-based silent guess. |
| P1 | Shared/recurrent target layers are rejected; implicit first/last selection is not scientifically neutral. | Add an explicit occurrence selector backed by traced call counts, with per-occurrence activations/gradients and cleanup tests. | First, middle and last occurrence analytical oracles pass; out-of-range/dynamic counts fail; default remains fail-closed. |
| P1 | Captum-backed DeepLIFT, DeepSHAP, and epsilon/gamma/z-plus LRP inherit Captum's restricted operator, graph, and version support. DeepSHAP's inherited single-baseline and ordinary-IG comparison helpers are deliberately quarantined/blocked; unsupported graphs, reused nonlinear DeepLIFT modules, and every shared LRP module must fail rather than approximate silently. Shared linear DeepLIFT modules remain within the declared surface. | **Owner: Captum parity workstream.** Maintain minimum/current Captum jobs, remove private API reliance where an upstream public hook exists, publish a method-by-operator support matrix, and retain explicit errors for every quarantined helper and graph. | Each supported method/rule passes analytical plus Captum parity on every declared version. The named DeepSHAP helpers remain `NotImplementedError` until a background-expected definition and oracle exist; unsupported graphs fail before returning attribution. Unquarantine only the individually evidenced surface. |
| P1 | Python `Explanation.to_dict()` intentionally preserves NumPy and arbitrary Python payload types and is not JSON serialization. | If demand exists, add a separately named `to_wire_dict()`/`from_wire_dict()` with recursive finite-JSON validation and a versioned schema; do not change `to_dict()` silently. | Python→JSON→JS→JSON→Python producer/consumer tests round-trip the schema exactly, including rejection fixtures. |
| P1 | The project currently has one release operator. The protected `pypi` environment therefore uses the maintainer as its reviewer and permits self-review; although deployment-time administrator bypass is disabled, administrators can still change external settings. This is not segregation of duties. | When a second trusted maintainer exists, require an independent environment reviewer and prevent self-review. Until then, preserve signed tags, immutable artifacts, settings snapshots, and a public release audit trail. | Two distinct principals approve and execute a staged release, or the release record explicitly discloses single-operator approval and includes the archived external-control preflight. |
| P1 | macOS/ARM, package typing, real-browser, and assistive-technology behavior are uncertified. | Add macOS ARM CI, publish `py.typed` only after public annotations pass strict consumer mypy/pyright, and run Playwright plus screen-reader/manual accessibility checks for the demo. | Dedicated gates are required and green before those platforms/capabilities appear in supported claims. |
| P1 | `scikit-image` is required by base LIME and future major versions are untested. | Keep the `<1.0` cap; test the next major prerelease before widening. | A prerelease/current compatibility job passes localisation, LIME, packaging, and tutorial gates before the bound changes. |

## Numerical and statistical limitations

| Priority | Residual limitation and current control | Mitigation work | Falsifiable retirement/acceptance gate |
|---|---|---|---|
| Permanent boundary | The supported finite-real contract excludes IEEE NaN and infinities, and a genuinely out-of-range real result cannot be represented as a finite value in the selected dtype. Current reductions rescale representable finite cases and reject unsupported or nonfinite results. | Continue scale-before-reduction design; add adversarial subnormal/near-max tests to every new formula. | Acceptance, not retirement: every mathematically representable scale transform matches a higher-precision/manual oracle; unrepresentable cases fail explicitly, never fabricate a finite score. |
| P1 | LayerCAM uses a scale-safe fused product-sum when each final float64 CAM cell is representable. Quarantined `gradcam_elementwise` and `eigengradcam` currently materialize `gradient * activation`; extreme finite operands can overflow or underflow before their later ReLU/sum or principal-projection step, even when a reformulated result might be representable. | **Owner: CAM numerical workstream.** Keep LayerCAM's fused boundary explicit. Develop scaled ReLU-sum and scaled projection/SVD formulations for the two variants without changing their formulas, and fail explicitly on any nonrepresentable intermediate until those formulations are proved. | Near-max, subnormal, cancellation, and genuinely out-of-range fixtures match a higher-precision analytical oracle for LayerCAM. The two variants return no `NaN`/infinity; their supported scope expands only after scaled implementations match direct formulas in the ordinary range and high-precision ReLU/projection references at extremes. |
| P1 | A scalar aggregate can be representable even when one requested float64 detail element is not. This occurs for individual prediction drops, squared errors, normalized curve points, or attribution aggregates whose later cancellation/scaling makes the scalar valid. Current scalar modes keep exact arithmetic through the aggregate; detail modes fail explicitly instead of serializing `NaN`, infinity, or a fabricated value. | Define a versioned scaled-detail representation carrying mantissa/exponent or an exact decimal string plus dtype metadata. Keep the existing float-array detail schema fail-closed until consumers can opt into that representation. | Every scalar/detail counterexample has a high-precision oracle. Scalar mode succeeds whenever its final result is representable; legacy detail mode raises a typed, detail-specific error; the opt-in schema round-trips every finite exact element and never changes ordinary float payloads. |
| P2 | Efficient MPRT treats histogram entropy less than or equal to machine epsilon as degenerate. A positive empirical histogram entropy that small requires on the order of `10^17` samples, beyond realizable ndarray memory, so no feasible current input reaches the distinction between zero and that threshold. | Replace the epsilon guard with exact-zero detection during a future Efficient-MPRT cleanup, and retain a synthetic count-domain oracle that does not allocate the impossible array. | A symbolic/count-based fixture distinguishes exact zero from positive entropy below epsilon, ordinary finite arrays are unchanged, and no degenerate normalization divides by zero. |
| Permanent boundary | SAGE, SHAP, LIME, sampled sensitivity/stability, randomisation, and perturbation metrics are finite estimators, not global proofs. | Expose seed, sample count, convergence diagnostics, replicate estimates, and confidence intervals where statistically justified; require sensitivity runs in tutorials. | Acceptance: repeated-seed studies report uncertainty and convergence; metadata never upgrades a finite estimate to a universal faithfulness/robustness claim. |
| Permanent boundary | Baselines, background distributions, feature dependence, masks, and off-manifold interventions define the estimand. No universal default is correct. | Add multi-baseline/background sensitivity helpers and require explicit intervention metadata in comparative suites. | Acceptance: comparisons fail without a shared intervention contract; reports show how conclusions vary across prespecified plausible references. |
| P1 | SSIM has no valid sliding-window result for a spatial axis smaller than 3. Current API rejects it. | Offer documented Pearson/cosine alternatives or caller-controlled upstream aggregation; do not invent a one-pixel SSIM. | Acceptance: <3 inputs fail with the alternative guidance; 3–7 and larger maps match the owned-window oracle and scikit-image where domains overlap. |
| P1 | Consistency cutoff ties depend on a policy. Current behavior is stable feature order. | Add detail-mode tie incidence and optional reject/include-all policies while recording the selected policy. | Every comparison carries one policy; adversarial ties give deterministic documented results and mixed-policy comparison is rejected. |
| Permanent boundary | Fairness-related metrics are diagnostics and cannot certify fairness; explanation scores cannot establish causality or deployment usefulness. | Require domain review, outcome/measurement analysis, subgroup uncertainty, and explicit decision-impact evaluation outside this library. | Acceptance: registry/docs/results never emit a fairness certificate, causal label, “best explainer,” or automatic deployment recommendation. |

| Permanent boundary | Fairness statistics retain two explicit extended/undefined conventions: unequal groups with exactly zero pooled variance report signed-infinite Cohen's d, while a completely tied pooled sample reports no Mann-Whitney p-value. These are mathematical boundary signals, not ordinary finite estimates. | Add explicit `effect_size_defined` and reason metadata in the next result-schema revision and require consumers to branch on it; never coerce either case to zero. | Acceptance: constant-equal, constant-unequal, and fully tied counterexamples expose distinct machine-readable states; finite-variance cases remain finite and exact, and reports label infinity or `None` as boundary states. |

## Ambiguity and defined-result limitations

| Priority | Residual limitation and current control | Mitigation work | Falsifiable retirement/acceptance gate |
|---|---|---|---|
| Permanent boundary | An unmarked one-dimensional or one-column numeric `{0,1}` array is observationally compatible with both endpoint probabilities and hard labels. | Require custom adapters to declare `prediction_output_kind`; keep legacy unmarked heuristics fail-closed in probability-only consumers. | Acceptance: declared probability/label counterexamples take different correct paths; an ambiguous undeclared endpoint is rejected with migration guidance. |
| P1 | `PyTorchAdapter(output_activation=None)` deliberately leaves multiclass output kind undeclared because the same matrix shape can carry raw scores or already-normalized probabilities. Value heuristics cannot prove which meaning the model intended. | Add an explicit caller declaration for score versus probability output, validate declared probability range/simplex constraints, and deprecate consumers that infer semantics from values alone. | Raw-score and normalized-probability counterexamples with the same shape take their declared paths; undeclared ambiguous matrices fail closed with migration guidance. |
| Permanent boundary | ProtoDash canonical objective mass less than or equal to the configured `epsilon` does not define normalized weights or a distributional MMD, even when that mass is positive but near zero. Current output preserves all-zero display weights with `normalized_weights_defined=False`. | **Owner: ProtoDash contract workstream.** Keep consumers branching on `mmd_defined`; expose the normalization threshold in result metadata and use a threshold-accurate undefined reason rather than describing every case as exactly zero mass. | Acceptance: masses below, equal to, and just above `epsilon` take the documented paths under several scales; no undefined path substitutes uniform mass or emits `mmd_score`, and every payload discloses the threshold used. |
| Permanent boundary | Dynamic model output width has no automatic target mapping. PDP/ALE now pin width per call and reject changes. | Support only an explicit caller-provided output mapping if a future model family has stable semantics. | Acceptance: width changes never silently select/relabel an output; an explicit mapping is validated against every perturbed call. |

## Quarantined, absent, and experimental capabilities

| Priority | Capability | Mitigation/retirement criteria |
|---|---|---|
| P2 | Historical `anchors` | Keep quarantined as fixed-sample compatibility search. Unquarantine only a separately named algorithm with sequential confidence certificate, budget-exhaustion behavior, primary-paper/reference parity, and categorical/numeric scope tests. |
| P2 | Historical `counterfactual` | Keep labeled constrained search, not DiCE. A DiCE key requires differentiable joint proximity/diversity optimization, supported-model contract, immutable/actionable constraints, and official/analytical reference oracles. |
| P2 | `eigengradcam` and `gradcam_elementwise` | Keep library variants quarantined and never cite them as canonical paper methods. Promotion requires a primary formula, score-space contract, and independent oracle; otherwise retain the variant names. |
| P2 | Score-CAM score-space variants | The verified `scorecam` key remains the paper Algorithm-1 transcription using raw target scores and a channel softmax; it must not be silently conflated with the authors' released probability-weighted implementation. **Owner: CAM reference workstream.** If demand exists, add the released behavior under a separate variant key pinned to an exact official commit. Retire the distinction or migrate an alias only after both keys pass distinct same-mask/channel analytical counterexamples, official-reference parity in their declared score spaces, explicit metadata, and a documented deprecation cycle. |
| P2 | DeepSHAP/Captum quarantined surfaces | Keep DeepSHAP's inherited single-baseline/comparison helpers and unsupported Captum graph/operator cases blocked. The verified DeepSHAP key remains scoped to its background-distribution API on explicitly supported Captum graphs. **Owner: Captum parity workstream.** A helper or graph leaves quarantine only with a stated estimand, public-backend/minimum-current parity, analytical conservation/completeness evidence where applicable, and fail-closed unsupported counterexamples. |
| P2 | `compute_effective_complexity` compatibility aliases | Keep quarantined. A genuine Nguyen–Martínez implementation needs its own endpoint, formula/reference tests, perturbation contract, and non-alias metadata. |
| P2 | Grad-CAM++ | Remain absent until the adapter exposes the necessary higher derivatives and the general formula matches primary/reference counterexamples. Do not restore a first-derivative approximation under that name. |
| P2 | One-logit LRP/unsupported graphs | Retain explicit class/rule/operator restrictions. Expand only through actual propagation of the selected score graph plus conservation and direct-reference parity; never infer class 0 by negating a sign-asymmetric rule result. |
| P2 | Experimental JavaScript package | Keep private, CommonJS-only, and limited to contracts/visualization. Before publication: versioned wire schema, exact Node 20.11 and React 18 peer-floor CI, ESM/browser matrix, security review, bundle budget, real accessibility gate, and an explicit decision whether algorithms will ever claim Python parity. |
| P2 | Tutorial curriculum | Four notebooks are verified, including finite-estimator uncertainty and intervention sensitivity. Promote one additional topic at a time with offline data, formula/output-space assertions, deterministic fresh-output equality, and current source/lock/runner provenance. Planned rows remain non-capabilities. |

## Execution order

1. Close every P0 row and every policy-selected stable gate, then rerun the full release gate
   before any stable tag.
2. Land P1 work as independent evidence-bearing changes; do not bundle platform claims with
   algorithm additions.
3. Keep P2 APIs quarantined/absent until their retirement criteria exist before implementation.
4. Re-audit permanent boundaries every release for truthful wording, not for impossible
   “elimination.”
