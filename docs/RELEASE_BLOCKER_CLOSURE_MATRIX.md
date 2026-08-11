# Stable-release blocker closure matrix

This is the current closure ledger for B01-B11. It was re-audited from clean commit
`dd76815c79076c43d88568ae10f43be7bb546d9c` on 2026-08-11. A repository guard can make
future evidence falsifiable, but it cannot close a live-service, hosted-runner, hardware, or
human-review blocker. `BLOCKED` therefore remains the only honest state until the acceptance
column is satisfied with direct evidence from the exact candidate commit. B10 governs a future
bound widening and B11 is an intentionally retained P2 quarantine, so their blocked states are
not stable-release failures while those surfaces remain unclaimed.

The stable-release recommendation is **NO**. In particular, every P0 external/hosted evidence
row remains open. No push, merge, tag, publication, release creation, settings mutation, staged
recovery drill, or representation of manual/GPU evidence was authorized or performed during
this audit.

## Fresh control observations

- The authenticated GitHub capture was made by repository administrator `jemsbhai` at
  `2026-08-11T08:39:57.921139+00:00` against `origin/main`
  `49e962c090e90e62f315837067e5adc3e3f04d1c` and planned tag `v0.15.0`. Snapshot SHA-256:
  `9e1be49a8aae8c6ef477297fe728d6907e55f5f609e01c6d718fb0bd785a6679`; reviewed-policy
  SHA-256: `dca9a3eeb1ad54d5931b15251d4685a7720bafed955c6937c7839291041c87db`.
- The capture is rejected with 19 violations. Live `main` protection has 10 required contexts,
  while the reviewed policy has 23. All 10 live contexts are bound to GitHub Actions app ID
  `15368`; 16 policy checks have no successful exact-SHA check run from that provider. Only the
  old Deploy Demo, JS CI, and Python CI workflows are registered on `main`; the audited commit is
  absent from GitHub. Actions variables, registered runners, repository secrets, and `pypi`
  environment secrets each have count zero. The repository-level immutable Releases control is
  disabled and is not enforced by the owner.
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
| B01 | **BLOCKED — PyPI owner/OIDC** | PyPI project owner `jemsbhai` | Directly verify or register the exact Trusted Publisher for owner `jemsbhai`, repository `explainiverse`, workflow `publish-pypi.yml`, environment `pypi`; separately authorize one release. | Archive the authenticated settings record. On the exact candidate, the sole publisher uses OIDC with no token/user/password/secret fallback; each exact PyPI file has Integrity provenance whose DSSE subject/digest and GitHub publisher repository/workflow/environment pass the hash-locked cryptographic verifier; repository and environment secret-name inventories remain empty. | Owner settings could not be read without login. Public 0.14.0 provenance is absent and 0.15.0 is unused. The repository now adversarially enforces the token-free publisher and provenance shape, but no authorized OIDC upload exists. |
| B02 | **BLOCKED — live GitHub controls/checks** | GitHub repository administrator and CI/merge authority | Land the reviewed workflows, make all 23 policy contexts required and bound to GitHub Actions app ID 15368, enable repository-level immutable Releases, and run every gate on the exact candidate. | A capture no older than 30 minutes has `repository_controls_accepted=true`, zero violations, immutable Releases enabled, exact policy/snapshot digests, exact context/app bindings, and one successful provider-bound check per required name on the candidate SHA; actor and triggering actor both equal the capture principal, and the retained run attempt/triggering actor match the Actions API source run. | Fresh capture is rejected: 10 live versus 23 policy contexts, 16 exact-SHA provider-bound runs missing, immutable Releases disabled, 19 violations. The audited commit/workflows are not live. |
| B03 | **BLOCKED — authorized recovery drill** | Separately authorized release operator | On a future build/attest/OIDC run, request the deliberate post-PyPI failure, then recover only the downstream GitHub Release from that original run. Do not reuse 0.14.0. | Source run concludes failure; build, attest, and publish each have exactly one successful attempt; the release job fails at the explicit drill step with later release steps skipped. Recovery verifies attestations and proves original, PyPI, and GitHub files byte-identical without a second upload. Retain source/all-attempt jobs, inventories, hashes, service JSON, and recovery evidence. | Verifier now distinguishes a staged drill from an unplanned downstream failure and rejects successful, missing, skipped, or duplicate source evidence. No authorized live drill exists; legacy 0.14.0 cannot satisfy it. |
| B04 | **BLOCKED — hosted one-/two-GPU evidence** | Repository administrator and GPU-infrastructure owner | Provision approved isolated runners, set `CUDA_SINGLE_RUNNER` and `CUDA_TWO_RUNNER`, and dispatch `cuda-ci.yml` on the candidate. | All four minimum/latest one-/two-GPU jobs complete successfully exactly once on the candidate, with declared device count, exact expected 15-node manifest, 15 executed, zero skips, and retained job/runner evidence bound into preflight. | Live variables/runners are zero, so no authorized hosted hardware record exists. The exact node-manifest and topology guards make silent suite or device-count erosion fail. |
| B05 | **BLOCKED — hosted dependency matrix** | CI/merge authority | Land and run Python 3.10-3.13 plus direct-floor, Captum, SHAP/XGBoost, and current/floor dependency jobs; require their contexts. | Every declared edge is green on the exact candidate, with the correct substantive test surface, resolved graph, `pip check`, and no unowned skip. | Local Python/floor lanes remain green. Captum 0.8/Torch 2.0 and Captum 0.9/Torch 2.10 each pass the five-file 306-test suite with zero skips. Six dependency jobs are now automatic required-policy contexts, but none has run on the audited candidate in GitHub. |
| B06 | **BLOCKED — two-host reproducibility** | Release-CI owner | Execute the artifact reproducibility workflow on two clean hosted Linux/Python-3.12 jobs. | Both builds use the same source SHA, Python/pip/platform, locked tools, runner image/OS/architecture, GitHub run, and attempt; matrix slots, job indexes, and build identities are distinct; wheel and sdist bytes match. Runner display names are retained but need not differ. The report retains both complete manifests and artifact hashes. | Repository comparison now rejects source, environment, runner, tool, lock, or identity drift and retains mismatch evidence. No two-hosted-runner execution exists. |
| B07 | **BLOCKED — hosted Captum contract** | Captum/CI owner | Run the mandatory five-file surface under exact Captum 0.8 and current after graph-integrity changes, locally and in required hosted jobs. | Both versions explicitly import Captum and pass all analytical, reference-parity, restoration, and fail-closed graph tests with zero skips before any surface is widened. | Both local 306-test version lanes are green with zero skips. The Quantus/fixture partition guard was hardened for the audited static import/fixture/helper patterns so those paths cannot silently erode the floor lane. Exact-candidate hosted Captum evidence is absent. |
| B08 | **BLOCKED — governance evidence** | Project governance/release manager | Add a second trusted principal and prevent self-review, or retain the approved single-operator route and disclose it in the actual release record. | Either approver and executor are distinct, or the release body explicitly states single-operator approval and includes the attested accepted external-control snapshot/preflight identity. The generated governance record validates actor, reviewer, self-review setting, commit, tag, run, and policy/snapshot digests. | Live `pypi` reviewer is only `jemsbhai` and self-review is permitted; the only other direct collaborator has write rather than release-admin authority. A fail-closed governance record and draft-before-finalization path now exist, but no future release record exists. |
| B09 | **MIXED — hosted release contexts and uncertified claims** | Platform, typing, JS, and independent accessibility owners | Run the policy-required exact-candidate macOS ARM and Node/React/browser gates. Separately finish typing before adding `py.typed`, and obtain physical NVDA/VoiceOver review before any AT-support claim. | For this stable release, the macOS ARM and JS/browser policy contexts must be green. A future typed claim additionally requires strict mypy zero, Pyright 100%, and clean installed-wheel consumers; a future AT claim requires both physical profiles, exact deployment/build binding, reviewed bytes/hashes, and evidence at most 180 days old. | Local JS has 88 tests, zero high audit findings, zero-retry Playwright 9/9, and green typecheck/lint/build. Exact strict mypy reports 1,375 errors in 44 files and Pyright remains 0% with no marker. No exact-candidate hosted macOS/browser or physical AT evidence exists. Stable Python may remain explicitly untyped and the demo explicitly AT-uncertified; those two absent claims are not publication gates. |
| B10 | **BLOCKED FOR BOUND WIDENING — not a claim of 1.x support** | Dependency maintainer; upstream supplies candidate | Keep `scikit-image<1.0`. Let the scheduled monitor capture PyPI metadata and select only a real non-yanked 1.x prerelease; propose any bound change separately. | The discovery record identifies a real candidate and preserves metadata hash/serial. A distinct compatibility proof passes a valid post-candidate dependency graph, localisation, LIME, package/twine, and all tutorial gates with retained freeze, JUnit/log, and distribution hashes before review of a bound change. No candidate produces a blocked/failing monitor and no green proof context. | PyPI current is 0.26.0 and exposes no qualifying 1.x prerelease (metadata SHA-256 `2a986d45ee278f7820a060319d53f6e4c9b68d9841907829aeb68b09c54ff7bc`, serial `33119953`). The `<1.0` cap remains. |
| B11 | **QUARANTINED — P2 publication capability** | JS, security, release, accessibility, and scientific-parity owners | Preserve `private=true`, CommonJS/experimental metadata, non-parity disclosure, and no npm publication until every separately reviewed publication and scientific prerequisite exists. | Direct ESM/browser library consumers, exact export/tarball contract, publication threat model, provenance/recovery, physical AT evidence, and an explicit algorithm-parity decision are all exact-candidate green before changing private/module/publication/parity metadata. | Local package tests, audit, browser checks, and 16-file dry-run tarball are diagnostic controls only. The npm name is unpublished. No publication, parity, AT, provenance, or recovery acceptance exists, so no promotion is permitted. |

## Interpretation

B10 is a blocker to widening the existing dependency bound, and B11 is a deliberately retained
P2 quarantine; neither may be reworded as supported capability. They do not offset the open P0
rows. The reviewed executable policy also promotes B06, B07, and the macOS/JS portions of B09 to
this stable release's gates even though their general roadmap priority is P1. Typing and physical
AT remain claim gates: stable Python can remain explicitly untyped and the private demo can remain
explicitly AT-uncertified. A stable release still cannot be recommended while B01-B08 and the
policy-selected hosted portions of B09 lack direct evidence.
