# Lambda JIT runner remote runtime

This directory contains the narrow remote-host portion of the Explainiverse CUDA runner
boundary. It is production-capable only when a trusted local controller has completed every
repository, authority-window, dispatch, history, JIT-response, and provider check represented by
the canonical per-job runtime plan. It does not make those observations itself.

The split is deliberate. The disposable Lambda host receives no controller GitHub API credential,
provider API key, SSH private key, long-lived repository token, or package-publishing credential.
Its only controller-supplied sensitive input is one GitHub-generated encoded JIT configuration.
GitHub may deliver the normal job-scoped `GITHUB_TOKEN` to the isolated Actions job after it is
claimed; that credential remains inside the same ephemeral container/tmpfs boundary. The remote
process accepts:

- the canonical public per-job runtime plan on inherited anonymous FD 4; and
- the encoded JIT configuration on inherited anonymous FD 3.

Regular files, named FIFOs, argv, Docker environment options, bind mounts, and command text are
not accepted as input transports. The fixed container launcher reads the JIT value from stdin,
exports `ACTIONS_RUNNER_INPUT_JITCONFIG` inside the container, and starts the runner without
putting the value in the Docker configuration or host process argv. Docker logging is disabled.
The host process disables core dumps and process dumping, requires zero host swap, and zeroizes
its mutable input buffer immediately after writing the anonymous pipe. Both fixed runtime entries
reject inherited credential-shaped environment-variable names before work begins. The runner's writable
state, home, work directory, diagnostics, temporary directory, and tool cache are tmpfs mounts.

## Immutable runtime source

The accepted runner is the repository-served Linux x64 2.336.0 build:

- archive SHA-256:
  `04cf0be1aff4c3ec3554466c39124ca250e3effd8873bb7e8d68535aa9505d5d`;
- image platform manifest:
  `ghcr.io/actions/actions-runner@sha256:a1919047b038c38871d667c58cfdc7a878452711ab1212fb6036188f27a7ab16`;
- OCI config digest:
  `sha256:bd6fe162bb4ab4821daa8d694e20d779865618825d30c94342a0228b89947305`;
- runner commit: `98aabcd429c4e8402406c56ce2d26387fed3b9ce`; and
- Node 20.20.2 binary SHA-256:
  `6295488653f0d93b0a157841746fef7e72cc4328cfb60c4bbe0ca2668a836ffd`.

The manifest observation was made at `2026-08-28T21:05:51.032Z` with
`docker manifest inspect --verbose ghcr.io/actions/actions-runner:2.336.0`. The platform manifest,
not the mutable tag and not merely a newer runner release asset, is the pull pin. Before dispatch,
the controller must run the fixed credential-free `probe-host` command, archive its canonical
receipt, and put the receipt digest and fresh observation time in the runtime plan. The command
waits for cloud-init to report `status: done` before checking the Docker/NVIDIA host posture. It
then records the complete ordered eight-device UUID/product inventory (each product must be
exactly `NVIDIA A100-SXM4-80GB`), pulls the immutable platform digest, and runs a networkless image
probe. The image probe verifies UID/GID 1001, the exact runner version and commit, and the exact
Node 20 executable required by the pinned checkout and setup-python actions.
Before any JIT is generated, a second networkless probe injects all eight GPUs with Docker's exact
UUID device request and verifies the ordered UUID/product rows again inside the nonroot container.

## Container and network boundary

The executor allows one process at a time using a Linux abstract-socket lock and rejects any
existing Explainiverse runtime container, network, or `EXJIT_*` firewall chain before mutation.
The scan covers globally labelled containers/networks, the fixed probe name, and every iptables
chain, including detached chains from a partial earlier setup.
It runs one container with:

- UID/GID 1001, all capabilities dropped, `no-new-privileges`, the default seccomp boundary, a
  read-only root filesystem, private IPC/UTS namespaces, bounded PIDs and file descriptors, and no
  bind mount, Docker socket, privileged mode, or published port;
- an executable `/runner` tmpfs copied from the immutable image plus separate tmpfs mounts for the
  exact nonce-bound work directory, home, temporary directory, tool cache, and diagnostics;
- Docker `--gpus device=<exact UUID list>`, exact `NVIDIA_VISIBLE_DEVICES`, and logical
  `CUDA_VISIBLE_DEVICES` equal to `0` or `0,1`; and
- a dedicated bridge with inter-container communication disabled. A nonce-bound `DOCKER-USER`
  chain rejects loopback, RFC1918, carrier-grade NAT, link-local/metadata, multicast, and reserved
  IPv4 destinations and permits only public DNS plus HTTPS. IPv6 and route-localnet are disabled
  inside the container.

The full ordered physical GPU UUID and product inventory must exactly match the plan. The supported
host is exactly eight A100 80 GB SXM4 devices. Assigned UUIDs must contain exactly one or two real
devices; every other physical UUID is recorded as unrequested and is excluded from the device
request. Jobs run sequentially. On success, failure, or timeout, the executor tries all
container/firewall/network cleanup operations and then proves all named and globally labelled
runtime resources are absent before emitting a receipt.

The authority expiry is an absolute deadline, not a resettable per-step timeout. Bootstrap pipe
transfer, host setup, final prelaunch, and runner execution share one derived monotonic deadline.
The workload is stopped no later than authority expiry; cleanup alone may use the fixed 60-second
grace. Successful receipts bind the absolute expiry, cleanup deadline, and both deadline results.

## Required controller sequence

The local controller must, in order:

1. invoke the fixed `probe-host` command, verify and archive its public canonical receipt, and bind
   its fresh bundle/image/GPU inventory observations before any JIT registration;
2. dispatch and observe the exact owner-triggered workflow, attempt 1, ref, head SHA, all current
   nonce inputs, and the queued exact job/name/sole label;
3. after that queued observation, close and freshly prove the sole-authority window, fresh selected
   nonce history, and zero runner inventory;
4. generate one JIT response, validate its runner ID/name/sole label and exact pre-start
   `os="unknown"`, compute the encoded config
   digest without logging or persisting the value, and construct a canonical short-lived runtime
   plan with `execution_authorized=true`;
5. stream the EXJIT01 header, canonical plan, and JIT value to the fixed no-argument bootstrap;
   the bootstrap supplies anonymous child FD 4 (plan) and FD 3 (JIT), and every policy SHA,
   ordinal, deadline, and GPU UUID value is read exclusively from the plan;
6. retain the sanitized remote lifecycle receipt; and
7. from the trusted local machine, query all-attempt GitHub job/check evidence, prove the exact
   runner ID/name/label and successful job, archive the 15/15 zero-skip evidence, and prove the
   repository runner inventory returned to zero before proceeding or registering the next runner.

The Lambda SSH principal is `ubuntu`, not root. Because cloud-init installs the runtime bundle,
the controller first uses this system-provided fixed, value-free, non-interactive readiness
command (no PTY):

```text
/usr/bin/sudo -n -- /usr/bin/cloud-init status --wait
```

It requires exit zero and first output line `status: done`; the controller archives hashes of its
stdout and stderr. This command avoids assuming the cloud-init-installed executor path already
exists. The controller then uses the only supported public pre-JIT runtime entry:

```text
/usr/bin/sudo -n -- /usr/bin/python3 -B /opt/explainiverse/bin/release_gpu_jit_lambda_runtime/executor.py probe-host
```

It accepts no options or stdin data. Its schema-v1
`explainiverse-lambda-jit-host-preflight` receipt binds cloud-init readiness, effective UID 0,
the root-owned non-writable four-file bundle digest, all eight ordered UUID/product pairs, the
immutable image pull/inspect/probe records, and absence of local runtime residue. It also records
that no JIT configuration or GitHub API credential was received and no accepted Actions evidence
was established. A nonzero exit, noncanonical receipt, stale receipt, wrong product/count, or
digest mismatch blocks JIT creation.

Only after that receipt and the controller-side GitHub observations are accepted may the
controller use this exact fixed, value-free job entry (also no PTY):

```text
/usr/bin/sudo -n -- /usr/bin/python3 -B /opt/explainiverse/bin/release_gpu_jit_lambda_runtime/bootstrap.py
```

`sudo -n` preserves the SSH stdin stream but never prompts for a password. Failure to obtain root
without a TTY fails before either entry. Successful execution of the fixed `probe-host` command is
the noninteractive-sudo readiness proof and must precede JIT generation. The bootstrap independently
requires effective UID 0, root-owned non-writable Python/Docker/iptables/nvidia-smi binaries, zero
swap, disabled core/process dumps, and a root-owned non-writable runtime bundle. `-B` and the child
`PYTHONDONTWRITEBYTECODE=1` prevent bytecode files in the installed bundle.

The bootstrap accepts no argument. SSH stdin is one strict binary frame:

```text
big-endian struct >8sHHII32s32s (84 bytes)
magic                 b"EXJIT01\n"
version               uint16 = 1
reserved flags        uint16 = 0
canonical plan length uint32, 1..1,048,576
JIT length            uint32, 100..1,048,576
plan SHA-256           32 raw bytes
JIT SHA-256            32 raw bytes
payload                exact plan bytes, then exact JIT bytes, then EOF
```

The controller must stream the header, plan, and JIT buffers separately; it should not concatenate
the secret into an additional immutable frame buffer. `bootstrap.frame_header()` renders only the
84-byte public header for this purpose. The bootstrap rejects non-anonymous stdin, truncation,
trailing bytes, wrong digests, noncanonical plans, and any reserved flag. It forks the fixed
executor, maps anonymous child pipes to JIT FD 3 and plan FD 4, and executes exactly
`/usr/bin/python3 -B executor.py run`. The executor's `run` subcommand accepts no option or value.

The controller must use the audited fixed remote command; it must not interpolate a plan field or
secret value into a remote shell command, environment variable, or SSH argument. The short-lived
plan additionally binds the SHA-256 of the root-owned, non-writable four-file runtime bundle using
this exact framing, in lexical order: two-byte big-endian basename length, basename bytes,
eight-byte big-endian file length, file bytes for `__init__.py`, `bootstrap.py`, `executor.py`, and
`runtime_contract.py`.

The plan permits only three phase shapes. `pull-request` binds
`.github/workflows/cuda-ci.yml` at
`refs/heads/codex/harden-cuda-runner-routing` and the two exact CUDA single-GPU jobs;
`final-main` binds that workflow at `refs/heads/main` and its four exact one-/two-GPU jobs; and
`publication` binds `.github/workflows/publish-pypi.yml` at the exact
`refs/tags/v0.15.0` source and only the two exact `Release CUDA single-GPU (..., zero skips)` jobs.
All require owner actor and triggering actor, `workflow_dispatch`, attempt 1, and distinct current
nonce inputs. Publication additionally requires `tag=v0.15.0`, positive preflight/CUDA run IDs,
`stage_recovery_drill=true`, and four distinct prior accepted final-main CUDA nonces that are
disjoint from both publication nonces. Two-GPU publication jobs and any other ref/tag/job fail
closed.

The runtime has one deployment path: the immutable launch request's cloud-init `write_files`
entries install the exact four public files directly under the fixed runtime directory. The
directory is root:root mode `0555`; each file is root:root mode `0444`; bytecode generation is
disabled. SSH/SCP staging and post-launch installation are not permitted. The retained deployment
receipt has schema version 1 and kind
`explainiverse-lambda-jit-runtime-deployment`; it records the instance ID/host-key fingerprint,
aware UTC observation time, exact fixed command, `sudo_noninteractive=true`, Python path and
resolved root-owned file identity, the four installed absolute paths with UID 0, GID 0, mode
`0444`, byte size and lowercase SHA-256, the framed runtime-bundle SHA-256, and
`bytecode_disabled=true`. Any missing/extra file, symlink, writable mode, digest drift, failed sudo
probe, or receipt mismatch blocks JIT generation.

## Evidence limits

A remote receipt proves only the local lifecycle facts it directly observes: exact image and GPU
inventory, container launch, JIT bytes sent through the anonymous pipe, process exit, JIT buffer
destruction, and zero local container/network/firewall residue. GitHub runner 2.336.0 can return a
zero `run.sh`/container exit code even when `Runner.Listener` reports an invalid JIT configuration,
so container exit zero is not job success. The receipt therefore explicitly records that job
success, pytest counts/skips, claimed-job count, post-exit registration absence, and accepted
Actions evidence were **not** verified by the remote runtime. Only the local controller's direct
GitHub records can close those fields.

The Docker daemon may remain rootful because the dedicated egress firewall requires host root;
the runner container itself is non-root. This component does not claim a rootless-daemon boundary.
It also does not claim provider teardown, VM/disk deletion, repository authority restoration, or
accepted CUDA release evidence.
