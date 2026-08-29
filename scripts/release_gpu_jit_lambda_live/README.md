# Lambda live-adapter boundary

This package is a production-oriented, fail-closed provider adapter for the
Explainiverse release GPU evidence window. It performs no work at import time
and has no enabled live mode by default.

The allowed target is exactly `gpu_8x_a100_80gb_sxm4` in `us-midwest-1`
(`Illinois, USA`), with an
action-time-discovered x86-64 image whose family is exactly
`lambda-stack-22-04`. The selected image ID, name, version, description, and
timestamps, the A100 80 GB SXM4 product string, eight physical GPUs, regional
capacity, and hourly price are all plan-bound. Ubuntu, GPU Base, other Lambda
Stack families, and caller-supplied image IDs absent from that filtered
discovery are rejected. H100 is intentionally unsupported:
the minimum Torch 2.0.0+cu117 lane has not been accepted on H100. Filesystems
are never created, mounted, renamed, or deleted.

Live construction requires all of the following:

1. A read-only inventory captured from the pinned Lambda OpenAPI 1.10.0
   production origin and used to build an immutable plan.
2. A Lambda API key supplied only over stdin or another anonymous file
   descriptor. This package cannot create or delete API keys.
3. Two separately asserted gates (`production_authorized` and
   `provider_mutation_authorized`) bound to the exact immutable plan SHA-256.
4. A fresh full-inventory receipt before every mutation. Receipts expire after
   45 seconds and can be consumed once.
5. A current public controller IPv4 `/32`; no ICMP and no open-world source.
6. An in-memory Ed25519 host identity whose fingerprint is plan-bound. Its
   private key is included only in launch `user_data` under cloud-init
   `ssh_keys`, then the serialized request body is zeroized on a best-effort
   basis. Only the public strict-known-hosts line and fingerprint are evidence.

The credentialed provider client paces every request start, including reads
and mutations, by at least one monotonic second to honor Lambda's general API
rate limit. A full eight-read inventory therefore spends at least seven
seconds in pacing, within the fixed 30-second observation window and before a
45-second receipt begins. Pacing clocks are injectable only alongside a
non-production transport for deterministic fixtures; the real HTTPS transport
rejects clock injection. Mutations, including the separately rate-limited
single launch, remain one-shot and are never retried.
The low-level public client accepts only the eight pinned read operations;
mutation requests are internal to `LambdaLiveAdapter`, whose methods consume
the matching fresh lifecycle receipt. API-key, filesystem-create, and every
other unlisted provider route are rejected before transport.

The live controller host must provide the lock-supported `cryptography`
package for in-memory Ed25519 generation and access-key/public-key binding.
It is imported only when one of those operations is requested; normal package
import and dry-run validation do not require it. Absence fails closed as
`cryptography_dependency_unavailable`; this operational prerequisite is not
added to Explainiverse's user-facing runtime or `all` extras. A Windows
controller additionally requires `pywin32` for native handle and security
descriptor inspection. Its absence fails closed before SSH execution; the
operator must inventory the exact Python executable and dependency versions in
the controller evidence rather than treating ambient site packages as release
evidence.

The exact four-file remote runtime bundle is loaded and hash-framed in the
same lexical order as the remote executor. Its digest is part of the immutable
plan. The public source files are base64-encoded into cloud-init `write_files`
entries under the fixed root-owned, non-writable
`/opt/explainiverse/bin/release_gpu_jit_lambda_runtime` directory. This makes
the only supported remote command concrete and value-free:
`/usr/bin/sudo -n -- /usr/bin/python3 -B
/opt/explainiverse/bin/release_gpu_jit_lambda_runtime/bootstrap.py`. The local
SSH invocation also fixes `-T`, `RequestTTY=no`, `IdentityAgent=none`, and the
platform null device so Windows never relies on `/dev/null` or a PTY.

Before any JIT configuration is generated, the controller must execute two
separate fixed public SSH bindings in order. The first waits for the path
written by cloud-init to exist:
`/usr/bin/sudo -n -- /usr/bin/cloud-init status --wait`. Its exit code, exact
first stdout line `status: done`, stdout/stderr digests, instance ID/IP,
known-hosts digest, and a fresh post-command provider inventory are bound in a
public receipt. Only then may the controller invoke:
`/usr/bin/sudo -n -- /usr/bin/python3 -B
/opt/explainiverse/bin/release_gpu_jit_lambda_runtime/executor.py probe-host`.
It accepts no options, credentials, plan, or JIT input. Its canonical receipt
must prove completed cloud-init, effective UID 0, the root-owned non-writable
plan-bound bundle, zero swap/residue, all eight exact A100 GPU UUID/product
records, and the pinned immutable image/Runner/Node probe. The local controller
must re-inventory the exact provider instance after each command and archive
and validate both receipts before asking GitHub for a JIT identity.
The host receipt also requires the runtime's exact GPU-injection probe record:
all eight ordered host UUIDs are bound into the Docker device request, the
container reports exactly eight `NVIDIA A100-SXM4-80GB` devices, the fixed
probe output digest matches, networking is `none`, and no ports are published.

Mutations are one-shot and never automatically retried. Any timeout, transport
failure, unbound response, non-JSON response, content-type drift, oversize
response, or non-200 status is classified as ambiguous. A new complete
inventory must prove either the exact before state or the exact owned after
state; partial or foreign state is a hard stop.

The required lifecycle is:

`baseline -> restrict global -> create instance ruleset -> launch -> bind exact
instance ID/IP -> terminate -> prove zero instances -> delete ruleset -> restore
global -> prove exact baseline`.

The remote runner executor is intentionally separate. Encoded GitHub JIT
configuration is accepted here only through an anonymous descriptor and must be
relayed to that executor without persistence. Its canonical public runtime
envelope is expected on FD 4 and JIT configuration on FD 3. No GitHub token is
sent to the Lambda host. An independently reviewed trusted-local controller
must perform and certify every GitHub run, job, runner-download, and runner
inventory query before and after remote execution; this provider adapter does
not manufacture or log any of those values.

The local SSH stdin framing is exact: an 84-byte big-endian
`>8sHHII32s32s` header containing magic `EXJIT01\n`, version 1, reserved flags
0, plan/JIT byte lengths, and their raw SHA-256 digests, followed by canonical
plan bytes and raw JIT bytes. `write_runtime_frame_and_close` requires an
anonymous output descriptor, writes exactly that frame, closes it to deliver
mandatory EOF, and destroys its local JIT buffer. The remote command is a fixed
bootstrap path with no options or values from either payload.

On Windows, ordinary OpenSSH cannot consume `/dev/fd/N` for a local
`UserKnownHostsFile`. `write_public_known_hosts` therefore exclusively creates
the public known-hosts line in the dedicated owner-private evidence directory;
the returned receipt binds its absolute path, content digest, controller ACL
audit-receipt digest, instance IP, and host fingerprint. The SSH builder
re-reads and verifies that exact file before returning executable argv. POSIX
callers may instead use an anonymous known-hosts descriptor. No private host,
access, API, or JIT material is written by either path.

The directory itself is not caller-asserted. `create_evidence_directory`
exclusively creates it and returns a held `EvidenceDirectoryReceipt`. On
Windows the directory is born with a protected, child-inheritable DACL
containing exactly full-control allow entries for the current user, SYSTEM,
and Administrators; its native handle denies delete sharing. On POSIX it is
owned by the effective user with mode `0700` and held through a no-follow
directory descriptor. The receipt binds the canonical path, filesystem
identity, and exact ACL while redacting the absolute path from public evidence.
`reopen_evidence_directory` permits interrupted recovery only when the same
stable receipt digest is supplied and reproduced. Every validation rejects
path, identity, owner, permission, reparse/symlink, or ACL drift.

Public evidence files are first fully written and flushed under a unique name
in that same directory. Windows publishes with a no-replace, write-through
move. POSIX publishes with a no-replace hard link, removes the temporary name,
and flushes the directory. A raced destination is never overwritten, a partial
file is never published under its final name, and any unresolved durability
error fails closed.

The pre-existing SSH access identity is separately sealed with
`capture_access_identity`. Its Ed25519 public-key digest must equal the
canonical provider key bound into the immutable plan. Windows requires the
file owner to be the current user, a protected DACL, and exactly full-control
allow entries for the current user, SYSTEM, and Administrators; the retained
native handle denies write and delete sharing. POSIX requires one regular link,
the effective-user owner, and mode `0600`. The receipt retains the private path,
file identity, and private digest only in memory, emits only redacted public
evidence, and must be revalidated immediately before every SSH process and
closed on every success or cleanup path.

## Remaining action-time blockers

The code does not claim that a release can proceed. Before using it, the sole
Lambda administrator must enroll MFA and create a temporary API key through the
authenticated console with user presence. GitHub App/collaborator authority,
runner-group state, workflow/job bindings, current public IP, current Lambda
capacity, selected image, and price must be re-audited immediately before the
window. The temporary key must be deleted through the console after cleanup.

The adapter deliberately refuses to claim that MFA, API-key lifecycle, GitHub
authority closure, or the remote runner execution has occurred.
