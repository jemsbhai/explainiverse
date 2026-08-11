# Python typing readiness

Explainiverse does **not** claim PEP 561 typed-package support. The source tree
must not contain `src/explainiverse/py.typed`, built archives must not contain a
package `py.typed` marker, and project metadata must not use the
`Typing :: Typed` classifier while the checked-in policy is blocked.

The baseline audit on the local mitigation working tree on 2026-08-10 was
falsifiable and not close to a support claim:

- `poetry run mypy --strict src/explainiverse` reported 487 errors in 41 files.
- Pyright 1.1.411 `--verifytypes explainiverse --ignoreexternal` found no
  `py.typed` marker and reported 0% type completeness.

`.github/typing-readiness-policy.json` records that evidence and the exact
acceptance commands. `scripts/audit_typing_readiness.py` is a non-claim guard:
it passes only while the policy says `blocked` and all source, metadata, wheel,
and source-distribution surfaces remain honestly untyped. It deliberately
fails if somebody merely changes the policy to `ready`.

Run the source guard with:

```sh
poetry run python scripts/audit_typing_readiness.py
```

After building distributions, include every archive:

```sh
poetry run python scripts/audit_typing_readiness.py \
  --distribution dist/explainiverse-VERSION-py3-none-any.whl \
  --distribution dist/explainiverse-VERSION.tar.gz
```

## Gate for a future typed-package claim

A future change may add `py.typed` only when all of the following are included
in the same reviewed change:

1. Public annotations pass `mypy --strict` with zero errors; suppressions must
   be narrow, justified, and reviewed rather than global.
2. Pyright `--verifytypes` reports 100% completeness with zero unknown or
   ambiguous public symbols.
3. Clean strict-mypy and Pyright consumer projects install the built wheel and
   exercise every documented public import path without importing the source
   checkout.
4. The wheel and source distribution are inspected for the marker, and the
   installed package is checked again.
5. The blocked guard is replaced with those positive consumer gates. Merely
   deleting the policy or changing `claim_status` is a gate failure.

Until that evidence is green, Python typing remains explicitly uncertified.
