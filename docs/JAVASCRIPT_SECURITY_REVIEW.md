# Private JavaScript workspace security review

## Scope and conclusion

This review covers the private CommonJS library tarball and the built synthetic
demo. It does not approve npm publication, third-party explainer execution, or
Python/JavaScript algorithm parity. The package remains `private`, experimental,
and publication-blocked.

The current demo accepts only local form input, keeps it in React memory, and
renders it as text through React. It has no API client, persistence, analytics,
cookies, authentication, remote fonts, or external content. The production HTML
uses a restrictive meta Content Security Policy and a no-referrer policy. Its
browser gate aborts and fails on every HTTP(S) request outside the local demo
origin and fails on console or page errors.

## Audited boundaries

- The tarball exports only root, `./core`, and `./visualizer` CommonJS entry
  points; experimental metrics/explainers and the demo are excluded.
- The versioned wire decoder rejects non-finite/unsafe values, accessors,
  custom prototypes, cycles, sparse/decorated arrays, unknown fields, and
  unsupported schema versions. JavaScript regression fixtures cover accepted
  and rejected payloads.
- Demo model and method names are inert labels over synthetic data. The UI and
  accessible disclosures say no explainer is executed or validated.
- React escapes feature names and values; no raw-HTML sink or dynamic script,
  style, URL, or module loading path is present.
- Browser tests block unexpected off-origin traffic. The CSP denies network
  connections, objects, forms, and non-self scripts/images/fonts; inline style
  remains allowed because the current React component uses inline style objects.
- GitHub Actions use read-only repository permissions for JavaScript tests and
  pin third-party actions to full commit SHAs. The npm lockfile fixes the exact
  dependency graph used by CI.

Run the reviewed checks with:

```sh
cd packages/js
npm ci
npm audit --audit-level=high
npm test
npm run lint
npm run typecheck
npm run test:browser
npm pack --dry-run
```

## Residual risk and publication blockers

- A meta CSP cannot set deployment response headers such as
  `frame-ancestors`; the hosting configuration must add reviewed security
  headers before a public production claim.
- Dependency audit results are point-in-time and require continuous review.
- The in-memory registry does not sandbox or safely execute untrusted plugins.
- There is no supported ESM entry point or library browser matrix. The
  Playwright matrix certifies only the built demo, not direct browser import of
  the private CommonJS tarball.
- Manual NVDA and VoiceOver evidence is absent. Follow
  `docs/ACCESSIBILITY_CERTIFICATION.md`; automated axe results are not a
  substitute.
- No npm identity, provenance, release recovery, incident response, or consumer
  threat model has been approved. Publishing remains prohibited by package
  metadata and policy.

Any change that adds network access, raw HTML, persistent storage, third-party
execution, new export paths, ESM output, or publication metadata requires a new
threat-model review and dedicated abuse tests.
