# Explainiverse JavaScript experiment

Status: **private, experimental, and not supported for publication or production use**.

This workspace is not the JavaScript edition of the Python `explainiverse`
package. It currently contains only:

- a validated `Explanation` data container;
- a minimal asynchronous `BaseExplainer` extension contract;
- an in-memory metadata registry with explicit claim-status boundaries; and
- a display-only React feature-attribution visualizer plus a synthetic demo.

There are no JavaScript explainer algorithms or evaluation metrics. The
`src/explainers` and `src/metrics` directories are quarantined placeholders and
are not exported from the package entry point. Names in the demo are labels over
synthetic values; no named method is executed or claimed.

The package is marked `private` and uses the local identifier
`explainiverse-js-experimental`. That identifier is not a proposed npm package
name. The build metadata exists only so the local tarball surface and TypeScript
exports can be audited.

## Local verification

Use Node.js 20.11 or newer, then run:

```sh
npm install
npm test
npm run typecheck
npm run build
npm run build:demo
npm run lint
npm pack --dry-run
```

The library build exposes only the root, `./core`, and `./visualizer` entry
points. React and ReactDOM are peer dependencies. The demo and all source,
tests, logs, and empty placeholder modules are excluded from the tarball.

## Residual limitations

- No compatibility or semantic-equivalence claim is made with Python beyond
  the JS-defined JSON-safe subset of the snake-case `Explanation` fields that
  is exercised by the shared fixture. Python's general `to_dict()` payload is
  intentionally broader and is not accepted automatically. The JS wire is a
  closed five-field schema; missing or unknown top-level fields are rejected. This subset
  is recursively restricted to finite JSON values, unique feature names, arrays,
  and plain string-keyed objects, and is tested through a shared Python/JS
  fixture. Maps, sets, dates, typed arrays, BigInt, undefined values, symbols,
  non-finite numbers, signed negative zero, integers outside JavaScript's safe-integer range,
  sparse or decorated arrays,
  accessors, custom prototypes, and cycles are rejected. Arbitrary JSON object
  keys (including `__proto__`) remain inert own data properties.
- The registry stores metadata and constructors in memory; it does not discover,
  recommend, sandbox, or execute third-party methods safely.
- The visualizer supports finite scalar feature-attribution maps only. It does
  not render heatmaps, rules, examples, uncertainty, or method diagnostics.
- The local library build is CommonJS-only; no ESM or browser support matrix has
  been established.
- Accessibility coverage is semantic linting and DOM tests, not validation with
  real browsers and assistive technologies.
- No bundle-size target, security review, or npm release process has been
  established.
