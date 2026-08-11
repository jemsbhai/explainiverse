# Demo accessibility certification

The private JavaScript demo has automated semantic and real-browser coverage,
but it is **not certified for assistive-technology support**. Axe and
Playwright cannot establish what NVDA or VoiceOver announces. This runbook is
the manual gate that must be completed before any accessibility support claim.

## Current blocker and ownership

- **Blocker:** there is no reviewed human test record from physical Windows 11
  and Apple-Silicon macOS hardware for the current commit.
- **Owner:** the release manager schedules an accessibility reviewer who did
  not implement the demo. The reviewer owns observations and signs the evidence
  manifest; the release manager verifies artifact hashes and workflow output.
- **Required access:** current Windows 11 with stable Edge and NVDA, and a
  currently supported macOS release on Apple Silicon with stable Safari and
  VoiceOver. Browser emulation is not acceptable evidence.
- **Acceptance:** both profiles in
  `.github/accessibility-certification-policy.json` pass every scenario below,
  all required artifacts are content-addressed, the evidence validator exits
  zero, and a reviewer confirms the referenced artifacts match their hashes.

Any failure, missing artifact, self-review, evidence older than 180 days, or UI
semantic change after the recorded commit keeps the claim blocked.

## Prepare the exact candidate

1. Record the full 40-character candidate commit SHA. Use a clean checkout of
   that commit and Node.js 20.11.0.
2. From `packages/js`, run `npm ci`, then
   `npx playwright install chromium firefox webkit`, then
   `npm run test:browser`. All Chromium, Firefox, WebKit, functional, reflow,
   off-origin-request, browser-error, and axe checks must pass.
3. Deploy that exact build to an HTTPS staging URL. Record the immutable build
   or workflow URL that ties the deployment to the candidate SHA. The Pages workflow's retained
   `deployment-evidence.json`, `demo-files.sha256`, and `demo-tree.sha256` must match the manifest's
   deployment URI, commit, and demo build digest. Do not use a mutable local source checkout as
   certification evidence.
4. Start a fresh browser profile with default settings. Disable extensions
   other than the screen reader. Record exact OS build, browser version, and
   assistive-technology version—not labels such as “latest.”
5. Start a screen recording with spoken output audible. Do not expose secrets,
   private datasets, notifications, or unrelated user content.

## Required profiles

### Windows Edge and NVDA

Use physical current Windows 11 hardware, stable Microsoft Edge, and current
stable NVDA. Test with keyboard and speech enabled. Record the NVDA speech-viewer
text as the interaction transcript. Use standard NVDA browse/focus modes; note
any non-default verbosity setting.

### macOS Safari and VoiceOver

Use Apple-Silicon hardware on a currently supported macOS release, stable
Safari, and built-in VoiceOver. Test with keyboard and speech enabled and Quick
Nav state recorded. Capture the VoiceOver caption-panel or equivalent spoken
output in the interaction transcript.

## Scenario checklist

Perform every scenario once per profile without a mouse. Mark only `pass` or
`fail` and write what was actually spoken in the notes.

1. `disclosure-and-landmarks`: load the page from a fresh tab. Confirm the page
   title, one main landmark, the “synthetic display data only” disclosure, form
   control labels, experimental visualizer region, and display-only disclosure
   are discoverable in a sensible reading order.
2. `keyboard-order-and-focus`: traverse forward and backward through all task,
   model, explainer, class, feature-name, feature-value, add, and remove
   controls. Focus must remain visible and must not enter a trap or invisible
   element. Disabled Add state must be announced before valid input exists.
3. `task-and-class-state-change`: select Tabular Classification and then Denied.
   Confirm the updated model/class choices and the polite status “Showing 4
   synthetic attributions for Denied using the SHAP label” are announced once,
   without claiming that a method was executed.
4. `signed-attribution-semantics`: navigate the attribution list. Confirm each
   feature name, signed formatted value, positive/negative/zero direction, and
   meter value are available without relying on green/red color.
5. `add-and-remove-feature`: add `manual_review` with value `0.33`, confirm the
   updated status and positive signed attribution, then remove it with the
   specifically named Remove control. Confirm focus remains usable.
6. `empty-state-announcement`: remove every feature. Confirm both “No features
   defined for this class yet” and “No feature attributions were supplied” are
   discoverable, and that no invalid meter is announced.
7. `zoom-and-320-css-px-reflow`: at 200% browser zoom and at an effective 320 CSS
   pixel viewport, repeat keyboard traversal. Text and controls must remain
   readable with no two-dimensional page scrolling, clipping, overlap, or loss
   of content.

Stop and record `fail` for crashes, browser console errors, unexpected network
requests, missing/duplicate announcements that impede operation, unlabeled
controls, focus loss, color-only meaning, or content that cannot be reached.

## Evidence artifact contract

Create one JSON manifest conforming to the checked-in policy. It must name an
independent reviewer, exact commit, HTTPS demo URL, immutable deployment/build
provenance URI, SHA-256 of the deployed demo build, completion timestamp,
and exactly one run for each profile. Each run must include all scenario IDs,
`pass` results with non-empty observation notes, exact version strings, and:

- `interaction-transcript`: UTF-8 text with ordered keystrokes/actions and the
  corresponding spoken output; and
- `screen-recording`: a recording that shows focus and includes spoken output.

Store each artifact at a durable access-controlled HTTPS URI and compute a
lowercase SHA-256 digest from the exact uploaded bytes. The validator checks
manifest completeness and URI/digest shape; the release manager must separately
download each artifact and verify its digest. The manifest contains no tokens,
embedded credentials, or sensitive user data.

Minimal top-level shape (repeat the complete scenario and artifact arrays for
both policy profiles):

```json
{
  "schema_version": 1,
  "commit_sha": "0123456789abcdef0123456789abcdef01234567",
  "demo_url": "https://staging.example.invalid/explainiverse/",
  "deployment_provenance_uri": "https://ci.example.invalid/builds/immutable-run-123",
  "demo_build_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "completed_at": "2026-08-10T17:00:00Z",
  "independent_reviewer": "Reviewer name or stable identity",
  "reviewer_independent_from_implementation": true,
  "runs": [
    {
      "profile_id": "windows-edge-nvda",
      "os_version": "Windows 11 exact build",
      "browser_version": "Microsoft Edge exact version",
      "assistive_technology_version": "NVDA exact version",
      "result": "pass",
      "scenarios": [
        {
          "id": "disclosure-and-landmarks",
          "result": "pass",
          "notes": "Exact observation and spoken output."
        }
      ],
      "artifacts": [
        {
          "kind": "interaction-transcript",
          "uri": "https://evidence.example.invalid/transcript.txt",
          "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        },
        {
          "kind": "screen-recording",
          "uri": "https://evidence.example.invalid/session.webm",
          "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        }
      ]
    }
  ]
}
```

Validate a completed manifest locally:

```sh
python scripts/validate_accessibility_evidence.py \
  --evidence PATH/manifest.json \
  --expected-commit "$(git rev-parse HEAD)"
```

Then run the manual **Accessibility Evidence Validation** workflow on the exact
candidate ref with the reviewed manifest path. The workflow binds
`commit_sha` to its checked-out `${{ github.sha }}` and retains the deployment
URI and build digest in the summary. It archives the exact tracked manifest,
the reviewed policy, the normalized summary, and a SHA-256 inventory for 180
days. Retain that artifact alongside the transcripts and recordings. A green validator is necessary but does not
substitute for reviewing the actual manual evidence or deployment bytes.
