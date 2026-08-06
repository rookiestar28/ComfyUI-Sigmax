# End-to-End Testing Notice

All ComfyUI-Sigmax end-to-end tests must follow:

- `tests/TEST_SOP.md` for the full acceptance gate;
- `tests/E2E_TESTING_SOP.md` for real ComfyUI host procedures;
- `tests/CI_TEST_MATRIX.md` for lane activation, blocking behavior, and artifacts.

For this repository, the initial E2E surface is a real supported ComfyUI host, not a browser
frontend.

E2E is mandatory when a change affects:

- custom-node import or registration;
- node schemas;
- workflow load/save behavior;
- ComfyUI model/profile inspection;
- schedule execution through host APIs;
- sampler execution;
- supported-host compatibility.

Rules:

- A package import test is not a substitute for real-host E2E.
- A workflow loading successfully is not enough when execution behavior changed.
- A missing host harness is `NOT_IMPLEMENTED`, not a pass.
- Browser Playwright E2E is `NOT_APPLICABLE` unless the repository deliberately adds a web
  extension.
- Strictly documentation-only changes do not run the full gate or E2E lanes; apply the exception
  in `tests/TEST_SOP.md`.
- Pure prose unrelated to executable behavior must not be made a test contract. Test the
  underlying executable API, schema, workflow, package, or release boundary instead.
- M4-13 has an explicit 2026-08-06 user waiver for unavailable GitHub Actions hosted CI because
  the quota was exhausted; its local WSL/native Windows/pinned-host evidence controls closure,
  without converting the unavailable hosted lane into a pass.
- M7-09 is separately activated for an explicit local H4 Krea 2 model/GPU lane. Its fixed-case
  image receipts and blind-review protocol are not substituted by the CPU H1/H2 lanes or by a
  hosted-CI result. The user may waive the scoring phase as an acceptance blocker; that closes
  only the local execution/provenance lane and does not support a quality or promotion claim.
- Bugfixes must reproduce the host-visible failure, pin it with a targeted test, and then run
  the full applicable sweep.
