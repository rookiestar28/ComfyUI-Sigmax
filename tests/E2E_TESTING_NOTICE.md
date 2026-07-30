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
- Strictly documentation-only changes may use the exception in `tests/TEST_SOP.md`.
- Bugfixes must reproduce the host-visible failure, pin it with a targeted test, and then run
  the full applicable sweep.
