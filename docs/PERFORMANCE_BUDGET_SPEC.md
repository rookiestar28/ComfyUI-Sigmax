# Performance Budget Specification

- **Observation:** `sigmax.performance-observation/1`
- **Evaluation:** `sigmax.performance-budget-evaluation/1`
- **Matrix:** `sigmax.performance-budget-matrix/1`
- **Envelope:** `sigmax.performance-budget-matrix-envelope/1`

## Contract

Every budget has an exact metric ID, integer unit, inclusive minimum/maximum, and workload
fingerprint. Supported units are nanoseconds, bytes, and count. Every accepted result contains
ordered first/repeat observations from one platform lane. Both values must remain inside the
range, and the canonical evaluation fingerprint must match.

Minimums prevent skipped work from appearing faster. Exact count ranges lock the current CPU
tensor boundary to one construction, one validation round-trip, and zero explicit device
transfers. Workload fingerprints bind source files and exact schedule output identities.

## Implemented Lanes

- Windows Python 3.13 and WSL Python 3.10:
  - official Turbo 8-step 1024×1024 schedule latency and peak Python allocation;
  - official RAW 52-step 1360×768 schedule latency and peak Python allocation;
  - CPU tensor-output latency and exact boundary-operation counts;
  - fresh isolated package/process startup and zero optional-framework imports.
- Windows pinned ComfyUI 0.29.0 revision
  `e651b7bef55a5376343dcb1c0edb79f0142c985e`:
  - two fresh CPU/no-model process starts;
  - complete H1/H2/H3 pass-to-pass host validation;
  - readiness below 30 seconds and complete controlled cleanup.

## Interpretation

Limits are deliberately broad regression ceilings. Observations are machine-specific and do
not promise identical timing across hardware, OS load, Python patch versions, or future hosts.
An observation cannot raise its own budget. A failed budget requires investigation and review;
silently increasing a threshold is not acceptance.

GPU, model-weight inference, latest-host, official-container, VRAM, image generation, and
end-to-end sampler throughput remain `not_evaluated`. CPU or synthetic PASS cannot be reused for
those lanes.

## Reproduction

Run the matching project-local interpreter:

```bash
python scripts/run_performance_budget_lane.py \
  --lane-id performance-wsl-py310 \
  --output tests/performance/fixtures/wsl_py310_v1.json \
  --check
python scripts/generate_performance_budget_matrix.py --check
```

Use `performance-windows-py313` and the Windows fixture on Windows. Pinned-host evidence must be
regenerated through the standard real-host SOP and sanitized with
`generate_performance_host_startup_evidence.py`; raw host evidence remains private.

## Fail-Closed Rules

The loader rejects noncanonical JSON, floats, invalid units/ranges, missing or misordered
attempts, workload mismatch, over-budget values, false PASS, undeclared sources, duplicate or
unsorted results, unsupported exclusion states, and evaluation or matrix fingerprint drift.
