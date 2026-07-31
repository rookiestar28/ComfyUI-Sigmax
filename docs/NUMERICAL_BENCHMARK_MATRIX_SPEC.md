# Numerical Benchmark Matrix Specification

## 1. Schemas

- Matrix: `sigmax.numerical-benchmark-matrix/1`
- Envelope: `sigmax.numerical-benchmark-matrix-envelope/1`
- Sanitized host attempts: `sigmax.known-good-host-attempts/1`

The packaged matrix is a canonical, dependency-free summary of already accepted numerical and
host evidence. It does not run a model, upgrade an execution receipt, or replace complete vectors
and traces in the source fixtures.

## 2. Coverage

The v1 matrix contains 23 sorted verified result rows:

| Lane | Rows | Evidence |
| --- | ---: | --- |
| Turbo schedule parity | 4 | 4/8/12/16-step official and framework comparisons |
| RAW schedule parity | 14 | 28/52-step cases over five square and two orientation geometries |
| H2 workflow | 4 | Turbo plus RAW square/landscape/portrait artifact/receipt workflows |
| H3 native Euler | 1 | Controlled deterministic 8-transition ComfyUI Euler execution |

Every verified row carries the exact capability decision
`{"level":"allow","reasons":["compatible"]}`. Rejected, unsupported, unavailable, or unapproved
lanes cannot enter the verified results array.

## 3. Result Rows

Every row records:

- stable ID and lane;
- profile ID/version, RAW/Turbo variant, recipe, and evidence level;
- requested/effective width, height, and transition count;
- requested/effective transition and model-evaluation counts;
- explicit schedule/model/sampler RNG ownership;
- model-weight presence and weight variant;
- device, dtype, Python, NumPy, Torch, Diffusers, ComfyUI, and host revision when applicable;
- exact source-precision baseline result fingerprints, decimal error statistics, and tolerances;
- construction/numerical artifact and execution-receipt identities when applicable;
- execution status;
- first-run/repeat status and stability for accepted host rows;
- deterministic rerun and native-Euler trace evidence for H3.

Unavailable runtime versions are `null`; they are never guessed. Schedule-only parity rows are
`not_executed`, have zero effective execution counts, and cannot be read as sampler or
image-quality evidence. H2 rows retain truthful `not_executed` receipts. Only H3 records completed
transitions and model evaluations.

## 4. Model-Weight Precision Boundary

The v1 verified rows use no real model weights. The matrix therefore records:

```json
[
  {
    "kind": "bf16",
    "reason": "gpu_model_weights_not_approved",
    "result": null,
    "status": "not_evaluated"
  },
  {
    "kind": "quantized",
    "reason": "gpu_model_weights_not_approved",
    "result": null,
    "status": "not_evaluated"
  }
]
```

BF16 and quantized evidence must be evaluated separately under an explicitly approved GPU/model
weight lane. Neither may inherit a schedule-parity, H2, synthetic-model, or other precision's
PASS status.

## 5. Source Binding and Regeneration

The generator reads only six fixed public repository-relative sources:

- packaged workflow fixtures;
- sanitized known-good host first/repeat attempts;
- capability/receipt conformance;
- native Euler parity;
- RAW authoritative parity;
- Turbo authoritative parity.

The matrix records the schema, status, and exact SHA-256 content identity of every source.
Regeneration is deterministic:

```bash
python scripts/generate_numerical_benchmark_matrix.py --check
```

`--check` fails if the packaged canonical bytes differ from a fresh source-derived result. A
reviewed source change requires explicit regeneration with `--write`, focused review, and the
full acceptance gate.

The sanitized host source contains only stable host/package revisions, lane/status/reason/result
identities, receipt identity where applicable, and first/repeat acceptance. It contains no
ports, durations, logs, local paths, credentials, prompts, or process details.

## 6. Loading and Identity

```python
from comfyui_sigmax.benchmark_matrix import (
    load_numerical_benchmark_matrix,
    serialize_numerical_benchmark_matrix,
)

matrix = load_numerical_benchmark_matrix()
canonical_envelope = serialize_numerical_benchmark_matrix(matrix)
print(matrix.matrix_fingerprint)
print(matrix.projection()["coverage"])
```

The loader rejects oversized or noncanonical JSON, BOMs, duplicate names, untyped floats,
unknown fields, malformed identities/decimals, private or absolute paths, secret-like fields,
inconsistent counts/coverage, unsupported capabilities, unstable repeats, invalid source
allowlists, and matrix fingerprint drift.

## 7. Evidence Hierarchy and Limits

The matrix preserves the repository correctness hierarchy:

1. closed-form and property tests;
2. complete golden vectors;
3. authoritative numerical parity;
4. step-level native sampler parity;
5. pinned real-host behavior;
6. optional image/model-weight evidence.

A summary row cannot make its source evidence stronger. Numerical schedule parity is not a real
model run; H2 construction evidence is not execution; the controlled H3 model is not a Krea
checkpoint; and no v1 row claims BF16, quantized, GPU, image-quality, or performance evidence.
