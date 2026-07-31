# Optional Image Benchmark Protocol Specification

## 1. Purpose and Schemas

The optional image benchmark boundary defines reproducible future image comparisons without
executing model weights in the default package or gate.

- Protocol: `sigmax.image-benchmark-protocol/1`
- Envelope: `sigmax.image-benchmark-protocol-envelope/1`
- Reviewer ballot: `sigmax.image-benchmark-blind-ballot/1`
- Post-vote reveal: `sigmax.image-benchmark-blind-reveal/1`

Every artifact carries the immutable authority level `supplemental_only`. Image metrics or blind
preferences cannot establish mathematical parity, schedule correctness, or official profile
status. The accepted numerical matrix remains a required prerequisite.

## 2. Fixed Cases

The packaged protocol contains four sorted cases derived from the accepted M7-02 H2 schedule
evidence:

| Case | Recipe | Requested geometry | Steps | ComfyUI CFG |
| --- | --- | ---: | ---: | ---: |
| RAW framework portrait | `krea2.raw.diffusers-reference-28` | 761×1353 | 28 | 5.5 |
| RAW official landscape | `krea2.raw.official-full-52` | 1353×761 | 52 | 4.5 |
| RAW official square | `krea2.raw.official-full-52` | 1024×1024 | 52 | 4.5 |
| Turbo official square | `krea2.turbo.official-8` | 1024×1024 | 8 | 1.0 |

Each case fixes an original public-safe prompt, an explicit negative prompt, a nonnegative seed,
requested and effective dimensions, Euler sampler, recipe/profile identity, candidate roles, and
canonical prompt/settings fingerprints. It also binds the source construction fingerprint,
numerical fingerprint, and truthful `not_executed` receipt from the numerical benchmark matrix.

The compared roles are:

- `reference_control` using the accepted reference replay;
- `sigmax_candidate` using the matching Sigmax profile schedule.

The protocol does not claim that either candidate has produced an image.

## 3. Execution and Hash Requirements

The packaged lifecycle state is exactly:

```json
{
  "component_hashes": null,
  "image_hashes": null,
  "reason": "gpu_model_weights_not_approved",
  "status": "not_executed"
}
```

This state is non-PASS. It contains no placeholder checkpoint, component, image, or metric result.

After a separately approved H4 run, each typed candidate evidence record must contain:

- exact case, candidate, and settings identities;
- `succeeded` execution status;
- construction, numerical, and execution-receipt fingerprints;
- checkpoint, text-encoder, VAE, and output-image SHA-256 identities;
- precision, ComfyUI, Python, Torch, and GPU identities.

All eight candidate images and all eight succeeded receipts must have distinct identities. The
builder rejects missing pairs, duplicate images/receipts, private paths, unsupported candidates,
and construction, numerical, or settings cross-link drift. It accepts hashes and bounded metadata
only; it never opens an image, model component, construction artifact, or receipt payload.

## 4. Blind Commit/Reveal Protocol

The operator chooses a lowercase 256-bit hexadecimal seed and keeps it from reviewers. The
`sha256-ranked-balanced-ab/1` algorithm ranks the four case IDs by SHA-256 of the seed and case ID.
Exactly half receive reference-as-A and half receive reference-as-B.

The reviewer ballot contains:

- protocol and seed-commitment fingerprints;
- case and prompt fingerprints;
- image A and image B hashes;
- no candidate IDs and no seed.

The ballot must be frozen before votes are frozen. Only then may the reveal disclose the seed and
A/B candidate mapping. Rebuilding the ballot from the reveal must reproduce its exact fingerprint;
a wrong seed, changed image, changed case, or changed mapping fails closed.

This protocol makes ordering reproducible and exactly balanced. Operational separation still
matters: giving reviewers the candidate evidence, seed, or reveal before votes are frozen defeats
blinding even if the hashes remain valid.

## 5. Metric Observations

The v1 packaged protocol contains no metric observations. A future completed observation must
record the metric ID, implementation, implementation version, and a canonical nonnegative decimal
value. Precision and quantized variants must be evaluated separately.

Metric observations and blind preferences remain `supplemental_only`. They may help compare two
already well-defined executions, but they cannot override a golden, parity, sampler-step, receipt,
or host failure.

## 6. Loading and Reproducible Generation

```python
from comfyui_sigmax.image_benchmark import load_image_benchmark_protocol

protocol = load_image_benchmark_protocol()
assert protocol.projection()["authority"]["level"] == "supplemental_only"
print(protocol.protocol_fingerprint)
```

The protocol is generated only from reviewed constants and the packaged numerical matrix:

```bash
python scripts/generate_image_benchmark_protocol.py --check
```

The generator records the numerical resource content identity and matrix fingerprint. `--check`
fails if the canonical packaged bytes drift. A source or fixed-case change requires explicit
regeneration, review, and the full acceptance gate.

## 7. Validation and Limits

The dependency-free loader rejects oversized or noncanonical JSON, BOMs, duplicate names,
untyped/non-finite numbers, unknown fields or schemas, malformed hashes/decimals/identifiers,
private paths, secret-like fields, control characters, invalid profile/settings/geometry, source
or prerequisite drift, fabricated execution state, metric authority escalation, and protocol
fingerprint mismatch.

The package does not download weights, launch ComfyUI, use an accelerator, compute image metrics,
or read image payloads. Real-model execution remains an explicit H4 operation governed by the E2E
SOP and requires separate authorization, hashes, compute, retention, cleanup, and evidence.
