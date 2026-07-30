# Execution Receipt and Portable Bundle Specification

## Status

- **Receipt projection schema:** `sigmax.execution-receipt/1`
- **Receipt envelope schema:** `sigmax.execution-receipt-envelope/1`
- **Portable bundle schema:** `sigmax.portable-execution-bundle/1`
- **Construction artifact dependency:** `sigmax.schedule-artifact-envelope/1`
- **Maturity:** implemented and normative for pre-alpha v1; not yet a stable public API

This specification keeps construction evidence and execution evidence separate. A schedule
artifact records what Sigmax constructed. An execution receipt records the explicit outcome
reported by an execution path. Building a schedule, registering a node, or passing static
capability checks MUST NOT imply that model or sampler execution succeeded.

## 1. Identity model

An execution receipt references:

- the existing construction fingerprint;
- the existing numerical schedule fingerprint;
- a model component fingerprint;
- a sampler component fingerprint;
- its own receipt fingerprint in the transport envelope.

The receipt does not reinterpret or replace the schedule artifact. A portable bundle contains
the complete existing artifact envelope and the complete receipt envelope as separate members,
then verifies their construction and numerical cross-links.

## 2. Receipt projection

The receipt projection has these exact members:

| Member | Meaning |
| --- | --- |
| `schema` | `sigmax.execution-receipt/1` |
| `artifact` | Referenced construction and numerical fingerprints |
| `effective_inputs` | Verified effective profile, dimensions, steps, precision, and compatibility metadata copied from the construction artifact |
| `profile` | Exact profile ID and version |
| `compatibility` | Complete allow/warn/reject decision, considered dimensions, and stable reason codes |
| `host` | Bounded host ID, version, revision, and public API version |
| `model` | Bounded model ID/version/fingerprint |
| `sampler` | Bounded sampler ID/version/fingerprint |
| `rng_ownership` | Explicit schedule/model/sampler random-source ownership |
| `counts` | Requested/effective transitions and requested/effective model evaluations |
| `execution` | Final status and optional stable reason code |

The requested transition count MUST match the effective schedule steps in the referenced
construction artifact. Effective counts cannot exceed requested counts.

## 3. Status contract

| Status | Required behavior |
| --- | --- |
| `not_executed` | Effective transitions and model evaluations are zero; no reason code |
| `succeeded` | Effective counts equal requested counts; no reason code; compatibility is not `reject` |
| `failed` | Effective counts may be partial; a stable bounded reason code is required |
| `interrupted` | Effective counts may be partial; a stable bounded reason code is required |

Raw exception messages, prompts, credentials, file paths, and machine-local details are not
reason codes and cannot enter the receipt.

## 4. Transport

`serialize_execution_receipt()` emits canonical UTF-8 bytes under
`sigmax.execution-receipt-envelope/1`. `deserialize_execution_receipt()` accepts bytes or
canonical text, bounds receipts to 1,048,576 bytes, rejects BOMs, duplicate keys, untyped JSON
floats, non-finite constants, unknown fields, unsupported enum values, inconsistent counts, and
stale fingerprints.

`serialize_portable_execution_bundle()` emits
`sigmax.portable-execution-bundle/1`, containing the existing artifact envelope and receipt
envelope. `deserialize_portable_execution_bundle()` bounds the bundle to 2,097,152 bytes,
strictly parses both nested transports, and rejects construction, numerical, or effective-input
cross-link mismatches.

Canonical output contains no clock-derived timestamps or random receipt IDs. Equal validated
inputs therefore produce byte-identical receipts and bundles across supported processes and
operating systems.

## 5. Current execution boundary

The runtime package exposes pure receipt and bundle contracts; those contracts do not execute a
sampler or start ComfyUI. A successful receipt can be built only after an execution boundary
supplies explicit, internally consistent evidence. Sigmax does not expose a production UI node
that lets a workflow manually claim successful execution.

The repository-owned M5-01 H3 harness is the first validated real-host producer. It stages a
release-excluded test node, executes the pinned host's actual deterministic `comfy.euler` over a
controlled eight-transition flow-velocity fixture, records and independently validates every
latent update and model evaluation, verifies deterministic rerun and single-shift ownership, and
only then calls `build_execution_receipt()` with `succeeded`. Its receipt is cross-linked to the
existing Turbo construction artifact. This proves the receipt path for that exact controlled
host execution; it does not let normal workflows self-assert success or establish real Krea
checkpoint, GPU, image, stochastic, resumable, partial-denoise-execution, or advanced-workflow
support.
