# Workflow Validation Specification

## Status

Implemented contract: `sigmax.workflow-validation-report/1`

Envelope: `sigmax.workflow-validation-report-envelope/1`

Fixture bundle: `sigmax.workflow-fixture-bundle/1`

Pinned host baseline: `sigmax.workflow-host-baseline/1`

This specification covers deterministic schema compatibility. Real-host execution is a separate
repository H1/H2 boundary described below; a validation report alone does not prove execution.

## Purpose

The workflow metadata layer preserves portable requirements but deliberately treats the
surrounding ComfyUI graph as host-owned data. The adjacent `comfyui_sigmax.workflows` package
validates canonical graph fixtures without importing or executing host node implementations.

The package ships four executable model-free fixtures:

- `krea2-turbo-1024`: official Turbo, 8 steps, 1024x1024;
- `krea2-raw-official-square-1024`: official RAW, 52 steps, 1024x1024;
- `krea2-raw-official-landscape-1353x761`: official RAW, 52 steps, requested
  1353x761 and effective 1360x768;
- `krea2-raw-diffusers-portrait-761x1353`: framework-reference RAW, 28 steps,
  requested 761x1353 and effective 768x1360.

Each graph connects `Sigmax.Krea2SigmaScheduler` to `Sigmax.ScheduleInspector` and then to the
variant-specific `Sigmax.RawWorkflowOutput` or `Sigmax.TurboWorkflowOutput`. The output verifies
the complete connected schedule and publishes a canonical artifact plus truthful
`not_executed` receipt. These fixtures exercise node identity, linked inputs, ordered widgets,
fixed variant values, versioned metadata, portable package identity, and an executable history
boundary without claiming model or sampler execution.

## Host Schema Inputs

Validation accepts either supported public representation:

- legacy/current `/object_info`;
- Node Definition JSON v2.

Both are normalized through the ComfyUI adapter. Ordered widget names remain an explicit fixture
contract because the adapter's semantic node definition intentionally sorts input names.

The pinned static baseline identifies:

- ComfyUI version `0.29.0`;
- revision `e651b7bef55a5376343dcb1c0edb79f0142c985e`;
- public module `comfyui_sigmax.nodes`.

`validate_pinned_workflow_fixtures()` runs the blocking static known-good scan.
`validate_live_workflow_fixtures()` validates a caller-observed `/object_info` object and requires
the caller to label the host version, revision, and lane.

## Issue Taxonomy

Reports use stable issue kinds:

- `missing_node`
- `missing_input`
- `widget_slot_drift`
- `input_type_drift`
- `invalid_fixed_combo_value`
- `deprecated_node`
- `experimental_node`
- `normalized_directory_failure`
- `malformed_metadata`
- `workflow_schema_malformed`
- `host_schema_malformed`

M4-10 compatibility findings use `error` severity. A known-good report passes only when it has no
issues.

## Lane Policy

`known_good` is blocking:

- any issue sets `compatible=false`;
- any issue sets `gate_passed=false`;
- `observational=false`.

`latest_host` is an informational compatibility signal:

- findings still set `compatible=false`;
- findings and their original severity remain in the report;
- `gate_passed=true` prevents an observational lane from masquerading as the supported-host gate;
- `observational=true` labels the result explicitly.

Latest-host evidence cannot silently redefine the pinned compatibility window.

## Machine-Readable Report

Every canonical report records:

- report schema;
- scan mode and lane;
- host version and revision;
- package ID/version;
- node IDs/versions;
- workflow count;
- ordered issues and severities;
- compatibility, gate, and observational results.

Serialization wraps the report with a SHA-256 fingerprint. Deserialization rejects unknown
schemas, non-canonical bytes, inconsistent lane/result flags, malformed issues, and fingerprint
drift.

## Live Acquisition Safety

`fetch_live_object_info()` is an optional literal-loopback acquisition helper with a deliberately
narrow boundary:

- HTTP only;
- literal `127.0.0.1` or `::1` only;
- exact `/object_info` path;
- no user information, query, fragment, DNS hostname, or redirect;
- timeout at most 30 seconds;
- `application/json` only;
- response limit of 2,000,000 bytes.

It returns an untrusted JSON object for normal validation. It does not load ComfyUI modules,
execute node code, start a host, or establish H1/H2 real-host evidence.

## Real-Host H1/H2 and Native-Euler H3 Boundary

`scripts/run_comfyui_e2e.py` owns the separate real-host lane for ComfyUI `0.29.0` revision
`e651b7bef55a5376343dcb1c0edb79f0142c985e`.

H1 stages the extension into an owned isolated directory and proves:

- all eight node IDs import and appear in live `/object_info`;
- the live four-workflow schema report passes;
- importing Sigmax does not replace `torch.nn.Module.__call__`, mutate ComfyUI's scheduler
  registry, or import Diffusers.

H2 retains the accepted Turbo regression and executes all three RAW fixtures. Every RAW case
must reach completed successful history and verify:

- requested and effective geometry, sequence length, recipe/evidence, and dynamic `mu`;
- canonical construction/numerical/receipt fingerprints;
- external-sigma ownership and exactly one `krea.exponential_mu` transform;
- zero effective model/sampler counts and `not_executed` status;
- workflow metadata reload from the retained prompt `extra_data`.

The lane also submits an ambiguous variant and an invalid step count. The ambiguous variant
must produce a terminal runtime rejection with a retained prompt ID, exact scheduler exception,
and no partial output. The invalid step count must produce a structured prequeue HTTP 400
rejection without a prompt ID or partial output. Host traffic is bounded credential-free
literal `127.0.0.1`, process state is isolated, and successful runs remove their owned temporary
state after shutdown and port-release verification.

The activated M5-01 H3 lane connects the same accepted Turbo schedule to a namespaced,
release-excluded controlled-flow test node. That node invokes the pinned host's actual native
Euler implementation twice on CPU float32 data. The driver requires all eight step inputs,
velocities, denoised values, outputs, sigma pairs, transitions, and model evaluations to match an
independent flow-Euler oracle; then it builds an artifact-linked `succeeded` receipt. H3 leaves
the production eight-node registry and H2 `not_executed` receipts unchanged.

## Package Identity and Normalized Directories

Physical custom-node directory names are not workflow identities. A canonical Sigmax node must
retain:

- its stable `Sigmax.<Name>` node ID;
- host schema `python_module=comfyui_sigmax.nodes`;
- workflow property `cnr_id=comfyui-sigmax`;
- M4-07 package/node versions in `extra.comfyui_sigmax`.

Renamed or version-normalized install directories must not leak into these public identities.

## Current Limitation

H1/H2 executes model-free schedule construction, inspection, publication, and metadata reload.
H3 additionally validates controlled deterministic native-Euler execution and a successful
sampler receipt. No lane loads Krea model weights, uses a GPU, generates/compares images, or
validates stochastic, resumable, partial-denoise-execution, or advanced-workflow behavior.
Evidence applies only to the exact pinned host revision; static and live schema reports remain
insufficient substitutes for real-host H1/H2/H3.
