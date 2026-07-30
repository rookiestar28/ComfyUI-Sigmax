# Workflow Validation Specification

## Status

Implemented contract: `sigmax.workflow-validation-report/1`

Envelope: `sigmax.workflow-validation-report-envelope/1`

Fixture bundle: `sigmax.workflow-fixture-bundle/1`

Pinned host baseline: `sigmax.workflow-host-baseline/1`

This specification covers deterministic schema compatibility. It does not claim that a real
ComfyUI process loaded or executed a workflow.

## Purpose

The workflow metadata layer preserves portable requirements but deliberately treats the
surrounding ComfyUI graph as host-owned data. The adjacent `comfyui_sigmax.workflows` package
validates canonical graph fixtures without importing or executing host node implementations.

The package ships two minimal model-free fixtures:

- `krea2-turbo-1024`: official Turbo, 8 steps, 1024x1024;
- `krea2-raw-1024`: official RAW, 52 steps, 1024x1024.

Each graph connects `Sigmax.Krea2SigmaScheduler` to `Sigmax.ScheduleInspector`. These fixtures
exercise node identity, linked inputs, ordered widgets, fixed variant values, M4-07 metadata, and
portable package identity. They are validator contracts, not the full user-facing workflows
owned by later roadmap items.

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

## Package Identity and Normalized Directories

Physical custom-node directory names are not workflow identities. A canonical Sigmax node must
retain:

- its stable `Sigmax.<Name>` node ID;
- host schema `python_module=comfyui_sigmax.nodes`;
- workflow property `cnr_id=comfyui-sigmax`;
- M4-07 package/node versions in `extra.comfyui_sigmax`.

Renamed or version-normalized install directories must not leak into these public identities.

## Current Limitation

The real ComfyUI H1/H2 harness remains `NOT_IMPLEMENTED`. Static scans and controlled loopback
HTTP tests are acceptance evidence for this pure validator only; they are not substitutes for a
running supported host.
