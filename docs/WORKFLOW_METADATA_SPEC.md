# Workflow Metadata Specification

- **Specification:** Sigmax workflow metadata
- **Version:** 1
- **Metadata schema:** `sigmax.workflow-metadata/1`
- **Transport envelope:** `sigmax.workflow-metadata-envelope/1`
- **Workflow namespace:** `extra.comfyui_sigmax`

## 1. Purpose and boundary

Workflow metadata preserves the exact Sigmax requirements and evidence references needed to
understand a saved ComfyUI graph. It does not replace the host workflow schema and does not validate
nodes, links, slots, widgets, positions, models, subgraphs, or live-host availability.

The contract references artifacts and receipts by SHA-256 identity. It never embeds their full
envelopes, prompts, images, latents, model files, private paths, credentials, timestamps, or raw
exceptions.

## 2. Metadata projection

The projection has these exact members:

| Member | Meaning |
| --- | --- |
| `schema` | Exact value `sigmax.workflow-metadata/1` |
| `requirements.package` | Exact public package ID and version |
| `requirements.nodes` | Sorted, unique namespaced node IDs and versions |
| `requirements.host` | Exact public host ID, host version, and API version |
| `profile` | Exact profile ID and version |
| `compatibility` | Complete allow/warn/reject decision, considered dimensions, and stable reasons |
| `artifact` | Construction/numerical fingerprints and supported artifact/receipt schemas |
| `receipts` | Sorted, unique receipt fingerprints, status, schema, and matching artifact links |

At least one node requirement is mandatory. Receipt references may be empty for a workflow that
has not executed. Every receipt reference must use the declared artifact's construction and
numerical fingerprints.

## 3. Canonical transport

`serialize_workflow_metadata()` wraps the projection, metadata fingerprint, and envelope schema
in canonical UTF-8 JSON. `deserialize_workflow_metadata()`:

- bounds transport to 1,048,576 bytes;
- rejects BOMs, duplicate names, untyped JSON floats, nonfinite constants, unknown members, and
  noncanonical encodings;
- validates every enum, identifier, schema, ordering rule, and SHA-256 identity;
- recomputes and verifies the metadata fingerprint.

Equal validated values produce byte-identical metadata across supported processes and operating
systems.

## 4. Workflow attachment

The official ComfyUI workflow specification defines current version `1` and permits additional
members under `extra`. The official frontend also preserves legacy version `0.4`. Sigmax
therefore supports these two exact mapping forms:

- legacy workflow version `0.4`;
- current workflow version `1`.

`attach_workflow_metadata()` copies the workflow root and `extra` object, then stores the complete
metadata envelope under `extra.comfyui_sigmax`. Nodes, links, widgets, positions, model records,
subgraphs, and unrelated `extra` members are not interpreted or rewritten.

Identical attachment is idempotent. A conflicting existing namespace fails closed.
`extract_workflow_metadata()` verifies the complete envelope before returning it.
`detach_workflow_metadata()` removes only a verified Sigmax namespace and keeps unrelated
members.

Official workflow references:

- <https://docs.comfy.org/specs/workflow_json>
- <https://docs.comfy.org/development/core-concepts/workflow>

## 5. Security and compatibility

Identifiers and versions are bounded public ASCII values. Secret-like names, credentials,
private local paths, malformed fingerprints, unsupported schema versions, arbitrary extension
data, and stale or cross-artifact receipt references fail closed.

The surrounding graph remains untrusted passthrough data. Successful metadata extraction proves
only the Sigmax metadata contract; it is not evidence that the complete workflow is valid,
loadable, executable, or compatible with a running ComfyUI host.
