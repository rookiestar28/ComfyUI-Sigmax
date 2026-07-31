# Dependency Compatibility Matrix Specification

## Status

- **Matrix schema:** `sigmax.dependency-compatibility-matrix/1`
- **Envelope schema:** `sigmax.dependency-compatibility-matrix-envelope/1`
- **Invariant contract:** `sigmax.compatibility-invariant-contract/1`
- **Maturity:** implemented and accepted local, pinned, and latest-host evidence; the official
  container is explicitly unavailable/non-blocking under the immutable-digest policy

## Purpose

The dependency compatibility matrix records exactly which environment combinations have
executed a fixed Sigmax invariant contract. It separates package compatibility evidence from
dependency resolution, source inspection, CI configuration, and assumptions based on another
version.

The matrix does not install dependencies, clone a host, pull an image, start ComfyUI, or execute
external code. Those are explicit, reviewed acquisition/execution stages. The matrix loader is
dependency-free.

## Lane Roles

| Role | Meaning | Can expand support automatically? |
| --- | --- | --- |
| `known_good` | Blocking exact framework or host baseline | No; only explicit policy updates |
| `supported` | Blocking project environment that executed the invariant contract | No |
| `latest_informational` | Observational upgrade signal | Never |

The policy fixes `latest_can_expand_support` and `unavailable_is_pass` to `false`.
`known_good` lanes are always blocking. Official-container execution additionally requires a
resolvable immutable official digest; an unavailable official digest is non-blocking, and a
third-party container may not substitute.

## Status and Attempt Semantics

| Status | Required evidence |
| --- | --- |
| `passed` | Exact result fingerprint plus independently retained `passed` first and repeat attempts |
| `not_evaluated` | No result fingerprint; first/repeat are `not_evaluated` |
| `unavailable` | No result fingerprint; a stable availability reason is required |

Current stable reasons are `compatible`, `approval_required`, and
`registry_access_denied`. A non-passed lane cannot use `compatible`, and a passed lane cannot use
an availability or approval reason.

## Components

Every lane carries explicit nullable component identities for Python, PyTorch, Diffusers,
ComfyUI, numbered Comfy API, and the official container. Executed container evidence requires
an immutable registry reference containing `@sha256:`. Mutable tags may be retained only on a
non-passed discovery record.

The current policy records Python 3.10 and 3.13, ComfyUI 0.29.0, Diffusers 0.39.0, and
`v0_0_2` as an experimental Comfy API. This does not claim every Cartesian combination.

## Fixed Invariant Contract

Native Windows 3.13 and WSL 3.10 execute the same fixed test selection twice. The contract binds
exact source fingerprints for Turbo and RAW goldens, authoritative/native-ComfyUI parity,
workflow fixtures, capability/receipt conformance, artifact/receipt identities, and the
packaged numerical benchmark matrix.

It also requires zero mandatory runtime dependencies. Both local lanes must emit the same
contract and test-selection fingerprints. Any failed first attempt remains failed evidence; a
repeat cannot upgrade it.

The generator refuses to publish a configured `passed` local lane unless its canonical evidence
actually reports `passed` for both attempts, the expected lane/platform/Python, and zero
mandatory dependencies.

Latest-host evidence is separately sanitized from the approved raw H1/H2/H3 run. Publication
requires the exact revision/version/runtime identity, CPU-only and no-model-weight declarations,
an unchanged import-safety probe, all nine required host transitions, matching first/repeat
result fingerprints, and a recomputed stable result fingerprint. Ports, durations, local paths,
and host log text are not published.

## Known-Good and External Boundaries

The accepted known-good host remains exact ComfyUI 0.29.0 revision
`e651b7bef55a5376343dcb1c0edb79f0142c985e`, Python 3.13, Torch 2.13.0, and experimental
numbered API `v0_0_2`. The framework-reference lane remains Diffusers 0.39.0 and Torch 2.9.0.
These rows reference already accepted public evidence; the matrix does not pretend to rerun it.

Official ComfyUI v0.29.2 revision `322122449c9d2ba8b8df1bb517364527dd0615f1` and repository
HEAD `5cc026f5b81b3f01fe7a1438a0fd4131d2ebda25` passed separate Windows Python 3.13.9 /
Torch 2.13.0+CPU H1/H2/H3 first/repeat observations without model weights. These
`latest_informational` rows cannot promote the supported baseline. The official
Comfy Org CI-container row remains `unavailable` because no immutable registry digest could be
obtained under approved access. This explicit non-PASS state does not block acceptance, source
review or a mutable tag is not container execution, and no third-party image may substitute.

## Canonical Transport and Validation

The resource is canonical UTF-8 JSON followed by exactly one LF. Objects use sorted keys,
compact separators, no BOM, and no untyped JSON floats or non-finite values. The loader rejects:

- duplicate or unknown fields and duplicate/reordered lane IDs;
- oversized, noncanonical, secret-like, or private-path-bearing values;
- unsupported roles, statuses, reasons, platforms, or API policy;
- nonzero mandatory runtime dependencies;
- false PASS, missing first/repeat evidence, or result fingerprints on non-passed lanes;
- latest lanes marked blocking or known-good lanes marked nonblocking;
- passed containers without immutable digests;
- envelope or matrix fingerprint drift.

## Regeneration and Local Evidence

Generate or verify the matrix:

```bash
python scripts/generate_dependency_compatibility_matrix.py --check
```

Execute an already-approved local environment without acquisition:

```powershell
.\.venv\Scripts\python.exe scripts\run_dependency_compatibility_lane.py `
  --lane-id core-windows-py313 `
  --output tests\compatibility\fixtures\windows_py313_v1.json
```

```bash
.venv-wsl/bin/python scripts/run_dependency_compatibility_lane.py \
  --lane-id core-wsl-py310 \
  --output tests/compatibility/fixtures/wsl_py310_v1.json
```

External host or container acquisition and execution remains a separate approval-controlled
procedure. The packaged matrix contains only validated, sanitized evidence from completed runs.
