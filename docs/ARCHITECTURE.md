# Architecture

## Status

This document separates the **implemented foundation** from the **planned product
architecture**. Planned components are design targets and do not imply working ComfyUI nodes
or validated model support.

## Vocabulary

Sigmax uses four terms deliberately:

- **Model sampling semantics** map an external sigma or time value to the model's internal
  conditioning.
- **Schedule** is the ordered sigma or timestep sequence.
- **Sampler** is the numerical integration method that updates the latent.
- **Profile** is a versioned model-specific contract combining supported semantics, schedule,
  sampler compatibility, evidence, and recommendations.

A Diffusers class that owns both schedule creation and step behavior cannot automatically be
reduced to a ComfyUI `SIGMAS` output. Controls that affect `step()` belong in a sampler
implementation.

## Implemented Foundation

The repository currently implements:

```text
comfyui_sigmax/
  __init__.py   side-effect-free package contract
  py.typed      typing marker
  core/
    base_grids.py
                Krea reciprocal-step and generic descending linear grids
    schedule_contracts.py
                ownership, sigma-domain, and transform-chain preflight
    request_result.py
                immutable requested/effective inputs and structural results

scripts/
  preflight.py          local environment validation
  run_full_gate.py      canonical ordered acceptance gate
  OS wrappers           repo-local environment selection

tests/
  import, package, quality, CI, and documentation contracts
```

The package exports empty ComfyUI node mappings. Importing it does not register schedulers,
patch PyTorch, import Diffusers, or alter host process state.

The implemented pure-core preflight requires exactly one ownership mode:

- `MODEL_NATIVE`: the host/model sampling object owns the schedule;
- `EXTERNAL_SIGMAS`: Sigmax owns a complete external sigma chain;
- `MODEL_PATCH`: an explicit host adapter replaces the model sampling object.

Native and patched ownership accept no external transform chain. External ownership uses
explicit domains and the ordered stages `PRIMARY_TIME_SHIFT`, `OPTIONAL_SPACING`, `TERMINAL`,
and `SLICE`. Domain discontinuity, stage regression, and duplicate shift/spacing transforms
fail before numerical execution.

Immutable request/result contracts now bind that preflight to:

- requested and effective steps/dimensions;
- base-grid identity and output domain;
- terminal and slicing policy;
- source/profile provenance and evidence level;
- warnings and explicit requested-to-effective overrides;
- the structural sigma tuple and validated final domain.

The contracts deliberately do not yet claim that sigma values are finite, monotonic, correctly
terminated, or numerically authoritative. Those checks follow the builders and canonical
artifact specification.

The first numerical builders are now implemented:

- Krea reciprocal-step returns the non-terminal unit-flow values from `1` through `1 / steps`;
- generic linear endpoint construction returns a finite strictly descending grid in an
  explicit non-opaque domain.

Terminal zero is not part of either builder. It remains a later terminal-stage operation.

## Planned Layered Design

```text
ComfyUI nodes and diagnostics
            |
ComfyUI model/host adapters
            |
Profile resolver and registry
            |
Pure schedule engine
            |
Optional sampler backends
```

### Pure schedule engine

Planned responsibilities:

- construct explicit base grids;
- apply named and domain-checked time shifts;
- apply at most one compatible optional spacing transform;
- append terminal values and slice schedules;
- validate monotonicity, finiteness, length, and domain;
- emit deterministic metadata and fingerprints.

This layer must not require ComfyUI or Diffusers for closed-form schedule formulas.

### Profile resolver and registry

Profiles will carry model identity, variant, evidence level, sigma domain, base-grid
construction, shift parameterization, terminal policy, sampler compatibility, and provenance.
Automatic resolution must expose confidence and fail closed when an official profile is
ambiguous.

### ComfyUI adapters and nodes

Adapters will inspect available host model metadata without assuming that one internal model
class uniquely identifies a checkpoint variant. Nodes will expose a simple model-aware path,
an advanced explicit schedule path, and schedule inspection/comparison outputs.

### Sampler strategy

The first integration target is ComfyUI's native deterministic Euler sampler with an explicit
validated sigma sequence. Sigmax should reuse the native sampler whenever it is mathematically
equivalent.

A separate full sampler is planned only for behavior that cannot be encoded by sigmas alone,
such as scheduler-owned stochastic noise, stateful indices, or per-token timesteps. Alternative
integrators remain profile-scoped and experimental until independently validated.

## Data Flow

```text
MODEL + explicit profile/variant + dimensions + steps
                         |
                  profile resolution
                         |
                schedule specification
                         |
       base grid -> shift -> optional transform
                         |
             terminal/slicing/validation
                         |
              SIGMAS + SCHEDULE_INFO
                         |
              compatible ComfyUI sampler
```

Every automatic or overridden decision must be reflected in `SCHEDULE_INFO`.

## Invariants

- No silent `mu = 0` fallback.
- No silent RAW/Turbo variant choice.
- No dummy sigmas for unsupported models.
- No double shifting.
- No automatic global PyTorch or ComfyUI monkey patches.
- No user control without an executed effect.
- Official, framework-reference, community, and experimental evidence remain distinct.
- Numerical parity precedes subjective quality optimization.

## Related Documents

- [Profile specification](PROFILE_SPEC.md)
- [Compatibility](COMPATIBILITY.md)
- [Contributing](../CONTRIBUTING.md)
