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
    shifts.py
                exponential-mu, direct-ratio, and explicit identity transforms
    terminal_slicing.py
                terminal append/preserve, step ranges, and denoise-tail slicing
    validation.py
                complete external-schedule numerical and policy validation
    fingerprints.py
                canonical projections and numerical/construction identities
    artifacts.py
                immutable artifact assembly and strict canonical JSON transport
    capabilities.py
                model/profile/sampler declarations and execution preflight decisions

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

Complete external schedules can now be checked for finite values, strict decrease, exact
transition count, domain bounds, and terminal policy. Validated results can be projected into
immutable schedule artifacts with separate numerical and construction identities, then
transported as bounded canonical JSON with strict untrusted-input rejection.

The first numerical builders are now implemented:

- Krea reciprocal-step returns the non-terminal unit-flow values from `1` through `1 / steps`;
- generic linear endpoint construction returns a finite strictly descending grid in an
  explicit non-opaque domain.

Terminal zero is not part of either builder. It remains a later terminal-stage operation.

The first pointwise shifts are also implemented:

- Krea/Flux exponential `mu`, including an explicit positive exponent;
- discrete-flow direct ratio, named separately from `mu`;
- an explicit no-shift identity path.

They accept only finite unit-flow tuples and reject model-native or other domains. Exact zero
and one endpoints are preserved, and algebraically equivalent stable evaluation prevents
extreme finite controls from overflowing. Resolution-to-`mu` derivation remains profile work.

Terminal and slicing operations are now implemented separately:

- append terminal zero or preserve an already terminal-inclusive vector;
- interpret `N + 1` sigmas as `N` transitions;
- retain strict `start_step:end_step + 1` transition ranges;
- calculate ComfyUI-compatible partial-denoise construction counts and retain the requested
  terminal-inclusive tail;
- represent zero denoise as an explicit empty no-execution vector.

Manual empty or out-of-range requests fail. ComfyUI's later `force_full_denoise` endpoint
replacement remains a host/sampler execution policy and is not hidden in terminal
construction.

Framework-independent capability declarations now define the compatibility boundary before
host integration:

- `ModelCapabilities` declares accepted prediction types, sigma domains, schedule ownerships,
  and support for partial denoise or per-token timesteps.
- `ProfileCapabilities` binds one model family/variant to prediction, domain, ownership,
  terminal behavior, permitted deterministic/stochastic modes, noise owners, sampler state,
  feature support, and reference sampler identifiers.
- `SamplerCapabilities` declares accepted semantics plus terminal, state, noise, and execution
  requirements.
- `CompatibilityDecision` returns a canonical `ALLOW`, `WARN`, or `REJECT` result with stable
  reason codes across every considered capability dimension.

The execution gate raises before sampler work on `REJECT`. A compatible non-reference sampler
is a warning rather than an unsupported claim. These declarations do not resolve a profile,
inspect ComfyUI, or execute a sampler; those remain later layers.

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

Responsibilities implemented or completed by the end of the pure-core stage:

- construct explicit base grids;
- compose the implemented named and domain-checked time shifts;
- apply at most one compatible optional spacing transform;
- compose the implemented terminal and slicing policies;
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
