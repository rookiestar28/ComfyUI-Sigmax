# Stable Public Contracts

## Status

- **Manifest schema:** `sigmax.public-contract-manifest/1`
- **Envelope schema:** `sigmax.public-contract-manifest-envelope/1`
- **Contract freeze version:** `1`
- **Migration policy version:** `1`
- **Canonical resource:** `comfyui_sigmax/contracts/manifest_v1.json`

The manifest is the release boundary frozen by M8-01. It records the exact built-in ComfyUI
node IDs, their versioned output schemas, the profile/capability schemas, construction and
execution schemas, and the stable capability reason-code vocabularies. Its SHA-256 fingerprint
binds that inventory to one canonical JSON projection.

The dependency-free loader rejects malformed, non-canonical, duplicated, unsafe, tampered, or
source-divergent manifests. Run the reproducibility check with:

```text
python scripts/generate_public_contract_manifest.py --check
```

## Frozen node contracts

| Node ID | Versioned schema |
| --- | --- |
| `Sigmax.AdvancedFlowMatchScheduler` | `sigmax.advanced-flowmatch-node/1` |
| `Sigmax.Krea2SigmaScheduler` | `sigmax.krea2-sigma-node/1` |
| `Sigmax.ModelAwareSigmaScheduler` | `sigmax.model-aware-sigma-node/1` |
| `Sigmax.ProfileInspector` | `sigmax.profile-inspector/1` |
| `Sigmax.RawWorkflowOutput` | `sigmax.raw-workflow-output/1` |
| `Sigmax.ScheduleComparison` | `sigmax.schedule-comparison/1` |
| `Sigmax.ScheduleInspector` | `sigmax.schedule-inspector/1` |
| `Sigmax.TurboWorkflowOutput` | `sigmax.turbo-workflow-output/1` |

The ID and associated schema are one pair. An existing ID cannot be reused for a different
contract.

## Frozen data contracts

Profile and capability:

- `sigmax.model-profile/1`
- `sigmax.capability-resolution/1`

Schedule construction:

- `sigmax.numerical-schedule/1`
- `sigmax.schedule-artifact/1`
- `sigmax.schedule-artifact-envelope/1`

Execution evidence:

- `sigmax.execution-receipt/1`
- `sigmax.execution-receipt-envelope/1`
- `sigmax.portable-execution-bundle/1`

The complete reason-code arrays are intentionally machine-readable in the manifest. The
`compatibility` group is the exact ordered `CompatibilityReason` vocabulary. The
`capability_resolution` group is the canonical sorted union of `core.<compatibility-code>` and
the host/model resolution codes. Free-form exception text is never a stable reason code.

Execution receipt failure/interruption reasons are producer-owned bounded identifiers. Sigmax
does not yet execute a sampler, so it does not claim or freeze a package-owned execution-failure
vocabulary in contract version 1.

## Migration policy v1

- A new compatible contract is added under a new identifier. Frozen identifiers are exact, not
  extensible by silently adding fields or changing semantics.
- An incompatible field, validation, numerical, or semantic change requires a new schema major,
  a project major-version plan, and a migration note.
- A built-in node rename or replacement requires an explicit compatibility alias and migration
  path. A removed ID cannot be reused.
- Deprecation must be documented before release and the deprecated frozen identifier remains
  supported through the current project major.
- Readers support every identifier in frozen contract version 1. Unknown schema identifiers fail
  closed rather than being interpreted as the closest known version.
- Additions to stable reason vocabularies require a new contract-manifest version. Removing,
  renaming, reordering ordered codes, or changing their meaning is breaking.

This policy does not promise forward compatibility with unknown schemas and does not convert old
payloads implicitly. When a future schema major is introduced, its release must provide explicit
conversion guidance and retain the v1 reader for the support window above.

## Scope exclusions

Internal benchmark matrices, CI evidence, environment diagnostics, workflow-validation reports,
and experimental future sampler contracts are not frozen by manifest version 1. They remain
versioned for reproducibility but are not part of this public release compatibility promise.
