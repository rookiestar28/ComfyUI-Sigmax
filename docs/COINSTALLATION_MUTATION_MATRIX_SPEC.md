# Co-Installation and Host-Mutation Matrix Specification

- **Snapshot schema:** `sigmax.host-mutation-snapshot/1`
- **Evaluation schema:** `sigmax.co-installation-evaluation/1`
- **Matrix schema:** `sigmax.co-installation-mutation-matrix/1`
- **Envelope schema:** `sigmax.co-installation-mutation-matrix-envelope/1`
- **Maturity:** implemented with deterministic repository-owned synthetic evidence

## Purpose

This boundary detects whether installation or reload behavior changed a protected host identity
or introduced a double-shift condition. It records semantic fingerprints rather than serializing
live Python objects, memory addresses, callables, or process state.

The packaged matrix is a reproducible regression contract. It is not a compatibility
certification for any named third-party node pack.

## Immutable Snapshot

`HostMutationSnapshot` records:

- sorted unique node IDs, providers, and definition fingerprints;
- sorted unique scheduler names, providers, and handler fingerprints;
- semantic fingerprints for the PyTorch `Module.__call__` path and shared model-patch state;
- schedule ownership (`external_sigmas` or `model_native`);
- construction-shift count and whether model-native sigmas are already shifted.

`evaluate_host_mutation` returns an immutable `CoInstallationEvaluation`. New unrelated node or
scheduler entries are allowed. The following findings produce `reject`:

| Finding | Protected condition |
| --- | --- |
| `node_registry_collision` | An existing node identity was removed or replaced |
| `sigmax_namespace_hijack` | An external provider added a new `Sigmax.` node ID |
| `scheduler_registry_overwrite` | An existing scheduler handler was removed or replaced |
| `torch_call_path_changed` | The protected PyTorch call-path identity changed |
| `model_patch_state_changed` | Shared model-patch state changed |
| `construction_shift_repeated` | More than one construction shift was applied |
| `model_native_external_double_shift` | External shift was combined with model-native shifting |

## Synthetic Matrix

The fixed fixture inventory contains ten sorted scenarios:

1. clean install;
2. idempotent reload;
3. unrelated node addition;
4. unrelated scheduler addition;
5. node-ID collision;
6. external `Sigmax.` namespace registration;
7. scheduler overwrite;
8. PyTorch call-path replacement;
9. shared model-patch mutation;
10. model-native/external double shift.

Every scenario executes twice from a fresh immutable baseline. Reload is additionally applied
twice to the same snapshot and must remain unchanged. Expected verdicts and exact finding sets
must match both observations, and first/repeat report fingerprints must be identical.

The baseline is bound to `builtin_node_registry()`. The matrix context is also bound to the
packaged M7-04 dependency compatibility matrix fingerprint, so dependency evidence drift cannot
be silently inherited.

## Canonical Transport and Fail-Closed Loading

The loader accepts only canonical UTF-8 JSON with a trailing newline. It rejects:

- duplicate keys, unknown fields, floats, non-finite values, BOMs, or oversized/deep input;
- private or absolute paths and secret-like field names;
- missing, duplicate, unsorted, failed, or non-repeatable rows;
- unsupported operations, findings, or verdicts;
- expectation/observation disagreement or a false `allow`;
- row-result, matrix, or dependency-compatibility fingerprint drift;
- an evidence path that is absent from the declared source inventory.

The source inventory binds the node registration catalog, dependency matrix, evaluator,
synthetic fixtures, and executed synthetic evidence by SHA-256.

## Reproduction

Use one repository-local Python environment:

```bash
python scripts/run_synthetic_coinstallation_matrix.py \
  --output tests/coinstallation/fixtures/synthetic_mutation_evidence_v1.json \
  --check
python scripts/generate_coinstallation_mutation_matrix.py --check
```

The first command independently executes the fixed fixture inventory. The second refuses to
publish if execution differs from the checked-in evidence and then verifies the canonical
packaged matrix.

## Security and Claim Boundary

Synthetic fixture fields select only a closed set of repository-owned operations. They cannot
provide an import path, expression, callable, code string, installer, lifecycle command,
container, model weight, or network artifact.

No external or `reference/` code executes in this lane. A future named-pack observation requires
an item-specific reviewed plan and explicit approval before acquisition or execution, and its
result must remain distinct from synthetic evidence.
