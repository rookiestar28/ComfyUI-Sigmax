# Contributing

ComfyUI-Sigmax 1.0.0 has a stable public-contract baseline. Contributions must preserve the
separation between model semantics, schedule construction, numerical samplers, and model profiles.

## Environment

Use a repository-local environment. The project supports Python 3.10 or newer. Node.js 18 or
newer is also required for contributors because the default full gate validates the scoped
ComfyUI frontend policy; hosted CI uses Node.js 20. No npm install or Playwright dependency is
required for that policy gate.

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Linux or WSL:

```bash
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install -e '.[dev]'
```

Do not mix global and environment-installed test tools in one validation run.

The full-gate wrappers run `sigmax.environment-diagnostics/1` before hooks or tests. Diagnostics
use stable issue codes and one-path remediation for local-venv mismatches, pre-commit cache
corruption/locks, conflicting executables, file operations, non-ASCII paths, repository-local
temp paths, and selected optional extras. Do not bypass a failure by switching to a global
Python or `pre-commit`; follow the reported local command and rerun.

## Architecture boundaries

- Add or update a failing contract test before production behavior.
- Keep pure schedule mathematics independent of ComfyUI and Diffusers where possible.
- Do not add model-specific defaults to a generic scheduler.
- Do not silently guess model variants or missing shift parameters.
- Do not register unfinished nodes or patch global framework behavior at import time.
- Do not expose controls that have no executed effect.
- Cite authoritative sources and numerical tolerances for parity claims.
- Label non-authoritative behavior as framework-reference, community, or experimental.
- Keep the Krea 2 conditioning rebalance isolated to its `CONDITIONING` adapter; do not turn
  diagnostics into default runtime telemetry or imply prompt-adherence improvement. The bounded
  H4 execution/provenance lane closed without blind scoring and does not establish a quality
  claim.
- Keep changes focused and preserve source attribution.

## Change workflow

1. Define objective acceptance criteria and the affected contract/evidence boundary.
2. For behavior changes and bug fixes, use `Reproduce -> Pin -> Sweep`: observe the failure, add a
   focused failing regression, implement the smallest correction, then run dependent and full
   gates.
3. Regenerate only artifacts whose declared sources changed; rerun first/repeat compatibility
   lanes when their invariant contract changes.
4. Update public documentation, changelog, migration impact, and rollback guidance with the code.
5. Keep commits scoped and use Conventional Commits.

## Public contract and migration changes

Treat identifiers in `sigmax.public-contract-manifest/1` as frozen. Run:

```powershell
python scripts/generate_public_contract_manifest.py --check
```

An intentional breaking node ID, schema, artifact, receipt, reason code, or schedule-semantics
change requires a new schema/project major as applicable, a reviewed migration plan, compatibility
evidence, and a user-facing migration note. A profile version bump cannot conceal a wire break.

## Validation

The canonical acceptance commands run secret scanning, all pre-commit hooks, static checks,
core-independence and frontend-policy checks, unit tests with coverage, and an isolated wheel
inventory.

Windows:

```powershell
powershell -File scripts/run_full_tests_windows.ps1
```

Linux or WSL:

```bash
bash scripts/run_full_tests_linux.sh
```

For non-documentation changes, run the complete applicable gate before requesting review. A hook
that modifies files is not a pass; review the change and rerun until the repository is clean. The
detailed workflow is documented in the [test SOP](tests/TEST_SOP.md).

Strictly documentation-only prose changes use the test-SOP exception: review Markdown structure,
commands, paths, links, and changed claims directly, but do not run the full gate or host E2E.
Changes to executable examples, generated manifests, workflows, CI behavior, package metadata, or
security/release contracts are not pure-prose changes and still require their applicable gates.

## Release-facing validation

Release-facing changes must also pass the local non-publishing audit with fresh paths:

```powershell
python scripts/run_release_audit.py --dist-dir .tmp/release-audit-dist --output .tmp/release-audit.json
```

Registry-facing changes must additionally stage the intended files and validate the exact
Git-indexed candidate without publishing:

```powershell
python scripts/validate_registry_artifact.py --archive .tmp/comfy-registry/comfyui-sigmax-1.0.0.zip --check-manifest --observe-registry --output .tmp/comfy-registry/report.json
```

Review `.comfyignore`, the canonical manifest, archive fingerprint, normalized-directory import,
and `publication_performed: false` together. Registry observation is read-only and does not prove
publisher ownership.

Never weaken `MANIFEST.in` or archive deny rules merely to include a local test, planning record,
reference clone, cache, model file, or private log.

## Review evidence

Describe:

- the problem and intended behavior;
- the model/profile and evidence class, if relevant;
- tests added and observed failure before the fix;
- complete validation results;
- compatibility or migration risks;
- documentation and changelog impact.

Do not describe attractive output alone as schedule correctness. Numerical construction,
provenance, and reproducibility are required.

Review is incomplete without the observed failing regression when applicable, targeted pass,
dependent generator/parity/host evidence, complete Windows and WSL gate results, environment and
interpreter identity, migration/rollback assessment, and known limitations. A retry cannot erase a
first-attempt P0 regression; record the cause and corrective rerun.

## Model Profile Contributions

A model profile is accepted as an evidence package, not as a set of preferred values. Include
the following evidence in the pull request.

At minimum, profile review requires:

- separate pinned software-source, framework, and model-weight provenance with a separate
  license declaration for each resource layer;
- explicit formulas, domains, ownership, grid/transforms/order, terminal/slicing, geometry,
  recipes, guidance, capabilities, and fail-closed detection;

Schedule-only generic FlowMatch declarations use the separate
`sigmax.generic-flowmatch-profile/1` contract and never substitute for this concrete-model
review. A generic declaration must remain explicitly selected, non-official, and outside
`ProfileRegistry`; onboarding any model still requires the complete evidence above.
- independent complete float64/float32 goldens, formula tests, pinned parity with dtype/device
  and error bounds, deterministic identities, and reproducible regeneration;
- a canonical validator-clean workflow plus supported-host evidence whenever node, adapter,
  workflow, or host behavior changes;
- evidence level, known limitations, migration impact, attribution, security, dependency,
  package, and rollback review.

An override, inherited child, step change, formula change, or other material deviation from
reference behavior uses `modified` evidence. Image quality, filenames, local headers, shared
model classes, or queue acceptance cannot substitute for numerical identity, exact variant
evidence, or completed execution.

## License

By contributing, you agree that your contribution is distributed under the repository's
[MIT License](LICENSE.TXT). Retain applicable attribution notices.

The repository license does not grant rights to upstream software, frameworks, model weights,
datasets, trademarks, or generated media. Missing or incompatible rights for any required
resource layer block acceptance.
