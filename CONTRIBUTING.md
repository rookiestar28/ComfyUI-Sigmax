# Contributing

ComfyUI-Sigmax is in pre-alpha development. Contributions should preserve the separation
between model semantics, schedule construction, numerical samplers, and model profiles.

## Environment

Use a repository-local environment. The project supports Python 3.10 or newer.

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

## Development Rules

- Add or update a failing contract test before production behavior.
- Keep pure schedule mathematics independent of ComfyUI and Diffusers where possible.
- Do not add model-specific defaults to a generic scheduler.
- Do not silently guess model variants or missing shift parameters.
- Do not register unfinished nodes or patch global framework behavior at import time.
- Do not expose controls that have no executed effect.
- Cite authoritative sources and numerical tolerances for parity claims.
- Label non-authoritative behavior as framework-reference, community, or experimental.
- Keep changes focused and preserve source attribution.

## Validation

The canonical acceptance commands run secret scanning, all pre-commit hooks, static checks,
unit tests with coverage, and an isolated wheel inventory.

Windows:

```powershell
powershell -File scripts/run_full_tests_windows.ps1
```

Linux or WSL:

```bash
bash scripts/run_full_tests_linux.sh
```

Run the complete gate before requesting review. A hook that modifies files is not a pass;
review the change and rerun until the repository is clean. The detailed workflow is documented
in the [test SOP](tests/TEST_SOP.md).

## Pull Requests

Describe:

- the problem and intended behavior;
- the model/profile and evidence class, if relevant;
- tests added and observed failure before the fix;
- complete validation results;
- compatibility or migration risks;
- documentation and changelog impact.

Do not describe attractive output alone as schedule correctness. Numerical construction,
provenance, and reproducibility are required.

## License

By contributing, you agree that your contribution is distributed under the repository's
[MIT License](LICENSE.TXT). Retain applicable attribution notices.
