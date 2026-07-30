#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_path="${repository_root}/.venv-wsl/bin/python"

if [[ ! -x "${python_path}" ]]; then
  echo "Missing .venv-wsl. Run: python3 -m venv .venv-wsl && .venv-wsl/bin/python -m pip install -e '.[dev]'" >&2
  exit 2
fi

cd "${repository_root}"
exec "${python_path}" scripts/run_full_gate.py
