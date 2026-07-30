#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_python="${repository_root}/.venv-wsl/bin/python"

if [[ ! -x "${project_python}" ]]; then
  echo "Missing project-local .venv-wsl. Create it and install the repository dev dependencies." >&2
  exit 2
fi
if [[ -z "${COMFYUI_ROOT:-}" ]]; then
  echo "COMFYUI_ROOT must point to the pinned supported ComfyUI checkout." >&2
  exit 2
fi
if [[ -z "${SIGMAX_COMFYUI_PYTHON:-}" ]]; then
  echo "SIGMAX_COMFYUI_PYTHON must point to that checkout's compatible Python interpreter." >&2
  exit 2
fi

cd "${repository_root}"
exec "${project_python}" scripts/run_comfyui_e2e.py "$@"
