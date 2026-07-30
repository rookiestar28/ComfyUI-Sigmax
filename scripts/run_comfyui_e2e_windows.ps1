$ErrorActionPreference = "Stop"

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProjectPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $ProjectPython -PathType Leaf)) {
    throw "Missing project-local .venv. Create it and install the repository dev dependencies."
}
if ([string]::IsNullOrWhiteSpace($env:COMFYUI_ROOT)) {
    throw "COMFYUI_ROOT must point to the pinned supported ComfyUI checkout."
}
if ([string]::IsNullOrWhiteSpace($env:SIGMAX_COMFYUI_PYTHON)) {
    throw "SIGMAX_COMFYUI_PYTHON must point to that checkout's compatible Python interpreter."
}

Push-Location $RepositoryRoot
try {
    & $ProjectPython "scripts/run_comfyui_e2e.py" @args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
