"""Static packaging and safety contract for the Sigmax frontend extension."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_krea2_frontend_extension_files_are_present_and_scoped() -> None:
    policy = ROOT / "web" / "krea2_strict_official_policy.js"
    extension = ROOT / "web" / "krea2_strict_official_extension.js"

    assert policy.is_file()
    assert extension.is_file()
    source = extension.read_text(encoding="utf-8")
    assert 'name: "Sigmax.Krea2StrictOfficialPolicy"' in source
    assert 'nodeData.name !== "Sigmax.Krea2SigmaScheduler"' in source
    assert "beforeRegisterNodeDef" in source
    assert "onNodeCreated" in source
    assert "onConfigure" in source
    for forbidden in ("fetch(", "XMLHttpRequest", "localStorage", "eval(", "Function("):
        assert forbidden not in source


def test_frontend_extension_is_selected_for_registry_packaging() -> None:
    comfyignore = (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines()

    assert "web/" not in {line.strip() for line in comfyignore}
