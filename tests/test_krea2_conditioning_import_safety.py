from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
from comfyui_sigmax.adapters import krea2_conditioning as adapter
from comfyui_sigmax.conditioning import (
    ConditioningModifierRequest,
    Krea2ConditioningProfileId,
    Krea2ConditioningVariant,
    get_krea2_profile,
)


def _request() -> ConditioningModifierRequest:
    return ConditioningModifierRequest(
        variant=Krea2ConditioningVariant.RAW,
        profile=get_krea2_profile(Krea2ConditioningProfileId.DISABLED),
        strength=0.0,
    )


def test_conditioning_adapter_imports_without_importing_torch() -> None:
    assert "torch" not in adapter.__dict__


def test_conditioning_adapter_fails_closed_when_torch_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def missing_torch(name: str, package: str | None = None) -> object:
        if name == "torch":
            raise ImportError("test-only missing torch")
        return real_import(name, package)

    with (
        patch("importlib.import_module", missing_torch),
        pytest.raises(adapter.ConditioningAdapterError, match="optional Torch runtime"),
    ):
        adapter.transform_krea2_conditioning([], _request())
