"""Phase 0 RED contract for the source-qualified Anima family profile."""

from __future__ import annotations

import importlib.util
from enum import Enum
from typing import Any

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain


def _module() -> Any:
    module = __import__("comfyui_sigmax.profiles.anima", fromlist=["*"])
    return module


def _api(name: str) -> Any:
    value = getattr(_module(), name, None)
    assert value is not None, f"Anima Phase 0 API is missing: {name}"
    return value


def test_anima_profile_module_exists_before_profile_contract_expands() -> None:
    """The Phase 0 profile module is a required implementation seam."""

    assert importlib.util.find_spec("comfyui_sigmax.profiles.anima") is not None


def test_anima_variants_are_explicit_and_released_v1_identities() -> None:
    variant_type = _api("AnimaVariant")
    assert isinstance(variant_type, type) and issubclass(variant_type, Enum)
    assert tuple(item.value for item in variant_type) == ("base", "aesthetic", "turbo")

    for name, expected_id in (
        ("ANIMA_BASE_SCHEMA", "anima.base.framework-reference"),
        ("ANIMA_AESTHETIC_SCHEMA", "anima.aesthetic.framework-reference"),
        ("ANIMA_TURBO_SCHEMA", "anima.turbo.framework-reference"),
    ):
        schema = _api(name)
        assert schema.profile_id == expected_id
        assert schema.profile_version == "1"
        assert schema.model_family == "anima"
        assert schema.evidence.value == "framework_reference"


def test_anima_fixed_shift_schedule_is_unit_flow_and_zero_terminated() -> None:
    variant_type = _api("AnimaVariant")
    builder = _api("build_anima_schedule")
    assert callable(builder)
    result = builder(variant=variant_type.BASE, steps=4)
    assert result.final_domain is SigmaDomain.UNIT_FLOW
    assert result.sigmas == pytest.approx((1.0, 0.9, 0.75, 0.5, 0.0), abs=1e-15)
    assert result.request.transforms[0].name == "rational_shift.fixed"
    assert result.request.transforms[0].stage.value == "PRIMARY_TIME_SHIFT"
    assert result.request.transforms[1].name == "terminal.append_zero"


def test_anima_requires_explicit_variant_and_rejects_double_shift() -> None:
    variant_type = _api("AnimaVariant")
    builder = _api("build_anima_schedule")
    with pytest.raises(ScheduleContractError, match="variant"):
        builder(variant="auto", steps=4)
    with pytest.raises(ScheduleContractError, match=r"already shifted|composition"):
        builder(variant=variant_type.BASE, steps=4, already_shifted=True)


@pytest.mark.parametrize(
    ("variant_name", "reference_steps", "guidance"),
    [("BASE", 50, 4.5), ("AESTHETIC", 50, 4.5), ("TURBO", 8, 1.0)],
)
def test_anima_recipe_metadata_is_variant_explicit(
    variant_name: str, reference_steps: int, guidance: float
) -> None:
    schemas = {
        "BASE": _api("ANIMA_BASE_SCHEMA"),
        "AESTHETIC": _api("ANIMA_AESTHETIC_SCHEMA"),
        "TURBO": _api("ANIMA_TURBO_SCHEMA"),
    }
    schema = schemas[variant_name]
    expected_steps = (30, 50) if variant_name != "TURBO" else (8, 12)
    assert schema.recipes[0].steps.reference_steps == expected_steps
    assert schema.recipes[0].guidance.model_value == guidance
    assert schema.parameters == tuple(sorted(schema.parameters, key=lambda field: field.name))
    assert (
        schema.model_variant
        == {
            "BASE": "base-v1.0",
            "AESTHETIC": "aesthetic-v1",
            "TURBO": "turbo-v1.0",
        }[variant_name]
    )
