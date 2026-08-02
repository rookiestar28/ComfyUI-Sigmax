"""Phase 0 RED contracts for the source-qualified Wan family profiles."""

from __future__ import annotations

import importlib.util
from enum import Enum
from typing import Any

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain


def _module() -> Any:
    module = __import__("comfyui_sigmax.profiles.wan", fromlist=["*"])
    return module


def _api(name: str) -> Any:
    value = getattr(_module(), name, None)
    assert value is not None, f"Wan Phase 0 API is missing: {name}"
    return value


def test_wan_profile_module_exists_before_profile_contract_expands() -> None:
    assert importlib.util.find_spec("comfyui_sigmax.profiles.wan") is not None


def test_wan_axes_are_explicit_and_not_derivative_aliases() -> None:
    generation = _api("WanGeneration")
    task = _api("WanTask")
    source = _api("WanSource")
    resolution = _api("WanResolution")
    assert issubclass(generation, Enum)
    assert tuple(item.value for item in generation) == ("wan2.1", "wan2.2")
    assert tuple(item.value for item in task) == ("t2v", "i2v", "ti2v")
    assert tuple(item.value for item in source) == (
        "comfy_native",
        "official_native",
        "diffusers_reference",
    )
    assert tuple(item.value for item in resolution) == ("none", "480p", "720p")


@pytest.mark.parametrize(
    ("name", "expected_family", "expected_shift", "expected_steps"),
    [
        ("WAN21_COMFY_NATIVE_SCHEMA", "wan", 8.0, 50),
        ("WAN21_T2V_OFFICIAL_SCHEMA", "wan", 5.0, 50),
        ("WAN21_I2V_480P_OFFICIAL_SCHEMA", "wan", 3.0, 40),
        ("WAN21_I2V_720P_OFFICIAL_SCHEMA", "wan", 5.0, 40),
        ("WAN22_TI2V_5B_NATIVE_SCHEMA", "wan", 5.0, 50),
        ("WAN22_T2V_A14B_NATIVE_SCHEMA", "wan", 12.0, 40),
        ("WAN22_I2V_A14B_NATIVE_SCHEMA", "wan", 5.0, 40),
    ],
)
def test_wan_schema_matrix_is_explicit(
    name: str, expected_family: str, expected_shift: float, expected_steps: int
) -> None:
    schema = _api(name)
    assert schema.profile_version == "1"
    assert schema.model_family == expected_family
    assert schema.parameters == tuple(sorted(schema.parameters, key=lambda item: item.name))
    assert schema.recipes[0].steps.default == expected_steps
    assert any(
        field.name in {"shift", "ratio"} and field.value == expected_shift
        for field in schema.parameters
    )


def test_wan_unit_flow_schedule_applies_one_shift_and_appends_zero() -> None:
    builder = _api("build_wan_schedule")
    profile = _api("WanProfileId")
    result = builder(profile=profile.WAN21_T2V_OFFICIAL, steps=4)
    assert result.final_domain is SigmaDomain.UNIT_FLOW
    assert result.sigmas == pytest.approx((1.0, 0.9375, 0.8333333333333334, 0.625, 0.0), abs=1e-15)
    assert result.request.transforms[0].name == "direct_ratio.shift"
    assert result.request.transforms[1].name == "terminal.append_zero"


def test_wan_requires_explicit_identity_and_rejects_double_shift() -> None:
    module = _module()
    builder = _api("build_wan_schedule")
    with pytest.raises(ScheduleContractError, match="profile"):
        builder(profile="auto", steps=4)
    with pytest.raises(ScheduleContractError, match=r"already shifted|composition"):
        builder(profile=module.WanProfileId.WAN21_T2V_OFFICIAL, steps=4, already_shifted=True)


def test_wan_resolution_and_derivative_requests_fail_closed() -> None:
    module = _module()
    builder = _api("build_wan_schedule")
    with pytest.raises(ScheduleContractError, match="resolution"):
        builder(profile=module.WanProfileId.WAN21_I2V_480P_OFFICIAL, steps=40, resolution="720p")
    with pytest.raises(ScheduleContractError, match=r"unsupported|derivative|profile"):
        builder(profile="wan2.1.fun-control", steps=40)


def test_wan_a14b_boundary_is_metadata_not_routing() -> None:
    module = _module()
    builder = _api("build_wan_schedule")
    result = builder(profile=module.WanProfileId.WAN22_T2V_A14B_NATIVE, steps=40)
    assert result.boundary is not None
    assert result.boundary.normalized == pytest.approx(0.875)
    assert result.boundary.transition_index >= 0
    assert result.boundary.routing_owner == "caller"
    assert result.boundary.model_dispatch is False
