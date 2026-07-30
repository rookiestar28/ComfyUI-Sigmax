"""Immutable request/result contracts for schedule construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast

import pytest
from comfyui_sigmax.core import (
    BaseGridSpec,
    EvidenceLevel,
    OverrideRecord,
    Provenance,
    ScheduleContractError,
    ScheduleInputs,
    ScheduleOwnership,
    ScheduleRequest,
    ScheduleResult,
    SigmaDomain,
    SliceSpec,
    TerminalPolicy,
    TransformContract,
    TransformStage,
)


def _provenance() -> Provenance:
    return Provenance(
        engine_version="0.1.0.dev0",
        evidence=EvidenceLevel.EXPERIMENTAL,
        source="unit-test fixture",
    )


def _inputs(*, width: int = 1024, height: int = 1024) -> ScheduleInputs:
    return ScheduleInputs(steps=8, width=width, height=height)


def _base_grid(domain: SigmaDomain = SigmaDomain.UNIT_FLOW) -> BaseGridSpec:
    return BaseGridSpec(identifier="krea_reciprocal_step", output_domain=domain)


def _external_request(
    *,
    inputs: ScheduleInputs | None = None,
    domain: SigmaDomain = SigmaDomain.UNIT_FLOW,
    transforms: tuple[TransformContract, ...] = (),
    overrides: tuple[OverrideRecord, ...] = (),
) -> ScheduleRequest:
    return ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=inputs or _inputs(),
        sigma_domain=domain,
        base_grid=_base_grid(domain),
        transforms=transforms,
        terminal_policy=TerminalPolicy.APPEND_ZERO,
        slicing=SliceSpec(),
        provenance=_provenance(),
        overrides=overrides,
    )


def test_new_enum_wire_values_are_explicit() -> None:
    assert {policy.value for policy in TerminalPolicy} == {
        "APPEND_ZERO",
        "PRESERVE",
    }
    assert {level.value for level in EvidenceLevel} == {
        "official",
        "framework_reference",
        "community_recommended",
        "experimental",
        "modified",
    }


@pytest.mark.parametrize("steps", [0, -1, True])
def test_schedule_inputs_require_positive_integer_steps(steps: int) -> None:
    with pytest.raises(ScheduleContractError, match="steps"):
        ScheduleInputs(steps=steps)


@pytest.mark.parametrize(
    ("width", "height"),
    [(1024, None), (None, 1024), (0, 1024), (1024, -1), (True, 1024)],
)
def test_schedule_inputs_require_paired_positive_integer_dimensions(
    width: int | None,
    height: int | None,
) -> None:
    with pytest.raises(ScheduleContractError, match="width and height"):
        ScheduleInputs(steps=8, width=width, height=height)


def test_schedule_inputs_allow_omitted_dimensions() -> None:
    assert ScheduleInputs(steps=8).width is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SliceSpec(start_step=-1),
        lambda: SliceSpec(start_step=4, end_step=4),
        lambda: SliceSpec(start_step=4, end_step=3),
        lambda: SliceSpec(denoise=-0.1),
        lambda: SliceSpec(denoise=1.1),
        lambda: SliceSpec(denoise=float("nan")),
    ],
)
def test_slice_spec_rejects_invalid_bounds(factory: Callable[[], SliceSpec]) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


def test_provenance_requires_profile_id_and_version_together() -> None:
    with pytest.raises(ScheduleContractError, match="profile_id and profile_version"):
        Provenance(
            engine_version="0.1.0.dev0",
            evidence=EvidenceLevel.OFFICIAL,
            source="official fixture",
            profile_id="krea2_turbo",
        )


def test_provenance_accepts_complete_profile_and_source_revision() -> None:
    provenance = Provenance(
        engine_version="0.1.0.dev0",
        evidence=EvidenceLevel.OFFICIAL,
        source="official fixture",
        source_revision="abc123",
        profile_id="krea2_turbo",
        profile_version="1",
    )

    assert provenance.profile_version == "1"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Provenance(
            engine_version="0.1.0.dev0",
            evidence=EvidenceLevel.OFFICIAL,
            source="official fixture",
            profile_id=" ",
            profile_version="1",
        ),
        lambda: Provenance(
            engine_version="0.1.0.dev0",
            evidence=EvidenceLevel.OFFICIAL,
            source="official fixture",
            profile_id="krea2",
            profile_version=" ",
        ),
        lambda: Provenance(
            engine_version="0.1.0.dev0",
            evidence=EvidenceLevel.OFFICIAL,
            source="official fixture",
            source_revision=" ",
        ),
    ],
)
def test_provenance_rejects_empty_optional_values(factory: Callable[[], Provenance]) -> None:
    with pytest.raises(ScheduleContractError, match="must not be empty"):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BaseGridSpec(identifier=" ", output_domain=SigmaDomain.UNIT_FLOW),
        lambda: Provenance(
            engine_version=" ",
            evidence=EvidenceLevel.EXPERIMENTAL,
            source="fixture",
        ),
        lambda: OverrideRecord(
            field=" ",
            requested_value="1024",
            effective_value="1024",
            reason="fixture",
        ),
    ],
)
def test_named_contracts_reject_empty_identifiers(factory: Callable[[], object]) -> None:
    with pytest.raises(ScheduleContractError, match="must not be empty"):
        factory()


def test_external_request_requires_construction_contracts() -> None:
    with pytest.raises(ScheduleContractError, match="requires base_grid"):
        ScheduleRequest(
            ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
            requested_inputs=_inputs(),
            sigma_domain=SigmaDomain.UNIT_FLOW,
            provenance=_provenance(),
        )


def test_native_request_forbids_external_construction_fields() -> None:
    with pytest.raises(ScheduleContractError, match="cannot define external"):
        ScheduleRequest(
            ownership=ScheduleOwnership.MODEL_NATIVE,
            requested_inputs=_inputs(),
            sigma_domain=SigmaDomain.MODEL_NATIVE,
            base_grid=_base_grid(),
            provenance=_provenance(),
        )


def test_request_rejects_base_grid_domain_mismatch() -> None:
    with pytest.raises(ScheduleContractError, match="base grid output domain"):
        ScheduleRequest(
            ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
            requested_inputs=_inputs(),
            sigma_domain=SigmaDomain.CONTINUOUS_EDM,
            base_grid=_base_grid(SigmaDomain.UNIT_FLOW),
            terminal_policy=TerminalPolicy.PRESERVE,
            slicing=SliceSpec(),
            provenance=_provenance(),
        )


def test_request_inherits_double_shift_rejection() -> None:
    shift = TransformContract(
        name="forbidden external shift",
        stage=TransformStage.PRIMARY_TIME_SHIFT,
        input_domain=SigmaDomain.MODEL_NATIVE,
        output_domain=SigmaDomain.MODEL_NATIVE,
    )
    with pytest.raises(ScheduleContractError, match="double shifting"):
        ScheduleRequest(
            ownership=ScheduleOwnership.MODEL_PATCH,
            requested_inputs=_inputs(),
            sigma_domain=SigmaDomain.MODEL_NATIVE,
            transforms=(shift,),
            provenance=_provenance(),
        )


def test_collection_fields_require_tuples() -> None:
    with pytest.raises(ScheduleContractError, match="transforms must be a tuple"):
        _external_request(
            transforms=cast(tuple[TransformContract, ...], []),
        )
    with pytest.raises(ScheduleContractError, match="overrides must be a tuple"):
        _external_request(
            overrides=cast(tuple[OverrideRecord, ...], []),
        )


def test_request_collection_fields_reject_wrong_element_types() -> None:
    with pytest.raises(ScheduleContractError, match="TransformContract"):
        _external_request(
            transforms=(cast(TransformContract, "not-a-transform"),),
        )
    with pytest.raises(ScheduleContractError, match="OverrideRecord"):
        _external_request(
            overrides=(cast(OverrideRecord, "not-an-override"),),
        )


def test_valid_external_and_native_requests_are_immutable() -> None:
    external = _external_request()
    native = ScheduleRequest(
        ownership=ScheduleOwnership.MODEL_NATIVE,
        requested_inputs=_inputs(),
        sigma_domain=SigmaDomain.MODEL_NATIVE,
        provenance=_provenance(),
    )

    assert external.base_grid == _base_grid()
    assert native.base_grid is None
    with pytest.raises(FrozenInstanceError):
        external.sigma_domain = SigmaDomain.CONTINUOUS_EDM  # type: ignore[misc]


def test_result_rejects_wrong_final_domain() -> None:
    with pytest.raises(ScheduleContractError, match="final_domain"):
        ScheduleResult(
            request=_external_request(),
            effective_inputs=_inputs(),
            sigmas=(1.0, 0.0),
            final_domain=SigmaDomain.CONTINUOUS_EDM,
        )


def test_result_requires_override_for_changed_effective_inputs() -> None:
    with pytest.raises(ScheduleContractError, match="missing override records"):
        ScheduleResult(
            request=_external_request(),
            effective_inputs=_inputs(width=1025),
            sigmas=(1.0, 0.0),
            final_domain=SigmaDomain.UNIT_FLOW,
        )


def test_result_records_effective_inputs_warnings_and_overrides() -> None:
    override = OverrideRecord(
        field="width",
        requested_value="1024",
        effective_value="1040",
        reason="pad to model multiple",
    )
    result = ScheduleResult(
        request=_external_request(),
        effective_inputs=_inputs(width=1040),
        sigmas=(1.0, 0.0),
        final_domain=SigmaDomain.UNIT_FLOW,
        warnings=("width padded",),
        overrides=(override,),
    )

    assert result.request.requested_inputs.width == 1024
    assert result.effective_inputs.width == 1040
    assert result.warnings == ("width padded",)
    assert result.overrides == (override,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sigmas", [1.0, 0.0]),
        ("warnings", ["warning"]),
        (
            "overrides",
            [
                OverrideRecord(
                    field="width",
                    requested_value="1",
                    effective_value="2",
                    reason="fixture",
                )
            ],
        ),
    ],
)
def test_result_collection_fields_require_tuples(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "request": _external_request(),
        "effective_inputs": _inputs(),
        "sigmas": (1.0, 0.0),
        "final_domain": SigmaDomain.UNIT_FLOW,
    }
    arguments[field] = value

    with pytest.raises(ScheduleContractError, match=f"{field} must be a tuple"):
        ScheduleResult(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sigmas", ("not-numeric",), "numeric"),
        ("warnings", ("",), "non-empty strings"),
        ("overrides", ("not-an-override",), "OverrideRecord"),
    ],
)
def test_result_collection_fields_reject_wrong_element_types(
    field: str,
    value: tuple[object, ...],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "request": _external_request(),
        "effective_inputs": _inputs(),
        "sigmas": (1.0, 0.0),
        "final_domain": SigmaDomain.UNIT_FLOW,
    }
    arguments[field] = value

    with pytest.raises(ScheduleContractError, match=message):
        ScheduleResult(**arguments)  # type: ignore[arg-type]
