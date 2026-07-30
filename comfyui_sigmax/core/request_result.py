"""Immutable request and result structures for schedule construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from comfyui_sigmax.core.schedule_contracts import (
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TransformContract,
    validate_transform_chain,
)


class TerminalPolicy(str, Enum):
    """How an externally constructed schedule handles its terminal value."""

    APPEND_ZERO = "APPEND_ZERO"
    PRESERVE = "PRESERVE"


class EvidenceLevel(str, Enum):
    """Evidence classification attached to schedule provenance."""

    OFFICIAL = "official"
    FRAMEWORK_REFERENCE = "framework_reference"
    COMMUNITY_RECOMMENDED = "community_recommended"
    EXPERIMENTAL = "experimental"
    MODIFIED = "modified"


def _require_nonempty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ScheduleContractError(f"{field_name} must not be empty")


def _require_tuple(field_name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise ScheduleContractError(f"{field_name} must be a tuple")


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleInputs:
    """User-requested or effective scalar inputs relevant to a schedule."""

    steps: int
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        if not _is_positive_integer(self.steps):
            raise ScheduleContractError("steps must be a positive integer")

        dimensions = (self.width, self.height)
        if dimensions == (None, None):
            return
        if not all(_is_positive_integer(value) for value in dimensions):
            raise ScheduleContractError(
                "width and height must be supplied together as positive integers"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseGridSpec:
    """Identity and output domain of a requested base-grid builder."""

    identifier: str
    output_domain: SigmaDomain

    def __post_init__(self) -> None:
        _require_nonempty("base grid identifier", self.identifier)


@dataclass(frozen=True, slots=True, kw_only=True)
class SliceSpec:
    """Requested start/end and denoise slicing parameters."""

    start_step: int = 0
    end_step: int | None = None
    denoise: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.start_step, int)
            or isinstance(self.start_step, bool)
            or self.start_step < 0
        ):
            raise ScheduleContractError("start_step must be a non-negative integer")
        if self.end_step is not None and (
            not isinstance(self.end_step, int)
            or isinstance(self.end_step, bool)
            or self.end_step <= self.start_step
        ):
            raise ScheduleContractError("end_step must be an integer greater than start_step")
        if (
            isinstance(self.denoise, bool)
            or not isinstance(self.denoise, (int, float))
            or not math.isfinite(float(self.denoise))
            or not 0.0 <= float(self.denoise) <= 1.0
        ):
            raise ScheduleContractError("denoise must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class Provenance:
    """Source and version information for a schedule request."""

    engine_version: str
    evidence: EvidenceLevel
    source: str
    source_revision: str | None = None
    profile_id: str | None = None
    profile_version: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty("engine_version", self.engine_version)
        _require_nonempty("source", self.source)
        if (self.profile_id is None) != (self.profile_version is None):
            raise ScheduleContractError("profile_id and profile_version must be supplied together")
        if self.profile_id is not None:
            _require_nonempty("profile_id", self.profile_id)
            _require_nonempty("profile_version", self.profile_version or "")
        if self.source_revision is not None:
            _require_nonempty("source_revision", self.source_revision)


@dataclass(frozen=True, slots=True, kw_only=True)
class OverrideRecord:
    """One explicit requested-to-effective value change."""

    field: str
    requested_value: str
    effective_value: str
    reason: str

    def __post_init__(self) -> None:
        for field_name in ("field", "requested_value", "effective_value", "reason"):
            _require_nonempty(field_name, getattr(self, field_name))


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleRequest:
    """A structurally complete request before numerical schedule construction."""

    ownership: ScheduleOwnership
    requested_inputs: ScheduleInputs
    sigma_domain: SigmaDomain
    provenance: Provenance
    base_grid: BaseGridSpec | None = None
    transforms: tuple[TransformContract, ...] = ()
    terminal_policy: TerminalPolicy | None = None
    slicing: SliceSpec | None = None
    overrides: tuple[OverrideRecord, ...] = ()

    def __post_init__(self) -> None:
        _require_tuple("transforms", self.transforms)
        _require_tuple("overrides", self.overrides)
        if not all(isinstance(item, TransformContract) for item in self.transforms):
            raise ScheduleContractError("transforms must contain TransformContract values")
        if not all(isinstance(item, OverrideRecord) for item in self.overrides):
            raise ScheduleContractError("overrides must contain OverrideRecord values")

        validate_transform_chain(self.ownership, self.sigma_domain, self.transforms)

        if self.ownership is ScheduleOwnership.EXTERNAL_SIGMAS:
            if self.base_grid is None or self.terminal_policy is None or self.slicing is None:
                raise ScheduleContractError(
                    "EXTERNAL_SIGMAS requires base_grid, terminal_policy, and slicing"
                )
            if self.base_grid.output_domain is not self.sigma_domain:
                raise ScheduleContractError(
                    "base grid output domain must equal the request sigma_domain"
                )
            return

        if any(value is not None for value in (self.base_grid, self.terminal_policy, self.slicing)):
            raise ScheduleContractError(
                f"{self.ownership.value} cannot define external construction fields"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleResult:
    """An immutable structural result produced from a validated request."""

    request: ScheduleRequest
    effective_inputs: ScheduleInputs
    sigmas: tuple[float, ...]
    final_domain: SigmaDomain
    warnings: tuple[str, ...] = ()
    overrides: tuple[OverrideRecord, ...] = ()

    def __post_init__(self) -> None:
        _require_tuple("sigmas", self.sigmas)
        _require_tuple("warnings", self.warnings)
        _require_tuple("overrides", self.overrides)
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in self.sigmas
        ):
            raise ScheduleContractError("sigmas must contain numeric values")
        if not all(isinstance(warning, str) and warning.strip() for warning in self.warnings):
            raise ScheduleContractError("warnings must contain non-empty strings")
        if not all(isinstance(item, OverrideRecord) for item in self.overrides):
            raise ScheduleContractError("overrides must contain OverrideRecord values")

        expected_domain = validate_transform_chain(
            self.request.ownership,
            self.request.sigma_domain,
            self.request.transforms,
        )
        if self.final_domain is not expected_domain:
            raise ScheduleContractError(
                f"final_domain must be {expected_domain.value} for the validated transform chain"
            )

        changed_fields = {
            field_name
            for field_name in ("steps", "width", "height")
            if getattr(self.request.requested_inputs, field_name)
            != getattr(self.effective_inputs, field_name)
        }
        recorded_fields = {
            override.field for override in (*self.request.overrides, *self.overrides)
        }
        missing_overrides = changed_fields - recorded_fields
        if missing_overrides:
            raise ScheduleContractError(
                f"missing override records for effective fields: {sorted(missing_overrides)}"
            )
