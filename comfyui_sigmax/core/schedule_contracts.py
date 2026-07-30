"""Framework-independent schedule ownership and domain contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class ScheduleContractError(ValueError):
    """Raised when a schedule cannot be constructed without ambiguity."""


class ScheduleOwnership(str, Enum):
    """The single component that owns schedule semantics for one execution."""

    MODEL_NATIVE = "MODEL_NATIVE"
    EXTERNAL_SIGMAS = "EXTERNAL_SIGMAS"
    MODEL_PATCH = "MODEL_PATCH"


class SigmaDomain(str, Enum):
    """Supported interpretations of values carried by a schedule."""

    UNIT_FLOW = "UNIT_FLOW"
    MODEL_NATIVE = "MODEL_NATIVE"
    CONTINUOUS_EDM = "CONTINUOUS_EDM"
    DISCRETE_TRAINING_INDEX = "DISCRETE_TRAINING_INDEX"


class TransformStage(str, Enum):
    """Canonical order for transforms applied to externally owned sigmas."""

    PRIMARY_TIME_SHIFT = "PRIMARY_TIME_SHIFT"
    OPTIONAL_SPACING = "OPTIONAL_SPACING"
    TERMINAL = "TERMINAL"
    SLICE = "SLICE"


_TRANSFORM_STAGE_ORDER = {stage: index for index, stage in enumerate(TransformStage)}
_SINGLETON_STAGES = {
    TransformStage.PRIMARY_TIME_SHIFT,
    TransformStage.OPTIONAL_SPACING,
}


@dataclass(frozen=True, slots=True)
class TransformContract:
    """A named schedule transform with explicit domain boundaries."""

    name: str
    stage: TransformStage
    input_domain: SigmaDomain
    output_domain: SigmaDomain

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ScheduleContractError("transform name must not be empty")


def require_single_ownership(*modes: ScheduleOwnership) -> ScheduleOwnership:
    """Return the selected ownership mode, rejecting missing or combined modes."""

    if len(modes) != 1:
        raise ScheduleContractError("exactly one schedule ownership mode is required")
    mode = modes[0]
    if not isinstance(mode, ScheduleOwnership):
        raise ScheduleContractError("unsupported schedule ownership mode")
    return mode


def validate_transform_chain(
    ownership: ScheduleOwnership,
    initial_domain: SigmaDomain,
    transforms: Iterable[TransformContract],
) -> SigmaDomain:
    """Validate ownership and transform compatibility before schedule execution."""

    chain = tuple(transforms)
    if ownership in {
        ScheduleOwnership.MODEL_NATIVE,
        ScheduleOwnership.MODEL_PATCH,
    }:
        if initial_domain is not SigmaDomain.MODEL_NATIVE:
            raise ScheduleContractError(
                f"{ownership.value} ownership requires the MODEL_NATIVE sigma domain"
            )
        if chain:
            raise ScheduleContractError(
                f"{ownership.value} ownership cannot apply an external transform chain; "
                "this would risk double shifting"
            )
        return initial_domain

    if ownership is not ScheduleOwnership.EXTERNAL_SIGMAS:
        raise ScheduleContractError("unsupported schedule ownership mode")
    if initial_domain is SigmaDomain.MODEL_NATIVE:
        raise ScheduleContractError(
            "EXTERNAL_SIGMAS ownership cannot consume the opaque MODEL_NATIVE sigma domain"
        )

    current_domain: SigmaDomain = initial_domain
    previous_stage_index = -1
    stage_counts: dict[TransformStage, int] = {}

    for transform in chain:
        stage_index = _TRANSFORM_STAGE_ORDER[transform.stage]
        if stage_index < previous_stage_index:
            raise ScheduleContractError(
                f"transform '{transform.name}' violates canonical stage order"
            )
        previous_stage_index = stage_index

        stage_counts[transform.stage] = stage_counts.get(transform.stage, 0) + 1
        if transform.stage in _SINGLETON_STAGES and stage_counts[transform.stage] > 1:
            raise ScheduleContractError(f"at most one {transform.stage.value} transform is allowed")

        if transform.input_domain is not current_domain:
            raise ScheduleContractError(
                f"transform '{transform.name}' expects {transform.input_domain.value}, "
                f"received {current_domain.value}"
            )
        current_domain = transform.output_domain

    return current_domain
