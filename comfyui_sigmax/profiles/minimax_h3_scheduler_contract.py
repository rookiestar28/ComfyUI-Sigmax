"""Pure qualification contracts for the MiniMax H3 ten-scheduler selector.

This private module records which schedules Sigmax owns and which schedules a pinned ComfyUI
host must construct.  It never imports ComfyUI, copies host scheduler formulas, registers a
runtime scheduler, or applies another shift to model-native MiniMax H3 sigmas.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Final, Literal, NoReturn

from comfyui_sigmax.core import ScheduleContractError, SigmaDomain, float_to_ieee_hex
from comfyui_sigmax.profiles.minimax_h3 import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_DIFFUSERS_REVISION,
    MINIMAX_H3_MAX_STEPS,
    MINIMAX_H3_VIDEO_SHIFT,
)
from comfyui_sigmax.profiles.minimax_h3_acceleration import (
    MINIMAX_H3_ACCELERATION_RECIPES,
)

MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_ID: Final = "sigmax.minimax-h3-ten-scheduler-contract/1"
MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_VERSION: Final = "1"
MINIMAX_H3_SCHEDULER_RESULT_SCHEMA_ID: Final = "sigmax.minimax-h3-scheduler-result/1"
MINIMAX_H3_SCHEDULER_CHOICES: Final = (
    "h3_endpoint",
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
)
MINIMAX_H3_DEFAULT_SCHEDULER: Final = MINIMAX_H3_SCHEDULER_CHOICES[0]
MINIMAX_H3_NATIVE_SCHEDULERS: Final = MINIMAX_H3_SCHEDULER_CHOICES[1:]

_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
FloatPrecision = Literal["float32", "float64"]


class MiniMaxH3SchedulerOwner(str, Enum):
    """Component that owns schedule construction."""

    SIGMAX_PURE = "sigmax_pure"
    COMFYUI_NATIVE = "comfyui_native"


class MiniMaxH3ModelPolicy(str, Enum):
    """Whether a connected model is meaningful for the selected schedule owner."""

    FORBIDDEN = "forbidden"
    REQUIRED = "required"


class MiniMaxH3CountPolicy(str, Enum):
    """How a raw handler result becomes the terminal-inclusive scheduler vector."""

    EXACT_ENDPOINT = "exact_steps_plus_one"
    BASIC_SCHEDULER_TAIL = "raw_then_basic_scheduler_tail"


class MiniMaxH3SchedulerHostRole(str, Enum):
    """Role of one exact ComfyUI source pin."""

    ACCEPTED_KNOWN_GOOD = "accepted_known_good"
    SUPPLIED_CURRENT = "supplied_current"


class MiniMaxH3SchedulerReasonCode(str, Enum):
    """Stable fail-closed qualification and result-validation reasons."""

    UNSUPPORTED_SCHEDULER = "UNSUPPORTED_SCHEDULER"
    INVALID_STEPS = "INVALID_STEPS"
    MODEL_FORBIDDEN = "MODEL_FORBIDDEN"
    MODEL_REQUIRED = "MODEL_REQUIRED"
    MODEL_FAMILY_MISMATCH = "MODEL_FAMILY_MISMATCH"
    MODEL_TASK_MISMATCH = "MODEL_TASK_MISMATCH"
    MODEL_SAMPLING_NOT_AV = "MODEL_SAMPLING_NOT_AV"
    MODEL_NOT_ALREADY_SHIFTED = "MODEL_NOT_ALREADY_SHIFTED"
    SHIFT_MISMATCH = "SHIFT_MISMATCH"
    UNSUPPORTED_HOST = "UNSUPPORTED_HOST"
    MISSING_HANDLER = "MISSING_HANDLER"
    UNSUPPORTED_RECIPE = "UNSUPPORTED_RECIPE"
    UNSUPPORTED_RECIPE_NFE = "UNSUPPORTED_RECIPE_NFE"
    RESULT_COUNT_INVALID = "RESULT_COUNT_INVALID"
    RESULT_NON_FINITE = "RESULT_NON_FINITE"
    RESULT_NOT_MONOTONIC = "RESULT_NOT_MONOTONIC"
    RESULT_DOMAIN_INVALID = "RESULT_DOMAIN_INVALID"
    RESULT_TERMINAL_INVALID = "RESULT_TERMINAL_INVALID"
    RESULT_DTYPE_INVALID = "RESULT_DTYPE_INVALID"
    RESULT_QUALIFICATION_INVALID = "RESULT_QUALIFICATION_INVALID"
    RESULT_SLICE_INVALID = "RESULT_SLICE_INVALID"


class MiniMaxH3SchedulerContractError(ScheduleContractError):
    """Contract error with a stable machine-readable reason code."""

    def __init__(self, reason_code: MiniMaxH3SchedulerReasonCode, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code.value}: {message}")


def _require_nonempty_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScheduleContractError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3SchedulerContract:
    """Immutable construction and input contract for one selector value."""

    name: str
    ordinal: int
    owner: MiniMaxH3SchedulerOwner
    model_policy: MiniMaxH3ModelPolicy
    handler_name: str | None
    count_policy: MiniMaxH3CountPolicy
    minimum_steps: int
    allowed_dtypes: tuple[FloatPrecision, ...]
    additional_shift_allowed: bool = False
    basic_scheduler_tail: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_text("scheduler name", self.name)
        if self.name not in MINIMAX_H3_SCHEDULER_CHOICES:
            raise ScheduleContractError("scheduler name is not in the frozen choice set")
        if (
            not isinstance(self.ordinal, int)
            or isinstance(self.ordinal, bool)
            or not 1 <= self.ordinal <= len(MINIMAX_H3_SCHEDULER_CHOICES)
        ):
            raise ScheduleContractError("scheduler ordinal is outside the frozen choice set")
        if not isinstance(self.owner, MiniMaxH3SchedulerOwner):
            raise ScheduleContractError("scheduler owner is unsupported")
        if not isinstance(self.model_policy, MiniMaxH3ModelPolicy):
            raise ScheduleContractError("scheduler MODEL policy is unsupported")
        if not isinstance(self.count_policy, MiniMaxH3CountPolicy):
            raise ScheduleContractError("scheduler count policy is unsupported")
        if (
            not isinstance(self.minimum_steps, int)
            or isinstance(self.minimum_steps, bool)
            or self.minimum_steps <= 0
        ):
            raise ScheduleContractError("scheduler minimum_steps must be positive")
        if not self.allowed_dtypes or self.allowed_dtypes != tuple(
            dict.fromkeys(self.allowed_dtypes)
        ):
            raise ScheduleContractError("scheduler allowed_dtypes must be non-empty and unique")
        if any(dtype not in {"float32", "float64"} for dtype in self.allowed_dtypes):
            raise ScheduleContractError("scheduler allowed_dtypes contains an unsupported dtype")
        if not isinstance(self.additional_shift_allowed, bool) or not isinstance(
            self.basic_scheduler_tail, bool
        ):
            raise ScheduleContractError("scheduler boolean policies must be explicit")

        if self.owner is MiniMaxH3SchedulerOwner.SIGMAX_PURE:
            if self.model_policy is not MiniMaxH3ModelPolicy.FORBIDDEN:
                raise ScheduleContractError("Sigmax-pure scheduler requires MODEL forbidden policy")
            if self.handler_name is not None:
                raise ScheduleContractError("Sigmax-pure scheduler handler must be absent")
            if self.count_policy is not MiniMaxH3CountPolicy.EXACT_ENDPOINT:
                raise ScheduleContractError("Sigmax-pure scheduler count policy must be exact")
            if self.basic_scheduler_tail:
                raise ScheduleContractError("Sigmax-pure scheduler cannot use BasicScheduler tail")
        else:
            if self.model_policy is not MiniMaxH3ModelPolicy.REQUIRED:
                raise ScheduleContractError("ComfyUI-native scheduler requires MODEL input")
            if self.handler_name != self.name:
                raise ScheduleContractError("ComfyUI-native scheduler handler must match its name")
            if self.count_policy is not MiniMaxH3CountPolicy.BASIC_SCHEDULER_TAIL:
                raise ScheduleContractError("ComfyUI-native scheduler count policy must use tail")
            if not self.basic_scheduler_tail:
                raise ScheduleContractError("ComfyUI-native scheduler requires BasicScheduler tail")
            if self.additional_shift_allowed:
                raise ScheduleContractError("ComfyUI-native scheduler cannot add a second shift")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3SchedulerHost:
    """Exact ComfyUI source pin used only to qualify delegation."""

    version: str
    revision: str
    role: MiniMaxH3SchedulerHostRole
    scheduler_names: tuple[str, ...]
    license_id: str
    delegation_only: bool
    url: str
    source_locators: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_text("host version", self.version)
        if not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("host revision must be a pinned lowercase commit")
        if not isinstance(self.role, MiniMaxH3SchedulerHostRole):
            raise ScheduleContractError("host role is unsupported")
        if self.scheduler_names != MINIMAX_H3_NATIVE_SCHEDULERS:
            raise ScheduleContractError("host scheduler_names must match the native choice set")
        if self.license_id != "GPL-3.0-only":
            raise ScheduleContractError(
                "host scheduler source must retain its GPL license boundary"
            )
        if self.delegation_only is not True:
            raise ScheduleContractError("host source is delegation-only")
        if not self.url.startswith("https://github.com/Comfy-Org/ComfyUI"):
            raise ScheduleContractError("host URL must identify the official ComfyUI repository")
        if self.source_locators != tuple(sorted(set(self.source_locators))):
            raise ScheduleContractError("host source_locators must be sorted and unique")
        if not {"comfy/model_sampling.py", "comfy/samplers.py"}.issubset(self.source_locators):
            raise ScheduleContractError("host source locators omit scheduler/model sampling owners")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3SchedulerRecipe:
    """Read-only projection of one source-qualified Turbo recipe."""

    recipe_id: str
    task: str
    allowed_nfe: tuple[int, ...]
    video_shift: float
    audio_shift: float

    def __post_init__(self) -> None:
        _require_nonempty_text("recipe_id", self.recipe_id)
        if self.task not in {"fl2va", "ref2va"}:
            raise ScheduleContractError("recipe task is unsupported")
        if not self.allowed_nfe or self.allowed_nfe != tuple(sorted(set(self.allowed_nfe))):
            raise ScheduleContractError("recipe allowed_nfe must be sorted and unique")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.allowed_nfe
        ):
            raise ScheduleContractError("recipe allowed_nfe must contain positive integers")
        if not math.isfinite(self.video_shift) or self.video_shift <= 0.0:
            raise ScheduleContractError("recipe video_shift must be finite and positive")
        if not math.isfinite(self.audio_shift) or self.audio_shift <= 0.0:
            raise ScheduleContractError("recipe audio_shift must be finite and positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3ModelSamplingEvidence:
    """Adapter-supplied observations needed before native scheduler delegation."""

    family_id: str
    task: str
    is_model_sampling_av: bool
    video_shift: float
    audio_shift: float
    already_shifted: bool

    def __post_init__(self) -> None:
        _require_nonempty_text("model family_id", self.family_id)
        if self.task not in {"fl2va", "ref2va"}:
            raise ScheduleContractError("model task must be fl2va or ref2va")
        if not isinstance(self.is_model_sampling_av, bool):
            raise ScheduleContractError("is_model_sampling_av must be boolean")
        if not isinstance(self.already_shifted, bool):
            raise ScheduleContractError("already_shifted must be boolean")
        for field_name, value in (
            ("model video_shift", self.video_shift),
            ("model audio_shift", self.audio_shift),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ScheduleContractError(f"{field_name} must be finite and positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3QualifiedSchedulerRequest:
    """Validated dispatch decision without a runtime handler object."""

    scheduler: str
    steps: int
    owner: MiniMaxH3SchedulerOwner
    handler_name: str | None
    additional_shift_allowed: bool
    expected_video_shift: float
    expected_audio_shift: float
    recipe_id: str | None
    recipe_task: str | None
    model_family_id: str | None
    model_task: str | None
    host_revision: str | None
    contract_fingerprint: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3SchedulerResultValidation:
    """Validated raw, BasicScheduler-tail, and Sigmax-sliced counts."""

    scheduler: str
    requested_steps: int
    dtype: FloatPrecision
    contract_fingerprint: str
    host_revision: str | None
    model_task: str | None
    recipe_id: str | None
    raw_count: int
    basic_scheduler_count: int
    basic_scheduler_sigmas: tuple[float, ...]
    output_sigmas: tuple[float, ...]
    output_transitions: int
    start_step: int
    end_step: int
    output_fingerprint: str


def _scheduler_contract(
    scheduler: object,
) -> MiniMaxH3SchedulerContract:
    if not isinstance(scheduler, str):
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.UNSUPPORTED_SCHEDULER,
            "scheduler must be one of the frozen MiniMax H3 choices",
        )
    contract = _CONTRACTS_BY_NAME.get(scheduler)
    if contract is None:
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.UNSUPPORTED_SCHEDULER,
            "scheduler is not in the frozen MiniMax H3 choice set",
        )
    return contract


def _require_steps(contract: MiniMaxH3SchedulerContract, steps: object) -> int:
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or not contract.minimum_steps <= steps <= MINIMAX_H3_MAX_STEPS
    ):
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.INVALID_STEPS,
            f"{contract.name} steps must be between {contract.minimum_steps} and "
            f"{MINIMAX_H3_MAX_STEPS}",
        )
    return steps


def _result_error(reason: MiniMaxH3SchedulerReasonCode, message: str) -> NoReturn:
    raise MiniMaxH3SchedulerContractError(reason, message)


def _validate_raw_sigmas(raw_sigmas: object) -> tuple[float, ...]:
    if not isinstance(raw_sigmas, tuple) or len(raw_sigmas) < 2:
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_COUNT_INVALID,
            "raw scheduler result must contain at least one transition and its terminal",
        )
    values: list[float] = []
    for value in raw_sigmas:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _result_error(
                MiniMaxH3SchedulerReasonCode.RESULT_NON_FINITE,
                "raw scheduler result must contain numeric values",
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            _result_error(
                MiniMaxH3SchedulerReasonCode.RESULT_NON_FINITE,
                "raw scheduler result must contain finite values",
            )
        values.append(numeric)
    if any(value < 0.0 or value > 1.0 for value in values):
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_DOMAIN_INVALID,
            f"raw scheduler result must remain in {SigmaDomain.UNIT_FLOW.value}",
        )
    if any(current < following for current, following in pairwise(values)):
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_NOT_MONOTONIC,
            "raw scheduler result must be monotonically non-increasing",
        )
    if values[-1] != 0.0:
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_TERMINAL_INVALID,
            "raw scheduler result must preserve an exact terminal zero",
        )
    return tuple(values)


def _slice_result(
    values: tuple[float, ...], *, start_step: object, end_step: object
) -> tuple[tuple[float, ...], int, int]:
    available_steps = len(values) - 1
    if (
        not isinstance(start_step, int)
        or isinstance(start_step, bool)
        or not 0 <= start_step < available_steps
    ):
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_SLICE_INVALID,
            "start_step is outside the normalized scheduler result",
        )
    effective_end = available_steps if end_step is None else end_step
    if (
        not isinstance(effective_end, int)
        or isinstance(effective_end, bool)
        or not start_step < effective_end <= available_steps
    ):
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_SLICE_INVALID,
            "end_step is outside the normalized scheduler result",
        )
    return values[start_step : effective_end + 1], start_step, effective_end


def _schedule_fingerprint(
    *,
    qualification: MiniMaxH3QualifiedSchedulerRequest,
    contract: MiniMaxH3SchedulerContract,
    precision: FloatPrecision,
    raw_values: tuple[float, ...],
    normalized: tuple[float, ...],
    output: tuple[float, ...],
    start_step: int,
    end_step: int,
) -> str:
    source_revision = (
        MINIMAX_H3_DIFFUSERS_REVISION
        if contract.owner is MiniMaxH3SchedulerOwner.SIGMAX_PURE
        else qualification.host_revision
    )
    projection = {
        "additional_shift_allowed": qualification.additional_shift_allowed,
        "audio_shift": float_to_ieee_hex(qualification.expected_audio_shift, "float64"),
        "basic_scheduler_count": len(normalized),
        "basic_scheduler_tail": contract.basic_scheduler_tail,
        "contract_fingerprint": qualification.contract_fingerprint,
        "count_policy": contract.count_policy.value,
        "domain": SigmaDomain.UNIT_FLOW.value,
        "end_step": end_step,
        "handler_name": qualification.handler_name,
        "host_revision": qualification.host_revision,
        "model_family_id": qualification.model_family_id,
        "model_task": qualification.model_task,
        "normalized_values": [float_to_ieee_hex(value, precision) for value in normalized],
        "output_transitions": len(output) - 1,
        "output_values": [float_to_ieee_hex(value, precision) for value in output],
        "owner": qualification.owner.value,
        "dtype": precision,
        "raw_count": len(raw_values),
        "raw_values": [float_to_ieee_hex(value, precision) for value in raw_values],
        "recipe_id": qualification.recipe_id,
        "recipe_task": qualification.recipe_task,
        "requested_steps": qualification.steps,
        "scheduler": qualification.scheduler,
        "contract_schema_id": MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_ID,
        "schema_version": MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_VERSION,
        "schema_id": MINIMAX_H3_SCHEDULER_RESULT_SCHEMA_ID,
        "source_revision": source_revision,
        "start_step": start_step,
        "terminal_policy": "require_exact_zero_preserve",
        "terminal_value": float_to_ieee_hex(raw_values[-1], precision),
        "video_shift": float_to_ieee_hex(qualification.expected_video_shift, "float64"),
    }
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def qualify_minimax_h3_scheduler_request(
    *,
    scheduler: object,
    steps: object,
    model_sampling: MiniMaxH3ModelSamplingEvidence | None = None,
    recipe_id: str | None = None,
    host_revision: str | None = None,
    available_handlers: tuple[str, ...] | None = None,
) -> MiniMaxH3QualifiedSchedulerRequest:
    """Fail closed before a future adapter selects a pure or host-native implementation."""

    contract = _scheduler_contract(scheduler)
    selected_steps = _require_steps(contract, steps)
    recipe = None
    if recipe_id is not None:
        recipe = _RECIPES_BY_ID.get(recipe_id)
        if recipe is None:
            raise MiniMaxH3SchedulerContractError(
                MiniMaxH3SchedulerReasonCode.UNSUPPORTED_RECIPE,
                "recipe is not in the source-qualified MiniMax H3 set",
            )
        if selected_steps not in recipe.allowed_nfe:
            raise MiniMaxH3SchedulerContractError(
                MiniMaxH3SchedulerReasonCode.UNSUPPORTED_RECIPE_NFE,
                "requested steps are outside the selected recipe",
            )

    expected_video_shift = MINIMAX_H3_VIDEO_SHIFT if recipe is None else recipe.video_shift
    expected_audio_shift = MINIMAX_H3_AUDIO_SHIFT if recipe is None else recipe.audio_shift

    if contract.model_policy is MiniMaxH3ModelPolicy.FORBIDDEN:
        if model_sampling is not None:
            raise MiniMaxH3SchedulerContractError(
                MiniMaxH3SchedulerReasonCode.MODEL_FORBIDDEN,
                "MODEL input is inert and therefore forbidden for h3_endpoint",
            )
        return MiniMaxH3QualifiedSchedulerRequest(
            scheduler=contract.name,
            steps=selected_steps,
            owner=contract.owner,
            handler_name=None,
            additional_shift_allowed=contract.additional_shift_allowed,
            expected_video_shift=expected_video_shift,
            expected_audio_shift=expected_audio_shift,
            recipe_id=recipe_id,
            recipe_task=None if recipe is None else recipe.task,
            model_family_id=None,
            model_task=None,
            host_revision=None,
            contract_fingerprint=minimax_h3_scheduler_contract_fingerprint(),
        )

    if model_sampling is None:
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.MODEL_REQUIRED,
            "MODEL sampling evidence is required for native scheduler delegation",
        )
    host = _HOSTS_BY_REVISION.get(host_revision) if isinstance(host_revision, str) else None
    if host is None:
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.UNSUPPORTED_HOST,
            "native scheduler delegation requires an exact supported ComfyUI revision",
        )
    if available_handlers is None or contract.handler_name not in available_handlers:
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.MISSING_HANDLER,
            "the selected native scheduler handler is unavailable",
        )
    if model_sampling.family_id != "minimax_h3":
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.MODEL_FAMILY_MISMATCH,
            "MODEL sampling evidence does not identify MiniMax H3",
        )
    if recipe is not None and model_sampling.task != recipe.task:
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.MODEL_TASK_MISMATCH,
            "MODEL task differs from the selected Turbo recipe",
        )
    if not model_sampling.is_model_sampling_av:
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.MODEL_SAMPLING_NOT_AV,
            "MODEL sampling evidence is not ModelSamplingAV-compatible",
        )
    if not model_sampling.already_shifted:
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.MODEL_NOT_ALREADY_SHIFTED,
            "MODEL sampling evidence does not prove model-native H3 shifting",
        )
    if not math.isclose(model_sampling.video_shift, expected_video_shift, abs_tol=1e-9) or not (
        math.isclose(model_sampling.audio_shift, expected_audio_shift, abs_tol=1e-9)
    ):
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.SHIFT_MISMATCH,
            "MODEL sampling shifts differ from the selected Base/Turbo recipe",
        )
    return MiniMaxH3QualifiedSchedulerRequest(
        scheduler=contract.name,
        steps=selected_steps,
        owner=contract.owner,
        handler_name=contract.handler_name,
        additional_shift_allowed=False,
        expected_video_shift=expected_video_shift,
        expected_audio_shift=expected_audio_shift,
        recipe_id=recipe_id,
        recipe_task=None if recipe is None else recipe.task,
        model_family_id=model_sampling.family_id,
        model_task=model_sampling.task,
        host_revision=host.revision,
        contract_fingerprint=minimax_h3_scheduler_contract_fingerprint(),
    )


def _validate_qualification(
    qualification: object,
) -> tuple[MiniMaxH3QualifiedSchedulerRequest, MiniMaxH3SchedulerContract]:
    if not isinstance(qualification, MiniMaxH3QualifiedSchedulerRequest):
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_QUALIFICATION_INVALID,
            "result validation requires a qualified scheduler request",
        )
    contract = _scheduler_contract(qualification.scheduler)
    _require_steps(contract, qualification.steps)
    if qualification.contract_fingerprint != minimax_h3_scheduler_contract_fingerprint():
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_QUALIFICATION_INVALID,
            "qualified request belongs to a different scheduler contract",
        )
    if (
        qualification.owner is not contract.owner
        or qualification.handler_name != contract.handler_name
        or qualification.additional_shift_allowed != contract.additional_shift_allowed
    ):
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_QUALIFICATION_INVALID,
            "qualified request ownership or handler identity drifted",
        )
    recipe = None
    if qualification.recipe_id is not None:
        recipe = _RECIPES_BY_ID.get(qualification.recipe_id)
        if recipe is None or qualification.recipe_task != recipe.task:
            _result_error(
                MiniMaxH3SchedulerReasonCode.RESULT_QUALIFICATION_INVALID,
                "qualified request recipe identity drifted",
            )
        if (
            qualification.steps not in recipe.allowed_nfe
            or qualification.expected_video_shift != recipe.video_shift
            or qualification.expected_audio_shift != recipe.audio_shift
        ):
            _result_error(
                MiniMaxH3SchedulerReasonCode.RESULT_QUALIFICATION_INVALID,
                "qualified request recipe steps or shifts drifted",
            )
    elif (
        qualification.recipe_task is not None
        or qualification.expected_video_shift != MINIMAX_H3_VIDEO_SHIFT
        or qualification.expected_audio_shift != MINIMAX_H3_AUDIO_SHIFT
    ):
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_QUALIFICATION_INVALID,
            "qualified Base request identity drifted",
        )
    if contract.owner is MiniMaxH3SchedulerOwner.SIGMAX_PURE:
        if (
            qualification.host_revision is not None
            or qualification.model_family_id is not None
            or qualification.model_task is not None
        ):
            _result_error(
                MiniMaxH3SchedulerReasonCode.RESULT_QUALIFICATION_INVALID,
                "pure qualification cannot carry host or MODEL identity",
            )
    elif (
        qualification.host_revision not in _HOSTS_BY_REVISION
        or qualification.model_family_id != "minimax_h3"
        or qualification.model_task not in {"fl2va", "ref2va"}
        or (recipe is not None and qualification.model_task != recipe.task)
    ):
        _result_error(
            MiniMaxH3SchedulerReasonCode.RESULT_QUALIFICATION_INVALID,
            "native qualification lacks exact host or MODEL task identity",
        )
    return qualification, contract


def validate_minimax_h3_scheduler_result(
    *,
    qualification: object,
    raw_sigmas: object,
    dtype: object,
    start_step: object = 0,
    end_step: object = None,
) -> MiniMaxH3SchedulerResultValidation:
    """Validate host output, then mirror BasicScheduler tail and Sigmax step slicing."""

    selected, contract = _validate_qualification(qualification)
    steps = selected.steps
    if dtype not in contract.allowed_dtypes:
        raise MiniMaxH3SchedulerContractError(
            MiniMaxH3SchedulerReasonCode.RESULT_DTYPE_INVALID,
            f"{contract.name} result dtype is outside its declared policy",
        )
    precision: FloatPrecision = dtype
    raw_values = _validate_raw_sigmas(raw_sigmas)
    if contract.count_policy is MiniMaxH3CountPolicy.EXACT_ENDPOINT:
        if len(raw_values) != steps + 1:
            raise MiniMaxH3SchedulerContractError(
                MiniMaxH3SchedulerReasonCode.RESULT_COUNT_INVALID,
                "h3_endpoint result must contain exactly requested_steps + 1 values",
            )
        normalized = raw_values
    else:
        # IMPORTANT: this mirrors ComfyUI BasicScheduler after its native handler returns.
        # Some handlers return more or fewer than requested_steps + 1; preserve that behavior.
        normalized = raw_values[-(steps + 1) :]
    output, effective_start, effective_end = _slice_result(
        normalized, start_step=start_step, end_step=end_step
    )
    return MiniMaxH3SchedulerResultValidation(
        scheduler=contract.name,
        requested_steps=steps,
        dtype=precision,
        contract_fingerprint=selected.contract_fingerprint,
        host_revision=selected.host_revision,
        model_task=selected.model_task,
        recipe_id=selected.recipe_id,
        raw_count=len(raw_values),
        basic_scheduler_count=len(normalized),
        basic_scheduler_sigmas=normalized,
        output_sigmas=output,
        output_transitions=len(output) - 1,
        start_step=effective_start,
        end_step=effective_end,
        output_fingerprint=_schedule_fingerprint(
            qualification=selected,
            contract=contract,
            precision=precision,
            raw_values=raw_values,
            normalized=normalized,
            output=output,
            start_step=effective_start,
            end_step=effective_end,
        ),
    )


def serialize_minimax_h3_scheduler_contract() -> dict[str, object]:
    """Return a deterministic public-safe projection for review and future adapter tests."""

    return {
        "schema_id": MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_ID,
        "schema_version": MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_VERSION,
        "choices": list(MINIMAX_H3_SCHEDULER_CHOICES),
        "default": MINIMAX_H3_DEFAULT_SCHEDULER,
        "qualification": {
            "family_id": "minimax_h3",
            "model_tasks": ["fl2va", "ref2va"],
            "native_model_required": True,
            "native_recipe_task_match": True,
            "native_requires_already_shifted": True,
            "pure_model_forbidden": True,
        },
        "pure_source": {
            "license_id": "Apache-2.0",
            "revision": MINIMAX_H3_DIFFUSERS_REVISION,
            "source_locators": [
                "src/diffusers/schedulers/scheduling_minimax_h3.py",
            ],
            "url": "https://github.com/huggingface/diffusers",
        },
        "result_identity": {
            "fields": [
                "additional_shift_allowed",
                "audio_shift",
                "basic_scheduler_count",
                "basic_scheduler_tail",
                "contract_fingerprint",
                "count_policy",
                "domain",
                "dtype",
                "end_step",
                "handler_name",
                "host_revision",
                "model_family_id",
                "model_task",
                "normalized_values",
                "output_transitions",
                "output_values",
                "owner",
                "raw_count",
                "raw_values",
                "recipe_id",
                "recipe_task",
                "requested_steps",
                "scheduler",
                "source_revision",
                "start_step",
                "terminal_policy",
                "terminal_value",
                "video_shift",
            ],
            "result_schema_id": MINIMAX_H3_SCHEDULER_RESULT_SCHEMA_ID,
            "terminal_policy": "require_exact_zero_preserve",
        },
        "contracts": [
            {
                "additional_shift_allowed": item.additional_shift_allowed,
                "allowed_dtypes": list(item.allowed_dtypes),
                "basic_scheduler_tail": item.basic_scheduler_tail,
                "count_policy": item.count_policy.value,
                "handler_name": item.handler_name,
                "minimum_steps": item.minimum_steps,
                "model_policy": item.model_policy.value,
                "name": item.name,
                "ordinal": item.ordinal,
                "owner": item.owner.value,
            }
            for item in MINIMAX_H3_SCHEDULER_CONTRACTS
        ],
        "hosts": [
            {
                "delegation_only": host.delegation_only,
                "license_id": host.license_id,
                "revision": host.revision,
                "role": host.role.value,
                "scheduler_names": list(host.scheduler_names),
                "source_locators": list(host.source_locators),
                "url": host.url,
                "version": host.version,
            }
            for host in MINIMAX_H3_SCHEDULER_HOSTS
        ],
        "recipes": [
            {
                "allowed_nfe": list(recipe.allowed_nfe),
                "audio_shift": recipe.audio_shift,
                "recipe_id": recipe.recipe_id,
                "task": recipe.task,
                "video_shift": recipe.video_shift,
            }
            for recipe in MINIMAX_H3_SCHEDULER_RECIPES
        ],
        "native_formula_copied": False,
        "runtime_registered": False,
    }


def minimax_h3_scheduler_contract_fingerprint() -> str:
    """Fingerprint the complete frozen contract projection."""

    encoded = json.dumps(
        serialize_minimax_h3_scheduler_contract(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _native_contract(name: str, ordinal: int) -> MiniMaxH3SchedulerContract:
    return MiniMaxH3SchedulerContract(
        name=name,
        ordinal=ordinal,
        owner=MiniMaxH3SchedulerOwner.COMFYUI_NATIVE,
        model_policy=MiniMaxH3ModelPolicy.REQUIRED,
        handler_name=name,
        count_policy=MiniMaxH3CountPolicy.BASIC_SCHEDULER_TAIL,
        minimum_steps=2 if name in {"beta", "kl_optimal"} else 1,
        allowed_dtypes=("float32",),
        additional_shift_allowed=False,
        basic_scheduler_tail=True,
    )


MINIMAX_H3_SCHEDULER_CONTRACTS: Final = (
    MiniMaxH3SchedulerContract(
        name="h3_endpoint",
        ordinal=1,
        owner=MiniMaxH3SchedulerOwner.SIGMAX_PURE,
        model_policy=MiniMaxH3ModelPolicy.FORBIDDEN,
        handler_name=None,
        count_policy=MiniMaxH3CountPolicy.EXACT_ENDPOINT,
        minimum_steps=1,
        allowed_dtypes=("float64", "float32"),
    ),
    *tuple(
        _native_contract(name, ordinal)
        for ordinal, name in enumerate(MINIMAX_H3_NATIVE_SCHEDULERS, start=2)
    ),
)
_CONTRACTS_BY_NAME: Final = {contract.name: contract for contract in MINIMAX_H3_SCHEDULER_CONTRACTS}

MINIMAX_H3_SCHEDULER_HOSTS: Final = (
    MiniMaxH3SchedulerHost(
        version="0.30.0",
        revision="14b05228cef127ce529bc0c08660770d4af3e9a8",  # pragma: allowlist secret
        role=MiniMaxH3SchedulerHostRole.ACCEPTED_KNOWN_GOOD,
        scheduler_names=MINIMAX_H3_NATIVE_SCHEDULERS,
        license_id="GPL-3.0-only",
        delegation_only=True,
        url="https://github.com/Comfy-Org/ComfyUI",
        source_locators=("comfy/model_sampling.py", "comfy/samplers.py"),
    ),
    MiniMaxH3SchedulerHost(
        version="0.32.0",
        revision="b323a345bbbfb2f3a95b5b73b68eb7919a26515e",  # pragma: allowlist secret
        role=MiniMaxH3SchedulerHostRole.SUPPLIED_CURRENT,
        scheduler_names=MINIMAX_H3_NATIVE_SCHEDULERS,
        license_id="GPL-3.0-only",
        delegation_only=True,
        url="https://github.com/Comfy-Org/ComfyUI",
        source_locators=("comfy/model_sampling.py", "comfy/samplers.py"),
    ),
)
_HOSTS_BY_REVISION: Final = {host.revision: host for host in MINIMAX_H3_SCHEDULER_HOSTS}

MINIMAX_H3_SCHEDULER_RECIPES: Final = tuple(
    MiniMaxH3SchedulerRecipe(
        recipe_id=recipe.recipe_id,
        task=recipe.task,
        allowed_nfe=recipe.allowed_nfe,
        video_shift=recipe.video_shift,
        audio_shift=recipe.audio_shift,
    )
    for recipe in MINIMAX_H3_ACCELERATION_RECIPES
)
_RECIPES_BY_ID: Final = {recipe.recipe_id: recipe for recipe in MINIMAX_H3_SCHEDULER_RECIPES}


__all__ = [
    "MINIMAX_H3_DEFAULT_SCHEDULER",
    "MINIMAX_H3_NATIVE_SCHEDULERS",
    "MINIMAX_H3_SCHEDULER_CHOICES",
    "MINIMAX_H3_SCHEDULER_CONTRACTS",
    "MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_ID",
    "MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_VERSION",
    "MINIMAX_H3_SCHEDULER_HOSTS",
    "MINIMAX_H3_SCHEDULER_RECIPES",
    "MINIMAX_H3_SCHEDULER_RESULT_SCHEMA_ID",
    "MiniMaxH3CountPolicy",
    "MiniMaxH3ModelPolicy",
    "MiniMaxH3ModelSamplingEvidence",
    "MiniMaxH3QualifiedSchedulerRequest",
    "MiniMaxH3SchedulerContract",
    "MiniMaxH3SchedulerContractError",
    "MiniMaxH3SchedulerHost",
    "MiniMaxH3SchedulerHostRole",
    "MiniMaxH3SchedulerOwner",
    "MiniMaxH3SchedulerReasonCode",
    "MiniMaxH3SchedulerRecipe",
    "MiniMaxH3SchedulerResultValidation",
    "minimax_h3_scheduler_contract_fingerprint",
    "qualify_minimax_h3_scheduler_request",
    "serialize_minimax_h3_scheduler_contract",
    "validate_minimax_h3_scheduler_result",
]
