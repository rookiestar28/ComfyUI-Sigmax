"""Thin Krea 2 SIGMAS node over the validated dependency-free profile builders."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core import (
    ScheduleContractError,
    SigmaDomain,
    canonical_projection_bytes,
    float_to_ieee_hex,
    numerical_fingerprint,
    slice_step_range,
    validate_sigma_schedule,
)
from comfyui_sigmax.profiles import (
    KREA2_RAW_DIFFUSERS_REFERENCE_28,
    KREA2_RAW_OFFICIAL_FULL_52,
    KREA2_TURBO_PROFILE,
    build_krea2_raw_schedule,
    build_krea2_turbo_schedule,
    derive_krea2_raw_shift,
)

KREA2_SIGMA_NODE_ID: Final = "Sigmax.Krea2SigmaScheduler"
KREA2_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.krea2-sigma-node/1"
_OUTPUT_FINGERPRINT_SCHEMA: Final = "sigmax.sigma-output/1"
_TURBO_OFFICIAL_RECIPE: Final = "krea2.turbo.official-8"
_MAX_STEPS: Final = 10_000
_MAX_DIMENSION: Final = 65_536


class Krea2SigmaVariant(str, Enum):
    """Explicit Krea 2 product variants supported by the first node."""

    TURBO = "Turbo"
    RAW = "RAW"


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2SigmaNodeResult:
    """Pure node output before the host-provided tensor conversion."""

    variant: Krea2SigmaVariant
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.variant, Krea2SigmaVariant):
            raise ScheduleContractError("node result variant is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("Krea 2 node result must use UNIT_FLOW")
        if not isinstance(self.sigmas, tuple) or len(self.sigmas) < 2:
            raise ScheduleContractError("node result requires at least one sigma transition")
        if not isinstance(self.schedule_info_json, str) or not self.schedule_info_json:
            raise ScheduleContractError("node result requires schedule information")


def _positive_integer(value: object, *, label: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > maximum:
        raise ScheduleContractError(f"{label} must be an integer between 1 and {maximum}")
    return value


def _variant(value: object) -> Krea2SigmaVariant:
    if not isinstance(value, str):
        raise ScheduleContractError("variant must be Turbo or RAW")
    try:
        return Krea2SigmaVariant(value)
    except ValueError as exc:
        raise ScheduleContractError("variant must be Turbo or RAW") from exc


def _slice_bounds(
    *,
    start_step: object,
    end_step: object,
    available_steps: int,
) -> tuple[int, int | None]:
    if not isinstance(start_step, int) or isinstance(start_step, bool) or start_step < 0:
        raise ScheduleContractError("start_step must be a non-negative integer")
    if not isinstance(end_step, int) or isinstance(end_step, bool) or end_step < -1:
        raise ScheduleContractError("end_step must be -1 or a non-negative integer")
    effective_end = None if end_step == -1 else end_step
    if start_step >= available_steps:
        raise ScheduleContractError("start_step must be below the constructed step count")
    if effective_end is not None and (
        effective_end <= start_step or effective_end > available_steps
    ):
        raise ScheduleContractError(
            "end_step must exceed start_step and not exceed the constructed step count"
        )
    return start_step, effective_end


def sigma_output_fingerprint(
    sigmas: tuple[float, ...],
    *,
    domain: SigmaDomain,
) -> str:
    """Fingerprint a selected monotonic sigma range without requiring terminal zero."""

    values = validate_sigma_schedule(
        sigmas,
        domain=domain,
        expected_steps=len(sigmas) - 1,
        require_terminal_zero=False,
    )
    projection: dict[str, object] = {
        "domain": domain.value.casefold(),
        "precision": "float64",
        "schema": _OUTPUT_FINGERPRINT_SCHEMA,
        "sigmas": [float_to_ieee_hex(value, "float64") for value in values],
    }
    return f"sha256:{hashlib.sha256(canonical_projection_bytes(projection)).hexdigest()}"


def _canonical_info(projection: dict[str, object]) -> str:
    try:
        return json.dumps(
            projection,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("schedule information is not canonical JSON") from exc


def build_krea2_sigma_schedule(
    *,
    variant: object,
    steps: object,
    width: object,
    height: object,
    strict_official: object,
    start_step: object,
    end_step: object,
) -> Krea2SigmaNodeResult:
    """Build and slice one explicit Krea 2 schedule without importing host frameworks."""

    selected_variant = _variant(variant)
    requested_steps = _positive_integer(steps, label="steps", maximum=_MAX_STEPS)
    requested_width = _positive_integer(width, label="width", maximum=_MAX_DIMENSION)
    requested_height = _positive_integer(height, label="height", maximum=_MAX_DIMENSION)
    if not isinstance(strict_official, bool):
        raise ScheduleContractError("strict_official must be boolean")

    if selected_variant is Krea2SigmaVariant.TURBO:
        if strict_official and requested_steps != 8:
            raise ScheduleContractError("strict official Turbo requires exactly 8 steps")
        complete = build_krea2_turbo_schedule(
            steps=requested_steps,
            width=requested_width,
            height=requested_height,
        )
        recipe_id = (
            _TURBO_OFFICIAL_RECIPE
            if requested_steps == 8
            else f"krea2.turbo.modified-{requested_steps}"
        )
        shift: dict[str, object] = {
            "kind": "fixed_exponential_mu",
            "mu": KREA2_TURBO_PROFILE.fixed_mu,
        }
    else:
        if strict_official and requested_steps != KREA2_RAW_OFFICIAL_FULL_52.steps:
            raise ScheduleContractError("strict official RAW requires exactly 52 steps")
        recipes = {
            KREA2_RAW_DIFFUSERS_REFERENCE_28.steps: KREA2_RAW_DIFFUSERS_REFERENCE_28,
            KREA2_RAW_OFFICIAL_FULL_52.steps: KREA2_RAW_OFFICIAL_FULL_52,
        }
        recipe = recipes.get(requested_steps)
        if recipe is None:
            raise ScheduleContractError("RAW steps must select the named 28-step or 52-step recipe")
        complete = build_krea2_raw_schedule(
            width=requested_width,
            height=requested_height,
            recipe=recipe,
        )
        recipe_id = recipe.recipe_id
        derivation = derive_krea2_raw_shift(requested_width, requested_height)
        shift = {
            "extrapolated": derivation.extrapolated,
            "image_seq_len": derivation.geometry.image_seq_len,
            "kind": "resolution_exponential_mu",
            "mu": derivation.mu,
        }

    start, end = _slice_bounds(
        start_step=start_step,
        end_step=end_step,
        available_steps=complete.effective_inputs.steps,
    )
    output_sigmas = slice_step_range(
        complete.sigmas,
        start_step=start,
        end_step=end,
    )
    effective_end = complete.effective_inputs.steps if end is None else end
    provenance = complete.request.provenance
    projection: dict[str, object] = {
        "dimensions": {
            "effective": {
                "height": complete.effective_inputs.height,
                "width": complete.effective_inputs.width,
            },
            "requested": {
                "height": complete.request.requested_inputs.height,
                "width": complete.request.requested_inputs.width,
            },
        },
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas,
                domain=complete.final_domain,
                precision="float64",
            ),
            "output": sigma_output_fingerprint(
                output_sigmas,
                domain=complete.final_domain,
            ),
        },
        "profile": {
            "evidence": provenance.evidence.value,
            "id": provenance.profile_id,
            "recipe": recipe_id,
            "variant": selected_variant.value.casefold(),
            "version": provenance.profile_version,
        },
        "schema": KREA2_SIGMA_NODE_SCHEMA_ID,
        "shift": shift,
        "slicing": {
            "available_steps": complete.effective_inputs.steps,
            "end_step": effective_end,
            "output_steps": len(output_sigmas) - 1,
            "start_step": start,
        },
        "strict_official": strict_official,
        "warnings": list(complete.warnings),
    }
    return Krea2SigmaNodeResult(
        variant=selected_variant,
        domain=complete.final_domain,
        sigmas=output_sigmas,
        schedule_info_json=_canonical_info(projection),
    )


class Krea2SigmaScheduler:
    """Construct validated Krea 2 external sigmas for ComfyUI custom sampling."""

    DESCRIPTION = (
        "Builds an explicit Krea 2 RAW or Turbo sigma schedule without patching model sampling."
    )
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "build"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        """Return a fresh deterministic legacy/current ComfyUI input schema."""

        return {
            "required": {
                "variant": (("Turbo", "RAW"),),
                "steps": (
                    "INT",
                    {"default": 8, "min": 1, "max": _MAX_STEPS, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 16, "max": _MAX_DIMENSION, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 16, "max": _MAX_DIMENSION, "step": 16},
                ),
                "strict_official": ("BOOLEAN", {"default": True}),
                "start_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": _MAX_STEPS - 1, "step": 1},
                ),
                "end_step": (
                    "INT",
                    {"default": -1, "min": -1, "max": _MAX_STEPS, "step": 1},
                ),
            }
        }

    def build(
        self,
        variant: object,
        steps: object,
        width: object,
        height: object,
        strict_official: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        """Build pure output first, then convert through host-provided Torch."""

        result = build_krea2_sigma_schedule(
            variant=variant,
            steps=steps,
            width=width,
            height=height,
            strict_official=strict_official,
            start_step=start_step,
            end_step=end_step,
        )
        try:
            torch = importlib.import_module("torch")
            float_tensor = torch.__dict__["FloatTensor"]
        except (ImportError, KeyError) as exc:
            # CRITICAL: keep Torch execution-only; package imports must remain dependency-free.
            raise RuntimeError("ComfyUI host execution requires Torch FloatTensor support") from exc
        return (float_tensor(result.sigmas), result.schedule_info_json)
