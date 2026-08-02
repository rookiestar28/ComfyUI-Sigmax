"""Experimental graph-local Krea 2 CONDITIONING rebalance node."""

from __future__ import annotations

from typing import Final

from comfyui_sigmax.adapters.krea2_conditioning import (
    transform_krea2_conditioning,
)
from comfyui_sigmax.conditioning import (
    ConditioningModifierRequest,
    Krea2ConditioningProfileId,
    Krea2ConditioningVariant,
    build_modifier_report,
    get_krea2_profile,
)
from comfyui_sigmax.core import ScheduleContractError

KREA2_CONDITIONING_NODE_ID: Final = "Sigmax.Krea2ConditioningRebalance"
KREA2_CONDITIONING_NODE_SCHEMA_ID: Final = "sigmax.krea2-conditioning-node/1"


def _variant(value: object) -> Krea2ConditioningVariant:
    if not isinstance(value, str):
        raise ScheduleContractError("conditioning node variant must explicitly be RAW or Turbo")
    try:
        return Krea2ConditioningVariant(value)
    except ValueError as exc:
        raise ScheduleContractError(
            "conditioning node variant must explicitly be RAW or Turbo"
        ) from exc


def _profile(value: object) -> Krea2ConditioningProfileId:
    if not isinstance(value, str):
        raise ScheduleContractError("conditioning node profile is unsupported")
    try:
        return Krea2ConditioningProfileId(value)
    except ValueError as exc:
        raise ScheduleContractError("conditioning node profile is unsupported") from exc


def _strength(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScheduleContractError("conditioning node strength must be numeric")
    return float(value)


class Krea2ConditioningRebalance:
    """Rebalance only the primary Krea 2 conditioning tensor before model fusion."""

    CATEGORY = "Sigmax/conditioning"
    DESCRIPTION = (
        "Experimental Krea 2 RAW/Turbo conditioning tap rebalance with fixed RMS preservation. "
        "Does not patch the model or scheduler and does not claim prompt-adherence improvement."
    )
    EXPERIMENTAL = True
    FUNCTION = "rebalance"
    RETURN_NAMES = ("conditioning", "modifier_info")
    RETURN_TYPES = ("CONDITIONING", "STRING")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, object]]:
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "variant": (
                    [
                        Krea2ConditioningVariant.RAW.value,
                        Krea2ConditioningVariant.TURBO.value,
                    ],
                ),
                "profile": ([profile.value for profile in Krea2ConditioningProfileId],),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
            }
        }

    def rebalance(
        self,
        conditioning: object,
        variant: object,
        profile: object,
        strength: object,
    ) -> tuple[list[list[object]], str]:
        selected_variant = _variant(variant)
        selected_profile = _profile(profile)
        request = ConditioningModifierRequest(
            variant=selected_variant,
            profile=get_krea2_profile(selected_profile),
            strength=_strength(strength),
        )
        transformed, stats = transform_krea2_conditioning(conditioning, request)
        warnings = (
            ()
            if selected_profile is Krea2ConditioningProfileId.DISABLED
            else ("experimental_profile",)
        )
        report = build_modifier_report(
            request=request,
            input_shape=stats.input_shape,
            input_shapes=stats.input_shapes,
            dtype=stats.dtype,
            device=stats.device,
            conditioning_entries=stats.conditioning_entries,
            transformed_entries=stats.transformed_entries,
            warnings=warnings,
        )
        return transformed, report.json_text
