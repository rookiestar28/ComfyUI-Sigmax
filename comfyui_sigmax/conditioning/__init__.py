"""Versioned, framework-independent conditioning contracts."""

from comfyui_sigmax.conditioning.contracts import (
    CONDITIONING_MODIFIER_ALGORITHM_ID,
    CONDITIONING_MODIFIER_REPORT_SCHEMA_ID,
    KREA2_CONDITIONING_PROFILE_SCHEMA_ID,
    KREA2_FEATURE_DIM,
    KREA2_TAP_COUNT,
    KREA2_TAP_DIM,
    ConditioningModifierReport,
    ConditioningModifierRequest,
    Krea2ConditioningProfile,
    Krea2ConditioningProfileId,
    Krea2ConditioningVariant,
    build_modifier_report,
    effective_gains,
    get_krea2_profile,
    validate_krea2_conditioning_shape,
)
from comfyui_sigmax.conditioning.diagnostics import (
    ConditioningDiagnostics,
    compute_conditioning_diagnostics,
)

__all__ = [
    "CONDITIONING_MODIFIER_ALGORITHM_ID",
    "CONDITIONING_MODIFIER_REPORT_SCHEMA_ID",
    "KREA2_CONDITIONING_PROFILE_SCHEMA_ID",
    "KREA2_FEATURE_DIM",
    "KREA2_TAP_COUNT",
    "KREA2_TAP_DIM",
    "ConditioningDiagnostics",
    "ConditioningModifierReport",
    "ConditioningModifierRequest",
    "Krea2ConditioningProfile",
    "Krea2ConditioningProfileId",
    "Krea2ConditioningVariant",
    "build_modifier_report",
    "compute_conditioning_diagnostics",
    "effective_gains",
    "get_krea2_profile",
    "validate_krea2_conditioning_shape",
]
