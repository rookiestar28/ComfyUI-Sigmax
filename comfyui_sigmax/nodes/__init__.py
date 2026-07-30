"""Thin user-facing ComfyUI nodes over validated Sigmax contracts."""

from comfyui_sigmax.nodes.advanced_flowmatch_scheduler import (
    ADVANCED_FLOWMATCH_NODE_ID,
    ADVANCED_FLOWMATCH_NODE_SCHEMA_ID,
    AdvancedFlowMatchNodeResult,
    AdvancedFlowMatchScheduler,
    AdvancedFlowMatchShiftMode,
    build_advanced_flowmatch_schedule,
)
from comfyui_sigmax.nodes.krea2_sigma_scheduler import (
    KREA2_SIGMA_NODE_ID,
    KREA2_SIGMA_NODE_SCHEMA_ID,
    Krea2SigmaNodeResult,
    Krea2SigmaScheduler,
    Krea2SigmaVariant,
    build_krea2_sigma_schedule,
    sigma_output_fingerprint,
)
from comfyui_sigmax.nodes.model_aware_sigma_scheduler import (
    MODEL_AWARE_SIGMA_NODE_ID,
    MODEL_AWARE_SIGMA_NODE_SCHEMA_ID,
    MODEL_FAMILY_PROBE_SCHEMA_ID,
    ModelAwareScheduleError,
    ModelAwareSigmaNodeResult,
    ModelAwareSigmaScheduler,
    ModelAwareVariant,
    ModelFamilyProbe,
    build_model_aware_sigma_schedule,
    probe_model_family,
)

__all__ = [
    "ADVANCED_FLOWMATCH_NODE_ID",
    "ADVANCED_FLOWMATCH_NODE_SCHEMA_ID",
    "KREA2_SIGMA_NODE_ID",
    "KREA2_SIGMA_NODE_SCHEMA_ID",
    "MODEL_AWARE_SIGMA_NODE_ID",
    "MODEL_AWARE_SIGMA_NODE_SCHEMA_ID",
    "MODEL_FAMILY_PROBE_SCHEMA_ID",
    "AdvancedFlowMatchNodeResult",
    "AdvancedFlowMatchScheduler",
    "AdvancedFlowMatchShiftMode",
    "Krea2SigmaNodeResult",
    "Krea2SigmaScheduler",
    "Krea2SigmaVariant",
    "ModelAwareScheduleError",
    "ModelAwareSigmaNodeResult",
    "ModelAwareSigmaScheduler",
    "ModelAwareVariant",
    "ModelFamilyProbe",
    "build_advanced_flowmatch_schedule",
    "build_krea2_sigma_schedule",
    "build_model_aware_sigma_schedule",
    "probe_model_family",
    "sigma_output_fingerprint",
]
