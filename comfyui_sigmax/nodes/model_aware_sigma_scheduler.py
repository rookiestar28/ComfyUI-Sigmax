"""Model-aware ComfyUI SIGMAS node over exact profiles and capability contracts."""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.adapters import adapt_krea2_model_evidence
from comfyui_sigmax.core import (
    CompatibilityLevel,
    ExecutionFeatureRequest,
    ScheduleContractError,
    SigmaDomain,
)
from comfyui_sigmax.nodes.krea2_sigma_scheduler import build_krea2_sigma_schedule
from comfyui_sigmax.profiles import (
    HostCapabilities,
    HostCapabilityEvidence,
    HostCapabilityLifecycle,
    Krea2Variant,
    Krea2VariantResolution,
    ProfileCapabilityDecision,
    ProfileKey,
    builtin_profile_registry,
    resolve_krea2_variant,
    resolve_profile_capabilities,
)

MODEL_AWARE_SIGMA_NODE_ID: Final = "Sigmax.ModelAwareSigmaScheduler"
MODEL_AWARE_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.model-aware-sigma-node/1"
MODEL_FAMILY_PROBE_SCHEMA_ID: Final = "sigmax.model-family-probe/1"
_COMFYUI_HOST_VERSION: Final = "0.29.0"
_COMFYUI_HOST_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"
_MAX_STEPS: Final = 10_000
_MAX_DIMENSION: Final = 65_536
_REASON_PATTERN: Final = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+")


class ModelAwareVariant(str, Enum):
    """Public model-aware variant modes."""

    AUTO = "Auto"
    TURBO = "Turbo"
    RAW = "RAW"


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelFamilyProbe:
    """Bounded public family evidence extracted without invoking model methods."""

    schema_id: str
    family: str | None
    supported: bool
    signals: tuple[str, ...]
    reason_code: str

    def __post_init__(self) -> None:
        if self.schema_id != MODEL_FAMILY_PROBE_SCHEMA_ID:
            raise ScheduleContractError("model family probe schema is unsupported")
        if self.family not in {None, "krea2"}:
            raise ScheduleContractError("model family probe family is unsupported")
        if not isinstance(self.supported, bool):
            raise ScheduleContractError("model family probe supported must be boolean")
        if self.supported != (self.family == "krea2"):
            raise ScheduleContractError("model family probe support is inconsistent")
        if not isinstance(self.signals, tuple) or not all(
            isinstance(signal, str) and signal for signal in self.signals
        ):
            raise ScheduleContractError("model family probe signals must be strings")
        if self.signals != tuple(sorted(set(self.signals))):
            raise ScheduleContractError("model family probe signals must be canonical")
        if not isinstance(self.reason_code, str) or not _REASON_PATTERN.fullmatch(self.reason_code):
            raise ScheduleContractError("model family probe reason is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelAwareSigmaNodeResult:
    """Pure model-aware result before host-provided tensor conversion."""

    selected_variant: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.selected_variant not in {"raw", "turbo"}:
            raise ScheduleContractError("model-aware result variant is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("model-aware result must use UNIT_FLOW")
        if not isinstance(self.sigmas, tuple) or len(self.sigmas) < 2:
            raise ScheduleContractError("model-aware result requires sigma transitions")
        if not isinstance(self.schedule_info_json, str) or not self.schedule_info_json:
            raise ScheduleContractError("model-aware result requires schedule information")


class ModelAwareScheduleError(ScheduleContractError):
    """Stable actionable rejection without exposing a foreign MODEL object."""

    reason_code: str
    action: str
    decision_json: str

    def __init__(self, *, reason_code: str, action: str, decision_json: str) -> None:
        if not isinstance(reason_code, str) or not _REASON_PATTERN.fullmatch(reason_code):
            raise ScheduleContractError("model-aware error reason is invalid")
        if not isinstance(action, str) or not action.strip():
            raise ScheduleContractError("model-aware error action is required")
        if not isinstance(decision_json, str) or not decision_json:
            raise ScheduleContractError("model-aware error decision is required")
        self.reason_code = reason_code
        self.action = action
        self.decision_json = decision_json
        super().__init__(f"{reason_code}: {action}")


def _static_attribute(value: object, name: str) -> object:
    try:
        candidate = inspect.getattr_static(value, name)
    except AttributeError:
        return _MISSING
    if isinstance(candidate, property) or inspect.isroutine(candidate):
        return _MISSING
    return candidate


_MISSING: Final = object()


def _class_name(value: object) -> str:
    return type(value).__name__.casefold()


def probe_model_family(model: object) -> ModelFamilyProbe:
    """Read a bounded public ComfyUI MODEL shape and return family-only evidence."""

    # CRITICAL: keep this probe static and bounded; arbitrary model calls or repr can leak data.
    inner = _static_attribute(model, "model")
    if inner is _MISSING:
        return ModelFamilyProbe(
            schema_id=MODEL_FAMILY_PROBE_SCHEMA_ID,
            family=None,
            supported=False,
            signals=(),
            reason_code="model.family_unsupported",
        )
    if inner is None:
        return ModelFamilyProbe(
            schema_id=MODEL_FAMILY_PROBE_SCHEMA_ID,
            family=None,
            supported=False,
            signals=(),
            reason_code="model.probe_invalid",
        )

    signals: list[str] = []
    if _class_name(inner) == "krea2":
        signals.append("model_class:krea2")

    model_config = _static_attribute(inner, "model_config")
    if model_config is not _MISSING and model_config is not None:
        if _class_name(model_config) == "krea2":
            signals.append("model_config_class:krea2")
        unet_config = _static_attribute(model_config, "unet_config")
        if unet_config is not _MISSING:
            if not isinstance(unet_config, Mapping):
                return ModelFamilyProbe(
                    schema_id=MODEL_FAMILY_PROBE_SCHEMA_ID,
                    family=None,
                    supported=False,
                    signals=(),
                    reason_code="model.probe_invalid",
                )
            image_model = unet_config.get("image_model")
            if isinstance(image_model, str) and image_model.casefold() == "krea2":
                signals.append("unet_config.image_model:krea2")

    canonical_signals = tuple(sorted(set(signals)))
    supported = bool(canonical_signals)
    return ModelFamilyProbe(
        schema_id=MODEL_FAMILY_PROBE_SCHEMA_ID,
        family="krea2" if supported else None,
        supported=supported,
        signals=canonical_signals,
        reason_code="model.family_krea2" if supported else "model.family_unsupported",
    )


def _canonical_json(projection: dict[str, object]) -> str:
    try:
        return json.dumps(
            projection,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("model-aware information is not canonical JSON") from exc


def _probe_projection(probe: ModelFamilyProbe) -> dict[str, object]:
    return {
        "family": probe.family,
        "reason_code": probe.reason_code,
        "schema": probe.schema_id,
        "signals": list(probe.signals),
        "supported": probe.supported,
    }


def _variant_resolution_projection(
    resolution: Krea2VariantResolution,
) -> dict[str, object]:
    reason_codes = tuple(
        sorted(
            {
                *(item.reason_code for item in resolution.evidence),
                *resolution.warnings,
            }
        )
    )
    return {
        "confidence": resolution.confidence.value,
        "decisive_source": (
            resolution.decisive_source.value if resolution.decisive_source is not None else None
        ),
        "reason_codes": list(reason_codes),
        "resolved_variant": (
            resolution.resolved_variant.value if resolution.resolved_variant is not None else None
        ),
        "status": resolution.status.value,
        "suggested_variant": (
            resolution.suggested_variant.value if resolution.suggested_variant is not None else None
        ),
    }


def _capability_projection(
    decision: ProfileCapabilityDecision,
) -> dict[str, object]:
    identity = decision.model_identity
    return {
        "core_decision": {
            "considered": [item.value for item in decision.core_decision.considered],
            "level": decision.core_decision.level.value,
            "reasons": [item.value for item in decision.core_decision.reasons],
        },
        "host": {
            "id": decision.host_id,
            "requirements": [
                {
                    "capability_id": item.capability_id,
                    "lifecycle": item.lifecycle.value if item.lifecycle is not None else None,
                    "reason_code": item.reason_code,
                    "satisfied": item.satisfied,
                }
                for item in decision.host_requirements
            ],
            "revision": decision.host_revision,
            "version": decision.host_version,
        },
        "level": decision.level.value,
        "model_identity": {
            "confidence": identity.confidence,
            "confirmed_variant": identity.confirmed_variant,
            "decisive_source": identity.decisive_source,
            "family": identity.model_family,
            "reason_codes": list(identity.reason_codes),
            "status": identity.status.value,
            "suggested_variant": identity.suggested_variant,
        },
        "profile_fingerprint": decision.profile_fingerprint,
        "profile_key": decision.profile_key,
        "reason_codes": list(decision.reason_codes),
        "schema": decision.schema_id,
        "schema_version": decision.schema_version,
    }


def _static_host_capabilities() -> HostCapabilities:
    return HostCapabilities(
        evidence_version="1",
        host_id="comfyui",
        host_version=_COMFYUI_HOST_VERSION,
        host_revision=_COMFYUI_HOST_REVISION,
        capabilities=(
            HostCapabilityEvidence(
                capability_id="sampler.comfy.euler",
                lifecycle=HostCapabilityLifecycle.LANDED,
            ),
            HostCapabilityEvidence(
                capability_id="schedule.external_sigmas",
                lifecycle=HostCapabilityLifecycle.LANDED,
            ),
        ),
    )


def _mode(value: object) -> ModelAwareVariant:
    if not isinstance(value, str):
        raise ScheduleContractError("variant must be Auto, Turbo, or RAW")
    try:
        return ModelAwareVariant(value)
    except ValueError as exc:
        raise ScheduleContractError("variant must be Auto, Turbo, or RAW") from exc


def _rejection(
    *,
    reason_code: str,
    action: str,
    probe: ModelFamilyProbe,
    resolution: Krea2VariantResolution | None = None,
    capability: ProfileCapabilityDecision | None = None,
) -> ModelAwareScheduleError:
    projection: dict[str, object] = {
        "capability_decision": (
            _capability_projection(capability) if capability is not None else None
        ),
        "model_probe": _probe_projection(probe),
        "schema": MODEL_AWARE_SIGMA_NODE_SCHEMA_ID,
        "variant_resolution": (
            _variant_resolution_projection(resolution) if resolution is not None else None
        ),
    }
    return ModelAwareScheduleError(
        reason_code=reason_code,
        action=action,
        decision_json=_canonical_json(projection),
    )


def build_model_aware_sigma_schedule(
    *,
    model: object,
    variant: object,
    steps: object,
    width: object,
    height: object,
    strict_official: object,
    start_step: object,
    end_step: object,
) -> ModelAwareSigmaNodeResult:
    """Resolve exact Krea 2 capabilities, then delegate numerical construction to M4-01."""

    selected_mode = _mode(variant)
    probe = probe_model_family(model)
    if not probe.supported:
        raise _rejection(
            reason_code=probe.reason_code,
            action="connect a supported Krea 2 MODEL before selecting a Sigmax profile",
            probe=probe,
        )

    explicit_variant = {
        ModelAwareVariant.AUTO: None,
        ModelAwareVariant.TURBO: Krea2Variant.TURBO,
        ModelAwareVariant.RAW: Krea2Variant.RAW,
    }[selected_mode]
    resolution = resolve_krea2_variant(
        strict_official=False,
        explicit_variant=explicit_variant,
        model_class="Krea2",
    )
    if resolution.resolved_variant is None:
        raise _rejection(
            reason_code=f"model.identity_{resolution.status.value}",
            action="select Turbo or RAW explicitly, or provide trusted variant evidence",
            probe=probe,
            resolution=resolution,
        )

    selected_variant = resolution.resolved_variant
    profile_key = ProfileKey(
        profile_id=f"krea2.{selected_variant.value}.official",
        profile_version="1",
    )
    registered_profile = builtin_profile_registry().resolve(profile_key)
    model_evidence = adapt_krea2_model_evidence(
        registered_profile=registered_profile,
        explicit_variant=selected_variant,
        model_class="Krea2",
    )
    capability = resolve_profile_capabilities(
        registered_profile=registered_profile,
        model=model_evidence,
        host=_static_host_capabilities(),
        sampler=registered_profile.schema.reference_sampler_capabilities,
        request=ExecutionFeatureRequest(),
    )
    if capability.level is CompatibilityLevel.REJECT:
        raise _rejection(
            reason_code=capability.reason_codes[0],
            action="resolve the reported model, host, or sampler capability mismatch",
            probe=probe,
            resolution=resolution,
            capability=capability,
        )

    display_variant = (
        ModelAwareVariant.TURBO.value
        if selected_variant is Krea2Variant.TURBO
        else ModelAwareVariant.RAW.value
    )
    schedule = build_krea2_sigma_schedule(
        variant=display_variant,
        steps=steps,
        width=width,
        height=height,
        strict_official=strict_official,
        start_step=start_step,
        end_step=end_step,
    )
    projection: dict[str, object] = {
        "capability_decision": _capability_projection(capability),
        "host_evidence": {
            "host_id": _static_host_capabilities().host_id,
            "host_revision": _COMFYUI_HOST_REVISION,
            "host_version": _COMFYUI_HOST_VERSION,
            "kind": "static_contract",
        },
        "model_probe": _probe_projection(probe),
        "profile": {
            "evidence": registered_profile.schema.evidence.value,
            "fingerprint": registered_profile.fingerprint,
            "key": registered_profile.key.canonical,
            "origin": registered_profile.origin.value,
        },
        "schedule": json.loads(schedule.schedule_info_json),
        "schema": MODEL_AWARE_SIGMA_NODE_SCHEMA_ID,
        "variant_resolution": _variant_resolution_projection(resolution),
    }
    return ModelAwareSigmaNodeResult(
        selected_variant=selected_variant.value,
        domain=schedule.domain,
        sigmas=schedule.sigmas,
        schedule_info_json=_canonical_json(projection),
    )


class ModelAwareSigmaScheduler:
    """Resolve a supported MODEL to exact Krea 2 external sigmas."""

    DESCRIPTION = (
        "Validates a Krea 2 MODEL, resolves an exact profile, and emits capability-gated sigmas."
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
                "model": ("MODEL",),
                "variant": (("Auto", "Turbo", "RAW"),),
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
        model: object,
        variant: object,
        steps: object,
        width: object,
        height: object,
        strict_official: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        """Resolve pure output first, then convert through host-provided Torch."""

        result = build_model_aware_sigma_schedule(
            model=model,
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
