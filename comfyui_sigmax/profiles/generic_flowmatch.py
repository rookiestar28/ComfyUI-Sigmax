"""Explicit framework-level FlowMatch profiles without model claims."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core import (
    EvidenceLevel,
    ExecutionBehavior,
    NoiseOwnership,
    PredictionType,
    SamplerCapabilities,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalPolicy,
    TerminalRequirement,
    TerminalSigma,
    TransformStage,
    float_to_ieee_hex,
)
from comfyui_sigmax.profiles.schema_v1 import (
    BaseGridDeclaration,
    DetectionDeclaration,
    FrameworkProvenance,
    LicenseDeclaration,
    ProfileField,
    SlicingDeclaration,
    TerminalDeclaration,
    TransformDeclaration,
)

GENERIC_FLOWMATCH_PROFILE_SCHEMA_ID: Final = "sigmax.generic-flowmatch-profile/1"
GENERIC_FLOWMATCH_PROFILE_SCHEMA_VERSION: Final = "1"

_PROFILE_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)+$")
_VERSION_PATTERN: Final = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ALLOWED_EVIDENCE: Final = frozenset(
    {EvidenceLevel.FRAMEWORK_REFERENCE, EvidenceLevel.EXPERIMENTAL}
)
_STEP_POLICY: Final = "explicit_positive_integer"
_GUIDANCE_POLICY: Final = "model_specific_unset"


class GenericFlowMatchShiftMode(str, Enum):
    """Framework-level shift modes that make no model compatibility claim."""

    FIXED = "fixed"
    DYNAMIC = "dynamic"


def _require_public_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ScheduleContractError(f"{field_name} must be bounded non-empty public text")
    return value


def _require_identifier(field_name: str, value: object) -> str:
    text = _require_public_text(field_name, value)
    if not _IDENTIFIER_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a stable lowercase identifier")
    return text


def _require_exact_selection(selection: object) -> None:
    if not isinstance(selection, DetectionDeclaration) or (
        selection.strict_default,
        selection.ambiguity_requires_explicit,
        selection.resolving_sources,
        selection.suggestion_sources,
        selection.family_only_sources,
    ) != (True, True, ("explicit_selection",), (), ()):
        raise ScheduleContractError("generic FlowMatch selection must be explicit-only")


def _require_canonical_identifiers(field_name: str, values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ScheduleContractError(f"{field_name} must be a tuple")
    if not all(isinstance(value, str) and _IDENTIFIER_PATTERN.fullmatch(value) for value in values):
        raise ScheduleContractError(f"{field_name} must contain stable lowercase identifiers")
    typed = tuple(values)
    if typed != tuple(sorted(set(typed))):
        raise ScheduleContractError(f"{field_name} must be unique and canonically ordered")
    return typed


@dataclass(frozen=True, slots=True, kw_only=True)
class GenericFlowMatchProfileV1:
    """Immutable schedule-framework declaration that cannot impersonate a model profile."""

    schema_id: str
    schema_version: str
    profile_id: str
    profile_version: str
    display_name: str
    evidence: EvidenceLevel
    primary_source_id: str
    shift_mode: GenericFlowMatchShiftMode
    prediction_type: PredictionType
    sigma_domain: SigmaDomain
    ownership: ScheduleOwnership
    base_grid: BaseGridDeclaration
    transform: TransformDeclaration
    terminal: TerminalDeclaration
    slicing: SlicingDeclaration
    selection: DetectionDeclaration
    reference_sampler_capabilities: SamplerCapabilities
    framework: FrameworkProvenance
    required_inputs: tuple[str, ...]
    step_policy: str
    guidance_policy: str
    parameters: tuple[ProfileField, ...]
    known_limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_id != GENERIC_FLOWMATCH_PROFILE_SCHEMA_ID
            or self.schema_version != GENERIC_FLOWMATCH_PROFILE_SCHEMA_VERSION
        ):
            raise ScheduleContractError(
                "generic FlowMatch schema identifier/version is unsupported"
            )
        if not isinstance(self.profile_id, str) or not _PROFILE_ID_PATTERN.fullmatch(
            self.profile_id
        ):
            raise ScheduleContractError("generic FlowMatch profile_id must be namespaced")
        if not isinstance(self.profile_version, str) or not _VERSION_PATTERN.fullmatch(
            self.profile_version
        ):
            raise ScheduleContractError("generic FlowMatch profile_version must be numeric")
        _require_public_text("display_name", self.display_name)
        if self.evidence not in _ALLOWED_EVIDENCE:
            raise ScheduleContractError(
                "generic FlowMatch evidence must be framework_reference or experimental"
            )
        _require_identifier("primary_source_id", self.primary_source_id)
        if not isinstance(self.shift_mode, GenericFlowMatchShiftMode):
            raise ScheduleContractError("generic FlowMatch shift_mode is unsupported")
        if self.prediction_type is not PredictionType.FLOW_VELOCITY:
            raise ScheduleContractError("generic FlowMatch prediction type must be flow velocity")
        if self.sigma_domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("generic FlowMatch sigma domain must be UNIT_FLOW")
        if self.ownership is not ScheduleOwnership.EXTERNAL_SIGMAS:
            raise ScheduleContractError("generic FlowMatch ownership must be EXTERNAL_SIGMAS")
        self._validate_schedule_contract()
        _require_exact_selection(self.selection)
        self._validate_sampler_contract()
        if not isinstance(self.framework, FrameworkProvenance):
            raise ScheduleContractError("generic FlowMatch framework provenance is required")
        if self.primary_source_id != self.framework.framework_id:
            raise ScheduleContractError("primary source must match framework provenance")
        required_inputs = _require_canonical_identifiers(
            "required_inputs",
            self.required_inputs,
        )
        if self.step_policy != _STEP_POLICY:
            raise ScheduleContractError("generic FlowMatch steps must be explicit")
        if self.guidance_policy != _GUIDANCE_POLICY:
            raise ScheduleContractError("generic FlowMatch guidance must remain model-specific")
        if not isinstance(self.parameters, tuple) or not all(
            isinstance(field, ProfileField) for field in self.parameters
        ):
            raise ScheduleContractError("generic FlowMatch parameters must be ProfileField values")
        if tuple(field.name for field in self.parameters) != tuple(
            sorted(field.name for field in self.parameters)
        ):
            raise ScheduleContractError("generic FlowMatch parameters must be canonically ordered")
        if (
            not isinstance(self.known_limitations, tuple)
            or not self.known_limitations
            or not all(
                isinstance(value, str) and value.strip() and len(value) <= 256
                for value in self.known_limitations
            )
        ):
            raise ScheduleContractError("generic FlowMatch known limitations are required")
        self._validate_mode(required_inputs)

    def _validate_schedule_contract(self) -> None:
        if not isinstance(self.base_grid, BaseGridDeclaration) or (
            self.base_grid.output_domain is not SigmaDomain.UNIT_FLOW
            or self.base_grid.terminal_included
        ):
            raise ScheduleContractError(
                "generic FlowMatch base grid must be terminal-free UNIT_FLOW"
            )
        if not isinstance(self.transform, TransformDeclaration) or (
            self.transform.stage is not TransformStage.PRIMARY_TIME_SHIFT
            or self.transform.input_domain is not SigmaDomain.UNIT_FLOW
            or self.transform.output_domain is not SigmaDomain.UNIT_FLOW
        ):
            raise ScheduleContractError("generic FlowMatch requires one UNIT_FLOW primary shift")
        if not isinstance(self.terminal, TerminalDeclaration) or (
            self.terminal.policy is not TerminalPolicy.APPEND_ZERO
            or self.terminal.sigma is not TerminalSigma.ZERO
            or self.terminal.value != 0.0
        ):
            raise ScheduleContractError("generic FlowMatch terminal must append zero")
        if not isinstance(self.slicing, SlicingDeclaration):
            raise ScheduleContractError("generic FlowMatch slicing declaration is required")

    def _validate_sampler_contract(self) -> None:
        sampler = self.reference_sampler_capabilities
        if not isinstance(sampler, SamplerCapabilities) or (
            sampler.sampler_id != "flowmatch.euler"
            or sampler.accepted_prediction_types != (PredictionType.FLOW_VELOCITY,)
            or sampler.accepted_sigma_domains != (SigmaDomain.UNIT_FLOW,)
            or sampler.accepted_ownerships != (ScheduleOwnership.EXTERNAL_SIGMAS,)
            or sampler.terminal_requirement is not TerminalRequirement.REQUIRES_ZERO
            or sampler.execution_behavior is not ExecutionBehavior.DETERMINISTIC
            or sampler.noise_ownership is not NoiseOwnership.NONE
            or sampler.required_state != ()
            or sampler.supports_partial_denoise
            or sampler.supports_per_token_timesteps
        ):
            raise ScheduleContractError(
                "generic FlowMatch sampler contract must be deterministic Euler structure"
            )

    def _validate_mode(self, required_inputs: tuple[str, ...]) -> None:
        if self.shift_mode is GenericFlowMatchShiftMode.FIXED:
            if (
                self.evidence is not EvidenceLevel.FRAMEWORK_REFERENCE
                or self.transform.identifier != "diffusers.direct_ratio"
                or self.transform.parameters != (ProfileField(name="ratio", value=1.0),)
                or required_inputs
            ):
                raise ScheduleContractError(
                    "fixed generic FlowMatch shift contract is inconsistent"
                )
            return
        if (
            self.evidence is not EvidenceLevel.EXPERIMENTAL
            or self.transform.identifier != "diffusers.exponential_mu"
            or self.transform.parameters
            != (
                ProfileField(name="exponent", value=1.0),
                ProfileField(name="mu", value=None),
                ProfileField(name="time_shift_type", value="exponential"),
            )
            or required_inputs != ("mu",)
        ):
            raise ScheduleContractError("dynamic generic FlowMatch shift contract is inconsistent")


_APACHE_2_LICENSE: Final = LicenseDeclaration(
    declaration_version="1",
    identifier="Apache-2.0",
    name="Apache License 2.0",
    url="https://www.apache.org/licenses/LICENSE-2.0",
)
_DIFFUSERS_FLOWMATCH_FRAMEWORK: Final = FrameworkProvenance(
    record_version="1",
    framework_id="diffusers.flowmatch.framework",
    resource_version="0.39.0",
    revision="a3608b512ed7248499a44c61d954965ed9bdae4d",  # pragma: allowlist secret
    url="https://github.com/huggingface/diffusers",
    license=_APACHE_2_LICENSE,
    locators=("src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",),
)
_BASE_GRID: Final = BaseGridDeclaration(
    identifier="diffusers.flowmatch_linear",
    output_domain=SigmaDomain.UNIT_FLOW,
    terminal_included=False,
    parameters=(
        ProfileField(name="num_train_timesteps", value=1000),
        ProfileField(name="sigma_end", value=0.001),
        ProfileField(name="sigma_start", value=1.0),
    ),
)
_TERMINAL: Final = TerminalDeclaration(
    policy=TerminalPolicy.APPEND_ZERO,
    sigma=TerminalSigma.ZERO,
    value=0.0,
)
_SLICING: Final = SlicingDeclaration(
    supports_step_range=True,
    supports_denoise_tail=True,
    zero_denoise_is_empty=True,
)
_EXPLICIT_SELECTION: Final = DetectionDeclaration(
    strategy_id="generic.flowmatch.explicit-v1",
    strict_default=True,
    ambiguity_requires_explicit=True,
    resolving_sources=("explicit_selection",),
    suggestion_sources=(),
    family_only_sources=(),
)
_DETERMINISTIC_EULER: Final = SamplerCapabilities(
    sampler_id="flowmatch.euler",
    sampler_version="1",
    accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
    accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
    accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
    terminal_requirement=TerminalRequirement.REQUIRES_ZERO,
    execution_behavior=ExecutionBehavior.DETERMINISTIC,
    noise_ownership=NoiseOwnership.NONE,
    required_state=(),
    supports_partial_denoise=False,
    supports_per_token_timesteps=False,
)

GENERIC_FLOWMATCH_FIXED_PROFILE: Final = GenericFlowMatchProfileV1(
    schema_id=GENERIC_FLOWMATCH_PROFILE_SCHEMA_ID,
    schema_version=GENERIC_FLOWMATCH_PROFILE_SCHEMA_VERSION,
    profile_id="flowmatch.generic.fixed",
    profile_version="1",
    display_name="Generic FlowMatch Fixed Shift (Framework Reference)",
    evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
    primary_source_id=_DIFFUSERS_FLOWMATCH_FRAMEWORK.framework_id,
    shift_mode=GenericFlowMatchShiftMode.FIXED,
    prediction_type=PredictionType.FLOW_VELOCITY,
    sigma_domain=SigmaDomain.UNIT_FLOW,
    ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
    base_grid=_BASE_GRID,
    transform=TransformDeclaration(
        identifier="diffusers.direct_ratio",
        stage=TransformStage.PRIMARY_TIME_SHIFT,
        input_domain=SigmaDomain.UNIT_FLOW,
        output_domain=SigmaDomain.UNIT_FLOW,
        parameters=(ProfileField(name="ratio", value=1.0),),
    ),
    terminal=_TERMINAL,
    slicing=_SLICING,
    selection=_EXPLICIT_SELECTION,
    reference_sampler_capabilities=_DETERMINISTIC_EULER,
    framework=_DIFFUSERS_FLOWMATCH_FRAMEWORK,
    required_inputs=(),
    step_policy=_STEP_POLICY,
    guidance_policy=_GUIDANCE_POLICY,
    parameters=(ProfileField(name="use_dynamic_shifting", value=False),),
    known_limitations=(
        "Explicit selection does not establish compatibility with any model.",
        "Guidance and inference step recommendations remain model-specific and unset.",
    ),
)

GENERIC_FLOWMATCH_DYNAMIC_PROFILE: Final = GenericFlowMatchProfileV1(
    schema_id=GENERIC_FLOWMATCH_PROFILE_SCHEMA_ID,
    schema_version=GENERIC_FLOWMATCH_PROFILE_SCHEMA_VERSION,
    profile_id="flowmatch.generic.dynamic",
    profile_version="1",
    display_name="Generic FlowMatch Dynamic Shift (Experimental)",
    evidence=EvidenceLevel.EXPERIMENTAL,
    primary_source_id=_DIFFUSERS_FLOWMATCH_FRAMEWORK.framework_id,
    shift_mode=GenericFlowMatchShiftMode.DYNAMIC,
    prediction_type=PredictionType.FLOW_VELOCITY,
    sigma_domain=SigmaDomain.UNIT_FLOW,
    ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
    base_grid=_BASE_GRID,
    transform=TransformDeclaration(
        identifier="diffusers.exponential_mu",
        stage=TransformStage.PRIMARY_TIME_SHIFT,
        input_domain=SigmaDomain.UNIT_FLOW,
        output_domain=SigmaDomain.UNIT_FLOW,
        parameters=(
            ProfileField(name="exponent", value=1.0),
            ProfileField(name="mu", value=None),
            ProfileField(name="time_shift_type", value="exponential"),
        ),
    ),
    terminal=_TERMINAL,
    slicing=_SLICING,
    selection=_EXPLICIT_SELECTION,
    reference_sampler_capabilities=_DETERMINISTIC_EULER,
    framework=_DIFFUSERS_FLOWMATCH_FRAMEWORK,
    required_inputs=("mu",),
    step_policy=_STEP_POLICY,
    guidance_policy=_GUIDANCE_POLICY,
    parameters=(
        ProfileField(name="base_image_seq_len", value=256),
        ProfileField(name="base_shift", value=0.5),
        ProfileField(name="max_image_seq_len", value=4096),
        ProfileField(name="max_shift", value=1.15),
        ProfileField(name="use_dynamic_shifting", value=True),
    ),
    known_limitations=(
        "Explicit selection does not establish compatibility with any model.",
        "Guidance and inference step recommendations remain model-specific and unset.",
        "The caller must provide mu explicitly; no generic geometry derivation is valid.",
    ),
)

_GENERIC_FLOWMATCH_PROFILES: Final = (
    GENERIC_FLOWMATCH_DYNAMIC_PROFILE,
    GENERIC_FLOWMATCH_FIXED_PROFILE,
)


def generic_flowmatch_profiles() -> tuple[GenericFlowMatchProfileV1, ...]:
    """Return the immutable canonical catalog of explicit generic profiles."""

    return _GENERIC_FLOWMATCH_PROFILES


def resolve_generic_flowmatch_profile(
    profile_id: str,
    profile_version: str,
) -> GenericFlowMatchProfileV1:
    """Resolve one exact generic profile key without aliases or fallback."""

    for profile in _GENERIC_FLOWMATCH_PROFILES:
        if profile.profile_id == profile_id and profile.profile_version == profile_version:
            return profile
    raise ScheduleContractError(
        f"generic FlowMatch profile {profile_id}@{profile_version} is not registered"
    )


def _profile_field_projection(fields: tuple[ProfileField, ...]) -> dict[str, object]:
    def value_projection(value: object) -> object:
        if isinstance(value, float):
            return {"bits": float_to_ieee_hex(value, "float64"), "precision": "float64"}
        return value

    return {field.name: value_projection(field.value) for field in fields}


def generic_flowmatch_profile_projection(
    profile: GenericFlowMatchProfileV1,
) -> dict[str, object]:
    """Return a canonical typed projection for one generic profile."""

    if not isinstance(profile, GenericFlowMatchProfileV1):
        raise ScheduleContractError("generic FlowMatch projection requires a profile")
    sampler = profile.reference_sampler_capabilities
    framework = profile.framework
    return {
        "base_grid": {
            "identifier": profile.base_grid.identifier,
            "output_domain": profile.base_grid.output_domain.value,
            "parameters": _profile_field_projection(profile.base_grid.parameters),
            "terminal_included": profile.base_grid.terminal_included,
        },
        "display_name": profile.display_name,
        "evidence": profile.evidence.value,
        "framework": {
            "framework_id": framework.framework_id,
            "license": {
                "declaration_version": framework.license.declaration_version,
                "identifier": framework.license.identifier,
                "name": framework.license.name,
                "url": framework.license.url,
            },
            "locators": list(framework.locators),
            "record_version": framework.record_version,
            "resource_version": framework.resource_version,
            "revision": framework.revision,
            "url": framework.url,
        },
        "guidance_policy": profile.guidance_policy,
        "known_limitations": list(profile.known_limitations),
        "ownership": profile.ownership.value,
        "parameters": _profile_field_projection(profile.parameters),
        "prediction_type": profile.prediction_type.value,
        "primary_source_id": profile.primary_source_id,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "reference_sampler_capabilities": {
            "accepted_ownerships": [value.value for value in sampler.accepted_ownerships],
            "accepted_prediction_types": [
                value.value for value in sampler.accepted_prediction_types
            ],
            "accepted_sigma_domains": [value.value for value in sampler.accepted_sigma_domains],
            "execution_behavior": sampler.execution_behavior.value,
            "noise_ownership": sampler.noise_ownership.value,
            "required_state": [value.value for value in sampler.required_state],
            "sampler_id": sampler.sampler_id,
            "sampler_version": sampler.sampler_version,
            "supports_partial_denoise": sampler.supports_partial_denoise,
            "supports_per_token_timesteps": sampler.supports_per_token_timesteps,
            "terminal_requirement": sampler.terminal_requirement.value,
        },
        "required_inputs": list(profile.required_inputs),
        "schema": profile.schema_id,
        "schema_version": profile.schema_version,
        "selection": {
            "ambiguity_requires_explicit": profile.selection.ambiguity_requires_explicit,
            "family_only_sources": list(profile.selection.family_only_sources),
            "resolving_sources": list(profile.selection.resolving_sources),
            "strategy_id": profile.selection.strategy_id,
            "strict_default": profile.selection.strict_default,
            "suggestion_sources": list(profile.selection.suggestion_sources),
        },
        "shift_mode": profile.shift_mode.value,
        "sigma_domain": profile.sigma_domain.value,
        "slicing": {
            "supports_denoise_tail": profile.slicing.supports_denoise_tail,
            "supports_step_range": profile.slicing.supports_step_range,
            "zero_denoise_is_empty": profile.slicing.zero_denoise_is_empty,
        },
        "step_policy": profile.step_policy,
        "terminal": {
            "policy": profile.terminal.policy.value,
            "sigma": profile.terminal.sigma.value,
            "value": {
                "bits": float_to_ieee_hex(profile.terminal.value, "float64"),
                "precision": "float64",
            },
        },
        "transform": {
            "identifier": profile.transform.identifier,
            "input_domain": profile.transform.input_domain.value,
            "output_domain": profile.transform.output_domain.value,
            "parameters": _profile_field_projection(profile.transform.parameters),
            "stage": profile.transform.stage.value,
        },
    }


def generic_flowmatch_profile_fingerprint(profile: GenericFlowMatchProfileV1) -> str:
    """Hash the canonical generic-profile projection."""

    payload = json.dumps(
        generic_flowmatch_profile_projection(profile),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
