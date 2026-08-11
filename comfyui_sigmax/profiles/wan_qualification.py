"""Pinned readiness contracts for planned Wan profiles.

This module records source identity and ownership without constructing schedules, registering
profiles, or changing ComfyUI behavior. Runtime support belongs to M6-10, M6-11, and M4-14.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final, NoReturn

from comfyui_sigmax.core import ScheduleContractError

WAN_QUALIFICATION_SCHEMA_ID: Final = "sigmax.wan-qualification/1"

_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")


class WanSourceLane(str, Enum):
    """Evidence owner for one planned recipe or host observation."""

    COMFYUI_MODEL_NATIVE = "comfyui_model_native"
    COMFYUI_WORKFLOW = "comfyui_workflow"
    FRAMEWORK_REFERENCE = "framework_reference"
    OFFICIAL_NATIVE = "official_native"


class WanSourceScope(str, Enum):
    """License boundary represented by a source pin."""

    SOFTWARE = "software"
    MODEL_WEIGHTS = "model_weights"


class WanReadiness(str, Enum):
    """M6-09 never promotes a planned identity to supported runtime behavior."""

    PLANNED = "planned"


class WanQualificationError(ScheduleContractError):
    """Fail-closed Wan identity error with a stable machine-readable reason."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class WanQualificationSource:
    """One immutable software-repository or model-card evidence pin."""

    source_id: str
    scope: WanSourceScope
    url: str
    revision: str
    license_id: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_id or self.source_id.strip() != self.source_id:
            raise ScheduleContractError("Wan qualification source_id is invalid")
        if not isinstance(self.scope, WanSourceScope):
            raise ScheduleContractError("Wan qualification source scope is invalid")
        if not self.url.startswith("https://"):
            raise ScheduleContractError("Wan qualification source URL must use HTTPS")
        if not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("Wan qualification source revision is invalid")
        if self.license_id not in {"Apache-2.0", "GPL-3.0-only"}:
            raise ScheduleContractError("Wan qualification source license is unsupported")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("Wan qualification locators must be sorted and unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class WanPlannedProfile:
    """Source-qualified identity awaiting its owning runtime implementation item."""

    profile_id: str
    family: str
    task: str
    variant: str
    source_lane: WanSourceLane
    resolution: str
    shift: float
    steps: int
    guidance: float | None
    guidance_mode: str
    solver: str | None
    allowed_solvers: tuple[str, ...]
    software_source_id: str
    model_card_source_id: str
    model_card_id: str
    model_card_revision: str
    implementation_item: str
    recipe_blockers: tuple[str, ...] = ()
    readiness: WanReadiness = WanReadiness.PLANNED
    runtime_registered: bool = False

    def __post_init__(self) -> None:
        if not self.profile_id or self.profile_id.strip() != self.profile_id:
            raise ScheduleContractError("Wan planned profile_id is invalid")
        if self.family not in {"wan2.1", "wan2.2", "wan-animate2"}:
            raise ScheduleContractError("Wan planned family is unsupported")
        if self.task not in {"animate", "flf2v", "s2v", "vace"}:
            raise ScheduleContractError("Wan planned task is unsupported")
        if not isinstance(self.source_lane, WanSourceLane):
            raise ScheduleContractError("Wan planned source lane is invalid")
        if self.resolution not in {"none", "720p"}:
            raise ScheduleContractError("Wan planned resolution is unsupported")
        if not math.isfinite(self.shift) or self.shift <= 0.0:
            raise ScheduleContractError("Wan planned shift must be finite and positive")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 0:
            raise ScheduleContractError("Wan planned steps must be positive")
        if self.guidance is not None and (not math.isfinite(self.guidance) or self.guidance < 0.0):
            raise ScheduleContractError("Wan planned guidance must be finite and non-negative")
        if self.guidance_mode not in {
            "cfg",
            "example_override",
            "framework_default",
            "no_cfg",
        }:
            raise ScheduleContractError("Wan planned guidance mode is unsupported")
        if not self.allowed_solvers or self.allowed_solvers != tuple(
            sorted(set(self.allowed_solvers))
        ):
            raise ScheduleContractError("Wan planned allowed solvers must be sorted and unique")
        if self.solver is not None and self.solver not in self.allowed_solvers:
            raise ScheduleContractError("Wan planned solver is outside the allowed set")
        if not _COMMIT_PATTERN.fullmatch(self.model_card_revision):
            raise ScheduleContractError("Wan planned model-card revision is invalid")
        if self.implementation_item not in {"M6-10", "M6-11"}:
            raise ScheduleContractError("Wan planned implementation item is unsupported")
        if self.recipe_blockers != tuple(sorted(set(self.recipe_blockers))):
            raise ScheduleContractError("Wan planned recipe blockers must be sorted and unique")
        if self.readiness is not WanReadiness.PLANNED or self.runtime_registered is not False:
            raise ScheduleContractError("M6-09 cannot register a Wan runtime profile")


@dataclass(frozen=True, slots=True, kw_only=True)
class WanComfyUIObservation:
    """Pinned current-host behavior that must not be relabeled official-native."""

    observation_id: str
    source_lane: WanSourceLane
    shift: float
    locator: str
    owner: str
    inheritance: str | None = None
    workflow: str | None = None
    official_native: bool = False

    def __post_init__(self) -> None:
        if self.source_lane not in {
            WanSourceLane.COMFYUI_MODEL_NATIVE,
            WanSourceLane.COMFYUI_WORKFLOW,
        }:
            raise ScheduleContractError("Wan ComfyUI observation lane is invalid")
        if not math.isfinite(self.shift) or self.shift <= 0.0:
            raise ScheduleContractError("Wan ComfyUI observation shift is invalid")
        if not self.locator or not self.owner:
            raise ScheduleContractError("Wan ComfyUI observation locator/owner is required")
        if self.official_native is not False:
            raise ScheduleContractError("ComfyUI observations cannot be official-native")


@dataclass(frozen=True, slots=True, kw_only=True)
class WanConflictRule:
    """One stable fail-closed qualification rule."""

    reason_code: str
    outcome: str
    description: str

    def __post_init__(self) -> None:
        if not self.reason_code.startswith("wan.identity."):
            raise ScheduleContractError("Wan conflict reason code is invalid")
        if self.outcome != "reject" or not self.description:
            raise ScheduleContractError("Wan conflict rule must reject with a description")


def _source(
    source_id: str,
    scope: WanSourceScope,
    url: str,
    revision: str,
    license_id: str,
    *locators: str,
) -> WanQualificationSource:
    return WanQualificationSource(
        source_id=source_id,
        scope=scope,
        url=url,
        revision=revision,
        license_id=license_id,
        locators=tuple(sorted(locators)),
    )


WAN_QUALIFICATION_SOURCES: Final = tuple(
    sorted(
        (
            _source(
                "comfyui.repository",
                WanSourceScope.SOFTWARE,
                "https://github.com/Comfy-Org/ComfyUI",
                "2a68ce33b4c9ea6ee4283e618a74560cefb32694",  # pragma: allowlist secret
                "GPL-3.0-only",
                "blueprints/Image to Video (Wan 2.2).json",
                "blueprints/Text to Video (Wan 2.2).json",
                "blueprints/Video Inpainting (Wan2.1 VACE).json",
                "comfy/supported_models.py",
                "comfy_extras/nodes_model_advanced.py",
            ),
            _source(
                "wan2.1.repository",
                WanSourceScope.SOFTWARE,
                "https://github.com/Wan-Video/Wan2.1",
                "9737cba9c1c3c4d04b33fcad41c111989865d315",  # pragma: allowlist secret
                "Apache-2.0",
                "LICENSE",
                "README.md",
                "generate.py",
            ),
            _source(
                "wan2.2.repository",
                WanSourceScope.SOFTWARE,
                "https://github.com/Wan-Video/Wan2.2",
                "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",  # pragma: allowlist secret
                "Apache-2.0",
                "LICENSE",
                "README.md",
                "generate.py",
                "wan/animate.py",
                "wan/configs/wan_animate_14B.py",
                "wan/configs/wan_s2v_14B.py",
            ),
            _source(
                "wan-animate2.repository",
                WanSourceScope.SOFTWARE,
                "https://github.com/Wan-Video/Wan-Animate-2",
                "3ad2fef7d61d6200c9c653e0fe47be7616b323f3",  # pragma: allowlist secret
                "Apache-2.0",
                "LICENSE",
                "README.md",
                "infer/wan_animate_2.yaml",
                "infer/wan_animate_2_demo.py",
                "infer/wan_animate_2_distillation.yaml",
                "pipelines/wan_animate_2_pipeline.py",
            ),
            _source(
                "wan2.1.flf2v.14b.720p.card",
                WanSourceScope.MODEL_WEIGHTS,
                "https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P",
                "c8db168d95d3ebeb63430b3b6d264885cb8a0df3",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
            ),
            _source(
                "wan2.1.vace.1.3b.card",
                WanSourceScope.MODEL_WEIGHTS,
                "https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B",
                "574e6a744642ce3bee319afc31496b88bde8aac4",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
            ),
            _source(
                "wan2.1.vace.14b.card",
                WanSourceScope.MODEL_WEIGHTS,
                "https://huggingface.co/Wan-AI/Wan2.1-VACE-14B",
                "539c162b1387eac9dc4c20bd3f74671309e76a4c",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
            ),
            _source(
                "wan2.2.s2v.14b.card",
                WanSourceScope.MODEL_WEIGHTS,
                "https://huggingface.co/Wan-AI/Wan2.2-S2V-14B",
                "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
            ),
            _source(
                "wan2.2.animate.14b.card",
                WanSourceScope.MODEL_WEIGHTS,
                "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B",
                "cb93a225fbaf1ca100f54e79da8f994995b689b3",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
            ),
            _source(
                "wan-animate2.14b.native.card",
                WanSourceScope.MODEL_WEIGHTS,
                "https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B",
                "6e8f1973bf0abc2aafd517992e8b6d88c3c46e69",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
            ),
            _source(
                "wan-animate2.14b.diffusers.card",
                WanSourceScope.MODEL_WEIGHTS,
                "https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
                "a84c891208322be6ea1130b1db95df1baedb0459",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
                "model_index.json",
                "scheduler/scheduler_config.json",
            ),
            _source(
                "wan-animate2.14b.distilled.diffusers.card",
                WanSourceScope.MODEL_WEIGHTS,
                "https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
                "36b185201c469c756601cb0779f6597dda1d6c01",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
                "model_index.json",
                "scheduler/scheduler_config.json",
            ),
        ),
        key=lambda item: item.source_id,
    )
)

_SOURCES_BY_ID: Final = {source.source_id: source for source in WAN_QUALIFICATION_SOURCES}


def _planned(
    *,
    profile_id: str,
    family: str,
    task: str,
    variant: str,
    source_lane: WanSourceLane,
    resolution: str,
    shift: float,
    steps: int,
    guidance: float | None,
    guidance_mode: str,
    solver: str | None,
    allowed_solvers: tuple[str, ...],
    software_source_id: str,
    model_card_source_id: str,
    model_card_id: str,
    implementation_item: str,
    recipe_blockers: tuple[str, ...] = (),
) -> WanPlannedProfile:
    card = _SOURCES_BY_ID[model_card_source_id]
    return WanPlannedProfile(
        profile_id=profile_id,
        family=family,
        task=task,
        variant=variant,
        source_lane=source_lane,
        resolution=resolution,
        shift=shift,
        steps=steps,
        guidance=guidance,
        guidance_mode=guidance_mode,
        solver=solver,
        allowed_solvers=allowed_solvers,
        software_source_id=software_source_id,
        model_card_source_id=model_card_source_id,
        model_card_id=model_card_id,
        model_card_revision=card.revision,
        implementation_item=implementation_item,
        recipe_blockers=recipe_blockers,
    )


WAN_PLANNED_PROFILES: Final = tuple(
    sorted(
        (
            _planned(
                profile_id="wan2.1.flf2v.14b.720p.official-native",
                family="wan2.1",
                task="flf2v",
                variant="14b",
                source_lane=WanSourceLane.OFFICIAL_NATIVE,
                resolution="720p",
                shift=16.0,
                steps=50,
                guidance=5.0,
                guidance_mode="cfg",
                solver="unipc",
                allowed_solvers=("dpm++", "unipc"),
                software_source_id="wan2.1.repository",
                model_card_source_id="wan2.1.flf2v.14b.720p.card",
                model_card_id="Wan-AI/Wan2.1-FLF2V-14B-720P",
                implementation_item="M6-10",
            ),
            _planned(
                profile_id="wan2.1.vace.1.3b.official-native",
                family="wan2.1",
                task="vace",
                variant="1.3b",
                source_lane=WanSourceLane.OFFICIAL_NATIVE,
                resolution="none",
                shift=16.0,
                steps=50,
                guidance=5.0,
                guidance_mode="cfg",
                solver="unipc",
                allowed_solvers=("dpm++", "unipc"),
                software_source_id="wan2.1.repository",
                model_card_source_id="wan2.1.vace.1.3b.card",
                model_card_id="Wan-AI/Wan2.1-VACE-1.3B",
                implementation_item="M6-10",
            ),
            _planned(
                profile_id="wan2.1.vace.14b.official-native",
                family="wan2.1",
                task="vace",
                variant="14b",
                source_lane=WanSourceLane.OFFICIAL_NATIVE,
                resolution="none",
                shift=16.0,
                steps=50,
                guidance=5.0,
                guidance_mode="cfg",
                solver="unipc",
                allowed_solvers=("dpm++", "unipc"),
                software_source_id="wan2.1.repository",
                model_card_source_id="wan2.1.vace.14b.card",
                model_card_id="Wan-AI/Wan2.1-VACE-14B",
                implementation_item="M6-10",
            ),
            _planned(
                profile_id="wan2.2.s2v.14b.official-native",
                family="wan2.2",
                task="s2v",
                variant="14b",
                source_lane=WanSourceLane.OFFICIAL_NATIVE,
                resolution="none",
                shift=3.0,
                steps=40,
                guidance=4.5,
                guidance_mode="cfg",
                solver="unipc",
                allowed_solvers=("dpm++", "unipc"),
                software_source_id="wan2.2.repository",
                model_card_source_id="wan2.2.s2v.14b.card",
                model_card_id="Wan-AI/Wan2.2-S2V-14B",
                implementation_item="M6-10",
            ),
            _planned(
                profile_id="wan2.2.animate.14b.official-native",
                family="wan2.2",
                task="animate",
                variant="14b",
                source_lane=WanSourceLane.OFFICIAL_NATIVE,
                resolution="none",
                shift=5.0,
                steps=20,
                guidance=1.0,
                guidance_mode="no_cfg",
                solver="unipc",
                allowed_solvers=("dpm++", "unipc"),
                software_source_id="wan2.2.repository",
                model_card_source_id="wan2.2.animate.14b.card",
                model_card_id="Wan-AI/Wan2.2-Animate-14B",
                implementation_item="M6-11",
            ),
            _planned(
                profile_id="wan-animate2.14b.base.official-native",
                family="wan-animate2",
                task="animate",
                variant="base-14b",
                source_lane=WanSourceLane.OFFICIAL_NATIVE,
                resolution="none",
                shift=5.0,
                steps=40,
                guidance=3.0,
                guidance_mode="cfg",
                solver="flow_dpm",
                allowed_solvers=("flow_dpm",),
                software_source_id="wan-animate2.repository",
                model_card_source_id="wan-animate2.14b.native.card",
                model_card_id="Wan-AI/Wan2.2-Animate-2-14B",
                implementation_item="M6-11",
            ),
            _planned(
                profile_id="wan-animate2.14b.distilled.official-native",
                family="wan-animate2",
                task="animate",
                variant="distilled-14b",
                source_lane=WanSourceLane.OFFICIAL_NATIVE,
                resolution="none",
                shift=5.0,
                steps=10,
                guidance=1.0,
                guidance_mode="no_cfg",
                solver="flow_dpm",
                allowed_solvers=("flow_dpm",),
                software_source_id="wan-animate2.repository",
                model_card_source_id="wan-animate2.14b.native.card",
                model_card_id="Wan-AI/Wan2.2-Animate-2-14B",
                implementation_item="M6-11",
            ),
            _planned(
                profile_id="wan-animate2.14b.base.diffusers-reference",
                family="wan-animate2",
                task="animate",
                variant="base-14b",
                source_lane=WanSourceLane.FRAMEWORK_REFERENCE,
                resolution="none",
                shift=5.0,
                steps=40,
                guidance=None,
                guidance_mode="framework_default",
                solver=None,
                allowed_solvers=("dpm_solver", "unipc"),
                software_source_id="wan-animate2.repository",
                model_card_source_id="wan-animate2.14b.diffusers.card",
                model_card_id="Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
                implementation_item="M6-11",
                recipe_blockers=(
                    "framework_release_unpinned",
                    "framework_schedule_ownership_unresolved",
                    "framework_support_pr_unmerged",
                    "scheduler_metadata_conflict",
                ),
            ),
            _planned(
                profile_id="wan-animate2.14b.distilled.diffusers-reference",
                family="wan-animate2",
                task="animate",
                variant="distilled-14b",
                source_lane=WanSourceLane.FRAMEWORK_REFERENCE,
                resolution="none",
                shift=5.0,
                steps=10,
                guidance=1.0,
                guidance_mode="no_cfg",
                solver="euler",
                allowed_solvers=("euler",),
                software_source_id="wan-animate2.repository",
                model_card_source_id="wan-animate2.14b.distilled.diffusers.card",
                model_card_id="Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
                implementation_item="M6-11",
                recipe_blockers=(
                    "framework_release_unpinned",
                    "framework_schedule_ownership_unresolved",
                    "framework_support_pr_unmerged",
                ),
            ),
        ),
        key=lambda item: item.profile_id,
    )
)


def _observation(
    observation_id: str,
    source_lane: WanSourceLane,
    shift: float,
    locator: str,
    owner: str,
    *,
    inheritance: str | None = None,
    workflow: str | None = None,
) -> WanComfyUIObservation:
    return WanComfyUIObservation(
        observation_id=observation_id,
        source_lane=source_lane,
        shift=shift,
        locator=locator,
        owner=owner,
        inheritance=inheritance,
        workflow=workflow,
    )


WAN_COMFYUI_OBSERVATIONS: Final = tuple(
    sorted(
        (
            _observation(
                "comfyui.wan21-t2v.model-default",
                WanSourceLane.COMFYUI_MODEL_NATIVE,
                8.0,
                "comfy/supported_models.py:WAN21_T2V.sampling_settings",
                "WAN21_T2V",
            ),
            _observation(
                "comfyui.wan21-i2v.inherited-default",
                WanSourceLane.COMFYUI_MODEL_NATIVE,
                8.0,
                "comfy/supported_models.py:WAN21_I2V",
                "WAN21_I2V",
                inheritance="WAN21_T2V",
            ),
            _observation(
                "comfyui.wan21-vace.inherited-default",
                WanSourceLane.COMFYUI_MODEL_NATIVE,
                8.0,
                "comfy/supported_models.py:WAN21_Vace",
                "WAN21_Vace",
                inheritance="WAN21_T2V",
            ),
            _observation(
                "comfyui.wan22-s2v.inherited-default",
                WanSourceLane.COMFYUI_MODEL_NATIVE,
                8.0,
                "comfy/supported_models.py:WAN22_S2V",
                "WAN22_S2V",
                inheritance="WAN21_T2V",
            ),
            _observation(
                "comfyui.wan22-animate.inherited-default",
                WanSourceLane.COMFYUI_MODEL_NATIVE,
                8.0,
                "comfy/supported_models.py:WAN22_Animate",
                "WAN22_Animate",
                inheritance="WAN21_T2V",
            ),
            _observation(
                "comfyui.wan-animate2.model-default",
                WanSourceLane.COMFYUI_MODEL_NATIVE,
                5.0,
                "comfy/supported_models.py:WAN_Animate2.sampling_settings",
                "WAN_Animate2",
                inheritance="WAN21_T2V",
            ),
            _observation(
                "comfyui.wan22-i2v.workflow-patch",
                WanSourceLane.COMFYUI_WORKFLOW,
                5.0,
                "definitions.subgraphs[*].nodes[type=ModelSamplingSD3]",
                "ModelSamplingSD3",
                workflow="blueprints/Image to Video (Wan 2.2).json",
            ),
            _observation(
                "comfyui.wan22-t2v.workflow-patch",
                WanSourceLane.COMFYUI_WORKFLOW,
                5.0,
                "definitions.subgraphs[*].nodes[type=ModelSamplingSD3]",
                "ModelSamplingSD3",
                workflow="blueprints/Text to Video (Wan 2.2).json",
            ),
            _observation(
                "comfyui.wan21-vace.workflow-patch",
                WanSourceLane.COMFYUI_WORKFLOW,
                5.0,
                "definitions.subgraphs[*].nodes[type=ModelSamplingSD3]",
                "ModelSamplingSD3",
                workflow="blueprints/Video Inpainting (Wan2.1 VACE).json",
            ),
        ),
        key=lambda item: item.observation_id,
    )
)

WAN_CONFLICT_RULES: Final = tuple(
    sorted(
        (
            WanConflictRule(
                reason_code="wan.identity.card_revision_mismatch",
                outcome="reject",
                description="A mutable or mismatched model-card revision cannot identify a profile.",
            ),
            WanConflictRule(
                reason_code="wan.identity.conflict",
                outcome="reject",
                description="Conflicting exact profile, card, or source-lane evidence is ambiguous.",
            ),
            WanConflictRule(
                reason_code="wan.identity.incomplete_recipe",
                outcome="reject",
                description="Unresolved solver, guidance, or framework ownership blocks promotion.",
            ),
            WanConflictRule(
                reason_code="wan.identity.runtime_not_implemented",
                outcome="reject",
                description="M6-09 readiness metadata never authorizes runtime support.",
            ),
            WanConflictRule(
                reason_code="wan.identity.unsupported",
                outcome="reject",
                description="Unknown exact identities and partial axis requests fail closed.",
            ),
            WanConflictRule(
                reason_code="wan.identity.weak_signal_only",
                outcome="reject",
                description="Names and family-only signals cannot promote a planned profile.",
            ),
        ),
        key=lambda item: item.reason_code,
    )
)

_PROFILES_BY_ID: Final = {profile.profile_id: profile for profile in WAN_PLANNED_PROFILES}


def _raise(reason_code: str, message: str) -> NoReturn:
    raise WanQualificationError(reason_code, message)


def _profile_from_card(*, model_card_id: str, model_card_revision: str) -> WanPlannedProfile:
    card_matches = tuple(
        profile for profile in WAN_PLANNED_PROFILES if profile.model_card_id == model_card_id
    )
    if not card_matches:
        _raise("wan.identity.unsupported", "model card is not a planned Wan identity")
    revision_matches = tuple(
        profile for profile in card_matches if profile.model_card_revision == model_card_revision
    )
    if not revision_matches:
        _raise(
            "wan.identity.card_revision_mismatch",
            "model-card revision does not match the pinned Wan identity",
        )
    if len(revision_matches) != 1:
        _raise(
            "wan.identity.conflict",
            "model card maps to multiple planned Wan identities; profile_id is required",
        )
    return revision_matches[0]


def qualify_planned_wan_identity(
    *,
    profile_id: str | None = None,
    model_card_id: str | None = None,
    model_card_revision: str | None = None,
    source_lane: WanSourceLane | str | None = None,
    weak_name: str | None = None,
    require_recipe_complete: bool = False,
    require_runtime: bool = False,
) -> WanPlannedProfile:
    """Resolve exact readiness evidence while refusing runtime promotion and weak inference."""

    if not isinstance(require_recipe_complete, bool) or not isinstance(require_runtime, bool):
        _raise("wan.identity.unsupported", "qualification flags must be boolean")
    explicit = _PROFILES_BY_ID.get(profile_id) if isinstance(profile_id, str) else None
    if profile_id is not None and explicit is None:
        _raise("wan.identity.unsupported", "profile_id is not a planned Wan identity")

    card: WanPlannedProfile | None = None
    if model_card_id is not None or model_card_revision is not None:
        if not isinstance(model_card_id, str) or not isinstance(model_card_revision, str):
            _raise(
                "wan.identity.card_revision_mismatch",
                "model-card ID and immutable revision are both required",
            )
        card = _profile_from_card(
            model_card_id=model_card_id, model_card_revision=model_card_revision
        )

    if explicit is not None and card is not None and explicit.profile_id != card.profile_id:
        _raise("wan.identity.conflict", "explicit profile and model-card evidence disagree")
    selected = explicit if explicit is not None else card
    if selected is None:
        if weak_name is not None:
            _raise("wan.identity.weak_signal_only", "weak model-name evidence cannot select Wan")
        _raise("wan.identity.unsupported", "an exact planned Wan identity is required")

    if source_lane is not None:
        try:
            selected_lane = (
                source_lane
                if isinstance(source_lane, WanSourceLane)
                else WanSourceLane(source_lane)
            )
        except (TypeError, ValueError):
            _raise("wan.identity.unsupported", "source lane is unsupported")
        if selected_lane is not selected.source_lane:
            _raise("wan.identity.conflict", "source lane conflicts with the exact Wan identity")
    if require_recipe_complete and selected.recipe_blockers:
        _raise("wan.identity.incomplete_recipe", "planned Wan recipe still has blocking evidence")
    if require_runtime:
        _raise(
            "wan.identity.runtime_not_implemented", "planned Wan runtime support is not implemented"
        )
    return selected


def _source_projection(source: WanQualificationSource) -> dict[str, object]:
    return {
        "license_id": source.license_id,
        "locators": list(source.locators),
        "revision": source.revision,
        "scope": source.scope.value,
        "source_id": source.source_id,
        "url": source.url,
    }


def _profile_projection(profile: WanPlannedProfile) -> dict[str, object]:
    return {
        "allowed_solvers": list(profile.allowed_solvers),
        "family": profile.family,
        "guidance": profile.guidance,
        "guidance_mode": profile.guidance_mode,
        "implementation_item": profile.implementation_item,
        "model_card_id": profile.model_card_id,
        "model_card_revision": profile.model_card_revision,
        "model_card_source_id": profile.model_card_source_id,
        "profile_id": profile.profile_id,
        "readiness": profile.readiness.value,
        "recipe_blockers": list(profile.recipe_blockers),
        "resolution": profile.resolution,
        "runtime_registered": profile.runtime_registered,
        "shift": profile.shift,
        "software_source_id": profile.software_source_id,
        "solver": profile.solver,
        "source_lane": profile.source_lane.value,
        "steps": profile.steps,
        "task": profile.task,
        "variant": profile.variant,
    }


def _observation_projection(observation: WanComfyUIObservation) -> dict[str, object]:
    return {
        "inheritance": observation.inheritance,
        "locator": observation.locator,
        "observation_id": observation.observation_id,
        "official_native": observation.official_native,
        "owner": observation.owner,
        "shift": observation.shift,
        "source_lane": observation.source_lane.value,
        "workflow": observation.workflow,
    }


def serialize_wan_qualification() -> dict[str, object]:
    """Return the canonical M6-09 readiness envelope."""

    return {
        "comfyui_observations": [
            _observation_projection(observation) for observation in WAN_COMFYUI_OBSERVATIONS
        ],
        "conflict_rules": [
            {
                "description": rule.description,
                "outcome": rule.outcome,
                "reason_code": rule.reason_code,
            }
            for rule in WAN_CONFLICT_RULES
        ],
        "planned_profiles": [_profile_projection(profile) for profile in WAN_PLANNED_PROFILES],
        "runtime_registration": False,
        "schema": WAN_QUALIFICATION_SCHEMA_ID,
        "sources": [_source_projection(source) for source in WAN_QUALIFICATION_SOURCES],
    }


def wan_qualification_fingerprint() -> str:
    """Fingerprint the canonical source/identity/ownership envelope."""

    encoded = json.dumps(
        serialize_wan_qualification(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "WAN_COMFYUI_OBSERVATIONS",
    "WAN_CONFLICT_RULES",
    "WAN_PLANNED_PROFILES",
    "WAN_QUALIFICATION_SCHEMA_ID",
    "WAN_QUALIFICATION_SOURCES",
    "WanComfyUIObservation",
    "WanConflictRule",
    "WanPlannedProfile",
    "WanQualificationError",
    "WanQualificationSource",
    "WanReadiness",
    "WanSourceLane",
    "WanSourceScope",
    "qualify_planned_wan_identity",
    "serialize_wan_qualification",
    "wan_qualification_fingerprint",
]
