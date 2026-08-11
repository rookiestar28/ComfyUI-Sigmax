"""Immutable namespaced registry for complete profile-schema declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from enum import Enum
from typing import Final, cast

from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError
from comfyui_sigmax.profiles.schema_v1 import (
    ProfileSchemaV1,
    profile_schema_fingerprint,
)

_PROFILE_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)+$")
_PROFILE_VERSION_PATTERN: Final = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_FINGERPRINT_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMPLICIT_INHERITANCE_FIELDS: Final = frozenset(
    {"schema_id", "schema_version", "profile_id", "profile_version"}
)
_OVERRIDABLE_FIELDS: Final = frozenset(
    field.name
    for field in fields(ProfileSchemaV1)
    if field.name not in _IMPLICIT_INHERITANCE_FIELDS
)


class ProfileOrigin(str, Enum):
    """Trust boundary for one registered profile."""

    BUILTIN = "builtin"
    EXTERNAL = "external"


class ConflictPolicy(str, Enum):
    """Explicit behavior when an exact profile key already exists."""

    REJECT = "reject"
    REPLACE_EXTERNAL = "replace_external"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileKey:
    """Exact namespaced profile identity with no fuzzy version selection."""

    profile_id: str
    profile_version: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.profile_id, str)
            or not self.profile_id.isascii()
            or not _PROFILE_ID_PATTERN.fullmatch(self.profile_id)
        ):
            raise ScheduleContractError("profile_id must be an exact namespaced identifier")
        if not isinstance(self.profile_version, str) or not _PROFILE_VERSION_PATTERN.fullmatch(
            self.profile_version
        ):
            raise ScheduleContractError("profile_version must be an exact numeric version")

    @classmethod
    def from_schema(cls, schema: ProfileSchemaV1) -> ProfileKey:
        if not isinstance(schema, ProfileSchemaV1):
            raise ScheduleContractError("profile key requires ProfileSchemaV1")
        return cls(
            profile_id=schema.profile_id,
            profile_version=schema.profile_version,
        )

    @property
    def canonical(self) -> str:
        return f"{self.profile_id}@{self.profile_version}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileInheritance:
    """Explicit parent and complete top-level semantic-difference declaration."""

    parent: ProfileKey
    overridden_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parent, ProfileKey):
            raise ScheduleContractError("inheritance parent must be a ProfileKey")
        if not isinstance(self.overridden_fields, tuple) or not self.overridden_fields:
            raise ScheduleContractError("overridden_fields must be a non-empty tuple")
        if not all(isinstance(name, str) for name in self.overridden_fields):
            raise ScheduleContractError("overridden_fields must contain strings")
        if len(self.overridden_fields) != len(set(self.overridden_fields)):
            raise ScheduleContractError("overridden_fields contains duplicates")
        if self.overridden_fields != tuple(sorted(self.overridden_fields)):
            raise ScheduleContractError("overridden_fields must use canonical order")
        if any(name not in _OVERRIDABLE_FIELDS for name in self.overridden_fields):
            raise ScheduleContractError("overridden_fields contains a protected or unknown field")


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisteredProfile:
    """One validated registry entry."""

    key: ProfileKey
    schema: ProfileSchemaV1
    fingerprint: str
    origin: ProfileOrigin
    inheritance: ProfileInheritance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, ProfileKey):
            raise ScheduleContractError("registered profile key must be ProfileKey")
        if not isinstance(self.schema, ProfileSchemaV1):
            raise ScheduleContractError("registered profile schema must be ProfileSchemaV1")
        if self.key != ProfileKey.from_schema(self.schema):
            raise ScheduleContractError("registered profile key does not match schema identity")
        if (
            not isinstance(self.fingerprint, str)
            or not _FINGERPRINT_PATTERN.fullmatch(self.fingerprint)
            or self.fingerprint != profile_schema_fingerprint(self.schema)
        ):
            raise ScheduleContractError("registered profile fingerprint is invalid")
        if not isinstance(self.origin, ProfileOrigin):
            raise ScheduleContractError("registered profile origin is invalid")
        if self.inheritance is not None and not isinstance(
            self.inheritance,
            ProfileInheritance,
        ):
            raise ScheduleContractError("registered inheritance is invalid")
        if self.origin is ProfileOrigin.BUILTIN and self.inheritance is not None:
            raise ScheduleContractError("built-in profiles cannot declare external inheritance")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileRegistry:
    """Canonical copy-on-write snapshot of registered complete profiles."""

    entries: tuple[RegisteredProfile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, RegisteredProfile) for entry in self.entries
        ):
            raise ScheduleContractError("registry entries must contain RegisteredProfile values")
        canonical_keys = tuple(entry.key.canonical for entry in self.entries)
        if canonical_keys != tuple(sorted(canonical_keys)):
            raise ScheduleContractError("registry entries must use canonical key order")
        if len(canonical_keys) != len(set(canonical_keys)):
            raise ScheduleContractError("registry contains duplicate profile keys")

    @classmethod
    def empty(cls) -> ProfileRegistry:
        return cls(entries=())

    def resolve(self, key: ProfileKey) -> RegisteredProfile:
        if not isinstance(key, ProfileKey):
            raise ScheduleContractError("profile lookup requires an exact ProfileKey")
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise ScheduleContractError(f"profile {key.canonical} is not registered")

    def register_external(
        self,
        schema: ProfileSchemaV1,
        *,
        inheritance: ProfileInheritance | None = None,
        conflict_policy: ConflictPolicy = ConflictPolicy.REJECT,
        expected_fingerprint: str | None = None,
    ) -> ProfileRegistry:
        if not isinstance(schema, ProfileSchemaV1):
            raise ScheduleContractError("external registration requires ProfileSchemaV1")
        if inheritance is not None and not isinstance(inheritance, ProfileInheritance):
            raise ScheduleContractError("inheritance must be ProfileInheritance")
        if not isinstance(conflict_policy, ConflictPolicy):
            raise ScheduleContractError("conflict_policy is invalid")
        if conflict_policy is ConflictPolicy.REJECT and expected_fingerprint is not None:
            raise ScheduleContractError("expected_fingerprint is only valid with REPLACE_EXTERNAL")

        key = ProfileKey.from_schema(schema)
        self._validate_inheritance(key, schema, inheritance)
        candidate = RegisteredProfile(
            key=key,
            schema=schema,
            fingerprint=profile_schema_fingerprint(schema),
            origin=ProfileOrigin.EXTERNAL,
            inheritance=inheritance,
        )
        existing_index = self._index_of(key)
        if existing_index is None:
            if conflict_policy is not ConflictPolicy.REJECT:
                raise ScheduleContractError(
                    "replacement policy requires an existing external profile"
                )
            return self._with_entry(candidate)

        existing = self.entries[existing_index]
        # CRITICAL: external callers can never replace a trusted built-in, even with identical bytes.
        if existing.origin is ProfileOrigin.BUILTIN:
            raise ScheduleContractError("external registration cannot replace a built-in profile")
        if (
            existing.fingerprint == candidate.fingerprint
            and existing.inheritance == candidate.inheritance
        ):
            return self
        if conflict_policy is ConflictPolicy.REJECT:
            raise ScheduleContractError(f"profile conflict for {key.canonical}")
        if (
            not isinstance(expected_fingerprint, str)
            or not _FINGERPRINT_PATTERN.fullmatch(expected_fingerprint)
            or expected_fingerprint != existing.fingerprint
        ):
            raise ScheduleContractError("expected external fingerprint does not match")

        updated = list(self.entries)
        updated[existing_index] = candidate
        return type(self)(entries=tuple(updated))

    def _validate_inheritance(
        self,
        child_key: ProfileKey,
        child_schema: ProfileSchemaV1,
        inheritance: ProfileInheritance | None,
    ) -> None:
        if inheritance is None:
            return
        if inheritance.parent == child_key:
            raise ScheduleContractError("profile cannot inherit from itself")
        try:
            parent = self.resolve(inheritance.parent)
        except ScheduleContractError as error:
            raise ScheduleContractError("inheritance parent is not registered") from error
        # Complete ProfileSchemaV1 instances already pin the schema ID/version; no partial
        # document can bypass that constructor and enter this comparison.
        if child_schema.evidence is not EvidenceLevel.MODIFIED:
            raise ScheduleContractError("inherited external profile evidence must be modified")

        changed = tuple(
            sorted(
                field.name
                for field in fields(ProfileSchemaV1)
                if field.name not in _IMPLICIT_INHERITANCE_FIELDS
                and getattr(parent.schema, field.name) != getattr(child_schema, field.name)
            )
        )
        if changed != inheritance.overridden_fields:
            raise ScheduleContractError(
                "overridden_fields must exactly match inherited semantic differences"
            )

    def _index_of(self, key: ProfileKey) -> int | None:
        for index, entry in enumerate(self.entries):
            if entry.key == key:
                return index
        return None

    def _with_entry(self, entry: RegisteredProfile) -> ProfileRegistry:
        entries = tuple(sorted((*self.entries, entry), key=lambda item: item.key.canonical))
        return type(self)(entries=entries)


def _builtin_entry(schema: ProfileSchemaV1) -> RegisteredProfile:
    return RegisteredProfile(
        key=ProfileKey.from_schema(schema),
        schema=schema,
        fingerprint=profile_schema_fingerprint(schema),
        origin=ProfileOrigin.BUILTIN,
    )


def builtin_profile_registry() -> ProfileRegistry:
    """Return the deterministic trusted registry shipped with this package."""

    # Local import avoids a schema -> profile -> registry import cycle.
    from comfyui_sigmax.profiles.anima import (
        ANIMA_AESTHETIC_SCHEMA,
        ANIMA_BASE_SCHEMA,
        ANIMA_TURBO_SCHEMA,
    )
    from comfyui_sigmax.profiles.aura_flow import AURAFLOW_V02_SCHEMA
    from comfyui_sigmax.profiles.flux1_schnell import FLUX1_SCHNELL_SCHEMA
    from comfyui_sigmax.profiles.hunyuan_image21 import (
        HUNYUAN_IMAGE21_BASE_SCHEMA,
        HUNYUAN_IMAGE21_DISTILLED_SCHEMA,
    )
    from comfyui_sigmax.profiles.krea2_lora_experimental import KREA2_LORA_EXPERIMENTAL_SCHEMA
    from comfyui_sigmax.profiles.krea2_raw import KREA2_RAW_SCHEMA
    from comfyui_sigmax.profiles.krea2_turbo import KREA2_TURBO_SCHEMA
    from comfyui_sigmax.profiles.ltx import (
        LTX2_19B_DISTILLED_STAGE1_PROFILE,
        LTX2_19B_DISTILLED_STAGE2_PROFILE,
        LTX2_19B_PROFILE,
        LTX23_22B_DISTILLED_STAGE1_PROFILE,
        LTX23_22B_DISTILLED_STAGE2_PROFILE,
        LTX23_22B_PROFILE,
        LTXV_098_PROFILE,
    )
    from comfyui_sigmax.profiles.lumina2 import LUMINA2_SCHEMA
    from comfyui_sigmax.profiles.minimax_h3 import (
        MINIMAX_H3_BASE_FL2VA_SCHEMA,
        MINIMAX_H3_BASE_REF2VA_SCHEMA,
    )
    from comfyui_sigmax.profiles.qwen_image import (
        QWEN_IMAGE_COMFY_FIXED_SCHEMA,
        QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA,
    )
    from comfyui_sigmax.profiles.sd3 import (
        SD3_COMFY_DIFFUSERS_SCHEMA,
        SD3_PUBLISHER_REFERENCE_SCHEMA,
    )
    from comfyui_sigmax.profiles.wan import (
        WAN21_COMFY_NATIVE_SCHEMA,
        WAN21_FLF2V_14B_720P_OFFICIAL_SCHEMA,
        WAN21_I2V_480P_DIFFUSERS_SCHEMA,
        WAN21_I2V_480P_OFFICIAL_SCHEMA,
        WAN21_I2V_720P_DIFFUSERS_SCHEMA,
        WAN21_I2V_720P_OFFICIAL_SCHEMA,
        WAN21_T2V_DIFFUSERS_SCHEMA,
        WAN21_T2V_OFFICIAL_SCHEMA,
        WAN21_VACE_1_3B_OFFICIAL_SCHEMA,
        WAN21_VACE_14B_OFFICIAL_SCHEMA,
        WAN22_ANIMATE_14B_OFFICIAL_SCHEMA,
        WAN22_I2V_A14B_DIFFUSERS_SCHEMA,
        WAN22_I2V_A14B_NATIVE_SCHEMA,
        WAN22_S2V_14B_OFFICIAL_SCHEMA,
        WAN22_T2V_A14B_DIFFUSERS_SCHEMA,
        WAN22_T2V_A14B_NATIVE_SCHEMA,
        WAN22_TI2V_5B_DIFFUSERS_SCHEMA,
        WAN22_TI2V_5B_NATIVE_SCHEMA,
        WAN_ANIMATE2_BASE_14B_OFFICIAL_SCHEMA,
        WAN_ANIMATE2_DISTILLED_14B_OFFICIAL_SCHEMA,
    )
    from comfyui_sigmax.profiles.z_image import Z_IMAGE_BASE_SCHEMA, Z_IMAGE_TURBO_SCHEMA

    entries = cast(
        tuple[RegisteredProfile, ...],
        tuple(
            sorted(
                (
                    _builtin_entry(AURAFLOW_V02_SCHEMA),
                    _builtin_entry(ANIMA_AESTHETIC_SCHEMA),
                    _builtin_entry(ANIMA_BASE_SCHEMA),
                    _builtin_entry(ANIMA_TURBO_SCHEMA),
                    _builtin_entry(FLUX1_SCHNELL_SCHEMA),
                    _builtin_entry(KREA2_LORA_EXPERIMENTAL_SCHEMA),
                    _builtin_entry(KREA2_RAW_SCHEMA),
                    _builtin_entry(KREA2_TURBO_SCHEMA),
                    _builtin_entry(LUMINA2_SCHEMA),
                    _builtin_entry(MINIMAX_H3_BASE_FL2VA_SCHEMA),
                    _builtin_entry(MINIMAX_H3_BASE_REF2VA_SCHEMA),
                    _builtin_entry(LTX2_19B_DISTILLED_STAGE1_PROFILE.schema),
                    _builtin_entry(LTX2_19B_DISTILLED_STAGE2_PROFILE.schema),
                    _builtin_entry(LTX2_19B_PROFILE.schema),
                    _builtin_entry(LTX23_22B_DISTILLED_STAGE1_PROFILE.schema),
                    _builtin_entry(LTX23_22B_DISTILLED_STAGE2_PROFILE.schema),
                    _builtin_entry(LTX23_22B_PROFILE.schema),
                    _builtin_entry(LTXV_098_PROFILE.schema),
                    _builtin_entry(HUNYUAN_IMAGE21_BASE_SCHEMA),
                    _builtin_entry(HUNYUAN_IMAGE21_DISTILLED_SCHEMA),
                    _builtin_entry(QWEN_IMAGE_COMFY_FIXED_SCHEMA),
                    _builtin_entry(QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA),
                    _builtin_entry(SD3_COMFY_DIFFUSERS_SCHEMA),
                    _builtin_entry(SD3_PUBLISHER_REFERENCE_SCHEMA),
                    _builtin_entry(Z_IMAGE_BASE_SCHEMA),
                    _builtin_entry(Z_IMAGE_TURBO_SCHEMA),
                    _builtin_entry(WAN21_COMFY_NATIVE_SCHEMA),
                    _builtin_entry(WAN21_FLF2V_14B_720P_OFFICIAL_SCHEMA),
                    _builtin_entry(WAN21_I2V_480P_DIFFUSERS_SCHEMA),
                    _builtin_entry(WAN21_I2V_480P_OFFICIAL_SCHEMA),
                    _builtin_entry(WAN21_I2V_720P_DIFFUSERS_SCHEMA),
                    _builtin_entry(WAN21_I2V_720P_OFFICIAL_SCHEMA),
                    _builtin_entry(WAN21_T2V_DIFFUSERS_SCHEMA),
                    _builtin_entry(WAN21_T2V_OFFICIAL_SCHEMA),
                    _builtin_entry(WAN21_VACE_1_3B_OFFICIAL_SCHEMA),
                    _builtin_entry(WAN21_VACE_14B_OFFICIAL_SCHEMA),
                    _builtin_entry(WAN22_I2V_A14B_DIFFUSERS_SCHEMA),
                    _builtin_entry(WAN22_I2V_A14B_NATIVE_SCHEMA),
                    _builtin_entry(WAN22_S2V_14B_OFFICIAL_SCHEMA),
                    _builtin_entry(WAN22_ANIMATE_14B_OFFICIAL_SCHEMA),
                    _builtin_entry(WAN22_T2V_A14B_DIFFUSERS_SCHEMA),
                    _builtin_entry(WAN22_T2V_A14B_NATIVE_SCHEMA),
                    _builtin_entry(WAN22_TI2V_5B_DIFFUSERS_SCHEMA),
                    _builtin_entry(WAN22_TI2V_5B_NATIVE_SCHEMA),
                    _builtin_entry(WAN_ANIMATE2_BASE_14B_OFFICIAL_SCHEMA),
                    _builtin_entry(WAN_ANIMATE2_DISTILLED_14B_OFFICIAL_SCHEMA),
                ),
                key=lambda entry: entry.key.canonical,
            )
        ),
    )
    return ProfileRegistry(entries=entries)
