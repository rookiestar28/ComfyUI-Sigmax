"""M6-02 namespaced profile registry and explicit inheritance contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError
from comfyui_sigmax.profiles import (
    ANIMA_AESTHETIC_SCHEMA,
    ANIMA_BASE_SCHEMA,
    ANIMA_TURBO_SCHEMA,
    AURAFLOW_V02_SCHEMA,
    FLUX1_SCHNELL_SCHEMA,
    HUNYUAN_IMAGE21_BASE_SCHEMA,
    HUNYUAN_IMAGE21_DISTILLED_SCHEMA,
    KREA2_LORA_EXPERIMENTAL_SCHEMA,
    KREA2_RAW_SCHEMA,
    KREA2_TURBO_SCHEMA,
    LTX2_19B_DISTILLED_STAGE1_PROFILE,
    LTX2_19B_DISTILLED_STAGE2_PROFILE,
    LTX2_19B_PROFILE,
    LTX23_22B_DISTILLED_STAGE1_PROFILE,
    LTX23_22B_DISTILLED_STAGE2_PROFILE,
    LTX23_22B_PROFILE,
    LTXV_098_PROFILE,
    LUMINA2_SCHEMA,
    MINIMAX_H3_BASE_FL2VA_SCHEMA,
    MINIMAX_H3_BASE_REF2VA_SCHEMA,
    QWEN_IMAGE_COMFY_FIXED_SCHEMA,
    QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA,
    SD3_COMFY_DIFFUSERS_SCHEMA,
    SD3_PUBLISHER_REFERENCE_SCHEMA,
    WAN21_COMFY_NATIVE_SCHEMA,
    WAN21_I2V_480P_DIFFUSERS_SCHEMA,
    WAN21_I2V_480P_OFFICIAL_SCHEMA,
    WAN21_I2V_720P_DIFFUSERS_SCHEMA,
    WAN21_I2V_720P_OFFICIAL_SCHEMA,
    WAN21_T2V_DIFFUSERS_SCHEMA,
    WAN21_T2V_OFFICIAL_SCHEMA,
    WAN22_I2V_A14B_DIFFUSERS_SCHEMA,
    WAN22_I2V_A14B_NATIVE_SCHEMA,
    WAN22_T2V_A14B_DIFFUSERS_SCHEMA,
    WAN22_T2V_A14B_NATIVE_SCHEMA,
    WAN22_TI2V_5B_DIFFUSERS_SCHEMA,
    WAN22_TI2V_5B_NATIVE_SCHEMA,
    Z_IMAGE_BASE_SCHEMA,
    Z_IMAGE_TURBO_SCHEMA,
    ConflictPolicy,
    ProfileInheritance,
    ProfileKey,
    ProfileOrigin,
    ProfileRegistry,
    RegisteredProfile,
    builtin_profile_registry,
    profile_schema_fingerprint,
)


def _external_turbo_schema(
    *,
    profile_id: str = "example.krea2.turbo.custom",
    profile_version: str = "1",
    display_name: str = "Example Krea 2 Turbo Custom",
    known_limitations: tuple[str, ...] = ("External test profile.",),
) -> Any:
    profile_capabilities = replace(
        KREA2_TURBO_SCHEMA.profile_capabilities,
        profile_id=profile_id,
        profile_version=profile_version,
    )
    return replace(
        KREA2_TURBO_SCHEMA,
        profile_id=profile_id,
        profile_version=profile_version,
        display_name=display_name,
        evidence=EvidenceLevel.MODIFIED,
        profile_capabilities=profile_capabilities,
        known_limitations=known_limitations,
    )


def _turbo_inheritance(
    *,
    overridden_fields: tuple[str, ...] = (
        "display_name",
        "evidence",
        "known_limitations",
        "profile_capabilities",
    ),
) -> ProfileInheritance:
    return ProfileInheritance(
        parent=ProfileKey.from_schema(KREA2_TURBO_SCHEMA),
        overridden_fields=overridden_fields,
    )


def test_profile_keys_are_exact_namespaced_and_versioned() -> None:
    key = ProfileKey.from_schema(KREA2_TURBO_SCHEMA)

    assert key.profile_id == "krea2.turbo.official"
    assert key.profile_version == "1"
    assert key.canonical == "krea2.turbo.official@1"
    assert key == ProfileKey(profile_id="krea2.turbo.official", profile_version="1")
    assert key != ProfileKey(profile_id="krea2.turbo.official", profile_version="2")


@pytest.mark.parametrize(
    ("profile_id", "profile_version"),
    [
        ("unnamespaced", "1"),
        (".leading", "1"),
        ("trailing.", "1"),
        ("Bad.Namespace", "1"),
        ("valid.namespace", "latest"),
        ("valid.namespace", ""),
    ],
)
def test_profile_keys_reject_ambiguous_or_invalid_identity(
    profile_id: str,
    profile_version: str,
) -> None:
    with pytest.raises(ScheduleContractError):
        ProfileKey(profile_id=profile_id, profile_version=profile_version)


@pytest.mark.parametrize(
    ("profile_id", "profile_version"),
    [
        (cast(Any, 1), "1"),
        ("例子.namespace", "1"),
        ("valid.namespace", cast(Any, 1)),
    ],
)
def test_profile_keys_reject_invalid_runtime_types_and_non_ascii(
    profile_id: Any,
    profile_version: Any,
) -> None:
    with pytest.raises(ScheduleContractError):
        ProfileKey(profile_id=profile_id, profile_version=profile_version)


def test_profile_key_from_schema_requires_a_complete_schema() -> None:
    with pytest.raises(ScheduleContractError):
        ProfileKey.from_schema(cast(Any, object()))


def test_builtin_registry_is_deterministic_exact_and_immutable() -> None:
    registry = builtin_profile_registry()

    assert tuple(entry.key for entry in registry.entries) == (
        ProfileKey.from_schema(ANIMA_AESTHETIC_SCHEMA),
        ProfileKey.from_schema(ANIMA_BASE_SCHEMA),
        ProfileKey.from_schema(ANIMA_TURBO_SCHEMA),
        ProfileKey.from_schema(AURAFLOW_V02_SCHEMA),
        ProfileKey.from_schema(FLUX1_SCHNELL_SCHEMA),
        ProfileKey.from_schema(HUNYUAN_IMAGE21_BASE_SCHEMA),
        ProfileKey.from_schema(HUNYUAN_IMAGE21_DISTILLED_SCHEMA),
        ProfileKey.from_schema(KREA2_LORA_EXPERIMENTAL_SCHEMA),
        ProfileKey.from_schema(KREA2_RAW_SCHEMA),
        ProfileKey.from_schema(KREA2_TURBO_SCHEMA),
        ProfileKey.from_schema(LTX2_19B_PROFILE.schema),
        ProfileKey.from_schema(LTX2_19B_DISTILLED_STAGE1_PROFILE.schema),
        ProfileKey.from_schema(LTX2_19B_DISTILLED_STAGE2_PROFILE.schema),
        ProfileKey.from_schema(LTX23_22B_PROFILE.schema),
        ProfileKey.from_schema(LTX23_22B_DISTILLED_STAGE1_PROFILE.schema),
        ProfileKey.from_schema(LTX23_22B_DISTILLED_STAGE2_PROFILE.schema),
        ProfileKey.from_schema(LTXV_098_PROFILE.schema),
        ProfileKey.from_schema(LUMINA2_SCHEMA),
        ProfileKey.from_schema(MINIMAX_H3_BASE_FL2VA_SCHEMA),
        ProfileKey.from_schema(MINIMAX_H3_BASE_REF2VA_SCHEMA),
        ProfileKey.from_schema(QWEN_IMAGE_COMFY_FIXED_SCHEMA),
        ProfileKey.from_schema(QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA),
        ProfileKey.from_schema(SD3_COMFY_DIFFUSERS_SCHEMA),
        ProfileKey.from_schema(SD3_PUBLISHER_REFERENCE_SCHEMA),
        ProfileKey.from_schema(WAN21_I2V_480P_DIFFUSERS_SCHEMA),
        ProfileKey.from_schema(WAN21_I2V_480P_OFFICIAL_SCHEMA),
        ProfileKey.from_schema(WAN21_I2V_720P_DIFFUSERS_SCHEMA),
        ProfileKey.from_schema(WAN21_I2V_720P_OFFICIAL_SCHEMA),
        ProfileKey.from_schema(WAN21_COMFY_NATIVE_SCHEMA),
        ProfileKey.from_schema(WAN21_T2V_DIFFUSERS_SCHEMA),
        ProfileKey.from_schema(WAN21_T2V_OFFICIAL_SCHEMA),
        ProfileKey.from_schema(WAN22_I2V_A14B_DIFFUSERS_SCHEMA),
        ProfileKey.from_schema(WAN22_I2V_A14B_NATIVE_SCHEMA),
        ProfileKey.from_schema(WAN22_T2V_A14B_DIFFUSERS_SCHEMA),
        ProfileKey.from_schema(WAN22_T2V_A14B_NATIVE_SCHEMA),
        ProfileKey.from_schema(WAN22_TI2V_5B_NATIVE_SCHEMA),
        ProfileKey.from_schema(WAN22_TI2V_5B_DIFFUSERS_SCHEMA),
        ProfileKey.from_schema(Z_IMAGE_BASE_SCHEMA),
        ProfileKey.from_schema(Z_IMAGE_TURBO_SCHEMA),
    )
    assert all(entry.origin is ProfileOrigin.BUILTIN for entry in registry.entries)
    assert registry.resolve(ProfileKey.from_schema(KREA2_RAW_SCHEMA)).schema is KREA2_RAW_SCHEMA
    assert registry.resolve(
        ProfileKey.from_schema(KREA2_TURBO_SCHEMA)
    ).fingerprint == profile_schema_fingerprint(KREA2_TURBO_SCHEMA)
    with pytest.raises(FrozenInstanceError):
        registry.entries = ()  # type: ignore[misc]


def test_exact_lookup_has_no_latest_prefix_or_fallback_behavior() -> None:
    registry = builtin_profile_registry()

    for key in (
        ProfileKey(profile_id="krea2.turbo.official", profile_version="2"),
        ProfileKey(profile_id="krea2.turbo", profile_version="1"),
        ProfileKey(profile_id="example.unknown", profile_version="1"),
    ):
        with pytest.raises(ScheduleContractError, match="not registered"):
            registry.resolve(key)


def test_external_registration_returns_a_new_canonical_snapshot() -> None:
    registry = builtin_profile_registry()
    schema = _external_turbo_schema()

    updated = registry.register_external(schema, inheritance=_turbo_inheritance())
    entry = updated.resolve(ProfileKey.from_schema(schema))

    assert updated is not registry
    assert len(registry.entries) == 39
    assert len(updated.entries) == 40
    assert entry.origin is ProfileOrigin.EXTERNAL
    assert entry.schema is schema
    assert entry.inheritance == _turbo_inheritance()
    assert tuple(item.key.canonical for item in updated.entries) == tuple(
        sorted(item.key.canonical for item in updated.entries)
    )


def test_identical_external_registration_is_idempotent() -> None:
    schema = _external_turbo_schema()
    inheritance = _turbo_inheritance()
    registry = builtin_profile_registry().register_external(
        schema,
        inheritance=inheritance,
    )

    assert registry.register_external(schema, inheritance=inheritance) is registry


def test_same_key_with_different_content_rejects_by_default() -> None:
    original = _external_turbo_schema()
    replacement = _external_turbo_schema(known_limitations=("Changed content.",))
    registry = builtin_profile_registry().register_external(
        original,
        inheritance=_turbo_inheritance(),
    )

    with pytest.raises(ScheduleContractError, match="conflict"):
        registry.register_external(
            replacement,
            inheritance=_turbo_inheritance(),
        )


@pytest.mark.parametrize("policy", tuple(ConflictPolicy))
def test_external_registration_can_never_replace_a_builtin(
    policy: ConflictPolicy,
) -> None:
    replacement = replace(
        KREA2_TURBO_SCHEMA,
        known_limitations=("Attempted external override.",),
    )
    registry = builtin_profile_registry()
    if policy is ConflictPolicy.REPLACE_EXTERNAL:
        with pytest.raises(ScheduleContractError, match="built-in"):
            registry.register_external(
                replacement,
                conflict_policy=policy,
                expected_fingerprint=profile_schema_fingerprint(KREA2_TURBO_SCHEMA),
            )
    else:
        with pytest.raises(ScheduleContractError, match="built-in"):
            registry.register_external(replacement, conflict_policy=policy)


def test_external_replacement_requires_exact_compare_and_swap_fingerprint() -> None:
    original = _external_turbo_schema()
    replacement = _external_turbo_schema(known_limitations=("Changed content.",))
    inheritance = _turbo_inheritance()
    registry = builtin_profile_registry().register_external(
        original,
        inheritance=inheritance,
    )

    for expected in (None, "sha256:" + ("0" * 64)):
        with pytest.raises(ScheduleContractError, match="fingerprint"):
            registry.register_external(
                replacement,
                inheritance=inheritance,
                conflict_policy=ConflictPolicy.REPLACE_EXTERNAL,
                expected_fingerprint=expected,
            )

    updated = registry.register_external(
        replacement,
        inheritance=inheritance,
        conflict_policy=ConflictPolicy.REPLACE_EXTERNAL,
        expected_fingerprint=profile_schema_fingerprint(original),
    )
    assert updated.resolve(ProfileKey.from_schema(replacement)).schema is replacement
    assert registry.resolve(ProfileKey.from_schema(original)).schema is original


def test_reject_policy_does_not_accept_an_expected_fingerprint() -> None:
    schema = _external_turbo_schema()

    with pytest.raises(ScheduleContractError, match="only valid"):
        builtin_profile_registry().register_external(
            schema,
            inheritance=_turbo_inheritance(),
            expected_fingerprint=profile_schema_fingerprint(schema),
        )


def test_inheritance_requires_an_existing_distinct_parent() -> None:
    schema = _external_turbo_schema()
    missing = ProfileInheritance(
        parent=ProfileKey(profile_id="example.missing", profile_version="1"),
        overridden_fields=_turbo_inheritance().overridden_fields,
    )
    same_key = ProfileInheritance(
        parent=ProfileKey.from_schema(schema),
        overridden_fields=_turbo_inheritance().overridden_fields,
    )

    with pytest.raises(ScheduleContractError, match="parent"):
        builtin_profile_registry().register_external(schema, inheritance=missing)
    with pytest.raises(ScheduleContractError, match="itself"):
        ProfileRegistry.empty().register_external(schema, inheritance=same_key)


@pytest.mark.parametrize(
    "overridden_fields",
    [
        ("evidence", "display_name", "known_limitations", "profile_capabilities"),
        ("display_name", "evidence", "evidence", "known_limitations", "profile_capabilities"),
        ("profile_id",),
        ("schema_version",),
        ("unknown_field",),
    ],
)
def test_inheritance_override_names_are_canonical_and_controlled(
    overridden_fields: tuple[str, ...],
) -> None:
    with pytest.raises(ScheduleContractError):
        ProfileInheritance(
            parent=ProfileKey.from_schema(KREA2_TURBO_SCHEMA),
            overridden_fields=overridden_fields,
        )


@pytest.mark.parametrize(
    ("parent", "overridden_fields"),
    [
        (cast(Any, object()), ("evidence",)),
        (ProfileKey.from_schema(KREA2_TURBO_SCHEMA), ()),
        (ProfileKey.from_schema(KREA2_TURBO_SCHEMA), cast(Any, ["evidence"])),
        (ProfileKey.from_schema(KREA2_TURBO_SCHEMA), cast(Any, (object(),))),
    ],
)
def test_inheritance_rejects_invalid_runtime_shapes(
    parent: Any,
    overridden_fields: Any,
) -> None:
    with pytest.raises(ScheduleContractError):
        ProfileInheritance(parent=parent, overridden_fields=overridden_fields)


@pytest.mark.parametrize(
    "overridden_fields",
    [
        ("display_name", "evidence", "profile_capabilities"),
        (
            "display_name",
            "evidence",
            "frameworks",
            "known_limitations",
            "profile_capabilities",
        ),
    ],
)
def test_inheritance_must_declare_the_exact_semantic_diff(
    overridden_fields: tuple[str, ...],
) -> None:
    with pytest.raises(ScheduleContractError, match="overridden_fields"):
        builtin_profile_registry().register_external(
            _external_turbo_schema(),
            inheritance=_turbo_inheritance(overridden_fields=overridden_fields),
        )


def test_inherited_external_profile_must_use_modified_evidence() -> None:
    schema = replace(_external_turbo_schema(), evidence=EvidenceLevel.EXPERIMENTAL)

    with pytest.raises(ScheduleContractError, match="modified"):
        builtin_profile_registry().register_external(
            schema,
            inheritance=_turbo_inheritance(),
        )


def test_external_profile_can_be_registered_without_inheritance() -> None:
    schema = _external_turbo_schema(profile_id="example.standalone.profile")
    registry = builtin_profile_registry().register_external(schema)

    entry = registry.resolve(ProfileKey.from_schema(schema))
    assert entry.inheritance is None
    assert entry.origin is ProfileOrigin.EXTERNAL


def test_inheritance_chain_is_acyclic_by_construction() -> None:
    child = _external_turbo_schema()
    registry = builtin_profile_registry().register_external(
        child,
        inheritance=_turbo_inheritance(),
    )
    grandchild = _external_turbo_schema(
        profile_id="example.krea2.turbo.grandchild",
        display_name="Example Krea 2 Turbo Grandchild",
        known_limitations=("Grandchild profile.",),
    )
    inheritance = ProfileInheritance(
        parent=ProfileKey.from_schema(child),
        overridden_fields=(
            "display_name",
            "known_limitations",
            "profile_capabilities",
        ),
    )

    updated = registry.register_external(grandchild, inheritance=inheritance)
    assert updated.resolve(ProfileKey.from_schema(grandchild)).inheritance == inheritance


def test_registry_constructor_rejects_duplicate_or_noncanonical_entries() -> None:
    registry = builtin_profile_registry()

    with pytest.raises(ScheduleContractError):
        ProfileRegistry(entries=(registry.entries[1], registry.entries[0]))
    with pytest.raises(ScheduleContractError):
        ProfileRegistry(entries=(registry.entries[0], registry.entries[0]))


def test_registered_profile_rejects_invalid_runtime_contracts() -> None:
    entry = builtin_profile_registry().entries[0]
    valid_inheritance = _turbo_inheritance()
    factories: tuple[Callable[[], object], ...] = (
        lambda: replace(entry, key=cast(Any, object())),
        lambda: replace(entry, schema=cast(Any, object())),
        lambda: replace(entry, key=ProfileKey.from_schema(KREA2_TURBO_SCHEMA)),
        lambda: replace(entry, fingerprint=cast(Any, object())),
        lambda: replace(entry, fingerprint="invalid"),
        lambda: replace(entry, fingerprint="sha256:" + ("0" * 64)),
        lambda: replace(entry, origin=cast(Any, object())),
        lambda: replace(entry, inheritance=cast(Any, object())),
        lambda: RegisteredProfile(
            key=entry.key,
            schema=entry.schema,
            fingerprint=entry.fingerprint,
            origin=ProfileOrigin.BUILTIN,
            inheritance=valid_inheritance,
        ),
    )

    for factory in factories:
        with pytest.raises(ScheduleContractError):
            factory()


@pytest.mark.parametrize("entries", [cast(Any, []), cast(Any, (object(),))])
def test_registry_rejects_invalid_entry_container(entries: Any) -> None:
    with pytest.raises(ScheduleContractError):
        ProfileRegistry(entries=entries)


def test_registry_rejects_invalid_runtime_argument_types() -> None:
    registry = builtin_profile_registry()

    with pytest.raises(ScheduleContractError):
        registry.resolve(object())  # type: ignore[arg-type]
    with pytest.raises(ScheduleContractError):
        registry.register_external(object())  # type: ignore[arg-type]
    with pytest.raises(ScheduleContractError):
        registry.register_external(
            _external_turbo_schema(),
            inheritance=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ScheduleContractError):
        registry.register_external(
            _external_turbo_schema(),
            conflict_policy=cast(Any, object()),
        )
    with pytest.raises(ScheduleContractError, match="existing external"):
        registry.register_external(
            _external_turbo_schema(),
            conflict_policy=ConflictPolicy.REPLACE_EXTERNAL,
            expected_fingerprint="sha256:" + ("0" * 64),
        )
