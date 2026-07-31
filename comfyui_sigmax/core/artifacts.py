"""Immutable versioned schedule artifacts and strict JSON transport."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from comfyui_sigmax.core.fingerprints import (
    FloatPrecision,
    build_numerical_projection,
    canonical_projection_bytes,
    construction_fingerprint,
    float_to_ieee_hex,
)
from comfyui_sigmax.core.request_result import (
    BaseGridSpec,
    EvidenceLevel,
    ScheduleResult,
    SliceSpec,
    TerminalPolicy,
)
from comfyui_sigmax.core.schedule_contracts import (
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TransformContract,
    TransformStage,
    validate_transform_chain,
)
from comfyui_sigmax.core.validation import validate_sigma_schedule

SCHEDULE_ARTIFACT_ENVELOPE_SCHEMA = "sigmax.schedule-artifact-envelope/1"
SCHEDULE_ARTIFACT_SCHEMA = "sigmax.schedule-artifact/1"
NUMERICAL_SCHEDULE_SCHEMA = "sigmax.numerical-schedule/1"
_ENVELOPE_SCHEMA = SCHEDULE_ARTIFACT_ENVELOPE_SCHEMA
_CONSTRUCTION_SCHEMA = SCHEDULE_ARTIFACT_SCHEMA
_NUMERICAL_SCHEMA = NUMERICAL_SCHEDULE_SCHEMA
_ENVELOPE_FIELDS = frozenset(
    {
        "construction",
        "construction_fingerprint",
        "numerical",
        "numerical_fingerprint",
        "schema",
    }
)
_NUMERICAL_FIELDS = frozenset({"domain", "precision", "schema", "sigmas"})
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_FIELD_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PRIVATE_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'=(])(?:[a-z]:[\\/]|\\\\[^\\]|/(?:home|users|mnt)/)",
    re.IGNORECASE,
)
_SECRET_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:api_?key|access_key|private_key|secret|password|passwd|credential|cookie|"
    r"token|authorization|auth)(?:_|$)"
)
_MAX_TRANSPORT_BYTES = 1_048_576

ArtifactScalar = str | int | bool | None
SlicePolicyName = Literal["full", "manual_range", "denoise_tail"]


def _require_public_text(field_name: str, value: str) -> None:
    if not value.strip():
        raise ScheduleContractError(f"{field_name} must not be empty")
    if _PRIVATE_PATH_PATTERN.search(value):
        raise ScheduleContractError(f"{field_name} must not contain a private local path")


@dataclass(frozen=True, slots=True, kw_only=True)
class TypedArtifactValue:
    """One semantic float with an explicit fingerprint precision."""

    value: float
    precision: FloatPrecision

    def __post_init__(self) -> None:
        float_to_ieee_hex(self.value, self.precision)

    def projection(self) -> dict[str, str]:
        return {
            "bits": float_to_ieee_hex(self.value, self.precision),
            "precision": self.precision,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactField:
    """One schema-controlled metadata field."""

    name: str
    value: ArtifactScalar | TypedArtifactValue

    def __post_init__(self) -> None:
        if not _FIELD_NAME_PATTERN.fullmatch(self.name):
            raise ScheduleContractError("artifact field name must be an ASCII identifier")
        if _SECRET_NAME_PATTERN.search(self.name) or self.name.endswith("_path"):
            raise ScheduleContractError("artifact field name is secret-like or path-like")
        if isinstance(self.value, str):
            _require_public_text("artifact field value", self.value)
        elif not isinstance(self.value, (int, bool, TypedArtifactValue)) and self.value is not None:
            raise ScheduleContractError("unsupported artifact field value")

    def projection_value(self) -> object:
        if isinstance(self.value, TypedArtifactValue):
            return self.value.projection()
        return self.value


def _require_field_group(
    field_name: str,
    fields: tuple[ArtifactField, ...],
) -> None:
    if not isinstance(fields, tuple) or not all(isinstance(item, ArtifactField) for item in fields):
        raise ScheduleContractError(f"{field_name} must be a tuple of ArtifactField values")
    names = [item.name for item in fields]
    if len(names) != len(set(names)):
        raise ScheduleContractError(f"{field_name} contains duplicate field names")


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactBuildMetadata:
    """Immutable computed and compatibility metadata required for artifact construction."""

    source_id: str
    source_label: str
    engine_name: str = "comfyui-sigmax"
    base_grid_parameters: tuple[ArtifactField, ...] = ()
    transform_parameters: tuple[tuple[ArtifactField, ...], ...] = ()
    compatibility: tuple[ArtifactField, ...] = ()

    def __post_init__(self) -> None:
        _require_public_text("source_id", self.source_id)
        _require_public_text("source_label", self.source_label)
        _require_public_text("engine_name", self.engine_name)
        _require_field_group("base_grid_parameters", self.base_grid_parameters)
        _require_field_group("compatibility", self.compatibility)
        if not isinstance(self.transform_parameters, tuple):
            raise ScheduleContractError("transform_parameters must be a tuple")
        for index, fields in enumerate(self.transform_parameters):
            _require_field_group(f"transform_parameters[{index}]", fields)


def _fields_projection(fields: tuple[ArtifactField, ...]) -> dict[str, object]:
    return {field.name: field.projection_value() for field in fields}


def _sha256_identity(preimage: bytes) -> str:
    return f"sha256:{hashlib.sha256(preimage).hexdigest()}"


def _decode_projection_copy(payload: bytes) -> dict[str, object]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ScheduleContractError("canonical projection must be an object")
    return cast(dict[str, object], value)


def _validate_numerical_projection(projection: Mapping[str, object]) -> bytes:
    if set(projection) != _NUMERICAL_FIELDS:
        raise ScheduleContractError("numerical projection fields do not match schema")
    if projection.get("schema") != _NUMERICAL_SCHEMA:
        raise ScheduleContractError("unsupported numerical projection schema")
    precision = projection.get("precision")
    if precision not in {"float32", "float64"}:
        raise ScheduleContractError("unsupported numerical projection precision")
    domain_name = projection.get("domain")
    domain_by_wire = {
        domain.value.casefold(): domain
        for domain in SigmaDomain
        if domain is not SigmaDomain.MODEL_NATIVE
    }
    if not isinstance(domain_name, str) or domain_name not in domain_by_wire:
        raise ScheduleContractError("unsupported numerical projection domain")
    tokens = projection.get("sigmas")
    if not isinstance(tokens, list) or len(tokens) < 2:
        raise ScheduleContractError("numerical projection sigmas must be a non-empty array")
    width = 8 if precision == "float32" else 16
    unpack_format = ">f" if precision == "float32" else ">d"
    values: list[float] = []
    for token in tokens:
        if (
            not isinstance(token, str)
            or len(token) != width
            or token != token.casefold()
            or not re.fullmatch(r"[0-9a-f]+", token)
        ):
            raise ScheduleContractError("numerical projection contains an invalid sigma token")
        values.append(struct.unpack(unpack_format, bytes.fromhex(token))[0])
    validate_sigma_schedule(
        values,
        domain=domain_by_wire[domain_name],
        expected_steps=len(values) - 1,
        require_terminal_zero=True,
    )
    return canonical_projection_bytes(projection)


def _require_exact_object(
    value: object,
    *,
    field_name: str,
    fields: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ScheduleContractError(f"{field_name} fields do not match schema")
    return cast(Mapping[str, object], value)


def _require_public_string(
    value: object,
    *,
    field_name: str,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ScheduleContractError(f"{field_name} must be a string")
    _require_public_text(field_name, value)
    return value


def _require_positive_integer(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScheduleContractError(f"{field_name} must be a positive integer")
    return value


def _decode_typed_value(
    value: object,
    *,
    field_name: str,
    expected_precision: FloatPrecision | None = None,
) -> tuple[float, FloatPrecision, str]:
    typed = _require_exact_object(
        value,
        field_name=field_name,
        fields=frozenset({"bits", "precision"}),
    )
    precision = typed.get("precision")
    if precision not in {"float32", "float64"}:
        raise ScheduleContractError(f"{field_name} has an unsupported precision")
    if expected_precision is not None and precision != expected_precision:
        raise ScheduleContractError(f"{field_name} precision does not match the schedule")
    token = typed.get("bits")
    width = 8 if precision == "float32" else 16
    if (
        not isinstance(token, str)
        or len(token) != width
        or token != token.casefold()
        or not re.fullmatch(r"[0-9a-f]+", token)
    ):
        raise ScheduleContractError(f"{field_name} contains an invalid float token")
    unpack_format = ">f" if precision == "float32" else ">d"
    decoded = struct.unpack(unpack_format, bytes.fromhex(token))[0]
    if not math.isfinite(decoded):
        raise ScheduleContractError(f"{field_name} must be finite")
    if decoded == 0.0 and token != ("0" * width):
        raise ScheduleContractError(f"{field_name} must normalize negative zero")
    return decoded, precision, token


def _validate_field_projection(value: object, *, field_name: str) -> None:
    if isinstance(value, Mapping):
        _decode_typed_value(value, field_name=field_name)
        return
    if isinstance(value, str):
        _require_public_text(field_name, value)
        return
    if value is None or isinstance(value, (int, bool)):
        return
    raise ScheduleContractError(f"{field_name} has an unsupported metadata value")


def _validate_field_projection_map(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ScheduleContractError(f"{field_name} must be an object")
    # JSON object names are strings; canonical projection validation handles key syntax.
    fields = cast(Mapping[str, object], value)
    for name, child in fields.items():
        ArtifactField(name=name, value=None)
        _validate_field_projection(child, field_name=f"{field_name}.{name}")
    return cast(Mapping[str, object], value)


def _validate_inputs_projection(
    value: object,
    *,
    field_name: str,
    effective: bool,
) -> Mapping[str, object]:
    fields = {
        "height",
        "precision",
        "profile",
        "profile_version",
        "steps",
        "width",
    }
    if effective:
        fields.add("compatibility")
    inputs = _require_exact_object(
        value,
        field_name=field_name,
        fields=frozenset(fields),
    )
    _require_positive_integer(inputs.get("steps"), field_name=f"{field_name}.steps")
    width = inputs.get("width")
    height = inputs.get("height")
    if (width is None) != (height is None):
        raise ScheduleContractError(f"{field_name} dimensions must be supplied together")
    if width is not None:
        _require_positive_integer(width, field_name=f"{field_name}.width")
        _require_positive_integer(height, field_name=f"{field_name}.height")
    if inputs.get("precision") not in {"float32", "float64"}:
        raise ScheduleContractError(f"{field_name}.precision is unsupported")
    profile = inputs.get("profile")
    profile_version = inputs.get("profile_version")
    if (profile is None) != (profile_version is None):
        raise ScheduleContractError(f"{field_name} profile identity is incomplete")
    if profile is not None:
        _require_public_string(profile, field_name=f"{field_name}.profile")
        _require_public_string(profile_version, field_name=f"{field_name}.profile_version")
    if effective:
        _validate_field_projection_map(
            inputs.get("compatibility"),
            field_name="effective.compatibility",
        )
    return inputs


def _validate_construction_projection(
    construction: Mapping[str, object],
    numerical: Mapping[str, object],
) -> None:
    if set(construction) != {
        "base_grid",
        "effective",
        "engine",
        "evidence",
        "numerical_fingerprint",
        "overrides",
        "ownership",
        "requested",
        "schema",
        "slicing",
        "source",
        "terminal",
        "transforms",
        "warnings",
    }:
        raise ScheduleContractError("construction projection fields do not match schema")

    precision = cast(FloatPrecision, numerical["precision"])
    numerical_tokens = cast(list[str], numerical["sigmas"])
    numerical_domain = cast(str, numerical["domain"])

    engine = _require_exact_object(
        construction.get("engine"),
        field_name="engine",
        fields=frozenset({"name", "version"}),
    )
    _require_public_string(engine.get("name"), field_name="engine.name")
    _require_public_string(engine.get("version"), field_name="engine.version")

    source = _require_exact_object(
        construction.get("source"),
        field_name="source",
        fields=frozenset({"id", "label", "revision"}),
    )
    _require_public_string(source.get("id"), field_name="source.id")
    _require_public_string(source.get("label"), field_name="source.label")
    _require_public_string(
        source.get("revision"),
        field_name="source.revision",
        allow_none=True,
    )

    evidence = _require_exact_object(
        construction.get("evidence"),
        field_name="evidence",
        fields=frozenset({"level", "reference"}),
    )
    if evidence.get("level") not in {level.value for level in EvidenceLevel}:
        raise ScheduleContractError("evidence.level is unsupported")
    _require_public_string(evidence.get("reference"), field_name="evidence.reference")

    ownership = _require_exact_object(
        construction.get("ownership"),
        field_name="ownership",
        fields=frozenset({"schedule", "shift"}),
    )
    if ownership != {
        "schedule": ScheduleOwnership.EXTERNAL_SIGMAS.value.casefold(),
        "shift": "construction_pipeline",
    }:
        raise ScheduleContractError("artifact ownership is unsupported")

    requested = _validate_inputs_projection(
        construction.get("requested"),
        field_name="requested",
        effective=False,
    )
    effective = _validate_inputs_projection(
        construction.get("effective"),
        field_name="effective",
        effective=True,
    )
    if requested.get("precision") != precision or effective.get("precision") != precision:
        raise ScheduleContractError("construction precision does not match the schedule")
    if effective.get("steps") != len(numerical_tokens) - 1:
        raise ScheduleContractError("effective steps do not match the numerical schedule")

    overrides = construction.get("overrides")
    if not isinstance(overrides, list):
        raise ScheduleContractError("overrides must be an array")
    override_fields: list[str] = []
    for index, override_value in enumerate(overrides):
        override = _require_exact_object(
            override_value,
            field_name=f"overrides[{index}]",
            fields=frozenset({"effective", "path", "reason", "requested"}),
        )
        path = _require_public_string(
            override.get("path"),
            field_name=f"overrides[{index}].path",
        )
        if _SECRET_NAME_PATTERN.search(cast(str, path)) or cast(str, path).endswith("_path"):
            raise ScheduleContractError("override path is secret-like or path-like")
        override_fields.append(cast(str, path))
        for name in ("effective", "reason", "requested"):
            _require_public_string(
                override.get(name),
                field_name=f"overrides[{index}].{name}",
            )
    if len(override_fields) != len(set(override_fields)):
        raise ScheduleContractError("overrides contain duplicate field paths")
    changed_inputs = {
        name for name in ("steps", "width", "height") if requested.get(name) != effective.get(name)
    }
    if not changed_inputs.issubset(override_fields):
        raise ScheduleContractError("requested/effective differences lack override records")

    base_grid = _require_exact_object(
        construction.get("base_grid"),
        field_name="base_grid",
        fields=frozenset({"id", "parameters"}),
    )
    _require_public_string(base_grid.get("id"), field_name="base_grid.id")
    _validate_field_projection_map(
        base_grid.get("parameters"),
        field_name="base_grid.parameters",
    )

    transforms = construction.get("transforms")
    if not isinstance(transforms, list):
        raise ScheduleContractError("transforms must be an array")
    transform_contracts: list[TransformContract] = []
    domain_by_wire = {
        domain.value.casefold(): domain
        for domain in SigmaDomain
        if domain is not SigmaDomain.MODEL_NATIVE
    }
    stage_by_wire = {stage.value.casefold(): stage for stage in TransformStage}
    for index, transform_value in enumerate(transforms):
        transform = _require_exact_object(
            transform_value,
            field_name=f"transforms[{index}]",
            fields=frozenset(
                {
                    "from_domain",
                    "id",
                    "parameters",
                    "stage",
                    "stage_kind",
                    "to_domain",
                }
            ),
        )
        if transform.get("stage") != index:
            raise ScheduleContractError("transform stage indices must be contiguous")
        identifier = _require_public_string(
            transform.get("id"),
            field_name=f"transforms[{index}].id",
        )
        stage = stage_by_wire.get(cast(str, transform.get("stage_kind")))
        input_domain = domain_by_wire.get(transform.get("from_domain"))
        output_domain = domain_by_wire.get(transform.get("to_domain"))
        if stage is None or input_domain is None or output_domain is None:
            raise ScheduleContractError("transform kind or domain is unsupported")
        _validate_field_projection_map(
            transform.get("parameters"),
            field_name=f"transforms[{index}].parameters",
        )
        transform_contracts.append(
            TransformContract(
                name=cast(str, identifier),
                stage=stage,
                input_domain=input_domain,
                output_domain=output_domain,
            )
        )
    initial_domain = (
        transform_contracts[0].input_domain
        if transform_contracts
        else domain_by_wire[numerical_domain]
    )
    final_domain = validate_transform_chain(
        ScheduleOwnership.EXTERNAL_SIGMAS,
        initial_domain,
        transform_contracts,
    )
    if final_domain is not domain_by_wire[numerical_domain]:
        raise ScheduleContractError("transform chain final domain does not match the schedule")

    terminal = _require_exact_object(
        construction.get("terminal"),
        field_name="terminal",
        fields=frozenset({"policy", "value"}),
    )
    if terminal.get("policy") not in {policy.value.casefold() for policy in TerminalPolicy}:
        raise ScheduleContractError("terminal policy is unsupported")
    _, _, terminal_token = _decode_typed_value(
        terminal.get("value"),
        field_name="terminal.value",
        expected_precision=precision,
    )
    if terminal_token != numerical_tokens[-1]:
        raise ScheduleContractError("terminal value does not match the numerical schedule")

    slicing = _require_exact_object(
        construction.get("slicing"),
        field_name="slicing",
        fields=frozenset({"denoise", "end_step", "policy", "start_step"}),
    )
    denoise, _, _ = _decode_typed_value(
        slicing.get("denoise"),
        field_name="slicing.denoise",
        expected_precision=precision,
    )
    if not 0.0 <= denoise <= 1.0:
        raise ScheduleContractError("slicing.denoise must be between zero and one")
    start_step = slicing.get("start_step")
    end_step = slicing.get("end_step")
    if (
        not isinstance(start_step, int)
        or isinstance(start_step, bool)
        or start_step < 0
        or not isinstance(end_step, int)
        or isinstance(end_step, bool)
        or end_step <= start_step
    ):
        raise ScheduleContractError("slicing bounds are invalid")
    if end_step - start_step != effective.get("steps"):
        raise ScheduleContractError("slicing bounds do not match effective steps")
    slicing_policy = slicing.get("policy")
    if slicing_policy not in {"full", "manual_range", "denoise_tail"}:
        raise ScheduleContractError("slicing policy is unsupported")
    if slicing_policy == "full" and (denoise != 1.0 or start_step != 0):
        raise ScheduleContractError("full slicing metadata is inconsistent")
    if slicing_policy == "manual_range" and denoise != 1.0:
        raise ScheduleContractError("manual slicing metadata is inconsistent")
    if slicing_policy == "denoise_tail" and denoise == 1.0:
        raise ScheduleContractError("denoise-tail slicing metadata is inconsistent")

    warnings = construction.get("warnings")
    if not isinstance(warnings, list):
        raise ScheduleContractError("warnings must be an array")
    for index, warning in enumerate(warnings):
        _require_public_string(warning, field_name=f"warnings[{index}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleArtifact:
    """Immutable canonical schedule construction artifact."""

    construction_bytes: bytes
    numerical_bytes: bytes
    construction_fingerprint: str
    numerical_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.construction_bytes, bytes) or not isinstance(
            self.numerical_bytes, bytes
        ):
            raise ScheduleContractError("artifact projections must be canonical bytes")
        if not _SHA256_PATTERN.fullmatch(self.construction_fingerprint):
            raise ScheduleContractError("invalid construction fingerprint")
        if not _SHA256_PATTERN.fullmatch(self.numerical_fingerprint):
            raise ScheduleContractError("invalid numerical fingerprint")

        numerical = _decode_projection_copy(self.numerical_bytes)
        numerical_bytes = _validate_numerical_projection(numerical)
        if numerical_bytes != self.numerical_bytes:
            raise ScheduleContractError("numerical projection bytes are not canonical")
        if _sha256_identity(numerical_bytes) != self.numerical_fingerprint:
            raise ScheduleContractError("numerical fingerprint mismatch")

        construction = _decode_projection_copy(self.construction_bytes)
        if construction.get("schema") != _CONSTRUCTION_SCHEMA:
            raise ScheduleContractError("unsupported construction projection schema")
        if construction.get("numerical_fingerprint") != self.numerical_fingerprint:
            raise ScheduleContractError("embedded numerical fingerprint mismatch")
        _validate_construction_projection(construction, numerical)
        if canonical_projection_bytes(construction) != self.construction_bytes:
            raise ScheduleContractError("construction projection bytes are not canonical")
        if construction_fingerprint(construction) != self.construction_fingerprint:
            raise ScheduleContractError("construction fingerprint mismatch")

    def construction_projection(self) -> dict[str, object]:
        """Return a fresh decoded construction projection."""

        return _decode_projection_copy(self.construction_bytes)

    def numerical_projection(self) -> dict[str, object]:
        """Return a fresh decoded numerical projection."""

        return _decode_projection_copy(self.numerical_bytes)


def _inputs_projection(
    result: ScheduleResult,
    *,
    effective: bool,
    precision: FloatPrecision,
    compatibility: tuple[ArtifactField, ...],
) -> dict[str, object]:
    inputs = result.effective_inputs if effective else result.request.requested_inputs
    provenance = result.request.provenance
    projection: dict[str, object] = {
        "height": inputs.height,
        "precision": precision,
        "profile": provenance.profile_id,
        "profile_version": provenance.profile_version,
        "steps": inputs.steps,
        "width": inputs.width,
    }
    if effective:
        projection["compatibility"] = _fields_projection(compatibility)
    return projection


def _slice_policy(slicing: SliceSpec) -> SlicePolicyName:
    if float(slicing.denoise) != 1.0:
        return "denoise_tail"
    if slicing.start_step != 0 or slicing.end_step is not None:
        return "manual_range"
    return "full"


def build_schedule_artifact(
    result: ScheduleResult,
    *,
    metadata: ArtifactBuildMetadata,
    precision: FloatPrecision,
) -> ScheduleArtifact:
    """Build an immutable versioned artifact from one validated external schedule result."""

    request = result.request
    if request.ownership is not ScheduleOwnership.EXTERNAL_SIGMAS:
        raise ScheduleContractError("artifact construction requires EXTERNAL_SIGMAS ownership")
    # EXTERNAL_SIGMAS construction completeness is enforced by ScheduleRequest.
    base_grid = cast(BaseGridSpec, request.base_grid)
    terminal_policy = cast(TerminalPolicy, request.terminal_policy)
    slicing = cast(SliceSpec, request.slicing)
    if len(metadata.transform_parameters) != len(request.transforms):
        raise ScheduleContractError(
            "transform parameter groups must align with the ordered transform chain"
        )

    validate_sigma_schedule(
        result.sigmas,
        domain=result.final_domain,
        expected_steps=result.effective_inputs.steps,
        require_terminal_zero=True,
    )
    numerical = build_numerical_projection(
        result.sigmas,
        domain=result.final_domain,
        precision=precision,
    )
    numerical_bytes = canonical_projection_bytes(numerical)
    numerical_identity = _sha256_identity(numerical_bytes)

    transforms: list[dict[str, object]] = []
    for index, (transform, parameters) in enumerate(
        zip(request.transforms, metadata.transform_parameters, strict=True)
    ):
        transforms.append(
            {
                "from_domain": transform.input_domain.value.casefold(),
                "id": transform.name,
                "parameters": _fields_projection(parameters),
                "stage": index,
                "stage_kind": transform.stage.value.casefold(),
                "to_domain": transform.output_domain.value.casefold(),
            }
        )

    overrides = [
        {
            "effective": override.effective_value,
            "path": override.field,
            "reason": override.reason,
            "requested": override.requested_value,
        }
        for override in (*request.overrides, *result.overrides)
    ]
    provenance = request.provenance
    construction: dict[str, object] = {
        "base_grid": {
            "id": base_grid.identifier,
            "parameters": _fields_projection(metadata.base_grid_parameters),
        },
        "effective": _inputs_projection(
            result,
            effective=True,
            precision=precision,
            compatibility=metadata.compatibility,
        ),
        "engine": {
            "name": metadata.engine_name,
            "version": provenance.engine_version,
        },
        "evidence": {
            "level": provenance.evidence.value,
            "reference": provenance.source_revision or metadata.source_id,
        },
        "numerical_fingerprint": numerical_identity,
        "overrides": overrides,
        "ownership": {
            "schedule": request.ownership.value.casefold(),
            "shift": "construction_pipeline",
        },
        "requested": _inputs_projection(
            result,
            effective=False,
            precision=precision,
            compatibility=(),
        ),
        "schema": _CONSTRUCTION_SCHEMA,
        "slicing": {
            "denoise": TypedArtifactValue(
                value=float(slicing.denoise),
                precision=precision,
            ).projection(),
            "end_step": slicing.end_step or result.effective_inputs.steps,
            "policy": _slice_policy(slicing),
            "start_step": slicing.start_step,
        },
        "source": {
            "id": metadata.source_id,
            "label": metadata.source_label,
            "revision": provenance.source_revision,
        },
        "terminal": {
            "policy": terminal_policy.value.casefold(),
            "value": TypedArtifactValue(
                value=result.sigmas[-1],
                precision=precision,
            ).projection(),
        },
        "transforms": transforms,
        "warnings": list(result.warnings),
    }
    construction_bytes = canonical_projection_bytes(construction)
    construction_identity = construction_fingerprint(construction)
    return ScheduleArtifact(
        construction_bytes=construction_bytes,
        numerical_bytes=numerical_bytes,
        construction_fingerprint=construction_identity,
        numerical_fingerprint=numerical_identity,
    )


def serialize_schedule_artifact(artifact: ScheduleArtifact) -> bytes:
    """Serialize one immutable artifact into canonical envelope bytes."""

    if not isinstance(artifact, ScheduleArtifact):
        raise ScheduleContractError("artifact must be a ScheduleArtifact")
    envelope: dict[str, object] = {
        "construction": artifact.construction_projection(),
        "construction_fingerprint": artifact.construction_fingerprint,
        "numerical": artifact.numerical_projection(),
        "numerical_fingerprint": artifact.numerical_fingerprint,
        "schema": _ENVELOPE_SCHEMA,
    }
    return canonical_projection_bytes(envelope)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleContractError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise ScheduleContractError(f"JSON floating literal is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise ScheduleContractError(f"non-standard JSON constant is forbidden: {value}")


def deserialize_schedule_artifact(payload: bytes | str) -> ScheduleArtifact:
    """Strictly parse, canonicalize, and verify one untrusted artifact envelope."""

    if isinstance(payload, str):
        if payload.startswith("\ufeff"):
            raise ScheduleContractError("artifact transport must not contain a BOM")
        raw = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        raw = payload
    else:
        raise ScheduleContractError("artifact transport must be bytes or str")
    if len(raw) > _MAX_TRANSPORT_BYTES:
        raise ScheduleContractError("artifact transport exceeds the maximum size")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ScheduleContractError("artifact transport must not contain a BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ScheduleContractError("artifact transport must be valid UTF-8") from error

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ScheduleContractError("artifact transport is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ScheduleContractError("artifact transport root must be an object")
    envelope = cast(dict[str, object], decoded)
    if set(envelope) != _ENVELOPE_FIELDS or envelope.get("schema") != _ENVELOPE_SCHEMA:
        raise ScheduleContractError("artifact envelope schema or fields are unsupported")
    if canonical_projection_bytes(envelope) != raw:
        raise ScheduleContractError("artifact transport is not canonical")

    construction = envelope.get("construction")
    numerical = envelope.get("numerical")
    construction_identity = envelope.get("construction_fingerprint")
    numerical_identity = envelope.get("numerical_fingerprint")
    if not isinstance(construction, Mapping) or not isinstance(numerical, Mapping):
        raise ScheduleContractError("artifact projections must be objects")
    if not isinstance(construction_identity, str) or not isinstance(numerical_identity, str):
        raise ScheduleContractError("artifact fingerprints must be strings")

    return ScheduleArtifact(
        construction_bytes=canonical_projection_bytes(construction),
        numerical_bytes=_validate_numerical_projection(numerical),
        construction_fingerprint=construction_identity,
        numerical_fingerprint=numerical_identity,
    )
