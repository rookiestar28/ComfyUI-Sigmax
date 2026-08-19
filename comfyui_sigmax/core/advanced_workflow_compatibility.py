"""Pure, bounded compatibility decisions for advanced ComfyUI workflows.

The contract in this module is deliberately not a sampler or a host adapter.  It records
which owner is responsible for an advanced feature and fails closed when the current pure
controller cannot make a truthful claim.  No framework objects, tensors, paths, prompts, or
weights are accepted at this boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import cast

from comfyui_sigmax.core.fingerprints import canonical_projection_bytes
from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

ADVANCED_WORKFLOW_REQUEST_SCHEMA = "sigmax.advanced-workflow-request/1"
ADVANCED_WORKFLOW_DECISION_SCHEMA = "sigmax.advanced-workflow-decision/1"
ADVANCED_WORKFLOW_COMPATIBILITY_SCHEMA = "sigmax.advanced-workflow-compatibility/1"

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_PRIVATE_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'=(])(?:[a-z]:[\\/]|\\\\[^\\]|/(?:home|users|mnt)/)",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(
    r"(?:^|[_.-])(?:api_?key|access_key|private_key|secret|password|passwd|credential|"
    r"cookie|token|authorization|auth)(?:[_.-]|$)",
    re.IGNORECASE,
)
_MAX_PAYLOAD_BYTES = 1_048_576


class AdvancedWorkflowFeature(str, Enum):
    """Advanced workflow seams whose ownership must be explicit."""

    IMAGE_TO_IMAGE = "image_to_image"
    INPAINTING = "inpainting"
    PARTIAL_DENOISE = "partial_denoise"
    CONTROLNET = "controlnet"
    MODEL_PATCHES = "model_patches"
    INTERRUPTION = "interruption"
    RESUME = "resume"


class AdvancedExecutionMode(str, Enum):
    """Execution boundary considered by the compatibility resolver."""

    NATIVE_HOST = "native_host"
    DETERMINISTIC_PURE = "deterministic_pure"
    STOCHASTIC_PURE = "stochastic_pure"


class AdvancedOwnership(str, Enum):
    """Owner of a feature when a request is allowed or rejected."""

    HOST = "host"
    CONTROLLER = "controller"
    UNSUPPORTED = "unsupported"


class AdvancedDecisionLevel(str, Enum):
    """Compatibility outcome before any external execution call."""

    ALLOW = "allow"
    WARN = "warn"
    REJECT = "reject"


class AdvancedReasonCode(str, Enum):
    """Stable reason codes, kept in deterministic serialization order."""

    COMPATIBLE = "compatible"
    HOST_CAPABILITY_MISSING = "host_capability_missing"
    PURE_HOST_FEATURE_UNSUPPORTED = "pure_host_feature_unsupported"
    PURE_MODEL_OWNERSHIP_UNSUPPORTED = "pure_model_ownership_unsupported"
    STOCHASTIC_PARTIAL_UNSUPPORTED = "stochastic_partial_unsupported"
    STOCHASTIC_RESUME_UNSUPPORTED = "stochastic_resume_unsupported"
    STOCHASTIC_INTERRUPTION_UNSUPPORTED = "stochastic_interruption_unsupported"
    RESUME_SNAPSHOT_REQUIRED = "resume_snapshot_required"
    RESUME_SNAPSHOT_MISMATCH = "resume_snapshot_mismatch"
    HOST_RESUME_UNSUPPORTED = "host_resume_unsupported"
    HOST_INTERRUPT_NON_RESUMABLE = "host_interrupt_non_resumable"
    CONTRADICTORY_OWNERSHIP = "contradictory_ownership"


class AdvancedReceiptStatus(str, Enum):
    """Truthful status of a compatibility receipt, not a model execution result."""

    NOT_EXECUTED = "not_executed"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"
    RESUMABLE = "resumable"
    SUCCEEDED = "succeeded"


def _sha256_identity(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_bool(field_name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ScheduleContractError(f"{field_name} must be boolean")


def _require_fingerprint(field_name: str, value: object, *, allow_none: bool = True) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ScheduleContractError(f"{field_name} must be a lowercase SHA-256 identity")
    return value


def _require_identifier(field_name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or _IDENTIFIER_PATTERN.fullmatch(value) is None
        or _PRIVATE_PATH_PATTERN.search(value)
        or _SECRET_PATTERN.search(value)
    ):
        raise ScheduleContractError(f"{field_name} must be a bounded public identifier")
    return value


def _require_enum_tuple(
    field_name: str,
    values: object,
    enum_type: type[Enum],
    *,
    allow_empty: bool,
) -> tuple[Enum, ...]:
    if not isinstance(values, tuple):
        raise ScheduleContractError(f"{field_name} must be a tuple")
    if not allow_empty and not values:
        raise ScheduleContractError(f"{field_name} must not be empty")
    if not all(isinstance(value, enum_type) for value in values):
        raise ScheduleContractError(f"{field_name} contains an unsupported value")
    if len(values) != len(set(values)):
        raise ScheduleContractError(f"{field_name} contains duplicate values")
    order = {value: index for index, value in enumerate(enum_type)}
    if tuple(values) != tuple(sorted(values, key=order.__getitem__)):
        raise ScheduleContractError(f"{field_name} must use canonical enum order")
    return cast(tuple[Enum, ...], values)


def _require_feature_tuple(
    field_name: str,
    values: object,
    *,
    allow_empty: bool,
) -> tuple[AdvancedWorkflowFeature, ...]:
    return cast(
        tuple[AdvancedWorkflowFeature, ...],
        _require_enum_tuple(
            field_name,
            values,
            AdvancedWorkflowFeature,
            allow_empty=allow_empty,
        ),
    )


def _require_state_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ScheduleContractError("required_state must be a tuple")
    normalized = tuple(_require_identifier("required_state item", value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ScheduleContractError("required_state contains duplicate values")
    if normalized != tuple(sorted(normalized)):
        raise ScheduleContractError("required_state must use canonical identifier order")
    return normalized


def _decode_json(payload: bytes | str) -> dict[str, object]:
    if isinstance(payload, bytes):
        if len(payload) > _MAX_PAYLOAD_BYTES:
            raise ScheduleContractError("compatibility payload exceeds the byte limit")
        raw_payload = payload
    elif isinstance(payload, str):
        if len(payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise ScheduleContractError("compatibility payload exceeds the byte limit")
        raw_payload = payload.encode("utf-8")
    else:
        raise ScheduleContractError("compatibility payload must be bytes or text")
    try:
        raw = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScheduleContractError("compatibility payload must be canonical JSON") from error
    if not isinstance(raw, dict):
        raise ScheduleContractError("compatibility payload must be an object")
    projection = cast(dict[str, object], raw)
    try:
        canonical = canonical_projection_bytes(projection)
    except ScheduleContractError as error:
        raise ScheduleContractError("compatibility payload must be canonical JSON") from error
    if canonical != raw_payload:
        raise ScheduleContractError("compatibility payload must be canonical JSON")
    return projection


def _exact_fields(value: dict[str, object], expected: set[str], field_name: str) -> None:
    if set(value) != expected:
        raise ScheduleContractError(f"{field_name} fields do not match its schema")


@dataclass(frozen=True, slots=True, kw_only=True)
class AdvancedWorkflowRequest:
    """Bounded request describing advanced features without framework objects."""

    features: tuple[AdvancedWorkflowFeature, ...]
    execution_mode: AdvancedExecutionMode
    host_capabilities: tuple[AdvancedWorkflowFeature, ...] = ()
    required_state: tuple[str, ...] = ()
    snapshot_fingerprint: str | None = None
    spec_fingerprint: str | None = None
    snapshot_spec_fingerprint: str | None = None
    model_object_supplied: bool = False
    patch_object_supplied: bool = False

    def __post_init__(self) -> None:
        _require_feature_tuple("features", self.features, allow_empty=False)
        if not isinstance(self.execution_mode, AdvancedExecutionMode):
            raise ScheduleContractError("execution_mode is unsupported")
        _require_feature_tuple("host_capabilities", self.host_capabilities, allow_empty=True)
        _require_state_tuple(self.required_state)
        _require_fingerprint("snapshot_fingerprint", self.snapshot_fingerprint)
        _require_fingerprint("spec_fingerprint", self.spec_fingerprint)
        _require_fingerprint("snapshot_spec_fingerprint", self.snapshot_spec_fingerprint)
        _require_bool("model_object_supplied", self.model_object_supplied)
        _require_bool("patch_object_supplied", self.patch_object_supplied)

    def projection(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode.value,
            "features": [item.value for item in self.features],
            "host_capabilities": [item.value for item in self.host_capabilities],
            "model_object_supplied": self.model_object_supplied,
            "patch_object_supplied": self.patch_object_supplied,
            "required_state": list(self.required_state),
            "schema": ADVANCED_WORKFLOW_REQUEST_SCHEMA,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "snapshot_spec_fingerprint": self.snapshot_spec_fingerprint,
            "spec_fingerprint": self.spec_fingerprint,
        }

    @property
    def request_fingerprint(self) -> str:
        return _sha256_identity(canonical_projection_bytes(self.projection()))


def _request_from_projection(value: object) -> AdvancedWorkflowRequest:
    if not isinstance(value, dict):
        raise ScheduleContractError("request must be an object")
    projection = cast(dict[str, object], value)
    _exact_fields(
        projection,
        {
            "execution_mode",
            "features",
            "host_capabilities",
            "model_object_supplied",
            "patch_object_supplied",
            "required_state",
            "schema",
            "snapshot_fingerprint",
            "snapshot_spec_fingerprint",
            "spec_fingerprint",
        },
        "request",
    )
    if projection["schema"] != ADVANCED_WORKFLOW_REQUEST_SCHEMA:
        raise ScheduleContractError("request schema is unsupported")
    features = projection["features"]
    host_capabilities = projection["host_capabilities"]
    required_state = projection["required_state"]
    if not isinstance(features, list) or not isinstance(host_capabilities, list):
        raise ScheduleContractError("request features must be arrays")
    if not isinstance(required_state, list):
        raise ScheduleContractError("request required_state must be an array")
    try:
        feature_values = tuple(AdvancedWorkflowFeature(cast(str, item)) for item in features)
        host_values = tuple(AdvancedWorkflowFeature(cast(str, item)) for item in host_capabilities)
        mode = AdvancedExecutionMode(cast(str, projection["execution_mode"]))
    except (TypeError, ValueError) as error:
        raise ScheduleContractError("request contains an unknown enum value") from error
    return AdvancedWorkflowRequest(
        features=feature_values,
        execution_mode=mode,
        host_capabilities=host_values,
        required_state=tuple(cast(str, item) for item in required_state),
        snapshot_fingerprint=cast(str | None, projection["snapshot_fingerprint"]),
        spec_fingerprint=cast(str | None, projection["spec_fingerprint"]),
        snapshot_spec_fingerprint=cast(str | None, projection["snapshot_spec_fingerprint"]),
        model_object_supplied=cast(bool, projection["model_object_supplied"]),
        patch_object_supplied=cast(bool, projection["patch_object_supplied"]),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class AdvancedWorkflowDecision:
    """Immutable pre-execution decision bound to one request fingerprint."""

    request_fingerprint: str
    features: tuple[AdvancedWorkflowFeature, ...]
    execution_mode: AdvancedExecutionMode
    ownership: AdvancedOwnership
    required_state: tuple[str, ...]
    level: AdvancedDecisionLevel
    reasons: tuple[AdvancedReasonCode, ...]
    snapshot_fingerprint: str | None = None
    spec_fingerprint: str | None = None
    snapshot_spec_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _require_fingerprint("request_fingerprint", self.request_fingerprint, allow_none=False)
        _require_feature_tuple("features", self.features, allow_empty=False)
        if not isinstance(self.execution_mode, AdvancedExecutionMode):
            raise ScheduleContractError("execution_mode is unsupported")
        if not isinstance(self.ownership, AdvancedOwnership):
            raise ScheduleContractError("ownership is unsupported")
        _require_state_tuple(self.required_state)
        if not isinstance(self.level, AdvancedDecisionLevel):
            raise ScheduleContractError("decision level is unsupported")
        _require_enum_tuple("reasons", self.reasons, AdvancedReasonCode, allow_empty=False)
        _require_fingerprint("snapshot_fingerprint", self.snapshot_fingerprint)
        _require_fingerprint("spec_fingerprint", self.spec_fingerprint)
        _require_fingerprint("snapshot_spec_fingerprint", self.snapshot_spec_fingerprint)
        if self.level is AdvancedDecisionLevel.ALLOW and self.reasons != (
            AdvancedReasonCode.COMPATIBLE,
        ):
            raise ScheduleContractError("ALLOW requires only the compatible reason")
        if self.level is AdvancedDecisionLevel.WARN and self.reasons != (
            AdvancedReasonCode.HOST_INTERRUPT_NON_RESUMABLE,
        ):
            raise ScheduleContractError("WARN requires the host interruption reason")
        if (
            self.level is AdvancedDecisionLevel.REJECT
            and AdvancedReasonCode.COMPATIBLE in self.reasons
        ):
            raise ScheduleContractError("REJECT cannot include the compatible reason")

    def _body_projection(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode.value,
            "features": [item.value for item in self.features],
            "level": self.level.value,
            "ownership": self.ownership.value,
            "reasons": [item.value for item in self.reasons],
            "request_fingerprint": self.request_fingerprint,
            "required_state": list(self.required_state),
            "schema": ADVANCED_WORKFLOW_DECISION_SCHEMA,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "snapshot_spec_fingerprint": self.snapshot_spec_fingerprint,
            "spec_fingerprint": self.spec_fingerprint,
        }

    def projection(self) -> dict[str, object]:
        return {**self._body_projection(), "decision_fingerprint": self.fingerprint}

    @property
    def fingerprint(self) -> str:
        return _sha256_identity(canonical_projection_bytes(self._body_projection()))


def _decision_from_projection(value: object) -> AdvancedWorkflowDecision:
    if not isinstance(value, dict):
        raise ScheduleContractError("decision must be an object")
    projection = cast(dict[str, object], value)
    expected = {
        "decision_fingerprint",
        "execution_mode",
        "features",
        "level",
        "ownership",
        "reasons",
        "request_fingerprint",
        "required_state",
        "schema",
        "snapshot_fingerprint",
        "snapshot_spec_fingerprint",
        "spec_fingerprint",
    }
    _exact_fields(projection, expected, "decision")
    if projection["schema"] != ADVANCED_WORKFLOW_DECISION_SCHEMA:
        raise ScheduleContractError("decision schema is unsupported")
    features = projection["features"]
    reasons = projection["reasons"]
    required_state = projection["required_state"]
    if not isinstance(features, list) or not isinstance(reasons, list):
        raise ScheduleContractError("decision features and reasons must be arrays")
    if not isinstance(required_state, list):
        raise ScheduleContractError("decision required_state must be an array")
    try:
        decision = AdvancedWorkflowDecision(
            request_fingerprint=cast(str, projection["request_fingerprint"]),
            features=tuple(AdvancedWorkflowFeature(cast(str, item)) for item in features),
            execution_mode=AdvancedExecutionMode(cast(str, projection["execution_mode"])),
            ownership=AdvancedOwnership(cast(str, projection["ownership"])),
            required_state=tuple(cast(str, item) for item in required_state),
            level=AdvancedDecisionLevel(cast(str, projection["level"])),
            reasons=tuple(AdvancedReasonCode(cast(str, item)) for item in reasons),
            snapshot_fingerprint=cast(str | None, projection["snapshot_fingerprint"]),
            spec_fingerprint=cast(str | None, projection["spec_fingerprint"]),
            snapshot_spec_fingerprint=cast(str | None, projection["snapshot_spec_fingerprint"]),
        )
    except (TypeError, ValueError) as error:
        raise ScheduleContractError("decision contains an unknown enum value") from error
    if projection["decision_fingerprint"] != decision.fingerprint:
        raise ScheduleContractError("decision fingerprint does not match its projection")
    return decision


def _sorted_reasons(reasons: set[AdvancedReasonCode]) -> tuple[AdvancedReasonCode, ...]:
    order = {reason: index for index, reason in enumerate(AdvancedReasonCode)}
    return tuple(sorted(reasons, key=order.__getitem__))


def _snapshot_reasons(request: AdvancedWorkflowRequest) -> set[AdvancedReasonCode]:
    values = (
        request.snapshot_fingerprint,
        request.spec_fingerprint,
        request.snapshot_spec_fingerprint,
    )
    if any(value is None for value in values):
        return {AdvancedReasonCode.RESUME_SNAPSHOT_REQUIRED}
    if request.snapshot_spec_fingerprint != request.spec_fingerprint:
        return {AdvancedReasonCode.RESUME_SNAPSHOT_MISMATCH}
    return set()


def resolve_advanced_workflow(request: AdvancedWorkflowRequest) -> AdvancedWorkflowDecision:
    """Resolve feature ownership without importing or calling a framework."""

    if not isinstance(request, AdvancedWorkflowRequest):
        raise ScheduleContractError("request must be an AdvancedWorkflowRequest")

    features = set(request.features)
    reasons: set[AdvancedReasonCode] = set()
    mode = request.execution_mode
    ownership = (
        AdvancedOwnership.HOST
        if mode is AdvancedExecutionMode.NATIVE_HOST
        else AdvancedOwnership.CONTROLLER
    )

    if mode is not AdvancedExecutionMode.NATIVE_HOST and request.host_capabilities:
        reasons.add(AdvancedReasonCode.CONTRADICTORY_OWNERSHIP)

    if mode is AdvancedExecutionMode.NATIVE_HOST:
        missing = features - set(request.host_capabilities)
        if missing:
            reasons.add(AdvancedReasonCode.HOST_CAPABILITY_MISSING)
        if AdvancedWorkflowFeature.RESUME in features:
            reasons.add(AdvancedReasonCode.HOST_RESUME_UNSUPPORTED)
        if (
            AdvancedWorkflowFeature.INTERRUPTION in features
            and not request.snapshot_fingerprint
            and not request.spec_fingerprint
            and not request.snapshot_spec_fingerprint
            and not reasons.intersection(
                {
                    AdvancedReasonCode.HOST_CAPABILITY_MISSING,
                    AdvancedReasonCode.HOST_RESUME_UNSUPPORTED,
                }
            )
        ):
            return AdvancedWorkflowDecision(
                request_fingerprint=request.request_fingerprint,
                features=request.features,
                execution_mode=mode,
                ownership=ownership,
                required_state=request.required_state,
                level=AdvancedDecisionLevel.WARN,
                reasons=(AdvancedReasonCode.HOST_INTERRUPT_NON_RESUMABLE,),
                snapshot_fingerprint=request.snapshot_fingerprint,
                spec_fingerprint=request.spec_fingerprint,
                snapshot_spec_fingerprint=request.snapshot_spec_fingerprint,
            )
    elif mode is AdvancedExecutionMode.STOCHASTIC_PURE:
        ownership = AdvancedOwnership.UNSUPPORTED
        if (
            AdvancedWorkflowFeature.PARTIAL_DENOISE in features
            or AdvancedWorkflowFeature.IMAGE_TO_IMAGE in features
        ):
            reasons.add(AdvancedReasonCode.STOCHASTIC_PARTIAL_UNSUPPORTED)
        if AdvancedWorkflowFeature.RESUME in features:
            reasons.add(AdvancedReasonCode.STOCHASTIC_RESUME_UNSUPPORTED)
        if AdvancedWorkflowFeature.INTERRUPTION in features:
            reasons.add(AdvancedReasonCode.STOCHASTIC_INTERRUPTION_UNSUPPORTED)
    else:
        pure_host_features = {
            AdvancedWorkflowFeature.INPAINTING,
            AdvancedWorkflowFeature.CONTROLNET,
            AdvancedWorkflowFeature.MODEL_PATCHES,
        }
        if features.intersection(pure_host_features):
            reasons.add(AdvancedReasonCode.PURE_HOST_FEATURE_UNSUPPORTED)
        if request.model_object_supplied or request.patch_object_supplied:
            reasons.add(AdvancedReasonCode.PURE_MODEL_OWNERSHIP_UNSUPPORTED)
        if (
            AdvancedWorkflowFeature.RESUME in features
            or AdvancedWorkflowFeature.INTERRUPTION in features
        ):
            reasons.update(_snapshot_reasons(request))

    if reasons:
        ownership = (
            ownership
            if mode is AdvancedExecutionMode.NATIVE_HOST
            and reasons <= {AdvancedReasonCode.HOST_INTERRUPT_NON_RESUMABLE}
            else AdvancedOwnership.UNSUPPORTED
            if mode is not AdvancedExecutionMode.NATIVE_HOST
            else ownership
        )
        return AdvancedWorkflowDecision(
            request_fingerprint=request.request_fingerprint,
            features=request.features,
            execution_mode=mode,
            ownership=ownership,
            required_state=request.required_state,
            level=AdvancedDecisionLevel.REJECT,
            reasons=_sorted_reasons(reasons),
            snapshot_fingerprint=request.snapshot_fingerprint,
            spec_fingerprint=request.spec_fingerprint,
            snapshot_spec_fingerprint=request.snapshot_spec_fingerprint,
        )

    return AdvancedWorkflowDecision(
        request_fingerprint=request.request_fingerprint,
        features=request.features,
        execution_mode=mode,
        ownership=ownership,
        required_state=request.required_state,
        level=AdvancedDecisionLevel.ALLOW,
        reasons=(AdvancedReasonCode.COMPATIBLE,),
        snapshot_fingerprint=request.snapshot_fingerprint,
        spec_fingerprint=request.spec_fingerprint,
        snapshot_spec_fingerprint=request.snapshot_spec_fingerprint,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class AdvancedWorkflowCompatibilityReceipt:
    """Portable decision receipt with no raw workflow or framework data."""

    request: AdvancedWorkflowRequest
    decision: AdvancedWorkflowDecision
    execution_status: AdvancedReceiptStatus
    resumable: bool
    snapshot_fingerprint: str | None = None
    result_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, AdvancedWorkflowRequest):
            raise ScheduleContractError("receipt request is unsupported")
        if not isinstance(self.decision, AdvancedWorkflowDecision):
            raise ScheduleContractError("receipt decision is unsupported")
        if self.decision.request_fingerprint != self.request.request_fingerprint:
            raise ScheduleContractError("decision does not match request")
        if not isinstance(self.execution_status, AdvancedReceiptStatus):
            raise ScheduleContractError("receipt execution status is unsupported")
        _require_bool("resumable", self.resumable)
        _require_fingerprint("snapshot_fingerprint", self.snapshot_fingerprint)
        _require_fingerprint("result_fingerprint", self.result_fingerprint)
        if self.snapshot_fingerprint != self.request.snapshot_fingerprint:
            raise ScheduleContractError("receipt snapshot fingerprint does not match request")
        if self.decision.level is AdvancedDecisionLevel.REJECT:
            if self.execution_status is not AdvancedReceiptStatus.REJECTED or self.resumable:
                raise ScheduleContractError(
                    "rejected decision requires a non-resumable rejected receipt"
                )
        elif self.execution_status is AdvancedReceiptStatus.REJECTED:
            raise ScheduleContractError("allowed decision cannot have rejected receipt status")
        if self.execution_status is AdvancedReceiptStatus.RESUMABLE:
            if not self.resumable or self.snapshot_fingerprint is None:
                raise ScheduleContractError("resumable receipt requires a snapshot fingerprint")
        elif self.resumable:
            raise ScheduleContractError("only resumable receipts may claim resumability")
        if (
            self.execution_status is AdvancedReceiptStatus.SUCCEEDED
            and self.result_fingerprint is None
        ):
            raise ScheduleContractError("succeeded receipt requires a result fingerprint")
        if (
            self.execution_status
            in {
                AdvancedReceiptStatus.NOT_EXECUTED,
                AdvancedReceiptStatus.REJECTED,
                AdvancedReceiptStatus.INTERRUPTED,
            }
            and self.result_fingerprint is not None
        ):
            raise ScheduleContractError("non-completed receipt cannot contain a result fingerprint")

    def _body_projection(self) -> dict[str, object]:
        return {
            "decision": self.decision.projection(),
            "execution_status": self.execution_status.value,
            "request": self.request.projection(),
            "resumable": self.resumable,
            "result_fingerprint": self.result_fingerprint,
            "schema": ADVANCED_WORKFLOW_COMPATIBILITY_SCHEMA,
            "snapshot_fingerprint": self.snapshot_fingerprint,
        }

    def projection(self) -> dict[str, object]:
        return {**self._body_projection(), "receipt_fingerprint": self.receipt_fingerprint}

    @property
    def receipt_fingerprint(self) -> str:
        return _sha256_identity(canonical_projection_bytes(self._body_projection()))


def build_advanced_workflow_receipt(
    request: AdvancedWorkflowRequest,
    decision: AdvancedWorkflowDecision,
    *,
    execution_status: AdvancedReceiptStatus | None = None,
    resumable: bool = False,
    result_fingerprint: str | None = None,
) -> AdvancedWorkflowCompatibilityReceipt:
    """Build a receipt while deriving the truthful default status from the decision."""

    if not isinstance(request, AdvancedWorkflowRequest):
        raise ScheduleContractError("request must be an AdvancedWorkflowRequest")
    if not isinstance(decision, AdvancedWorkflowDecision):
        raise ScheduleContractError("decision must be an AdvancedWorkflowDecision")
    if decision.request_fingerprint != request.request_fingerprint:
        raise ScheduleContractError("decision does not match request")
    if execution_status is None:
        execution_status = (
            AdvancedReceiptStatus.REJECTED
            if decision.level is AdvancedDecisionLevel.REJECT
            else AdvancedReceiptStatus.NOT_EXECUTED
        )
    return AdvancedWorkflowCompatibilityReceipt(
        request=request,
        decision=decision,
        execution_status=execution_status,
        resumable=resumable,
        snapshot_fingerprint=request.snapshot_fingerprint,
        result_fingerprint=result_fingerprint,
    )


def serialize_advanced_workflow_receipt(
    receipt: AdvancedWorkflowCompatibilityReceipt,
) -> bytes:
    """Serialize one receipt as canonical JSON bytes."""

    if not isinstance(receipt, AdvancedWorkflowCompatibilityReceipt):
        raise ScheduleContractError("receipt must be an AdvancedWorkflowCompatibilityReceipt")
    return canonical_projection_bytes(receipt.projection())


def deserialize_advanced_workflow_receipt(
    payload: bytes | str,
) -> AdvancedWorkflowCompatibilityReceipt:
    """Deserialize and verify one canonical compatibility receipt."""

    raw = _decode_json(payload)
    expected = {
        "decision",
        "execution_status",
        "receipt_fingerprint",
        "request",
        "resumable",
        "result_fingerprint",
        "schema",
        "snapshot_fingerprint",
    }
    _exact_fields(raw, expected, "receipt")
    if raw["schema"] != ADVANCED_WORKFLOW_COMPATIBILITY_SCHEMA:
        raise ScheduleContractError("receipt schema is unsupported")
    try:
        status = AdvancedReceiptStatus(cast(str, raw["execution_status"]))
    except (TypeError, ValueError) as error:
        raise ScheduleContractError("receipt execution status is unsupported") from error
    receipt = AdvancedWorkflowCompatibilityReceipt(
        request=_request_from_projection(raw["request"]),
        decision=_decision_from_projection(raw["decision"]),
        execution_status=status,
        resumable=cast(bool, raw["resumable"]),
        snapshot_fingerprint=cast(str | None, raw["snapshot_fingerprint"]),
        result_fingerprint=cast(str | None, raw["result_fingerprint"]),
    )
    if raw["receipt_fingerprint"] != receipt.receipt_fingerprint:
        raise ScheduleContractError("receipt fingerprint does not match its projection")
    return receipt


__all__ = [
    "ADVANCED_WORKFLOW_COMPATIBILITY_SCHEMA",
    "ADVANCED_WORKFLOW_DECISION_SCHEMA",
    "ADVANCED_WORKFLOW_REQUEST_SCHEMA",
    "AdvancedDecisionLevel",
    "AdvancedExecutionMode",
    "AdvancedOwnership",
    "AdvancedReasonCode",
    "AdvancedReceiptStatus",
    "AdvancedWorkflowCompatibilityReceipt",
    "AdvancedWorkflowDecision",
    "AdvancedWorkflowFeature",
    "AdvancedWorkflowRequest",
    "build_advanced_workflow_receipt",
    "deserialize_advanced_workflow_receipt",
    "resolve_advanced_workflow",
    "serialize_advanced_workflow_receipt",
]
