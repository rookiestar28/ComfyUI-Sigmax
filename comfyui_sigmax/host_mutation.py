"""Pure co-installation snapshots and protected host-mutation detection."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

HOST_MUTATION_SNAPSHOT_SCHEMA: Final = "sigmax.host-mutation-snapshot/1"
COINSTALLATION_EVALUATION_SCHEMA: Final = "sigmax.co-installation-evaluation/1"
_FINGERPRINT: Final = re.compile(r"sha256:[0-9a-f]{64}")
_NODE_ID: Final = re.compile(r"[A-Z][A-Za-z0-9]{0,63}(?:\.[A-Z][A-Za-z0-9]{0,63})+")
_PACK_ID: Final = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+){1,15}")
_PROVIDER_ID: Final = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_SCHEDULER_NAME: Final = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_OWNERSHIP: Final = frozenset({"external_sigmas", "model_native"})


class MutationVerdict(str, Enum):
    """Stable co-installation decision."""

    ALLOW = "allow"
    REJECT = "reject"


class HostMutationFinding(str, Enum):
    """Protected host seams detected between two immutable snapshots."""

    CONSTRUCTION_SHIFT_REPEATED = "construction_shift_repeated"
    MODEL_NATIVE_EXTERNAL_DOUBLE_SHIFT = "model_native_external_double_shift"
    MODEL_PATCH_STATE_CHANGED = "model_patch_state_changed"
    NODE_REGISTRY_COLLISION = "node_registry_collision"
    SCHEDULER_REGISTRY_OVERWRITE = "scheduler_registry_overwrite"
    SIGMAX_NAMESPACE_HIJACK = "sigmax_namespace_hijack"
    TORCH_CALL_PATH_CHANGED = "torch_call_path_changed"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _fingerprint(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT.fullmatch(value):
        raise ScheduleContractError(f"{label} must be a lowercase SHA-256 identity")
    return value


def _pack_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _PACK_ID.fullmatch(value):
        raise ScheduleContractError(f"{label} must be a stable namespaced identifier")
    return value


def _provider_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _PROVIDER_ID.fullmatch(value):
        raise ScheduleContractError(f"{label} must be a stable provider identifier")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class NodeRegistryIdentity:
    """Stable semantic identity for one global node registration."""

    node_id: str
    provider: str
    definition_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not _NODE_ID.fullmatch(self.node_id):
            raise ScheduleContractError("node registry identity has an invalid node ID")
        _provider_id(self.provider, label="node registry provider")
        _fingerprint(
            self.definition_fingerprint,
            label="node definition fingerprint",
        )

    def projection(self) -> dict[str, str]:
        return {
            "definition_fingerprint": self.definition_fingerprint,
            "node_id": self.node_id,
            "provider": self.provider,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerRegistryIdentity:
    """Stable semantic identity for one global scheduler handler."""

    name: str
    provider: str
    handler_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SCHEDULER_NAME.fullmatch(self.name):
            raise ScheduleContractError("scheduler registry identity has an invalid name")
        _provider_id(self.provider, label="scheduler registry provider")
        _fingerprint(
            self.handler_fingerprint,
            label="scheduler handler fingerprint",
        )

    def projection(self) -> dict[str, str]:
        return {
            "handler_fingerprint": self.handler_fingerprint,
            "name": self.name,
            "provider": self.provider,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class HostMutationSnapshot:
    """Immutable protected host state before or after one pack operation."""

    nodes: tuple[NodeRegistryIdentity, ...]
    schedulers: tuple[SchedulerRegistryIdentity, ...]
    torch_call_fingerprint: str
    model_patch_fingerprint: str
    schedule_ownership: str
    construction_shift_count: int
    model_native_shifted: bool

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple) or any(
            not isinstance(item, NodeRegistryIdentity) for item in self.nodes
        ):
            raise ScheduleContractError("host snapshot nodes must be immutable identities")
        if not isinstance(self.schedulers, tuple) or any(
            not isinstance(item, SchedulerRegistryIdentity) for item in self.schedulers
        ):
            raise ScheduleContractError("host snapshot schedulers must be immutable identities")
        node_ids = tuple(item.node_id for item in self.nodes)
        scheduler_names = tuple(item.name for item in self.schedulers)
        if node_ids != tuple(sorted(set(node_ids))):
            raise ScheduleContractError("host snapshot node IDs must be unique and sorted")
        if scheduler_names != tuple(sorted(set(scheduler_names))):
            raise ScheduleContractError("host snapshot scheduler names must be unique and sorted")
        _fingerprint(self.torch_call_fingerprint, label="PyTorch call-path fingerprint")
        _fingerprint(self.model_patch_fingerprint, label="model-patch fingerprint")
        if self.schedule_ownership not in _OWNERSHIP:
            raise ScheduleContractError("host snapshot schedule ownership is invalid")
        if type(self.construction_shift_count) is not int or self.construction_shift_count < 0:
            raise ScheduleContractError("construction shift count must be a non-negative integer")
        if type(self.model_native_shifted) is not bool:
            raise ScheduleContractError("model-native shifted state must be boolean")

    @property
    def snapshot_fingerprint(self) -> str:
        return _identity(self.projection())

    def projection(self) -> dict[str, object]:
        return {
            "construction_shift_count": self.construction_shift_count,
            "model_native_shifted": self.model_native_shifted,
            "model_patch_fingerprint": self.model_patch_fingerprint,
            "nodes": [item.projection() for item in self.nodes],
            "schedule_ownership": self.schedule_ownership,
            "schedulers": [item.projection() for item in self.schedulers],
            "schema": HOST_MUTATION_SNAPSHOT_SCHEMA,
            "torch_call_fingerprint": self.torch_call_fingerprint,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CoInstallationEvaluation:
    """Stable allow/reject report for one synthetic or approved pack observation."""

    pack_id: str
    before_fingerprint: str
    after_fingerprint: str
    verdict: MutationVerdict
    findings: tuple[HostMutationFinding, ...]

    def __post_init__(self) -> None:
        _pack_id(self.pack_id, label="co-installation pack ID")
        _fingerprint(self.before_fingerprint, label="before-snapshot fingerprint")
        _fingerprint(self.after_fingerprint, label="after-snapshot fingerprint")
        if not isinstance(self.verdict, MutationVerdict):
            raise ScheduleContractError("co-installation verdict is invalid")
        if not isinstance(self.findings, tuple) or any(
            not isinstance(item, HostMutationFinding) for item in self.findings
        ):
            raise ScheduleContractError("co-installation findings must be immutable")
        if self.findings != tuple(sorted(set(self.findings), key=lambda item: item.value)):
            raise ScheduleContractError("co-installation findings must be unique and sorted")
        if (self.verdict is MutationVerdict.ALLOW) != (not self.findings):
            raise ScheduleContractError("co-installation verdict disagrees with findings")

    @property
    def report_fingerprint(self) -> str:
        return _identity(self.projection())

    def projection(self) -> dict[str, object]:
        return {
            "after_fingerprint": self.after_fingerprint,
            "before_fingerprint": self.before_fingerprint,
            "findings": [item.value for item in self.findings],
            "pack_id": self.pack_id,
            "schema": COINSTALLATION_EVALUATION_SCHEMA,
            "verdict": self.verdict.value,
        }


def evaluate_host_mutation(
    *,
    before: HostMutationSnapshot,
    after: HostMutationSnapshot,
    pack_id: str,
) -> CoInstallationEvaluation:
    """Detect changes to protected existing identities and schedule ownership."""

    if not isinstance(before, HostMutationSnapshot) or not isinstance(after, HostMutationSnapshot):
        raise ScheduleContractError("co-installation evaluation requires host snapshots")
    stable_pack_id = _pack_id(pack_id, label="co-installation pack ID")
    findings: set[HostMutationFinding] = set()

    before_nodes = {item.node_id: item for item in before.nodes}
    after_nodes = {item.node_id: item for item in after.nodes}
    if any(after_nodes.get(node_id) != identity for node_id, identity in before_nodes.items()):
        findings.add(HostMutationFinding.NODE_REGISTRY_COLLISION)
    if any(
        node_id.startswith("Sigmax.")
        and node_id not in before_nodes
        and identity.provider != "comfyui_sigmax"
        for node_id, identity in after_nodes.items()
    ):
        findings.add(HostMutationFinding.SIGMAX_NAMESPACE_HIJACK)

    before_schedulers = {item.name: item for item in before.schedulers}
    after_schedulers = {item.name: item for item in after.schedulers}
    if any(after_schedulers.get(name) != identity for name, identity in before_schedulers.items()):
        findings.add(HostMutationFinding.SCHEDULER_REGISTRY_OVERWRITE)

    if before.torch_call_fingerprint != after.torch_call_fingerprint:
        findings.add(HostMutationFinding.TORCH_CALL_PATH_CHANGED)
    if before.model_patch_fingerprint != after.model_patch_fingerprint:
        findings.add(HostMutationFinding.MODEL_PATCH_STATE_CHANGED)
    if after.construction_shift_count > 1:
        findings.add(HostMutationFinding.CONSTRUCTION_SHIFT_REPEATED)
    if after.construction_shift_count > 0 and (
        after.model_native_shifted or after.schedule_ownership == "model_native"
    ):
        findings.add(HostMutationFinding.MODEL_NATIVE_EXTERNAL_DOUBLE_SHIFT)

    ordered = tuple(sorted(findings, key=lambda item: item.value))
    return CoInstallationEvaluation(
        pack_id=stable_pack_id,
        before_fingerprint=before.snapshot_fingerprint,
        after_fingerprint=after.snapshot_fingerprint,
        verdict=MutationVerdict.REJECT if ordered else MutationVerdict.ALLOW,
        findings=ordered,
    )


__all__ = [
    "COINSTALLATION_EVALUATION_SCHEMA",
    "HOST_MUTATION_SNAPSHOT_SCHEMA",
    "CoInstallationEvaluation",
    "HostMutationFinding",
    "HostMutationSnapshot",
    "MutationVerdict",
    "NodeRegistryIdentity",
    "SchedulerRegistryIdentity",
    "evaluate_host_mutation",
]
