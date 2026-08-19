"""Ephemeral H4 dispatch observation nodes and testable tracing helpers.

This module is copied into a harness-owned temporary custom-node package.  It is intentionally
dependency-free at import time: ComfyUI, Torch, and Comfy Kitchen are imported only after the
observer node is executed inside the explicitly authorized validation host.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Final

ADAPTER_VERSION: Final = "m7-13-h4-dispatch-observer/1"
TRACE_SCHEMA: Final = "sigmax.h4-dispatch-observation/1"
MAX_TRACE_BYTES: Final = 64 * 1024
MAX_EVENTS: Final = 32
MAX_REASON_CODES: Final = 16

_HOOK_MARKER: Final = "_sigmax_h4_dispatch_observer_hook"
_WRAPPED_MARKER: Final = "_sigmax_h4_dispatch_observer_wrapped"
_ALLOWED_OPERATION_NAMES: Final = frozenset(
    {
        "convrot_w4a4_linear",
        "gemv_awq_w4a16",
        "int8_linear",
        "scaled_mm_mxfp8",
        "scaled_mm_nvfp4",
        "scaled_mm_svdquant_w4a4",
        "w4a8_int8_linear",
    }
)
_ALLOWED_OPERATION_BACKENDS: Final = frozenset({"cuda", "eager", "hip", "triton"})
_ATTENTION_FUNCTION_NAMES: Final = {
    "ck_int8": "comfy_kitchen_int8",
    "flash": "flash",
    "pytorch": "pytorch",
    "sage": "sage",
    "sage3": "sage3",
    "split": "split",
    "sub_quad": "sub_quad",
    "xformers": "xformers",
}
_ALLOWED_ATTENTION_BACKENDS: Final = frozenset(_ATTENTION_FUNCTION_NAMES)
_ALLOWED_OPERATION_REQUESTS: Final = frozenset({"auto", "unavailable"})


class DispatchObserverError(RuntimeError):
    """Raised when the bounded observer cannot preserve its lifecycle contract."""


def _backend_from_callable(value: object) -> str | None:
    """Map only an allowlisted Comfy Kitchen backend module to its public backend name."""

    module = getattr(value, "__module__", None)
    if not isinstance(module, str):
        return None
    parts = module.split(".")
    if len(parts) < 3 or tuple(parts[:2]) != ("comfy_kitchen", "backends"):
        return None
    backend = parts[2]
    return backend if backend in _ALLOWED_OPERATION_BACKENDS else None


def _safe_trace_file(value: object) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        path = Path(value)
    else:
        raise DispatchObserverError("dispatch trace file must be text")
    if "\x00" in str(path) or not path.is_absolute() or ".." in path.parts:
        raise DispatchObserverError("dispatch trace file must be absolute and traversal-free")
    return path


def _append_bounded(events: list[dict[str, object]], event: dict[str, object]) -> None:
    if len(events) < MAX_EVENTS:
        events.append(event)


@dataclass
class _TraceState:
    """In-memory bounded state for one observer arm/disarm lifecycle."""

    trace_file: Path
    requested_attention_backend: str
    requested_operation_backend: str
    status: str = "NEW"
    operation_call_count: int = 0
    operation_backend_counts: dict[str, int] = field(default_factory=dict)
    operation_events: list[dict[str, object]] = field(default_factory=list)
    attention_call_count: int = 0
    attention_success_count: int = 0
    attention_failure_count: int = 0
    attention_backend_counts: dict[str, int] = field(default_factory=dict)
    attention_events: list[dict[str, object]] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    registry_method: object | None = None
    registry_hook: object | None = None

    def __post_init__(self) -> None:
        self.trace_file = _safe_trace_file(self.trace_file)
        if self.requested_attention_backend not in _ALLOWED_ATTENTION_BACKENDS:
            raise DispatchObserverError("requested attention backend is not allowlisted")
        if self.requested_operation_backend not in _ALLOWED_OPERATION_REQUESTS:
            raise DispatchObserverError("requested operation backend is not allowlisted")

    def add_reason(self, reason: str) -> None:
        if reason not in self.reason_codes and len(self.reason_codes) < MAX_REASON_CODES:
            self.reason_codes.append(reason)

    def _maybe_write(self, count: int) -> None:
        # Keep a crash-visible first event and bounded periodic checkpoints without serializing
        # callable arguments, tensors, prompts, paths, or exceptions.
        if count == 1 or count % MAX_EVENTS == 0:
            self.write()

    def record_operation(self, operation: str, backend: str, outcome: str) -> None:
        self.operation_call_count += 1
        self.operation_backend_counts[backend] = self.operation_backend_counts.get(backend, 0) + 1
        _append_bounded(
            self.operation_events,
            {
                "backend": backend,
                "operation": operation,
                "ordinal": self.operation_call_count,
                "outcome": outcome,
            },
        )
        self._maybe_write(self.operation_call_count)

    def record_attention(self, backend: str, outcome: str) -> None:
        self.attention_call_count += 1
        self.attention_backend_counts[backend] = self.attention_backend_counts.get(backend, 0) + 1
        if outcome == "returned":
            self.attention_success_count += 1
        else:
            self.attention_failure_count += 1
        _append_bounded(
            self.attention_events,
            {
                "backend": backend,
                "ordinal": self.attention_call_count,
                "outcome": outcome,
            },
        )
        self._maybe_write(self.attention_call_count)

    def wrap_operation(
        self, operation: str, implementation: Callable[..., object]
    ) -> Callable[..., object]:
        """Wrap one allowlisted returned implementation without changing its call contract."""

        if operation not in _ALLOWED_OPERATION_NAMES:
            return implementation
        if getattr(implementation, _WRAPPED_MARKER, False):
            raise DispatchObserverError("operation callable is already wrapped")
        backend = _backend_from_callable(implementation)
        if backend is None:
            self.add_reason("operation_callable_backend_unavailable")
            return implementation

        @wraps(implementation)
        def observed(*args: object, **kwargs: object) -> object:
            try:
                result = implementation(*args, **kwargs)
            except Exception:
                self.record_operation(operation, backend, "raised")
                raise
            self.record_operation(operation, backend, "returned")
            return result

        setattr(observed, _WRAPPED_MARKER, ADAPTER_VERSION)
        return observed

    def write(self) -> None:
        projected = _project_trace(self)
        encoded = json.dumps(
            projected, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > MAX_TRACE_BYTES:
            raise DispatchObserverError("dispatch trace exceeds bounded size")
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self.trace_file.write_bytes(encoded + b"\n")


def _project_trace(state: _TraceState) -> dict[str, object]:
    operation_backends = tuple(state.operation_backend_counts)
    actual_operation = (
        operation_backends[0]
        if state.operation_call_count > 0 and len(operation_backends) == 1
        else "not_observed"
    )
    attention_backends = tuple(state.attention_backend_counts)
    actual_attention = (
        attention_backends[0]
        if state.attention_call_count > 0 and len(attention_backends) == 1
        else "not_observed"
    )
    observation_source = (
        "authorized_host_dispatch"
        if actual_operation != "not_observed" or actual_attention != "not_observed"
        else "not_observed"
    )
    return {
        "adapter_version": ADAPTER_VERSION,
        "actual_attention_backend": actual_attention,
        "actual_operation_backend": actual_operation,
        "attention": {
            "backend_counts": dict(state.attention_backend_counts),
            "calls": state.attention_call_count,
            "events": list(state.attention_events),
            "failed_calls": state.attention_failure_count,
            "successful_calls": state.attention_success_count,
        },
        "component_versions": {},
        "observation_source": observation_source,
        "operation": {
            "backend_counts": dict(state.operation_backend_counts),
            "calls": state.operation_call_count,
            "events": list(state.operation_events),
        },
        "reason_codes": list(state.reason_codes),
        "requested_attention_backend": state.requested_attention_backend,
        "requested_operation_backend": state.requested_operation_backend,
        "schema": TRACE_SCHEMA,
        "status": state.status,
    }


def _wrap_attention_delegate(
    state: _TraceState, backend: str, function: Callable[..., object]
) -> Callable[..., object]:
    @wraps(function)
    def observed(*args: object, **kwargs: object) -> object:
        try:
            result = function(*args, **kwargs)
        except Exception:
            state.record_attention(backend, "raised")
            raise
        state.record_attention(backend, "returned")
        return result

    setattr(observed, _WRAPPED_MARKER, ADAPTER_VERSION)
    return observed


def _wrap_attention_callable(
    state: _TraceState, backend: str, function: Callable[..., object]
) -> Callable[..., object]:
    """Wrap the exact host attention callable selected by the requested semantic label."""

    if backend not in _ALLOWED_ATTENTION_BACKENDS:
        raise DispatchObserverError("attention backend is not allowlisted")
    if getattr(function, _WRAPPED_MARKER, False):
        raise DispatchObserverError("attention callable is already wrapped")
    module = getattr(function, "__module__", None)
    name = getattr(function, "__name__", None)
    # ComfyUI's wrap_attn decorator exposes the bounded host module and the generic name
    # ``wrapper``; un-decorated test/host callables may retain the semantic function name.
    if module != "comfy.ldm.modules.attention" or name not in {
        "wrapper",
        f"attention_{_ATTENTION_FUNCTION_NAMES[backend]}",
    }:
        raise DispatchObserverError("attention callable identity is not allowlisted")
    observed = _wrap_attention_delegate(state, backend, function)
    container = getattr(function, "container_function", None)
    if container is not None:
        if not callable(container):
            raise DispatchObserverError("attention container callable is invalid")
        observed.container_function = _wrap_attention_delegate(state, backend, container)  # type: ignore[attr-defined]
    return observed


def _arm_registry(state: _TraceState, registry: object) -> None:
    if state.status != "NEW":
        raise DispatchObserverError("observer state is not new")
    original = getattr(registry, "get_implementation", None)
    if not callable(original):
        raise DispatchObserverError("Comfy Kitchen registry method is unavailable")
    if getattr(original, _HOOK_MARKER, False):
        raise DispatchObserverError("Comfy Kitchen registry is already observed")

    @wraps(original)
    def observed(function_name: str, *args: object, **kwargs: object) -> object:
        implementation = original(function_name, *args, **kwargs)
        if function_name not in _ALLOWED_OPERATION_NAMES:
            return implementation
        if not callable(implementation):
            state.add_reason("operation_callable_unavailable")
            return implementation
        return state.wrap_operation(function_name, implementation)

    setattr(observed, _HOOK_MARKER, ADAPTER_VERSION)
    state.registry_method = original
    state.registry_hook = observed
    namespace = getattr(registry, "__dict__", None)
    if not isinstance(namespace, dict):
        raise DispatchObserverError("Comfy Kitchen registry does not expose an instance namespace")
    namespace["get_implementation"] = observed
    if getattr(registry, "get_implementation", None) is not observed:
        raise DispatchObserverError("Comfy Kitchen registry hook could not be installed")
    state.status = "ARMED"
    state.write()


def _disarm_registry(state: _TraceState, registry: object) -> None:
    if state.status == "DISARMED":
        return
    if state.status != "ARMED" or state.registry_method is None or state.registry_hook is None:
        raise DispatchObserverError("observer registry lifecycle is not armed")
    if getattr(registry, "get_implementation", None) is not state.registry_hook:
        raise DispatchObserverError("Comfy Kitchen registry hook identity changed")
    namespace = getattr(registry, "__dict__", None)
    if not isinstance(namespace, dict):
        raise DispatchObserverError("Comfy Kitchen registry does not expose an instance namespace")
    namespace["get_implementation"] = state.registry_method
    if getattr(registry, "get_implementation", None) is not state.registry_method:
        raise DispatchObserverError("Comfy Kitchen registry hook could not be restored")
    state.status = "DISARMED"
    state.write()


_ACTIVE_STATE: _TraceState | None = None
_ACTIVE_REGISTRY: object | None = None


def install(
    trace_file: str | Path,
    requested_attention_backend: str,
    requested_operation_backend: str = "auto",
) -> _TraceState:
    """Install one process-local registry observer in the isolated H4 host."""

    global _ACTIVE_REGISTRY, _ACTIVE_STATE
    if _ACTIVE_STATE is not None:
        raise DispatchObserverError("a dispatch observer is already active")
    state = _TraceState(
        trace_file=_safe_trace_file(trace_file),
        requested_attention_backend=requested_attention_backend,
        requested_operation_backend=requested_operation_backend,
    )
    registry_module = importlib.import_module("comfy_kitchen.registry")
    registry = getattr(registry_module, "registry", None)
    if registry is None:
        raise DispatchObserverError("Comfy Kitchen registry singleton is unavailable")
    _arm_registry(state, registry)
    _ACTIVE_STATE = state
    _ACTIVE_REGISTRY = registry
    return state


def install_model(
    model: object,
    trace_file: str | Path,
    requested_attention_backend: str,
    requested_operation_backend: str = "auto",
) -> object:
    """Clone a model and install a model-scoped attention observer on the clone."""

    state = install(trace_file, requested_attention_backend, requested_operation_backend)
    try:
        clone = getattr(model, "clone", None)
        if not callable(clone):
            raise DispatchObserverError("model does not expose a clone method")
        observed_model = clone()
        if observed_model is model:
            raise DispatchObserverError("model clone would mutate the source model")
        attention_module = importlib.import_module("comfy.ldm.modules.attention")
        resolver = getattr(attention_module, "get_attention_function", None)
        if not callable(resolver):
            raise DispatchObserverError("ComfyUI attention resolver is unavailable")
        host_name = _ATTENTION_FUNCTION_NAMES[requested_attention_backend]
        selected = resolver(host_name)
        if not callable(selected):
            raise DispatchObserverError("selected attention callable is unavailable")
        observed_attention = _wrap_attention_callable(state, requested_attention_backend, selected)
        setter = getattr(observed_model, "set_model_optimized_attention", None)
        if not callable(setter):
            raise DispatchObserverError("model clone lacks scoped attention setter")
        setter(observed_attention)
        state.write()
        return observed_model
    except Exception as exc:
        try:
            disarm()
        except Exception as restore_error:
            raise DispatchObserverError(
                "observer cleanup failed after arm error"
            ) from restore_error
        raise exc


def disarm() -> None:
    """Restore the registry method and write the final bounded trace."""

    global _ACTIVE_REGISTRY, _ACTIVE_STATE
    if _ACTIVE_STATE is None or _ACTIVE_REGISTRY is None:
        return
    state = _ACTIVE_STATE
    registry = _ACTIVE_REGISTRY
    _disarm_registry(state, registry)
    _ACTIVE_STATE = None
    _ACTIVE_REGISTRY = None


class SigmaxH4DispatchObserver:
    """Temporary validation node that returns a model clone with scoped observation."""

    CATEGORY = "Sigmax/Validation"
    FUNCTION = "observe"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, object]]:
        return {
            "required": {
                "model": ("MODEL",),
                "trace_file": ("STRING", {"default": "", "multiline": False}),
                "requested_attention_backend": (tuple(sorted(_ALLOWED_ATTENTION_BACKENDS)),),
                "requested_operation_backend": (tuple(sorted(_ALLOWED_OPERATION_REQUESTS)),),
            }
        }

    def observe(
        self,
        model: object,
        trace_file: str,
        requested_attention_backend: str,
        requested_operation_backend: str = "auto",
    ) -> tuple[object]:
        return (
            install_model(
                model,
                trace_file,
                requested_attention_backend,
                requested_operation_backend,
            ),
        )


class SigmaxH4DispatchFinalize:
    """Temporary output node that restores the process-global registry hook."""

    CATEGORY = "Sigmax/Validation"
    FUNCTION = "finalize"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, object]]:
        return {
            "required": {
                "video": ("VIDEO",),
                "trace_file": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def finalize(self, video: object, trace_file: str) -> tuple[object]:
        expected = _safe_trace_file(trace_file)
        if _ACTIVE_STATE is None or _ACTIVE_STATE.trace_file != expected:
            raise DispatchObserverError("dispatch observer trace lifecycle is not active")
        disarm()
        return (video,)


NODE_CLASS_MAPPINGS: Final = {
    "Sigmax.H4DispatchObserver": SigmaxH4DispatchObserver,
    "Sigmax.H4DispatchFinalize": SigmaxH4DispatchFinalize,
}
NODE_DISPLAY_NAME_MAPPINGS: Final = {
    "Sigmax.H4DispatchObserver": "Sigmax H4 Dispatch Observer (temporary)",
    "Sigmax.H4DispatchFinalize": "Sigmax H4 Dispatch Finalize (temporary)",
}

__all__ = [
    "ADAPTER_VERSION",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TRACE_SCHEMA",
    "DispatchObserverError",
    "SigmaxH4DispatchFinalize",
    "SigmaxH4DispatchObserver",
    "_TraceState",
    "_arm_registry",
    "_backend_from_callable",
    "_disarm_registry",
    "_project_trace",
    "_wrap_attention_callable",
    "disarm",
    "install",
    "install_model",
]
