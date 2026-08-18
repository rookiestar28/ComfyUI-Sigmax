"""Hermetic tests for the temporary H4 dispatch observer."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from scripts.h4_dispatch_observer import (
    DispatchObserverError,
    _backend_from_callable,
    _project_trace,
    _TraceState,
    _wrap_attention_callable,
)


def _callable(
    *, module: str, name: str, result: object = "ok"
) -> Callable[..., object]:
    def implementation(*args: object, **kwargs: object) -> object:
        return result

    implementation.__module__ = module
    implementation.__name__ = name
    return implementation


def test_backend_identity_is_bounded_to_allowlisted_comfy_kitchen_modules() -> None:
    assert _backend_from_callable(
        _callable(module="comfy_kitchen.backends.cuda", name="convrot_w4a4_linear")
    ) == "cuda"
    assert (
        _backend_from_callable(
            _callable(module="comfy_kitchen.backends.eager.quantization", name="int8_linear")
        )
        == "eager"
    )
    assert _backend_from_callable(_callable(module="private.module", name="convrot_w4a4_linear")) is None


def test_operation_is_not_observed_until_wrapped_callable_enters(tmp_path: Path) -> None:
    state = _TraceState(
        trace_file=tmp_path / "trace.json",
        requested_attention_backend="pytorch",
        requested_operation_backend="auto",
    )
    implementation = _callable(
        module="comfy_kitchen.backends.cuda", name="convrot_w4a4_linear", result=17
    )
    wrapped = state.wrap_operation("convrot_w4a4_linear", implementation)
    assert _project_trace(state)["actual_operation_backend"] == "not_observed"
    assert wrapped("private tensor", opaque="private") == 17
    projected = _project_trace(state)
    assert projected["actual_operation_backend"] == "cuda"
    operation = cast(dict[str, object], projected["operation"])
    assert operation["calls"] == 1
    serialized = json.dumps(projected, sort_keys=True)
    assert "private tensor" not in serialized
    assert "secret" not in serialized


def test_wrapped_operation_preserves_exception_and_rejects_stacking(tmp_path: Path) -> None:
    state = _TraceState(
        trace_file=tmp_path / "trace.json",
        requested_attention_backend="pytorch",
        requested_operation_backend="auto",
    )

    def raises(*args: object, **kwargs: object) -> object:
        raise LookupError("synthetic failure")

    raises.__module__ = "comfy_kitchen.backends.eager"
    raises.__name__ = "int8_linear"
    wrapped = state.wrap_operation("int8_linear", raises)
    with pytest.raises(LookupError, match="synthetic failure"):
        wrapped()
    with pytest.raises(DispatchObserverError, match="already wrapped"):
        state.wrap_operation("int8_linear", wrapped)


def test_attention_wrapper_records_success_and_preserves_return(tmp_path: Path) -> None:
    state = _TraceState(
        trace_file=tmp_path / "trace.json",
        requested_attention_backend="ck_int8",
        requested_operation_backend="auto",
    )
    attention = _callable(
        module="comfy.ldm.modules.attention", name="wrapper", result="attention-result"
    )
    wrapped = _wrap_attention_callable(state, "ck_int8", attention)
    assert wrapped("private q", "private k", "private v") == "attention-result"
    projected = _project_trace(state)
    assert projected["actual_attention_backend"] == "ck_int8"
    assert projected["observation_source"] == "authorized_host_dispatch"
    attention_summary = cast(dict[str, object], projected["attention"])
    assert attention_summary["calls"] == 1


def test_trace_file_is_bounded_and_disarm_restores_registry(tmp_path: Path) -> None:
    class Registry:
        def get_implementation(
            self, name: str, *args: object, **kwargs: object
        ) -> Callable[..., object]:
            return _callable(
                module="comfy_kitchen.backends.eager", name=name, result="implementation"
            )

    from scripts.h4_dispatch_observer import _arm_registry, _disarm_registry

    registry = Registry()
    original = registry.get_implementation
    state = _TraceState(
        trace_file=tmp_path / "trace.json",
        requested_attention_backend="pytorch",
        requested_operation_backend="auto",
    )
    _arm_registry(state, registry)
    assert registry.get_implementation is not original
    _disarm_registry(state, registry)
    restored = registry.__dict__["get_implementation"]
    assert getattr(restored, "__func__", None) is getattr(original, "__func__", None)
    assert getattr(restored, "__self__", None) is registry
    assert state.status == "DISARMED"
    assert (tmp_path / "trace.json").is_file()
