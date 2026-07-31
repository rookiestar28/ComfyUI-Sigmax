"""Lazy headless rendering for immutable schedule reports."""

from __future__ import annotations

import importlib
import io
import math
import struct
from collections.abc import Callable
from typing import Any, Final, cast

from comfyui_sigmax.core import (
    ScheduleComparisonReport,
    ScheduleContractError,
    ScheduleReport,
)

_FORMATS: Final = frozenset({"png", "svg"})


class SchedulePlotDependencyError(RuntimeError):
    """Raised when the explicitly optional plotting extra is unavailable."""


def _format(value: object) -> str:
    if not isinstance(value, str) or value not in _FORMATS:
        raise ScheduleContractError("plot format must be png or svg")
    return value


def _decode_token(value: object) -> float:
    if not isinstance(value, dict) or set(value) != {"bits", "precision"}:
        raise ScheduleContractError("plot input contains an invalid typed float")
    bits = value["bits"]
    precision = value["precision"]
    if precision == "float32":
        expected_length = 8
        format_code = ">f"
    elif precision == "float64":
        expected_length = 16
        format_code = ">d"
    else:
        raise ScheduleContractError("plot input precision is unsupported")
    if (
        not isinstance(bits, str)
        or len(bits) != expected_length
        or any(character not in "0123456789abcdef" for character in bits)
    ):
        raise ScheduleContractError("plot input float bits are invalid")
    number = struct.unpack(format_code, bytes.fromhex(bits))[0]
    if not math.isfinite(number):
        raise ScheduleContractError("plot input must be finite")
    return 0.0 if number == 0.0 else number


def _matplotlib() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        figure_module = importlib.import_module("matplotlib.figure")
        agg_module = importlib.import_module("matplotlib.backends.backend_agg")
    except (ImportError, ModuleNotFoundError) as exc:
        raise SchedulePlotDependencyError(
            "schedule plotting requires the optional extra: pip install 'comfyui-sigmax[plot]'"
        ) from exc
    figure = getattr(figure_module, "Figure", None)
    canvas = getattr(agg_module, "FigureCanvasAgg", None)
    if not callable(figure) or not callable(canvas):
        raise SchedulePlotDependencyError("installed Matplotlib lacks the required headless API")
    return cast(Callable[..., Any], figure), cast(Callable[..., Any], canvas)


def _save(figure: Any, *, image_format: str) -> bytes:
    payload = io.BytesIO()
    metadata: dict[str, object] = (
        {"Date": None, "Creator": "ComfyUI-Sigmax"}
        if image_format == "svg"
        else {"Software": "ComfyUI-Sigmax"}
    )
    try:
        figure.savefig(
            payload,
            format=image_format,
            dpi=120,
            metadata=metadata,
        )
        rendered = payload.getvalue()
    finally:
        figure.clear()
        payload.close()
    signature = b"\x89PNG\r\n\x1a\n" if image_format == "png" else b"<?xml"
    if not rendered.startswith(signature):
        raise ScheduleContractError("plot renderer returned an invalid image payload")
    return rendered


def render_schedule_plot(
    report: ScheduleReport,
    *,
    image_format: str = "png",
) -> bytes:
    """Render one report to in-memory PNG/SVG presentation bytes."""

    selected_format = _format(image_format)
    if not isinstance(report, ScheduleReport):
        raise ScheduleContractError("schedule plot requires a ScheduleReport")
    projection = report.projection()
    samples = cast(list[dict[str, object]], projection["samples"])
    indices = [cast(int, sample["index"]) for sample in samples]
    sigmas = [_decode_token(sample["sigma"]) for sample in samples]
    figure_type, canvas_type = _matplotlib()
    figure = figure_type(figsize=(8.0, 5.0), dpi=120)
    canvas_type(figure)
    axis = figure.subplots()
    axis.plot(indices, sigmas, marker="o", linewidth=1.8, label="sigma")
    axis.set_title("Sigma schedule")
    axis.set_xlabel("Terminal-inclusive sigma index")
    axis.set_ylabel(f"Sigma ({projection['domain']})")
    axis.grid(True, alpha=0.25)
    axis.legend()
    return _save(figure, image_format=selected_format)


def render_schedule_comparison_plot(
    report: ScheduleComparisonReport,
    *,
    image_format: str = "png",
) -> bytes:
    """Render two aligned schedules and their absolute differences."""

    selected_format = _format(image_format)
    if not isinstance(report, ScheduleComparisonReport):
        raise ScheduleContractError("comparison plot requires a ScheduleComparisonReport")
    projection = report.projection()
    if projection["comparable"] is not True:
        raise ScheduleContractError("comparison plot requires comparable schedules")
    samples = cast(list[dict[str, object]], projection["samples"])
    indices = [cast(int, sample["index"]) for sample in samples]
    sigmas_a = [_decode_token(sample["sigma_a"]) for sample in samples]
    sigmas_b = [_decode_token(sample["sigma_b"]) for sample in samples]
    absolute = [_decode_token(sample["absolute_difference"]) for sample in samples]
    figure_type, canvas_type = _matplotlib()
    figure = figure_type(figsize=(8.0, 7.0), dpi=120)
    canvas_type(figure)
    schedule_axis, difference_axis = figure.subplots(2, 1, sharex=True)
    schedule_axis.plot(indices, sigmas_a, marker="o", linewidth=1.6, label="schedule A")
    schedule_axis.plot(indices, sigmas_b, marker="s", linewidth=1.6, label="schedule B")
    schedule_axis.set_title("Sigma schedule comparison")
    schedule_axis.set_ylabel("Sigma")
    schedule_axis.grid(True, alpha=0.25)
    schedule_axis.legend()
    difference_axis.plot(indices, absolute, color="#b34747", linewidth=1.6)
    difference_axis.set_xlabel("Terminal-inclusive sigma index")
    difference_axis.set_ylabel("Absolute difference")
    difference_axis.grid(True, alpha=0.25)
    return _save(figure, image_format=selected_format)


__all__ = [
    "SchedulePlotDependencyError",
    "render_schedule_comparison_plot",
    "render_schedule_plot",
]
