"""Validate the optional headless schedule plot renderer without test-only imports."""

from __future__ import annotations

import importlib.metadata
import json

from comfyui_sigmax.core import (
    ArtifactBuildMetadata,
    ArtifactField,
    BaseGridSpec,
    EvidenceLevel,
    Provenance,
    ScheduleArtifact,
    ScheduleInputs,
    ScheduleOwnership,
    ScheduleRequest,
    ScheduleResult,
    SigmaDomain,
    SliceSpec,
    TerminalPolicy,
    TransformContract,
    TransformStage,
    TypedArtifactValue,
    build_schedule_artifact,
    build_schedule_comparison_report,
    build_schedule_report,
    serialize_schedule_comparison_report,
    serialize_schedule_report,
)
from comfyui_sigmax.plotting import (
    render_schedule_comparison_plot,
    render_schedule_plot,
)


def _artifact(sigmas: tuple[float, ...]) -> ScheduleArtifact:
    steps = len(sigmas) - 1
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=steps, width=1024, height=768),
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version="0.1.0.dev0",
            evidence=EvidenceLevel.EXPERIMENTAL,
            source="plot-validation",
            source_revision="m7-01",
            profile_id="fixture.plot-validation",
            profile_version="1",
        ),
        base_grid=BaseGridSpec(
            identifier="fixture.plot-grid",
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
        transforms=(
            TransformContract(
                name="fixture.plot-shift",
                stage=TransformStage.PRIMARY_TIME_SHIFT,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
            ),
        ),
        terminal_policy=TerminalPolicy.APPEND_ZERO,
        slicing=SliceSpec(start_step=0, end_step=steps, denoise=1.0),
    )
    result = ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=steps, width=1024, height=768),
        sigmas=sigmas,
        final_domain=SigmaDomain.UNIT_FLOW,
    )
    return build_schedule_artifact(
        result,
        metadata=ArtifactBuildMetadata(
            source_id="fixture.schedule-plot-validation",
            source_label="Schedule plot validation",
            base_grid_parameters=(ArtifactField(name="points", value=steps),),
            transform_parameters=(
                (
                    ArtifactField(
                        name="mu",
                        value=TypedArtifactValue(value=1.15, precision="float64"),
                    ),
                ),
            ),
            compatibility=(ArtifactField(name="decision", value="allow"),),
        ),
        precision="float64",
    )


def build_validation_summary() -> dict[str, object]:
    """Render both supported formats and prove the source reports remain immutable."""

    report = build_schedule_report(_artifact((1.0, 0.75, 0.5, 0.25, 0.0)))
    comparison = build_schedule_comparison_report(
        report,
        build_schedule_report(_artifact((1.0, 0.5, 0.375, 0.125, 0.0))),
    )
    report_bytes = serialize_schedule_report(report)
    comparison_bytes = serialize_schedule_comparison_report(comparison)
    rendered: dict[str, bytes] = {}
    for image_format in ("png", "svg"):
        rendered[f"schedule_{image_format}"] = render_schedule_plot(
            report,
            image_format=image_format,
        )
        rendered[f"comparison_{image_format}"] = render_schedule_comparison_plot(
            comparison,
            image_format=image_format,
        )
    if not rendered["schedule_png"].startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("schedule PNG signature is invalid")
    if not rendered["comparison_png"].startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("comparison PNG signature is invalid")
    if not rendered["schedule_svg"].startswith(b"<?xml"):
        raise RuntimeError("schedule SVG signature is invalid")
    if not rendered["comparison_svg"].startswith(b"<?xml"):
        raise RuntimeError("comparison SVG signature is invalid")
    if (
        serialize_schedule_report(report) != report_bytes
        or serialize_schedule_comparison_report(comparison) != comparison_bytes
    ):
        raise RuntimeError("plot rendering mutated canonical report evidence")
    return {
        "format_sizes": {
            identifier: len(payload) for identifier, payload in sorted(rendered.items())
        },
        "matplotlib": importlib.metadata.version("matplotlib"),
        "report_unchanged": True,
        "schema": "sigmax.schedule-plot-validation/1",
        "status": "PASS",
    }


def main() -> int:
    print(
        json.dumps(
            build_validation_summary(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
