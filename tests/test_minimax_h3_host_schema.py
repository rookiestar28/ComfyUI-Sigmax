"""Model-free MiniMax H3 host-schema preflight contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest
from comfyui_sigmax.adapters.comfyui import ComfyAdapterCompatibilityError
from comfyui_sigmax.workflows import (
    MiniMaxH3HostSchemaReport,
    MiniMaxH3PublicVariant,
    MiniMaxH3SchemaIssueKind,
    MiniMaxH3WorkflowSpec,
    build_minimax_h3_host_workflow,
    resolve_minimax_h3_host_workflow_prompt,
    validate_minimax_h3_host_workflow_schema,
)


def _node(
    name: str,
    *,
    required: dict[str, object],
    optional: dict[str, object] | None = None,
    hidden: dict[str, object] | None = None,
    outputs: tuple[str, ...],
) -> dict[str, object]:
    inputs: dict[str, object] = {"required": required}
    if optional:
        inputs["optional"] = optional
    if hidden:
        inputs["hidden"] = hidden
    return {
        "input": inputs,
        "output": list(outputs),
        "name": name,
    }


def _type(name: str) -> list[str]:
    return [name]


def _combo(*options: str) -> list[object]:
    return [list(options)]


def _host_object_info() -> dict[str, object]:
    return {
        "UNETLoader": _node(
            "UNETLoader",
            required={
                "unet_name": _combo(
                    "H3\\minimax_h3_fl2va_bf16.safetensors",
                    "H3\\minimax_h3_ref2va_bf16.safetensors",
                ),
                "weight_dtype": _combo("default"),
            },
            outputs=("MODEL",),
        ),
        "CLIPLoader": _node(
            "CLIPLoader",
            required={
                "clip_name": _combo("qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
                "type": _combo("minimax"),
            },
            optional={"device": _combo("default", "cpu")},
            outputs=("CLIP",),
        ),
        "VAELoader": _node(
            "VAELoader",
            required={
                "vae_name": _combo(
                    "minimax_h3_video_vae_fp16.safetensors",
                    "minimax_h3_audio_vae_fp32.safetensors",
                )
            },
            outputs=("VAE",),
        ),
        "MiniMaxH3SigmaShift": _node(
            "MiniMaxH3SigmaShift",
            required={
                "model": _type("MODEL"),
                "shift_video": _type("FLOAT"),
                "shift_audio": _type("FLOAT"),
            },
            outputs=("MODEL",),
        ),
        "MiniMaxH3ImageToVideo": _node(
            "MiniMaxH3ImageToVideo",
            required={
                "clip": _type("CLIP"),
                "vae": _type("VAE"),
                "prompt": _type("STRING"),
                "width": _type("INT"),
                "height": _type("INT"),
                "length": _type("INT"),
            },
            optional={"first_frame": _type("IMAGE"), "last_frame": _type("IMAGE")},
            outputs=("CONDITIONING", "LATENT"),
        ),
        "MiniMaxH3ReferenceToVideo": _node(
            "MiniMaxH3ReferenceToVideo",
            required={
                "clip": _type("CLIP"),
                "vae": _type("VAE"),
                "audio_vae": _type("VAE"),
                "prompt": _type("STRING"),
                "width": _type("INT"),
                "height": _type("INT"),
                "length": _type("INT"),
                "ref_image_size": ["COMBO", {"options": ["match", "max"]}],
            },
            optional={
                "ref_images": [
                    "COMFY_AUTOGROW_V3",
                    {
                        "template": {
                            "input": {"required": {"ref_image": _type("IMAGE")}},
                            "prefix": "ref_image_",
                            "min": 0,
                            "max": 9,
                        }
                    },
                ]
            },
            outputs=("CONDITIONING", "LATENT"),
        ),
        "Sigmax.MiniMaxH3SigmaScheduler": _node(
            "Sigmax.MiniMaxH3SigmaScheduler",
            required={
                "variant": _combo("H3 Base FL2VA", "H3 Base Ref2VA"),
                "steps": _type("INT"),
                "start_step": _type("INT"),
                "end_step": _type("INT"),
            },
            optional={
                "turbo": _combo(
                    "disabled",
                    "h3.fl2va.lightx2v-turbo-4-v0.1-544p",
                    "h3.fl2va.lightx2v-turbo-8-v1.0-544p",
                    "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
                    "h3.ref2va.lightx2v-turbo-4-v0.1-544p",
                ),
                "recipe_id": _combo(
                    "disabled",
                    "h3.fl2va.lightx2v-turbo-4-v0.1-544p",
                    "h3.fl2va.lightx2v-turbo-8-v1.0-544p",
                    "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
                    "h3.ref2va.lightx2v-turbo-4-v0.1-544p",
                ),
                "scheduler": _combo(
                    "h3_endpoint",
                    "simple",
                    "sgm_uniform",
                    "karras",
                    "exponential",
                    "ddim_uniform",
                    "beta",
                    "normal",
                    "linear_quadratic",
                    "kl_optimal",
                ),
                "model": _type("MODEL"),
            },
            outputs=("SIGMAS", "STRING"),
        ),
        "KSamplerSelect": _node(
            "KSamplerSelect",
            required={"sampler_name": _combo("euler", "res_multistep")},
            outputs=("SAMPLER",),
        ),
        "BasicGuider": _node(
            "BasicGuider",
            required={"model": _type("MODEL"), "conditioning": _type("CONDITIONING")},
            outputs=("GUIDER",),
        ),
        "RandomNoise": _node(
            "RandomNoise", required={"noise_seed": _type("INT")}, outputs=("NOISE",)
        ),
        "SamplerCustomAdvanced": _node(
            "SamplerCustomAdvanced",
            required={
                "noise": _type("NOISE"),
                "guider": _type("GUIDER"),
                "sampler": _type("SAMPLER"),
                "sigmas": _type("SIGMAS"),
                "latent_image": _type("LATENT"),
            },
            outputs=("LATENT", "LATENT"),
        ),
        "LTXVSeparateAVLatent": _node(
            "LTXVSeparateAVLatent",
            required={"av_latent": _type("LATENT")},
            outputs=("LATENT", "LATENT"),
        ),
        "VAEDecode": _node(
            "VAEDecode",
            required={"samples": _type("LATENT"), "vae": _type("VAE")},
            outputs=("IMAGE",),
        ),
        "VAEDecodeAudio": _node(
            "VAEDecodeAudio",
            required={"samples": _type("LATENT"), "vae": _type("VAE")},
            outputs=("AUDIO",),
        ),
        "PreviewImage": _node(
            "PreviewImage",
            required={"images": _type("IMAGE")},
            hidden={"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            outputs=("IMAGE",),
        ),
        "PreviewAny": _node("PreviewAny", required={"source": _type("*")}, outputs=("STRING",)),
        "LoadImage": _node(
            "LoadImage", required={"image": _combo("input.png")}, outputs=("IMAGE",)
        ),
    }


def _set_unet_choices(host: dict[str, object], *choices: str) -> None:
    unet = cast(dict[str, object], host["UNETLoader"])
    input_section = cast(dict[str, object], unet["input"])
    required = cast(dict[str, object], input_section["required"])
    required["unet_name"] = _combo(*choices)


def test_schema_preflight_accepts_both_explicit_variants() -> None:
    fl2va = build_minimax_h3_host_workflow(
        MiniMaxH3WorkflowSpec(variant="H3 Base FL2VA", prompt="schema preflight")
    )
    ref2va = build_minimax_h3_host_workflow(
        MiniMaxH3WorkflowSpec(
            variant="H3 Base Ref2VA",
            prompt="schema preflight",
            reference_images=("ref.png",),
        )
    )

    for workflow in (fl2va, ref2va):
        report = validate_minimax_h3_host_workflow_schema(workflow, _host_object_info())
        assert isinstance(report, MiniMaxH3HostSchemaReport)
        assert report.compatible is True
        assert report.issues == ()
        assert report.report_fingerprint.startswith("sha256:")


def test_schema_preflight_accepts_experimental_turbo_optional_input() -> None:
    workflow = build_minimax_h3_host_workflow(
        MiniMaxH3WorkflowSpec(variant="H3 Base FL2VA", prompt="turbo schema preflight")
    )
    schedule_inputs = cast(dict[str, object], workflow.prompt["7"]["inputs"])
    schedule_inputs["turbo"] = "h3.fl2va.lightx2v-turbo-4-v0.1-544p"
    report = validate_minimax_h3_host_workflow_schema(workflow, _host_object_info())
    assert report.compatible is True
    assert report.issues == ()


def test_schema_preflight_accepts_native_scheduler_model_connection() -> None:
    workflow = build_minimax_h3_host_workflow(
        MiniMaxH3WorkflowSpec(
            variant="H3 Base FL2VA",
            prompt="native scheduler schema preflight",
            scheduler="simple",
        )
    )
    assert cast(dict[str, object], workflow.prompt["7"]["inputs"])["model"] == ["5", 0]
    report = validate_minimax_h3_host_workflow_schema(workflow, _host_object_info())
    assert report.compatible is True
    assert report.issues == ()


def test_schema_preflight_reports_artifact_and_dynamic_schema_drift() -> None:
    workflow = build_minimax_h3_host_workflow(
        MiniMaxH3WorkflowSpec(
            variant="H3 Base Ref2VA",
            prompt="schema drift",
            reference_images=("ref.png",),
        )
    )
    host = _host_object_info()
    cast_host = host["UNETLoader"]
    assert isinstance(cast_host, dict)
    cast_host["input"]["required"]["unet_name"] = _combo("other.safetensors")
    reference = host["MiniMaxH3ReferenceToVideo"]
    assert isinstance(reference, dict)
    reference["input"]["optional"]["ref_images"][1]["template"]["max"] = 0

    report = validate_minimax_h3_host_workflow_schema(workflow, host)
    kinds = {(item.kind, item.node_id, item.input_name) for item in report.issues}
    assert report.compatible is False
    assert (
        MiniMaxH3SchemaIssueKind.COMBO_VALUE_UNAVAILABLE,
        "1",
        "unet_name",
    ) in kinds
    assert (
        MiniMaxH3SchemaIssueKind.DYNAMIC_INPUT_LIMIT,
        "6",
        "ref_images",
    ) in kinds

    malformed_host = _host_object_info()
    malformed_reference = malformed_host["MiniMaxH3ReferenceToVideo"]
    assert isinstance(malformed_reference, dict)
    malformed_reference["input"]["optional"]["ref_images"][1]["template"]["prefix"] = ""
    malformed_report = validate_minimax_h3_host_workflow_schema(workflow, malformed_host)
    assert any(
        item.kind is MiniMaxH3SchemaIssueKind.DYNAMIC_INPUT_MALFORMED
        for item in malformed_report.issues
    )


def test_schema_preflight_is_deterministic_and_rejects_malformed_schema() -> None:
    workflow = build_minimax_h3_host_workflow(
        MiniMaxH3WorkflowSpec(variant="H3 Base FL2VA", prompt="fingerprint")
    )
    first = validate_minimax_h3_host_workflow_schema(workflow, _host_object_info())
    second = validate_minimax_h3_host_workflow_schema(workflow, _host_object_info())
    assert first.projection() == second.projection()
    assert first.report_fingerprint == second.report_fingerprint

    malformed = deepcopy(_host_object_info())
    malformed["UNETLoader"] = {"input": {"required": {"unet_name": []}}}
    with pytest.raises(ComfyAdapterCompatibilityError):
        validate_minimax_h3_host_workflow_schema(workflow, malformed)


@pytest.mark.parametrize("variant", ["H3 Base FL2VA", "H3 Base Ref2VA"])
def test_host_prompt_resolves_separator_variant_to_exact_windows_combo_choice(
    variant: MiniMaxH3PublicVariant,
) -> None:
    spec = MiniMaxH3WorkflowSpec(
        variant=variant,
        prompt="runtime path resolution",
        reference_images=("ref.png",) if variant == "H3 Base Ref2VA" else (),
    )
    workflow = build_minimax_h3_host_workflow(spec)
    resolved = resolve_minimax_h3_host_workflow_prompt(workflow, _host_object_info())

    expected = (
        "H3\\minimax_h3_fl2va_bf16.safetensors"
        if variant == "H3 Base FL2VA"
        else "H3\\minimax_h3_ref2va_bf16.safetensors"
    )
    resolved_inputs = cast(dict[str, object], resolved["1"]["inputs"])
    workflow_inputs = cast(dict[str, object], workflow.prompt["1"]["inputs"])
    assert resolved_inputs["unet_name"] == expected
    assert workflow_inputs["unet_name"] == expected.replace("\\", "/")


def test_host_prompt_resolver_preserves_linux_choice_spelling() -> None:
    workflow = build_minimax_h3_host_workflow(
        MiniMaxH3WorkflowSpec(variant="H3 Base FL2VA", prompt="linux choice")
    )
    host = _host_object_info()
    _set_unet_choices(
        host,
        "H3/minimax_h3_fl2va_bf16.safetensors",
        "H3/minimax_h3_ref2va_bf16.safetensors",
    )

    resolved = resolve_minimax_h3_host_workflow_prompt(workflow, host)

    resolved_inputs = cast(dict[str, object], resolved["1"]["inputs"])
    assert resolved_inputs["unet_name"] == "H3/minimax_h3_fl2va_bf16.safetensors"


def test_host_prompt_resolver_rejects_ambiguous_normalized_choice() -> None:
    workflow = build_minimax_h3_host_workflow(
        MiniMaxH3WorkflowSpec(variant="H3 Base FL2VA", prompt="ambiguous choice")
    )
    host = _host_object_info()
    _set_unet_choices(
        host,
        "H3/minimax_h3_fl2va_bf16.safetensors",
        "H3\\minimax_h3_fl2va_bf16.safetensors",
    )

    with pytest.raises(ComfyAdapterCompatibilityError, match="ambiguous"):
        resolve_minimax_h3_host_workflow_prompt(workflow, host)


def test_host_prompt_resolver_rejects_missing_host_choice() -> None:
    workflow = build_minimax_h3_host_workflow(
        MiniMaxH3WorkflowSpec(variant="H3 Base FL2VA", prompt="missing choice")
    )
    host = _host_object_info()
    _set_unet_choices(host, "other.safetensors")

    with pytest.raises(ComfyAdapterCompatibilityError, match="missing"):
        resolve_minimax_h3_host_workflow_prompt(workflow, host)
