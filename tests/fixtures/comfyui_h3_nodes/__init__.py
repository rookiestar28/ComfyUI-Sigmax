"""Release-excluded ComfyUI H3 fixture nodes for deterministic Euler proof."""

from __future__ import annotations

import json
import math
from itertools import pairwise
from typing import Any

import torch  # type: ignore[import-not-found]
from comfy.k_diffusion.sampling import sample_euler  # type: ignore[import-not-found]
from comfy.model_sampling import CONST  # type: ignore[import-not-found]

_INITIAL = (0.75, -0.5, 1.25, -1.0)
_BIASES = (0.0625, -0.125, 0.1875, -0.25)
_UI_KEY = "sigmax_native_euler_trace"
_ALGEBRA_UI_KEY = "sigmax_schedule_algebra"
_CHECKPOINT_UI_KEY = "sigmax_checkpoint_evidence"
_Z_IMAGE_UI_KEY = "sigmax_z_image_schedule"
_FLUX1_SCHNELL_UI_KEY = "sigmax_flux1_schnell_schedule"
_QWEN_IMAGE_UI_KEY = "sigmax_qwen_image_schedule"
_SD3_UI_KEY = "sigmax_sd3_schedule"
_AURAFLOW_UI_KEY = "sigmax_auraflow_schedule"
_KREA2_LORA_UI_KEY = "sigmax_krea2_lora_experimental"
_KREA2_CONDITIONING_UI_KEY = "sigmax_krea2_conditioning"
_KREA2_CONDITIONING_FEATURES = 12 * 2560


def _vector(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _execute_once(sigmas: torch.Tensor) -> tuple[list[dict[str, object]], list[float]]:
    calls: list[dict[str, object]] = []
    converter = CONST()

    class ControlledFlowModel:
        def __call__(
            self,
            state: torch.Tensor,
            sigma: torch.Tensor,
            **_extra: object,
        ) -> torch.Tensor:
            index = len(calls)
            velocity = (
                state * 0.125
                + sigma.reshape(-1, 1) * 0.25
                + state.new_tensor(_BIASES).reshape(1, -1)
                + (index + 1) * 0.03125
            )
            denoised = converter.calculate_denoised(sigma, velocity, state)
            calls.append(
                {
                    "denoised": _vector(denoised),
                    "index": index,
                    "input_state": _vector(state),
                    "sigma": float(sigma.detach().cpu().reshape(-1)[0]),
                    "velocity": _vector(velocity),
                }
            )
            return denoised

    initial = torch.tensor([_INITIAL], dtype=torch.float32, device="cpu")
    final = sample_euler(
        ControlledFlowModel(),
        initial,
        sigmas,
        disable=True,
        s_churn=0.0,
    )
    final_vector = _vector(final)
    steps: list[dict[str, object]] = []
    for index, call in enumerate(calls):
        output = calls[index + 1]["input_state"] if index + 1 < len(calls) else final_vector
        steps.append(
            {
                **call,
                "output_state": output,
                "sigma_next": float(sigmas[index + 1].detach().cpu()),
            }
        )
    return steps, final_vector


class NativeEulerProbe:
    """Invoke the host's native deterministic Euler on a controlled CPU fixture."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only native Euler H3 execution probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, sigmas: object, schedule_info: object) -> dict[str, object]:
        if not isinstance(sigmas, torch.Tensor) or sigmas.device.type != "cpu":
            raise ValueError("H3 sigmas must be a CPU tensor")
        if sigmas.dtype != torch.float32 or sigmas.ndim != 1 or len(sigmas) != 9:
            raise ValueError("H3 sigmas must be one float32 eight-transition schedule")
        if not isinstance(schedule_info, str):
            raise ValueError("H3 schedule information must be text")
        info = json.loads(schedule_info)
        if (
            not isinstance(info, dict)
            or info.get("profile", {}).get("id") != "krea2.turbo.official"
            or info.get("slicing", {}).get("output_steps") != 8
        ):
            raise ValueError("H3 requires the complete official Turbo schedule")

        first_steps, first_final = _execute_once(sigmas)
        second_steps, second_final = _execute_once(sigmas)
        if first_steps != second_steps or first_final != second_final:
            raise ValueError("native Euler deterministic rerun drifted")
        trace: dict[str, Any] = {
            "counts": {
                "effective_model_evaluations": len(first_steps),
                "effective_transitions": len(first_steps),
                "requested_model_evaluations": len(sigmas) - 1,
                "requested_transitions": len(sigmas) - 1,
            },
            "deterministic_rerun": True,
            "initial_state": list(_INITIAL),
            "native_final": first_final,
            "native_steps": first_steps,
            "rerun_final": second_final,
            "sigmas": _vector(sigmas),
            "steps": len(sigmas) - 1,
        }
        return {
            "ui": {
                _UI_KEY: [
                    json.dumps(
                        trace,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        }


class ScheduleAlgebraProbe:
    """Return bounded test-only H2 evidence for an executed algebra schedule."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only schedule algebra H2 execution probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
                "schedule_report": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(
        self,
        sigmas: object,
        schedule_info: object,
        schedule_report: object,
    ) -> dict[str, object]:
        if not isinstance(sigmas, torch.Tensor) or sigmas.device.type != "cpu":
            raise ValueError("H2 algebra sigmas must be a CPU tensor")
        if sigmas.dtype != torch.float32 or sigmas.ndim != 1 or len(sigmas) != 5:
            raise ValueError("H2 algebra sigmas must contain four transitions")
        if not isinstance(schedule_info, str) or not isinstance(schedule_report, str):
            raise ValueError("H2 algebra information and report must be text")
        info = json.loads(schedule_info)
        report = json.loads(schedule_report)
        if (
            not isinstance(info, dict)
            or info.get("schema") != "sigmax.schedule-resample-node/1"
            or info.get("operation") != "resample"
            or info.get("evidence") != "modified"
            or info.get("parameters")
            != {"input_steps": 8, "method": "index_linear_v1", "output_steps": 4}
        ):
            raise ValueError("H2 algebra information is not the expected modified resample")
        if (
            not isinstance(report, dict)
            or report.get("source_schema") != "sigmax.schedule-resample-node/1"
            or report.get("fingerprints", {}).get("verified") is not True
        ):
            raise ValueError("H2 algebra inspector report is not verified")
        evidence = {
            "schedule_info": info,
            "schedule_report": report,
            "sigmas": _vector(sigmas),
        }
        return {
            "ui": {
                _ALGEBRA_UI_KEY: [
                    json.dumps(
                        evidence,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        }


class CheckpointEvidenceProbe:
    """Return bounded test-only H2 evidence from the production inspector node."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only checkpoint evidence H2 execution probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "checkpoint_evidence": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, checkpoint_evidence: object) -> dict[str, object]:
        if not isinstance(checkpoint_evidence, str) or len(checkpoint_evidence) > 100_000:
            raise ValueError("H2 checkpoint evidence must be bounded text")
        report = json.loads(checkpoint_evidence)
        if (
            not isinstance(report, dict)
            or report.get("schema") != "sigmax.checkpoint-evidence-inspection/1"
            or report.get("status") != "inspected"
            or report.get("source", {}).get("payload_bytes_read") != 0
            or report.get("structure", {}).get("tensor_count") != 4
            or report.get("model_identity", {}).get("confirmed_variant") is not None
            or report.get("model_identity", {}).get("suggested_variant") != "turbo"
        ):
            raise ValueError("H2 checkpoint evidence contract drifted")
        return {"ui": {_CHECKPOINT_UI_KEY: [checkpoint_evidence]}}


class ZImageScheduleProbe:
    """Return model-free H2 evidence for one executed Z-Image scheduler node."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only Z-Image schedule H2 execution probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, sigmas: object, schedule_info: object) -> dict[str, object]:
        if not isinstance(sigmas, torch.Tensor) or sigmas.device.type != "cpu":
            raise ValueError("Z-Image H2 sigmas must be a CPU tensor")
        if sigmas.dtype != torch.float32 or sigmas.ndim != 1:
            raise ValueError("Z-Image H2 sigmas must be a float32 vector")
        if not isinstance(schedule_info, str):
            raise ValueError("Z-Image H2 schedule information must be text")
        info = json.loads(schedule_info)
        steps = len(sigmas) - 1
        if (
            not isinstance(info, dict)
            or info.get("schema") != "sigmax.z-image-sigma-node/1"
            or info.get("profile", {}).get("evidence") != "official"
            or info.get("slicing", {}).get("output_steps") != steps
            or info.get("shift", {}).get("dynamic") is not False
        ):
            raise ValueError("Z-Image H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _Z_IMAGE_UI_KEY: [
                    json.dumps(
                        trace,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        }


class Flux1SchnellScheduleProbe:
    """Return model-free H2 evidence for an executed FLUX.1-schnell scheduler."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only FLUX.1-schnell schedule H2 execution probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, sigmas: object, schedule_info: object) -> dict[str, object]:
        if not isinstance(sigmas, torch.Tensor) or sigmas.device.type != "cpu":
            raise ValueError("FLUX.1-schnell H2 sigmas must be a CPU tensor")
        if sigmas.dtype != torch.float32 or sigmas.ndim != 1 or len(sigmas) != 5:
            raise ValueError("FLUX.1-schnell H2 sigmas must contain four transitions")
        if not isinstance(schedule_info, str):
            raise ValueError("FLUX.1-schnell H2 schedule information must be text")
        info = json.loads(schedule_info)
        if (
            not isinstance(info, dict)
            or info.get("schema") != "sigmax.flux1-schnell-sigma-node/1"
            or info.get("profile", {}).get("id") != "flux1.schnell.official"
            or info.get("profile", {}).get("evidence") != "official"
            or info.get("slicing", {}).get("output_steps") != 4
            or info.get("shift") != {"dynamic": False, "kind": "none"}
        ):
            raise ValueError("FLUX.1-schnell H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _FLUX1_SCHNELL_UI_KEY: [
                    json.dumps(
                        trace,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        }


class QwenImageScheduleProbe:
    """Return model-free H2 evidence for one original Qwen Image schedule."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only original Qwen Image schedule H2 execution probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, sigmas: object, schedule_info: object) -> dict[str, object]:
        if not isinstance(sigmas, torch.Tensor) or sigmas.device.type != "cpu":
            raise ValueError("Qwen Image H2 sigmas must be a CPU tensor")
        if sigmas.dtype != torch.float32 or sigmas.ndim != 1 or len(sigmas) != 51:
            raise ValueError("Qwen Image H2 sigmas must contain fifty transitions")
        if not isinstance(schedule_info, str):
            raise ValueError("Qwen Image H2 schedule information must be text")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("Qwen Image H2 schedule information must be an object")
        profile = info.get("profile", {})
        shift = info.get("shift", {})
        if (
            info.get("schema") != "sigmax.qwen-image-sigma-node/1"
            or not isinstance(profile, dict)
            or profile.get("evidence") not in {"official", "framework_reference"}
            or profile.get("id")
            not in {
                "qwen_image.comfy-fixed.official",
                "qwen_image.diffusers-dynamic.framework-reference",
            }
            or not isinstance(shift, dict)
            or info.get("slicing", {}).get("output_steps") != 50
            or float(sigmas[0]) != 1.0
            or float(sigmas[-1]) != 0.0
            or any(float(left) <= float(right) for left, right in pairwise(sigmas))
        ):
            raise ValueError("Qwen Image H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _QWEN_IMAGE_UI_KEY: [
                    json.dumps(
                        trace,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        }


class SD3ScheduleProbe:
    """Return model-free H2 evidence for one original SD3 source mode."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only original Stable Diffusion 3 schedule H2 execution probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, sigmas: object, schedule_info: object) -> dict[str, object]:
        if not isinstance(sigmas, torch.Tensor) or sigmas.device.type != "cpu":
            raise ValueError("SD3 H2 sigmas must be a CPU tensor")
        if not isinstance(schedule_info, str):
            raise ValueError("SD3 H2 schedule information must be text")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("SD3 H2 schedule information must be an object")
        profile = info.get("profile", {})
        mode = (
            "publisher"
            if isinstance(profile, dict) and profile.get("id") == "sd3.publisher-reference.official"
            else "framework"
        )
        expected_steps = 50 if mode == "publisher" else 28
        expected_profile = (
            "sd3.publisher-reference.official"
            if mode == "publisher"
            else "sd3.comfy-diffusers-fixed.framework-reference"
        )
        expected_evidence = "official" if mode == "publisher" else "framework_reference"
        expected_ratio = 1.0 if mode == "publisher" else 3.0
        if (
            sigmas.dtype != torch.float32
            or sigmas.ndim != 1
            or len(sigmas) != expected_steps + 1
            or info.get("schema") != "sigmax.sd3-sigma-node/1"
            or not isinstance(profile, dict)
            or profile.get("id") != expected_profile
            or profile.get("evidence") != expected_evidence
            or info.get("shift") != {"kind": "direct_ratio", "ratio": expected_ratio}
            or info.get("slicing", {}).get("output_steps") != expected_steps
            or float(sigmas[0]) != 1.0
            or float(sigmas[-1]) != 0.0
            or any(float(left) <= float(right) for left, right in pairwise(sigmas))
        ):
            raise ValueError("SD3 H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _SD3_UI_KEY: [
                    json.dumps(
                        trace,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        }


class AuraFlowScheduleProbe:
    """Return model-free H2 evidence for original AuraFlow v0.2."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only original AuraFlow v0.2 schedule H2 execution probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, sigmas: object, schedule_info: object) -> dict[str, object]:
        if not isinstance(sigmas, torch.Tensor) or sigmas.device.type != "cpu":
            raise ValueError("AuraFlow H2 sigmas must be a CPU tensor")
        if not isinstance(schedule_info, str):
            raise ValueError("AuraFlow H2 schedule information must be text")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("AuraFlow H2 schedule information must be an object")
        profile = info.get("profile", {})
        if (
            sigmas.dtype != torch.float32
            or sigmas.ndim != 1
            or len(sigmas) != 51
            or info.get("schema") != "sigmax.aura-flow-sigma-node/1"
            or not isinstance(profile, dict)
            or profile.get("id") != "auraflow.v0-2.official"
            or profile.get("evidence") != "official"
            or info.get("shift") != {"kind": "direct_ratio", "multiplier": 1.0, "ratio": 1.73}
            or info.get("slicing", {}).get("output_steps") != 50
            or float(sigmas[0]) != 1.0
            or float(sigmas[-1]) != 0.0
            or any(float(left) <= float(right) for left, right in pairwise(sigmas))
        ):
            raise ValueError("AuraFlow H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _AURAFLOW_UI_KEY: [
                    json.dumps(
                        trace,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        }


class Krea2LoraExperimentalProbe:
    """Return model-free H2 evidence for one experimental Krea 2 LoRA schedule."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only experimental Krea 2 LoRA H2 schedule probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, sigmas: object, schedule_info: object) -> dict[str, object]:
        if not isinstance(sigmas, torch.Tensor) or sigmas.device.type != "cpu":
            raise ValueError("experimental Krea 2 H2 sigmas must be a CPU tensor")
        if sigmas.dtype != torch.float32 or sigmas.ndim != 1 or len(sigmas) != 13:
            raise ValueError("experimental Krea 2 H2 sigmas must contain 12 transitions")
        if not isinstance(schedule_info, str):
            raise ValueError("experimental Krea 2 H2 information must be text")
        info = json.loads(schedule_info)
        if (
            not isinstance(info, dict)
            or info.get("schema") != "sigmax.krea2-sigma-node/1"
            or info.get("profile", {}).get("id") != "krea2.raw-turbo-lora.experimental"
            or info.get("profile", {}).get("evidence") != "experimental"
            or info.get("shift", {}).get("mu_source") not in {"raw", "turbo"}
            or info.get("slicing", {}).get("output_steps") != 12
            or info.get("strict_official") is not False
        ):
            raise ValueError("experimental Krea 2 H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _KREA2_LORA_UI_KEY: [
                    json.dumps(
                        trace,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        }


class Krea2ConditioningSource:
    """Build deterministic CPU conditioning for the M4-12 host lane only."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only long/multimodal-shaped Krea 2 conditioning source."
    FUNCTION = "execute"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "variant": (["RAW", "Turbo"],),
                "sequence_length": ("INT", {"default": 97, "min": 1, "max": 128}),
            }
        }

    def execute(self, variant: object, sequence_length: object) -> tuple[list[list[object]]]:
        if variant not in {"RAW", "Turbo"}:
            raise ValueError("M4-12 source variant must be explicit")
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
            raise ValueError("M4-12 source sequence length must be an integer")
        if not 1 <= sequence_length <= 128:
            raise ValueError("M4-12 source sequence length is out of bounds")
        total = sequence_length * _KREA2_CONDITIONING_FEATURES
        tensor = (torch.arange(total, dtype=torch.float32) % 257).reshape(
            1, sequence_length, _KREA2_CONDITIONING_FEATURES
        )
        tensor = (tensor - 128.0) / 64.0
        metadata = {
            "attention_mask": torch.ones((1, sequence_length), dtype=torch.bool),
            "area": (0, 0, 64, 64),
            "pooled_output": {"source": "m4-12-test", "variant": variant},
            "reference_latents": {"source": "m4-12-test"},
            "source_marker": "krea2-conditioning-h2-v1",
        }
        return ([[tensor, metadata]],)


class Krea2ConditioningProbe:
    """Verify model-free M4-12 conditioning output and report contracts."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only Krea 2 conditioning H2 probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "modifier_info": ("STRING", {"default": "", "multiline": True}),
                "variant": (["RAW", "Turbo"],),
            }
        }

    def execute(
        self,
        conditioning: object,
        modifier_info: object,
        variant: object,
    ) -> dict[str, object]:
        if variant not in {"RAW", "Turbo"}:
            raise ValueError("M4-12 probe variant must be explicit")
        if not isinstance(conditioning, list) or len(conditioning) != 1:
            raise ValueError("M4-12 probe requires one conditioning entry")
        entry = conditioning[0]
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("M4-12 probe conditioning pair is malformed")
        tensor, metadata = entry
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError("M4-12 probe requires a CPU tensor")
        if tensor.dtype != torch.float32 or tensor.ndim != 3 or tensor.shape[-1] != 30720:
            raise ValueError("M4-12 probe tensor shape or dtype drifted")
        if (
            not isinstance(metadata, dict)
            or metadata.get("source_marker") != "krea2-conditioning-h2-v1"
        ):
            raise ValueError("M4-12 conditioning metadata was not preserved")
        if not isinstance(modifier_info, str) or len(modifier_info) > 65_536:
            raise ValueError("M4-12 modifier report is malformed")
        report = json.loads(modifier_info)
        if (
            not isinstance(report, dict)
            or report.get("schema") != "sigmax.conditioning-modifier/1"
            or report.get("algorithm") != "sigmax.krea2-tap-rms-rebalance/1"
            or report.get("schedule_affected") is not False
            or report.get("evidence") != "experimental"
            or report.get("variant") != {"evidence": "user_selected", "value": variant}
            or report.get("input", {}).get("shape") != list(tensor.shape)
            or report.get("input", {}).get("shapes") != [list(tensor.shape)]
        ):
            raise ValueError("M4-12 conditioning report contract drifted")
        rms = float(torch.sqrt(torch.mean(tensor.float() * tensor.float())).item())
        if not math.isfinite(rms) or rms <= 0.0:
            raise ValueError("M4-12 conditioning output RMS is invalid")
        trace = {
            "metadata_keys": sorted(metadata),
            "report_fingerprint": report.get("fingerprint"),
            "rms": rms,
            "shape": list(tensor.shape),
            "variant": variant,
        }
        return {
            "ui": {
                _KREA2_CONDITIONING_UI_KEY: [
                    json.dumps(
                        trace,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        }


NODE_CLASS_MAPPINGS = {
    "SigmaxTest.Flux1SchnellScheduleProbe": Flux1SchnellScheduleProbe,
    "SigmaxTest.QwenImageScheduleProbe": QwenImageScheduleProbe,
    "SigmaxTest.SD3ScheduleProbe": SD3ScheduleProbe,
    "SigmaxTest.AuraFlowScheduleProbe": AuraFlowScheduleProbe,
    "SigmaxTest.CheckpointEvidenceProbe": CheckpointEvidenceProbe,
    "SigmaxTest.Krea2LoraExperimentalProbe": Krea2LoraExperimentalProbe,
    "SigmaxTest.Krea2ConditioningProbe": Krea2ConditioningProbe,
    "SigmaxTest.Krea2ConditioningSource": Krea2ConditioningSource,
    "SigmaxTest.NativeEulerProbe": NativeEulerProbe,
    "SigmaxTest.ScheduleAlgebraProbe": ScheduleAlgebraProbe,
    "SigmaxTest.ZImageScheduleProbe": ZImageScheduleProbe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SigmaxTest.Flux1SchnellScheduleProbe": "Sigmax Test — FLUX.1-schnell Schedule Probe",
    "SigmaxTest.QwenImageScheduleProbe": "Sigmax Test — Qwen Image Schedule Probe",
    "SigmaxTest.SD3ScheduleProbe": "Sigmax Test — SD3 Schedule Probe",
    "SigmaxTest.AuraFlowScheduleProbe": "Sigmax Test — AuraFlow Schedule Probe",
    "SigmaxTest.CheckpointEvidenceProbe": "Sigmax Test — Checkpoint Evidence Probe",
    "SigmaxTest.Krea2LoraExperimentalProbe": "Sigmax Test — Krea 2 LoRA Experimental Probe",
    "SigmaxTest.Krea2ConditioningProbe": "Sigmax Test — Krea 2 Conditioning Probe",
    "SigmaxTest.Krea2ConditioningSource": "Sigmax Test — Krea 2 Conditioning Source",
    "SigmaxTest.NativeEulerProbe": "Sigmax Test — Native Euler Probe",
    "SigmaxTest.ScheduleAlgebraProbe": "Sigmax Test — Schedule Algebra Probe",
    "SigmaxTest.ZImageScheduleProbe": "Sigmax Test — Z-Image Schedule Probe",
}
