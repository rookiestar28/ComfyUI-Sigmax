"""Release-excluded ComfyUI H3 fixture nodes for deterministic Euler proof."""

from __future__ import annotations

import json
from typing import Any

import torch  # type: ignore[import-not-found]
from comfy.k_diffusion.sampling import sample_euler  # type: ignore[import-not-found]
from comfy.model_sampling import CONST  # type: ignore[import-not-found]

_INITIAL = (0.75, -0.5, 1.25, -1.0)
_BIASES = (0.0625, -0.125, 0.1875, -0.25)
_UI_KEY = "sigmax_native_euler_trace"


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


NODE_CLASS_MAPPINGS = {"SigmaxTest.NativeEulerProbe": NativeEulerProbe}
NODE_DISPLAY_NAME_MAPPINGS = {"SigmaxTest.NativeEulerProbe": "Sigmax Test — Native Euler Probe"}
