"""Release-excluded ComfyUI H3 fixture nodes for deterministic Euler proof."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import torch  # type: ignore[import-not-found]
from comfy import model_sampling as comfy_model_sampling  # type: ignore[import-not-found]
from comfy import samplers as comfy_samplers
from comfy import supported_models
from comfy.k_diffusion.sampling import sample_euler  # type: ignore[import-not-found]
from comfy.model_sampling import CONST  # type: ignore[import-not-found]
from comfyui_sigmax.adapters.comfyui_flow_euler import (
    ComfyDenoisedFlowVelocityEvaluator,
    TorchFlowEulerNoiseProvider,
    TorchFlowEulerStateOperations,
)
from comfyui_sigmax.core import (
    AdvancedExecutionMode,
    AdvancedReceiptStatus,
    AdvancedWorkflowFeature,
    AdvancedWorkflowRequest,
    CapabilityDimension,
    ExecutionBehavior,
    ExecutionReceipt,
    NoiseOwnership,
    PredictionType,
    SamplerCapabilities,
    SamplerExecutionSpec,
    SamplerState,
    SamplerStateSnapshot,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalRequirement,
    build_advanced_workflow_receipt,
    canonical_projection_bytes,
    deserialize_advanced_workflow_receipt,
    deserialize_sampler_execution_spec,
    deserialize_sampler_state_snapshot,
    execute_deterministic_flow_euler,
    execute_stochastic_flow_euler,
    resolve_advanced_workflow,
    sampler_execution_spec_fingerprint,
    sampler_state_snapshot_fingerprint,
    serialize_advanced_workflow_receipt,
    serialize_sampler_execution_spec,
    serialize_sampler_state_snapshot,
)

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
_LUMINA2_UI_KEY = "sigmax_lumina2_schedule"
_HUNYUAN_IMAGE21_UI_KEY = "sigmax_hunyuan_image21_schedule"
_ANIMA_UI_KEY = "sigmax_anima_schedule"
_WAN_UI_KEY = "sigmax_wan_schedule"
_LTX_UI_KEY = "sigmax_ltx_schedule"
_MINIMAX_H3_H2_UI_KEY = "sigmax_minimax_h3_h2"
_MINIMAX_H3_NATIVE_H2_UI_KEY = "sigmax_minimax_h3_native_h2"
_MINIMAX_H3_NATIVE_SCHEDULERS = (
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
)
_KREA2_LORA_UI_KEY = "sigmax_krea2_lora_experimental"
_KREA2_CONDITIONING_UI_KEY = "sigmax_krea2_conditioning"
_SAMPLER_STATE_UI_KEY = "sigmax_sampler_state_contract"
_FLOW_EULER_UI_KEY = "sigmax_flow_euler_contract"
_STOCHASTIC_FLOW_EULER_UI_KEY = "sigmax_stochastic_flow_euler_contract"
_ADVANCED_WORKFLOW_UI_KEY = "sigmax_advanced_workflow_compatibility_contract"
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


def _fixture_fingerprint(character: str) -> str:
    return "sha256:" + character * 64


def _sampler_state_not_executed_receipt(spec: SamplerExecutionSpec) -> ExecutionReceipt:
    projection: dict[str, object] = {
        "artifact": {
            "construction_fingerprint": _fixture_fingerprint("a"),
            "numerical_fingerprint": _fixture_fingerprint("b"),
        },
        "compatibility": {
            "considered": [item.value for item in CapabilityDimension],
            "level": "allow",
            "reasons": ["compatible"],
        },
        "counts": {
            "effective_model_evaluations": 0,
            "effective_transitions": 0,
            "requested_model_evaluations": spec.requested_model_evaluations,
            "requested_transitions": spec.requested_transitions,
        },
        "effective_inputs": {
            "compatibility": {},
            "height": None,
            "precision": "float64",
            "profile": "fixture.profile",
            "profile_version": "1",
            "steps": spec.requested_transitions,
            "width": None,
        },
        "execution": {"reason_code": None, "status": "not_executed"},
        "host": {
            "api_version": "test",
            "id": "comfyui",
            "revision": "fixture",
            "version": "contract",
        },
        "model": {
            "fingerprint": _fixture_fingerprint("c"),
            "id": "fixture.model",
            "version": "1",
        },
        "profile": {"id": "fixture.profile", "version": "1"},
        "rng_ownership": {"model": "none", "sampler": "none", "schedule": "none"},
        "sampler": {
            "fingerprint": _fixture_fingerprint("d"),
            "id": spec.capabilities.sampler_id,
            "version": spec.capabilities.sampler_version,
        },
        "schema": "sigmax.execution-receipt/1",
    }
    payload = canonical_projection_bytes(projection)
    return ExecutionReceipt(
        receipt_bytes=payload,
        receipt_fingerprint="sha256:" + hashlib.sha256(payload).hexdigest(),
        construction_fingerprint=_fixture_fingerprint("a"),
        numerical_fingerprint=_fixture_fingerprint("b"),
    )


class SamplerStateContractProbe:
    """Exercise only the M5-02 portable contract; never call a numerical sampler."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only M5-02 sampler-state contract probe without sampler execution."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {"required": {}}

    def execute(self) -> dict[str, object]:
        before_call = torch.nn.Module.__call__
        before_schedulers = tuple(comfy_samplers.SCHEDULER_NAMES)
        capabilities = SamplerCapabilities(
            sampler_id="comfy.euler",
            sampler_version="native",
            accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
            accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
            accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
            terminal_requirement=TerminalRequirement.REQUIRES_ZERO,
            execution_behavior=ExecutionBehavior.DETERMINISTIC,
            noise_ownership=NoiseOwnership.NONE,
            required_state=(
                SamplerState.BEGIN_INDEX,
                SamplerState.STEP_INDEX,
                SamplerState.MULTISTEP_HISTORY,
                SamplerState.RESUME,
            ),
            supports_partial_denoise=True,
            supports_per_token_timesteps=True,
        )
        spec = SamplerExecutionSpec(
            capabilities=capabilities,
            scheduler_index=4,
            begin_index=2,
            solver_order=1,
            timestep_spacing="native",
            random_source_ownership=NoiseOwnership.NONE,
            per_token_time=(0.25, 0.5),
            requested_transitions=2,
            requested_model_evaluations=2,
        )
        spec_bytes = serialize_sampler_execution_spec(spec)
        restored_spec = deserialize_sampler_execution_spec(spec_bytes)
        initial = SamplerStateSnapshot.initial(spec)
        initial_bytes = serialize_sampler_state_snapshot(initial, spec)
        restored_initial = deserialize_sampler_state_snapshot(initial_bytes, spec)
        receipt = _sampler_state_not_executed_receipt(spec)
        bound = initial.attach_execution_receipt_evidence(spec, receipt)
        bound_bytes = serialize_sampler_state_snapshot(bound, spec)
        restored_bound = deserialize_sampler_state_snapshot(bound_bytes, spec)
        global_mutation = (
            torch.nn.Module.__call__ is not before_call
            or tuple(comfy_samplers.SCHEDULER_NAMES) != before_schedulers
        )
        trace = {
            "bound_snapshot_fingerprint": sampler_state_snapshot_fingerprint(bound, spec),
            "execution_receipt_fingerprint": receipt.receipt_fingerprint,
            "global_mutation": global_mutation,
            "history_length": len(bound.history),
            "initial_snapshot_fingerprint": sampler_state_snapshot_fingerprint(initial, spec),
            "python_version": platform.python_version(),
            "receipt_bound": bound.execution_receipt_fingerprint == receipt.receipt_fingerprint,
            "receipt_status": "not_executed",
            "round_trip_stable": (
                restored_spec == spec
                and restored_initial == initial
                and restored_bound == bound
                and serialize_sampler_execution_spec(restored_spec) == spec_bytes
                and serialize_sampler_state_snapshot(restored_initial, spec) == initial_bytes
                and serialize_sampler_state_snapshot(restored_bound, spec) == bound_bytes
            ),
            "sampler_execution_performed": False,
            "schema": "sigmax.sampler-state-host-contract/1",
            "spec_fingerprint": sampler_execution_spec_fingerprint(spec),
            "status": bound.status.value,
        }
        return {
            "ui": {
                _SAMPLER_STATE_UI_KEY: [
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


def _flow_euler_capabilities() -> SamplerCapabilities:
    return SamplerCapabilities(
        sampler_id="sigmax.flow_euler",
        sampler_version="1",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        terminal_requirement=TerminalRequirement.REQUIRES_ZERO,
        execution_behavior=ExecutionBehavior.DETERMINISTIC,
        noise_ownership=NoiseOwnership.NONE,
        required_state=(
            SamplerState.BEGIN_INDEX,
            SamplerState.STEP_INDEX,
            SamplerState.MULTISTEP_HISTORY,
            SamplerState.RESUME,
        ),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )


def _flow_euler_spec(*, begin_index: int, transitions: int) -> SamplerExecutionSpec:
    return SamplerExecutionSpec(
        capabilities=_flow_euler_capabilities(),
        scheduler_index=begin_index,
        begin_index=begin_index,
        solver_order=1,
        timestep_spacing="explicit_unit_flow",
        random_source_ownership=NoiseOwnership.NONE,
        per_token_time=None,
        requested_transitions=transitions,
        requested_model_evaluations=transitions,
    )


def _stochastic_flow_euler_spec() -> SamplerExecutionSpec:
    return SamplerExecutionSpec(
        capabilities=SamplerCapabilities(
            sampler_id="sigmax.flow_euler_stochastic",
            sampler_version="1",
            accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
            accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
            accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
            terminal_requirement=TerminalRequirement.REQUIRES_ZERO,
            execution_behavior=ExecutionBehavior.STOCHASTIC,
            noise_ownership=NoiseOwnership.CALLER,
            required_state=(
                SamplerState.BEGIN_INDEX,
                SamplerState.STEP_INDEX,
                SamplerState.MULTISTEP_HISTORY,
            ),
            supports_partial_denoise=False,
            supports_per_token_timesteps=False,
        ),
        scheduler_index=0,
        begin_index=0,
        solver_order=1,
        timestep_spacing="explicit_unit_flow",
        random_source_ownership=NoiseOwnership.CALLER,
        per_token_time=None,
        requested_transitions=3,
        requested_model_evaluations=3,
    )


def _flow_model(call_sigmas: list[float]) -> Any:
    bias = torch.tensor([_BIASES], dtype=torch.float32, device="cpu")

    def model(
        state: torch.Tensor,
        sigma: torch.Tensor,
        **_extra: object,
    ) -> torch.Tensor:
        sigma_value = sigma.reshape(-1, 1)
        call_sigmas.append(float(sigma_value.detach().cpu().reshape(-1)[0]))
        velocity = state * 0.125 + sigma_value * 0.25 + bias
        return state - velocity * sigma_value

    return model


class FlowEulerContractProbe:
    """Execute M5-03 against native Euler on bounded model-free CPU tensors."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only M5-03 deterministic Flow Euler execution and native parity probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {"required": {}}

    def execute(self) -> dict[str, object]:
        before_call = torch.nn.Module.__call__
        before_schedulers = tuple(comfy_samplers.SCHEDULER_NAMES)
        before_rng = torch.random.get_rng_state().clone()
        sigmas = torch.tensor((1.0, 0.75, 0.25, 0.0), dtype=torch.float32, device="cpu")
        sigma_values = tuple(float(item) for item in sigmas.tolist())
        initial = torch.tensor([_INITIAL], dtype=torch.float32, device="cpu")
        operations = TorchFlowEulerStateOperations(torch_module=torch)

        native_full_calls: list[float] = []
        native_full = sample_euler(
            _flow_model(native_full_calls),
            initial.clone(),
            sigmas,
            disable=True,
            s_churn=0.0,
        )
        full_calls: list[float] = []
        full = execute_deterministic_flow_euler(
            spec=_flow_euler_spec(begin_index=0, transitions=3),
            sigmas=sigma_values,
            state=initial.clone(),
            evaluator=ComfyDenoisedFlowVelocityEvaluator(model=_flow_model(full_calls)),
            operations=operations,
        )

        prefix_calls: list[float] = []
        partial_initial = sample_euler(
            _flow_model(prefix_calls),
            initial.clone(),
            sigmas[:2],
            disable=True,
            s_churn=0.0,
        )
        native_partial_calls: list[float] = []
        native_partial = sample_euler(
            _flow_model(native_partial_calls),
            partial_initial.clone(),
            sigmas[1:],
            disable=True,
            s_churn=0.0,
        )
        partial_calls: list[float] = []
        partial = execute_deterministic_flow_euler(
            spec=_flow_euler_spec(begin_index=1, transitions=2),
            sigmas=sigma_values,
            state=partial_initial.clone(),
            evaluator=ComfyDenoisedFlowVelocityEvaluator(model=_flow_model(partial_calls)),
            operations=operations,
        )

        resume_calls: list[float] = []
        resume_spec = _flow_euler_spec(begin_index=0, transitions=3)
        interrupted = execute_deterministic_flow_euler(
            spec=resume_spec,
            sigmas=sigma_values,
            state=initial.clone(),
            evaluator=ComfyDenoisedFlowVelocityEvaluator(model=_flow_model(resume_calls)),
            operations=operations,
            transition_limit=1,
        )
        resumed = execute_deterministic_flow_euler(
            spec=resume_spec,
            sigmas=sigma_values,
            state=interrupted.state,
            evaluator=ComfyDenoisedFlowVelocityEvaluator(model=_flow_model(resume_calls)),
            operations=operations,
            snapshot=interrupted.snapshot,
        )

        invalid_terminal_calls: list[float] = []
        try:
            execute_deterministic_flow_euler(
                spec=_flow_euler_spec(begin_index=0, transitions=2),
                sigmas=(1.0, 0.5, 0.25),
                state=initial.clone(),
                evaluator=ComfyDenoisedFlowVelocityEvaluator(
                    model=_flow_model(invalid_terminal_calls)
                ),
                operations=operations,
            )
        except ScheduleContractError:
            pass
        else:
            raise RuntimeError("M5-03 invalid terminal schedule was accepted")

        resume_mismatch_calls: list[float] = []
        try:
            execute_deterministic_flow_euler(
                spec=resume_spec,
                sigmas=sigma_values,
                state=interrupted.state + 1.0,
                evaluator=ComfyDenoisedFlowVelocityEvaluator(
                    model=_flow_model(resume_mismatch_calls)
                ),
                operations=operations,
                snapshot=interrupted.snapshot,
            )
        except ScheduleContractError:
            pass
        else:
            raise RuntimeError("M5-03 mismatched resume state was accepted")

        full_error = (full.state - native_full).abs()
        partial_error = (partial.state - native_partial).abs()
        all_calls = (
            native_full_calls
            + full_calls
            + prefix_calls
            + native_partial_calls
            + partial_calls
            + resume_calls
        )
        global_mutation = (
            torch.nn.Module.__call__ is not before_call
            or tuple(comfy_samplers.SCHEDULER_NAMES) != before_schedulers
            or not torch.equal(torch.random.get_rng_state(), before_rng)
        )
        trace = {
            "full_effective_model_evaluations": full.snapshot.effective_model_evaluations,
            "full_effective_transitions": full.snapshot.effective_transitions,
            "full_result_fingerprint": full.result_fingerprint,
            "full_scheduler_indexes": [step.scheduler_index for step in full.snapshot.history],
            "global_mutation": global_mutation,
            "model_weights_used": False,
            "negative_rejections": {
                "invalid_terminal_evaluator_calls": len(invalid_terminal_calls),
                "resume_mismatch_evaluator_calls": len(resume_mismatch_calls),
            },
            "native_full_max_abs_error_hex": float(full_error.max().item()).hex(),
            "native_full_mean_abs_error_hex": float(full_error.mean().item()).hex(),
            "native_partial_max_abs_error_hex": float(partial_error.max().item()).hex(),
            "partial_effective_model_evaluations": partial.snapshot.effective_model_evaluations,
            "partial_effective_transitions": partial.snapshot.effective_transitions,
            "partial_result_fingerprint": partial.result_fingerprint,
            "partial_scheduler_indexes": [
                step.scheduler_index for step in partial.snapshot.history
            ],
            "python_version": platform.python_version(),
            "resumed_matches_full": (
                torch.equal(resumed.state, full.state) and resumed.snapshot == full.snapshot
            ),
            "resumed_result_fingerprint": resumed.result_fingerprint,
            "sampler_execution_performed": True,
            "schedule_fingerprint": full.schedule_fingerprint,
            "schema": "sigmax.flow-euler-host-contract/1",
            "status": "succeeded",
            "terminal_model_evaluations": sum(value == 0.0 for value in all_calls),
            "torch_version": str(torch.__version__),
        }
        return {
            "ui": {
                _FLOW_EULER_UI_KEY: [
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


class StochasticFlowEulerContractProbe:
    """Execute M5-04 with local CPU generators and no model weights."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only M5-04 caller-RNG stochastic Flow Euler contract probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {"required": {}}

    def execute(self) -> dict[str, object]:
        before_call = torch.nn.Module.__call__
        before_schedulers = tuple(comfy_samplers.SCHEDULER_NAMES)
        before_rng = torch.random.get_rng_state().clone()
        sigmas = (1.0, 0.75, 0.25, 0.0)
        initial = torch.tensor([_INITIAL], dtype=torch.float32, device="cpu")
        operations = TorchFlowEulerStateOperations(torch_module=torch)
        spec = _stochastic_flow_euler_spec()

        def evaluator(state: torch.Tensor, sigma: float, scheduler_index: int) -> torch.Tensor:
            return state * 0.25 + sigma + scheduler_index * 0.125

        def run(seed: int) -> Any:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return execute_stochastic_flow_euler(
                spec=spec,
                sigmas=sigmas,
                state=initial.clone(),
                evaluator=evaluator,
                noise_provider=TorchFlowEulerNoiseProvider(
                    generator=generator,
                    torch_module=torch,
                ),
                operations=operations,
            )

        first = run(123)
        repeat = run(123)
        alternate = run(124)
        global_mutation = (
            torch.nn.Module.__call__ is not before_call
            or tuple(comfy_samplers.SCHEDULER_NAMES) != before_schedulers
            or not torch.equal(torch.random.get_rng_state(), before_rng)
        )
        trace = {
            "alternate_result_fingerprint": alternate.result_fingerprint,
            "different_seed_diverges": alternate.result_fingerprint != first.result_fingerprint,
            "full_effective_model_evaluations": first.snapshot.effective_model_evaluations,
            "full_effective_noise_draws": len(first.noise_fingerprints),
            "full_effective_transitions": first.snapshot.effective_transitions,
            "full_result_fingerprint": first.result_fingerprint,
            "global_mutation": global_mutation,
            "model_weights_used": False,
            "noise_fingerprints": list(first.noise_fingerprints),
            "python_version": platform.python_version(),
            "repeat_result_fingerprint": repeat.result_fingerprint,
            "same_seed_repeat": repeat.result_fingerprint == first.result_fingerprint,
            "sampler_execution_performed": True,
            "schedule_fingerprint": first.schedule_fingerprint,
            "schema": "sigmax.stochastic-flow-euler-host-contract/1",
            "status": "succeeded",
            "terminal_noise_draw": len(first.noise_fingerprints) == len(sigmas) - 1,
            "torch_version": str(torch.__version__),
        }
        return {
            "ui": {
                _STOCHASTIC_FLOW_EULER_UI_KEY: [
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


class AdvancedWorkflowCompatibilityProbe:
    """Exercise M5-05 decisions and receipts without model or framework execution."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only M5-05 advanced-workflow compatibility contract probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {"required": {}}

    def execute(self) -> dict[str, object]:
        before_call = torch.nn.Module.__call__
        before_schedulers = tuple(comfy_samplers.SCHEDULER_NAMES)
        fixed_spec = _fixture_fingerprint("a")
        fixed_snapshot = _fixture_fingerprint("b")

        requests = {
            "native_missing_capability": AdvancedWorkflowRequest(
                features=(
                    AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
                    AdvancedWorkflowFeature.INPAINTING,
                ),
                execution_mode=AdvancedExecutionMode.NATIVE_HOST,
                host_capabilities=(AdvancedWorkflowFeature.IMAGE_TO_IMAGE,),
            ),
            "deterministic_controller": AdvancedWorkflowRequest(
                features=(
                    AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
                    AdvancedWorkflowFeature.PARTIAL_DENOISE,
                ),
                execution_mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
                required_state=("latent", "sigma_cursor"),
            ),
            "stochastic_rejected": AdvancedWorkflowRequest(
                features=(
                    AdvancedWorkflowFeature.PARTIAL_DENOISE,
                    AdvancedWorkflowFeature.RESUME,
                ),
                execution_mode=AdvancedExecutionMode.STOCHASTIC_PURE,
            ),
            "deterministic_resume": AdvancedWorkflowRequest(
                features=(AdvancedWorkflowFeature.RESUME,),
                execution_mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
                required_state=("execution_cursor", "snapshot"),
                snapshot_fingerprint=fixed_snapshot,
                spec_fingerprint=fixed_spec,
                snapshot_spec_fingerprint=fixed_spec,
            ),
            "native_interruption": AdvancedWorkflowRequest(
                features=(AdvancedWorkflowFeature.INTERRUPTION,),
                execution_mode=AdvancedExecutionMode.NATIVE_HOST,
                host_capabilities=(AdvancedWorkflowFeature.INTERRUPTION,),
            ),
            "pure_inpainting_rejected": AdvancedWorkflowRequest(
                features=(AdvancedWorkflowFeature.INPAINTING,),
                execution_mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
            ),
        }
        decisions = {name: resolve_advanced_workflow(request) for name, request in requests.items()}
        receipts = {
            name: build_advanced_workflow_receipt(
                requests[name],
                decision,
                execution_status=(
                    AdvancedReceiptStatus.INTERRUPTED
                    if name == "native_interruption"
                    else AdvancedReceiptStatus.RESUMABLE
                    if name == "deterministic_resume"
                    else None
                ),
                resumable=name == "deterministic_resume",
            )
            for name, decision in decisions.items()
        }
        restored = {
            name: deserialize_advanced_workflow_receipt(
                serialize_advanced_workflow_receipt(receipt)
            )
            for name, receipt in receipts.items()
        }
        global_mutation = (
            torch.nn.Module.__call__ is not before_call
            or tuple(comfy_samplers.SCHEDULER_NAMES) != before_schedulers
        )
        trace = {
            "cleanup": True,
            "decision_fingerprints": {
                name: decision.fingerprint for name, decision in decisions.items()
            },
            "decision_levels": {name: decision.level.value for name, decision in decisions.items()},
            "expected_rejections": 3,
            "global_mutation": global_mutation,
            "model_weights_used": False,
            "receipt_fingerprints": {
                name: receipt.receipt_fingerprint for name, receipt in receipts.items()
            },
            "receipt_statuses": {
                name: receipt.execution_status.value for name, receipt in receipts.items()
            },
            "registry_mutation": global_mutation,
            "round_trip_stable": all(restored[name] == receipts[name] for name in receipts),
            "schema": "sigmax.advanced-workflow-host-contract/1",
            "status": "succeeded",
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
        }
        return {
            "ui": {
                _ADVANCED_WORKFLOW_UI_KEY: [
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


class MiniMaxH3ScheduleProbe:
    """Return bounded model-free host evidence for one explicit MiniMax H3 variant."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only MiniMax H3 sigma-node H2 execution probe."
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
            raise ValueError("MiniMax H3 H2 sigmas must be a CPU tensor")
        if sigmas.dtype != torch.float32 or sigmas.ndim != 1 or len(sigmas) != 21:
            raise ValueError("MiniMax H3 H2 sigmas must be one float32 20-transition schedule")
        if not isinstance(schedule_info, str):
            raise ValueError("MiniMax H3 H2 schedule information must be text")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("MiniMax H3 H2 schedule information must be an object")
        profile = info.get("profile")
        expected_profiles = {"minimax-h3.base_fl2va", "minimax-h3.base_ref2va"}
        counts = info.get("counts")
        audio = info.get("audio")
        shift = info.get("shift")
        velocity = info.get("velocity")
        slicing = info.get("slicing")
        canonical_info = json.dumps(
            info,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if (
            schedule_info != canonical_info
            or not isinstance(profile, dict)
            or profile.get("id") not in expected_profiles
            or info.get("schema") != "sigmax.minimax-h3-sigma-node/1"
            or info.get("lane") != "diffusers_endpoint_inclusive"
            or not isinstance(counts, dict)
            or counts
            != {
                "effective_grid_points": 21,
                "effective_model_evaluations": 20,
                "effective_steps": 20,
                "effective_transitions": 20,
                "requested_grid_points": 21,
                "requested_model_evaluations": 20,
                "requested_steps": 20,
                "requested_transitions": 20,
            }
            or not isinstance(audio, dict)
            or audio.get("ownership") != "model_native"
            or audio.get("derivative") != "model_native"
            or audio.get("shift") != 3.0
            or not isinstance(shift, dict)
            or shift.get("video") != 12.0
            or shift.get("audio") != 3.0
            or not isinstance(velocity, dict)
            or velocity != {"direction": "data_ward", "sign_adapter": "explicit_only"}
            or not isinstance(slicing, dict)
            or slicing.get("output_steps") != 20
            or float(sigmas[0]) != 1.0
            or float(sigmas[-1]) != 0.0
            or any(float(left) <= float(right) for left, right in pairwise(sigmas))
        ):
            raise ValueError("MiniMax H3 H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _MINIMAX_H3_H2_UI_KEY: [
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


class _SyntheticMiniMaxH3Model:
    """Weight-free MODEL seam around the real host's H3 sampling implementation."""

    def __init__(
        self,
        *,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
        source_mode: str = "h3",
    ) -> None:
        if (
            not math.isfinite(video_shift)
            or video_shift <= 0.0
            or not math.isfinite(audio_shift)
            or audio_shift <= 0.0
            or source_mode not in {"h3", "non_h3"}
        ):
            raise ValueError("MiniMax H3 native model source inputs are invalid")
        config = supported_models.MiniMaxH3({"image_model": "minimax_h3"})

        sampling_base = getattr(
            comfy_model_sampling,
            "ModelSamplingAV",
            comfy_model_sampling.ModelSamplingDiscreteFlow,
        )

        # IMPORTANT: the qualified 0.30 host predates ModelSamplingAV and uses DiscreteFlow.
        class SyntheticSampling(sampling_base, CONST):  # type: ignore[misc, valid-type]
            pass

        self._sampling = SyntheticSampling(config)
        setter = getattr(self._sampling, "set_parameters", None)
        if not callable(setter):
            raise ValueError("MiniMax H3 native sampling source lacks set_parameters")
        if hasattr(comfy_model_sampling, "ModelSamplingAV"):
            setter(shift=video_shift, audio_shift=audio_shift)
        else:
            # IMPORTANT: ComfyUI 0.30 owns audio shift through complete H3 option markers.
            setter(shift=video_shift)
        self.model = SimpleNamespace(model_config=config if source_mode == "h3" else object())
        self.model_options = {
            "transformer_options": {
                "minimax_h3_sigma_shift_video": video_shift,
                "minimax_h3_sigma_shift_audio": audio_shift,
            }
        }

    def get_model_object(self, name: str) -> object:
        if name != "model_sampling":
            raise KeyError(name)
        return self._sampling


class MiniMaxH3NativeModelSource:
    """Return a bounded weight-free H3 MODEL for native scheduler H2."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only host-qualified MiniMax H3 sampling source; loads no weights."
    FUNCTION = "build"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {},
            "optional": {
                "video_shift": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01},
                ),
                "audio_shift": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01},
                ),
                "source_mode": (("h3", "non_h3"), {"default": "h3"}),
            },
        }

    def build(
        self,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
        source_mode: str = "h3",
    ) -> tuple[object]:
        return (
            _SyntheticMiniMaxH3Model(
                video_shift=video_shift,
                audio_shift=audio_shift,
                source_mode=source_mode,
            ),
        )


class MiniMaxH3NativeScheduleProbe:
    """Compare delegated sigmas to BasicScheduler's host-owned calculation."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only MiniMax H3 native scheduler differential probe."
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES: tuple[()] = ()
    RETURN_NAMES: tuple[()] = ()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
                "scheduler": (_MINIMAX_H3_NATIVE_SCHEDULERS, {"default": "simple"}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 1000}),
            },
            "optional": {
                "case_id": ("STRING", {"default": "m4-17.simple"}),
                "variant": (("", "H3 Base FL2VA", "H3 Base Ref2VA"), {"default": ""}),
                "recipe_id": ("STRING", {"default": ""}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 999}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 1000}),
                "video_shift": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01},
                ),
                "audio_shift": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01},
                ),
            },
        }

    def execute(
        self,
        model: object,
        sigmas: object,
        schedule_info: object,
        scheduler: object,
        steps: object,
        case_id: object = "m4-17.simple",
        variant: object = "",
        recipe_id: object = "",
        start_step: object = 0,
        end_step: object = -1,
        video_shift: object = 12.0,
        audio_shift: object = 3.0,
    ) -> dict[str, object]:
        if scheduler not in _MINIMAX_H3_NATIVE_SCHEDULERS:
            raise ValueError("MiniMax H3 native H2 scheduler is outside the fixed set")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("MiniMax H3 native H2 steps are invalid")
        if not isinstance(case_id, str) or not case_id or len(case_id) > 256:
            raise ValueError("MiniMax H3 native H2 case ID is invalid")
        if variant not in {"", "H3 Base FL2VA", "H3 Base Ref2VA"}:
            raise ValueError("MiniMax H3 native H2 variant is invalid")
        if not isinstance(recipe_id, str) or len(recipe_id) > 128:
            raise ValueError("MiniMax H3 native H2 recipe ID is invalid")
        if (
            not isinstance(start_step, int)
            or isinstance(start_step, bool)
            or start_step < 0
            or not isinstance(end_step, int)
            or isinstance(end_step, bool)
            or end_step < -1
        ):
            raise ValueError("MiniMax H3 native H2 slice is invalid")
        if (
            not isinstance(video_shift, int | float)
            or isinstance(video_shift, bool)
            or not math.isfinite(float(video_shift))
            or float(video_shift) <= 0.0
            or not isinstance(audio_shift, int | float)
            or isinstance(audio_shift, bool)
            or not math.isfinite(float(audio_shift))
            or float(audio_shift) <= 0.0
        ):
            raise ValueError("MiniMax H3 native H2 shifts are invalid")
        if not isinstance(sigmas, torch.Tensor) or sigmas.dtype != torch.float32:
            raise ValueError("MiniMax H3 native H2 sigmas must be float32")
        if not isinstance(schedule_info, str):
            raise ValueError("MiniMax H3 native H2 schedule information must be text")
        getter = getattr(model, "get_model_object", None)
        if not callable(getter):
            raise ValueError("MiniMax H3 native H2 MODEL lacks model sampling")
        raw_reference = comfy_samplers.calculate_sigmas(
            getter("model_sampling"), scheduler, steps
        ).cpu()
        basic_reference = raw_reference[-(steps + 1) :]
        available_steps = len(basic_reference) - 1
        effective_end = available_steps if end_step == -1 else end_step
        if not 0 <= start_step < effective_end <= available_steps:
            raise ValueError("MiniMax H3 native H2 slice is outside the BasicScheduler result")
        reference = basic_reference[start_step : effective_end + 1]
        if sigmas.device.type != "cpu" or reference.dtype != torch.float32:
            raise ValueError("MiniMax H3 native H2 must remain on the float32 CPU boundary")
        if sigmas.shape != reference.shape or not torch.equal(sigmas, reference):
            raise ValueError("MiniMax H3 native schedule differs from BasicScheduler semantics")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("MiniMax H3 native H2 metadata must be an object")
        native = info.get("scheduler")
        host_version = (
            native.get("host", {}).get("observed_version") if isinstance(native, dict) else None
        )
        expected_sampling_api = (
            {
                "0.30.0": "model_sampling_discrete_flow_h3_v030",
                "0.32.0": "model_sampling_av_v032",
            }.get(host_version)
            if isinstance(host_version, str)
            else None
        )
        expected_task = (
            "fl2va"
            if variant == "H3 Base FL2VA"
            else "ref2va"
            if variant == "H3 Base Ref2VA"
            else None
        )
        expected_recipe = recipe_id or None
        expected_terminal = float(reference[-1])
        native_counts = native.get("counts") if isinstance(native, dict) else None
        native_terminal = native.get("terminal") if isinstance(native, dict) else None
        native_shift = native.get("shift") if isinstance(native, dict) else None
        native_slice = native.get("slicing") if isinstance(native, dict) else None
        native_fingerprints = native.get("fingerprints") if isinstance(native, dict) else None
        if (
            info.get("lane") != "m4_17_comfyui_native_scheduler"
            or info.get("mode") != "experimental_comfyui_native_scheduler"
            or not isinstance(native, dict)
            or native.get("owner") != "comfyui_native"
            or native.get("scheduler") != scheduler
            or native.get("model_task") not in {"fl2va", "ref2va"}
            or (expected_task is not None and native.get("model_task") != expected_task)
            or native.get("recipe_id") != expected_recipe
            or native.get("dtype") != "float32"
            or not isinstance(native.get("host"), dict)
            or expected_sampling_api is None
            or native.get("sampling_api") != expected_sampling_api
            or not isinstance(native_counts, dict)
            or native_counts.get("requested_steps") != steps
            or native_counts.get("raw_sigmas") != len(raw_reference)
            or native_counts.get("actual_sigmas") != len(sigmas)
            or native_counts.get("actual_transitions") != len(sigmas) - 1
            or native_shift
            != {
                "already_applied": True,
                "audio": float(audio_shift),
                "video": float(video_shift),
            }
            or native_slice != {"end_step": effective_end, "start_step": start_step}
            or native_terminal != {"included": expected_terminal == 0.0, "value": expected_terminal}
            or not isinstance(native_fingerprints, dict)
            or not isinstance(native_fingerprints.get("contract"), str)
            or not isinstance(native_fingerprints.get("output"), str)
        ):
            raise ValueError("MiniMax H3 native scheduler metadata drifted")
        errors = torch.abs(sigmas - reference)
        max_abs_error = float(errors.max().item()) if errors.numel() else 0.0
        mean_abs_error = float(errors.mean().item()) if errors.numel() else 0.0
        raw_values = _vector(raw_reference)
        basic_values = _vector(basic_reference)
        reference_values = _vector(reference)
        sigma_values = _vector(sigmas)
        finite = all(
            math.isfinite(value)
            for values in (raw_values, basic_values, reference_values, sigma_values)
            for value in values
        )
        monotonic = all(
            left >= right
            for values in (raw_values, basic_values, reference_values, sigma_values)
            for left, right in pairwise(values)
        )
        trace = {
            "basic_scheduler_sigmas": basic_values,
            "case_id": case_id,
            "finite": finite,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "monotonic_nonincreasing": monotonic,
            "raw_reference_sigmas": raw_values,
            "reference_sigmas": reference_values,
            "schedule_info": info,
            "scheduler": scheduler,
            "schema": "sigmax.minimax-h3-native-matrix-trace/1",
            "sigmas": sigma_values,
            "steps": steps,
        }
        return {
            "ui": {
                _MINIMAX_H3_NATIVE_H2_UI_KEY: [
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


class MiniMaxH3NativeUnexpectedSuccessProbe:
    """Make a negative graph executable while rejecting any unexpected upstream success."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only MiniMax H3 negative execution boundary."
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
        del sigmas, schedule_info
        raise ValueError("MiniMax H3 negative graph unexpectedly reached its output probe")


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


class Lumina2ScheduleProbe:
    """Return model-free H2 evidence for Lumina-Image 2.0."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only Lumina-Image 2.0 schedule H2 execution probe."
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
            raise ValueError("Lumina2 H2 sigmas must be a CPU tensor")
        if not isinstance(schedule_info, str):
            raise ValueError("Lumina2 H2 schedule information must be text")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("Lumina2 H2 schedule information must be an object")
        profile = info.get("profile", {})
        if (
            sigmas.dtype != torch.float32
            or sigmas.ndim != 1
            or len(sigmas) != 51
            or info.get("schema") != "sigmax.lumina2-sigma-node/1"
            or not isinstance(profile, dict)
            or profile.get("id") != "lumina2.v2.official"
            or profile.get("evidence") != "official"
            or info.get("shift") != {"kind": "direct_ratio", "multiplier": 1.0, "ratio": 6.0}
            or info.get("slicing", {}).get("output_steps") != 50
            or float(sigmas[0]) != 1.0
            or float(sigmas[-1]) != 0.0
            or any(float(left) <= float(right) for left, right in pairwise(sigmas))
        ):
            raise ValueError("Lumina2 H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _LUMINA2_UI_KEY: [
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


class LTXScheduleProbe:
    """Return model-free H2 evidence for one explicit LTX generation/stage."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only LTX schedule H2 execution probe."
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
            raise ValueError("LTX H2 sigmas must be a CPU tensor")
        if not isinstance(schedule_info, str):
            raise ValueError("LTX H2 schedule information must be text")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("LTX H2 schedule information must be an object")
        generation = info.get("generation")
        stage = info.get("stage")
        expected_map: dict[tuple[str, str], tuple[str, int, float | None]] = {
            ("LTXV 0.9.8", "Dev"): ("ltxv.0.9.8.dev", 20, 2.05),
            ("LTX-2 19B", "Distilled Stage 1"): (
                "ltx2.19b.distilled.stage1",
                8,
                None,
            ),
            ("LTX-2.3 22B", "Dev"): ("ltx2.3.22b.dev", 30, 2.05),
            ("LTX-2.3 22B", "Distilled Stage 2"): (
                "ltx2.3.22b.distilled.stage2",
                3,
                None,
            ),
        }
        if not isinstance(generation, str) or not isinstance(stage, str):
            raise ValueError("LTX H2 generation/stage metadata must be text")
        expected = expected_map.get((generation, stage))
        profile = info.get("profile")
        shift = info.get("shift")
        expected_start = 0.909375 if stage == "Distilled Stage 2" else 1.0
        valid_shift = (
            shift is None
            if expected is not None and expected[2] is None
            else isinstance(shift, int | float)
            and expected is not None
            and expected[2] is not None
            and abs(float(shift) - expected[2]) <= 1e-12
        )
        if (
            sigmas.dtype != torch.float32
            or sigmas.ndim != 1
            or expected is None
            or len(sigmas) != expected[1] + 1
            or info.get("schema") != "sigmax.ltx-sigma-node/1"
            or profile != expected[0]
            or info.get("slicing", {}).get("output_steps") != expected[1]
            or not valid_shift
            or abs(float(sigmas[0]) - expected_start) > 1e-6
            or float(sigmas[-1]) != 0.0
            or any(float(left) <= float(right) for left, right in pairwise(sigmas))
        ):
            raise ValueError("LTX H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _LTX_UI_KEY: [
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


class HunyuanImage21ScheduleProbe:
    """Return model-free H2 evidence for one HunyuanImage 2.1 variant."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only HunyuanImage 2.1 schedule H2 execution probe."
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
            raise ValueError("HunyuanImage 2.1 H2 sigmas must be a CPU tensor")
        if not isinstance(schedule_info, str):
            raise ValueError("HunyuanImage 2.1 H2 schedule information must be text")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("HunyuanImage 2.1 H2 schedule information must be an object")
        profile = info.get("profile", {})
        if not isinstance(profile, dict):
            raise ValueError("HunyuanImage 2.1 H2 profile is malformed")
        variant = profile.get("variant")
        expected = (
            {
                "2.1": ("hunyuan-image-2-1.base.official", 5.0, 50),
                "2.1-distilled": ("hunyuan-image-2-1.distilled.official", 4.0, 8),
            }.get(variant)
            if isinstance(variant, str)
            else None
        )
        if (
            sigmas.dtype != torch.float32
            or sigmas.ndim != 1
            or expected is None
            or len(sigmas) != expected[2] + 1
            or info.get("schema") != "sigmax.hunyuan-image-2-1-sigma-node/1"
            or profile.get("id") != expected[0]
            or profile.get("evidence") != "official"
            or info.get("shift")
            != {"kind": "direct_ratio", "multiplier": 1.0, "ratio": expected[1]}
            or info.get("slicing", {}).get("output_steps") != expected[2]
            or float(sigmas[0]) != 1.0
            or float(sigmas[-1]) != 0.0
            or any(float(left) <= float(right) for left, right in pairwise(sigmas))
        ):
            raise ValueError("HunyuanImage 2.1 H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _HUNYUAN_IMAGE21_UI_KEY: [
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


class AnimaScheduleProbe:
    """Return model-free H2 evidence for one explicit Anima v1 variant."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only Anima v1 schedule H2 execution probe."
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
            raise ValueError("Anima H2 sigmas must be a CPU tensor")
        if not isinstance(schedule_info, str):
            raise ValueError("Anima H2 schedule information must be text")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("Anima H2 schedule information must be an object")
        profile = info.get("profile", {})
        if not isinstance(profile, dict):
            raise ValueError("Anima H2 profile is malformed")
        variant = profile.get("variant")
        if not isinstance(variant, str):
            raise ValueError("Anima H2 profile variant is malformed")
        expected = {
            "base-v1.0": ("anima.base.framework-reference", 30, 50),
            "aesthetic-v1": ("anima.aesthetic.framework-reference", 30, 50),
            "turbo-v1.0": ("anima.turbo.framework-reference", 8, 12),
        }.get(variant)
        steps = info.get("slicing", {}).get("output_steps")
        if (
            expected is None
            or sigmas.dtype != torch.float32
            or sigmas.ndim != 1
            or type(steps) is not int
            or steps < expected[1]
            or steps > expected[2]
            or len(sigmas) != steps + 1
            or info.get("schema") != "sigmax.anima-sigma-node/1"
            or profile.get("id") != expected[0]
            or profile.get("evidence") != "framework_reference"
            or info.get("shift") != {"kind": "rational", "multiplier": 1.0, "shift": 3.0}
            or float(sigmas[0]) != 1.0
            or float(sigmas[-1]) != 0.0
            or any(float(left) <= float(right) for left, right in pairwise(sigmas))
        ):
            raise ValueError("Anima H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": _vector(sigmas)}
        return {
            "ui": {
                _ANIMA_UI_KEY: [
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


class WanScheduleProbe:
    """Return model-free H2 evidence for explicit Wan profiles."""

    CATEGORY = "SigmaxTest"
    DESCRIPTION = "Test-only Wan 2.1/2.2 and Wan Animate 2 schedule H2 execution probe."
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
            raise ValueError("Wan H2 sigmas must be a CPU tensor")
        if sigmas.dtype != torch.float32 or sigmas.ndim != 1:
            raise ValueError("Wan H2 sigmas must be a float32 vector")
        if not isinstance(schedule_info, str):
            raise ValueError("Wan H2 schedule information must be text")
        info = json.loads(schedule_info)
        if not isinstance(info, dict):
            raise ValueError("Wan H2 schedule information must be an object")
        profile = info.get("profile", {})
        shift = info.get("shift", {})
        slicing = info.get("slicing", {})
        boundary = info.get("boundary", {})
        if not all(isinstance(item, dict) for item in (profile, shift, slicing, boundary)):
            raise ValueError("Wan H2 schedule information is malformed")
        expected = {
            "wan2.1.t2v.official-native": ("official", 5.0, 50, None),
            "wan2.1.i2v.480p.official-native": ("official", 3.0, 40, None),
            "wan2.2.ti2v.5b.comfy-native": ("framework_reference", 5.0, 50, None),
            "wan2.2.t2v-a14b.official-native": ("official", 12.0, 40, 0.875),
            "wan2.1.flf2v.14b.720p.official-native": ("official", 16.0, 50, None),
            "wan2.1.vace.1.3b.official-native": ("official", 16.0, 50, None),
            "wan2.1.vace.14b.official-native": ("official", 16.0, 50, None),
            "wan2.2.s2v.14b.official-native": ("official", 3.0, 40, None),
            "wan2.2.animate.14b.official-native": ("official", 5.0, 20, None),
            "wan-animate2.14b.base.official-native": ("official", 5.0, 40, None),
            "wan-animate2.14b.distilled.official-native": ("official", 5.0, 10, None),
            "wan-animate2.14b.comfy-optimized-6.framework-reference": (
                "framework_reference",
                5.0,
                6,
                None,
            ),
        }.get(profile.get("id"))
        steps = slicing.get("output_steps")
        float_sigmas = _vector(sigmas)
        expected_boundary = expected[3] if expected is not None else None
        if (
            info.get("schema") != "sigmax.wan-sigma-node/1"
            or expected is None
            or profile.get("evidence") != expected[0]
            or shift != {"kind": "direct_ratio", "multiplier": 1.0, "ratio": expected[1]}
            or info.get("strict_source") is not True
            or type(steps) is not int
            or steps != expected[2]
            or len(float_sigmas) != steps + 1
            or float_sigmas[0] != 1.0
            or float_sigmas[-1] != 0.0
            or any(float(left) <= float(right) for left, right in pairwise(float_sigmas))
            or boundary.get("model_dispatch") is not False
            or boundary.get("routing_owner") != "caller"
            or (expected_boundary is None and boundary.get("step") != -1)
            or (expected_boundary is not None and boundary.get("normalized") != expected_boundary)
        ):
            raise ValueError("Wan H2 schedule contract drifted")
        trace = {"schedule_info": info, "sigmas": float_sigmas}
        return {
            "ui": {
                _WAN_UI_KEY: [
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
        if not isinstance(entry, list | tuple) or len(entry) != 2:
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
    "SigmaxTest.Lumina2ScheduleProbe": Lumina2ScheduleProbe,
    "SigmaxTest.HunyuanImage21ScheduleProbe": HunyuanImage21ScheduleProbe,
    "SigmaxTest.AnimaScheduleProbe": AnimaScheduleProbe,
    "SigmaxTest.WanScheduleProbe": WanScheduleProbe,
    "SigmaxTest.LTXScheduleProbe": LTXScheduleProbe,
    # IMPORTANT: keep H3 probes exported; defining the class alone does not register it in ComfyUI.
    "SigmaxTest.MiniMaxH3ScheduleProbe": MiniMaxH3ScheduleProbe,
    "SigmaxTest.MiniMaxH3NativeModelSource": MiniMaxH3NativeModelSource,
    "SigmaxTest.MiniMaxH3NativeScheduleProbe": MiniMaxH3NativeScheduleProbe,
    "SigmaxTest.MiniMaxH3NativeUnexpectedSuccessProbe": MiniMaxH3NativeUnexpectedSuccessProbe,
    "SigmaxTest.CheckpointEvidenceProbe": CheckpointEvidenceProbe,
    "SigmaxTest.Krea2LoraExperimentalProbe": Krea2LoraExperimentalProbe,
    "SigmaxTest.Krea2ConditioningProbe": Krea2ConditioningProbe,
    "SigmaxTest.Krea2ConditioningSource": Krea2ConditioningSource,
    "SigmaxTest.NativeEulerProbe": NativeEulerProbe,
    "SigmaxTest.FlowEulerContractProbe": FlowEulerContractProbe,
    "SigmaxTest.StochasticFlowEulerContractProbe": StochasticFlowEulerContractProbe,
    "SigmaxTest.AdvancedWorkflowCompatibilityProbe": AdvancedWorkflowCompatibilityProbe,
    "SigmaxTest.SamplerStateContractProbe": SamplerStateContractProbe,
    "SigmaxTest.ScheduleAlgebraProbe": ScheduleAlgebraProbe,
    "SigmaxTest.ZImageScheduleProbe": ZImageScheduleProbe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SigmaxTest.Flux1SchnellScheduleProbe": "Sigmax Test — FLUX.1-schnell Schedule Probe",
    "SigmaxTest.QwenImageScheduleProbe": "Sigmax Test — Qwen Image Schedule Probe",
    "SigmaxTest.SD3ScheduleProbe": "Sigmax Test — SD3 Schedule Probe",
    "SigmaxTest.AuraFlowScheduleProbe": "Sigmax Test — AuraFlow Schedule Probe",
    "SigmaxTest.Lumina2ScheduleProbe": "Sigmax Test — Lumina-Image 2.0 Schedule Probe",
    "SigmaxTest.HunyuanImage21ScheduleProbe": "Sigmax Test — HunyuanImage 2.1 Schedule Probe",
    "SigmaxTest.AnimaScheduleProbe": "Sigmax Test — Anima Schedule Probe",
    "SigmaxTest.WanScheduleProbe": "Sigmax Test — Wan Schedule Probe",
    "SigmaxTest.LTXScheduleProbe": "Sigmax Test — LTX Schedule Probe",
    "SigmaxTest.MiniMaxH3ScheduleProbe": "Sigmax Test — MiniMax H3 Schedule Probe",
    "SigmaxTest.MiniMaxH3NativeModelSource": "Sigmax Test — MiniMax H3 Native Model Source",
    "SigmaxTest.MiniMaxH3NativeScheduleProbe": "Sigmax Test — MiniMax H3 Native Schedule Probe",
    "SigmaxTest.MiniMaxH3NativeUnexpectedSuccessProbe": (
        "Sigmax Test — MiniMax H3 Native Unexpected Success Probe"
    ),
    "SigmaxTest.CheckpointEvidenceProbe": "Sigmax Test — Checkpoint Evidence Probe",
    "SigmaxTest.Krea2LoraExperimentalProbe": "Sigmax Test — Krea 2 LoRA Experimental Probe",
    "SigmaxTest.Krea2ConditioningProbe": "Sigmax Test — Krea 2 Conditioning Probe",
    "SigmaxTest.Krea2ConditioningSource": "Sigmax Test — Krea 2 Conditioning Source",
    "SigmaxTest.NativeEulerProbe": "Sigmax Test — Native Euler Probe",
    "SigmaxTest.FlowEulerContractProbe": "Sigmax Test — Flow Euler Contract Probe",
    "SigmaxTest.StochasticFlowEulerContractProbe": (
        "Sigmax Test — Stochastic Flow Euler Contract Probe"
    ),
    "SigmaxTest.AdvancedWorkflowCompatibilityProbe": (
        "Sigmax Test — Advanced Workflow Compatibility Probe"
    ),
    "SigmaxTest.SamplerStateContractProbe": "Sigmax Test — Sampler State Contract Probe",
    "SigmaxTest.ScheduleAlgebraProbe": "Sigmax Test — Schedule Algebra Probe",
    "SigmaxTest.ZImageScheduleProbe": "Sigmax Test — Z-Image Schedule Probe",
}
