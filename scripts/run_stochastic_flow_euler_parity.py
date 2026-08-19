"""Run exact M5-04 parity against Diffusers 0.39.0 on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from comfyui_sigmax.adapters.comfyui_flow_euler import (
    TorchFlowEulerNoiseProvider,
    TorchFlowEulerStateOperations,
)
from comfyui_sigmax.core import (
    ExecutionBehavior,
    NoiseOwnership,
    PredictionType,
    SamplerCapabilities,
    SamplerExecutionSpec,
    SamplerState,
    ScheduleOwnership,
    SigmaDomain,
    TerminalRequirement,
    execute_stochastic_flow_euler,
)
from scripts.parity.stochastic_flow_euler_report import (
    DIFFUSERS_VERSION,
    SIGMAS,
    TORCH_VERSION,
    build_parity_report,
    canonical_json,
)


def _require_distribution_version(distribution: str, expected: str) -> str:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"missing parity dependency {distribution}=={expected}") from error
    cpu_build = distribution == "torch" and actual == f"{expected}+cpu"
    if actual != expected and not cpu_build:
        raise RuntimeError(f"parity dependency {distribution} must be {expected}, found {actual}")
    return expected


def _fingerprint_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _spec() -> SamplerExecutionSpec:
    capabilities = SamplerCapabilities(
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
    )
    return SamplerExecutionSpec(
        capabilities=capabilities,
        scheduler_index=0,
        begin_index=0,
        solver_order=1,
        timestep_spacing="explicit_unit_flow",
        random_source_ownership=NoiseOwnership.CALLER,
        per_token_time=None,
        requested_transitions=3,
        requested_model_evaluations=3,
    )


def _run_case() -> tuple[dict[str, object], dict[str, str]]:
    versions = {
        "device": "cpu",
        "diffusers": _require_distribution_version("diffusers", DIFFUSERS_VERSION),
        "torch": _require_distribution_version("torch", TORCH_VERSION),
    }
    torch = cast(Any, importlib.import_module("torch"))
    scheduler_module = importlib.import_module(
        "diffusers.schedulers.scheduling_flow_match_euler_discrete"
    )
    scheduler_type = cast(Any, scheduler_module.FlowMatchEulerDiscreteScheduler)
    initial = torch.tensor((0.75, -0.5, 1.25, -1.0), dtype=torch.float32, device="cpu")

    class RecordingOperations(TorchFlowEulerStateOperations):
        def __init__(self) -> None:
            super().__init__(torch_module=torch)
            self.states: list[Any] = []

        def interpolate(self, x0: object, noise: object, weight: float) -> object:
            result = super().interpolate(x0, noise, weight)
            self.states.append(cast(Any, result).detach().clone())
            return result

    def velocity(state: Any, sigma: float, scheduler_index: int) -> Any:
        return 0.25 * state + sigma + scheduler_index * 0.125

    def run_sigmax(seed: int) -> tuple[Any, RecordingOperations, Any]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        operations = RecordingOperations()
        result = execute_stochastic_flow_euler(
            spec=_spec(),
            sigmas=SIGMAS,
            state=initial.clone(),
            evaluator=velocity,
            noise_provider=TorchFlowEulerNoiseProvider(
                generator=generator,
                torch_module=torch,
            ),
            operations=operations,
        )
        return result, operations, generator

    global_before = torch.random.get_rng_state().clone()
    result, operations, generator = run_sigmax(123)
    repeat, _, _ = run_sigmax(123)
    alternate, _, _ = run_sigmax(124)

    scheduler = scheduler_type(stochastic_sampling=True)
    scheduler.set_timesteps(sigmas=list(SIGMAS[:-1]), device="cpu")
    reference_generator = torch.Generator(device="cpu").manual_seed(123)
    reference_state = initial.clone()
    reference_states: list[Any] = []
    for index, timestep in enumerate(scheduler.timesteps):
        model_output = velocity(reference_state, float(SIGMAS[index]), index)
        reference_state = scheduler.step(
            model_output,
            timestep,
            reference_state,
            generator=reference_generator,
        ).prev_sample
        reference_states.append(reference_state.detach().clone())
    global_after = torch.random.get_rng_state().clone()

    state_operations = TorchFlowEulerStateOperations(torch_module=torch)
    steps: list[dict[str, object]] = []
    for index, (actual, expected) in enumerate(
        zip(operations.states, reference_states, strict=True)
    ):
        errors = (actual - expected).abs()
        steps.append(
            {
                "max_abs_error": repr(float(errors.max().item())),
                "mean_abs_error": repr(float(errors.mean().item())),
                "reference_state_fingerprint": state_operations.fingerprint(expected),
                "scheduler_index": index,
                "sigmax_state_fingerprint": state_operations.fingerprint(actual),
            }
        )

    generator_state = generator.get_state().cpu().contiguous().numpy().tobytes(order="C")
    return (
        {
            "different_seed_diverges": alternate.result_fingerprint != result.result_fingerprint,
            "final_result_fingerprint": result.result_fingerprint,
            "global_rng_unchanged": bool(torch.equal(global_before, global_after)),
            "local_generator_state_fingerprint": _fingerprint_bytes(generator_state),
            "local_generator_state_matches": bool(
                torch.equal(generator.get_state(), reference_generator.get_state())
            ),
            "model_evaluation_count": result.snapshot.effective_model_evaluations,
            "noise_draw_count": len(result.noise_fingerprints),
            "noise_fingerprints": list(result.noise_fingerprints),
            "same_seed_repeat": repeat.result_fingerprint == result.result_fingerprint,
            "schedule_fingerprint": result.schedule_fingerprint,
            "sigmas": list(SIGMAS),
            "steps": steps,
            "terminal_noise_draw": len(result.noise_fingerprints) == len(SIGMAS) - 1,
            "transition_count": result.snapshot.effective_transitions,
        },
        versions,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    try:
        case, environment = _run_case()
        encoded = canonical_json(build_parity_report(case, environment=environment))
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as error:
        print(f"PARITY=FAIL\n{error}", file=sys.stderr)
        return 2
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(f"PARITY=PASS\nREPORT={parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
