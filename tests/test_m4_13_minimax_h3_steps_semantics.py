"""M4-13 regression contracts for public MiniMax H3 transition semantics."""

from __future__ import annotations

import json
import math

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.minimax_h3_sigma_scheduler import (
    MiniMaxH3SigmaScheduler,
    build_minimax_h3_sigma_schedule,
)
from comfyui_sigmax.profiles.minimax_h3 import (
    MINIMAX_H3_BASE_FL2VA_SCHEMA,
    MINIMAX_H3_BASE_REF2VA_SCHEMA,
    MiniMaxH3Variant,
    build_minimax_h3_schedule,
)
from comfyui_sigmax.workflows.minimax_h3 import (
    MiniMaxH3WorkflowSpec,
    build_minimax_h3_host_workflow_prompt,
)


@pytest.mark.parametrize("variant", ("H3 Base FL2VA", "H3 Base Ref2VA"))
def test_public_h3_schema_exposes_steps_only(variant: str) -> None:
    del variant
    required = MiniMaxH3SigmaScheduler.INPUT_TYPES()["required"]
    assert tuple(required) == ("variant", "steps", "start_step", "end_step")
    assert "grid_points" not in required
    assert "already_shifted" not in required


@pytest.mark.parametrize("variant", ("H3 Base FL2VA", "H3 Base Ref2VA"))
def test_public_steps_are_transition_count_and_map_to_n_plus_one_sigmas(variant: str) -> None:
    result = build_minimax_h3_sigma_schedule(
        variant=variant,
        steps=20,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)
    assert len(result.sigmas) == 21
    assert info["counts"] == {
        "effective_grid_points": 21,
        "effective_model_evaluations": 20,
        "effective_steps": 20,
        "effective_transitions": 20,
        "requested_grid_points": 21,
        "requested_model_evaluations": 20,
        "requested_steps": 20,
        "requested_transitions": 20,
    }


@pytest.mark.parametrize(
    ("public_variant", "source_variant"),
    (
        ("H3 Base FL2VA", MiniMaxH3Variant.BASE_FL2VA),
        ("H3 Base Ref2VA", MiniMaxH3Variant.BASE_REF2VA),
    ),
)
def test_steps_twenty_matches_source_grid_points_twenty_one(
    public_variant: str, source_variant: MiniMaxH3Variant
) -> None:
    public = build_minimax_h3_sigma_schedule(
        variant=public_variant,
        steps=20,
        start_step=0,
        end_step=-1,
    )
    source = build_minimax_h3_schedule(
        variant=source_variant,
        grid_points=21,
        precision="float32",
    )
    assert public.sigmas == source.sigmas


@pytest.mark.parametrize("variant", ("H3 Base FL2VA", "H3 Base Ref2VA"))
@pytest.mark.parametrize("steps", (1, 4, 8, 20, 128))
def test_public_step_sweep_preserves_endpoint_and_transition_invariants(
    variant: str, steps: int
) -> None:
    result = build_minimax_h3_sigma_schedule(
        variant=variant,
        steps=steps,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)
    assert len(result.sigmas) == steps + 1
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0
    assert all(math.isfinite(value) for value in result.sigmas)
    assert all(left > right for left, right in zip(result.sigmas, result.sigmas[1:], strict=False))
    assert info["counts"]["effective_steps"] == steps
    assert info["counts"]["effective_transitions"] == steps


def test_public_steps_slicing_preserves_transition_boundaries() -> None:
    result = build_minimax_h3_sigma_schedule(
        variant="H3 Base FL2VA",
        steps=20,
        start_step=5,
        end_step=12,
    )
    info = json.loads(result.schedule_info_json)
    assert len(result.sigmas) == 8
    assert info["slicing"] == {
        "available_steps": 20,
        "end_step": 12,
        "output_steps": 7,
        "start_step": 5,
    }


def test_old_grid_points_and_already_shifted_public_contracts_are_rejected() -> None:
    with pytest.raises(TypeError):
        build_minimax_h3_sigma_schedule(
            variant="H3 Base FL2VA",
            grid_points=20,
            start_step=0,
            end_step=-1,
        )  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        build_minimax_h3_sigma_schedule(
            variant="H3 Base FL2VA",
            steps=20,
            start_step=0,
            end_step=-1,
            already_shifted=False,
        )  # type: ignore[call-arg]


def test_workflow_prompt_uses_steps_and_not_legacy_controls() -> None:
    prompt = build_minimax_h3_host_workflow_prompt(
        MiniMaxH3WorkflowSpec(
            variant="H3 Base FL2VA",
            prompt="M4-13 workflow contract",
            steps=20,
        )
    )
    inputs = prompt["7"]["inputs"]
    assert inputs == {
        "end_step": -1,
        "start_step": 0,
        "steps": 20,
        "variant": "H3 Base FL2VA",
    }


def test_profile_diffusers_recipe_uses_public_default_steps() -> None:
    for schema in (MINIMAX_H3_BASE_FL2VA_SCHEMA, MINIMAX_H3_BASE_REF2VA_SCHEMA):
        diffusers = next(
            recipe for recipe in schema.recipes if recipe.recipe_id.endswith("diffusers")
        )
        assert diffusers.steps.default == 20
        assert diffusers.steps.reference_steps == (20,)


def test_steps_bounds_are_transition_based() -> None:
    with pytest.raises(ScheduleContractError, match="steps"):
        build_minimax_h3_sigma_schedule(
            variant="H3 Base FL2VA",
            steps=0,
            start_step=0,
            end_step=-1,
        )
