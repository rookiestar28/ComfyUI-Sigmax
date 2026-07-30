"""Canonical report construction for Krea 2 RAW schedule parity."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from comfyui_sigmax.profiles import (
    KREA2_RAW_DIFFUSERS_REFERENCE_28,
    KREA2_RAW_OFFICIAL_FULL_52,
    build_krea2_raw_schedule,
)
from scripts.parity.krea2_official import (
    KREA_LOCATOR,
    KREA_REVISION,
    KREA_SOURCE_URL,
    official_krea2_raw_case,
)
from scripts.parity.krea2_turbo_report import (
    DIFFUSERS_PIPELINE_BLOB,
    DIFFUSERS_REVISION,
    DIFFUSERS_SCHEDULER_BLOB,
    DIFFUSERS_SOURCE_URL,
    DIFFUSERS_TAG,
    DIFFUSERS_VERSION,
    NUMPY_VERSION,
    TORCH_VERSION,
    _comparison,
    _float32_vector,
    _git_object_chunks,
    _require_mapping,
    _validate_comparison,
)

REPORT_SCHEMA: Final = "sigmax.krea2-raw-parity/1"
KREA_TOLERANCE: Final = 1e-8
DIFFUSERS_TOLERANCE: Final = 1e-6
MU_TOLERANCE: Final = 1e-6
CANONICAL_FLOAT64_SIGNIFICANT_DIGITS: Final = 15
_GEOMETRIES: Final = (
    (256, 256),
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1280, 1280),
    (1360, 768),
    (768, 1360),
)
_RECIPES: Final = (
    KREA2_RAW_DIFFUSERS_REFERENCE_28,
    KREA2_RAW_OFFICIAL_FULL_52,
)
CASE_SPECS: Final = tuple(
    (recipe.recipe_id, recipe.steps, width, height)
    for width, height in _GEOMETRIES
    for recipe in _RECIPES
)
_ROOT_FIELDS: Final = frozenset(
    {
        "cases",
        "configuration",
        "environment",
        "profile",
        "schema",
        "sources",
        "status",
        "tolerances",
    }
)
_CASE_FIELDS: Final = frozenset(
    {
        "case_id",
        "comparisons",
        "diffusers_mu",
        "effective_height",
        "effective_width",
        "evidence",
        "image_seq_len",
        "mu",
        "recipe_id",
        "requested_height",
        "requested_width",
        "steps",
    }
)


def raw_case_id(recipe_id: str, width: int, height: int) -> str:
    suffix = "framework_28" if recipe_id.endswith("reference-28") else "official_52"
    return f"{suffix}_{width}x{height}"


def _canonical_float64_vector(values: Sequence[float]) -> tuple[float, ...]:
    """Remove platform-libm noise below the authoritative float64 tolerance."""

    return tuple(
        float(format(float(value), f".{CANONICAL_FLOAT64_SIGNIFICANT_DIGITS}g")) for value in values
    )


def _fixed_metadata(environment: Mapping[str, str]) -> dict[str, object]:
    return {
        "configuration": {
            "base_grid": "krea.reciprocal_step",
            "canonical_float64_significant_digits": CANONICAL_FLOAT64_SIGNIFICANT_DIGITS,
            "mu": "resolution_linear_unclamped",
            "terminal": "zero",
        },
        "environment": dict(environment),
        "profile": {"id": "krea2.raw.official", "version": "1"},
        "schema": REPORT_SCHEMA,
        "sources": {
            "diffusers": {
                "evidence": "framework_reference",
                "pipeline_blob_chunks": _git_object_chunks(DIFFUSERS_PIPELINE_BLOB),
                "pipeline_locator": "src/diffusers/pipelines/krea2/pipeline_krea2.py:71-80,613-630",
                "revision_chunks": _git_object_chunks(DIFFUSERS_REVISION),
                "scheduler_blob_chunks": _git_object_chunks(DIFFUSERS_SCHEDULER_BLOB),
                "scheduler_locator": (
                    "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py:283-378"
                ),
                "tag": DIFFUSERS_TAG,
                "url": DIFFUSERS_SOURCE_URL,
            },
            "krea": {
                "evidence": "official",
                "locator": f"sampling.py:10-17,{KREA_LOCATOR.split(':', maxsplit=1)[1]}",
                "revision_chunks": _git_object_chunks(KREA_REVISION),
                "url": KREA_SOURCE_URL,
            },
        },
        "tolerances": {
            "diffusers_float32_max_abs": "1e-6",
            "krea_float64_max_abs": "1e-8",
            "mu_max_abs": "1e-6",
        },
    }


def build_parity_report(
    diffusers_cases: Mapping[str, Mapping[str, object]],
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Build all 14 official/framework RAW comparisons."""

    expected_ids = tuple(
        raw_case_id(recipe_id, width, height) for recipe_id, _, width, height in CASE_SPECS
    )
    if tuple(sorted(diffusers_cases)) != tuple(sorted(expected_ids)):
        raise ValueError("diffusers RAW case set is incomplete")
    recipes = {recipe.recipe_id: recipe for recipe in _RECIPES}
    cases: list[dict[str, object]] = []
    for recipe_id, steps, width, height in CASE_SPECS:
        case_id = raw_case_id(recipe_id, width, height)
        framework = _require_mapping(diffusers_cases[case_id], name="diffusers case")
        if set(framework) != {"mu", "sigmas"}:
            raise ValueError("diffusers RAW case fields are incomplete")
        diffusers_mu = float(format(float(cast(float, framework["mu"])), ".15g"))
        diffusers_sigmas = tuple(
            float(value) for value in cast(Sequence[float], framework["sigmas"])
        )
        product = build_krea2_raw_schedule(
            width=width,
            height=height,
            recipe=recipes[recipe_id],
        )
        official = official_krea2_raw_case(width=width, height=height, steps=steps)
        product_mu = float(format(float(official["mu"]), ".15g"))
        canonical_product = _canonical_float64_vector(product.sigmas)
        canonical_official = _canonical_float64_vector(cast(tuple[float, ...], official["sigmas"]))
        if abs(diffusers_mu - product_mu) > MU_TOLERANCE:
            raise ValueError("Diffusers RAW mu exceeds tolerance")
        cases.append(
            {
                "case_id": case_id,
                "comparisons": {
                    "diffusers_float32": _comparison(
                        sigmax=_float32_vector(canonical_product),
                        reference=diffusers_sigmas,
                        dtype="float32",
                        tolerance=DIFFUSERS_TOLERANCE,
                    ),
                    "krea_float64": _comparison(
                        sigmax=canonical_product,
                        reference=canonical_official,
                        dtype="float64",
                        tolerance=KREA_TOLERANCE,
                    ),
                },
                "diffusers_mu": repr(diffusers_mu),
                "effective_height": product.effective_inputs.height,
                "effective_width": product.effective_inputs.width,
                "evidence": recipes[recipe_id].evidence.value,
                "image_seq_len": int(official["image_seq_len"]),
                "mu": repr(product_mu),
                "recipe_id": recipe_id,
                "requested_height": height,
                "requested_width": width,
                "steps": steps,
            }
        )
    report: dict[str, Any] = {
        **_fixed_metadata(environment),
        "cases": cases,
        "status": "PASS",
    }
    return validate_parity_report(report)


def validate_parity_report(report: object) -> dict[str, Any]:
    """Fail closed unless all pinned RAW parity evidence is internally consistent."""

    root = _require_mapping(report, name="report")
    if (
        set(root) != _ROOT_FIELDS
        or root.get("schema") != REPORT_SCHEMA
        or root.get("status") != "PASS"
    ):
        raise ValueError("RAW parity report root is invalid")
    expected_environment = {
        "device": "cpu",
        "diffusers": DIFFUSERS_VERSION,
        "numpy": NUMPY_VERSION,
        "torch": TORCH_VERSION,
    }
    fixed = _fixed_metadata(expected_environment)
    for field in ("configuration", "environment", "profile", "schema", "sources", "tolerances"):
        if root.get(field) != fixed[field]:
            raise ValueError(f"RAW parity {field} metadata is invalid")
    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(CASE_SPECS):
        raise ValueError("RAW parity case set is incomplete")
    recipes = {recipe.recipe_id: recipe for recipe in _RECIPES}
    for raw_case, spec in zip(raw_cases, CASE_SPECS, strict=True):
        case = _require_mapping(raw_case, name="RAW case")
        if set(case) != _CASE_FIELDS:
            raise ValueError("RAW parity case fields are invalid")
        recipe_id, steps, width, height = spec
        official = official_krea2_raw_case(width=width, height=height, steps=steps)
        canonical_mu = float(format(float(official["mu"]), ".15g"))
        expected_scalars = {
            "case_id": raw_case_id(recipe_id, width, height),
            "effective_height": official["effective_height"],
            "effective_width": official["effective_width"],
            "evidence": recipes[recipe_id].evidence.value,
            "image_seq_len": official["image_seq_len"],
            "mu": repr(canonical_mu),
            "recipe_id": recipe_id,
            "requested_height": height,
            "requested_width": width,
            "steps": steps,
        }
        if any(case.get(field) != value for field, value in expected_scalars.items()):
            raise ValueError("RAW parity case identity or geometry is invalid")
        try:
            diffusers_mu = float(cast(str, case.get("diffusers_mu")))
        except (TypeError, ValueError) as error:
            raise ValueError("RAW parity Diffusers mu is invalid") from error
        if abs(diffusers_mu - canonical_mu) > MU_TOLERANCE:
            raise ValueError("RAW parity Diffusers mu exceeds tolerance")
        comparisons = _require_mapping(case.get("comparisons"), name="RAW comparisons")
        if set(comparisons) != {"diffusers_float32", "krea_float64"}:
            raise ValueError("RAW parity comparisons are incomplete")
        _validate_comparison(
            comparisons["krea_float64"],
            steps=steps,
            name="krea_float64",
            dtype="float64",
            tolerance=KREA_TOLERANCE,
        )
        _validate_comparison(
            comparisons["diffusers_float32"],
            steps=steps,
            name="diffusers_float32",
            dtype="float32",
            tolerance=DIFFUSERS_TOLERANCE,
        )
    return cast(dict[str, Any], report)


def canonical_json(report: object) -> str:
    """Serialize validated RAW parity evidence deterministically."""

    return (
        json.dumps(
            validate_parity_report(report),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
