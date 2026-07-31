"""Dependency-free optional image benchmark and blind-review contracts."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

IMAGE_BENCHMARK_PROTOCOL_SCHEMA: Final = "sigmax.image-benchmark-protocol/1"
IMAGE_BENCHMARK_PROTOCOL_ENVELOPE_SCHEMA: Final = "sigmax.image-benchmark-protocol-envelope/1"
BLIND_BALLOT_SCHEMA: Final = "sigmax.image-benchmark-blind-ballot/1"
BLIND_REVEAL_SCHEMA: Final = "sigmax.image-benchmark-blind-reveal/1"

_MAX_BYTES: Final = 1_048_576
_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}")
_SEED: Final = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER: Final = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_DECIMAL: Final = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e-[0-9]+)?")
_PRIVATE_PATH: Final = re.compile(r"(?:[A-Za-z]:[\\/]|/|\\\\)")
_SECRET_WORDS: Final = ("authorization", "cookie", "credential", "password", "secret", "token")
_CANDIDATES: Final = (
    ("reference_control", "reference_replay"),
    ("sigmax_candidate", "sigmax_profile"),
)
_CASE_SPECS: Final = {
    "krea2.raw.framework_portrait": {
        "cfg": "5.5",
        "evidence": "framework_reference",
        "profile": "krea2.raw.official",
        "recipe": "krea2.raw.diffusers-reference-28",
        "steps": 28,
        "variant": "RAW",
    },
    "krea2.raw.official_landscape": {
        "cfg": "4.5",
        "evidence": "official",
        "profile": "krea2.raw.official",
        "recipe": "krea2.raw.official-full-52",
        "steps": 52,
        "variant": "RAW",
    },
    "krea2.raw.official_square": {
        "cfg": "4.5",
        "evidence": "official",
        "profile": "krea2.raw.official",
        "recipe": "krea2.raw.official-full-52",
        "steps": 52,
        "variant": "RAW",
    },
    "krea2.turbo.official_square": {
        "cfg": "1",
        "evidence": "official",
        "profile": "krea2.turbo.official",
        "recipe": "krea2.turbo.official-8",
        "steps": 8,
        "variant": "Turbo",
    },
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleContractError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise ScheduleContractError(f"untyped JSON float is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise ScheduleContractError(f"non-finite JSON value is forbidden: {value}")


def _object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ScheduleContractError(f"{name} must be an array")
    return value


def _exact(value: dict[str, object], fields: frozenset[str], *, name: str) -> None:
    if set(value) != fields:
        raise ScheduleContractError(f"{name} fields do not match schema")


def _text(value: object, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > 4096 or (not allow_empty and not value):
        raise ScheduleContractError(f"{name} must be bounded text")
    if any(ord(character) < 32 and character not in "\n\t" for character in value):
        raise ScheduleContractError(f"{name} contains a forbidden control character")
    return value


def _identifier(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if len(text) > 128 or not _IDENTIFIER.fullmatch(text):
        raise ScheduleContractError(f"{name} must be a canonical identifier")
    return text


def _integer(value: object, *, name: str, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ScheduleContractError(f"{name} must be an integer in the allowed range")
    return value


def _fingerprint(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ScheduleContractError(f"{name} must be a lowercase SHA-256 identity")
    return value


def _decimal(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if not _DECIMAL.fullmatch(text):
        raise ScheduleContractError(f"{name} must be a canonical non-negative decimal string")
    return text


def _scan_safe(value: object, *, depth: int = 0) -> None:
    if depth > 24:
        raise ScheduleContractError("image benchmark exceeds maximum depth")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        if len(value) > 4096:
            raise ScheduleContractError("image benchmark string exceeds limit")
        if _PRIVATE_PATH.match(value):
            raise ScheduleContractError("image benchmark contains a private or absolute path")
        return
    if isinstance(value, list):
        if len(value) > 256:
            raise ScheduleContractError("image benchmark collection exceeds limit")
        for child in value:
            _scan_safe(child, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 256:
            raise ScheduleContractError("image benchmark collection exceeds limit")
        for key, child in value.items():
            if not isinstance(key, str) or any(word in key.lower() for word in _SECRET_WORDS):
                raise ScheduleContractError("image benchmark contains a forbidden field name")
            _scan_safe(child, depth=depth + 1)
        return
    raise ScheduleContractError("image benchmark contains an unsupported JSON value")


def _decode(payload: bytes | str) -> dict[str, object]:
    if isinstance(payload, str):
        if payload.startswith("\ufeff"):
            raise ScheduleContractError("image benchmark must not contain a BOM")
        raw = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        raw = payload
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ScheduleContractError("image benchmark must not contain a BOM")
    else:
        raise ScheduleContractError("image benchmark transport must be bytes or text")
    if not raw or len(raw) > _MAX_BYTES:
        raise ScheduleContractError("image benchmark transport size is outside the allowed range")
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, RecursionError, UnicodeError) as exc:
        raise ScheduleContractError("image benchmark transport is not valid JSON") from exc
    root = _object(decoded, name="image benchmark envelope")
    _scan_safe(root)
    if _canonical(root) + b"\n" != raw:
        raise ScheduleContractError("image benchmark transport must use canonical JSON")
    return root


def _validate_dimensions(value: object, *, name: str) -> dict[str, object]:
    dimensions = _object(value, name=name)
    _exact(dimensions, frozenset({"height", "width"}), name=name)
    _integer(dimensions["height"], name=f"{name} height", minimum=16, maximum=16384)
    _integer(dimensions["width"], name=f"{name} width", minimum=16, maximum=16384)
    return dimensions


def _validate_case(value: object) -> tuple[str, int]:
    case = _object(value, name="image benchmark case")
    _exact(
        case,
        frozenset(
            {
                "candidates",
                "id",
                "negative_prompt",
                "profile",
                "prompt",
                "prompt_fingerprint",
                "schedule_evidence",
                "seed",
                "settings",
                "settings_fingerprint",
                "workload",
            }
        ),
        name="image benchmark case",
    )
    identifier = _identifier(case["id"], name="image benchmark case id")
    expected = _CASE_SPECS.get(identifier)
    if expected is None:
        raise ScheduleContractError("image benchmark case is unsupported")
    prompt = _text(case["prompt"], name="image benchmark prompt")
    negative = _text(
        case["negative_prompt"],
        name="image benchmark negative prompt",
        allow_empty=True,
    )
    expected_prompt_identity = _identity(
        _canonical({"negative_prompt": negative, "prompt": prompt})
    )
    if case["prompt_fingerprint"] != expected_prompt_identity:
        raise ScheduleContractError("image benchmark prompt fingerprint drifted")
    seed = _integer(case["seed"], name="image benchmark seed", maximum=2**64 - 1)

    profile = _object(case["profile"], name="image benchmark profile")
    _exact(
        profile,
        frozenset({"evidence", "id", "recipe", "variant", "version"}),
        name="image benchmark profile",
    )
    if profile != {
        "evidence": expected["evidence"],
        "id": expected["profile"],
        "recipe": expected["recipe"],
        "variant": expected["variant"],
        "version": "1",
    }:
        raise ScheduleContractError("image benchmark profile/recipe drifted")

    workload = _object(case["workload"], name="image benchmark workload")
    _exact(workload, frozenset({"effective", "requested"}), name="image benchmark workload")
    _validate_dimensions(workload["requested"], name="requested image dimensions")
    effective = _validate_dimensions(workload["effective"], name="effective image dimensions")
    if cast(int, effective["width"]) % 16 or cast(int, effective["height"]) % 16:
        raise ScheduleContractError("effective image dimensions must be divisible by 16")

    settings = _object(case["settings"], name="image benchmark settings")
    _exact(settings, frozenset({"cfg", "sampler", "steps"}), name="image benchmark settings")
    _decimal(settings["cfg"], name="image benchmark CFG")
    if settings != {"cfg": expected["cfg"], "sampler": "euler", "steps": expected["steps"]}:
        raise ScheduleContractError("image benchmark settings drifted")
    settings_identity = _identity(
        _canonical(
            {
                "profile": profile,
                "prompt_fingerprint": case["prompt_fingerprint"],
                "seed": seed,
                "settings": settings,
                "workload": workload,
            }
        )
    )
    if case["settings_fingerprint"] != settings_identity:
        raise ScheduleContractError("image benchmark settings fingerprint drifted")

    schedule = _object(case["schedule_evidence"], name="image benchmark schedule evidence")
    _exact(
        schedule,
        frozenset(
            {
                "construction_fingerprint",
                "numerical_fingerprint",
                "receipt_fingerprint",
                "receipt_status",
            }
        ),
        name="image benchmark schedule evidence",
    )
    _fingerprint(schedule["construction_fingerprint"], name="construction fingerprint")
    _fingerprint(schedule["numerical_fingerprint"], name="numerical fingerprint")
    _fingerprint(schedule["receipt_fingerprint"], name="source receipt fingerprint")
    if schedule["receipt_status"] != "not_executed":
        raise ScheduleContractError("source schedule receipt must remain not_executed")

    candidates = _array(case["candidates"], name="image benchmark candidates")
    observed_candidates: list[tuple[str, str]] = []
    for candidate_value in candidates:
        candidate = _object(candidate_value, name="image benchmark candidate")
        _exact(
            candidate,
            frozenset({"id", "schedule_provider"}),
            name="image benchmark candidate",
        )
        observed_candidates.append(
            (
                _identifier(candidate["id"], name="image benchmark candidate id"),
                _identifier(candidate["schedule_provider"], name="image schedule provider"),
            )
        )
    if tuple(observed_candidates) != _CANDIDATES:
        raise ScheduleContractError("image benchmark candidate order/roles drifted")
    return identifier, seed


def _validate_protocol(value: object) -> dict[str, object]:
    protocol = _object(value, name="image benchmark protocol")
    _exact(
        protocol,
        frozenset(
            {
                "authority",
                "blind_protocol",
                "cases",
                "execution_state",
                "metrics",
                "numerical_prerequisite",
                "schema",
                "sources",
            }
        ),
        name="image benchmark protocol",
    )
    if protocol["schema"] != IMAGE_BENCHMARK_PROTOCOL_SCHEMA:
        raise ScheduleContractError("image benchmark protocol schema is unsupported")
    authority = _object(protocol["authority"], name="image benchmark authority")
    expected_authority = {
        "cannot_establish": [
            "mathematical_parity",
            "official_profile_status",
            "schedule_correctness",
        ],
        "level": "supplemental_only",
    }
    if authority != expected_authority:
        raise ScheduleContractError("image benchmark authority must remain supplemental only")

    prerequisite = _object(
        protocol["numerical_prerequisite"],
        name="image benchmark numerical prerequisite",
    )
    _exact(
        prerequisite,
        frozenset({"matrix_fingerprint", "required_status", "schema"}),
        name="image benchmark numerical prerequisite",
    )
    _fingerprint(prerequisite["matrix_fingerprint"], name="numerical matrix fingerprint")
    if (
        prerequisite["schema"] != "sigmax.numerical-benchmark-matrix/1"
        or prerequisite["required_status"] != "PASS"
    ):
        raise ScheduleContractError("image benchmark numerical prerequisite is invalid")

    execution_state = _object(protocol["execution_state"], name="image benchmark execution state")
    if execution_state != {
        "component_hashes": None,
        "image_hashes": None,
        "reason": "gpu_model_weights_not_approved",
        "status": "not_executed",
    }:
        raise ScheduleContractError("image benchmark execution state must remain non-PASS")

    metrics = _object(protocol["metrics"], name="image benchmark metrics")
    if metrics != {
        "authority": "supplemental_only",
        "observations": [],
        "required_fields": ["implementation", "metric_id", "value", "version"],
        "value_encoding": "canonical_decimal_string",
    }:
        raise ScheduleContractError("image benchmark metric authority/schema drifted")

    blind = _object(protocol["blind_protocol"], name="blind review protocol")
    if blind != {
        "assignment_algorithm": "sha256-ranked-balanced-ab/1",
        "ballot_schema": BLIND_BALLOT_SCHEMA,
        "commitment_algorithm": "sha256-secret-seed/1",
        "labels": ["A", "B"],
        "reveal_schema": BLIND_REVEAL_SCHEMA,
        "review_sequence": ["ballot_frozen", "votes_frozen", "seed_revealed"],
        "seed_encoding": "lowercase_hex_256bit",
    }:
        raise ScheduleContractError("blind review protocol drifted")

    cases = _array(protocol["cases"], name="image benchmark cases")
    identities_and_seeds = [_validate_case(case) for case in cases]
    identities = [identity for identity, _ in identities_and_seeds]
    seeds = [seed for _, seed in identities_and_seeds]
    if identities != list(_CASE_SPECS) or len(set(seeds)) != len(seeds):
        raise ScheduleContractError("image benchmark case identity/order/seed coverage drifted")

    sources = _array(protocol["sources"], name="image benchmark sources")
    if len(sources) != 1:
        raise ScheduleContractError("image benchmark source coverage is invalid")
    source = _object(sources[0], name="image benchmark source")
    _exact(
        source,
        frozenset({"matrix_fingerprint", "path", "schema", "sha256", "status"}),
        name="image benchmark source",
    )
    _fingerprint(source["matrix_fingerprint"], name="source matrix fingerprint")
    _fingerprint(source["sha256"], name="image benchmark source identity")
    if (
        source["path"] != "comfyui_sigmax/benchmarks/numerical_matrix_v1.json"
        or source["schema"] != "sigmax.numerical-benchmark-matrix-envelope/1"
        or source["status"] != "PASS"
    ):
        raise ScheduleContractError("image benchmark source allowlist/schema/status drifted")
    if source["matrix_fingerprint"] != prerequisite["matrix_fingerprint"]:
        raise ScheduleContractError("image benchmark numerical matrix cross-link drifted")
    return protocol


@dataclass(frozen=True, slots=True)
class ImageBenchmarkProtocol:
    """Canonical immutable optional image benchmark protocol."""

    _protocol_bytes: bytes
    protocol_fingerprint: str

    def __post_init__(self) -> None:
        protocol = _validate_protocol(json.loads(self._protocol_bytes))
        if _canonical(protocol) != self._protocol_bytes:
            raise ScheduleContractError("image benchmark protocol bytes are not canonical")
        if _identity(self._protocol_bytes) != self.protocol_fingerprint:
            raise ScheduleContractError("image benchmark protocol fingerprint drifted")

    def projection(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self._protocol_bytes))


def serialize_image_benchmark_protocol(protocol: ImageBenchmarkProtocol) -> bytes:
    if not isinstance(protocol, ImageBenchmarkProtocol):
        raise ScheduleContractError("protocol must be an ImageBenchmarkProtocol")
    projection = _validate_protocol(protocol.projection())
    protocol_bytes = _canonical(projection)
    if _identity(protocol_bytes) != protocol.protocol_fingerprint:
        raise ScheduleContractError("image benchmark protocol fingerprint drifted")
    return (
        _canonical(
            {
                "protocol": projection,
                "protocol_fingerprint": protocol.protocol_fingerprint,
                "schema": IMAGE_BENCHMARK_PROTOCOL_ENVELOPE_SCHEMA,
            }
        )
        + b"\n"
    )


def load_image_benchmark_protocol(payload: bytes | str | None = None) -> ImageBenchmarkProtocol:
    if payload is None:
        payload = (
            importlib.resources.files("comfyui_sigmax.benchmarks")
            .joinpath("image_protocol_v1.json")
            .read_bytes()
        )
    envelope = _decode(payload)
    _exact(
        envelope,
        frozenset({"protocol", "protocol_fingerprint", "schema"}),
        name="image benchmark protocol envelope",
    )
    if envelope["schema"] != IMAGE_BENCHMARK_PROTOCOL_ENVELOPE_SCHEMA:
        raise ScheduleContractError("image benchmark protocol envelope schema is unsupported")
    protocol = _validate_protocol(envelope["protocol"])
    protocol_bytes = _canonical(protocol)
    fingerprint = _fingerprint(
        envelope["protocol_fingerprint"],
        name="image benchmark protocol fingerprint",
    )
    if fingerprint != _identity(protocol_bytes):
        raise ScheduleContractError("image benchmark protocol fingerprint does not match content")
    return ImageBenchmarkProtocol(
        _protocol_bytes=protocol_bytes,
        protocol_fingerprint=fingerprint,
    )


@dataclass(frozen=True, slots=True)
class CandidateImageEvidence:
    """Hash-only evidence for one explicitly approved completed candidate execution."""

    case_id: str
    candidate_id: str
    execution_status: str
    settings_fingerprint: str
    construction_fingerprint: str
    numerical_fingerprint: str
    receipt_fingerprint: str
    checkpoint_sha256: str
    text_encoder_sha256: str
    vae_sha256: str
    image_sha256: str
    precision: str
    comfyui_version: str
    python_version: str
    torch_version: str
    gpu: str

    def __post_init__(self) -> None:
        _identifier(self.case_id, name="candidate case id")
        _identifier(self.candidate_id, name="candidate id")
        if self.execution_status != "succeeded":
            raise ScheduleContractError("candidate execution status must be succeeded")
        for field_name in (
            "construction_fingerprint",
            "numerical_fingerprint",
            "receipt_fingerprint",
            "settings_fingerprint",
            "checkpoint_sha256",
            "text_encoder_sha256",
            "vae_sha256",
            "image_sha256",
        ):
            _fingerprint(getattr(self, field_name), name=field_name.replace("_", " "))
        if self.precision not in {"bf16", "float32", "quantized"}:
            raise ScheduleContractError("candidate precision is unsupported")
        for field_name in ("comfyui_version", "python_version", "torch_version", "gpu"):
            runtime_value = _text(
                getattr(self, field_name),
                name=f"candidate {field_name.replace('_', ' ')}",
            )
            if _PRIVATE_PATH.match(runtime_value):
                raise ScheduleContractError("candidate runtime contains a private or absolute path")


def _protocol_cases(protocol: ImageBenchmarkProtocol) -> list[dict[str, object]]:
    if not isinstance(protocol, ImageBenchmarkProtocol):
        raise ScheduleContractError("protocol must be an ImageBenchmarkProtocol")
    projection = _validate_protocol(protocol.projection())
    return cast(list[dict[str, object]], projection["cases"])


def _evidence_map(
    protocol: ImageBenchmarkProtocol,
    evidence: Sequence[CandidateImageEvidence],
) -> dict[tuple[str, str], CandidateImageEvidence]:
    cases = _protocol_cases(protocol)
    expected = {
        (cast(str, case["id"]), candidate_id) for case in cases for candidate_id, _ in _CANDIDATES
    }
    observed: dict[tuple[str, str], CandidateImageEvidence] = {}
    for item in evidence:
        if not isinstance(item, CandidateImageEvidence):
            raise ScheduleContractError("candidate evidence entries must be typed")
        key = (item.case_id, item.candidate_id)
        if key not in expected:
            if item.case_id not in {case_id for case_id, _ in expected}:
                raise ScheduleContractError("candidate evidence case is unsupported")
            raise ScheduleContractError("candidate evidence candidate is unsupported")
        if key in observed:
            raise ScheduleContractError("candidate evidence contains a duplicate candidate")
        observed[key] = item
    if set(observed) != expected:
        raise ScheduleContractError("candidate evidence coverage is incomplete")
    if len({item.image_sha256 for item in observed.values()}) != len(observed):
        raise ScheduleContractError("candidate image identities must be unique")
    if len({item.receipt_fingerprint for item in observed.values()}) != len(observed):
        raise ScheduleContractError("candidate receipt identities must be unique")
    cases_by_id = {cast(str, case["id"]): case for case in cases}
    for (case_id, _), item in observed.items():
        schedule = cast(dict[str, object], cases_by_id[case_id]["schedule_evidence"])
        if item.settings_fingerprint != cases_by_id[case_id]["settings_fingerprint"]:
            raise ScheduleContractError("candidate settings fingerprint cross-link drifted")
        if item.construction_fingerprint != schedule["construction_fingerprint"]:
            raise ScheduleContractError("candidate construction fingerprint cross-link drifted")
        if item.numerical_fingerprint != schedule["numerical_fingerprint"]:
            raise ScheduleContractError("candidate numerical fingerprint cross-link drifted")
    return observed


def _assignment(
    protocol: ImageBenchmarkProtocol,
    evidence: Sequence[CandidateImageEvidence],
    *,
    secret_seed: str,
) -> tuple[list[dict[str, str]], dict[tuple[str, str], CandidateImageEvidence]]:
    if not isinstance(secret_seed, str) or not _SEED.fullmatch(secret_seed):
        raise ScheduleContractError("blind review seed must be lowercase 256-bit hexadecimal")
    cases = _protocol_cases(protocol)
    observed = _evidence_map(protocol, evidence)
    ranked = sorted(
        (cast(str, case["id"]) for case in cases),
        key=lambda case_id: hashlib.sha256(f"{secret_seed}:{case_id}".encode()).digest(),
    )
    if len(ranked) % 2:
        raise ScheduleContractError("blind review case coverage must be even for balance")
    reference_first = set(ranked[: len(ranked) // 2])
    assignments: list[dict[str, str]] = []
    for case_id in sorted(ranked):
        if case_id in reference_first:
            image_a_candidate_id, image_b_candidate_id = "reference_control", "sigmax_candidate"
        else:
            image_a_candidate_id, image_b_candidate_id = "sigmax_candidate", "reference_control"
        assignments.append(
            {
                "case_id": case_id,
                "image_a_candidate_id": image_a_candidate_id,
                "image_a_sha256": observed[(case_id, image_a_candidate_id)].image_sha256,
                "image_b_candidate_id": image_b_candidate_id,
                "image_b_sha256": observed[(case_id, image_b_candidate_id)].image_sha256,
            }
        )
    return assignments, observed


def _validate_ballot(value: object) -> dict[str, object]:
    ballot = _object(value, name="blind review ballot")
    _exact(
        ballot,
        frozenset({"authority", "protocol_fingerprint", "schema", "seed_commitment", "trials"}),
        name="blind review ballot",
    )
    if ballot["schema"] != BLIND_BALLOT_SCHEMA or ballot["authority"] != "supplemental_only":
        raise ScheduleContractError("blind review ballot schema/authority is invalid")
    _fingerprint(ballot["protocol_fingerprint"], name="ballot protocol fingerprint")
    _fingerprint(ballot["seed_commitment"], name="ballot seed commitment")
    trials = _array(ballot["trials"], name="blind review trials")
    case_ids: list[str] = []
    images: list[str] = []
    for value in trials:
        trial = _object(value, name="blind review trial")
        _exact(
            trial,
            frozenset({"case_id", "image_a_sha256", "image_b_sha256", "prompt_fingerprint"}),
            name="blind review trial",
        )
        case_ids.append(_identifier(trial["case_id"], name="blind trial case id"))
        _fingerprint(trial["prompt_fingerprint"], name="blind trial prompt fingerprint")
        images.extend(
            [
                _fingerprint(trial["image_a_sha256"], name="blind trial image A"),
                _fingerprint(trial["image_b_sha256"], name="blind trial image B"),
            ]
        )
    if case_ids != sorted(_CASE_SPECS) or len(set(images)) != len(images):
        raise ScheduleContractError("blind review trial coverage/image identities are invalid")
    return ballot


@dataclass(frozen=True, slots=True)
class BlindReviewBallot:
    """Canonical reviewer-facing ballot with candidate identities withheld."""

    _ballot_bytes: bytes
    ballot_fingerprint: str

    def __post_init__(self) -> None:
        ballot = _validate_ballot(json.loads(self._ballot_bytes))
        if _canonical(ballot) != self._ballot_bytes:
            raise ScheduleContractError("blind ballot bytes are not canonical")
        if _identity(self._ballot_bytes) != self.ballot_fingerprint:
            raise ScheduleContractError("blind ballot fingerprint drifted")

    def projection(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self._ballot_bytes))


def build_blind_ballot(
    protocol: ImageBenchmarkProtocol,
    evidence: Sequence[CandidateImageEvidence],
    *,
    secret_seed: str,
) -> BlindReviewBallot:
    assignments, _ = _assignment(protocol, evidence, secret_seed=secret_seed)
    cases = {cast(str, case["id"]): case for case in _protocol_cases(protocol)}
    trials = [
        {
            "case_id": assignment["case_id"],
            "image_a_sha256": assignment["image_a_sha256"],
            "image_b_sha256": assignment["image_b_sha256"],
            "prompt_fingerprint": cast(str, cases[assignment["case_id"]]["prompt_fingerprint"]),
        }
        for assignment in assignments
    ]
    projection = _validate_ballot(
        {
            "authority": "supplemental_only",
            "protocol_fingerprint": protocol.protocol_fingerprint,
            "schema": BLIND_BALLOT_SCHEMA,
            "seed_commitment": _identity(secret_seed.encode("ascii")),
            "trials": trials,
        }
    )
    ballot_bytes = _canonical(projection)
    return BlindReviewBallot(
        _ballot_bytes=ballot_bytes,
        ballot_fingerprint=_identity(ballot_bytes),
    )


def serialize_blind_ballot(ballot: BlindReviewBallot) -> bytes:
    if not isinstance(ballot, BlindReviewBallot):
        raise ScheduleContractError("ballot must be a BlindReviewBallot")
    projection = _validate_ballot(ballot.projection())
    ballot_bytes = _canonical(projection)
    if _identity(ballot_bytes) != ballot.ballot_fingerprint:
        raise ScheduleContractError("blind ballot fingerprint drifted")
    return ballot_bytes + b"\n"


def _validate_reveal(value: object) -> dict[str, object]:
    reveal = _object(value, name="blind review reveal")
    _exact(
        reveal,
        frozenset(
            {
                "assignments",
                "authority",
                "ballot_fingerprint",
                "protocol_fingerprint",
                "schema",
                "secret_seed",
            }
        ),
        name="blind review reveal",
    )
    if reveal["schema"] != BLIND_REVEAL_SCHEMA or reveal["authority"] != "supplemental_only":
        raise ScheduleContractError("blind review reveal schema/authority is invalid")
    _fingerprint(reveal["ballot_fingerprint"], name="reveal ballot fingerprint")
    _fingerprint(reveal["protocol_fingerprint"], name="reveal protocol fingerprint")
    if not isinstance(reveal["secret_seed"], str) or not _SEED.fullmatch(reveal["secret_seed"]):
        raise ScheduleContractError("blind review reveal seed is invalid")
    assignments = _array(reveal["assignments"], name="blind reveal assignments")
    case_ids: list[str] = []
    for value in assignments:
        assignment = _object(value, name="blind reveal assignment")
        _exact(
            assignment,
            frozenset(
                {
                    "case_id",
                    "image_a_candidate_id",
                    "image_a_sha256",
                    "image_b_candidate_id",
                    "image_b_sha256",
                }
            ),
            name="blind reveal assignment",
        )
        case_ids.append(_identifier(assignment["case_id"], name="reveal case id"))
        candidates = {
            _identifier(assignment["image_a_candidate_id"], name="reveal candidate A"),
            _identifier(assignment["image_b_candidate_id"], name="reveal candidate B"),
        }
        if candidates != {candidate_id for candidate_id, _ in _CANDIDATES}:
            raise ScheduleContractError("blind reveal candidate mapping is invalid")
        _fingerprint(assignment["image_a_sha256"], name="reveal image A")
        _fingerprint(assignment["image_b_sha256"], name="reveal image B")
    if case_ids != sorted(_CASE_SPECS):
        raise ScheduleContractError("blind reveal assignment coverage is invalid")
    return reveal


@dataclass(frozen=True, slots=True)
class BlindReviewReveal:
    """Canonical post-vote reveal that verifies one ballot commitment and mapping."""

    _reveal_bytes: bytes
    reveal_fingerprint: str

    def __post_init__(self) -> None:
        reveal = _validate_reveal(json.loads(self._reveal_bytes))
        if _canonical(reveal) != self._reveal_bytes:
            raise ScheduleContractError("blind reveal bytes are not canonical")
        if _identity(self._reveal_bytes) != self.reveal_fingerprint:
            raise ScheduleContractError("blind reveal fingerprint drifted")

    def projection(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self._reveal_bytes))


def build_blind_reveal(
    protocol: ImageBenchmarkProtocol,
    ballot: BlindReviewBallot,
    evidence: Sequence[CandidateImageEvidence],
    *,
    secret_seed: str,
) -> BlindReviewReveal:
    if not isinstance(ballot, BlindReviewBallot):
        raise ScheduleContractError("ballot must be a BlindReviewBallot")
    expected_ballot = build_blind_ballot(protocol, evidence, secret_seed=secret_seed)
    if expected_ballot != ballot:
        raise ScheduleContractError("blind seed commitment or ballot evidence does not match")
    assignments, _ = _assignment(protocol, evidence, secret_seed=secret_seed)
    projection = _validate_reveal(
        {
            "assignments": assignments,
            "authority": "supplemental_only",
            "ballot_fingerprint": ballot.ballot_fingerprint,
            "protocol_fingerprint": protocol.protocol_fingerprint,
            "schema": BLIND_REVEAL_SCHEMA,
            "secret_seed": secret_seed,
        }
    )
    reveal_bytes = _canonical(projection)
    return BlindReviewReveal(
        _reveal_bytes=reveal_bytes,
        reveal_fingerprint=_identity(reveal_bytes),
    )


def serialize_blind_reveal(reveal: BlindReviewReveal) -> bytes:
    if not isinstance(reveal, BlindReviewReveal):
        raise ScheduleContractError("reveal must be a BlindReviewReveal")
    projection = _validate_reveal(reveal.projection())
    reveal_bytes = _canonical(projection)
    if _identity(reveal_bytes) != reveal.reveal_fingerprint:
        raise ScheduleContractError("blind reveal fingerprint drifted")
    return reveal_bytes + b"\n"


__all__ = [
    "BLIND_BALLOT_SCHEMA",
    "BLIND_REVEAL_SCHEMA",
    "IMAGE_BENCHMARK_PROTOCOL_ENVELOPE_SCHEMA",
    "IMAGE_BENCHMARK_PROTOCOL_SCHEMA",
    "BlindReviewBallot",
    "BlindReviewReveal",
    "CandidateImageEvidence",
    "ImageBenchmarkProtocol",
    "build_blind_ballot",
    "build_blind_reveal",
    "load_image_benchmark_protocol",
    "serialize_blind_ballot",
    "serialize_blind_reveal",
    "serialize_image_benchmark_protocol",
]
