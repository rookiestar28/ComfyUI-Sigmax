"""Isolated CPU model-free Qwen migration H1/H2; exact host pins, owned cleanup."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_sigmax.adapters.registration import builtin_node_registry  # noqa: E402
from comfyui_sigmax.core import ScheduleContractError  # noqa: E402
from comfyui_sigmax.nodes.qwen_image21_sigma_scheduler import (  # noqa: E402
    build_qwen_image21_sigma_schedule,
)
from comfyui_sigmax.nodes.qwen_image_sigma_scheduler import (  # noqa: E402
    bind_qwen_image_sigma_output_info,
    build_qwen_image_sigma_schedule,
)
from comfyui_sigmax.workflows.qwen_alignment import (  # noqa: E402
    validate_qwen_alignment_workflow,
    validate_qwen_image21_workflows,
)
from scripts import run_comfyui_e2e as host  # noqa: E402

_HOSTS = {
    "0.39.0": "87c32827017c50c6a629da941439015b4ad656e6",  # pragma: allowlist secret
    "0.30.0": "14b05228cef127ce529bc0c08660770d4af3e9a8",  # pragma: allowlist secret
}

# Explicit cases preserve the geometry authority in each trace; no reference-image tokens.
QWEN21_CASES: tuple[dict[str, Any], ...] = (
    {"steps": 25},
    {"steps": 7, "strict_official": False},
    {"steps": 25, "start_step": 3, "end_step": 15},
    {
        "mode": "Diffusers Dynamic",
        "steps": 40,
        "target_source": "Dimensions",
        "width": 1024,
        "height": 1024,
    },
    {
        "mode": "Diffusers Dynamic",
        "steps": 40,
        "target_source": "Dimensions",
        "width": 1024,
        "height": 1536,
    },
    {
        "mode": "Diffusers Dynamic",
        "steps": 40,
        "target_source": "Target Tokens",
        "image_seq_len": 6144,
    },
    {
        "mode": "Diffusers Dynamic",
        "steps": 40,
        "target_source": "Target Tokens",
        "image_seq_len": 9216,
    },
    {
        "mode": "Diffusers Dynamic",
        "steps": 2,
        "target_source": "Target Tokens",
        "image_seq_len": 4096,
        "strict_official": False,
    },
    {
        "mode": "Diffusers Dynamic",
        "steps": 40,
        "target_source": "Target Tokens",
        "image_seq_len": 4096,
        "start_step": 3,
        "end_step": 15,
    },
)


def qwen21_prompt(changes: dict[str, Any]) -> dict[str, Any]:
    inputs = {
        "mode": "Comfy Native",
        "steps": 25,
        "target_source": "None (Native)",
        "width": 0,
        "height": 0,
        "image_seq_len": 0,
        "strict_official": True,
        "start_step": 0,
        "end_step": -1,
        **changes,
    }
    return {
        "1": {"class_type": "Sigmax.QwenImage21SigmaScheduler", "inputs": inputs},
        "2": {
            "class_type": "SigmaxTest.QwenNativeScheduleProbe",
            "inputs": {"sigmas": ["1", 0], "schedule_info": ["1", 1]},
        },
    }


def verify_qwen21_history(
    history: Any, prompt_id: str, *, changes: dict[str, Any]
) -> dict[str, Any]:
    try:
        entry = history[prompt_id]
        if entry["status"]["status_str"] != "success" or entry["status"]["completed"] is not True:
            raise ScheduleContractError("Qwen 2.1 history did not succeed")
        payload = entry["outputs"]["2"]["sigmax_qwen_native"]
        if len(payload) != 1:
            raise ScheduleContractError("Qwen 2.1 history has ambiguous traces")
        trace = json.loads(payload[0])
        inputs = qwen21_prompt(changes)["1"]["inputs"]
        built = build_qwen_image21_sigma_schedule(**inputs)
        quantized = tuple(struct.unpack("!f", struct.pack("!f", x))[0] for x in built.sigmas)
        info = json.loads(bind_qwen_image_sigma_output_info(built, output_sigmas=quantized))
        if tuple(trace["sigmas"]) != quantized or trace["schedule_info"] != info:
            raise ScheduleContractError("Qwen 2.1 vector/profile/authority/fingerprint drift")
        if trace["dtype"] != "float32" or trace["device"] != "cpu":
            raise ScheduleContractError("Qwen 2.1 runtime dtype/device drift")
        # CRITICAL: verify against independently generated Decimal goldens, not only the
        # production builder. A shared transform-order/geometry bug must fail host proof.
        goldens = json.loads((ROOT / "tests/golden/qwen_native_v1.json").read_text())
        candidates = [
            case
            for case in goldens["cases"]
            if case["lane"] == info["profile"]["id"]
            and case["steps"] == inputs["steps"]
            and case["image_seq_len"] == info["target"]["image_seq_len"]
        ]
        if len(candidates) != 1:
            raise ScheduleContractError("Qwen 2.1 case requires a unique independent golden")
        golden = candidates[0]
        bounds = info["slicing"]
        expected = golden["float32"][bounds["start_step"] : bounds["end_step"] + 1]
        if tuple(expected) != quantized:
            raise ScheduleContractError("Qwen 2.1 independent golden mismatch")
        if inputs["mode"] == "Comfy Native":
            native = trace["native_sigmas"]
            if len(native) != len(expected) or not all(math.isfinite(x) for x in native):
                raise ScheduleContractError("Qwen 2.1 native vector is incomplete")
            errors = [abs(a - b) for a, b in zip(expected, native, strict=True)]
            if (
                max(errors) > 2e-7
                or trace["max_abs_error"] != max(errors)
                or trace["mean_abs_error"] != sum(errors) / len(errors)
                or trace["tolerance"] != 2e-7
                or trace["mu"] != 0.69
            ):
                raise ScheduleContractError("Qwen 2.1 native differential drift")
        elif trace["reference_kind"] != "independent_source_golden_checked_by_runner":
            raise ScheduleContractError("Qwen 2.1 dynamic reference label drift")
        return {"status": "succeeded", "golden_case_id": golden["case_id"], **trace}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ScheduleContractError("Qwen 2.1 history is malformed") from exc


def native_prompt(*, steps: int, start: int = 0, end: int = -1) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "Sigmax.QwenImageSigmaScheduler",
            "inputs": {
                "mode": "Comfy Native",
                "steps": steps,
                "image_seq_len": 0,
                "strict_official": steps == 50,
                "start_step": start,
                "end_step": end,
            },
        },
        "2": {
            "class_type": "SigmaxTest.QwenNativeScheduleProbe",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
            },
        },
    }


def verify_native_history(
    history: Any, prompt_id: str, *, steps: int, start: int = 0, end: int = -1
) -> dict[str, Any]:
    try:
        entry = history[prompt_id]
        if entry["status"]["status_str"] != "success" or entry["status"]["completed"] is not True:
            raise ScheduleContractError("Qwen native history did not succeed")
        payload = entry["outputs"]["2"]["sigmax_qwen_native"]
        if len(payload) != 1:
            raise ScheduleContractError("Qwen native history has ambiguous traces")
        trace = json.loads(payload[0])
        expected = build_qwen_image_sigma_schedule(
            mode="Comfy Native",
            steps=steps,
            image_seq_len=0,
            strict_official=steps == 50,
            start_step=start,
            end_step=end,
        )
        quantized = tuple(struct.unpack("!f", struct.pack("!f", x))[0] for x in expected.sigmas)
        info = json.loads(bind_qwen_image_sigma_output_info(expected, output_sigmas=quantized))
        vector = trace["sigmas"]
        native = trace["native_sigmas"]
        if tuple(vector) != quantized or trace["schedule_info"] != info:
            raise ScheduleContractError("Qwen complete vector/profile/output fingerprint drift")
        if len(native) != len(vector) or not all(math.isfinite(x) for x in native):
            raise ScheduleContractError("Qwen actual native vector is incomplete")
        errors = [abs(a - b) for a, b in zip(vector, native, strict=True)]
        if (
            max(errors) > 2e-7
            or trace["max_abs_error"] != max(errors)
            or trace["mean_abs_error"] != sum(errors) / len(errors)
            or trace["tolerance"] != 2e-7
            or trace["mu"] != 1.15
            or trace["dtype"] != "float32"
            or trace["device"] != "cpu"
        ):
            raise ScheduleContractError("Qwen native error/source/dtype evidence drift")
        return {"status": "succeeded", **trace}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ScheduleContractError("Qwen native history is malformed") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.comfyui_root).resolve()
    python = Path(args.host_python).resolve()
    expected_revision = _HOSTS.get(args.host_version)
    if expected_revision is None or args.expected_revision != expected_revision:
        raise ScheduleContractError("Qwen host requires an exact qualified source/version pair")
    if args.include_qwen21 and args.host_version != "0.39.0":
        raise ScheduleContractError("Qwen 2.1 requires the qualified current 0.39 host")
    if not (source / "main.py").is_file() or not python.is_file():
        raise ScheduleContractError("Qwen host source/interpreter is missing")
    if host._git_revision(source) != expected_revision:
        raise ScheduleContractError("Qwen host source revision drift")
    owned_root = Path(args.temp_root).resolve()
    run_path = host.require_owned_run_path(
        repository_root=ROOT,
        owned_root=owned_root,
        candidate=owned_root / f"qwen-run-{uuid.uuid4().hex}",
    )
    run_path.mkdir(parents=True)
    port = host._select_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = run_path / "comfyui.log"
    process: subprocess.Popen[bytes] | None = None
    succeeded = False
    receipt: dict[str, Any] = {
        "schema": "sigmax.qwen-alignment-host/1",
        "model_execution": "not_performed",
        "host": {"version": args.host_version, "revision": expected_revision},
        "attempt_transitions": {},
        "native_cases": [],
        "legacy_cases": [],
        "qwen21_cases": [],
    }
    try:
        for name in ("base", "input", "output", "temp", "user"):
            (run_path / name).mkdir()
        staged = host._stage_extension(run_path)
        host._stage_h3_test_pack(run_path)
        receipt["import_probe"] = host._run_import_probe(
            host_python=python,
            comfyui_root=source,
            staged_node=staged,
        )
        creationflags = (
            int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
        )
        with log_path.open("wb") as log:
            # SECURITY: fixed inspected host command; exact explicit interpreter/source, owned
            # directories, CPU and loopback whitelist; never invoke a reference setup script.
            process = subprocess.Popen(  # noqa: S603
                host._host_command(
                    host_python=python, comfyui_root=source, run_path=run_path, port=port
                ),
                cwd=run_path,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                start_new_session=os.name == "posix",
            )
            live = host._readiness(
                base_url=base_url, process=process, deadline=time.monotonic() + 90
            )
            receipt["host"]["reported_version"] = host._verify_minimax_h3_live_host_version(
                host._http_json(f"{base_url}/system_stats"),
                expected_version=args.host_version,
            )
            expected_ids = set(builtin_node_registry().class_mappings())
            if not expected_ids <= live.keys():
                raise ScheduleContractError("Qwen H1 is missing Sigmax registrations")
            h1 = validate_qwen_alignment_workflow(live)
            repeat_h1 = validate_qwen_alignment_workflow(
                host._object(host._http_json(f"{base_url}/object_info"), label="repeat Qwen schema")
            )
            receipt["h1"] = h1
            receipt["attempt_transitions"]["h1"] = host.build_verified_host_repeat_transition(
                lane="QWEN_H1",
                first_summary=h1,
                repeat_summary=repeat_h1,
            )
            for steps, start, end in ((50, 0, -1), (27, 0, -1), (1, 0, -1), (50, 3, 20)):

                def submit(
                    ordinal: int, *, n: int = steps, a: int = start, b: int = end
                ) -> tuple[str, dict[str, object]]:
                    return host._submit_successful_prompt(
                        base_url=base_url,
                        client_id=f"sigmax-qwen-native-{n}-{a}-{b}-{ordinal}",
                        prompt=native_prompt(steps=n, start=a, end=b),
                        execution_timeout=45,
                    )

                def verify(
                    history: object, prompt_id: str, *, n: int = steps, a: int = start, b: int = end
                ) -> dict[str, Any]:
                    return verify_native_history(history, prompt_id, steps=n, start=a, end=b)

                summary, transition = host.execute_verified_host_repeat(
                    lane="QWEN_NATIVE_H2",
                    submit=submit,
                    verify=verify,
                )
                receipt["native_cases"].append(summary)
                receipt["attempt_transitions"][f"native.{steps}.{start}.{end}"] = transition
            for mode in ("Comfy Fixed", "Diffusers Dynamic"):

                def submit_legacy(
                    ordinal: int, *, selected: str = mode
                ) -> tuple[str, dict[str, object]]:
                    return host._submit_successful_prompt(
                        base_url=base_url,
                        client_id=f"sigmax-qwen-legacy-{ordinal}",
                        prompt=host.build_qwen_image_h2_api_prompt(selected),
                        execution_timeout=45,
                    )

                def verify_legacy(
                    history: object, prompt_id: str, *, selected: str = mode
                ) -> dict[str, object]:
                    return host.verify_qwen_image_h2_history(
                        history, prompt_id=prompt_id, mode=selected
                    )

                summary, transition = host.execute_verified_host_repeat(
                    lane="QWEN_LEGACY_H2",
                    submit=submit_legacy,
                    verify=verify_legacy,
                )
                receipt["legacy_cases"].append(summary)
                receipt["attempt_transitions"][f"legacy.{mode}"] = transition
            if args.include_qwen21:
                h1_21 = validate_qwen_image21_workflows(live)
                repeat_21 = validate_qwen_image21_workflows(
                    host._object(
                        host._http_json(f"{base_url}/object_info"), label="repeat Qwen 2.1 schema"
                    )
                )
                receipt["qwen21_h1"] = h1_21
                receipt["attempt_transitions"]["qwen21.h1"] = (
                    host.build_verified_host_repeat_transition(
                        lane="QWEN21_H1", first_summary=h1_21, repeat_summary=repeat_21
                    )
                )
                for index, changes in enumerate(QWEN21_CASES):

                    def submit_21(
                        ordinal: int, *, case: dict[str, Any] = changes, n: int = index
                    ) -> tuple[str, dict[str, object]]:
                        return host._submit_successful_prompt(
                            base_url=base_url,
                            client_id=f"sigmax-qwen21-{n}-{ordinal}",
                            prompt=qwen21_prompt(case),
                            execution_timeout=45,
                        )

                    def verify_21(
                        history: object, prompt_id: str, *, case: dict[str, Any] = changes
                    ) -> dict[str, Any]:
                        return verify_qwen21_history(history, prompt_id, changes=case)

                    summary, transition = host.execute_verified_host_repeat(
                        lane="QWEN21_H2", submit=submit_21, verify=verify_21
                    )
                    receipt["qwen21_cases"].append(summary)
                    receipt["attempt_transitions"][f"qwen21.{index}"] = transition
            succeeded = True
    finally:
        if process is not None:
            receipt["shutdown"] = host._terminate_owned_process(process, base_url=base_url)
        host._wait_for_port_release(port)
        receipt["port_released"] = True
        receipt["cleanup"] = "removed" if succeeded else "retained_failure_artifacts"
        if log_path.exists():
            receipt["host_log_tail"] = host.redact_text(
                log_path.read_text(encoding="utf-8", errors="replace"),
                sensitive_paths=(
                    ROOT,
                    source,
                    run_path,
                    *host._host_python_redaction_paths(python),
                ),
            )[-8000:]
        host._write_evidence(Path(args.evidence_file), receipt)
        if succeeded:
            # SECURITY: validate the resolved owned descendant again immediately before deletion.
            host.require_owned_run_path(
                repository_root=ROOT, owned_root=owned_root, candidate=run_path
            )
            shutil.rmtree(run_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-qwen21", action="store_true")
    for name in (
        "comfyui-root",
        "host-python",
        "host-version",
        "expected-revision",
        "temp-root",
        "evidence-file",
    ):
        parser.add_argument(f"--{name}", required=True)
    receipt = run(parser.parse_args())
    print(
        json.dumps(
            {
                "host": receipt["host"],
                "transitions": len(receipt["attempt_transitions"]),
                "cleanup": receipt["cleanup"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
