"""Generate or verify the canonical M8-01 public-contract manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

from comfyui_sigmax.public_contracts import (
    PUBLIC_CONTRACT_MANIFEST_ENVELOPE_SCHEMA,
    source_contract_projection,
)

ROOT: Final = Path(__file__).resolve().parents[1]
TARGET: Final = ROOT / "comfyui_sigmax" / "contracts" / "manifest_v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def build_envelope() -> dict[str, object]:
    """Build the source-derived public contract envelope."""

    manifest = source_contract_projection()
    fingerprint = "sha256:" + hashlib.sha256(_canonical(manifest)).hexdigest()
    return {
        "manifest": manifest,
        "manifest_fingerprint": fingerprint,
        "schema": PUBLIC_CONTRACT_MANIFEST_ENVELOPE_SCHEMA,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    expected = _canonical(build_envelope()) + b"\n"
    if args.check:
        if not TARGET.is_file() or TARGET.read_bytes() != expected:
            raise SystemExit("public contract manifest is stale; regenerate it")
        return 0
    TARGET.write_bytes(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
