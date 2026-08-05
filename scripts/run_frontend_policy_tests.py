"""Run dependency-free frontend policy tests with a supported Node.js runtime."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests" / "frontend" / "krea2_strict_official_policy.test.mjs"
EXTENSION = ROOT / "web" / "krea2_strict_official_extension.js"
MINIMUM_NODE_MAJOR = 18


def main() -> int:
    node = shutil.which("node")
    if node is None:
        raise SystemExit(
            "[node.missing] Node.js 18+ is required for the frontend policy gate. "
            "Install an active LTS release and retry."
        )
    # SECURITY: the active PATH resolves the executable; every argument and test path is fixed.
    version = subprocess.run(  # noqa: S603
        [node, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    match = re.fullmatch(r"v(\d+)\.\d+\.\d+", version)
    if match is None or int(match.group(1)) < MINIMUM_NODE_MAJOR:
        raise SystemExit(
            f"[node.incompatible] Node.js 18+ is required; active version is {version!r}."
        )
    subprocess.run(  # noqa: S603
        [node, "--experimental-default-type=module", "--test", str(TEST)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(  # noqa: S603
        [node, "--experimental-default-type=module", "--check", str(EXTENSION)],
        cwd=ROOT,
        check=True,
    )
    print(f"FRONTEND_POLICY=PASS node={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
