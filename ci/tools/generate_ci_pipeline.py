#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render the CI pipeline diagram from its Mermaid source."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "ci" / "ci-pipeline.mmd"
OUTPUT = REPO_ROOT / "ci" / "ci-pipeline.svg"
MERMAID_CLI_IMAGE = (
    "ghcr.io/mermaid-js/mermaid-cli/mermaid-cli"
    "@sha256:fa995339034aae7e5cd4f61482248b7f5c51be355b1a6f6eda11a2bbf8401f5f"  # 12.0.0
)
SVGO_VERSION = "4.1.0"
SPDX_HEADER = (
    "<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->\n"
    "<!-- SPDX-License-Identifier: Apache-2.0 -->\n"
)


def _render() -> bytes:
    docker = shutil.which("docker")
    npx = shutil.which("npx")
    if docker is None or npx is None:
        raise RuntimeError("Docker and npx are required; install Docker and Node.js 16 or newer")

    with tempfile.TemporaryDirectory(prefix="cuda-python-ci-pipeline-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        source_path = temporary_path / SOURCE.name
        rendered_path = temporary_path / "ci-pipeline.svg"
        optimized_path = temporary_path / "ci-pipeline-optimized.svg"
        shutil.copyfile(SOURCE, source_path)

        user_args = []
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            user_args = ["--user", f"{os.getuid()}:{os.getgid()}"]
        command = [
            docker,
            "run",
            "--rm",
            *user_args,
            "--env",
            "HOME=/tmp",
            "--volume",
            f"{temporary_path}:/data",
            MERMAID_CLI_IMAGE,
            "--input",
            f"/data/{source_path.name}",
            "--output",
            f"/data/{rendered_path.name}",
            "--backgroundColor",
            "white",
            "--size",
            "1400",
            "--no-font-embed",
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)  # noqa: S603 - fixed command and paths.
        optimize_command = [
            npx,
            "--yes",
            "--package",
            f"svgo@{SVGO_VERSION}",
            "svgo",
            str(rendered_path),
            "--output",
            str(optimized_path),
            "--multipass",
            "--precision",
            "2",
            "--final-newline",
            "--quiet",
        ]
        subprocess.run(  # noqa: S603 - fixed command and temporary paths.
            optimize_command,
            cwd=REPO_ROOT,
            check=True,
        )
        return SPDX_HEADER.encode("utf-8") + optimized_path.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that ci/ci-pipeline.svg matches the Mermaid source",
    )
    args = parser.parse_args()

    try:
        rendered = _render()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: could not render {SOURCE.relative_to(REPO_ROOT)}: {error}", file=sys.stderr)
        return 2

    if args.check:
        try:
            current = OUTPUT.read_bytes()
        except OSError as error:
            print(f"error: could not read {OUTPUT.relative_to(REPO_ROOT)}: {error}", file=sys.stderr)
            return 2
        if current != rendered:
            print(
                "error: ci/ci-pipeline.svg is stale; regenerate it with `python ci/tools/generate_ci_pipeline.py`",
                file=sys.stderr,
            )
            return 1
        print("ci/ci-pipeline.svg is up to date")
        return 0

    OUTPUT.write_bytes(rendered)
    print(f"generated {OUTPUT.relative_to(REPO_ROOT)} from {SOURCE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
