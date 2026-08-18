#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a package test in the caller-selected Pixi environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PATHS = {
    "pathfinder": Path("cuda_pathfinder"),
    "bindings": Path("cuda_bindings"),
    "core": Path("cuda_core"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=PROJECT_PATHS)
    args = parser.parse_args()

    pixi = shutil.which("pixi")
    if pixi is None:
        raise RuntimeError("pixi is required for this task but was not found on PATH")

    command = [
        pixi,
        "run",
        "--manifest-path",
        str(REPO_ROOT / PROJECT_PATHS[args.project] / "pixi.toml"),
    ]
    environment = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if environment:
        command.extend(["--environment", environment])
    command.append("test")

    print(f"+ {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)  # noqa: S603


if __name__ == "__main__":
    main()
