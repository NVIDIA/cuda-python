#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Install the wheel and dependency inputs used to build native test assets."""

from __future__ import annotations

import argparse
import sys

from ci.tools.artifacts import artifact_wheel, project_path, run


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    wheels = [
        artifact_wheel("pathfinder", "pure"),
        artifact_wheel("bindings", "current"),
        artifact_wheel("core", "current"),
    ]
    command = [sys.executable, "-m", "pip", "install", *(str(wheel) for wheel in wheels)]
    for project in ("bindings", "core"):
        command.extend(["--group", f"{project_path(project) / 'pyproject.toml'}:test"])
    run(command)


if __name__ == "__main__":
    main()
