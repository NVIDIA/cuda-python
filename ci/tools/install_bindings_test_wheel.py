# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install a cuda.bindings wheel with its source tree's test dependencies."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_SECTION_PATTERN = re.compile(r"\s*\[([^]]+)]\s*(?:#.*)?$")
_KEY_PATTERN = re.compile(r"\s*([A-Za-z0-9_-]+)\s*=")


def _section_has_key(pyproject: Path, section: str, key: str) -> bool:
    active_section = ""
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if match := _SECTION_PATTERN.fullmatch(line):
            active_section = match.group(1).strip()
            continue
        if active_section == section and (match := _KEY_PATTERN.match(line)) and match.group(1) == key:
            return True
    return False


def pip_install_command(wheel: Path, pyproject: Path, with_all: bool) -> list[str]:
    """Return the pip command for either modern dependency groups or legacy extras."""
    wheel_requirement = str(wheel)
    if _section_has_key(pyproject, "dependency-groups", "test"):
        if with_all:
            wheel_requirement += "[all]"
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            wheel_requirement,
            "--group",
            f"{pyproject}:test",
        ]

    if _section_has_key(pyproject, "project.optional-dependencies", "test"):
        extras = "all,test" if with_all else "test"
        return [sys.executable, "-m", "pip", "install", f"{wheel_requirement}[{extras}]"]

    raise ValueError(f"{pyproject} defines neither a test dependency group nor a test extra")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("pyproject", type=Path)
    parser.add_argument("--all", action="store_true", dest="with_all")
    args = parser.parse_args(argv)

    command = pip_install_command(args.wheel, args.pyproject, args.with_all)
    return subprocess.run(command, check=False).returncode  # noqa: S603


if __name__ == "__main__":
    raise SystemExit(main())
