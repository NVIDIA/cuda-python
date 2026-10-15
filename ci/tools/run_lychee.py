# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Lychee with this checkout's main-branch source links mapped to local paths."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def repo_link_remaps(root: Path = ROOT) -> list[str]:
    """Apply the same URL-prefix boundaries as the dedicated CI link check."""
    local_prefix = root.resolve().as_uri() + "/"
    arguments = []
    for view in ("blob", "tree"):
        arguments.extend(["--remap", f"^https://github[.]com/NVIDIA/cuda-python/{view}/main/ {local_prefix}"])
    return arguments


def main(arguments: list[str] | None = None) -> int:
    # Use this Python environment's installation, independent of PATH and the working directory.
    executable = Path(sysconfig.get_path("scripts")) / ("lychee.exe" if sys.platform == "win32" else "lychee")
    if not executable.is_file():
        raise SystemExit(
            f"lychee was not found at {executable}; run this script through the configured pre-commit hook"
        )
    arguments = sys.argv[1:] if arguments is None else arguments
    return subprocess.run([str(executable), *repo_link_remaps(), *arguments], check=False).returncode  # noqa: S603


if __name__ == "__main__":
    raise SystemExit(main())
