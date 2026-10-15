# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Lychee with this checkout's main-branch source links mapped to local paths."""

from __future__ import annotations

import shutil
import subprocess
import sys
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
    executable = shutil.which("lychee")
    if executable is None:
        raise SystemExit("lychee was not found; run this script through the configured pre-commit hook")
    arguments = sys.argv[1:] if arguments is None else arguments
    # Let Python launch the native binary on every platform, including Windows.
    return subprocess.run([executable, *repo_link_remaps(), *arguments], check=False).returncode  # noqa: S603


if __name__ == "__main__":
    raise SystemExit(main())
