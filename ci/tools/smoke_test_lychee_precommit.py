# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline smoke check for the configured native Lychee hook.

Exercise the hook's environment, Python launcher, and native executable against
the README's local links without making external link requests. As an accepted
tradeoff for using a real document, a broken local README link also fails this
check even when the hook itself works.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    # Use the checked-in hook, including its entry point and pinned dependency.
    # PyYAML is already installed as a dependency of pre-commit.
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    [hook] = [hook for repo in config["repos"] for hook in repo["hooks"] if hook["id"] == "lychee"]
    hook.setdefault("args", []).append("--offline")

    # pre-commit resolves --config relative to the checkout; keep it on the same drive.
    with TemporaryDirectory(prefix="lychee offline smoke ", dir=ROOT) as directory:
        config_path = Path(directory) / "pre-commit.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "pre_commit",
                "run",
                "lychee",
                "--config",
                str(config_path),
                "--files",
                "README.md",
                "--verbose",
            ],
            cwd=ROOT,
            env={**os.environ, "SKIP": "", "PYTHONUTF8": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=False,
        )
    print(result.stdout, end="")
    if result.returncode == 0 and ("- hook id: lychee" not in result.stdout or "Skipped" in result.stdout):
        raise SystemExit("The configured Lychee hook did not run")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
