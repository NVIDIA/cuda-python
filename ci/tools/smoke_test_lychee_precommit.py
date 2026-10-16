# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline smoke check for the configured native Lychee hook.

Exercise the hook's environment, Python launcher, and native executable against
the README's local links without making external link requests. As an accepted
tradeoff for using a real document, a broken local README link also fails this
check even when the hook itself works.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def write_config(directory: Path) -> None:
    # Use the checked-in hook, including its entry point and pinned dependency.
    # PyYAML is already installed as a dependency of pre-commit.
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    [hook] = [hook for repo in config["repos"] for hook in repo["hooks"] if hook["id"] == "lychee"]
    hook.setdefault("args", []).append("--offline")
    config_path = directory / "pre-commit.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def verify(output: str, exit_code: int) -> int:
    print(output, end="")
    if exit_code == 0 and ("- hook id: lychee" not in output or "Skipped" in output):
        raise SystemExit("The configured Lychee hook did not run")
    return exit_code


def main() -> int:
    # CI uses separate phases so each shell invokes pre-commit directly.
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path, metavar="DIRECTORY")
    mode.add_argument("--verify", nargs=2, metavar=("LOG", "EXIT_CODE"))
    mode.add_argument("--cleanup", type=Path, metavar="DIRECTORY")
    args = parser.parse_args()
    if args.prepare is not None:
        write_config(ROOT / args.prepare)
        return 0
    if args.verify is not None:
        log, exit_code = args.verify
        return verify((ROOT / log).read_text(encoding="utf-8"), int(exit_code))
    directory = (ROOT / args.cleanup).resolve()
    if directory.parent != ROOT or not directory.name.startswith("lychee offline smoke "):
        raise SystemExit("Refusing to remove a directory outside the smoke check's temporary files")
    shutil.rmtree(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
