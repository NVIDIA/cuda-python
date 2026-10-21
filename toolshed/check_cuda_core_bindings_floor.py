# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that everything that depends on the cuda-bindings floor agrees with it.

The floor of each CUDA major is declared once, by the `cu12`/`cu13` extras in
cuda_core/pyproject.toml (see cuda_core/cuda/core/_bindings_floor.py). Most
consumers read it from there. These constraints cannot be derived from it, so
this script checks them, as a pre-commit hook and from
cuda_core/tests/test_bindings_floor.py:

1. The extras parse: each `cu<N>` extra pins exactly
   `cuda-bindings[...]>=<N>.<minor>.<patch>,<<N+1>`.
2. No documentation page spells a floor out by hand. docs/source/conf.py
   provides |cuda-bindings-floor-cu12| and |cuda-bindings-floor-cu13|.
   Release notes are history and are exempt.

Usage: python toolshed/check_cuda_core_bindings_floor.py [--repo-root DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOOR_MODULE = Path("cuda_core", "cuda", "core", "_bindings_floor.py")
PYPROJECT = Path("cuda_core", "pyproject.toml")
DOCS_SOURCE = Path("cuda_core", "docs", "source")

# `cuda-bindings >= 13.4.1`, ``cuda-bindings`` 13.4.1 or later, ... (release notes are exempt).
_HAND_WRITTEN_FLOOR_RE = re.compile(r"cuda-bindings[`'\" ]{0,4}(?:>=\s*)?\d+\.\d+\.\d+")


def load_floor_module(repo_root: Path):
    path = repo_root / FLOOR_MODULE
    spec = importlib.util.spec_from_file_location("_cuda_core_bindings_floor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_floors(repo_root: Path) -> dict[int, tuple[int, int, int]]:
    """The floors declared by the pyproject extras. Raises ValueError if they do not parse."""
    with open(repo_root / PYPROJECT, "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    return load_floor_module(repo_root).floors_from_extras(extras)


def docs_problems(repo_root: Path) -> list[str]:
    """Documentation pages that spell out a floor by hand. Release notes are exempt."""
    problems = []
    for path in sorted((repo_root / DOCS_SOURCE).rglob("*.rst")):
        if "release" in path.relative_to(repo_root / DOCS_SOURCE).parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _HAND_WRITTEN_FLOOR_RE.search(line):
                problems.append(
                    f"{path.relative_to(repo_root).as_posix()}:{number}: spells out a cuda-bindings floor. "
                    "Use the |cuda-bindings-floor-cu<major>| substitution from conf.py"
                )
    return problems


def check(repo_root: Path) -> list[str]:
    try:
        read_floors(repo_root)
    except ValueError as exc:
        return [f"{PYPROJECT}: {exc}"]
    problems = docs_problems(repo_root)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    problems = check(args.repo_root)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
