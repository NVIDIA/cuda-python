# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that everything that depends on the cuda-bindings floor agrees with it.

The floor of each CUDA major is declared once, by the `cu12`/`cu13` extras in
cuda_core/pyproject.toml (see cuda_core/cuda/core/_bindings_floor.py). Most
consumers read it from there, but two constraints cannot be derived and are
checked here, as a pre-commit hook and from cuda_core/tests/test_bindings_floor.py:

1. The extras parse: each `cu<N>` extra pins exactly
   `cuda-bindings[...]>=<N>.<minor>.<patch>,==<N>.*`.
2. ci/versions.yml builds each major against a CUDA Toolkit of at least the
   floor's major.minor. A toolkit below the floor's minor cannot build the
   floor's cuda-bindings (the build requires the header cuda-bindings was
   generated from); a toolkit ahead of the floor is the toolkit-bump window
   described in cuda_core/AGENTS.md.
3. No documentation page spells a floor out by hand; docs/source/conf.py
   provides |cuda-bindings-floor-cu12| and |cuda-bindings-floor-cu13|.
   Release notes are history and are exempt.

Usage: python toolshed/check_cuda_core_bindings_floor.py [--repo-root DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOOR_MODULE = Path("cuda_core", "cuda", "core", "_bindings_floor.py")
PYPROJECT = Path("cuda_core", "pyproject.toml")
CI_VERSIONS = Path("ci", "versions.yml")
DOCS_SOURCE = Path("cuda_core", "docs", "source")

_CI_PIN_RE = re.compile(r"^\s+(build|prev_build):\s*\n\s+version:\s*\"(\d+)\.(\d+)(?:\.\d+)?\"", re.M)
# `cuda-bindings >= 13.4.1`, ``cuda-bindings`` 13.4.1 or later, ... (release notes are exempt).
_HAND_WRITTEN_FLOOR_RE = re.compile(r"cuda-bindings[`'\" ]{0,4}(?:>=\s*)?\d+\.\d+\.\d+")


def load_floor_module(repo_root: Path):
    path = repo_root / FLOOR_MODULE
    spec = importlib.util.spec_from_file_location("_cuda_core_bindings_floor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_floors(repo_root: Path) -> dict[int, tuple[int, int, int]]:
    """The floors declared by the pyproject extras; ValueError if they do not parse."""
    with open(repo_root / PYPROJECT, "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    return load_floor_module(repo_root).floors_from_extras(extras)


def ci_pin_problems(floors: dict[int, tuple[int, int, int]], versions_yml: str) -> list[str]:
    """Toolkit pins in ci/versions.yml whose major.minor is not the floor's."""
    pins = {int(major): (int(major), int(minor), key) for key, major, minor in _CI_PIN_RE.findall(versions_yml)}
    problems = []
    for major, floor in floors.items():
        if major not in pins:
            problems.append(f"{CI_VERSIONS}: no build or prev_build toolkit pin for CUDA {major} (floor {floor})")
            continue
        pinned_major, pinned_minor, key = pins[major]
        if pinned_major != floor[0] or pinned_minor < floor[1]:
            problems.append(
                f"{CI_VERSIONS}: cuda.{key}.version is {pinned_major}.{pinned_minor} but the CUDA {major} floor "
                f"is cuda-bindings {'.'.join(map(str, floor))}; the toolkit must not sit below the floor's "
                "major.minor (ahead of it is the toolkit-bump window, see cuda_core/AGENTS.md)"
            )
    for major in pins:
        if major not in floors:
            problems.append(f"{CI_VERSIONS}: pins CUDA {major}, which has no cu{major} extra in {PYPROJECT}")
    return problems


def docs_problems(repo_root: Path) -> list[str]:
    """Documentation pages (release notes excepted) that spell out a floor by hand."""
    problems = []
    for path in sorted((repo_root / DOCS_SOURCE).rglob("*.rst")):
        if "release" in path.relative_to(repo_root / DOCS_SOURCE).parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _HAND_WRITTEN_FLOOR_RE.search(line):
                problems.append(
                    f"{path.relative_to(repo_root).as_posix()}:{number}: spells out a cuda-bindings floor; "
                    "use the |cuda-bindings-floor-cu<major>| substitution from conf.py"
                )
    return problems


def check(repo_root: Path) -> list[str]:
    try:
        floors = read_floors(repo_root)
    except ValueError as exc:
        return [f"{PYPROJECT}: {exc}"]
    problems = ci_pin_problems(floors, (repo_root / CI_VERSIONS).read_text(encoding="utf-8"))
    problems += docs_problems(repo_root)
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
