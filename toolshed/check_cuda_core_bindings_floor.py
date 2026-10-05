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
2. ci/versions.yml builds each major against a CUDA Toolkit of at least the
   floor's major.minor. A toolkit below the floor's minor cannot build the
   floor's cuda-bindings, because the build requires the header that
   cuda-bindings was generated from. A toolkit ahead of the floor is the
   toolkit-bump window described in cuda_core/AGENTS.md.
3. No documentation page spells a floor out by hand. docs/source/conf.py
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

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOOR_MODULE = Path("cuda_core", "cuda", "core", "_bindings_floor.py")
PYPROJECT = Path("cuda_core", "pyproject.toml")
CI_VERSIONS = Path("ci", "versions.yml")
DOCS_SOURCE = Path("cuda_core", "docs", "source")

_TOOLKIT_VERSION_RE = re.compile(r"([1-9]\d*)\.(\d+)\.(\d+)")
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


def _ci_toolkit_pins(versions_yml: str) -> dict[int, tuple[int, int, str]]:
    """Read toolkit pins without importing the Python 3.11-only CI package."""
    config = yaml.safe_load(versions_yml)
    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise ValueError("expected the schema_version: 2 bindings package registry")
    cuda = config.get("cuda")
    bindings = cuda.get("bindings") if isinstance(cuda, dict) else None
    roots = bindings.get("package_roots") if isinstance(bindings, dict) else None
    if not isinstance(roots, dict) or not roots:
        raise ValueError("cuda.bindings.package_roots must be a nonempty mapping")

    pins = {}
    for root, package in roots.items():
        key = f"cuda.bindings.package_roots.{root}.toolkit_version"
        version = package.get("toolkit_version") if isinstance(package, dict) else None
        match = _TOOLKIT_VERSION_RE.fullmatch(version) if isinstance(version, str) else None
        if match is None:
            raise ValueError(f"{key} must be a major.minor.patch version string")
        major, minor, _ = map(int, match.groups())
        if major in pins:
            raise ValueError(f"multiple bindings package roots pin CUDA {major}")
        pins[major] = (major, minor, key)
    return pins


def ci_pin_problems(floors: dict[int, tuple[int, int, int]], versions_yml: str) -> list[str]:
    """Registered toolkit pins that sit below the floor's major.minor."""
    try:
        pins = _ci_toolkit_pins(versions_yml)
    except (ValueError, yaml.YAMLError) as exc:
        return [f"{CI_VERSIONS}: {exc}"]
    problems = []
    for major, floor in floors.items():
        if major not in pins:
            problems.append(f"{CI_VERSIONS}: no registered toolkit pin for CUDA {major} (floor {floor})")
            continue
        pinned_major, pinned_minor, key = pins[major]
        if pinned_major != floor[0] or pinned_minor < floor[1]:
            problems.append(
                f"{CI_VERSIONS}: {key} is {pinned_major}.{pinned_minor} but the CUDA {major} floor "
                f"is cuda-bindings {'.'.join(map(str, floor))}. The toolkit must not sit below the floor's "
                "major.minor. A toolkit ahead of the floor is the toolkit-bump window. See cuda_core/AGENTS.md"
            )
    for major in pins:
        if major not in floors:
            problems.append(f"{CI_VERSIONS}: pins CUDA {major}, which has no cu{major} extra in {PYPROJECT}")
    return problems


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
