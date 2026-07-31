#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute the package build and test closure for a set of changed paths."""

from __future__ import annotations

import argparse
from pathlib import Path

PACKAGES = {
    "cuda_pathfinder": "pathfinder",
    "cuda_bindings": "bindings",
    "cuda_core": "core",
    "cuda_python": "python",
}

SHARED_PREFIXES = (
    ".github/",
    "ci/",
    "scripts/",
    "toolshed/",
)

SHARED_FILES = {
    ".pre-commit-config.yaml",
    "conftest.py",
    "pixi.lock",
    "pixi.toml",
    "pytest.ini",
    "ruff.toml",
}

KNOWN_REPOSITORY_FILES = {
    ".git-blame-ignore-revs",
    ".gitignore",
    "AGENTS.md",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
}


def _bool(value: bool) -> str:
    return str(value).lower()


def _read_paths(path: Path) -> list[str]:
    return [value.decode("utf-8", errors="surrogateescape") for value in path.read_bytes().split(b"\0") if value]


def compute_plan(paths: list[str]) -> dict[str, bool]:
    source = dict.fromkeys(PACKAGES.values(), False)
    tests = dict.fromkeys(PACKAGES.values(), False)
    docs = False
    test_helpers = False
    shared = False
    unknown = False

    for path in paths:
        package_dir, separator, relative_path = path.partition("/")
        package = PACKAGES.get(package_dir)
        if package is not None and separator:
            if relative_path.startswith("docs/"):
                docs = True
            elif relative_path.startswith(("tests/", "examples/")):
                tests[package] = True
            else:
                source[package] = True
            continue

        if path.startswith("cuda_python_test_helpers/"):
            test_helpers = True
        elif path.startswith("benchmarks/cuda_bindings/"):
            tests["bindings"] = True
        elif path.startswith(SHARED_PREFIXES) or path in SHARED_FILES:
            shared = True
        elif path in KNOWN_REPOSITORY_FILES:
            # Repository policy and prose files do not affect package artifacts.
            continue
        else:
            unknown = True

    full = shared or unknown

    build_pathfinder = full or source["pathfinder"]
    # Development cuda-python wheels exactly pin cuda-bindings, so a
    # metapackage change needs a matching bindings artifact for its smoke test.
    build_bindings = full or source["pathfinder"] or source["bindings"] or source["python"]
    build_core = full or source["pathfinder"] or source["bindings"] or source["core"]
    # A core-only change can reuse the baseline cuda-python wheel: rebuilding
    # it would also require rebuilding the exact-version cuda-bindings pin.
    build_python = full or source["pathfinder"] or source["bindings"] or source["python"]

    test_pathfinder = full or source["pathfinder"] or tests["pathfinder"]
    test_bindings = full or source["pathfinder"] or source["bindings"] or tests["bindings"] or test_helpers
    test_core = full or source["pathfinder"] or source["bindings"] or source["core"] or tests["core"] or test_helpers
    test_python = (
        full or source["pathfinder"] or source["bindings"] or source["core"] or source["python"] or tests["python"]
    )

    return {
        "shared": shared,
        "unknown": unknown,
        "docs": docs,
        "test_helpers": test_helpers,
        "pathfinder_source": source["pathfinder"],
        "bindings_source": source["bindings"],
        "core_source": source["core"],
        "python_source": source["python"],
        "pathfinder_tests": tests["pathfinder"],
        "bindings_tests": tests["bindings"],
        "core_tests": tests["core"],
        "python_tests": tests["python"],
        "build_pathfinder": build_pathfinder,
        "build_bindings": build_bindings,
        "build_core": build_core,
        "build_python": build_python,
        "test_pathfinder": test_pathfinder,
        "test_bindings": test_bindings,
        "test_core": test_core,
        "test_python": test_python,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths_file",
        type=Path,
        help="NUL-separated changed-path list produced by git diff --name-only -z",
    )
    args = parser.parse_args()

    plan = compute_plan(_read_paths(args.paths_file))
    for key, value in plan.items():
        print(f"{key}={_bool(value)}")


if __name__ == "__main__":
    main()
