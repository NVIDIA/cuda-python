#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute the CI build and test workplan for a pull request."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES = ("pathfinder", "bindings", "core", "python")
PLATFORMS = ("linux", "windows")
PACKAGE_MODULES = {
    "cuda_pathfinder": "pathfinder",
    "cuda_bindings": "bindings",
    "cuda_core": "core",
    "cuda_python": "python",
}

# Source changes have different build and test consumers. In particular,
# cuda-python source needs a same-version bindings wheel, while a core-only
# change can reuse the baseline cuda-python wheel.
SOURCE_IMPACT = {
    "pathfinder": (set(MODULES), set(MODULES)),
    "bindings": ({"bindings", "core", "python"}, {"bindings", "core", "python"}),
    "core": ({"core"}, {"core", "python"}),
    "python": ({"bindings", "python"}, {"python"}),
}

IGNORED_BASENAMES = {"AGENTS.md", "CLAUDE.md", "pixi.lock", "pixi.toml"}
IGNORED_SUFFIXES = {".md", ".svg"}
IGNORED_PATHS = {
    ".coveragerc",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".spdx-ignore",
    "LICENSE",
    "context7.json",
    "greptile.json",
    "ruff.toml",
}
IGNORED_PREFIXES = (".agents/", "toolshed/")

# Only infrastructure exclusive to one OS belongs here; other CI paths force a full run.
TEST_INFRA_PLATFORMS = {
    ".github/workflows/test-wheel-linux.yml": {"linux"},
    ".github/workflows/test-wheel-windows.yml": {"windows"},
    "ci/tools/configure_driver_mode.ps1": {"windows"},
    "ci/tools/guess_latest.sh": {"linux"},
    "ci/tools/install_gpu_driver.ps1": {"windows"},
    "ci/tools/install_gpu_driver.sh": {"linux"},
    "ci/tools/setup-sanitizer": {"linux"},
}


def compute_workplan(
    paths: list[str],
    *,
    merge_base: str,
    baseline_run_id: str,
    baseline_sha: str,
    linked_paths: set[str] | None = None,
) -> dict[str, object]:
    """Return the final CI decisions for the supplied changed paths."""
    linked_paths = linked_paths or set()
    source_changes: set[str] = set()
    test_changes: set[str] = set()
    test_platforms: set[str] = set()
    force_all = not merge_base or not baseline_run_id or not baseline_sha

    if not force_all:
        for path in paths:
            path_parts = PurePosixPath(path).parts
            if not path_parts:
                continue

            if platforms := TEST_INFRA_PLATFORMS.get(path):
                test_platforms.update(platforms)
                continue

            if path_parts[0] == "ci" or (
                len(path_parts) >= 2 and path_parts[:2] in {(".github", "actions"), (".github", "workflows")}
            ):
                force_all = True
                break

            if path_parts[0] == ".github" or path_parts[-1] in IGNORED_BASENAMES:
                continue

            module = PACKAGE_MODULES.get(path_parts[0])
            if module is not None and len(path_parts) > 1:
                relative = path_parts[1:]
                if relative[0] == "docs":
                    continue
                if (
                    any(part in {"test", "tests"} for part in relative[:-1])
                    or relative[0] == "examples"
                    or (module == "core" and relative == ("pytest.ini",))
                ):
                    test_changes.add(module)
                elif PurePosixPath(path).suffix in IGNORED_SUFFIXES and path not in linked_paths:
                    continue
                else:
                    source_changes.add(module)
                continue

            is_test_path = any(part in {"test", "tests"} for part in path_parts[:-1])
            if is_test_path:
                test_changes.update(MODULES)
            elif (
                path in IGNORED_PATHS
                or path_parts[-1] in IGNORED_BASENAMES
                or PurePosixPath(path).suffix in IGNORED_SUFFIXES
                or path.startswith(IGNORED_PREFIXES)
            ):
                continue
            elif path_parts[0] in {"benchmarks", "cuda_python_test_helpers"}:
                test_changes.update(MODULES)
            else:
                force_all = True
                break

    if force_all:
        builds = set(MODULES)
        tests = set(MODULES)
        test_platforms = set(PLATFORMS)
    else:
        builds: set[str] = set()
        tests = set(MODULES) if test_platforms else set(test_changes)
        for module in source_changes:
            build_impact, test_impact = SOURCE_IMPACT[module]
            builds.update(build_impact)
            tests.update(test_impact)
        if source_changes or test_changes:
            test_platforms.update(PLATFORMS)

    modules = {
        module: {
            "needs_build": module in builds,
            "needs_test": module in tests,
        }
        for module in MODULES
    }
    return {
        "modules": modules,
        "jobs": {
            # These gates cover both optional artifact builds and wheel tests.
            "platforms": {platform: platform in test_platforms for platform in PLATFORMS},
            "sdist_tests": bool(builds),
            "core_api_checks": force_all or "core" in source_changes,
        },
        "merge_base": merge_base,
        "baseline": {
            "run_id": baseline_run_id if not force_all else "",
            "sha": baseline_sha if not force_all else "",
        },
    }


def _changed_paths(merge_base: str, head: str) -> tuple[list[str], set[str]]:
    result = subprocess.run(  # noqa: S603 - argv is passed directly to git without a shell.
        ["git", "diff", "--no-renames", "--name-only", "-z", f"{merge_base}...{head}"],  # noqa: S607
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
    )
    paths = [path.decode("utf-8", errors="surrogateescape") for path in result.stdout.split(b"\0") if path]
    head_symlinks = _tracked_symlink_paths(head)
    # Base links preserve the packaging impact of deleted or replaced symlinks.
    linked_paths = set(head_symlinks) | set(_tracked_symlink_paths(merge_base))
    return _expand_linked_paths(paths, head_symlinks, root=REPO_ROOT), linked_paths


def _tracked_symlink_paths(ref: str) -> list[str]:
    result = subprocess.run(  # noqa: S603 - the Git ref is passed as an argv element.
        ["git", "ls-tree", "--full-tree", "-r", "-z", ref],  # noqa: S607
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
    )
    return [
        entry.partition(b"\t")[2].decode("utf-8", errors="surrogateescape")
        for entry in result.stdout.split(b"\0")
        if entry.startswith(b"120000 ")
    ]


def _expand_linked_paths(paths: list[str], symlink_paths: list[str], *, root: Path) -> list[str]:
    """Include tracked symlinks whose resolved targets changed."""
    resolved_paths = {(root / path).resolve(strict=False) for path in paths}
    expanded = list(paths)
    selected = set(paths)
    expanded.extend(
        path for path in symlink_paths if path not in selected and (root / path).resolve(strict=False) in resolved_paths
    )
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-base", default="")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--baseline-run-id", default="")
    parser.add_argument("--baseline-sha", default="")
    args = parser.parse_args()

    if bool(args.baseline_run_id) != bool(args.baseline_sha):
        parser.error("baseline run ID and SHA must be supplied together")
    if args.baseline_sha and args.baseline_sha != args.merge_base:
        parser.error("baseline SHA must match the merge base")

    reusable_baseline = bool(args.merge_base and args.baseline_run_id)
    paths, linked_paths = _changed_paths(args.merge_base, args.head) if reusable_baseline else ([], set())
    plan = compute_workplan(
        paths,
        merge_base=args.merge_base,
        baseline_run_id=args.baseline_run_id,
        baseline_sha=args.baseline_sha,
        linked_paths=linked_paths,
    )
    print(json.dumps(plan, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
