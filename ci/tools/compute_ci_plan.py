#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute the CI build and test workplan for a pull request."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import PurePosixPath

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

IGNORED_BASENAMES = {"AGENTS.md", "CLAUDE.md"}
IGNORED_PATHS = {
    ".coveragerc",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".spdx-ignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "SECURITY.md",
    "context7.json",
    "greptile.json",
    "pixi.lock",
    "pixi.toml",
    "pytest.ini",
    "ruff.toml",
    "benchmarks/cuda_bindings/pixi.lock",
    "benchmarks/cuda_bindings/pixi.toml",
    "ci/.ci-pipeline-regen.md",
    "ci/ci-pipeline.svg",
    "ci/cleanup-pr-previews",
    "ci/tools/check_mempool_hygiene.py",
    "ci/tools/check_pixi_cuda_version.py",
    "ci/tools/check_release_notes.py",
    "ci/tools/download-wheels",
    "ci/tools/run_pytest_with_stack.py",
    "ci/tools/validate-release-wheels",
    "cuda_bindings/pixi.lock",
    "cuda_bindings/pixi.toml",
    "cuda_core/pixi.lock",
    "cuda_core/pixi.toml",
    "cuda_pathfinder/pixi.lock",
    "cuda_pathfinder/pixi.toml",
}
IGNORED_PREFIXES = (
    ".agents/",
    "benchmarks/cuda_core/",
    "ci/tools/tests/",
    "cuda_python_test_helpers/",
    "toolshed/",
)

TEST_INFRA_PLATFORMS = {
    ".github/workflows/test-wheel-linux.yml": {"linux"},
    ".github/workflows/test-wheel-windows.yml": {"windows"},
    "ci/test-matrix.yml": set(PLATFORMS),
    "ci/tools/configure_driver_mode.ps1": {"windows"},
    "ci/tools/guess_latest.sh": {"linux"},
    "ci/tools/install_gpu_driver.ps1": {"windows"},
    "ci/tools/install_gpu_driver.sh": {"linux"},
    "ci/tools/run-tests": set(PLATFORMS),
    "ci/tools/setup-sanitizer": {"linux"},
}

INDEPENDENT_GITHUB_PATHS = {
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/RELEASE-core.md",
    ".github/actionlint.yaml",
    ".github/copy-pr-bot.yaml",
    ".github/dependabot.yml",
    ".github/labeler.yml",
}
INDEPENDENT_WORKFLOWS = {
    "backport.yml",
    "bandit.yml",
    "build-docs.yml",
    "ci-nightly.yml",
    "ci-pixi-source-test.yml",
    "cleanup-pr-previews.yml",
    "coverage.yml",
    "pr-auto-label.yml",
    "pr-metadata-check.yml",
    "release-cuda-pathfinder.yml",
    "release-upload.yml",
    "release.yml",
    "security-suite.yml",
    "triagelabel.yml",
}
INDEPENDENT_ACTIONS = {"doc_preview", "get_pr_number"}


def _is_independent(path: str) -> bool:
    if path.startswith(IGNORED_PREFIXES):
        return True

    if path in INDEPENDENT_GITHUB_PATHS or path.startswith(".github/ISSUE_TEMPLATE/"):
        return True

    parts = PurePosixPath(path).parts
    if len(parts) >= 3 and parts[:2] == (".github", "workflows"):
        return parts[2] in INDEPENDENT_WORKFLOWS
    if len(parts) >= 3 and parts[:2] == (".github", "actions"):
        return parts[2] in INDEPENDENT_ACTIONS
    return False


def compute_workplan(
    paths: list[str],
    *,
    merge_base: str,
    baseline_run_id: str,
    baseline_sha: str,
) -> dict[str, object]:
    """Return the final CI decisions for the supplied changed paths."""
    source_changes: set[str] = set()
    test_changes: set[str] = set()
    test_platforms: set[str] = set()
    force_all = not merge_base or not baseline_run_id or not baseline_sha

    if not force_all:
        for path in paths:
            path_parts = PurePosixPath(path).parts
            if not path_parts or path in IGNORED_PATHS or path_parts[-1] in IGNORED_BASENAMES:
                continue

            if path == "README.md":
                # cuda_python/README.md is a tracked symlink to this sdist input.
                source_changes.add("python")
                continue

            module = PACKAGE_MODULES.get(path_parts[0])
            if module is not None and len(path_parts) > 1:
                relative = path_parts[1:]
                if relative[0] == "docs":
                    continue
                if relative[0] in {"tests", "examples"} or (module == "core" and relative == ("pytest.ini",)):
                    test_changes.add(module)
                else:
                    source_changes.add(module)
                continue

            if path.startswith("cuda_python_test_helpers/cuda_python_test_helpers/"):
                test_changes.update(("bindings", "core"))
            elif path.startswith("benchmarks/cuda_bindings/"):
                test_changes.add("bindings")
            elif platforms := TEST_INFRA_PLATFORMS.get(path):
                test_platforms.update(platforms)
            elif not _is_independent(path):
                force_all = True

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


def _changed_paths(merge_base: str, head: str) -> list[str]:
    result = subprocess.run(  # noqa: S603 - argv is passed directly to git without a shell.
        ["git", "diff", "--no-renames", "--name-only", "-z", f"{merge_base}...{head}"],  # noqa: S607
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path.decode("utf-8", errors="surrogateescape") for path in result.stdout.split(b"\0") if path]


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
    paths = _changed_paths(args.merge_base, args.head) if reusable_baseline else []
    plan = compute_workplan(
        paths,
        merge_base=args.merge_base,
        baseline_run_id=args.baseline_run_id,
        baseline_sha=args.baseline_sha,
    )
    print(json.dumps(plan, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
