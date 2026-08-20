#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compute the CI build and test workplan for a pull request."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES = ("pathfinder", "bindings", "core", "python")
VARIANT_MODULES = ("bindings", "core", "python")
CUDA_VARIANTS = ("cu12", "cu13")
PLATFORMS = ("linux", "windows")
PACKAGE_TARGETS = {
    "cuda_pathfinder": ("pathfinder", None),
    "cuda_bindings_12": ("bindings", "cu12"),
    "cuda_bindings": ("bindings", "cu13"),
    "cuda_core": ("core", None),
    "cuda_python": ("python", None),
}
ALL_TARGETS = frozenset(
    {("pathfinder", None), *((module, variant) for module in VARIANT_MODULES for variant in CUDA_VARIANTS)}
)

# Source changes have different build and test consumers. In particular,
# cuda-python source needs a same-version bindings wheel, while a core-only
# change can reuse the baseline cuda-python wheel.
SOURCE_IMPACT = {
    ("pathfinder", None): (ALL_TARGETS, ALL_TARGETS),
    ("bindings", "cu12"): (
        frozenset(
            {
                ("bindings", "cu12"),
                ("core", "cu12"),
                ("core", "cu13"),
                ("python", "cu12"),
            }
        ),
        frozenset({("bindings", "cu12"), ("core", "cu12"), ("python", "cu12")}),
    ),
    ("bindings", "cu13"): (
        frozenset(
            {
                ("bindings", "cu13"),
                ("core", "cu12"),
                ("core", "cu13"),
                ("python", "cu13"),
            }
        ),
        frozenset({("bindings", "cu13"), ("core", "cu13"), ("python", "cu13")}),
    ),
    ("core", None): (
        frozenset({("core", variant) for variant in CUDA_VARIANTS}),
        frozenset({(module, variant) for module in ("core", "python") for variant in CUDA_VARIANTS}),
    ),
    ("python", None): (
        frozenset({(module, variant) for module in ("bindings", "python") for variant in CUDA_VARIANTS}),
        frozenset({("python", variant) for variant in CUDA_VARIANTS}),
    ),
}

SOURCE_SDIST_VARIANTS = {
    ("pathfinder", None): frozenset(CUDA_VARIANTS),
    ("bindings", "cu12"): frozenset({"cu12"}),
    ("bindings", "cu13"): frozenset({"cu13"}),
    ("core", None): frozenset(CUDA_VARIANTS),
    ("python", None): frozenset(CUDA_VARIANTS),
}

RELEASE_TAG_VARIANTS = (
    (re.compile(r"^v12\.9\.\d+$"), "cu12"),
    (re.compile(r"^v13\.\d+\.\d+(?:[ab]\d+)?$"), "cu13"),
)

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
    ".github/workflows/test-wheel-linux.yml": "linux",
    ".github/workflows/test-wheel-windows.yml": "windows",
    "ci/tools/configure_driver_mode.ps1": "windows",
    "ci/tools/guess_latest.sh": "linux",
    "ci/tools/install_gpu_driver.ps1": "windows",
    "ci/tools/install_gpu_driver.sh": "linux",
    "ci/tools/setup-sanitizer": "linux",
}


def compute_workplan(
    paths: list[str],
    *,
    merge_base: str,
    baseline_run_id: str,
    linked_paths: set[str] | None = None,
    release_tag: str = "",
) -> dict[str, object]:
    """Return the final CI decisions for the supplied changed paths."""
    linked_paths = linked_paths or set()
    source_changes: set[tuple[str, str | None]] = set()
    test_changes: set[tuple[str, str | None]] = set()
    test_platforms: set[str] = set()
    release_variant = next(
        (variant for pattern, variant in RELEASE_TAG_VARIANTS if pattern.fullmatch(release_tag)),
        None,
    )
    force_all = (bool(release_tag) and release_variant is None) or (
        release_variant is None and (not merge_base or not baseline_run_id)
    )

    if release_variant is None and not force_all:
        for path in paths:
            path_parts = PurePosixPath(path).parts
            if not path_parts:
                continue

            if platform := TEST_INFRA_PLATFORMS.get(path):
                test_platforms.add(platform)
                continue

            if path_parts[0] == "ci" or (
                len(path_parts) >= 2 and path_parts[:2] in {(".github", "actions"), (".github", "workflows")}
            ):
                force_all = True
                break

            if path_parts[0] == ".github" or path_parts[-1] in IGNORED_BASENAMES:
                continue

            target = PACKAGE_TARGETS.get(path_parts[0])
            if target is not None and len(path_parts) > 1:
                module, variant = target
                relative = path_parts[1:]
                if relative[0] == "docs":
                    continue
                if (
                    any(part in {"test", "tests"} for part in relative[:-1])
                    or relative[0] == "examples"
                    or (module == "core" and relative == ("pytest.ini",))
                ):
                    if variant is None and module in VARIANT_MODULES:
                        test_changes.update((module, cuda_variant) for cuda_variant in CUDA_VARIANTS)
                    else:
                        test_changes.add(target)
                elif PurePosixPath(path).suffix in IGNORED_SUFFIXES and path not in linked_paths:
                    continue
                else:
                    source_changes.add(target)
                continue

            is_test_path = any(part in {"test", "tests"} for part in path_parts[:-1])
            if is_test_path:
                test_changes.update(ALL_TARGETS)
            elif (
                path in IGNORED_PATHS
                or PurePosixPath(path).suffix in IGNORED_SUFFIXES
                or path.startswith(IGNORED_PREFIXES)
            ):
                continue
            elif path_parts[0] in {"benchmarks", "cuda_python_test_helpers"}:
                test_changes.update(ALL_TARGETS)
            else:
                force_all = True
                break

    if release_variant is not None:
        builds = {("bindings", release_variant), ("python", release_variant)}
        tests = set(builds)
        test_platforms = set(PLATFORMS)
        sdist_cuda_variants = {release_variant}
    elif force_all:
        builds = set(ALL_TARGETS)
        tests = set(ALL_TARGETS)
        test_platforms = set(PLATFORMS)
        sdist_cuda_variants = set(CUDA_VARIANTS)
    else:
        builds: set[tuple[str, str | None]] = set()
        tests = set(ALL_TARGETS) if test_platforms else set(test_changes)
        sdist_cuda_variants: set[str] = set()
        for target in source_changes:
            build_impact, test_impact = SOURCE_IMPACT[target]
            builds.update(build_impact)
            tests.update(test_impact)
            sdist_cuda_variants.update(SOURCE_SDIST_VARIANTS[target])
        if source_changes or test_changes:
            test_platforms.update(PLATFORMS)

    modules: dict[str, dict[str, object]] = {
        "pathfinder": {
            "needs_build": ("pathfinder", None) in builds,
            "needs_test": ("pathfinder", None) in tests,
        }
    }
    for module in VARIANT_MODULES:
        variants = {
            variant: {
                "needs_build": (module, variant) in builds,
                "needs_test": (module, variant) in tests,
            }
            for variant in CUDA_VARIANTS
        }
        modules[module] = {
            "needs_build": any(decision["needs_build"] for decision in variants.values()),
            "needs_test": any(decision["needs_test"] for decision in variants.values()),
            "variants": variants,
        }

    test_cuda_variants = {
        variant
        for variant in CUDA_VARIANTS
        if modules["pathfinder"]["needs_test"]
        or any(modules[module]["variants"][variant]["needs_test"] for module in VARIANT_MODULES)
    }
    return {
        "modules": modules,
        "jobs": {
            # These gates cover both optional artifact builds and wheel tests.
            "platforms": {platform: platform in test_platforms for platform in PLATFORMS},
            "sdist_tests": bool(builds),
            "core_api_checks": force_all or ("core", None) in source_changes,
            "test_cuda_majors": {variant: variant in test_cuda_variants for variant in CUDA_VARIANTS},
            "sdist_cuda_majors": {variant: variant in sdist_cuda_variants for variant in CUDA_VARIANTS},
        },
        "merge_base": merge_base,
        "baseline": {
            "run_id": baseline_run_id if release_variant is None and not force_all else "",
            "sha": merge_base if release_variant is None and not force_all else "",
        },
    }


def _git_output(*args: str) -> bytes:
    return subprocess.check_output(  # noqa: S603 - argv is passed directly without a shell.
        ["git", *args],  # noqa: S607
        cwd=REPO_ROOT,
    )


def _changed_paths(merge_base: str) -> tuple[list[str], set[str]]:
    output = _git_output("diff", "--no-renames", "--name-only", "-z", merge_base, "HEAD")
    paths = [path.decode("utf-8", errors="surrogateescape") for path in output.split(b"\0") if path]
    head_symlinks = _tracked_symlink_paths("HEAD")
    # Base links preserve the packaging impact of deleted or replaced symlinks.
    linked_paths = set(head_symlinks) | set(_tracked_symlink_paths(merge_base))
    return _expand_linked_paths(paths, head_symlinks, root=REPO_ROOT), linked_paths


def _tracked_symlink_paths(ref: str) -> list[str]:
    output = _git_output("ls-tree", "--full-tree", "-r", "-z", ref)
    return [
        entry.partition(b"\t")[2].decode("utf-8", errors="surrogateescape")
        for entry in output.split(b"\0")
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
    parser.add_argument("--baseline-run-id", default="")
    parser.add_argument("--release-tag", default="")
    args = parser.parse_args()

    reusable_baseline = bool(args.merge_base and args.baseline_run_id and not args.release_tag)
    paths, linked_paths = _changed_paths(args.merge_base) if reusable_baseline else ([], set())
    plan = compute_workplan(
        paths,
        merge_base=args.merge_base,
        baseline_run_id=args.baseline_run_id,
        linked_paths=linked_paths,
        release_tag=args.release_tag,
    )
    print(json.dumps(plan, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
