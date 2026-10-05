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
from typing import TYPE_CHECKING

from .bindings_config import BindingsConfigError, load_config

if TYPE_CHECKING:
    from .bindings_config import BindingsConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORMS = ("linux", "windows")
PACKAGE_TARGETS: dict[str, tuple[str, None]] = {
    "cuda_pathfinder": ("pathfinder", None),
    "cuda_core": ("core", None),
    "cuda_python": ("python", None),
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
    ".github/workflows/test-wheel-linux.yml": "linux",
    ".github/workflows/test-wheel-windows.yml": "windows",
    "ci/tools/configure_driver_mode.ps1": "windows",
    "ci/tools/guess_latest.sh": "linux",
    "ci/tools/install_gpu_driver.ps1": "windows",
    "ci/tools/install_gpu_driver.sh": "linux",
    "ci/tools/setup-sanitizer": "linux",
}

Target = tuple[str, str | None]


def compute_workplan(
    *,
    bindings_config: BindingsConfig,
    paths: list[str],
    linked_paths: set[str],
    merge_base: str,
    baseline_run_id: str,
    release_tag: str,
) -> dict[str, object]:
    """Return the final CI decisions for the supplied changed paths."""
    packages = bindings_config.package_roots
    package_roots = tuple(package.package_root for package in packages)
    cuda_variants = tuple(package.cuda_variant for package in packages)

    bindings_targets = frozenset(("bindings", package_root) for package_root in package_roots)
    core_targets = frozenset(("core", variant) for variant in cuda_variants)
    python_targets = frozenset(("python", package_root) for package_root in package_roots)
    all_targets = frozenset({("pathfinder", None), *bindings_targets, *core_targets, *python_targets})

    package_targets = {PurePosixPath(package.package_root): ("bindings", package.package_root) for package in packages}
    package_targets.update((PurePosixPath(source_dir), target) for source_dir, target in PACKAGE_TARGETS.items())

    # Source changes have different build/test consumers. Core-only changes can
    # reuse baseline cuda-python wheels; cuda-python changes cannot reuse bindings.
    source_impacts: dict[Target, tuple[frozenset[Target], frozenset[Target], frozenset[str]]] = {
        ("pathfinder", None): (all_targets, all_targets, frozenset(package_roots)),
        ("core", None): (core_targets, frozenset({*core_targets, *python_targets}), frozenset(package_roots)),
        ("python", None): (
            frozenset({*bindings_targets, *python_targets}),
            python_targets,
            frozenset(package_roots),
        ),
        **{
            ("bindings", package.package_root): (
                frozenset(
                    {
                        ("bindings", package.package_root),
                        ("python", package.package_root),
                        *core_targets,
                    }
                ),
                frozenset(
                    {
                        ("bindings", package.package_root),
                        ("core", package.cuda_variant),
                        ("python", package.package_root),
                    }
                ),
                frozenset({package.package_root}),
            )
            for package in packages
        },
    }

    source_changes: set[tuple[str, str | None]] = set()
    test_changes: set[tuple[str, str | None]] = set()
    test_platforms: set[str] = set()
    release_package = bindings_config.match_tag(release_tag) if release_tag else None
    if release_tag.startswith("v") and release_package is None:
        raise BindingsConfigError(f"no configured CUDA bindings package root matches release tag: {release_tag!r}")
    force_all = release_package is None and (bool(release_tag) or not merge_base or not baseline_run_id)

    if release_package is None and not force_all:
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

            target_match = next(
                (
                    (root, target)
                    for root, target in sorted(
                        package_targets.items(), key=lambda item: len(item[0].parts), reverse=True
                    )
                    if len(path_parts) > len(root.parts) and path_parts[: len(root.parts)] == root.parts
                ),
                None,
            )
            if target_match is not None:
                root, target = target_match
                module, _ = target
                relative = path_parts[len(root.parts) :]
                if relative[0] == "docs":
                    continue
                if (
                    any(part in {"test", "tests"} for part in relative[:-1])
                    or relative[0] == "examples"
                    or (module == "core" and relative == ("pytest.ini",))
                ):
                    if module == "core":
                        test_changes.update(core_targets)
                    elif module == "python":
                        test_changes.update(python_targets)
                    else:
                        test_changes.add(target)
                elif PurePosixPath(path).suffix in IGNORED_SUFFIXES and path not in linked_paths:
                    continue
                else:
                    source_changes.add(target)
                continue

            is_test_path = any(part in {"test", "tests"} for part in path_parts[:-1])
            if is_test_path:
                test_changes.update(all_targets)
            elif (
                path in IGNORED_PATHS
                or PurePosixPath(path).suffix in IGNORED_SUFFIXES
                or path.startswith(IGNORED_PREFIXES)
            ):
                continue
            elif path_parts[0] in {"benchmarks", "cuda_python_test_helpers"}:
                test_changes.update(all_targets)
            else:
                force_all = True
                break

    if release_package is not None:
        builds = {
            ("bindings", release_package.package_root),
            ("python", release_package.package_root),
        }
        tests = set(builds)
        test_platforms = set(PLATFORMS)
        sdist_package_roots = {release_package.package_root}
    elif force_all:
        builds = set(all_targets)
        tests = set(all_targets)
        test_platforms = set(PLATFORMS)
        sdist_package_roots = set(package_roots)
    else:
        builds: set[tuple[str, str | None]] = set()
        tests = set(all_targets) if test_platforms else set(test_changes)
        sdist_package_roots: set[str] = set()
        for target in source_changes:
            build_impact, test_impact, sdist_impact = source_impacts[target]
            builds.update(build_impact)
            tests.update(test_impact)
            sdist_package_roots.update(sdist_impact)
        if source_changes or test_changes:
            test_platforms.update(PLATFORMS)

    def flags(module: str, selectors: tuple[str | None, ...]) -> dict[str, bool]:
        return {
            "needs_build": any((module, selector) in builds for selector in selectors),
            "needs_test": any((module, selector) in tests for selector in selectors),
        }

    modules: dict[str, dict[str, object]] = {"pathfinder": flags("pathfinder", (None,))}
    for module in ("bindings", "python"):
        package_decisions = {package.package_root: flags(module, (package.package_root,)) for package in packages}
        modules[module] = {
            **flags(module, package_roots),
            "package_roots": package_decisions,
        }

    cuda_major_decisions = {variant: flags("core", (variant,)) for variant in cuda_variants}
    modules["core"] = {
        **flags("core", cuda_variants),
        "cuda_majors": cuda_major_decisions,
    }

    pathfinder_test = ("pathfinder", None) in tests
    test_cuda_variants = {
        package.cuda_variant
        for package in packages
        if pathfinder_test
        or ("core", package.cuda_variant) in tests
        or any((module, package.package_root) in tests for module in ("bindings", "python"))
    }
    return {
        "modules": modules,
        "sources": {
            # Focused bindings releases intentionally omit unrelated artifacts.
            "pathfinder": "published" if release_package is not None else "artifact",
        },
        "jobs": {
            # These gates cover both optional artifact builds and wheel tests.
            "platforms": {platform: platform in test_platforms for platform in PLATFORMS},
            "sdist_tests": bool(sdist_package_roots),
            "core_api_checks": force_all or ("core", None) in source_changes,
            "test_cuda_majors": {variant: variant in test_cuda_variants for variant in cuda_variants},
            "sdist_package_roots": {
                package_root: package_root in sdist_package_roots for package_root in package_roots
            },
        },
        "merge_base": merge_base,
        "baseline": {
            "run_id": baseline_run_id if release_package is None and not force_all else "",
            "sha": merge_base if release_package is None and not force_all else "",
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-base", default="")
    parser.add_argument("--baseline-run-id", default="")
    parser.add_argument("--release-tag", default="")
    parser.add_argument(
        "--github-output",
        type=Path,
        help="append normalized CI records to this GitHub output file",
    )
    parser.add_argument(
        "--github-step-summary",
        type=Path,
        help="append a formatted workplan to this GitHub step summary file",
    )
    args = parser.parse_args(argv)
    if args.github_step_summary is not None and args.github_output is None:
        parser.error("--github-step-summary requires --github-output")

    config = load_config()
    reusable_baseline = bool(args.merge_base and args.baseline_run_id and not args.release_tag)
    paths, linked_paths = _changed_paths(args.merge_base) if reusable_baseline else ([], set())
    try:
        plan = compute_workplan(
            bindings_config=config,
            paths=paths,
            linked_paths=linked_paths,
            merge_base=args.merge_base,
            baseline_run_id=args.baseline_run_id,
            release_tag=args.release_tag,
        )
    except BindingsConfigError as error:
        parser.error(str(error))
    workplan = json.dumps(plan, separators=(",", ":"), sort_keys=True)
    if args.github_output is None:
        print(workplan)
        return

    records = {
        "bindings-config": config.to_json(),
        "workplan": workplan,
    }
    with args.github_output.open("a", encoding="utf-8") as output:
        for name, value in records.items():
            output.write(f"{name}={value}\n")
    if args.github_step_summary is not None:
        formatted = json.dumps(plan, indent=2, sort_keys=True)
        with args.github_step_summary.open("a", encoding="utf-8") as summary:
            summary.write(f"\n### CI workplan\n```json\n{formatted}\n```\n")


if __name__ == "__main__":
    main()
