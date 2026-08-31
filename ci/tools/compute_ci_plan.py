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

if __package__:
    from .bindings_config import load_config
else:
    from bindings_config import load_config

if TYPE_CHECKING:
    if __package__:
        from .bindings_config import BindingsConfig
    else:
        from bindings_config import BindingsConfig

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


def compute_workplan(
    paths: list[str],
    *,
    merge_base: str,
    baseline_run_id: str,
    linked_paths: set[str] | None = None,
    release_tag: str = "",
    bindings_config: BindingsConfig | None = None,
) -> dict[str, object]:
    """Return the final CI decisions for the supplied changed paths."""
    config = bindings_config or load_config()
    lines = config.public_lines
    line_by_id = {line.line_id: line for line in lines}
    line_ids = tuple(line_by_id)
    cuda_variants = tuple(dict.fromkeys(line.cuda_variant for line in lines))
    cuda_major_by_variant = {line.cuda_variant: line.cuda_major for line in lines}

    bindings_targets = frozenset(("bindings", line_id) for line_id in line_ids)
    core_targets = frozenset(("core", variant) for variant in cuda_variants)
    python_targets = frozenset(("python", line_id) for line_id in line_ids)
    all_targets = frozenset({("pathfinder", None), *bindings_targets, *core_targets, *python_targets})

    package_targets = {
        PurePosixPath(source_dir): ("bindings", line_id)
        for line_id, source_dir in ((line.line_id, line.source_dir) for line in lines)
    }
    package_targets.update((PurePosixPath(source_dir), target) for source_dir, target in PACKAGE_TARGETS.items())

    # Source changes have different build and test consumers. In particular,
    # cuda-python source needs same-version bindings wheels, while a core-only
    # change can reuse baseline cuda-python wheels.
    def source_impact(
        target: tuple[str, str | None],
    ) -> tuple[frozenset[tuple[str, str | None]], frozenset[tuple[str, str | None]], frozenset[str]]:
        module, selector = target
        if module == "pathfinder":
            return all_targets, all_targets, frozenset(line_ids)
        if module == "bindings":
            assert selector is not None
            line = line_by_id[selector]
            return (
                frozenset({("bindings", selector), ("python", selector), *core_targets}),
                frozenset({("bindings", selector), ("core", line.cuda_variant), ("python", selector)}),
                frozenset({selector}),
            )
        if module == "core":
            return (
                core_targets,
                frozenset({*core_targets, *python_targets}),
                frozenset(line_ids),
            )
        if module == "python":
            return (
                frozenset({*bindings_targets, *python_targets}),
                python_targets,
                frozenset(line_ids),
            )
        raise AssertionError(f"unhandled source target: {target!r}")

    linked_paths = linked_paths or set()
    source_changes: set[tuple[str, str | None]] = set()
    test_changes: set[tuple[str, str | None]] = set()
    test_platforms: set[str] = set()
    matched_line = config.match_tag(release_tag) if release_tag else None
    release_line = matched_line if matched_line is not None and matched_line.line_id in line_by_id else None
    force_all = (bool(release_tag) and release_line is None) or (
        release_line is None and (not merge_base or not baseline_run_id)
    )

    if release_line is None and not force_all:
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

    if release_line is not None:
        builds = {("bindings", release_line.line_id), ("python", release_line.line_id)}
        tests = set(builds)
        test_platforms = set(PLATFORMS)
        sdist_lines = {release_line.line_id}
    elif force_all:
        builds = set(all_targets)
        tests = set(all_targets)
        test_platforms = set(PLATFORMS)
        sdist_lines = set(line_ids)
    else:
        builds: set[tuple[str, str | None]] = set()
        tests = set(all_targets) if test_platforms else set(test_changes)
        sdist_lines: set[str] = set()
        for target in source_changes:
            build_impact, test_impact, sdist_impact = source_impact(target)
            builds.update(build_impact)
            tests.update(test_impact)
            sdist_lines.update(sdist_impact)
        if source_changes or test_changes:
            test_platforms.update(PLATFORMS)

    modules: dict[str, dict[str, object]] = {
        "pathfinder": {
            "needs_build": ("pathfinder", None) in builds,
            "needs_test": ("pathfinder", None) in tests,
        }
    }
    for module in ("bindings", "python"):
        line_decisions = {
            line.line_id: {
                **config.line_to_dict(line),
                "needs_build": (module, line.line_id) in builds,
                "needs_test": (module, line.line_id) in tests,
            }
            for line in lines
        }
        # Transitional compatibility for workflows that still address a CUDA
        # major directly. OR aggregation prevents same-major lines from
        # overwriting one another while consumers migrate to `lines`.
        variants = {
            variant: {
                "needs_build": any(
                    decision["needs_build"]
                    for line_id, decision in line_decisions.items()
                    if line_by_id[line_id].cuda_variant == variant
                ),
                "needs_test": any(
                    decision["needs_test"]
                    for line_id, decision in line_decisions.items()
                    if line_by_id[line_id].cuda_variant == variant
                ),
            }
            for variant in cuda_variants
        }
        modules[module] = {
            "needs_build": any(decision["needs_build"] for decision in line_decisions.values()),
            "needs_test": any(decision["needs_test"] for decision in line_decisions.values()),
            "lines": line_decisions,
            "variants": variants,
        }

    cuda_major_decisions = {
        variant: {
            "cuda_major": cuda_major_by_variant[variant],
            "cuda_variant": variant,
            "needs_build": ("core", variant) in builds,
            "needs_test": ("core", variant) in tests,
        }
        for variant in cuda_variants
    }
    modules["core"] = {
        "needs_build": any(decision["needs_build"] for decision in cuda_major_decisions.values()),
        "needs_test": any(decision["needs_test"] for decision in cuda_major_decisions.values()),
        "cuda_majors": cuda_major_decisions,
        # Transitional compatibility; `cuda_majors` is canonical.
        "variants": {
            variant: {
                "needs_build": decision["needs_build"],
                "needs_test": decision["needs_test"],
            }
            for variant, decision in cuda_major_decisions.items()
        },
    }

    test_cuda_variants = {
        variant
        for variant in cuda_variants
        if modules["pathfinder"]["needs_test"]
        or modules["bindings"]["variants"][variant]["needs_test"]
        or modules["core"]["cuda_majors"][variant]["needs_test"]
        or modules["python"]["variants"][variant]["needs_test"]
    }
    sdist_cuda_variants = {line.cuda_variant for line in lines if line.line_id in sdist_lines}
    return {
        "modules": modules,
        "jobs": {
            # These gates cover both optional artifact builds and wheel tests.
            "platforms": {platform: platform in test_platforms for platform in PLATFORMS},
            "sdist_tests": bool(builds),
            "core_api_checks": force_all or ("core", None) in source_changes,
            "test_cuda_majors": {variant: variant in test_cuda_variants for variant in cuda_variants},
            "sdist_lines": {line_id: line_id in sdist_lines for line_id in line_ids},
            # Transitional compatibility; `sdist_lines` is canonical.
            "sdist_cuda_majors": {variant: variant in sdist_cuda_variants for variant in cuda_variants},
        },
        "merge_base": merge_base,
        "baseline": {
            "run_id": baseline_run_id if release_line is None and not force_all else "",
            "sha": merge_base if release_line is None and not force_all else "",
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
