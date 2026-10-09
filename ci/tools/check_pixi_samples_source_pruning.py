# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize one temporary Pixi 0.73.0 source-pruning defect, failing closed."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import tomllib

LOCAL_PACKAGES = {
    "cuda-bindings": "../cuda_bindings",
    "cuda-core": ".",
    "cuda-pathfinder": "../cuda_pathfinder",
}
PLATFORMS = {"p1": "linux-64", "p2": "linux-aarch64", "p3": "win-64"}
SOURCE_REFERENCE = re.compile(
    r"      - conda_source: (cuda-bindings|cuda-core|cuda-pathfinder)\[([0-9a-f]{8})\] @ (\S+)\n"
)


def is_known_samples_source_pruning(original: str, repaired: str) -> bool:
    """Accept only removal of the nine known samples references, with nothing else changed.

    This deliberately understands only the pinned version's exact serialization.
    Source record definitions remain checked byte-for-byte along with the rest of
    the lockfile; missing definitions and the reverse (addition) cycle are rejected.
    """
    lines = original.splitlines(keepends=True)
    if not lines or lines[0] != "version: 7\n" or lines.count("environments:\n") != 1:
        return False
    for alias, platform in PLATFORMS.items():
        if original.count(f"- name: {alias}\n  subdir: {platform}\n") != 1:
            return False

    environment_start = lines.index("environments:\n") + 1
    environment_end = next(
        (index for index in range(environment_start, len(lines)) if not lines[index].startswith(" ")),
        len(lines),
    )
    sample_headers = [index for index in range(environment_start, environment_end) if lines[index] == "  samples:\n"]
    if len(sample_headers) != 1:
        return False
    samples_start = sample_headers[0] + 1
    samples_end = next(
        (index for index in range(samples_start, environment_end) if not lines[index].startswith("    ")),
        environment_end,
    )
    if any(lines[samples_start:samples_end].count(f"      {alias}:\n") != 1 for alias in PLATFORMS):
        return False

    platform = None
    seen: set[tuple[str, str]] = set()
    removed: set[int] = set()
    for index in range(samples_start, samples_end):
        line = lines[index]
        platform_match = re.fullmatch(r"      ([^\s:]+):\n", line)
        if platform_match:
            platform = platform_match[1] if platform_match[1] in PLATFORMS else None
        reference = SOURCE_REFERENCE.fullmatch(line)
        if reference is None:
            continue
        name, identifier, path = reference.groups()
        if platform is None or path != LOCAL_PACKAGES[name] or (platform, name) in seen:
            return False
        definition = f"- conda_source: {name}[{identifier}] @ {path}\n"
        if lines[environment_end:].count(definition) != 1:
            return False
        seen.add((platform, name))
        removed.add(index)

    expected = {(platform, name) for platform in PLATFORMS for name in LOCAL_PACKAGES}
    return seen == expected and "".join(line for index, line in enumerate(lines) if index not in removed) == repaired


def _manifest_matches(manifest: dict) -> bool:
    features = manifest.get("feature", {})
    if not isinstance(features, dict):
        return False
    samples = features.get("samples", {})
    local_deps = features.get("local-deps", {})
    environments = manifest.get("environments", {})
    if not all(isinstance(table, dict) for table in (samples, local_deps, environments)):
        return False
    options = samples.get("pypi-options", {})
    sample_dependencies = samples.get("dependencies", {})
    local_dependencies = local_deps.get("dependencies", {})
    if not all(isinstance(table, dict) for table in (options, sample_dependencies, local_dependencies)):
        return False
    overrides = options.get("dependency-overrides", {})
    if overrides != {name: {"version": "*", "env-markers": "python_version < '0'"} for name in LOCAL_PACKAGES}:
        return False
    if sample_dependencies.get("cuda-core") != {"path": "."}:
        return False
    if any(
        local_dependencies.get(name) != {"path": path} for name, path in LOCAL_PACKAGES.items() if name != "cuda-core"
    ):
        return False
    environment = environments.get("samples", {})
    return isinstance(environment, dict) and environment.get("features") == ["cu13", "samples", "local-deps"]


def main(argv: list[str] | None = None) -> int:
    """Exit 0 for the known defect, 1 otherwise, and 2 for an operational error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-blob", required=True)
    parser.add_argument("--lockfile", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--pixi-version", required=True)
    parser.add_argument("--check-status", type=int, required=True)
    args = parser.parse_args(argv)
    if args.pixi_version.removeprefix("v") != "0.73.0" or args.check_status != 1:
        return 1
    if re.fullmatch(r"[0-9a-f]{40}", args.original_blob) is None:
        print("Source-pruning check: original blob must be a full Git SHA-1.", file=sys.stderr)
        return 2
    try:
        original = subprocess.run(  # noqa: S603 - validated SHA, argv passed without a shell.
            ["git", "cat-file", "blob", args.original_blob],  # noqa: S607
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8")
        repaired = args.lockfile.read_bytes().decode("utf-8")
        manifest = tomllib.loads(args.manifest_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, subprocess.CalledProcessError) as error:
        print(f"Source-pruning check could not read its inputs: {error}", file=sys.stderr)
        return 2
    if not _manifest_matches(manifest):
        return 1
    return 0 if is_known_samples_source_pruning(original, repaired) else 1


if __name__ == "__main__":
    sys.exit(main())
