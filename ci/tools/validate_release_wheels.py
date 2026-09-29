#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate downloaded release wheels against the requested release tag."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml
from packaging.tags import Tag
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from .check_release_notes import COMPONENTS, parse_version_from_tag

_BINARY_DISTRIBUTIONS = frozenset({"cuda_bindings", "cuda_core"})
_BUILD_WORKFLOW = Path(".github/workflows/build-wheel.yml")
_CI_WORKFLOW = Path(".github/workflows/ci.yml")
_RELEASE_EXCLUDED_PYTHON_PREFIXES = ("3.15",)


@dataclass(frozen=True, order=True)
class WheelTarget:
    interpreter: str
    abi: str
    platform: str

    def __str__(self) -> str:
        return f"{self.interpreter}-{self.abi}-{self.platform}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that wheel versions match the release tag. "
            "This rejects dev/local wheel versions for release uploads."
        )
    )
    parser.add_argument("git_tag", help="Release git tag (for example: v13.0.0)")
    parser.add_argument("component", choices=[*sorted(COMPONENTS), "all"])
    parser.add_argument("wheel_dir", help="Directory containing wheel files")
    parser.add_argument(
        "--bindings-package",
        default="",
        help="normalized CUDA bindings package JSON from the release resolver",
    )
    parser.add_argument(
        "--release-source-root",
        type=Path,
        help="tagged source tree whose CI build matrix produced the wheels",
    )
    return parser.parse_args()


def version_from_tag(
    tag: str,
    component: str,
    bindings_package: Mapping[str, object] | None = None,
) -> str:
    versions = {
        version
        for tag_component in (COMPONENTS if component == "all" else (component,))
        if (
            version := parse_version_from_tag(
                tag,
                tag_component,
                bindings_package if tag_component == "cuda-bindings" else None,
            )
        )
        is not None
    }
    if len(versions) == 1:
        return versions.pop()
    raise ValueError(
        "Unsupported git tag format "
        f"{tag!r} for component {component!r}; expected vX.Y.Z[.postN], "
        "cuda-core-vX.Y.Z[.postN], or cuda-pathfinder-vX.Y.Z[.postN]."
    )


def _load_yaml_mapping(path: Path) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read release build configuration {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"release build configuration {path} must contain a mapping")
    return raw


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} must be a non-empty list of strings")
    return value


def _release_python_versions(release_source_root: Path) -> list[str]:
    path = release_source_root / _BUILD_WORKFLOW
    raw = _load_yaml_mapping(path)
    try:
        value = raw["jobs"]["build"]["strategy"]["matrix"]["python-version"]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError(f"could not find the Python build matrix in {path}") from error
    return [
        version
        for version in _string_list(value, f"{path} Python build matrix")
        if not version.startswith(_RELEASE_EXCLUDED_PYTHON_PREFIXES)
    ]


def _release_platforms(release_source_root: Path) -> list[str]:
    path = release_source_root / _CI_WORKFLOW
    raw = _load_yaml_mapping(path)
    try:
        jobs = raw["jobs"]
    except KeyError as error:
        raise ValueError(f"could not find jobs in {path}") from error
    if not isinstance(jobs, dict):
        raise ValueError(f"{path} jobs must be a mapping")

    platforms: list[str] = []
    for job_name, value in jobs.items():
        if not isinstance(value, dict) or value.get("uses") != "./.github/workflows/build-wheel.yml":
            continue
        inputs = value.get("with", {})
        if not isinstance(inputs, dict):
            raise ValueError(f"{path} job {job_name!r} inputs must be a mapping")
        platform = inputs.get("host-platform")
        if isinstance(platform, str) and "${{" not in platform:
            platforms.append(platform)
            continue
        try:
            matrix = value["strategy"]["matrix"]["host-platform"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"could not resolve the host-platform matrix for {path} job {job_name!r}") from error
        platforms.extend(_string_list(matrix, f"{path} job {job_name!r} host-platform matrix"))

    if not platforms:
        raise ValueError(f"could not find any build-wheel host platforms in {path}")
    if len(platforms) != len(set(platforms)):
        raise ValueError(f"{path} contains duplicate build-wheel host platforms")
    return platforms


def _python_target(version: str) -> tuple[str, str]:
    free_threaded = version.endswith("t")
    digits = version.removesuffix("t").replace(".", "")
    if not digits.isdecimal():
        raise ValueError(f"unsupported Python build-matrix version: {version!r}")
    interpreter = f"cp{digits}"
    return interpreter, f"{interpreter}t" if free_threaded else interpreter


def _platform_target(platform: str) -> str:
    targets = {
        "linux-64": "manylinux-x86_64",
        "linux-aarch64": "manylinux-aarch64",
        "win-64": "win_amd64",
        "win-arm64": "win_arm64",
    }
    try:
        return targets[platform]
    except KeyError as error:
        raise ValueError(f"unsupported release build platform: {platform!r}") from error


def expected_binary_targets(
    release_source_root: Path,
    bindings_package: Mapping[str, object] | None,
) -> set[WheelTarget]:
    python_versions = _release_python_versions(release_source_root)
    platforms = _release_platforms(release_source_root)

    if "win-arm64" in platforms:
        toolkit_version = None if bindings_package is None else bindings_package.get("toolkit_version")
        if not isinstance(toolkit_version, str):
            raise ValueError("resolved CUDA bindings package must supply toolkit_version for matrix validation")
        try:
            toolkit_major, toolkit_minor = (int(part) for part in toolkit_version.split(".", maxsplit=2)[:2])
        except ValueError as error:
            raise ValueError(f"invalid resolved toolkit_version: {toolkit_version!r}") from error
        if (toolkit_major, toolkit_minor) < (13, 4):
            platforms.remove("win-arm64")

    targets: set[WheelTarget] = set()
    for version in python_versions:
        interpreter, abi = _python_target(version)
        for platform in platforms:
            if platform == "win-arm64" and version == "3.10":
                continue
            targets.add(WheelTarget(interpreter, abi, _platform_target(platform)))
    return targets


def _normalized_platform(platform: str) -> str:
    if platform.startswith("manylinux") and platform.endswith("_x86_64"):
        return "manylinux-x86_64"
    if platform.startswith("manylinux") and platform.endswith("_aarch64"):
        return "manylinux-aarch64"
    return platform


def _wheel_target(tags: frozenset[Tag], wheel_name: str) -> WheelTarget:
    targets = {WheelTarget(tag.interpreter, tag.abi, _normalized_platform(tag.platform)) for tag in tags}
    if len(targets) != 1:
        raise ValueError(f"{wheel_name}: expected one wheel target, found {len(targets)}")
    return targets.pop()


def main() -> int:
    args = parse_args()
    try:
        bindings_package = json.loads(args.bindings_package) if args.bindings_package else None
        if bindings_package is not None and not isinstance(bindings_package, dict):
            raise ValueError("resolved CUDA bindings package must be a JSON object")
        expected_version = version_from_tag(args.git_tag, args.component, bindings_package)
        expected_targets = {
            distribution: (
                expected_binary_targets(args.release_source_root, bindings_package)
                if distribution in _BINARY_DISTRIBUTIONS and args.release_source_root is not None
                else {WheelTarget("py3", "none", "any")}
            )
            for distribution in (
                {package for package, _ in COMPONENTS.values()}
                if args.component == "all"
                else {COMPONENTS[args.component][0]}
            )
        }
        binary_distributions = set(expected_targets) & _BINARY_DISTRIBUTIONS
        if binary_distributions and args.release_source_root is None:
            raise ValueError("--release-source-root is required to validate binary wheel matrices")
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    expected_distributions = set(expected_targets)
    wheel_dir = Path(args.wheel_dir)

    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        print(f"Error: No wheel files found in {wheel_dir}", file=sys.stderr)
        return 1

    seen_targets: dict[str, set[WheelTarget]] = {distribution: set() for distribution in expected_distributions}
    errors: list[str] = []

    for wheel in wheels:
        try:
            parsed_distribution, version, build, tags = parse_wheel_filename(wheel.name)
            distribution = str(parsed_distribution).replace("-", "_")
            target = _wheel_target(tags, wheel.name)
        except (InvalidWheelFilename, ValueError) as exc:
            errors.append(f"Invalid wheel filename {wheel.name!r}: {exc}")
            continue

        if distribution not in expected_distributions:
            errors.append(
                f"{wheel.name}: unexpected distribution {distribution!r} for component "
                f"{args.component!r}; expected one of: " + ", ".join(sorted(expected_distributions))
            )
            continue

        if build:
            errors.append(f"{wheel.name}: wheel build tags are not expected for a release")

        if target in seen_targets[distribution]:
            errors.append(f"{wheel.name}: duplicate wheel target {target} for {distribution}")
        seen_targets[distribution].add(target)

        if version.is_devrelease or version.local is not None:
            errors.append(
                f"{wheel.name}: wheel version {str(version)!r} contains dev/local markers "
                "(.dev or +), which is not allowed for release uploads."
            )

        if str(version) != expected_version:
            errors.append(
                f"{wheel.name}: wheel version {str(version)!r} does not match expected "
                f"release version {expected_version!r} from git tag {args.git_tag!r}."
            )

    missing_distributions = sorted(distribution for distribution, targets in seen_targets.items() if not targets)
    if missing_distributions:
        errors.append("Missing expected component wheels in download set: " + ", ".join(missing_distributions))

    for distribution, expected in sorted(expected_targets.items()):
        missing = sorted(expected - seen_targets[distribution])
        unexpected = sorted(seen_targets[distribution] - expected)
        if missing:
            errors.append(f"Missing expected {distribution} wheel targets: " + ", ".join(map(str, missing)))
        if unexpected:
            errors.append(f"Unexpected {distribution} wheel targets: " + ", ".join(map(str, unexpected)))

    if errors:
        print("Wheel validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "Validated release wheels for component "
        f"{args.component} at version {expected_version} from tag {args.git_tag}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
