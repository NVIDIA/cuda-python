# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write pip constraints that select exact, locally built wheel artifacts."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from email import policy
from email.errors import MessageError
from email.parser import BytesParser
from pathlib import Path

_CANONICALIZE_PROJECT_RE = re.compile(r"[-_.]+")
_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_RELEASE_MAJOR_RE = re.compile(r"^([0-9]+)(?:\.|$)")


class WheelConstraintError(RuntimeError):
    """Raised when an exact local wheel constraint cannot be produced."""


@dataclass(frozen=True)
class WheelRequirement:
    """A distribution that must resolve to one wheel in a local directory."""

    project: str
    directory: Path
    expected_major: str | None = None


@dataclass(frozen=True)
class _WheelMetadata:
    path: Path
    name: str
    version: str


def _canonicalize_project(name: str) -> str:
    if _PROJECT_NAME_RE.fullmatch(name) is None:
        raise WheelConstraintError(f"Invalid project name: {name!r}")
    return _CANONICALIZE_PROJECT_RE.sub("-", name).lower()


def _read_wheel_metadata(wheel_path: Path) -> _WheelMetadata:
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            metadata_entries = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_entries) != 1:
                raise WheelConstraintError(
                    f"Wheel {wheel_path} contains {len(metadata_entries)} .dist-info/METADATA entries; expected exactly one"
                )
            metadata_bytes = wheel.read(metadata_entries[0])
        metadata = BytesParser(policy=policy.compat32).parsebytes(metadata_bytes)
    except WheelConstraintError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, MessageError) as exc:
        raise WheelConstraintError(f"Cannot read wheel {wheel_path}: {exc}") from exc

    name = metadata.get("Name", "").strip()
    version = metadata.get("Version", "").strip()
    if not name or not version:
        raise WheelConstraintError(f"Wheel {wheel_path} METADATA must contain non-empty Name and Version fields")
    _canonicalize_project(name)
    try:
        resolved_path = wheel_path.resolve()
    except OSError as exc:
        raise WheelConstraintError(f"Cannot resolve wheel path {wheel_path}: {exc}") from exc
    return _WheelMetadata(path=resolved_path, name=name, version=version)


def _release_major(version: str, wheel_path: Path) -> str:
    match = _RELEASE_MAJOR_RE.match(version)
    if match is None:
        raise WheelConstraintError(
            f"Wheel {wheel_path} has version {version!r}, which does not start with a numeric release major"
        )
    return match.group(1)


def _describe_wheels(wheels: Sequence[_WheelMetadata]) -> str:
    if not wheels:
        return "none"
    return ", ".join(f"{wheel.name}=={wheel.version} ({wheel.path.name})" for wheel in wheels)


def _select_wheel(requirement: WheelRequirement) -> _WheelMetadata:
    project = _canonicalize_project(requirement.project)
    try:
        directory = requirement.directory.resolve()
    except OSError as exc:
        raise WheelConstraintError(
            f"Cannot resolve wheel directory for {project}: {requirement.directory}: {exc}"
        ) from exc
    if not directory.is_dir():
        raise WheelConstraintError(f"Wheel directory for {project} does not exist or is not a directory: {directory}")

    try:
        wheel_paths = sorted(directory.glob("*.whl"))
    except OSError as exc:
        raise WheelConstraintError(f"Cannot inspect wheel directory {directory}: {exc}") from exc
    if not wheel_paths:
        raise WheelConstraintError(f"Wheel directory for {project} contains no .whl files: {directory}")

    inspected = [_read_wheel_metadata(path) for path in wheel_paths]
    matching_project = [wheel for wheel in inspected if _canonicalize_project(wheel.name) == project]

    expected_major = requirement.expected_major
    if expected_major is not None:
        if re.fullmatch(r"[0-9]+", expected_major) is None:
            raise WheelConstraintError(f"Expected major for {project} must contain only digits, got {expected_major!r}")
        matching = [wheel for wheel in matching_project if _release_major(wheel.version, wheel.path) == expected_major]
    else:
        matching = matching_project

    if not matching:
        qualifier = f" with release major {expected_major}" if expected_major is not None else ""
        raise WheelConstraintError(
            f"Found no wheel for {project}{qualifier} in {directory}; inspected: {_describe_wheels(inspected)}"
        )
    if len(matching) > 1:
        qualifier = f" with release major {expected_major}" if expected_major is not None else ""
        raise WheelConstraintError(
            f"Found multiple wheels for {project}{qualifier} in {directory}: {_describe_wheels(matching)}"
        )
    return matching[0]


def _consumer_path(wheel_path: Path, container_mount: Path | None) -> Path:
    if container_mount is None:
        return wheel_path
    if not container_mount.is_absolute():
        raise WheelConstraintError(f"Container mount must be an absolute path: {container_mount}")
    if os.name != "posix":
        raise WheelConstraintError("--container-mount is only supported on POSIX hosts")
    return container_mount / wheel_path.relative_to(wheel_path.anchor)


def _remove_stale_output(output_path: Path) -> Path:
    try:
        resolved_output = output_path.resolve()
        resolved_output.unlink(missing_ok=True)
    except OSError as exc:
        raise WheelConstraintError(f"Cannot remove stale constraints file {output_path}: {exc}") from exc
    return resolved_output


def write_constraints(
    output_path: Path,
    requirements: Sequence[WheelRequirement],
    *,
    container_mount: Path | None = None,
) -> None:
    """Write exact direct-reference constraints for the requested wheels."""
    output_path = _remove_stale_output(output_path)

    if not requirements:
        raise WheelConstraintError("At least one wheel requirement is required")

    seen_projects: set[str] = set()
    selected: list[tuple[str, _WheelMetadata, str]] = []
    for requirement in requirements:
        project = _canonicalize_project(requirement.project)
        if not project:
            raise WheelConstraintError(f"Project name must not be empty: {requirement.project!r}")
        if project in seen_projects:
            raise WheelConstraintError(f"Project {project} was requested more than once")
        seen_projects.add(project)

        wheel = _select_wheel(requirement)
        consumer_uri = _consumer_path(wheel.path, container_mount).as_uri()
        selected.append((project, wheel, consumer_uri))

    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for project, _, consumer_uri in selected:
                output.write(f"{project} @ {consumer_uri}\n")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, output_path)
    except OSError as exc:
        cleanup_details = ""
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_details = f"; temporary-file cleanup also failed: {cleanup_exc}"
        raise WheelConstraintError(f"Cannot write constraints file {output_path}: {exc}{cleanup_details}") from exc

    for project, wheel, consumer_uri in selected:
        print(f"Selected {project}=={wheel.version} from {wheel.path} as {consumer_uri}")
    print(f"Wrote exact wheel constraints to {output_path}")


def _parse_requirements(
    wheels: Sequence[Sequence[str]], expected_majors: Sequence[Sequence[str]]
) -> list[WheelRequirement]:
    majors_by_project: dict[str, str] = {}
    for project, major in expected_majors:
        canonical_project = _canonicalize_project(project)
        if canonical_project in majors_by_project:
            raise WheelConstraintError(f"Expected major for {canonical_project} was specified more than once")
        majors_by_project[canonical_project] = major

    requested_projects = {_canonicalize_project(project) for project, _ in wheels}
    unused_majors = sorted(majors_by_project.keys() - requested_projects)
    if unused_majors:
        raise WheelConstraintError(
            f"Expected major was specified for an unrequested project: {', '.join(unused_majors)}"
        )

    return [
        WheelRequirement(
            project=project,
            directory=Path(directory),
            expected_major=majors_by_project.get(_canonicalize_project(project)),
        )
        for project, directory in wheels
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Constraints file to write")
    parser.add_argument(
        "--wheel",
        action="append",
        nargs=2,
        required=True,
        metavar=("PROJECT", "DIRECTORY"),
        help="Project and directory containing its locally built wheel",
    )
    parser.add_argument(
        "--expected-major",
        action="append",
        nargs=2,
        default=[],
        metavar=("PROJECT", "MAJOR"),
        help="Require the selected project's wheel version to have this release major",
    )
    parser.add_argument(
        "--container-mount",
        type=Path,
        help="Map absolute host wheel paths below this container-visible mount",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write a constraints file from command-line arguments."""
    args = _parser().parse_args(argv)
    try:
        _remove_stale_output(args.output)
        requirements = _parse_requirements(args.wheel, args.expected_major)
        write_constraints(args.output, requirements, container_mount=args.container_mount)
    except WheelConstraintError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
