# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write pip constraints that refer to exact wheel files."""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Sequence


class WheelConstraintError(RuntimeError):
    """Raised when an exact wheel cannot be selected safely."""


def normalize_project_name(name: str) -> str:
    """Normalize a project name using the package-name rules from PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_wheel_identity(wheel_path: Path) -> tuple[str, str]:
    """Read the project name and version from a wheel's core metadata."""
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            metadata_paths = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_paths) != 1:
                raise WheelConstraintError(
                    f"expected exactly one .dist-info/METADATA entry in {wheel_path}, found {len(metadata_paths)}"
                )
            metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))
    except zipfile.BadZipFile as exc:
        raise WheelConstraintError(f"invalid wheel archive: {wheel_path}") from exc

    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise WheelConstraintError(f"wheel metadata is missing Name or Version: {wheel_path}")
    return name, version


def find_exact_wheel(directory: Path, project: str) -> tuple[Path, str]:
    """Find exactly one wheel for project in directory."""
    if not directory.is_dir():
        raise WheelConstraintError(f"wheel directory does not exist: {directory}")

    expected_name = normalize_project_name(project)
    inspected: list[tuple[Path, str, str]] = []
    matches: list[tuple[Path, str]] = []
    for wheel_path in sorted(directory.glob("*.whl")):
        name, version = read_wheel_identity(wheel_path)
        inspected.append((wheel_path, name, version))
        if normalize_project_name(name) == expected_name:
            matches.append((wheel_path.resolve(), version))

    if len(matches) != 1:
        details = "\n".join(f"  {path}: {name} {version}" for path, name, version in inspected)
        if not details:
            details = "  <no wheel files>"
        raise WheelConstraintError(
            f"expected exactly one {project} wheel in {directory}, found {len(matches)}:\n{details}"
        )

    return matches[0]


def wheel_uri(wheel_path: Path, host_platform: str) -> str:
    """Return the wheel URI as seen by the cibuildwheel build environment."""
    wheel_path = wheel_path.resolve()
    if host_platform.startswith("linux"):
        if wheel_path.anchor != "/":
            raise WheelConstraintError(f"expected an absolute POSIX wheel path, got {wheel_path}")
        wheel_path = Path("/host", *wheel_path.parts[1:])
    elif not host_platform.startswith("win"):
        raise WheelConstraintError(f"unsupported host platform: {host_platform}")
    return wheel_path.as_uri()


def write_constraints(
    output_path: Path,
    host_platform: str,
    wheels: Sequence[tuple[str, Path]],
) -> None:
    """Resolve wheels and write direct-reference pip constraints."""
    constraints: list[str] = []
    seen_projects: set[str] = set()
    for project, directory in wheels:
        normalized_name = normalize_project_name(project)
        if normalized_name in seen_projects:
            raise WheelConstraintError(f"duplicate project requested: {project}")
        seen_projects.add(normalized_name)

        wheel_path, version = find_exact_wheel(directory, project)
        constraints.append(f"{normalized_name} @ {wheel_uri(wheel_path, host_platform)}")
        print(f"Resolved {normalized_name} {version}: {wheel_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(constraints) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Constraint file to write")
    parser.add_argument("--host-platform", required=True, help="CI host platform, such as linux-64 or win-64")
    parser.add_argument(
        "--wheel",
        action="append",
        nargs=2,
        required=True,
        metavar=("PROJECT", "DIRECTORY"),
        help="Project name and directory containing its wheel; may be repeated",
    )
    args = parser.parse_args(argv)

    try:
        write_constraints(
            args.output,
            args.host_platform,
            [(project, Path(directory)) for project, directory in args.wheel],
        )
    except WheelConstraintError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
