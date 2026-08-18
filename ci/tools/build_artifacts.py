#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build cacheable Python artifacts declared by the Moon project graph."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from ci.tools.artifacts import artifact_wheel, find_one, output_path, project_path, reset_output, run

PACKAGE_PROJECTS = ("pathfinder", "bindings", "core", "metapackage")


def _cuda_major(lane: str) -> str:
    variable = "BUILD_CUDA_MAJOR" if lane == "current" else "BUILD_PREV_CUDA_MAJOR"
    value = os.environ.get(variable, "")
    if value:
        return value
    if lane == "current":
        version = os.environ.get("BUILD_CUDA_VER") or os.environ.get("CUDA_VER", "")
        if version:
            return version.split(".", maxsplit=1)[0]
    raise RuntimeError(f"{variable} is required for the {lane} CUDA lane")


def _constraint_uri(path: Path, *, in_linux_container: bool) -> str:
    resolved = path.resolve()
    return f"file:///host{resolved.as_posix()}" if in_linux_container else resolved.as_uri()


def _constraint_environment(
    project: str,
    lane: str,
    *,
    cibuildwheel: bool,
    from_sdist: bool = False,
) -> dict[str, str]:
    if project not in {"bindings", "core"}:
        return os.environ.copy()

    constraints = output_path(project, f"constraints-{lane}")
    reset_output(constraints)
    constraint_file = constraints / "build.txt"
    linux_container = cibuildwheel and os.name != "nt"
    pathfinder_wheel = (
        find_one(output_path("pathfinder", "sdist"), "*.whl", "cuda.pathfinder sdist wheel")
        if from_sdist
        else artifact_wheel("pathfinder", "pure")
    )
    requirements = [("cuda-pathfinder", pathfinder_wheel)]
    if project == "core":
        bindings_wheel = (
            find_one(output_path("bindings", "sdist"), "*.whl", "cuda.bindings sdist wheel")
            if from_sdist
            else artifact_wheel("bindings", lane)
        )
        requirements.append(("cuda-bindings", bindings_wheel))
    constraint_file.write_text(
        "".join(
            f"{distribution} @ {_constraint_uri(wheel, in_linux_container=linux_container)}\n"
            for distribution, wheel in requirements
        ),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    host_constraint = str(constraint_file.resolve())
    environment["PIP_BUILD_CONSTRAINT"] = host_constraint
    environment["PIP_CONSTRAINT"] = host_constraint
    if project == "core":
        environment["CUDA_CORE_BUILD_MAJOR"] = _cuda_major(lane)
    if cibuildwheel:
        setting = "CIBW_ENVIRONMENT_WINDOWS" if os.name == "nt" else "CIBW_ENVIRONMENT_LINUX"
        container_constraint = f"/host{constraint_file.resolve().as_posix()}" if linux_container else host_constraint
        additions = [
            f'PIP_BUILD_CONSTRAINT="{container_constraint}"',
            f'PIP_CONSTRAINT="{container_constraint}"',
        ]
        if project == "core":
            additions.append(f"CUDA_CORE_BUILD_MAJOR={_cuda_major(lane)}")
        environment[setting] = " ".join(filter(None, [environment.get(setting, ""), *additions]))
    return environment


def _ensure_owned(output: Path) -> None:
    if os.name == "nt":
        return
    owners = {path.stat().st_uid for path in output.rglob("*")}
    if not owners or owners == {os.getuid()}:
        return
    sudo = shutil.which("sudo")
    if sudo is None:
        raise RuntimeError(f"cibuildwheel output is not owned by this user and sudo was not found: {output}")
    run([sudo, "chown", "-R", f"{os.getuid()}:{os.getgid()}", str(output)])


def _pure_wheel(project: str) -> None:
    if project not in {"pathfinder", "metapackage"}:
        raise ValueError("pure-wheel only supports pathfinder and metapackage")
    output = output_path(project, "wheel-pure")
    reset_output(output)
    run(
        [sys.executable, "-m", "pip", "wheel", "--verbose", "--no-deps", "--wheel-dir", str(output), "."],
        cwd=project_path(project),
    )
    find_one(output, "*.whl", f"{project} wheel")


def _native_wheel(project: str, lane: str) -> None:
    if project not in {"bindings", "core"}:
        raise ValueError("native-wheel only supports bindings and core")
    if project == "bindings" and lane != "current":
        raise ValueError("cuda.bindings is only built in the current lane")
    output = output_path(project, f"wheel-{lane}")
    reset_output(output)
    environment = _constraint_environment(project, lane, cibuildwheel=True)
    run(
        [sys.executable, "-m", "cibuildwheel", "--output-dir", str(output), str(project_path(project))],
        env=environment,
    )
    _ensure_owned(output)
    wheel = find_one(output, "*.whl", f"{project} {lane} wheel")
    if project == "core":
        wheel.rename(wheel.with_name(f"{wheel.stem}.cu{_cuda_major(lane)}.whl"))


def _sdist(project: str) -> None:
    project_root = project_path(project)
    output = output_path(project, "sdist")
    reset_output(output)
    environment = (
        _constraint_environment(project, "current", cibuildwheel=False, from_sdist=True)
        if project in {"bindings", "core"}
        else os.environ.copy()
    )
    run([sys.executable, "-m", "build", "--sdist", "--outdir", str(output), str(project_root)], env=environment)
    archive = find_one(output, "*.tar.gz", f"{project} source distribution")
    run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(output), str(archive)],
        env=environment,
    )
    find_one(output, "*.whl", f"{project} wheel from source distribution")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pure = subparsers.add_parser("pure-wheel")
    pure.add_argument("project", choices=("pathfinder", "metapackage"))
    native = subparsers.add_parser("native-wheel")
    native.add_argument("project", choices=("bindings", "core"))
    native.add_argument("--lane", choices=("current", "previous"), required=True)
    sdist = subparsers.add_parser("sdist")
    sdist.add_argument("project", choices=PACKAGE_PROJECTS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "pure-wheel":
        _pure_wheel(args.project)
    elif args.command == "native-wheel":
        _native_wheel(args.project, args.lane)
    else:
        _sdist(args.project)


if __name__ == "__main__":
    main()
