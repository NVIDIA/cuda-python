#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared path and artifact helpers for Moon CI tasks."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PATHS = {
    "root": Path("."),
    "pathfinder": Path("cuda_pathfinder"),
    "bindings": Path("cuda_bindings"),
    "core": Path("cuda_core"),
    "metapackage": Path("cuda_python"),
}


def run(command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
    print(f"+ {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)  # noqa: S603


def project_path(project: str) -> Path:
    try:
        relative = PROJECT_PATHS[project]
    except KeyError as error:
        raise ValueError(f"unknown project: {project}") from error
    return REPO_ROOT / relative


def output_path(project: str, directory: str) -> Path:
    repo_root = Path(os.path.abspath(REPO_ROOT))
    project_root = Path(os.path.abspath(project_path(project)))
    if project_root != repo_root and repo_root not in project_root.parents:
        raise ValueError(f"project must be within {repo_root}: {project_root}")
    output_root = project_root / ".moon-out"
    output = Path(os.path.abspath(output_root / directory))
    if output != output_root and output_root not in output.parents:
        raise ValueError(f"output must be within {output_root}: {output}")
    current = output
    while current != repo_root:
        if current.is_symlink():
            raise ValueError(f"output path must not traverse a symlink: {current}")
        current = current.parent
    return output


def reset_output(output: Path) -> None:
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError(f"refusing to replace non-directory output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def find_one(directory: Path, pattern: str, description: str) -> Path:
    selected = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(selected) != 1:
        raise RuntimeError(f"expected one {description} in {directory}, found {len(selected)}")
    return selected[0]


def find_one_in(directories: list[Path], pattern: str, description: str) -> Path:
    for directory in directories:
        selected = sorted(path for path in directory.glob(pattern) if path.is_file())
        if len(selected) == 1:
            return selected[0]
        if len(selected) > 1:
            raise RuntimeError(f"expected one {description} in {directory}, found {len(selected)}")
    searched = ", ".join(str(path) for path in directories)
    raise RuntimeError(f"expected one {description}; searched {searched}")


def artifact_wheel(project: str, lane: str) -> Path:
    if project == "pathfinder":
        directories = [output_path(project, "wheel-pure"), project_path(project)]
    elif project == "bindings":
        environment = os.environ.get("CUDA_BINDINGS_ARTIFACTS_DIR")
        directories = [output_path(project, f"wheel-{lane}")]
        if lane == "previous":
            directories.append(project_path(project) / "dist-prev")
        elif environment:
            directories.append(Path(environment))
        directories.append(project_path(project) / "dist")
    elif project == "core":
        environment = os.environ.get("CUDA_CORE_ARTIFACTS_DIR")
        directories = [output_path(project, f"wheel-{lane}")]
        if environment:
            directories.append(Path(environment))
        directories.append(project_path(project) / "dist")
    elif project == "metapackage":
        directories = [output_path(project, "wheel-pure"), REPO_ROOT, project_path(project)]
    else:
        raise ValueError(f"project does not produce wheel artifacts: {project}")
    return find_one_in(directories, "*.whl", f"{project} {lane} wheel")
