#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Print a deterministic build-environment fingerprint for a Moon task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sysconfig
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCM_MATCH = {
    "pathfinder": "cuda-pathfinder-v*[0-9]*",
    "bindings": "v*[0-9]*",
    "core": "cuda-core-v*[0-9]*",
    "metapackage": "v*[0-9]*",
}
SCM_DISTRIBUTION = {
    "pathfinder": "CUDA_PATHFINDER",
    "bindings": "CUDA_BINDINGS",
    "core": "CUDA_CORE",
    "metapackage": "CUDA_PYTHON",
}
SCM_GLOBAL_VARIABLES = (
    "SETUPTOOLS_SCM_PRETEND_METADATA",
    "SETUPTOOLS_SCM_PRETEND_VERSION",
    "SOURCE_DATE_EPOCH",
    "VCS_VERSIONING_PRETEND_METADATA",
    "VCS_VERSIONING_PRETEND_VERSION",
)
SCM_DISTRIBUTION_VARIABLES = (
    "SETUPTOOLS_SCM_OVERRIDES_FOR_{distribution}",
    "SETUPTOOLS_SCM_PRETEND_METADATA_FOR_{distribution}",
    "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{distribution}",
    "VCS_VERSIONING_PRETEND_METADATA_FOR_{distribution}",
    "VCS_VERSIONING_PRETEND_VERSION_FOR_{distribution}",
)
LANE_VARIABLES = (
    "BUILD_CUDA_MAJOR",
    "BUILD_CUDA_VER",
    "BUILD_PREV_CUDA_MAJOR",
    "CIBW_ARCHS",
    "CIBW_BUILD",
    "CIBW_ENABLE",
    "CUDA_CORE_BUILD_MAJOR",
    "CUDA_PATH",
    "CUDA_PYTHON_LANE",
    "CUDA_VER",
    "HOST_PLATFORM",
    "PY_VER",
)
PYTHON_TOOLS = ("build", "cibuildwheel", "packaging", "pip", "setuptools", "setuptools-scm", "wheel")


def _git_describe(pattern: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed git command with a package-defined tag pattern.
        ["git", "describe", "--dirty", "--tags", "--long", "--match", pattern],  # noqa: S607
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<not-installed>"


def _scm_environment(project: str) -> dict[str, str]:
    distribution = SCM_DISTRIBUTION[project]
    distribution_variables = tuple(name.format(distribution=distribution) for name in SCM_DISTRIBUTION_VARIABLES)
    variables = (*SCM_GLOBAL_VARIABLES, *distribution_variables)
    return {name: os.environ.get(name, "") for name in variables}


def _scm_identity(project: str) -> dict[str, object]:
    environment = _scm_environment(project)
    pretend_variables = ["SETUPTOOLS_SCM_PRETEND_VERSION"]
    # cuda_python/setup.py calls get_version without a distribution name, so
    # setuptools-scm cannot apply its distribution-specific override there.
    if project != "metapackage":
        pretend_variables.append(f"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{SCM_DISTRIBUTION[project]}")
    describe = (
        "<pretend-version>"
        if any(environment[name] for name in pretend_variables)
        else _git_describe(SCM_MATCH[project])
    )
    return {"describe": describe, "environment": environment}


def fingerprint(project: str, lane: str) -> str:
    payload: dict[str, object] = {
        "lane": lane,
        "project": project,
        "python": {
            "implementation": platform.python_implementation(),
            "soabi": sysconfig.get_config_var("SOABI") or "",
            "version": platform.python_version(),
        },
        "python_tools": {name: _distribution_version(name) for name in PYTHON_TOOLS},
        "scm": _scm_identity(project),
    }
    if lane != "portable":
        payload.update(
            {
                "environment": {name: os.environ.get(name, "") for name in LANE_VARIABLES},
                "platform": {"machine": platform.machine(), "system": platform.system()},
            }
        )
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=tuple(SCM_MATCH))
    parser.add_argument("lane", choices=("portable", "native", "previous", "sdist", "test-assets"))
    args = parser.parse_args()
    print(fingerprint(args.project, args.lane))


if __name__ == "__main__":
    main()
