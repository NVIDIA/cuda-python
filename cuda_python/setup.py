# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import tomllib
from pathlib import Path

from packaging.version import Version
from setuptools import setup
from setuptools_scm import get_version

# The selector table ships with standalone sdists, which have neither the
# CI registry nor sibling bindings packages. The metadata hook checks drift.
with Path(__file__).with_name("pyproject.toml").open("rb") as stream:
    scm_selectors = tomllib.load(stream)["tool"]["cuda-python"]["scm-describe-match-by-major"]

# Preserve the current-major default for ordinary source builds. CI selects
# a major explicitly when building either metapackage.
build_major = os.environ.get("CUDA_PYTHON_BUILD_MAJOR", "13")
if build_major not in {"12", "13"}:
    raise ValueError(f"CUDA_PYTHON_BUILD_MAJOR must be 12 or 13, got {build_major!r}")

version_options = {
    "root": "..",
    "relative_to": __file__,
    "dist_name": "cuda-python",
    # Keep metapackage tag selection identical to its bindings source line.
    "git_describe_command": [
        "git",
        "describe",
        "--dirty",
        "--tags",
        "--long",
        "--match",
        scm_selectors[build_major],
    ],
}

version = get_version(**version_options)


base_version = Version(version).base_version


if base_version == version:
    # Tagged release
    matcher = "~="
else:
    # Pre-release version
    matcher = "=="

install_requires = [f"cuda-bindings{matcher}{version}"]
if int(build_major) >= 13:
    install_requires.extend(
        [
            # Bump this with every cuda-core release and release the current-major
            # cuda-python right after it; see .github/RELEASE-core.md.
            "cuda-core~=1.2.0",
            "cuda-pathfinder~=1.1",
        ]
    )


setup(
    version=version,
    install_requires=install_requires,
    extras_require={
        "all": [f"cuda-bindings[all]{matcher}{version}"],
    },
)
