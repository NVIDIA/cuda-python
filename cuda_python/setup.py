# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

from packaging.version import Version
from setuptools import setup
from setuptools_scm import get_version

build_major = os.environ.get("CUDA_PYTHON_BUILD_MAJOR", "13")
if build_major not in {"12", "13"}:
    raise ValueError(f"CUDA_PYTHON_BUILD_MAJOR must be 12 or 13, got {build_major!r}")

version_options = {
    "root": "..",
    "relative_to": __file__,
    "dist_name": "cuda-python",
    # Preserve the established prerelease policy of each line and post-release
    # suffixes for both: CUDA 12.9 strips prereleases, CUDA 13 preserves a/b.
    "tag_regex": (
        r"^(?P<version>v12\.9\.\d+(?:\.post\d+)?)"
        if build_major == "12"
        else r"^(?P<version>v13\.\d+\.\d+(?:[ab]\d+)?(?:\.post\d+)?)"
    ),
    "git_describe_command": [
        "git",
        "describe",
        "--dirty",
        "--tags",
        "--long",
        "--match",
        "v12.9.[1-9]*" if build_major == "12" else "v13.*",
    ],
}
if build_major == "12":
    # Main predates the active 12.9 tags. This fallback is used until the first
    # post-migration v12.9 tag is reachable from main.
    version_options["fallback_version"] = "12.9.8.dev0"

version = get_version(**version_options)


base_version = Version(version).base_version


if base_version == version:
    # Tagged release
    matcher = "~="
else:
    # Pre-release version
    matcher = "=="

install_requires = [f"cuda-bindings{matcher}{version}"]
if build_major == "13":
    install_requires.extend(
        [
            "cuda-core~=1.1.0",
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
