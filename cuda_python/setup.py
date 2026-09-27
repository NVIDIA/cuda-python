# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

from packaging.version import Version
from setuptools import setup
from setuptools_scm import get_version

# The standalone metapackage sdist has neither ci/versions.yml nor the
# sibling bindings pyprojects. CI tests keep these selectors in sync with them.
SCM_TAG_REGEX_BY_MAJOR = {
    "12": (
        r"^(?P<version>v12\.9\.(?:0|[1-9][0-9]*)"
        r"(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?"
        r"(?:\.post(?:0|[1-9][0-9]*))?"
        r"(?:\.dev(?:0|[1-9][0-9]*))?)$"
    ),
    "13": (
        r"^(?P<version>v13\.4\.(?:0|[1-9][0-9]*)"
        r"(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?"
        r"(?:\.post(?:0|[1-9][0-9]*))?"
        r"(?:\.dev(?:0|[1-9][0-9]*))?)$"
    ),
}
SCM_DESCRIBE_MATCH_BY_MAJOR = {
    "12": "v12.9.[1-9]*",
    "13": "v13.4.*",
}
MAINTENANCE_FALLBACK_VERSION = "12.9.10.dev0"

build_major = os.environ.get("CUDA_PYTHON_BUILD_MAJOR")
if build_major not in {"12", "13"}:
    raise ValueError(f"CUDA_PYTHON_BUILD_MAJOR must be 12 or 13, got {build_major!r}")

version_options = {
    "root": "..",
    "relative_to": __file__,
    "dist_name": "cuda-python",
    # Keep metapackage tag selection identical to its bindings source line.
    "tag_regex": SCM_TAG_REGEX_BY_MAJOR[build_major],
    "git_describe_command": [
        "git",
        "describe",
        "--dirty",
        "--tags",
        "--long",
        "--match",
        SCM_DESCRIBE_MATCH_BY_MAJOR[build_major],
    ],
}
if build_major == "12":
    # Main predates the active 12.9 tags. This fallback is used until the first
    # post-migration v12.9 tag is reachable from main.
    version_options["fallback_version"] = MAINTENANCE_FALLBACK_VERSION

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
