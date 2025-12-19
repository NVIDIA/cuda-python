# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from packaging.version import Version
from setuptools import setup
from setuptools_scm import get_version

version = get_version(
    root="..",
    relative_to=__file__,
    # We deliberately do not want to include the version suffixes (a/b/rc) in cuda-python versioning
    tag_regex="^(?P<version>v\\d+\\.\\d+\\.\\d+)",
    git_describe_command=["git", "describe", "--dirty", "--tags", "--long", "--match", "v*[0-9]*"],
)


base_version = Version(version).base_version


if base_version == version:
    # Tagged release
    matcher = "~="
else:
    # Pre-release version
    matcher = "=="


setup(
    version=version,
    install_requires=[
        f"cuda-bindings{matcher}{version}",
    ],
    extras_require={
        "all": [f"cuda-bindings[all]{matcher}{version}"],
    },
)
