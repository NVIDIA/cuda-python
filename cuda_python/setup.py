# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ast

from packaging.version import Version
from setuptools import setup
from setuptools_scm import get_version


def get_version_for_module(module_name: str | None = None) -> str:
    if module_name is None:
        module_name = ""
    else:
        module_name = f"{module_name}-"
    return get_version(
        root="..",
        relative_to=__file__,
        # Preserve a/b pre-release suffixes, but intentionally strip rc suffixes.
        tag_regex=f"^{module_name}(?P<version>v\\d+\\.\\d+\\.\\d+(?:[ab]\\d+)?)",
        git_describe_command=["git", "describe", "--dirty", "--tags", "--long", "--match", f"{module_name}v*[0-9]*"],
        version_scheme="post-release",
    )


install_requires: list[str] = []
extras_require: dict[str, list[str]] = {}

for module in ["cuda-bindings", "cuda-core", "cuda-pathfinder"]:
    if module == "cuda-bindings":
        module_prefix = None
    else:
        module_prefix = module
    version = get_version_for_module(module_prefix)
    base_version: str = Version(version).base_version

    install_requires.append(f"{module}~={base_version}")

    if module == "cuda-bindings":
        extras_require["all"] = [f"{module}[all]~={base_version}"]


setup(
    version=version,
    install_requires=install_requires,
    extras_require=extras_require,
)
