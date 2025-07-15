# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ast
from setuptools import setup

# We want to keep the version in sync with cuda.bindings, but setuptools does not
# provide a nice way to construct the dependencies in pyproject.toml, so we need
# to manually grab the version and do it ourselves.
with open("_version.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.value

setup(
    version=version,
    install_requires=[
        f"cuda-bindings~={version}",
    ],
    extras_require={
        "all": [f"cuda-bindings[all]~={version}"],
    },
)
