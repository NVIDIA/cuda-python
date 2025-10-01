# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ast

from setuptools import setup

# We want to keep the version in sync with cuda.bindings, but setuptools would not let
# us to refer to any files outside of the project root, so we have to employ our own
# run-time lookup using setup()...
with open("../cuda_bindings/cuda/bindings/_version.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.value

setup(
    version=version,
    install_requires=[
        f"cuda-bindings~={version}",
        "cuda-pathfinder~=1.1",
    ],
    extras_require={
        "all": [f"cuda-bindings[all]~={version}"],
    },
)
