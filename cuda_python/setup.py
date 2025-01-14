# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from setuptools import setup

# We want to keep the version in sync with cuda.bindings, but setuptools would not let
# us to refer to any files outside of the project root, so we have to employ our own
# run-time lookup using setup()...
with open("../cuda_bindings/cuda/bindings/_version.py") as f:
    exec(f.read())
version = __version__  # noqa: F821
del __version__  # noqa: F821

setup(
    version=version,
    install_requires=[
        f"cuda-bindings~={version}",
    ],
    extras_require={
        "all": [f"cuda-bindings[all]~={version}"],
    },
)
