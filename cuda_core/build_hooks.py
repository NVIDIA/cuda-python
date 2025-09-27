# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module implements basic PEP 517 backend support, see e.g.
# - https://peps.python.org/pep-0517/
# - https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
# Specifically, there are 5 APIs required to create a proper build backend, see below.
# For now it's mostly a pass-through to setuptools, except that we need to determine
# some dependencies at build time.
#
# TODO: also implement PEP-660 API hooks

import os
import re
import subprocess  # nosec: B404

from setuptools import build_meta as _build_meta

prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_wheel = _build_meta.build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist


def _get_proper_cuda_bindings_major_version() -> str:
    # for local development (with/without build isolation)
    try:
        import cuda.bindings

        return cuda.bindings.__version__.split(".")[0]
    except ImportError:
        pass

    # for custom overwrite, e.g. in CI
    cuda_major = os.environ.get("CUDA_CORE_BUILD_MAJOR")
    if cuda_major is not None:
        return cuda_major

    # also for local development
    try:
        out = subprocess.run("nvidia-smi", env=os.environ, capture_output=True, check=True)  # nosec: B603, B607
        m = re.search(r"CUDA Version:\s*([\d\.]+)", out.stdout.decode())
        if m:
            return m.group(1).split(".")[0]
    except FileNotFoundError:
        # the build machine has no driver installed
        pass

    # default fallback
    return "13"


# Note: this function returns a list of *build-time* dependencies, so it's not affected
# by "--no-deps" based on the PEP-517 design.
def get_requires_for_build_wheel(config_settings=None):
    cuda_major = _get_proper_cuda_bindings_major_version()
    cuda_bindings_require = [f"cuda-bindings=={cuda_major}.*"]
    return _build_meta.get_requires_for_build_wheel(config_settings) + cuda_bindings_require
