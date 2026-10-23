# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""What this build of cuda.core compiled against.

``CUDA_MAJOR`` and ``CUDA_BINDINGS_FLOOR`` come from the compile-time environment
that build_hooks.py sets. ``CUDA_VERSION`` is the ``CUDA_VERSION`` macro of the
``cuda.h`` that the compiler resolved when it built this module, so the record
cannot disagree with the binaries. ``cuda/core/__init__.py`` reads this module
before it selects the build. The module imports nothing.
"""

cdef extern from "cuda.h":
    enum: CUDA_H_VERSION "CUDA_VERSION"

CUDA_MAJOR: int = CUDA_CORE_BUILD_MAJOR
CUDA_VERSION: int = CUDA_H_VERSION
CUDA_BINDINGS_FLOOR: tuple[int, int, int] = CUDA_CORE_BINDINGS_FLOOR
