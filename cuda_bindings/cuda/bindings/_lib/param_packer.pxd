# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Include "param_packer.h" so its contents get compiled into every
# Cython extension module that depends on param_packer.pxd.
cdef extern from "param_packer.h":
    # Populates ctypes pointers + the feeder table once, at import. `except +`
    # translates a C++ throw (e.g. failed `import ctypes`) into a Python
    # exception during module import, preserving any already-set Python error.
    void init_param_packer() except +
    # Hot path: read-only lookup, does not throw -> intentionally left as an
    # implicit-noexcept extern (no `except +` needed).
    int feed(void* ptr, object o, object ct)
