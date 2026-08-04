# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Include "param_packer.h" so its contents get compiled into every
# Cython extension module that depends on param_packer.pxd. Each such module
# therefore owns an independent copy of the header's static state and must call
# init_param_packer() from its own module body.
cdef extern from "param_packer.h":
    # `except +` translates a C++ throw (failed `import ctypes`, or an
    # allocation failure while building the feeder table) into a Python
    # exception. Without it Cython treats an extern cdef function as implicitly
    # noexcept and generates no handler, so a throw would hit std::terminate.
    void init_param_packer() except +*
    int feed(void* ptr, object o, object ct) except -1
