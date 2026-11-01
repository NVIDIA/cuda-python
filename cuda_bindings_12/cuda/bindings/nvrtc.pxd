# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated with version 12.9.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=ac973884786458b1e5df96532862991c5f4bdf2e9c448f656ab63c297ff61372
cimport cuda.bindings.cynvrtc as cynvrtc

include "_lib/utils.pxd"

cdef class nvrtcProgram:
    """ nvrtcProgram is the unit of compilation, and an opaque handle for a program.

    To compile a CUDA program string, an instance of nvrtcProgram must be created first with nvrtcCreateProgram, then compiled with nvrtcCompileProgram.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cynvrtc.nvrtcProgram  _pvt_val
    cdef cynvrtc.nvrtcProgram* _pvt_ptr
