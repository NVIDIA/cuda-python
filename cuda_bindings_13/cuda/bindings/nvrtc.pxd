# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated with version 13.5.0. Do not modify it directly.
# !!! WARNING: THIS FILE CONTAINS PRERELEASE APIs !!!
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=8624adff129b8d53ce64f44d09df5e98e9d67618f93d357aff7b386453ad5e1a
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

cdef class nvrtcBundledHeadersInfo:
    """
    Structure containing information about bundled headers.

    Attributes
    ----------

    available : int
        Non-zero if bundled headers are available


    compressedSize : size_t
        Size of compressed archive in bytes


    uncompressedSize : size_t
        Estimated size when extracted in bytes


    cudaVersionMajor : int
        CUDA major version of bundled headers


    cudaVersionMinor : int
        CUDA minor version of bundled headers


    numFiles : unsigned int
        Number of header files in the bundle


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef object _owner
    cdef cynvrtc.nvrtcBundledHeadersInfo _pvt_val
    cdef cynvrtc.nvrtcBundledHeadersInfo* _pvt_ptr
