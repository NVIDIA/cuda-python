# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated with version 13.4.0. Do not modify it directly.
# !!! WARNING: THIS FILE CONTAINS PRERELEASE APIs !!!
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=315bc7e377beef4b6e7e6dbb58c18dd44c06655387f58ccdfacd96a4fa464c60
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

cdef class anon_struct0:
    """
    Attributes
    ----------

    available : int



    compressedSize : size_t



    uncompressedSize : size_t



    cudaVersionMajor : int



    cudaVersionMinor : int



    numFiles : unsigned int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cynvrtc.nvrtcBundledHeadersInfo* _pvt_ptr

cdef class nvrtcBundledHeadersInfo(anon_struct0):
    """
    Attributes
    ----------

    available : int



    compressedSize : size_t



    uncompressedSize : size_t



    cudaVersionMajor : int



    cudaVersionMinor : int



    numFiles : unsigned int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cynvrtc.nvrtcBundledHeadersInfo _pvt_val
