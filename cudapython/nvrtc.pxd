# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cudapython.cnvrtc as cnvrtc
cimport cudapython._lib.utils as utils

cdef class nvrtcProgram:
    cdef cnvrtc.nvrtcProgram* _ptr
    cdef bint _ptr_owner
