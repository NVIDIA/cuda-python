# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cuda.cnvrtc as cnvrtc
cimport cuda._lib.utils as utils

cdef class nvrtcProgram:
    """


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cnvrtc.nvrtcProgram* _ptr
    cdef bint _ptr_owner
