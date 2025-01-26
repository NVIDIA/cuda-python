# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated with version 12.6.1. Do not modify it directly.

cimport cython  # NOQA

from enum import IntEnum as _IntEnum


###############################################################################
# Enum
###############################################################################

class Result(_IntEnum):
    """See `nvvmResult`."""
    SUCCESS = NVVM_SUCCESS
    ERROR_OUT_OF_MEMORY = NVVM_ERROR_OUT_OF_MEMORY
    ERROR_PROGRAM_CREATION_FAILURE = NVVM_ERROR_PROGRAM_CREATION_FAILURE
    ERROR_IR_VERSION_MISMATCH = NVVM_ERROR_IR_VERSION_MISMATCH
    ERROR_INVALID_INPUT = NVVM_ERROR_INVALID_INPUT
    ERROR_INVALID_PROGRAM = NVVM_ERROR_INVALID_PROGRAM
    ERROR_INVALID_IR = NVVM_ERROR_INVALID_IR
    ERROR_INVALID_OPTION = NVVM_ERROR_INVALID_OPTION
    ERROR_NO_MODULE_IN_PROGRAM = NVVM_ERROR_NO_MODULE_IN_PROGRAM
    ERROR_COMPILATION = NVVM_ERROR_COMPILATION


###############################################################################
# Error handling
###############################################################################

class nvvmError(Exception):

    def __init__(self, status):
        self.status = status
        s = Result(status)
        cdef str err = f"{s.name} ({s.value})"
        super(nvvmError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cdef int check_status(int status) except 1 nogil:
    if status != 0:
        with gil:
            raise nvvmError(status)
    return status


###############################################################################
# Wrapper functions
###############################################################################

## cpdef destroy(intptr_t handle):
##     """nvvmDestroy frees the memory associated with the given handle.
## 
##     Args:
##         handle (intptr_t): nvvm handle.
## 
##     .. seealso:: `nvvmDestroy`
##     """
##     cdef Handle h = <Handle>handle
##     with nogil:
##         status = nvvmDestroy(&h)
##     check_status(status)


cpdef tuple version():
    """Get the NVVM version.

    Returns:
        A 2-tuple containing:

        - int: NVVM major version number.
        - int: NVVM minor version number.

    .. seealso:: `nvvmVersion`
    """
    cdef int major
    cdef int minor
    with nogil:
        status = nvvmVersion(&major, &minor)
    check_status(status)
    return (major, minor)


cpdef tuple ir_version():
    """Get the NVVM IR version.

    Returns:
        A 4-tuple containing:

        - int: NVVM IR major version number.
        - int: NVVM IR minor version number.
        - int: NVVM IR debug metadata major version number.
        - int: NVVM IR debug metadata minor version number.

    .. seealso:: `nvvmIRVersion`
    """
    cdef int major_ir
    cdef int minor_ir
    cdef int major_dbg
    cdef int minor_dbg
    with nogil:
        status = nvvmIRVersion(&major_ir, &minor_ir, &major_dbg, &minor_dbg)
    check_status(status)
    return (major_ir, minor_ir, major_dbg, minor_dbg)
