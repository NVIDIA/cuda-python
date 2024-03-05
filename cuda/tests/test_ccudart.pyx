# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from libc.string cimport (
    memset,
    memcmp
    )
cimport cuda.ccudart as ccudart

def test_ccudart_memcpy():
    # Allocate dev memory
    cdef void* dptr
    err = ccudart.cudaMalloc(&dptr, 1024)
    assert(err == ccudart.cudaSuccess)

    # Set h1 and h2 memory to be different
    cdef char[1024] hptr1
    memset(hptr1, 1, 1024)
    cdef char[1024] hptr2
    memset(hptr2, 2, 1024)
    assert(memcmp(hptr1, hptr2, 1024) != 0)

    # h1 to D
    err = ccudart.cudaMemcpy(dptr, <void*>hptr1, 1024, ccudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    assert(err == ccudart.cudaSuccess)

    # D to h2
    err = ccudart.cudaMemcpy(<void*>hptr2, dptr, 1024, ccudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    assert(err == ccudart.cudaSuccess)

    # Validate h1 == h2
    assert(memcmp(hptr1, hptr2, 1024) == 0)

    # Cleanup
    err = ccudart.cudaFree(dptr)
    assert(err == ccudart.cudaSuccess)

from cuda.ccudart cimport dim3
from cuda.ccudart cimport cudaMemAllocationHandleType
from cuda.ccudart cimport CUuuid, cudaUUID_t

cdef extern from *:
    """
    #include <cuda_runtime_api.h>
    dim3 copy_and_append_dim3(dim3 copy) {
        return dim3(copy.x + 1, copy.y + 1, copy.z + 1);
    }
    void foo(cudaMemAllocationHandleType x) {
        return;
    }
    int compareUUID(CUuuid cuType, cudaUUID_t cudaType) {
        return memcmp(&cuType, &cudaType, sizeof(CUuuid));
    }
    """
    void foo(cudaMemAllocationHandleType x)
    dim3 copy_and_append_dim3(dim3 copy)
    int compareUUID(CUuuid cuType, cudaUUID_t cudaType)

def test_ccudart_interoperable():
    # struct
    cdef dim3 oldDim, newDim
    oldDim.x = 1
    oldDim.y = 2
    oldDim.z = 3
    newDim = copy_and_append_dim3(oldDim)
    assert oldDim.x + 1 == newDim.x
    assert oldDim.y + 1 == newDim.y
    assert oldDim.z + 1 == newDim.z

    # Enum
    foo(cudaMemAllocationHandleType.cudaMemHandleTypeNone)

    # typedef struct
    cdef CUuuid type_one
    cdef cudaUUID_t type_two
    memset(type_one.bytes, 1, sizeof(type_one.bytes))
    memset(type_two.bytes, 1, sizeof(type_one.bytes))
    assert compareUUID(type_one, type_two) == 0
    memset(type_two.bytes, 2, sizeof(type_one.bytes))
    assert compareUUID(type_one, type_two) != 0
