# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
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