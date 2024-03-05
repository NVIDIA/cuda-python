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
cimport cuda.ccuda as ccuda

def test_ccuda_memcpy():
    # Init CUDA
    err = ccuda.cuInit(0)
    assert(err == 0)

    # Get device
    cdef ccuda.CUdevice device
    err = ccuda.cuDeviceGet(&device, 0)
    assert(err == 0)

    # Construct context
    cdef ccuda.CUcontext ctx
    err = ccuda.cuCtxCreate(&ctx, 0, device)
    assert(err == 0)

    # Allocate dev memory
    cdef ccuda.CUdeviceptr dptr
    err = ccuda.cuMemAlloc(&dptr, 1024)
    assert(err == 0)

    # Set h1 and h2 memory to be different
    cdef char[1024] hptr1
    memset(hptr1, 1, 1024)
    cdef char[1024] hptr2
    memset(hptr2, 2, 1024)
    assert(memcmp(hptr1, hptr2, 1024) != 0)

    # h1 to D
    err = ccuda.cuMemcpyHtoD(dptr, <void*>hptr1, 1024)
    assert(err == 0)

    # D to h2
    err = ccuda.cuMemcpyDtoH(<void*>hptr2, dptr, 1024)
    assert(err == 0)

    # Validate h1 == h2
    assert(memcmp(hptr1, hptr2, 1024) == 0)

    # Cleanup
    err = ccuda.cuMemFree(dptr)
    assert(err == 0)
    err = ccuda.cuCtxDestroy(ctx)
    assert(err == 0)