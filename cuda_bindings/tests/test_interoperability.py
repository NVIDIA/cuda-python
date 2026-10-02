# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cudart

# Cap for the memory pool created by test_interop_memPool. Touching the device's
# default pool (e.g. via cuDeviceGetDefaultMemPool, cudaDeviceGetDefaultMemPool,
# or cuDeviceGetMemPool / cudaDeviceGetMemPool before a pool has been set)
# reserves virtual address space of about twice the device memory, which
# cannot be satisfied in a 39-bit address space.
POOL_SIZE = 2 * 1024 * 1024  # 2 MiB


def supportsMemoryPool():
    err, isSupported = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, 0)
    return err == cudart.cudaError_t.cudaSuccess and isSupported


def test_interop_stream():
    # DRV to RT
    err_dr, stream = cuda.cuStreamCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaStreamDestroy(stream)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, stream = cudart.cudaStreamCreate()
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuStreamDestroy(stream)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_event():
    # DRV to RT
    err_dr, event = cuda.cuEventCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaEventDestroy(event)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, event = cudart.cudaEventCreate()
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuEventDestroy(event)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_graph():
    # DRV to RT
    err_dr, graph = cuda.cuGraphCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaGraphDestroy(graph)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, graph = cudart.cudaGraphCreate(0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuGraphDestroy(graph)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_graphNode():
    err_dr, graph = cuda.cuGraphCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # DRV to RT
    err_dr, node = cuda.cuGraphAddEmptyNode(graph, [], 0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaGraphDestroyNode(node)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, node = cudart.cudaGraphAddEmptyNode(graph, [], 0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuGraphDestroyNode(node)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    (err_rt,) = cudart.cudaGraphDestroy(graph)
    assert err_rt == cudart.cudaError_t.cudaSuccess


# cudaUserObject_t
# TODO


# cudaFunction_t
# TODO


@pytest.mark.agent_authored(model="claude-sonnet-5.5")
@pytest.mark.skipif(not supportsMemoryPool(), reason="Requires mempool operations")
def test_interop_memPool():
    # Use a small, capped pool instead of the device's default pool. The pool
    # must be set before it is queried below because the getters create (and
    # reserve address space for) the default pool if none has been set.
    props = cuda.CUmemPoolProps()
    props.allocType = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    props.handleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    props.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    props.location.id = 0
    props.maxSize = POOL_SIZE
    err_dr, pool = cuda.cuMemPoolCreate(props)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    try:
        # DRV to RT
        (err_rt,) = cudart.cudaDeviceSetMemPool(0, pool)
        assert err_rt == cudart.cudaError_t.cudaSuccess

        # RT to DRV
        err_rt, rt_pool = cudart.cudaDeviceGetMemPool(0)
        assert err_rt == cudart.cudaError_t.cudaSuccess
        assert int(rt_pool) == int(pool)
        (err_dr,) = cuda.cuDeviceSetMemPool(0, rt_pool)
        assert err_dr == cuda.CUresult.CUDA_SUCCESS
    finally:
        (err_dr,) = cuda.cuMemPoolDestroy(pool)
        assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_graphExec():
    err_dr, graph = cuda.cuGraphCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, node = cuda.cuGraphAddEmptyNode(graph, [], 0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # DRV to RT
    err_dr, graphExec = cuda.cuGraphInstantiate(graph, 0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaGraphExecDestroy(graphExec)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, graphExec = cudart.cudaGraphInstantiate(graph, 0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuGraphExecDestroy(graphExec)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    (err_rt,) = cudart.cudaGraphDestroy(graph)
    assert err_rt == cudart.cudaError_t.cudaSuccess


def test_interop_deviceptr():
    # Allocate dev memory
    size = 1024 * np.uint8().itemsize
    err_dr, dptr = cuda.cuMemAlloc(size)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # Allocate host memory
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert np.array_equal(h1, h2) is False

    # Initialize device memory
    (err_rt,) = cudart.cudaMemset(dptr, 1, size)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # D to h2
    (err_rt,) = cudart.cudaMemcpy(h2, dptr, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # Validate h1 == h2
    assert np.array_equal(h1, h2)

    # Cleanup
    (err_dr,) = cuda.cuMemFree(dptr)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
