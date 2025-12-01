# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cudart
import numpy as np
import pytest


def supportsMemoryPool():
    err, isSupported = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, 0)
    return err == cudart.cudaError_t.cudaSuccess and isSupported


@pytest.mark.usefixtures("ctx")
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


@pytest.mark.usefixtures("ctx")
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


@pytest.mark.usefixtures("ctx")
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


@pytest.mark.usefixtures("ctx")
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


@pytest.mark.skipif(not supportsMemoryPool(), reason="Requires mempool operations")
@pytest.mark.usefixtures("ctx")
def test_interop_memPool():
    # DRV to RT
    err_dr, pool = cuda.cuDeviceGetDefaultMemPool(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaDeviceSetMemPool(0, pool)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, pool = cudart.cudaDeviceGetDefaultMemPool(0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuDeviceSetMemPool(0, pool)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.usefixtures("ctx")
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


@pytest.mark.usefixtures("ctx")
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
